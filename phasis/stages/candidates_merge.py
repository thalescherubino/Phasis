from __future__ import annotations

import configparser
import os
from functools import partial

import pandas as pd

import phasis.runtime as rt
from phasis.cache import MEM_FILE_DEFAULT, getmd5, phase2_basename
from phasis.parallel import run_parallel_with_progress


def chromosome_clusters_to_candidate_loci(
    chromosome_df: pd.DataFrame,
    *,
    minClusterLength: int | None = None,
):
    """
    For a single chromosome, convert clusters to candidate loci based on minimum length.

    Returns a list-of-lists rows in the format:
        [clusterID, 0, chromosome, min_pos, max_pos]
    """
    # Resolve threshold locally (avoid legacy globals)
    mcl = int(getattr(rt, "minClusterLength", 0) or 0) if minClusterLength is None else int(minClusterLength)

    # Group by clusterID and calculate min/max positions
    cluster_positions = chromosome_df.groupby("clusterID")["pos"].agg(["min", "max"]).reset_index()

    # Merge to get chromosome for each clusterID (matches legacy behavior, may duplicate rows)
    cluster_info = chromosome_df.merge(cluster_positions, on="clusterID")

    # Clean up clusterID
    cluster_info["clusterID"] = cluster_info["clusterID"].astype(str).str.strip()

    # Mask for clusters longer than minClusterLength
    mask = (cluster_info["max"] - cluster_info["min"]) >= mcl
    sub = cluster_info.loc[mask, ["clusterID", "chromosome", "min", "max"]]

    lociTablelist = []
    # Avoid lambdas/nested functions; preserve row-order behavior
    for cid, achr, s, e in sub.itertuples(index=False, name=None):
        lociTablelist.append([
            str(cid).replace("\t", "").strip(),
            0,
            int(achr),
            int(s),
            int(e),
        ])

    return lociTablelist


def loci_table_from_clusters(
    allClusters: pd.DataFrame,
    *,
    memFile: str | None = None,
    minClusterLength: int | None = None,
    outfname: str | None = None,
) -> pd.DataFrame:
    print("### Building loci table from clusters per chromosome ###")

    memFile_local = memFile or getattr(rt, "memFile", None) or MEM_FILE_DEFAULT
    outfname = outfname or phase2_basename("candidate.loci_table.tab")

    # Step 0: Check if output is up-to-date (file exists and hash matches memory file)
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(memFile_local)
    section = "LOCI_TABLE"
    if not config.has_section(section):
        config.add_section(section)

    if os.path.isfile(outfname):
        _, current_md5 = getmd5(outfname)
        prev_md5 = config[section].get(outfname)
        if prev_md5 and current_md5 == prev_md5:
            print(f"File {outfname} is up-to-date (hash match). Skipping recomputation.")
            print(f"Loci table written to {outfname}")
            with open(outfname, "r") as fh:
                file_lines = fh.readlines()[1:]  # skip header
                lociTablelist_unique = [ln.strip().split("\t") for ln in file_lines]
            return pd.DataFrame(lociTablelist_unique, columns=["name", "pval", "chr", "start", "end"])

    # Resolve minClusterLength and bind into worker
    mcl = int(getattr(rt, "minClusterLength", 0) or 0) if minClusterLength is None else int(minClusterLength)
    worker_fn = partial(chromosome_clusters_to_candidate_loci, minClusterLength=mcl)

    # Step 1: Split data into chromosome groups
    chromosome_groups = [df for _, df in allClusters.groupby("chromosome")]

    # Step 2: Multicore processing (unit 'lib-chr')
    lociTablelist = run_parallel_with_progress(
        worker_fn,
        chromosome_groups,
        desc="LociTable chromosomes",
        min_chunk=1,
        unit="lib-chr",
    )

    # Step 3: Flatten the results
    lociTablelist = [item for sublist in lociTablelist for item in sublist]

    # Step 4: Remove duplicates (stable)
    seen = set()
    lociTablelist_unique = []
    for item in lociTablelist:
        row_tuple = tuple(item)
        if row_tuple not in seen:
            seen.add(row_tuple)
            lociTablelist_unique.append(item)

    # Step 5: Write results to file
    with open(outfname, "w") as fh:
        fh.write("Cluster\tvalue1\tchromosome\tStart\tEnd\n")
        for row in lociTablelist_unique:
            fh.write("\t".join(map(str, row)) + "\n")

    # Step 6: Update memory file with output hash
    if os.path.isfile(outfname):
        _, out_md5 = getmd5(outfname)
        config[section][outfname] = out_md5
        print(f"Hash for {outfname}: {out_md5}")
    with open(memFile_local, "w") as fh:
        config.write(fh)

    print(f"Loci table written to {outfname}")
    return pd.DataFrame(lociTablelist_unique, columns=["name", "pval", "chr", "start", "end"])


def merge_candidate_clusters_across_libs(
    loci_table_path: str,
    out_path: str,
    *,
    memFile: str | None = None,
    concat_libs: bool | None = None,
) -> pd.DataFrame:
    """
    Produce the per-(chromosome, library) merged candidates and write `out_path`.
    On cache hit, LOAD + RETURN the cached file so callers always get a DataFrame.

    In --concat_libs mode (single logical library), the merge is effectively a
    pass-through of the loci table with alib="ALL_LIBS".
    """
    print("### Merging candidate clusters across libraries (per chromosome) ###")

    memFile_local = memFile or getattr(rt, "memFile", None) or MEM_FILE_DEFAULT
    concat_local = bool(getattr(rt, "concat_libs", False)) if concat_libs is None else bool(concat_libs)

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(memFile_local)

    section = "MERGED_CANDIDATES"
    if not config.has_section(section):
        config.add_section(section)

    # --- Cache check
    if os.path.isfile(out_path):
        _, curr_md5 = getmd5(out_path)
        prev_md5 = config[section].get(out_path)
        if prev_md5 and prev_md5 == curr_md5:
            print("Outputs up-to-date (hash match). Skipping merge computation.")
            df_cached = pd.read_csv(out_path, sep="\t")

            # Normalize required columns
            if "chromosome" not in df_cached.columns and "chr" in df_cached.columns:
                df_cached = df_cached.rename(columns={"chr": "chromosome"})
            if "alib" not in df_cached.columns and concat_local:
                df_cached["alib"] = "ALL_LIBS"  # only in concat mode
            # If not concat and 'alib' is missing, warn but still return (failsafe)
            if "alib" not in df_cached.columns and not concat_local:
                print("[WARN] Cached merged table lacks 'alib' in non-concat mode.")
            return df_cached

    # --- Compute (for concat_libs this is a pass-through of the loci table)
    if not os.path.isfile(loci_table_path):
        print(f"[WARN] Loci table not found: {loci_table_path}. Returning empty DataFrame.")
        return pd.DataFrame()

    merged_df = pd.read_csv(loci_table_path, sep="\t")

    # Normalize required columns
    if "chromosome" not in merged_df.columns and "chr" in merged_df.columns:
        merged_df = merged_df.rename(columns={"chr": "chromosome"})
    if "alib" not in merged_df.columns:
        # In concat mode, set the single logical library id
        if concat_local:
            merged_df["alib"] = "ALL_LIBS"
        else:
            # Non-concat: don't guess; warn and provide a safe default.
            # Downstream grouping will still work but may be degenerate; your aggregator should
            # ideally have written 'alib' per row already for non-concat runs.
            print("[WARN] 'alib' missing in loci table on non-concat run; setting 'alib'='UNKNOWN'.")
            merged_df["alib"] = "UNKNOWN"

    # Persist + hash
    merged_df.to_csv(out_path, sep="\t", index=False)
    _, new_md5 = getmd5(out_path)
    config[section][out_path] = new_md5
    with open(memFile_local, "w") as fh:
        config.write(fh)
    print(f"Hash for {os.path.basename(out_path)}:")

    return merged_df
