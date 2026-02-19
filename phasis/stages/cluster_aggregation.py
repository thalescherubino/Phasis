from __future__ import annotations

"""
phasis.stages.cluster_aggregation
--------------------------------

Phase II helper stage: read one or more *.PHAS.candidate.clusters files and
write {phase}_processed_clusters.tab.

Design constraints:
- spawn-safe (top-level functions only)
- no imports inside functions
- minimal behavior drift vs legacy implementation
"""

import configparser
import os
import re
from typing import Iterable, List, Sequence, Tuple

import pandas as pd

import phasis.runtime as rt
from phasis.cache import getmd5, phase2_basename
from phasis.parallel import run_parallel_with_progress


# Canonical column order for processed cluster rows
PROCESSED_CLUSTER_COLUMNS: List[str] = [
    "alib", "clusterID", "chromosome", "strand", "pos", "len", "hits", "abun",
    "pval_h_f", "N_f", "X_f", "pval_r_f", "pval_corr_f", "pval_h_r", "N_r", "X_r",
    "pval_r_r", "pval_corr_r", "tag_id", "tag_seq",
]


def process_single_lib_cluster(filename: str) -> List[Tuple]:
    """
    Parse a single *.PHAS.candidate.clusters file into a list of tuples
    matching PROCESSED_CLUSTER_COLUMNS.

    filename: path to a cluster file (either per-library or ALL_LIBS.* in concat mode)
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Cluster file not found: {filename}")

    clustlist: List[Tuple] = []

    # library name from file basename: AR_1_nocontam.21-PHAS.candidate.clusters -> AR_1_nocontam
    base = os.path.basename(filename)
    alib = re.sub(r"\.\d+-PHAS\.candidate\.clusters$", "", base)

    with open(filename) as fh:
        lines = fh.readlines()

    aid = None
    for line in lines:
        if line.startswith(">"):
            # header like: ">cluster = lobe_3_nocontam-1_3894_1"
            m = re.search(r"cluster\s*=\s*([^\s]+)", line)
            if not m:
                aid = None
                continue
            aid = m.group(1).strip()  # e.g. lobe_3_nocontam-1_3894_1
            continue

        # data lines belong to the most recent header (aid)
        if not aid:
            continue

        ent = line.rstrip("\n").split("\t")
        if len(ent) < 18:
            # tolerate malformed lines (legacy would throw); keep it permissive
            continue

        achr = str(ent[0])
        astrand = str(ent[1])
        apos = int(ent[2])
        alen = int(ent[3])
        ahits = int(ent[4])
        abun = int(ent[5])
        pval_h_f = float(ent[6])
        N_f = int(ent[7])
        X_f = int(ent[8])
        pval_r_f = float(ent[9])
        pval_corr_f = float(ent[10])
        pval_h_r = float(ent[11])
        N_r = int(ent[12])
        X_r = int(ent[13])
        pval_r_r = float(ent[14])
        pval_corr_r = float(ent[15])
        tag_id = str(ent[16])
        tag_seq = str(ent[17])

        # clusterID is the clean per-lib id (no filename glue)
        clustlist.append(
            (
                alib, aid, achr, astrand, apos, alen, ahits, abun,
                pval_h_f, N_f, X_f, pval_r_f, pval_corr_f, pval_h_r, N_r, X_r,
                pval_r_r, pval_corr_r, tag_id, tag_seq,
            )
        )

    return clustlist


def _coerce_paths(clusterFiles: Sequence[str] | str) -> List[str]:
    if isinstance(clusterFiles, str):
        return [clusterFiles]
    return [str(x) for x in clusterFiles if str(x).strip()]


def _raise_on_parallel_errors(results: List[object]) -> None:
    errs = [x for x in results if isinstance(x, RuntimeError)]
    if errs:
        # bubble up the *first* error to match "fail fast" expectations
        raise errs[0]


def _update_memfile_md5(memFile: str, outfname: str) -> None:
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(memFile)

    section = "PROCESSED"
    if not config.has_section(section):
        config.add_section(section)

    if os.path.isfile(outfname):
        _, out_md5 = getmd5(outfname)
        config[section][outfname] = out_md5
        print(f"Hash for {outfname}: {out_md5}")

    with open(memFile, "w") as fh:
        config.write(fh)


def aggregate_and_write_processed_clusters(
    clusterFiles: Sequence[str] | str,
    *,
    memFile: str | None = None,
) -> pd.DataFrame:
    """
    Aggregate candidate cluster files and write {phase}_processed_clusters.tab.

    Returns: allClusters dataframe (sorted by clusterID, pos)

    memFile: optional explicit path; if None, uses rt.memFile.
    """
    print("### Aggregating and processing candidate cluster files per library ###")

    paths = _coerce_paths(clusterFiles)
    if not paths:
        raise ValueError("No cluster files provided to aggregate_and_write_processed_clusters().")

    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Missing cluster file(s): {missing}")

    # --- Parallel processing ---
    all_clustlists = run_parallel_with_progress(
        process_single_lib_cluster,
        paths,
        desc="Aggregating cluster files",
        min_chunk=1,
        unit="lib",
    )
    _raise_on_parallel_errors(all_clustlists)

    # Flatten list of lists
    flat_clustlist = [item for sublist in all_clustlists for item in sublist]

    allClusters = pd.DataFrame(flat_clustlist, columns=PROCESSED_CLUSTER_COLUMNS)
    allClusters = allClusters.sort_values(by=["clusterID", "pos"])

    outfname = phase2_basename("processed_clusters.tab")
    allClusters.to_csv(outfname, sep="\t", index=False, header=True)

    # --- Update hash in memory file ---
    mem_path = memFile or getattr(rt, "memFile", None)
    if mem_path and str(mem_path).strip():
        _update_memfile_md5(str(mem_path), outfname)
    else:
        print("[WARN] memFile not set; skipping processed_clusters md5 update.")

    print(f"Processed clusters written to {outfname}")
    return allClusters
