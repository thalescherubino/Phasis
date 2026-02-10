"""phasis.stages.output

Output writer + plotting stage extracted from legacy.py.

Goals:
- No behavior drift: writes the same TSV/GFF outputs and runs the same plots.
- Spawn-safe on macOS: plotting happens in a dedicated pool (kind="plot") and
  workers can rehydrate outdir/phase via runtime snapshot.
- No nested functions; no imports inside functions.
"""

from __future__ import annotations

import os
import re
from multiprocessing import cpu_count
from typing import Any

import numpy as np
import pandas as pd

import phasis.runtime as rt
from phasis.parallel import make_pool
from phasis import ids as st_ids

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, Bbox

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Module globals used by plotting and GFF writing (set by finalize_and_write_results or _plot_wrapper)
outdir: str | None = None
phase: str | int | None = None


def _join_outdir(dirpath: str | None, name: str) -> str:
    if not dirpath:
        return name
    return dirpath + name if dirpath.endswith("/") else dirpath + "/" + name


def _fallback_series(nrows: int) -> pd.Series:
    return pd.Series([np.nan] * int(nrows))


def _parse_identifiers_and_alib(features: pd.DataFrame, job_phase: str | int | None):
    """
    Return achr, start, end, cleaned alib arrays from features['identifier']/['alib'].

    If 'identifier' is not in 'chr:start..end' form, try to resolve via rt.mergedClusterReverse
    using the row's 'cID'. Falls back to blanks to avoid IndexError.
    """
    # Ensure reverse map exists (spawn-safe)
    rmap = getattr(rt, "mergedClusterReverse", None)
    if not isinstance(rmap, dict) or not rmap:
        try:
            st_ids.ensure_mergedClusterDict(str(job_phase) if job_phase is not None else None)
        except Exception:
            pass
        rmap = getattr(rt, "mergedClusterReverse", {}) or {}

    achr, start, end = [], [], []

    # iterate row-wise so we can look at both identifier and cID
    cids = features.get("cID", pd.Series([None] * len(features)))
    for id_str, cID in zip(features["identifier"].astype(str), cids):
        u = None
        if ":" in id_str and ".." in id_str:
            u = id_str
        else:
            # try reverse map by cID first, then by the identifier string itself
            key = None
            if cID is not None and str(cID) != "nan":
                key = str(cID).strip()
            if key and key in rmap:
                u = rmap.get(key)
            elif id_str in rmap:
                u = rmap.get(id_str)

        if u and ":" in u and ".." in u:
            left, right = u.split(":", 1)
            achr.append(left)
            s_val, e_val = right.split("..", 1)
            start.append(s_val)
            end.append(e_val)
        else:
            achr.append("")
            start.append("")
            end.append("")

    # keep alib as-is unless it ends with ".{phase}-PHAS.candidate"
    alib_src = features["alib"].astype(str).tolist()
    alib_ids = [re.sub(rf"\.{re.escape(str(job_phase))}-PHAS\.candidate$", "", x) for x in alib_src]
    return achr, start, end, alib_ids


def format_attributes(row):
    attributes = {
        'id':row['identifier'],
        'complexity': row['complexity'],
        'strand_bias': row['strand_bias'],
        'log_clust_len_norm_counts': row['log_clust_len_norm_counts'],
        'ratio_abund_len_phase': row['ratio_abund_len_phase']
    }
    attr_str = ';'.join([f"{key}={value}" for key, value in attributes.items()])
    return attr_str


def write_gff(phasis_result_df,gff_filename):
    unique_df = phasis_result_df.groupby('identifier').first().reset_index()
    # Write to GFF3 file
    
    with open(gff_filename, 'w') as gff_file:
        for index, row in unique_df.iterrows():
            seq_id = row['achr']
            source = 'Phasis'
            feature_type = f'{phase}-PHAS'
            start_pos = str(row['start'])
            end_pos = str(row['end'])
            score = str(row['phasis_score'])
            strand = '.'
            t_phase = '.'
            attributes = format_attributes(row)
            gff_line = f"{seq_id}\t{source}\t{feature_type}\t{start_pos}\t{end_pos}\t{score}\t{strand}\t{t_phase}\t{attributes}\n"
            gff_file.write(gff_line)

    return None


def plot_report_heat_map(phasis_result_df, plot_type):
    print("#### Plotting heatmap ######")

    # Create a DataFrame to store heatmap data
    #data = pd.DataFrame(data=0, columns=list(phasis_result_df["....unique()), index=list(phasis_result_df["identifier"].unique()))

    data = pd.DataFrame(
    data=0.0, 
    columns=sorted(list(phasis_result_df["alib"].unique())),  # Sort columns alphanumerically
    index=list(phasis_result_df["identifier"].unique())
    )

    # Iterate over rows and columns to fill heatmap data
    for i in tqdm(data.index, desc="Processing Rows"):
        tempRows = phasis_result_df[phasis_result_df["identifier"] == i]
        for j in data.columns:
            k = 0.0
            subSetData = tempRows[tempRows["alib"] == j]
            if not subSetData.empty:
                k = float(subSetData["label"].iloc[0] == "PHAS")
            data.at[i, j] = k

    # Define a normalization to represent the boundaries between each level
    norm = Normalize(vmin=0, vmax=1)

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap = sns.heatmap(
        data,
        cmap="viridis",
        norm=norm,
        ax=ax,
        cbar_kws={"ticks": [0, 1], "label": "PHAS call (0/1)"}
    )

    # Improve readability
    ax.set_xlabel("Library")
    ax.set_ylabel("PHAS locus")
    ax.set_title(f"{phase}-PHAS {plot_type} heatmap")

    # Save
    try:
        os.makedirs(outdir, exist_ok=True)
    except Exception:
        pass
    out = f"{outdir}/{phase}_{plot_type}_report_heatmap.pdf" if outdir else f"{phase}_{plot_type}_report_heatmap.pdf"
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    return None


def plot_phasAbundance_heat_map(phasis_result_df, plot_type):
    print("#### Plotting phasAbundance heatmap ######")

    # Filter PHAS only
    phas = phasis_result_df[phasis_result_df["label"] == "PHAS"].copy()

    # Create a DataFrame to store heatmap data: abundance of phase-length reads per (locus, lib)
    data = pd.DataFrame(
        data=0.0,
        columns=sorted(list(phas["alib"].unique())),
        index=list(phas["identifier"].unique())
    )

    for i in tqdm(data.index, desc="Processing Rows"):
        tempRows = phas[phas["identifier"] == i]
        for j in data.columns:
            sub = tempRows[tempRows["alib"] == j]
            if not sub.empty:
                # use total_abund if present; else fall back to phasis_score
                if "total_abund" in sub.columns:
                    data.at[i, j] = float(sub["total_abund"].iloc[0])
                else:
                    data.at[i, j] = float(sub["phasis_score"].iloc[0])

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        data,
        cmap="mako",
        ax=ax,
        cbar_kws={"label": "Abundance"}
    )

    ax.set_xlabel("Library")
    ax.set_ylabel("PHAS locus")
    ax.set_title(f"{phase}-PHAS {plot_type} abundance heatmap")

    try:
        os.makedirs(outdir, exist_ok=True)
    except Exception:
        pass
    out = f"{outdir}/{phase}_{plot_type}_phasAbundance_heatmap.pdf" if outdir else f"{phase}_{plot_type}_phasAbundance_heatmap.pdf"
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    return None


def plot_totalAbundance_heat_map(phasis_result_df, plot_type):
    print("#### Plotting totalAbundance heatmap ######")

    data = pd.DataFrame(
        data=0.0,
        columns=sorted(list(phasis_result_df["alib"].unique())),
        index=list(phasis_result_df["identifier"].unique())
    )

    for i in data.index:
        tempRows = phasis_result_df[phasis_result_df["identifier"] == i]
        for j in data.columns:
            sub = tempRows[tempRows["alib"] == j]
            if not sub.empty:
                if "total_abund" in sub.columns:
                    data.at[i, j] = float(sub["total_abund"].iloc[0])
                else:
                    data.at[i, j] = float(sub["phasis_score"].iloc[0])

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        data,
        cmap="rocket_r",
        ax=ax,
        cbar_kws={"label": "Total abundance"}
    )

    ax.set_xlabel("Library")
    ax.set_ylabel("PHAS locus")
    ax.set_title(f"{phase}-PHAS {plot_type} total abundance heatmap")

    try:
        os.makedirs(outdir, exist_ok=True)
    except Exception:
        pass
    out = f"{outdir}/{phase}_{plot_type}_totalAbundance_heatmap.pdf" if outdir else f"{phase}_{plot_type}_totalAbundance_heatmap.pdf"
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    return None


def _plot_wrapper(job):
    """
    Worker-safe plot wrapper.

    Accepts:
      - (fn, df, mname)                              [legacy format]
      - (fn, df, mname, job_outdir, job_phase)       [spawn-safe format]

    Ensures `outdir` and `phase` globals are defined in the worker
    without importing inside the function (macOS spawn-safe).
    """
    if not isinstance(job, (tuple, list)):
        raise TypeError(f"_plot_wrapper expected tuple/list, got: {type(job)}")

    n = len(job)
    if n == 3:
        fn, df, mname = job
        job_outdir, job_phase = None, None
    elif n == 5:
        fn, df, mname, job_outdir, job_phase = job
    else:
        raise ValueError(f"Unexpected plot job tuple size: {n}")

    # Ensure globals exist in worker (spawn/fork safe)
    global outdir, phase

    # Prefer explicit values passed in the job (most reliable across spawn)
    if job_outdir is not None:
        outdir = job_outdir
    elif outdir is None:
        # Fallback to runtime snapshot loaded in worker initializer
        outdir = getattr(rt, "outdir", outdir)

    if job_phase is not None:
        phase = job_phase
    elif phase is None:
        phase = getattr(rt, "phase", phase)

    # Ensure output directory exists (safe in parallel)
    if outdir:
        try:
            os.makedirs(outdir, exist_ok=True)
        except Exception:
            pass

    return fn(df, mname)


def finalize_and_write_results(method_name: str, features: pd.DataFrame, *, job_outdir: str | None = None, job_phase: str | int | None = None):
    
    """
    Build result dataframe (all clusters + filtered PHAS),
    write standardized outputs, run 3 tree plots in parallel, and write GFF.
    """
    # Resolve config (prefer explicit args; fallback to runtime)
    global outdir, phase
    if job_outdir is not None:
        outdir = job_outdir
    elif outdir is None:
        outdir = getattr(rt, "outdir", outdir)

    if job_phase is not None:
        phase = job_phase
    elif phase is None:
        phase = getattr(rt, "phase", phase)

    try:
        if outdir:
            os.makedirs(outdir, exist_ok=True)
    except Exception:
        pass

    # Decompose identifiers and clean alib tags
    achr, start, end, alib_ids = _parse_identifiers_and_alib(features, phase)

    nrows = len(features)

    # ---- Full 'all clusters' table (unchanged, includes strict columns) ----
    all_df = pd.DataFrame({
        'identifier': features['identifier'],
        'phasis_score': features['phasis_score'],
        'achr': achr,
        'start': start,
        'end': end,
        'complexity': features['complexity'],
        'strand_bias': features['strand_bias'],
        'log_clust_len_norm_counts': features['log_clust_len_norm_counts'],
        'ratio_abund_len_phase': features['ratio_abund_len_phase'],
        'label': features['label'],
        'alib': alib_ids,
        'combined_fishers': features.get('combined_fishers', _fallback_series(nrows)),
        'total_abund': features.get('total_abund', _fallback_series(nrows)),
        'w_Howell_score': features.get('w_Howell_score', _fallback_series(nrows)),
        'w_window_start': features.get('w_window_start', _fallback_series(nrows)),
        'w_window_end': features.get('w_window_end', _fallback_series(nrows)),
        'c_Howell_score': features.get('c_Howell_score', _fallback_series(nrows)),
        'c_window_start': features.get('c_window_start', _fallback_series(nrows)),
        'c_window_end': features.get('c_window_end', _fallback_series(nrows)),
        'Peak_Howell_score': features.get('Peak_Howell_score', _fallback_series(nrows)),
        # strict (classic) Howell
        'w_Howell_score_strict': features.get('w_Howell_score_strict', _fallback_series(nrows)),
        'w_window_start_strict': features.get('w_window_start_strict', _fallback_series(nrows)),
        'w_window_end_strict': features.get('w_window_end_strict', _fallback_series(nrows)),
        'c_Howell_score_strict': features.get('c_Howell_score_strict', _fallback_series(nrows)),
        'c_window_start_strict': features.get('c_window_start_strict', _fallback_series(nrows)),
        'c_window_end_strict': features.get('c_window_end_strict', _fallback_series(nrows)),
        'Peak_Howell_score_strict': features.get('Peak_Howell_score_strict', _fallback_series(nrows)),
    })

    # Standardized filenames
    all_out   = _join_outdir(outdir, f"{phase}_{method_name}_all_clusters.tsv")
    calls_out = _join_outdir(outdir, f"{phase}_{method_name}_calls.tsv")
    gff_out   = _join_outdir(outdir, f"{phase}_PHAS.gff")
 
    # Write all clusters with labels
    all_df.to_csv(all_out, sep="\t", index=False)

    # Keep only PHAS
    phas_df = all_df[all_df['label'] == 'PHAS'].copy()

    # Write GFF
    write_gff(phas_df,gff_out)

    # ---- Compact calls table (same as before + Peak_Howell_score_strict) ----
    calls_cols = [
        'identifier', 'phasis_score', 'achr', 'start', 'end', 'alib',
        'Peak_Howell_score', 'Peak_Howell_score_strict'
    ]
    # Ensure missing columns are created as NaN so write doesn't fail
    for col in calls_cols:
        if col not in phas_df.columns:
            phas_df[col] = np.nan
    compact_calls = phas_df[calls_cols].copy()
    compact_calls.to_csv(calls_out, sep="\t", index=False)

    # --- Run the 3 plots in parallel (each on a core) ---
    plot_jobs = [
    (plot_report_heat_map, all_df, method_name, outdir, phase),
    (plot_phasAbundance_heat_map, all_df, method_name, outdir, phase),
    (plot_totalAbundance_heat_map, all_df, method_name, outdir, phase),
    ]
    with make_pool(min(3, cpu_count()), kind="plot") as p:
        p.map(_plot_wrapper, plot_jobs)
    print(f"  - Wrote: {all_out}, {calls_out}, and {gff_out}")
