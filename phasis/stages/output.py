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
from matplotlib.colors import Normalize, LinearSegmentedColormap
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


def _tqdm(iterable, **kwargs):
    # tqdm is optional; fall back to plain iteration if not available.
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def _phase_is_24() -> bool:
    try:
        return str(phase) == "24"
    except Exception:
        return False


def _filter_df_keep_any_phas_call(df: pd.DataFrame) -> pd.DataFrame:
    """
    For 24-PHAS runs only: keep loci (identifier rows) that have at least one PHAS call
    in any library. Leave 21-PHAS behavior unchanged.
    """
    if not _phase_is_24():
        return df
    if df is None or df.empty:
        return df
    if "identifier" not in df.columns or "label" not in df.columns:
        return df
    keep = df.groupby("identifier")["label"].apply(lambda s: (s == "PHAS").any())
    keep_ids = keep[keep].index
    return df[df["identifier"].isin(keep_ids)].copy()


def _save_empty_plot(path: str, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14)
    ax.set_title(title, fontsize=16, pad=20)
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

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

    all_libs = sorted(list(phasis_result_df["alib"].unique())) if (phasis_result_df is not None and not phasis_result_df.empty and "alib" in phasis_result_df.columns) else []

    phasis_result_df = _filter_df_keep_any_phas_call(phasis_result_df)

    data = pd.DataFrame(
        data=0.0,
        columns=all_libs,
        index=list(phasis_result_df["identifier"].unique()) if (phasis_result_df is not None and not phasis_result_df.empty and "identifier" in phasis_result_df.columns) else []
    )

    if data.shape[0] == 0 or data.shape[1] == 0:
        out = _join_outdir(outdir, f"{phase}_{plot_type}_PHAS.pdf")
        _save_empty_plot(out, f"{phase}-PHAS {plot_type}", "No loci to plot for this heatmap.")
        return None

    # Iterate over rows and columns to fill heatmap data
    for i in _tqdm(data.index, desc="Processing Rows"):
        tempRows = phasis_result_df[phasis_result_df["identifier"] == i]
        for j in data.columns:
            k = 1.0 if _phase_is_24() else 0.0
            subSetData = tempRows[tempRows["alib"] == j]
            if not subSetData.empty:
                if "non-PHAS" in subSetData["label"].values:
                    k = 1.0
                elif "PHAS" in subSetData["label"].values:
                    k = 2.0
            data.at[i, j] = k

    # Extract chromosome data from the identifier
    chrom_data = [identifier.split(":")[0] for identifier in data.index]

    # Sort chromosomes numerically instead of lexicographically (i.e., 1, 2, 3, ... not 1, 10, 2, 3, ...)
    sorted_indices = sorted(
        range(len(chrom_data)),
        key=lambda idx: int(chrom_data[idx]) if chrom_data[idx].isdigit() else chrom_data[idx],
    )

    # Reorder the DataFrame based on sorted chromosomes
    data = data.iloc[sorted_indices]

    # Re-extract the sorted chromosome data
    chrom_data = [chrom_data[idx] for idx in sorted_indices]
    unique_chromosomes = sorted(np.unique(chrom_data), key=lambda x: int(x) if x.isdigit() else x)

    # Create the heatmap figure with specified figure size
    f, ax = plt.subplots(figsize=(11, 11))

    # Define color map for the heatmap
    colors = ["#C3D8EA", "#3662A5", "#C24F4E"]
    cmap = LinearSegmentedColormap.from_list("Custom", colors, len(colors))

    # Create the actual heatmap for the PHAS data
    heat = sns.heatmap(data, square=False, cmap=cmap, cbar=False, xticklabels=True, yticklabels=False, ax=ax)

    # Add a new axis for the chromosome bar to the left
    cax = f.add_axes([0.17, 0.1, 0.02, 0.8])  # Adjust the position for the left side

    # Manually create a discrete grayscale colormap for the chromosomes
    base_cmap = plt.get_cmap("Greys")
    chrom_cmap = base_cmap(np.linspace(0.3, 0.9, len(unique_chromosomes)))  # Discretizing to avoid full black/white

    # Assign colors based on the order of unique chromosomes
    chrom_color_map = {chrom: idx for idx, chrom in enumerate(unique_chromosomes)}

    # Map chromosomes to grayscale colors
    chrom_colors = np.array([chrom_color_map[chrom] for chrom in chrom_data]).reshape(-1, 1)

    # Plot the chromosome colorbar (no x-axis)
    cax.imshow(
        chrom_colors,
        cmap=LinearSegmentedColormap.from_list("CustomGreys", chrom_cmap, len(unique_chromosomes)),
        aspect="auto",
    )
    cax.set_xticks([])  # Remove x-axis ticks
    cax.set_yticks([])  # Remove y-axis ticks

    # Remove the black border around the colorbar
    cax.spines["top"].set_visible(False)
    cax.spines["right"].set_visible(False)
    cax.spines["bottom"].set_visible(False)
    cax.spines["left"].set_visible(False)

    # Align chromosome bar with heatmap y-coordinates
    cax.set_ylim(len(data.index), 0)  # Set y-limits to invert the chromosome bar

    # Calculate the midpoints for each chromosome based on its range of rows
    chrom_ranges = {}
    for chrom in unique_chromosomes:
        chrom_loci_indices = np.where(np.array(chrom_data) == chrom)[0]
        chrom_midpoint = (chrom_loci_indices[0] + chrom_loci_indices[-1]) / 2
        chrom_ranges[chrom] = chrom_midpoint

    # Set chromosome names in the middle of the chromosome color bar using calculated midpoints
    midpoints = [chrom_ranges[chrom] for chrom in unique_chromosomes]
    cax.set_yticks(midpoints)
    cax.set_yticklabels(unique_chromosomes, fontsize=16, rotation=0, ha="center")
    cax.yaxis.set_tick_params(labelsize=16, pad=11)

    # Add a custom color bar (legend) to the top right
    cax2 = inset_axes(
        ax,
        width="2.5%",
        height="9%",
        loc="lower left",
        bbox_to_anchor=(-0.25, 1.01, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cax2.spines["top"].set_visible(False)
    cax2.spines["right"].set_visible(False)
    cax2.spines["bottom"].set_visible(False)
    cax2.spines["left"].set_visible(False)

    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax2, orientation="vertical")
    cbar.set_ticklabels(["Not detected", r"non-$\it{PHAS}$ cluster", r"$\it{PHAS}$"])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)

    fig = heat.get_figure()
    out = _join_outdir(outdir, f"{phase}_{plot_type}_PHAS.pdf")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return None


def plot_phasAbundance_heat_map(phasis_result_df, plot_type):
    print("#### Plotting PHAS Abundance heatmap ######")

    all_libs = sorted(list(phasis_result_df["alib"].unique())) if (phasis_result_df is not None and not phasis_result_df.empty and "alib" in phasis_result_df.columns) else []

    phasis_result_df = _filter_df_keep_any_phas_call(phasis_result_df)

    data = pd.DataFrame(
        data=0.0,
        columns=all_libs,
        index=list(phasis_result_df["identifier"].unique()) if (phasis_result_df is not None and not phasis_result_df.empty and "identifier" in phasis_result_df.columns) else [],
    )

    if data.shape[0] == 0 or data.shape[1] == 0:
        out = _join_outdir(outdir, f"{plot_type}_{phase}_Abundance_PHAS.pdf")
        _save_empty_plot(out, f"{phase}-PHAS {plot_type}", "No loci to plot for this heatmap.")
        return None

    for i in _tqdm(data.index, desc="Processing Rows"):
        tempRows = phasis_result_df[phasis_result_df["identifier"] == i]
        for j in data.columns:
            k = 0.0
            subSetData = tempRows[tempRows["alib"] == j]
            if not subSetData.empty:
                if "non-PHAS" in subSetData["label"].values:
                    k = 0.0
                elif "PHAS" in subSetData["label"].values:
                    if "log_clust_len_norm_counts" in subSetData.columns and len(subSetData["log_clust_len_norm_counts"]) > 0:
                        k = float(subSetData["log_clust_len_norm_counts"].iloc[0])
            data.at[i, j] = k

    chrom_data = [identifier.split(":")[0] for identifier in data.index]
    sorted_indices = sorted(
        range(len(chrom_data)),
        key=lambda idx: int(chrom_data[idx]) if chrom_data[idx].isdigit() else chrom_data[idx],
    )
    data = data.iloc[sorted_indices]
    chrom_data = [chrom_data[idx] for idx in sorted_indices]
    unique_chromosomes = sorted(np.unique(chrom_data), key=lambda x: int(x) if x.isdigit() else x)

    max_value = float(data.max().max())
    if max_value <= 0:
        max_value = 1.0

    f, ax = plt.subplots(figsize=(11, 11))

    colors = ["#C3D8EA", "#3F2F13"]
    cmap = LinearSegmentedColormap.from_list("Custom", colors, 10)
    norm = Normalize(vmin=0, vmax=max_value)

    cax = inset_axes(
        ax,
        width="5%",
        height="9%",
        loc="lower left",
        bbox_to_anchor=(-0.25, 1.01, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, orientation="vertical")
    cbar.set_ticks([0, max_value / 3, 2 * max_value / 3, max_value])
    cbar.set_ticklabels([f"{0:.1f}", f"{max_value / 3:.1f}", f"{2 * max_value / 3:.1f}", f"{max_value:.1f}"])

    heat = sns.heatmap(data, square=False, cmap=cmap, cbar=False, norm=norm, xticklabels=True, yticklabels=False, ax=ax)

    cax2 = f.add_axes([0.17, 0.1, 0.02, 0.8])

    base_cmap = plt.get_cmap("Greys")
    chrom_cmap = base_cmap(np.linspace(0.3, 0.9, len(unique_chromosomes)))
    chrom_color_map = {chrom: idx for idx, chrom in enumerate(unique_chromosomes)}
    chrom_colors = np.array([chrom_color_map[chrom] for chrom in chrom_data]).reshape(-1, 1)

    cax2.imshow(
        chrom_colors,
        cmap=LinearSegmentedColormap.from_list("CustomGreys", chrom_cmap, len(unique_chromosomes)),
        aspect="auto",
    )
    cax2.set_xticks([])
    cax2.set_yticks([])
    cax2.spines["top"].set_visible(False)
    cax2.spines["right"].set_visible(False)
    cax2.spines["bottom"].set_visible(False)
    cax2.spines["left"].set_visible(False)
    cax2.set_ylim(len(data.index), 0)

    chrom_positions = np.array([np.mean(np.where(np.array(chrom_data) == chrom)) for chrom in unique_chromosomes])
    cax2.set_yticks(chrom_positions)
    cax2.set_yticklabels(unique_chromosomes, fontsize=16, rotation=0, ha="center")
    cax2.yaxis.set_tick_params(labelsize=16, pad=11)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)

    fig = heat.get_figure()
    out = _join_outdir(outdir, f"{plot_type}_{phase}_Abundance_PHAS.pdf")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return None


def plot_totalAbundance_heat_map(phasis_result_df, plot_type):
    print("#### Plotting PHAS and non-PHAS Heatmaps ######")

    all_libs = sorted(list(phasis_result_df["alib"].unique())) if (phasis_result_df is not None and not phasis_result_df.empty and "alib" in phasis_result_df.columns) else []

    phasis_result_df = _filter_df_keep_any_phas_call(phasis_result_df)

    data_phas = pd.DataFrame(
        data=0.0,
        columns=all_libs,
        index=list(phasis_result_df["identifier"].unique()) if (phasis_result_df is not None and not phasis_result_df.empty and "identifier" in phasis_result_df.columns) else [],
    )
    data_non_phas = data_phas.copy()

    if data_phas.shape[0] == 0 or data_phas.shape[1] == 0:
        out = _join_outdir(outdir, f"{plot_type}_{phase}_Abundance_PHAS_and_nonPHAS.pdf")
        _save_empty_plot(out, f"{phase}-PHAS {plot_type}", "No loci to plot for this heatmap.")
        return None

    for i in _tqdm(data_phas.index, desc="Processing Rows"):
        tempRows = phasis_result_df[phasis_result_df["identifier"] == i]
        for j in data_phas.columns:
            subSetData = tempRows[tempRows["alib"] == j]
            if not subSetData.empty and "total_abund" in subSetData.columns and len(subSetData["total_abund"]) > 0:
                if "PHAS" in subSetData["label"].values:
                    data_phas.at[i, j] = float(subSetData["total_abund"].iloc[0])
                elif "non-PHAS" in subSetData["label"].values:
                    data_non_phas.at[i, j] = float(subSetData["total_abund"].iloc[0])

    # Apply log10 normalization
    data_phas = np.log10(data_phas.replace(0, np.nan).fillna(1e-10))
    data_non_phas = np.log10(data_non_phas.replace(0, np.nan).fillna(1e-10))

    chrom_data = [identifier.split(":")[0] for identifier in data_phas.index]
    sorted_indices = sorted(
        range(len(chrom_data)),
        key=lambda idx: int(chrom_data[idx]) if chrom_data[idx].isdigit() else chrom_data[idx],
    )
    data_phas = data_phas.iloc[sorted_indices]
    data_non_phas = data_non_phas.iloc[sorted_indices]

    chrom_data = [chrom_data[idx] for idx in sorted_indices]
    unique_chromosomes = sorted(np.unique(chrom_data), key=lambda x: int(x) if x.isdigit() else x)

    max_value_phas = float(data_phas.max().max())
    max_value_non_phas = float(data_non_phas.max().max())
    if max_value_phas <= 0:
        max_value_phas = 1.0
    if max_value_non_phas <= 0:
        max_value_non_phas = 1.0

    f, ax = plt.subplots(figsize=(11, 11))

    phas_colors = ["#D5E6D6", "#FF9999", "#FF0000"]
    non_phas_colors = ["#D5E6D6", "#9999FF", "#0000FF"]

    norm_phas = Normalize(vmin=0, vmax=max_value_phas)
    norm_non_phas = Normalize(vmin=0, vmax=max_value_non_phas)

    cmap_phas = LinearSegmentedColormap.from_list("PHAS", phas_colors, N=256)
    sns.heatmap(data_phas, square=False, cmap=cmap_phas, cbar=False, norm=norm_phas, xticklabels=True, yticklabels=False, ax=ax)

    cmap_non_phas = LinearSegmentedColormap.from_list("non-PHAS", non_phas_colors, N=256)
    sns.heatmap(
        data_non_phas,
        square=False,
        cmap=cmap_non_phas,
        cbar=False,
        norm=norm_non_phas,
        xticklabels=True,
        yticklabels=False,
        ax=ax,
        alpha=0.5,
    )

    cax_phas = inset_axes(
        ax,
        width="5%",
        height="30%",
        loc="lower left",
        bbox_to_anchor=(-0.25, 0.6, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar_phas = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_phas, norm=norm_phas), cax=cax_phas, orientation="vertical")
    cbar_phas.set_ticks([0, max_value_phas / 3, 2 * max_value_phas / 3, max_value_phas])
    cbar_phas.set_ticklabels([f"{0:.1f}", f"{max_value_phas / 3:.1f}", f"{2 * max_value_phas / 3:.1f}", f"{max_value_phas:.1f}"])
    cbar_phas.set_label(r"log of $\it{PHAS}$ abundance", rotation=90, labelpad=15)

    cax_non_phas = inset_axes(
        ax,
        width="5%",
        height="30%",
        loc="lower left",
        bbox_to_anchor=(-0.25, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar_non_phas = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_non_phas, norm=norm_non_phas), cax=cax_non_phas, orientation="vertical")
    cbar_non_phas.set_ticks([0, max_value_non_phas / 3, 2 * max_value_non_phas / 3, max_value_non_phas])
    cbar_non_phas.set_ticklabels([f"{0:.1f}", f"{max_value_non_phas / 3:.1f}", f"{2 * max_value_non_phas / 3:.1f}", f"{max_value_non_phas:.1f}"])
    cbar_non_phas.set_label(r"log of non-$\it{PHAS}$ abundance", rotation=90, labelpad=10)

    cax2 = f.add_axes([0.17, 0.1, 0.02, 0.8])
    base_cmap = plt.get_cmap("Greys")
    chrom_cmap = base_cmap(np.linspace(0.3, 0.9, len(unique_chromosomes)))
    chrom_color_map = {chrom: idx for idx, chrom in enumerate(unique_chromosomes)}
    chrom_colors = np.array([chrom_color_map[chrom] for chrom in chrom_data]).reshape(-1, 1)

    cax2.imshow(
        chrom_colors,
        cmap=LinearSegmentedColormap.from_list("CustomGreys", chrom_cmap, len(unique_chromosomes)),
        aspect="auto",
    )
    cax2.set_xticks([])
    cax2.set_yticks([])

    chrom_positions = np.array([np.mean(np.where(np.array(chrom_data) == chrom)) for chrom in unique_chromosomes])
    cax2.set_yticks(chrom_positions)
    cax2.set_yticklabels(unique_chromosomes, fontsize=12, rotation=0, ha="center")
    cax2.yaxis.set_tick_params(labelsize=16, pad=11)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)

    fig = ax.get_figure()
    out = _join_outdir(outdir, f"{plot_type}_{phase}_Abundance_PHAS_and_nonPHAS.pdf")
    fig.savefig(out, dpi=300)
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
