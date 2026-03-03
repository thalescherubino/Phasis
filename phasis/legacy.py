#!/usr/bin/env python

version = 'v 2.5.3'

##                  Authors : Thales H Cherubino Ribeiro, Atul Kakrana, Blake C. Meyers
##                  Affilations  : Meyers Lab (Donald Danforth Plant Science Center, St. Louis, MO)
##                  License copy: Included and found at https://opensource.org/licenses/Artistic-2.0
#### IMPORTS ##############################################
import phasis.runtime as rt
import os
import threading
from collections import defaultdict, OrderedDict, Counter
from scipy.stats import hypergeom, mannwhitneyu, combine_pvalues
from os.path import expanduser
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, Bbox
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from typing import List, Sequence, Dict, Tuple, Any
from .cache import *   # re-export cache public API
from .cache import __all__  # make "import *" stable
from .cache import MEM_FILE_DEFAULT
from . import parallel as parallel_utils
from . import state as st
from .stages import window_scoring as ws
from phasis.stages import feature_assembly as st_feature_assembly
from phasis import ids as st_ids
from phasis.stages import classify as st_classify
from phasis.stages import output as st_output
from .config import Phase2Config
from phasis.stages import candidates_merge as st_cmerge
from phasis.stages import cluster_aggregation as st_cluster_aggregation
from phasis.stages import phas_clusters as st_phas_clusters
from phasis.stages import window_selection as st_winsel
from phasis.stages import phase2_pipeline as st_phase2_pipeline
from phasis.stages.phase2_pipeline import run_phase2_pipeline
from phasis.stages.phase1_pipeline import run_phase1_pipeline, Phase1Hooks
from phasis.stages import sam_parsing as st_sam_parsing
from phasis.stages import cluster_build as st_cluster_build
from phasis.stages import cluster_scoring as st_cluster_scoring
from phasis.stages import mapping as st_mapping
from phasis.stages import library_processing as st_library_processing
from phasis.stages import indexing as st_indexing
from phasis.stages import input_validation as st_input_validation
from phasis.stages import dependency_check as st_dependency_check
from phasis.stages import folder_setup as st_folder_setup
import phasis.cache as cache
import phasis.index_integrity as index_integrity
import phasis.libprep as libprep


memFile = MEM_FILE_DEFAULT

# Ensure these exist at module scope so workers never hit NameError
outdir = None
phase = None
runtype = None

# ---- legacy config is populated from phasis.runtime (spawn-safe) ----

def sync_from_runtime() -> None:
    """
    Populate legacy module globals from phasis.runtime.
    Call this exactly once at the start of legacy_entrypoint().
    """
    global libs, reference, norm, norm_factor, maxhits, runtype, mindepth
    global uniqueRatioCut, max_complexity, mismat, libformat, phase
    global clustbuffer, phasisScoreCutoff, minClusterLength, window_len, sliding
    global cores, classifier, steps, class_cluster_file, min_Howell_score
    global concat_libs, outdir, memFile

    libs = rt.libs
    reference = rt.reference
    norm = rt.norm
    norm_factor = rt.norm_factor
    maxhits = rt.maxhits
    runtype = rt.runtype
    mindepth = rt.mindepth
    libprep.mindepth = mindepth
    uniqueRatioCut = rt.uniqueRatioCut
    max_complexity = rt.max_complexity
    mismat = rt.mismat
    libformat = rt.libformat
    phase = rt.phase
    clustbuffer = rt.clustbuffer
    phasisScoreCutoff = rt.phasisScoreCutoff
    minClusterLength = rt.minClusterLength
    cores = rt.cores
    classifier = rt.classifier
    steps = rt.steps
    class_cluster_file = rt.class_cluster_file
    min_Howell_score = rt.min_Howell_score
    concat_libs = rt.concat_libs
    outdir = rt.outdir
    window_len = rt.window_len
    sliding = rt.sliding

    # Ensure outdir exists even if someone calls legacy directly
    if outdir:
        outdir_abs = os.path.abspath(os.path.expanduser(outdir))
        if outdir_abs != outdir:
            outdir = outdir_abs
            rt.outdir = outdir_abs
        os.makedirs(outdir, exist_ok=True)

    # Anchor memFile under outdir (prevents collisions across runs)
    # Use runtime override if present; otherwise set a default.
    mem_override = getattr(rt, "memFile", None)
    if mem_override:
        memFile = mem_override
    else:
        if outdir:
            memFile = os.path.join(outdir, MEM_FILE_DEFAULT)
        else:
            memFile = MEM_FILE_DEFAULT
        rt.memFile = memFile

    # Fallbacks (should already be set by CLI, but keep legacy robust)
    if window_len is None or sliding is None:
        if phase and phase > 21:
            window_len, sliding = 26, 8
        else:
            window_len, sliding = 23, 5
        rt.window_len = window_len
        rt.sliding = sliding

    # ------------------------------------------------------------
    # Spawn-safe: rebuild WIN_SCORE_LOOKUP in this process if missing.
    # On macOS (spawn), workers don't inherit populated globals.
    # We load from the scored TSV path saved in rt.clusters_scored_tsv.
    # ------------------------------------------------------------
    try:
        # Prefer the canonical cache owner (phasis.state).
        if not st.WIN_SCORE_LOOKUP:
            p = getattr(rt, "clusters_scored_tsv", None)
            if p and os.path.isfile(p):
                load_win_score_lookup_from_tsv(p)  # wrapper -> st.load_win_score_lookup_from_tsv
    except Exception:
        pass

def checkLibs():
    """
    Thin wrapper to stage implementation (keeps legacy call sites stable).
    """
    return st_input_validation.checkLibs()

## ADVANCED SETTINGS ######################################
UNIQRATIO_HIT   = 2             ## number of hits cutoff to consider sRNA as multihit for computing uniqness ratio of cluster
DOMSIZE_CUT     = 0.50          ## among all sRNAs, the user defined size should be more abundant than this cutoff
WINDOW_SIZE     = 15            ## Arbitrary window size;
                                ## alternatively you can compute max phases from start and end coords
###########################################################


def checkDependency():
    """
    Thin wrapper to stage implementation (keeps legacy call sites stable).
    """
    return st_dependency_check.checkDependency()

def match_pattern(filename, patterns):
    """
    Compatibility wrapper; canonical implementation lives in phasis.cache.
    """
    return cache.match_pattern(filename, patterns)

def cleanup():
    """
    Compatibility wrapper; canonical implementation lives in phasis.cache.
    Uses the recorded run directory when available.
    """
    return cache.cleanup(getattr(rt, "run_dir", None))

def refClean(filename):
    """
    Compatibility wrapper; canonical implementation lives in phasis.stages.indexing.
    """
    return st_indexing.refClean(filename)


def indexBuilder(reference,ncores):
    """
    Compatibility wrapper; canonical implementation lives in phasis.stages.indexing.
    """
    return st_indexing.indexBuilder(reference, ncores)


def getindex(fh_run):
    """
    Thin wrapper to stage implementation (keeps legacy call sites stable).
    """
    return st_indexing.getindex(fh_run)


def indexIntegrityCheck(index):

    """
    Compatibility wrapper; canonical implementation lives in phasis.index_integrity.
    """
    return index_integrity.indexIntegrityCheck(index)


def readMem(memFile):
    """
    Compatibility wrapper:
    - Keeps EXACT signature + return values (memflag, index)
    - Keeps legacy globals for the rest of legacy.py
    - cache.py owns the mem-file parsing/printing
    """
    global existRefHash, existIndexHash, index

    memflag, index, mem = cache.read_mem_verbose(memFile)

    if mem.genomehash is not None:
        existRefHash = str(mem.genomehash)

    if mem.indexhash is not None:
        existIndexHash = str(mem.indexhash)

    if mem.index is not None:
        index = str(mem.index)
    else:
        index = ""

    return bool(memflag), index


def libraryprocess(libs):
    """
    Thin wrapper to stage implementation (keeps legacy call sites stable).
    """
    return st_library_processing.libraryprocess(libs)

def mapprocess(libs, genoIndex):
    """
    Thin wrapper to stage implementation (keeps legacy call sites stable).
    """
    global nproc, nspread

    libs_mapped = st_mapping.mapprocess(
        libs,
        genoIndex,
        ncores_local=ncores,
    )

    return libs_mapped

def createfolders(currdir):
    """
    Thin wrapper to stage implementation (keeps legacy call sites stable).
    """
    return st_folder_setup.createfolders(currdir)

def coreReserve(cores):
    """
    Compatibility wrapper; canonical implementation lives in phasis.parallel.
    """
    return parallel_utils.coreReserve(cores)

def process_single_lib_cluster(filename):
    return st_cluster_aggregation.process_single_lib_cluster(filename)
def aggregate_and_write_processed_clusters(clusterFiles, memFile_override=None):
    return st_cluster_aggregation.aggregate_and_write_processed_clusters(
        clusterFiles,
        memFile=memFile_override,
    )

# ---- Single-library preprocessing ----
def preprocess_single_library_clusters(mergedClusters):
    """
    Compatibility wrapper; canonical implementation lives in phasis.ids.
    """
    return st_ids.preprocess_single_library_clusters(mergedClusters)

# ---- Step 2: assemble candidate sets per chromosome ----
def assemble_clusters_by_chromosome(mergedClusters_chr):
    """
    Compatibility wrapper; canonical implementation lives in phasis.ids.
    """
    return st_ids.assemble_clusters_by_chromosome(mergedClusters_chr)

# ---- Step 3: assign final IDs per chromosome (genomic span keys) ----
def assign_final_ids_by_chromosome(chromosome_df):
    """
    Compatibility wrapper; canonical implementation lives in phasis.ids.
    """
    return st_ids.assign_final_ids_by_chromosome(chromosome_df)

def assign_final_cluster_ids(mergedClusterDict, allClusters):
    """
    Compatibility wrapper; canonical implementation lives in phasis.ids.
    """
    return st_ids.assign_final_cluster_ids(mergedClusterDict, allClusters)

MERGED_REVERSE_BUILT = False

def _ensure_reverse_index() -> dict:
    """
    Compatibility wrapper; canonical implementation lives in phasis.ids.
    """
    return st_ids._ensure_reverse_index()

def _strip_fileprefix_from_id(
    cid: str,
    lib: str | None = None,
    phase: str | int | None = None,
    alib: str | None = None,
) -> str:
    """
    Compatibility wrapper; canonical implementation lives in phasis.ids.

    Accepts both the newer ``lib=`` form and the legacy ``alib=`` form.
    """
    if lib is None and alib is not None:
        lib = alib
    return st_ids._strip_fileprefix_from_id(cid, lib=lib, phase=phase)

def _normalize_cluster_id_for_lookup(x: str) -> str:
    """
    Compatibility wrapper; canonical implementation lives in phasis.ids.
    """
    return st_ids.normalize_cluster_id_for_lookup(x, phase=getattr(rt, "phase", None))

def process_chromosome_data(loci_group):
    """
    Compatibility wrapper; canonical implementation lives in phasis.stages.phas_clusters.
    """
    return st_phas_clusters.process_chromosome_data(loci_group)

def assemble_candidate_clusters_parametric(
    mergedClusters: Sequence[Sequence[Any]],
    allClusters_df: pd.DataFrame,
    phase: str,
) -> Dict[str, List[str]]:
    """
    Compatibility wrapper; canonical implementation lives in phasis.ids.
    """
    return st_ids.assemble_candidate_clusters_parametric(mergedClusters, allClusters_df, phase)

def merge_candidate_clusters_parametric(loci_df, allClusters_df, phase, memFile, **kwargs):
    return st_ids.merge_candidate_clusters_parametric(
        loci_df, allClusters_df, phase, memFile, **kwargs
    )


_MERGED_DICT_LOCK = threading.Lock()
_MERGED_REVERSE_BUILT = False

def _identity_dict_from_tsv_firstcol(path: str, id_col=("Cluster","cluster","clusterID","name","cID")):
    """
    Compatibility wrapper; canonical implementation lives in phasis.ids.
    """
    return st_ids._identity_dict_from_tsv_firstcol(path, id_col=id_col)

def _set_reverse_merged_map(mcd: dict) -> None:
    """
    Compatibility wrapper; canonical implementation lives in phasis.ids.
    """
    return st_ids._set_reverse_merged_map(mcd)

def _load_simple_tab_dict(path: str) -> dict:
    """
    Compatibility wrapper; canonical implementation lives in phasis.ids.
    """
    return st_ids._load_simple_tab_dict(path)

def ensure_mergedClusterDict_always(*, concat_libs, phase, merged_out_path, loci_table_df, allClusters_df, memFile):
    return st_ids.ensure_mergedClusterDict_always(
        concat_libs=concat_libs,
        phase=phase,
        merged_out_path=merged_out_path,
        loci_table_df=loci_table_df,
        allClusters_df=allClusters_df,
        memFile=memFile,
    )

def load_processed_clusters_fallback(phase: str) -> pd.DataFrame:
    """
    Compatibility wrapper; canonical implementation lives in phasis.stages.phas_clusters.
    Keeps the legacy signature but ignores the explicit phase argument.
    """
    return st_phas_clusters.load_processed_clusters_fallback()

def _normalize_cluster_df(df: pd.DataFrame, is_concat: bool) -> pd.DataFrame:
    """
    Compatibility wrapper; canonical implementation lives in phasis.stages.phase2_pipeline.
    """
    return st_phase2_pipeline._normalize_cluster_df(df, is_concat)

def set_win_score_lookup(win_df: pd.DataFrame) -> dict:
    """
    Build and set a compact lookup: cID -> (phasis_score, combined_fishers).
    Backward-compatible wrapper around phasis.state.
    """
    return st.set_win_score_lookup(win_df)


def load_win_score_lookup_from_tsv(path: str) -> dict:
    """
    Backward-compatible wrapper around phasis.state.
    Required on macOS (spawn) where workers do NOT inherit parent globals.
    """
    return st.load_win_score_lookup_from_tsv(path)

def clear_win_score_lookup() -> None:
    """Clear the process-local lookup (wrapper)."""
    st.clear_win_score_lookup()

# --- helpers ---------------------------------------------------------------

DCL_OVERHANG = 3          # 2-nt 3' overhang in duplex -> 3-nt genomic offset
WINDOW_MULTIPLIER = 10    # 10 cycles per window
# ---------- Howell utilities (exact-phase only) ----------

def _strand_masks(df: pd.DataFrame):
    """
    Compatibility wrapper; canonical implementation lives in phasis.stages.feature_assembly.
    """
    return st_feature_assembly._strand_masks(df)

def _build_pos_abun_exact_phase(df: pd.DataFrame, seq_start: int, seq_end: int, phase: int):
    """
    Compatibility wrapper; canonical implementation lives in phasis.stages.feature_assembly.
    """
    return st_feature_assembly._build_pos_abun_exact_phase(df, seq_start, seq_end, phase)

def best_sliding_window_score_forward(pos_abun, phase, win_size, seq_start=None, seq_end=None):
    return _best_sliding_window_score_generic(
        pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=True
    )


def best_sliding_window_score_reverse(pos_abun, phase, win_size, seq_start=None, seq_end=None):
    return _best_sliding_window_score_generic(
        pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=False
    )

# ---------- STRICT Howell (NO positional wobble; still ONLY len == phase) ----------
def _evaluate_register_strict_exact(window_positions, pos_abun, win_start, win_end, phase, reg, forward=True):
    """
    Compatibility wrapper; canonical implementation lives in phasis.stages.feature_assembly.
    """
    return st_feature_assembly._evaluate_register_strict_exact(
        window_positions, pos_abun, win_start, win_end, phase, reg, forward=forward
    )

def _evaluate_register(window_positions, pos_abun, win_start, win_end, phase, reg, forward=True):
    """
    Compatibility wrapper; canonical implementation lives in phasis.stages.feature_assembly.
    """
    return st_feature_assembly._evaluate_register(
        window_positions, pos_abun, win_start, win_end, phase, reg, forward=forward
    )

def best_sliding_window_score_forward_strict(pos_abun, phase, win_size, seq_start=None, seq_end=None):
    return _best_sliding_window_score_generic_strict(
        pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=True
    )


def best_sliding_window_score_reverse_strict(pos_abun, phase, win_size, seq_start=None, seq_end=None):
    return _best_sliding_window_score_generic_strict(
        pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=False
    )

def _best_sliding_window_score_generic(pos_abun, phase, win_size, seq_start=None, seq_end=None, forward=True):
    return st_feature_assembly._best_sliding_window_score_generic(
        pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=forward
    )

def compute_phasing_score_Howell(aclust: pd.DataFrame):
    return st_feature_assembly.compute_phasing_score_Howell(aclust)

def _best_sliding_window_score_generic_strict(pos_abun, phase, win_size, seq_start=None, seq_end=None, forward=True):
    return st_feature_assembly._best_sliding_window_score_generic_strict(
        pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=forward
    )

def compute_phasing_score_Howell_strict(aclust: pd.DataFrame):
    return st_feature_assembly.compute_phasing_score_Howell_strict(aclust)

# ---------------------------------------------------------------------
# Output + plotting stage wrappers (extracted to phasis.stages.output)
# ---------------------------------------------------------------------

def format_attributes(row):
    return st_output.format_attributes(row)


def write_gff(phasis_result_df, gff_filename):
    # ensure stage module sees the same phase as legacy
    st_output.phase = phase
    return st_output.write_gff(phasis_result_df, gff_filename)


def plot_report_heat_map(phasis_result_df, plot_type):
    st_output.outdir = outdir
    st_output.phase = phase
    return st_output.plot_report_heat_map(phasis_result_df, plot_type)


def plot_phasAbundance_heat_map(phasis_result_df, plot_type):
    st_output.outdir = outdir
    st_output.phase = phase
    return st_output.plot_phasAbundance_heat_map(phasis_result_df, plot_type)


def plot_totalAbundance_heat_map(phasis_result_df, plot_type):
    st_output.outdir = outdir
    st_output.phase = phase
    return st_output.plot_totalAbundance_heat_map(phasis_result_df, plot_type)


def _plot_wrapper(job):
    return st_output._plot_wrapper(job)


def _finalize_and_write_results(
    method_name: str,
    features: pd.DataFrame,
    *,
    job_outdir: str | None = None,
    job_phase: str | int | None = None,
):
    """
    Finalize outputs via output stage without depending on legacy globals.

    - If job_outdir/job_phase are not provided, fall back to rt.* (not legacy module globals).
    """
    if job_outdir is None:
        job_outdir = getattr(rt, "outdir", None)

    if job_phase is None:
        job_phase = getattr(rt, "phase", None)

    return st_output.finalize_and_write_results(
        method_name,
        features,
        job_outdir=job_outdir,
        job_phase=job_phase,
    )

def KNN_phas_clustering(
    features: pd.DataFrame,
    *,
    cfg: Phase2Config | None = None,
    phasisScoreCutoff: float | None = None,
    min_Howell_score: float | None = None,
    max_complexity: float | None = None,
    job_outdir: str | None = None,
    job_phase: str | int | None = None,
):
    """
    KNN classifier (legacy wrapper).
    Behavior preserved; now supports explicit args/config.
    """
    print("### KNN classifier ###")

    if cfg is not None:
        phasisScoreCutoff = cfg.phasisScoreCutoff
        min_Howell_score = cfg.min_Howell_score
        max_complexity = cfg.max_complexity
        job_outdir = cfg.outdir
        job_phase = cfg.phase

    if phasisScoreCutoff is None:
        phasisScoreCutoff = globals()["phasisScoreCutoff"]
    if min_Howell_score is None:
        min_Howell_score = globals()["min_Howell_score"]
    if max_complexity is None:
        max_complexity = globals()["max_complexity"]

    labeled = st_classify.knn_classify(
        features,
        phasisScoreCutoff=float(phasisScoreCutoff),
        min_Howell_score=float(min_Howell_score),
        max_complexity=float(max_complexity),
    )

    _finalize_and_write_results("KNN", labeled, job_outdir=job_outdir, job_phase=job_phase)


def GMM_phas_clustering(
    features: pd.DataFrame,
    n_clusters: int = 2,
    *,
    cfg: Phase2Config | None = None,
    phasisScoreCutoff: float | None = None,
    min_Howell_score: float | None = None,
    max_complexity: float | None = None,
    job_outdir: str | None = None,
    job_phase: str | int | None = None,
):
    """
    GMM classifier (legacy wrapper).
    Behavior preserved; now supports explicit args/config.
    """
    print("### GMM classifier ###")

    if cfg is not None:
        phasisScoreCutoff = cfg.phasisScoreCutoff
        min_Howell_score = cfg.min_Howell_score
        max_complexity = cfg.max_complexity
        job_outdir = cfg.outdir
        job_phase = cfg.phase

    if phasisScoreCutoff is None:
        phasisScoreCutoff = globals()["phasisScoreCutoff"]
    if min_Howell_score is None:
        min_Howell_score = globals()["min_Howell_score"]
    if max_complexity is None:
        max_complexity = globals()["max_complexity"]

    labeled = st_classify.gmm_classify(
        features,
        phasisScoreCutoff=float(phasisScoreCutoff),
        min_Howell_score=float(min_Howell_score),
        max_complexity=float(max_complexity),
        n_clusters=int(n_clusters),
    )

    _finalize_and_write_results("GMM", labeled, job_outdir=job_outdir, job_phase=job_phase)

def compute_scores_for_group(chromosome_data_group):
    return ws.compute_scores_for_group(chromosome_data_group)


def _record_clusters_scored_tsv_path(path: str) -> None:
    return ws._record_clusters_scored_tsv_path(path)


def infer_library_from_cluster_id(cid: str, phase_value: int) -> str:
    return ws.infer_library_from_cluster_id(cid, phase_value)


def compute_and_save_phasis_scores(clusters: pd.DataFrame) -> pd.DataFrame:
    return ws.compute_and_save_phasis_scores(clusters)

def ensure_mergedClusterDict(phase: str):
    return st_ids.ensure_mergedClusterDict(phase)


def getUniversalID(clusterID: str):
    return st_ids.getUniversalID(clusterID)


def process_chromosome_features(chromosome_df):
    return st_feature_assembly.process_chromosome_features(chromosome_df)


def features_to_detection(
    clusters_data,
    *,
    phase=None,
    outdir=None,
    concat_libs=None,
    memFile=None,
    outfname=None,
):
    return st_feature_assembly.features_to_detection(
        clusters_data,
        phase=phase,
        outdir=outdir,
        concat_libs=concat_libs,
        memFile=memFile,
        outfname=outfname,
    )

def chromosome_clusters_to_candidate_loci(chromosome_df, **kwargs):
    """Legacy compatibility wrapper → stage implementation."""
    return st_cmerge.chromosome_clusters_to_candidate_loci(chromosome_df, **kwargs)


def loci_table_from_clusters(allClusters, **kwargs):
    """Legacy compatibility wrapper → stage implementation."""
    return st_cmerge.loci_table_from_clusters(allClusters, **kwargs)


def merge_candidate_clusters_across_libs(loci_table_path: str, out_path: str, **kwargs):
    """Legacy compatibility wrapper → stage implementation."""
    return st_cmerge.merge_candidate_clusters_across_libs(loci_table_path, out_path, **kwargs)

def build_and_save_phas_clusters(allClusters):
    return st_phas_clusters.build_and_save_phas_clusters(allClusters)

def select_scoring_windows(clusters_data, **kwargs):
    return st_winsel.select_scoring_windows(clusters_data, **kwargs)

def run_phase2(clusterFilePaths, cfg: Phase2Config | None = None):
    return run_phase2_pipeline(clusterFilePaths, cfg=cfg)

def parserprocess(libs, load_dicts=False):
    return st_sam_parsing.parserprocess(libs, load_dicts=load_dicts)

def run_phase1(libs_checked, cfg=None):
    hooks = Phase1Hooks(
        createfolders=createfolders,
        getindex=getindex,
        libraryprocess=libraryprocess,
        mapprocess=mapprocess,
        parserprocess=parserprocess,
        clusterprocess=clusterprocess,
        scoringprocess=scoringprocess,
    )
    return run_phase1_pipeline(libs_checked, cfg=cfg, hooks=hooks)

def clusterprocess(libs_poscountdict, clustfolder):
    return st_cluster_build.clusterprocess(libs_poscountdict, clustfolder)

def scoringprocess(
    libs,
    libs_clustdicts,
    libs_nestdict,
    clustfolder,
    force_rescore=False,
    verify_outputs=True,
    scored_dir=None,
    purge_existing=False,
    concat_mode=None,
    merged_name="ALL_LIBS",
):
    return st_cluster_scoring.scoringprocess(
        libs=libs,
        libs_clustdicts=libs_clustdicts,
        libs_nestdict=libs_nestdict,
        clustfolder=clustfolder,
        force_rescore=force_rescore,
        verify_outputs=verify_outputs,
        scored_dir=scored_dir,
        purge_existing=purge_existing,
        concat_mode=concat_mode,
        merged_name=merged_name,
    )

def legacy_entrypoint():
    """
    Compatibility wrapper: top-level dispatch now lives in phasis.pipeline.
    """
    from phasis.pipeline import run_pipeline

    return run_pipeline()


if __name__ == "__main__":
    legacy_entrypoint()
