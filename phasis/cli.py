# phasis/cli.py
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional
from .pipeline import run_pipeline

from . import __version__
from . import runtime as rt
from .deps_check import require_dependencies


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    reqflags = parser.add_argument_group("required arguments")
    reqflags.add_argument("-libs", help="Quality controlled libraries to process", required=False, nargs="*")
    reqflags.add_argument("-reference", help="Genome or transcriptome reference FASTA", required=False)

    parser.add_argument("-maxhits", default=25, type=int,
                        help="Number of genome/transcriptome hits passed as -k to hisat2 [default 25]")
    parser.add_argument("-runtype", default="G", type=str,
                        help="G: genome | T: transcriptome | S: scaffolded genome [default G]")
    parser.add_argument("-mindepth", default=2, type=int,
                        help="Minimum depth for p-value computation [default 2]")
    parser.add_argument("-force", action="store_true",
                        help="Allow resource-intensive parameter combinations", required=False)
    parser.add_argument("-uniqueRatioCut", default=0.2, type=float,
                        help="Proportion of uniquely mapped reads filter [default 0.2]")
    parser.add_argument("-max_complexity", default=0.3, type=float,
                        help="Max complexity filter [default 0.3]")
    parser.add_argument("-mismat", default=0, type=int,
                        help="Mismatches allowed for mapping [default 0]")
    parser.add_argument("-libformat", default="F", type=str,
                        help="QC format: FASTA (F) or tag-count (T) [default F]")
    parser.add_argument("-phase", default=21, type=int,
                        help="Desired phase length [default 21]")
    parser.add_argument("-clustbuffer", default=300, type=int,
                        help="Max distance to merge clusters [default 300]")
    parser.add_argument("-phasisScoreCutoff", default=50, type=int,
                        help="Min score to report PHAS [21:50 | 24:250]")
    parser.add_argument("-minClusterLength", default=350, type=int,
                        help="Min length to score a PHAS locus [default 350]")
    parser.add_argument("-cores", default=0, type=int,
                        help="0: most free cores | >0: exact cores [default 0]")
    parser.add_argument("-norm", action="store_true",
                        help="Enable normalization (CP10M) [default False]")
    parser.add_argument("-norm_factor", type=float, default=1e7,
                        help="Normalization factor (default 1e7)")
    parser.add_argument("-classifier", default="KNN", type=str,
                        help="Classifier: KNN or GMM [default KNN]")
    parser.add_argument("-cleanup", action="store_true",
                        help="Delete intermediate files (not recommended for successive runs)")
    parser.add_argument("-steps", default="both", type=str,
                        help="both | cfind | class [default both]")
    parser.add_argument("-class_cluster_file", nargs="*",
                        help="Cluster file(s) for -steps class")
    parser.add_argument("-min_Howell_score", type=float, default=12.5,
                        help="Minimum Howell score [default 12.5]")
    parser.add_argument("--concat_libs", dest="concat_libs", action="store_true",
                        help="Concatenate all input libs into a single virtual library")
    parser.add_argument("--outdir", dest="outdir", metavar="DIR", type=str,
                        default="{phase}_results",
                        help="Output directory (supports {phase}); default {phase}_results")

    parser.add_argument(
        "-version", action="version",
        help="Print the version and quit",
        version="%(prog)s " + __version__,
    )

    return parser


def _normalize_outdir(outdir: str, phase: int) -> str:
    if not outdir or outdir == "{phase}_results":
        outdir = f"{phase}_results"
    else:
        outdir = outdir.replace("{phase}", str(phase))

    outdir = os.path.abspath(os.path.expanduser(outdir))
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _validate_args(args: argparse.Namespace) -> None:
    if args.classifier not in ("KNN", "GMM"):
        print("\nERROR: Wrong classifier option (use KNN or GMM)\n")
        raise SystemExit(2)

    if args.steps not in ("both", "cfind", "class"):
        print("ERROR: Invalid value for -steps. Use: both, cfind, class\n")
        raise SystemExit(2)


def configure_runtime(args: argparse.Namespace) -> None:
    """
    Single source of truth: write configuration into phasis.runtime (rt.*).
    Legacy globals will be mirrored from rt right before execution.
    """
    _validate_args(args)

    # phase-dependent window settings + cutoff clamping
    if args.phase > 21:
        window_len, sliding = 26, 8
        score_cutoff = int(args.phasisScoreCutoff)
        min_cutoff, max_cutoff = 250, 300
        orig = score_cutoff
        if score_cutoff < min_cutoff:
            score_cutoff = min_cutoff
            print(f"[INFO] phase={args.phase} > 21: phasisScoreCutoff raised from {orig} to {score_cutoff} (min {min_cutoff}).")
        elif score_cutoff > max_cutoff:
            score_cutoff = max_cutoff
            print(f"[INFO] phase={args.phase} > 21: phasisScoreCutoff lowered from {orig} to {score_cutoff} (max {max_cutoff}).")
        else:
            print(f"[INFO] phase={args.phase} > 21: phasisScoreCutoff kept at {score_cutoff} (within [{min_cutoff}, {max_cutoff}]).")
    else:
        window_len, sliding = 23, 5
        score_cutoff = int(args.phasisScoreCutoff)

    outdir = _normalize_outdir(args.outdir, args.phase)

    # write everything to runtime
    rt.libs = args.libs
    rt.reference = args.reference
    rt.norm = args.norm
    rt.norm_factor = args.norm_factor
    rt.maxhits = args.maxhits
    rt.runtype = args.runtype
    rt.mindepth = args.mindepth
    rt.uniqueRatioCut = args.uniqueRatioCut
    rt.max_complexity = args.max_complexity
    rt.mismat = args.mismat
    rt.libformat = args.libformat
    rt.phase = args.phase
    rt.clustbuffer = args.clustbuffer
    rt.phasisScoreCutoff = score_cutoff
    rt.minClusterLength = args.minClusterLength
    rt.window_len = window_len
    rt.sliding = sliding
    rt.cores = args.cores
    rt.classifier = args.classifier
    rt.steps = args.steps
    rt.class_cluster_file = args.class_cluster_file
    rt.min_Howell_score = args.min_Howell_score
    rt.concat_libs = args.concat_libs
    rt.outdir = outdir

    # store optional flags too (even if runtime.py doesn't predeclare them yet)
    rt.force = args.force
    rt.cleanup = args.cleanup
    rt.run_dir = os.path.abspath(os.getcwd())
    rt.memFile = os.path.join(rt.run_dir, "phasis.mem")

def sync_legacy_globals_from_runtime(legacy_module) -> None:
    """
    Backward-compat bridge: legacy still reads many globals directly.
    Mirror rt.* into legacy module attributes so the old code keeps working.
    """
    names = [
        "libs", "reference", "norm", "norm_factor", "maxhits", "runtype", "mindepth",
        "uniqueRatioCut", "max_complexity", "mismat", "libformat", "phase", "clustbuffer",
        "phasisScoreCutoff", "minClusterLength", "window_len", "sliding", "cores",
        "classifier", "steps", "class_cluster_file", "min_Howell_score", "concat_libs",
        "outdir", "memFile",
        # optional (may be referenced in some places)
        "force", "cleanup",
        # state objects (phase II)
        "mergedClusterDict", "mergedClusterReverse", "WIN_SCORE_LOOKUP",
    ]
    for n in names:
        if hasattr(rt, n):
            setattr(legacy_module, n, getattr(rt, n))


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    args = parser.parse_args(argv)

    # Dependencies should be checked only in the real run path.
    # (argparse will have already exited for -h/-version)
    require_dependencies()

    configure_runtime(args)

    # MUST be before any multiprocessing pool creation
    import phasis.runtime as rt
    rt.save_snapshot()

    # IMPORTANT for macOS + multiprocessing + matplotlib:
    # set this BEFORE importing anything that might import matplotlib.
    os.environ.setdefault("MPLBACKEND", "Agg")

    # Prefer: run pipeline orchestration (which may call legacy under the hood)
    from . import legacy
    sync_legacy_globals_from_runtime(legacy)
    return run_pipeline()
