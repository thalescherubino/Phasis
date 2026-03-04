# phasis/pipeline.py
from __future__ import annotations

import os

from . import cache
from . import parallel as parallel_utils


def run_pipeline() -> int:
    # Ensure headless plotting (safe on macOS/HPC)
    os.environ.setdefault("MPLBACKEND", "Agg")

    # Heavy imports happen here (run path only)
    from . import bridge
    from . import runtime as rt
    from .stages.phase1_pipeline import run_phase1_pipeline
    from .stages.phase2_pipeline import run_phase2_pipeline

    # Sync bridge globals from runtime snapshot/config
    bridge.sync_from_runtime()

    # startup sequence (same order as before)
    bridge.ncores = parallel_utils.coreReserve(bridge.cores)

    # Step-2 bridge (parallel.py will read rt.ncores)
    rt.ncores = bridge.ncores

    bridge.checkDependency()
    libs_checked = bridge.checkLibs()

    # keep legacy + runtime consistent
    bridge.libs = libs_checked
    rt.libs = libs_checked

    # ---- dispatcher (moved out of legacy) ----
    steps_local = getattr(rt, "steps", None) or getattr(bridge, "steps", "both")
    steps_local = str(steps_local).strip().lower()
    cleanup_requested = bool(getattr(rt, "cleanup", False))

    if cleanup_requested and steps_local != "both":
        print("[ERROR] -cleanup is only supported when steps is 'both'.")
        return 1

    if steps_local == "cfind":
        run_phase1_pipeline(libs_checked)
    elif steps_local == "class":
        # run_phase2() will pull cfg from runtime and override clusterFilePaths
        run_phase2_pipeline([])
    elif steps_local == "both":
        clusterFilePaths = run_phase1_pipeline(libs_checked)
        run_phase2_pipeline(clusterFilePaths)
    else:
        raise ValueError(
            f"Unknown steps value: {steps_local!r} (expected 'cfind', 'class', or 'both')"
        )

    if cleanup_requested:
        cache.cleanup(getattr(rt, "run_dir", None))

    return 0
