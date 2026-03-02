# phasis/pipeline.py
from __future__ import annotations

import os


def run_pipeline() -> int:
    # Ensure headless plotting (safe on macOS/HPC)
    os.environ.setdefault("MPLBACKEND", "Agg")

    # Heavy imports happen here (run path only)
    from . import legacy
    from . import runtime as rt

    # Sync legacy globals from runtime snapshot/config
    legacy.sync_from_runtime()

    # legacy startup sequence (same order as before)
    legacy.ncores = legacy.coreReserve(legacy.cores)

    # Step-2 bridge (parallel.py will read rt.ncores)
    rt.ncores = legacy.ncores

    legacy.checkDependency()
    libs_checked = legacy.checkLibs()

    # keep legacy + runtime consistent
    legacy.libs = libs_checked
    rt.libs = libs_checked

    # ---- dispatcher (moved out of legacy) ----
    steps_local = getattr(rt, "steps", None) or getattr(legacy, "steps", "both")
    steps_local = str(steps_local).strip().lower()

    if steps_local == "cfind":
        legacy.run_phase1(libs_checked)
    elif steps_local == "class":
        # run_phase2() will pull cfg from runtime and override clusterFilePaths
        legacy.run_phase2([])
    elif steps_local == "both":
        clusterFilePaths = legacy.run_phase1(libs_checked)
        legacy.run_phase2(clusterFilePaths)
    else:
        raise ValueError(
            f"Unknown steps value: {steps_local!r} (expected 'cfind', 'class', or 'both')"
        )

    return 0
