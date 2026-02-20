from __future__ import annotations

"""
phasis.stages.phase1_pipeline
-----------------------------

Phase I ("cfind") orchestrator.

This module intentionally does NOT import phasis.legacy to avoid circular imports.
Instead, legacy passes in a small bundle of callables (Phase1Hooks) that provide
the legacy implementations of each Phase I step.

Constraints:
- spawn-safe (top-level functions only)
- no nested functions; no imports inside functions
- minimal behavior drift: preserves legacy's execution order and runtime log file
"""

import datetime
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence


@dataclass(frozen=True)
class Phase1Hooks:
    """
    A small bundle of callables that implement the Phase I steps.

    We keep signatures loose (Any) to avoid tight coupling during migration.
    These callables are expected to come from phasis.legacy.
    """
    createfolders: Callable[..., Any]
    getindex: Callable[..., Any]
    libraryprocess: Callable[..., Any]
    mapprocess: Callable[..., Any]
    parserprocess: Callable[..., Any]
    clusterprocess: Callable[..., Any]
    scoringprocess: Callable[..., Any]


def run_phase1_pipeline(
    libs: Sequence[str],
    *,
    cfg: Optional[Any] = None,
    hooks: Optional[Phase1Hooks] = None,
    workdir: Optional[str] = None,
) -> Any:
    """
    Phase I (cfind): preprocess → map → parse → cluster → window scoring.
    Returns clusterFilePaths (same object legacy.main previously used).

    Parameters
    ----------
    libs:
        List of library inputs (already validated by legacy.checkLibs()).
    cfg:
        Reserved for future Phase I config. Currently unused (kept for API symmetry).
    hooks:
        Required. A Phase1Hooks instance providing the legacy implementations.
    workdir:
        Optional override of working directory (defaults to os.getcwd()).
    """
    if hooks is None:
        raise ValueError("run_phase1_pipeline() requires hooks=Phase1Hooks(...)")

    # Preserve legacy behavior: print header + write runtime log
    print("######            Starting Phase I           #########")

    runLog = "runtime_%s" % datetime.datetime.now().strftime("%m_%d_%H_%M")
    fh_run = open(runLog, "w")

    clusterFilePaths = None
    try:
        wd = workdir if workdir is not None else os.getcwd()

        # Create folders, build/reuse index
        clustfolder = hooks.createfolders(wd)
        genoIndex = hooks.getindex(fh_run)

        # Preprocess → map → parse
        libs_processed = hooks.libraryprocess(libs)
        _ = hooks.mapprocess(libs_processed, genoIndex)
        libs_nestdict, libs_poscountdict = hooks.parserprocess(libs_processed)

        # Find clusters
        libs_clustdicts = hooks.clusterprocess(libs_poscountdict, clustfolder)

        # Score (bucketed hashing will skip when cached)
        clusterFilePaths = hooks.scoringprocess(
            libs_processed, libs_clustdicts, libs_nestdict, clustfolder
        )
    finally:
        try:
            fh_run.close()
        except Exception:
            pass

    return clusterFilePaths
