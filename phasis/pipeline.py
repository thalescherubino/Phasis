# phasis/pipeline.py
from __future__ import annotations

import os


def run_pipeline() -> int:
    # Ensure headless plotting (safe on macOS/HPC)
    os.environ.setdefault("MPLBACKEND", "Agg")

    # Heavy import happens here (run path only)
    from . import legacy

    # If you still need legacy globals mirrored from runtime,
    # keep doing that in cli.py for now (minimal change strategy).
    legacy.legacy_entrypoint()
    return 0