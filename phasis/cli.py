# phasis/cli.py
from __future__ import annotations

import sys
from typing import List, Optional

from .deps_check import require_dependencies


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # Make legacy argparse see the same argv it used to see
    sys.argv = ["phasis"] + list(argv)

    # Keep your current behavior for now (later you can skip this for -h/--help)
    require_dependencies()

    # Import legacy as a real module so multiprocessing can pickle functions
    from . import legacy

    # Run the legacy pipeline through the stable entrypoint
    legacy.legacy_entrypoint()
    return 0