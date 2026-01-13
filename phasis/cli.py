# phasis/cli.py
from __future__ import annotations

import runpy
import sys
from typing import List, Optional

from .deps_check import require_dependencies


# Run checks at import time, before legacy execution.
require_dependencies()


def main(argv: Optional[List[str]] = None) -> int:
    # Entry point for the `phasis` command.
    # Runs the legacy module as __main__ so its argparse/globals work unchanged.
    if argv is None:
        argv = sys.argv[1:]

    # Emulate: python phasis.py <args>
    sys.argv = ["phasis"] + list(argv)

    # Execute phasis.legacy as a script.
    runpy.run_module("phasis.legacy", run_name="__main__")
    return 0

