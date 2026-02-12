from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import phasis.runtime as rt


@dataclass(frozen=True)
class Phase2Config:
    # identity/output
    phase: str
    outdir: Optional[str]
    concat_libs: bool

    # orchestration
    steps: str
    class_cluster_file: Any

    # classification
    classifier: str
    phasisScoreCutoff: float
    min_Howell_score: float
    max_complexity: float

    # cache
    memFile: Optional[str]

    @classmethod
    def from_runtime(cls) -> "Phase2Config":
        return cls(
            phase=str(getattr(rt, "phase", "")),
            outdir=getattr(rt, "outdir", None),
            concat_libs=bool(getattr(rt, "concat_libs", False)),
            steps=str(getattr(rt, "steps", "")),
            class_cluster_file=getattr(rt, "class_cluster_file", None),
            classifier=str(getattr(rt, "classifier", "")),
            phasisScoreCutoff=float(getattr(rt, "phasisScoreCutoff", 0.0)),
            min_Howell_score=float(getattr(rt, "min_Howell_score", 0.0)),
            max_complexity=float(getattr(rt, "max_complexity", 1.0)),
            memFile=getattr(rt, "memFile", None),
        )