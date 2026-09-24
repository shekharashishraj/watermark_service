"""End-to-end entry points: build a baseline, inspect a walkthrough against it."""

from __future__ import annotations

import time

from .detect import DetectConfig, InspectionResult, detect_changes
from .model import BaselineModel, build_baseline
from .register import localize
from .session import Session


def build(session: Session, rooms=None, voxel: float = 0.05) -> BaselineModel:
    return build_baseline(session, rooms=rooms, voxel=voxel)


def inspect(model: BaselineModel, session: Session, cfg: DetectConfig | None = None,
            refine_chunks: bool = True) -> InspectionResult:
    t0 = time.time()
    reg = localize(model, session, refine_chunks=refine_chunks)
    result = detect_changes(model, session, reg, cfg)
    result.timings["localize_s"] = round(reg.seconds, 2)
    result.timings["inspection_total_s"] = round(time.time() - t0, 2)
    result.timings["frames"] = len(session)
    return result
