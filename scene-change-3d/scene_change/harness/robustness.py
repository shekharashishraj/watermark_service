"""Robustness sweeps: change detection under controlled capture degradation.

    python -m scene_change.harness.robustness --bench runs/bench --out runs/robust \
        --seeds 0 1 2 3 4 --stressors blur dark bright relight coverage sparse \
        --methods ours video2d oscd --stride 2

For every scenario (``<bench>/scenarios/s<seed>_<kind>``, generated if missing), the
inspection walkthrough (every ``stride``-th frame) is degraded by each stressor of
``scene_change.stress.STRESSORS`` at each severity, and every method runs on the
degraded walkthrough against the same clean baseline. Ground truth is unchanged by
construction. One JSON line per (scenario, stressor, level, method variant) goes to
``<out>/results.jsonl``; finished entries are skipped on restart.

Method variants
    ours          3D geometric + appearance detector, all flags (offline: whole walk)
    ours-confirmed  same run, confirmed detections only
    video2d       2D video comparison (per frame)
    oscd-online   O-SCD re-implementation, mask rendered right after each frame
    oscd-offline  same run after refinement over all frames (to 3000 iterations)

Each record holds the PASLCD frame scores at the method's own threshold, pixel scores,
an error breakdown (false positives next to a real change vs elsewhere, misses per
change), calibration statistics of the continuous scores (fine histogram, ECE, Brier,
AP, AUROC, threshold sweep, confidently-wrong errors), per-frame error counts and, for
``ours``, the object-level metrics and per-detection scores.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
from scipy import ndimage

from .. import calibration as cal
from ..baseline2d import build_retrieval, compare_videos
from ..oscd import OSCDConfig, ReferenceIndex, make_encoder, run_oscd
from ..pipeline import build, inspect
from ..stress import STRESSORS, apply_stressor
from .benchmark import get_scenario
from .evaluate import (evaluate_3d, frame_metrics, gt_masks, gt_regions, match_detections, pixel_metrics,
                       predicted_masks, predicted_scores, registration_errors)

INSP_KEYS = ("insp_poses_world", "insp_oid", "insp_oid_basestate")


# ----------------------------------------------------------------------------
# Ground-truth helpers
# ----------------------------------------------------------------------------

def change_masks(arrays: dict, changes: list) -> dict:
    """Per GT change id: (N, H, W) mask of the pixels showing it (as ``gt_masks``)."""
    ob, oi = arrays["insp_oid_basestate"], arrays["insp_oid"]
    out = {}
    for c in changes:
        m = np.zeros(oi.shape, bool)
        if c["type"] in ("removed", "moved") or c.get("swap"):
            m |= ob == c["oid"]
        if c["type"] in ("added", "moved", "appearance"):
            m |= oi == c["oid"]
        out[c["id"]] = m
    return out


def change_size(c: dict) -> dict:
    boxes = [np.asarray(c[k], np.float64) for k in ("aabb_before", "aabb_after") if c.get(k) is not None]
    if not boxes:
        return {"dims": None, "volume": None, "area": None}
    d = np.sort(boxes[0][1] - boxes[0][0])[::-1]
    return {"dims": np.round(d, 3).tolist(), "volume": float(np.prod(d)), "area": float(d[0] * d[1])}


def subset_arrays(arrays: dict, idx) -> dict:
    return {k: (v[idx] if k in INSP_KEYS else v) for k, v in arrays.items()}


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

def error_breakdown(pred: np.ndarray, gt: np.ndarray, band: int = 3) -> dict:
    """False positives within ``band`` px of a real change (boundary / misalignment) vs elsewhere."""
    near = ndimage.binary_dilation(gt, structure=np.ones((1, 3, 3), bool), iterations=band)
    has = gt.any(axis=(1, 2))
    fp = pred & ~gt
    return {"tp_px": int(np.sum(pred & gt)), "fn_px": int(np.sum(~pred & gt)), "fp_px": int(fp.sum()),
            "fp_near_px": int(np.sum(fp & near)), "fp_far_px": int(np.sum(fp & ~near)),
            "fp_changefree_px": int(fp[~has].sum()), "px": int(gt.size),
            "changefree_frames": int((~has).sum()),
            "changefree_frames_flagged": int(np.sum(np.sum(fp[~has], axis=(1, 2)) >= 20))}


def per_change(pred: np.ndarray, scores: np.ndarray | None, cmasks: dict, changes: list,
               min_hit: float = 0.25) -> list[dict]:
    out = []
    for c in changes:
        m = cmasks[c["id"]]
        n = int(m.sum())
        row = {"id": c["id"], "type": c["type"], "name": c["name"], "room": c["room"], "gt_px": n,
               "frames": int(np.sum(m.any(axis=(1, 2)))), **change_size(c)}
        if n:
            hit = float(np.sum(pred & m) / n)
            row.update(hit_frac=round(hit, 4), detected=hit >= min_hit)
            if scores is not None:
                s = np.asarray(scores[m], np.float32)
                row.update(score_mean=round(float(s.mean()), 4), score_max=round(float(s.max()), 4))
        out.append(row)
    return out


def per_frame(pred: np.ndarray, gt: np.ndarray) -> dict:
    ax = (1, 2)
    return {"tp": np.sum(pred & gt, axis=ax).astype(int).tolist(), "fp": np.sum(pred & ~gt, axis=ax).astype(int).tolist(),
            "fn": np.sum(~pred & gt, axis=ax).astype(int).tolist()}


def score_variant(pred, scores, gt, cmasks, changes, threshold: float) -> dict:
    rec = {"frames": frame_metrics(pred, gt), "pixels": pixel_metrics(pred, gt), "errors": error_breakdown(pred, gt),
           "changes": per_change(pred, scores, cmasks, changes), "per_frame": per_frame(pred, gt)}
    if scores is not None:
        rec["calibration"] = cal.calibration_report(scores, gt, threshold)
    return rec


def region_flags(result, model, changes: list, T_wb: np.ndarray, pad: float = 0.1) -> dict:
    """Per GT change: was its region reported unverified by our detector (>= half of its baseline surfaces)?"""
    mu = model.gaussians.means.astype(np.float64)
    Pw = mu @ T_wb[:3, :3].T + T_wb[:3, 3]
    out = {}
    for c in changes:
        inside = np.zeros(len(Pw), bool)
        for k in ("aabb_before", "aabb_after"):
            if c.get(k) is None:
                continue
            lo, hi = np.asarray(c[k][0]) - pad, np.asarray(c[k][1]) + pad
            inside |= np.all((Pw >= lo) & (Pw <= hi), axis=1)
        out[c["id"]] = None if inside.sum() < 5 else bool(np.mean(result.status[inside] == 3) >= 0.5)
    return out


# ----------------------------------------------------------------------------
# Scene context
# ----------------------------------------------------------------------------

class SceneContext:
    """A scenario with its baseline model and per-method indexes, shared by all stress levels."""

    def __init__(self, bench: Path, seed: int, kind: str, stride: int):
        self.seed, self.kind = seed, kind
        self.sc = get_scenario(bench, seed, kind)
        self.frames = np.arange(0, len(self.sc["inspection"]), stride)
        self.insp = self.sc["inspection"].subset(self.frames)
        self.arrays = subset_arrays(self.sc["gt_arrays"], self.frames)
        self.changes = self.sc["gt"]["changes"]
        t = time.time()
        self.model = build(self.sc["baseline"], rooms=self.sc["rooms"])
        self.build_s = time.time() - t
        self.T_wb = np.linalg.inv(np.asarray(self.sc["gt"]["T_baseline_world"]))
        self._oscd_index = None
        self._retrieval = None
        self._models = {self.model.gaussians.voxel: self.model}

    def model_for(self, voxel: float):
        """Baseline model fused at another voxel size (map compression)."""
        if voxel not in self._models:
            self._models[voxel] = build(self.sc["baseline"], rooms=self.sc["rooms"], voxel=voxel)
        return self._models[voxel]

    @property
    def oscd_index(self):
        if self._oscd_index is None:
            c = OSCDConfig()
            self._oscd_index = ReferenceIndex(self.sc["baseline"], stride=c.ref_stride, n_features=c.n_features,
                                              min_sim=c.min_sim)
        return self._oscd_index

    @property
    def retrieval(self):
        if self._retrieval is None:
            self._retrieval = build_retrieval(self.sc["baseline"])
        return self._retrieval

    def view(self, keep: np.ndarray, session) -> dict:
        """Scenario dict restricted to the kept inspection frames (for ``evaluate_3d``)."""
        sc = dict(self.sc)
        sc["inspection"] = session
        sc["gt_arrays"] = subset_arrays(self.arrays, keep)
        return sc


# ----------------------------------------------------------------------------
# Methods
# ----------------------------------------------------------------------------

def run_methods(ctx: SceneContext, session, keep: np.ndarray, methods, encoder=None, cache_dir=None,
                model=None) -> list[dict]:
    model = model or ctx.model
    arrays = subset_arrays(ctx.arrays, keep)
    gt = gt_masks(arrays, ctx.changes)
    cm = change_masks(arrays, ctx.changes)
    recs = []

    def add(variant, fn):
        t = time.time()
        try:
            for r in fn():
                r.setdefault("seconds", round(time.time() - t, 2))
                recs.append(r)
        except Exception as e:  # keep the sweep going; the failure is recorded
            recs.append({"method": variant, "error": f"{type(e).__name__}: {e}",
                         "trace": traceback.format_exc()[-2000:]})

    def ours():
        t = time.time()
        res = inspect(model, session)
        secs = round(time.time() - t, 2)
        vox = model.gaussians.voxel
        pred = predicted_masks(res, session, vox, include_review=True)
        pred_c = predicted_masks(res, session, vox, include_review=False)
        scores = predicted_scores(res, session, vox, include_review=True)
        ev = evaluate_3d(ctx.view(keep, session), model, res)
        regions = gt_regions(ctx.sc["scene"], ctx.sc["inspection_scene"], ctx.changes)
        hits = match_detections(res, regions, ctx.T_wb)
        dets = [{"score": ch.score, "review": ch.review, "type": ch.type, "correct": bool(hits.get(ch.id)),
                 "geometry": ch.geometry_score, "appearance": ch.appearance_score} for ch in res.changes]
        flags = region_flags(res, model, ctx.changes, ctx.T_wb)
        common = {"objects": ev["objects"], "unverified": ev["unverified"], "registration": ev["registration"],
                  "self_registration": {k: res.registration.get(k) for k in ("inlier_ratio", "rmse_m", "confident")},
                  "quality": getattr(res, "quality", {}),
                  "verdict": res.verdict, "seconds": secs, "fps": round(len(session) / max(secs, 1e-9), 2)}
        r_all = {"method": "ours", **score_variant(pred, scores, gt, cm, ctx.changes, 0.0), **common,
                 "detections": dets,
                 "detection_calibration": cal.detection_reliability([d["score"] for d in dets],
                                                                    [d["correct"] for d in dets])}
        for row in r_all["changes"]:
            row["unverified_flag"] = flags.get(row["id"])
        scores_c = predicted_scores(res, session, vox, include_review=False)
        r_conf = {"method": "ours-confirmed", **score_variant(pred_c, scores_c, gt, cm, ctx.changes, 0.0),
                  "seconds": secs}
        return [r_all, r_conf]

    def video2d():
        r2 = compare_videos(ctx.sc["baseline"], session, retrieval=ctx.retrieval)
        return [{"method": "video2d", **score_variant(r2.masks, r2.scores, gt, cm, ctx.changes, 0.5),
                 "matched_frames": int((r2.matched >= 0).sum()), "coverage": float(r2.valid.mean()),
                 "fps": r2.timings.get("fps")}]

    def oscd():
        t = time.time()
        ro = run_oscd(model.gaussians, ctx.sc["baseline"], session, OSCDConfig(backbone="sam2"),
                      index=ctx.oscd_index, encoder=encoder, cache_dir=cache_dir)
        secs = round(time.time() - t, 2)
        reg = registration_errors(ro.poses, arrays["insp_poses_world"], ctx.T_wb)
        common = {"registration": reg, "localized": int(ro.localized.sum()), "usable": int(ro.usable.sum()),
                  "backbone": ro.backbone, "timings": ro.timings, "seconds": secs,
                  "fps": round(len(session) / max(secs, 1e-9), 3)}
        out = [{"method": "oscd-online", **score_variant(ro.masks, ro.scores, gt, cm, ctx.changes, 0.5), **common}]
        # no frame localised -> nothing to refine: the offline output is empty too, and scored as such
        m_off = ro.masks_refined if ro.masks_refined is not None else np.zeros_like(ro.masks)
        s_off = ro.scores_refined if ro.scores_refined is not None else np.zeros(ro.masks.shape, np.float16)
        out.append({"method": "oscd-offline", **score_variant(m_off, s_off, gt, cm, ctx.changes, 0.5), **common,
                    "no_output": ro.masks_refined is None})
        return out

    runners = {"ours": ours, "video2d": video2d, "oscd": oscd}
    for m in methods:
        add(m, runners[m])
    return recs


# ----------------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------------

VARIANTS = {"ours": ("ours", "ours-confirmed"), "video2d": ("video2d",), "oscd": ("oscd-online", "oscd-offline")}


def levels_of(name: str) -> list:
    return [("none", 0.0)] if name == "none" else [(name, float(v)) for v in STRESSORS[name].levels]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default="runs/bench", help="folder holding scenarios/ (generated if missing)")
    ap.add_argument("--out", default="runs/robust")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--kind", default="mixed")
    ap.add_argument("--stressors", nargs="+", default=["none"] + list(STRESSORS))
    ap.add_argument("--methods", nargs="+", default=["ours", "video2d", "oscd"])
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--stress-seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="write results<tag>.jsonl (one file per concurrent process)")
    ap.add_argument("--feature-cache", action="store_true",
                    help="cache image embeddings per scene for the clean, coverage and sparse levels, whose "
                         "frames repeat (deleted when the scene is done)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res_file = out / f"results{args.tag}.jsonl"
    done = set()
    if res_file.exists():
        for line in res_file.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                done.add((r["seed"], r["kind"], r["stressor"], r["level"], r["method"]))
    stressors = ["none"] + [s for s in args.stressors if s != "none"] if "none" in args.stressors else args.stressors
    encoder = make_encoder("sam2") if "oscd" in args.methods else None
    for seed in args.seeds:
        ctx = None
        cache = out / "featcache" / f"s{seed:03d}" if args.feature_cache else None
        for name in stressors:
            for sname, level in levels_of(name):
                baseline_only = sname in STRESSORS and STRESSORS[sname].target == "baseline"
                depth_only = sname in STRESSORS and STRESSORS[sname].depth_only
                todo = [m for m in args.methods
                        if any((seed, args.kind, sname, level, v) not in done for v in VARIANTS[m])
                        and not (baseline_only and m == "video2d")       # the 2D method has no map
                        and not (depth_only and m != "ours")]            # the others ignore the visit's depth
                if not todo:
                    continue
                if ctx is None:
                    ctx = SceneContext(Path(args.bench), seed, args.kind, args.stride)
                t = time.time()
                session, keep = apply_stressor(sname, level, ctx.insp, sc=ctx.sc, frames=ctx.frames,
                                               seed=args.stress_seed + seed)
                stress_s = round(time.time() - t, 2)
                cdir = cache if sname in ("none", "coverage", "sparse") else None
                model = ctx.model_for(level) if baseline_only else ctx.model
                for r in run_methods(ctx, session, keep, todo, encoder, cache_dir=cdir, model=model):
                    r.update(seed=seed, kind=args.kind, stressor=sname, level=level, stride=args.stride,
                             n_frames=int(len(keep)), stress_s=stress_s)
                    if len(keep) < len(ctx.insp):
                        r["kept"] = keep.tolist()
                    if baseline_only:
                        r["gaussians"] = len(model.gaussians)
                    with open(res_file, "a") as f:
                        f.write(json.dumps(r, default=float) + "\n")
                    if "error" in r:
                        print(f"s{seed} {sname}={level} {r['method']}: ERROR {r['error']}", flush=True)
                        continue
                    fr, c = r["frames"], r.get("calibration") or {}

                    def f(v, d=3):
                        return "-" if v is None else f"{v:.{d}f}"
                    print(f"s{seed} {sname}={level} {r['method']}: F1 {f(fr['F1'])} IoU {f(fr['IoU'])} "
                          f"FA {f(fr['false_alarm_rate'], 4)} ECE {f(c.get('ece'), 4)} ({r.get('seconds')}s)", flush=True)
        if cache is not None and cache.exists():
            shutil.rmtree(cache)


if __name__ == "__main__":
    main()
