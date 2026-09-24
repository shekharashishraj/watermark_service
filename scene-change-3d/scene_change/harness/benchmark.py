"""Benchmark change detection methods on generated harness scenarios.

    python -m scene_change.harness.benchmark --out runs/bench --seeds 0 1 2 --kinds mixed clean partial \
        --methods ours video2d oscd

Methods
    ours      3D geometric + appearance change detection (scene_change.pipeline)
    video2d   2D video comparison: retrieval + homography + differencing (scene_change.baseline2d)
    oscd      O-SCD re-implementation (scene_change.oscd); ``--oscd-poses`` picks its own PnP
              poses (default, as in the paper), ours, or ground truth

Every method is scored on the inspection frames with the PASLCD protocol used by
O-SCD (per-frame change IoU/F1, averaged over frames showing a change) and with pooled
pixel metrics; ``ours`` also gets the object-level metrics of ``evaluate_3d``.
Results go to ``<out>/results.jsonl`` and a summary table to ``<out>/summary.md``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..baseline2d import Video2DConfig, compare_videos
from ..oscd import OSCDConfig, run_oscd
from ..pipeline import build, inspect
from .evaluate import evaluate_3d, frame_metrics, gt_masks, pixel_metrics, predicted_masks, registration_errors
from .scenario import build_scenario, load_scenario, save_scenario


def scenario_path(out: Path, seed: int, kind: str) -> Path:
    return out / "scenarios" / f"s{seed:03d}_{kind}"


def get_scenario(out: Path, seed: int, kind: str) -> dict:
    p = scenario_path(out, seed, kind)
    if not (p / "gt.json").exists():
        save_scenario(build_scenario(seed, kind), p)
    return load_scenario(p)


def _gt_pose(sc):
    T_bw = np.asarray(sc["gt"]["T_baseline_world"])
    return np.einsum("ij,njk->nik", T_bw, sc["gt_arrays"]["insp_poses_world"]), np.linalg.inv(T_bw)


def run_one(sc: dict, methods, oscd_poses="pnp", oscd_backbone="auto", cache_dir=None) -> list[dict]:
    base, insp = sc["baseline"], sc["inspection"]
    gt = gt_masks(sc["gt_arrays"], sc["gt"]["changes"])
    gt_pose, T_wb = _gt_pose(sc)
    t = time.time()
    model = build(base, rooms=sc["rooms"])
    build_s = time.time() - t
    recs = []
    ours = None
    if "ours" in methods or oscd_poses == "ours":
        t = time.time()
        ours = inspect(model, insp)
        ours_s = time.time() - t
    if "ours" in methods:
        pred = predicted_masks(ours, insp, model.gaussians.voxel, include_review=True)
        pred_c = predicted_masks(ours, insp, model.gaussians.voxel, include_review=False)
        ev = evaluate_3d(sc, model, ours)
        recs.append({"method": "ours", "frames": frame_metrics(pred, gt), "frames_confirmed": frame_metrics(pred_c, gt),
                     "pixels": pixel_metrics(pred, gt), "objects": ev["objects"], "registration": ev["registration"],
                     "seconds": round(ours_s, 2), "fps": round(len(insp) / max(ours_s, 1e-9), 2),
                     "baseline_build_s": round(build_s, 2), "verdict": ours.verdict})
    if "video2d" in methods:
        t = time.time()
        r2 = compare_videos(base, insp, Video2DConfig())
        s2 = time.time() - t
        recs.append({"method": "video2d", "frames": frame_metrics(r2.masks, gt), "pixels": pixel_metrics(r2.masks, gt),
                     "coverage": float(r2.valid.mean()), "matched_frames": int((r2.matched >= 0).sum()),
                     "seconds": round(s2, 2), "fps": round(len(insp) / max(s2, 1e-9), 2)})
    if "oscd" in methods:
        poses = {"gt": gt_pose, "ours": None if ours is None else ours.poses}.get(oscd_poses)
        t = time.time()
        ro = run_oscd(model.gaussians, base, insp, OSCDConfig(backbone=oscd_backbone), poses=poses,
                      cache_dir=cache_dir)
        so = time.time() - t
        recs.append({"method": "oscd", "poses": oscd_poses, "backbone": ro.backbone,
                     "frames": frame_metrics(ro.masks, gt), "pixels": pixel_metrics(ro.masks, gt),
                     "frames_refined": frame_metrics(ro.masks_refined, gt) if ro.masks_refined is not None else None,
                     "registration": registration_errors(ro.poses, sc["gt_arrays"]["insp_poses_world"], T_wb),
                     "localized": int(ro.localized.sum()), "seconds": round(so, 2),
                     "fps": round(len(insp) / max(so, 1e-9), 2), "timings": ro.timings})
    return recs


def summarize(records: list[dict]) -> str:
    rows = defaultdict(list)
    for r in records:
        key = r["method"] + (f" ({r['poses']} poses)" if r["method"] == "oscd" else "")
        rows[key].append(r)
    lines = ["| method | scenarios | IoU | F1 | false-alarm px (no-change frames) | pooled F1 | fps |",
             "|---|---|---|---|---|---|---|"]

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    for key, rs in rows.items():
        iou = mean([r["frames"]["IoU"] for r in rs if r["kind"] != "clean"])
        f1 = mean([r["frames"]["F1"] for r in rs if r["kind"] != "clean"])
        far = mean([r["frames"]["false_alarm_rate"] for r in rs])
        pf1 = mean([r["pixels"]["F1"] for r in rs if r["kind"] != "clean"])
        fps = mean([r["fps"] for r in rs])
        lines.append(f"| {key} | {len(rs)} | {iou:.3f} | {f1:.3f} | {100 * far:.2f}% | {pf1:.3f} | {fps:.2f} |")
    obj = [r for r in records if r["method"] == "ours"]
    if obj:
        rec = mean([r["objects"]["recall"] for r in obj if r["kind"] != "clean"])
        fp = mean([r["objects"]["false_positives"] for r in obj])
        fpc = mean([r["objects"]["false_positives_confirmed"] for r in obj])
        lines += ["", f"ours, object level: recall {rec:.3f}, false positives per inspection {fp:.2f} "
                      f"({fpc:.2f} confirmed, the rest flagged for review)"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/bench")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--kinds", nargs="+", default=["mixed", "clean", "partial"])
    ap.add_argument("--methods", nargs="+", default=["ours", "video2d", "oscd"])
    ap.add_argument("--oscd-poses", default="pnp", choices=["pnp", "ours", "gt"])
    ap.add_argument("--oscd-backbone", default="auto")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res_file = out / "results.jsonl"
    done = set()
    if res_file.exists():
        for line in res_file.read_text().splitlines():
            r = json.loads(line)
            done.add((r["seed"], r["kind"], r["method"]))
    for seed in args.seeds:
        for kind in args.kinds:
            todo = [m for m in args.methods if (seed, kind, m) not in done]
            if not todo:
                continue
            sc = get_scenario(out, seed, kind)
            for r in run_one(sc, todo, args.oscd_poses, args.oscd_backbone, cache_dir=out / "featcache"):
                r.update(seed=seed, kind=kind)
                with open(res_file, "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(f"s{seed} {kind} {r['method']}: IoU {r['frames']['IoU']} F1 {r['frames']['F1']} "
                      f"FA {r['frames']['false_alarm_rate']} ({r['seconds']}s)", flush=True)
    records = [json.loads(line) for line in res_file.read_text().splitlines()]
    (out / "summary.md").write_text(summarize(records))
    print(summarize(records))


if __name__ == "__main__":
    main()
