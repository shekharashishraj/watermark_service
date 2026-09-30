"""Evaluator used for every method (plan patch P7). NumPy + OpenCV only; runs in any environment.

Differences from the released evaluators (O-SCD utils/evaluate.py, MV3DCD evaluate.py):
  * a frame without a prediction (pose failure, registration failure) is scored as an EMPTY mask
    instead of being silently skipped; the released behaviour is reported too ("paper_*" fields);
  * per-frame TP/FP/FN/TN, pooled metrics, a boundary-tolerant variant and object-level recall;
  * continuous scores -> per-frame 256-bin histograms for calibration analysis;
  * no-change ("null") pairs: every pixel is negative, false-alarm rates are reported.
Outputs (prefix = <run>/eval/<output name>): _frames.csv, _components.csv, _hists.npz, _summary.json
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from . import IMAGE_EXTS
from .imageio import find_by_stem
from .metrics import counts, rates, score_hist, summarize_hists

_READ_GRAY = cv2.IMREAD_GRAYSCALE | getattr(cv2, "IMREAD_IGNORE_ORIENTATION", 0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# how a stored score becomes p in [0, 1], and the threshold each method actually uses on p
SCORE_KINDS = {
    "oscd_raw": (lambda s: sigmoid(s.astype(np.float32)), float(sigmoid(0.5))),   # decision: raw > 0.5
    "mv3dcd": (lambda s: np.clip(s.astype(np.float32), 0.0, 1.0), 0.5),
    "prob": (lambda s: np.clip(s.astype(np.float32), 0.0, 1.0), 0.5),
}


def _read_bool(path):
    m = cv2.imread(str(path), _READ_GRAY)
    if m is None:
        raise IOError("could not read %s" % path)
    return m > 127


def _resize(arr, wh, nearest=True):
    if arr.shape[1] == wh[0] and arr.shape[0] == wh[1]:
        return arr
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    if arr.dtype == bool:
        return cv2.resize(arr.astype(np.uint8), wh, interpolation=interp).astype(bool)
    return cv2.resize(arr, wh, interpolation=interp)


def input_stats(img_path, work_scale):
    """Blur / exposure indicators of the input frame at the method's working resolution."""
    g = cv2.imread(str(img_path), _READ_GRAY)
    if g is None:
        return {}
    h, w = g.shape
    s = max(1, int(work_scale))
    small = cv2.resize(g, (max(1, w // s), max(1, h // s)), interpolation=cv2.INTER_AREA)
    return {
        "img_h": h, "img_w": w,
        "lap_var": float(cv2.Laplacian(small, cv2.CV_64F).var()),
        "brightness": float(small.mean() / 255.0),
        "clipped_frac": float(((small <= 2) | (small >= 253)).mean()),
    }


def _band(gt, radius):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    g = gt.astype(np.uint8)
    return (cv2.dilate(g, k) > 0) & ~(cv2.erode(g, k) > 0)


def _nanmean(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _load_frame_log(path):
    log = {}
    if path and Path(path).exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    log[rec.get("image_name")] = rec
    return log


def evaluate_output(frames, image_dir, pred_dir, score_dir, score_kind, gt_dir, work_scale, out_prefix,
                    frame_log=None, boundary_px_work=3, min_component_px_work=50, far_frame_thresh=0.005):
    """Evaluate one output (e.g. O-SCD online masks) of one run over the given frame stems."""
    to_p, decision = SCORE_KINDS[score_kind]
    null = gt_dir is None
    pred_dir, score_dir = Path(pred_dir), Path(score_dir) if score_dir else None
    log = _load_frame_log(frame_log)
    rows, comps, pos_all, neg_all = [], [], [], []

    for stem in frames:
        row = {"frame": stem}
        img_path = find_by_stem(image_dir, stem, IMAGE_EXTS)
        row.update(input_stats(img_path, work_scale) if img_path else {})
        pred_path = find_by_stem(pred_dir, stem, (".png", ".jpg")) if pred_dir.exists() else None
        pred = _read_bool(pred_path) if pred_path else None
        row["pred_missing"] = pred is None

        if not null:
            gt_path = find_by_stem(gt_dir, stem, IMAGE_EXTS)
            gt = _read_bool(gt_path)
            H, W = gt.shape
        elif pred is not None:
            H, W = pred.shape
        else:
            H, W = int(row.get("img_h", 1)) // max(1, int(work_scale)), int(row.get("img_w", 1)) // max(1, int(work_scale))
        work_w = pred.shape[1] if pred is not None else row.get("img_w", W * work_scale) / float(work_scale)
        scale = W / float(work_w)                       # eval pixels per working-resolution pixel
        pred_eval = _resize(pred, (W, H)) if pred is not None else np.zeros((H, W), bool)

        if null:
            row["far"] = float(pred_eval.mean())
            row["false_alarm_frame"] = bool(row["far"] > far_frame_thresh)
            row["pred_area_frac"] = row["far"]
        else:
            c = counts(gt, pred_eval)
            row.update(c)
            row.update(rates(**c))
            row["gt_area_frac"] = float(gt.mean())
            row["pred_area_frac"] = float(pred_eval.mean())
            # boundary-tolerant: ignore a band of +-boundary_px_work working pixels around GT edges
            radius = max(1, int(round(boundary_px_work * scale)))
            keep = ~_band(gt, radius)
            cb = counts(gt[keep], pred_eval[keep])
            row["iou_btol"] = rates(**cb)["iou"]
            # object-level recall
            n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), connectivity=8)
            min_px = min_component_px_work * scale * scale
            for i in range(1, n_lab):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < min_px:
                    continue
                comp = lab == i
                comps.append({"frame": stem, "component": i, "area_frac": area / float(H * W),
                              "detected_frac": float(pred_eval[comp].mean())})

        score_path = score_dir / (stem + ".npy") if score_dir else None
        if score_path is not None and score_path.exists():
            p = _resize(to_p(np.load(score_path)), (W, H))
            ph, nh = score_hist(p, None if null else gt)
            row["score_mean"] = float(p.mean())
            ent = -(p * np.log(p + 1e-7) + (1 - p) * np.log(1 - p + 1e-7))
            row["score_entropy"] = float(ent.mean())
            row["score_missing"] = False
        else:
            ph = nh = None
            row["score_missing"] = True
        pos_all.append(ph)
        neg_all.append(nh)

        for key, value in log.get(stem, {}).items():
            if key not in ("image_name", "frame"):
                row["log_" + key] = value
        rows.append(row)

    # ---- write per-frame outputs
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(Path(str(out_prefix) + "_frames.csv"), rows)
    _write_csv(Path(str(out_prefix) + "_components.csv"), comps,
               ["frame", "component", "area_frac", "detected_frac"])
    have = [i for i, h in enumerate(pos_all) if h is not None]
    pos = np.stack([pos_all[i] for i in have]) if have else np.zeros((0, 256), np.int64)
    neg = np.stack([neg_all[i] for i in have]) if have else np.zeros((0, 256), np.int64)
    np.savez_compressed(str(out_prefix) + "_hists.npz", frames=np.array([rows[i]["frame"] for i in have]),
                        pos=pos, neg=neg)

    # ---- summary
    summary = {
        "n_frames": len(rows),
        "n_pred_missing": sum(r["pred_missing"] for r in rows),
        "n_score_missing": sum(r["score_missing"] for r in rows),
        "null": null,
    }
    if null:
        summary["far_mean"] = _nanmean([r["far"] for r in rows])
        summary["ffr"] = _nanmean([float(r["false_alarm_frame"]) for r in rows])
    else:
        with_pred = [r for r in rows if not r["pred_missing"]]
        summary["paper_miou"] = _nanmean([r["iou"] for r in with_pred])     # released: missing skipped
        summary["paper_f1"] = _nanmean([r["f1"] for r in with_pred])
        summary["miou"] = _nanmean([r["iou"] for r in rows])                # missing = empty mask
        summary["mf1"] = _nanmean([r["f1"] for r in rows])
        summary["miou_btol"] = _nanmean([r["iou_btol"] for r in rows])
        tot = {k: sum(r[k] for r in rows) for k in ("tp", "fp", "fn", "tn")}
        summary.update({"pooled_" + k: v for k, v in rates(**tot).items()})
        summary["fp_frac_mean"] = _nanmean([r["fp"] / float(r["tp"] + r["fp"] + r["fn"] + r["tn"]) for r in rows])
        summary["fn_frac_mean"] = _nanmean([r["fn"] / float(r["tp"] + r["fp"] + r["fn"] + r["tn"]) for r in rows])
    if len(pos):
        summary.update(summarize_hists(pos.sum(0), neg.sum(0), decision))
    with open(str(out_prefix) + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def evaluate_run(spec: dict) -> dict:
    """Evaluate every output of a finished run described by a manifest line."""
    ev = spec.get("eval")
    if not ev:
        return {}
    data_dir, out_dir = Path(spec["data_dir"]), Path(spec["out_dir"])
    meta = json.loads((data_dir / "variant.json").read_text())
    frames = meta["frames_kept"]
    n_no_gt = 0
    if ev.get("gt_dir"):
        gt_stems = {Path(p).stem for p in Path(ev["gt_dir"]).iterdir()}
        n_no_gt = sum(1 for f in frames if f not in gt_stems)
        frames = [f for f in frames if f in gt_stems]
    results = {}
    for out in ev["outputs"]:
        summary = evaluate_output(
            frames=frames, image_dir=ev["image_dir"], pred_dir=out_dir / out["pred"],
            score_dir=out_dir / out["score"] if out.get("score") else None, score_kind=out["score_kind"],
            gt_dir=ev.get("gt_dir"), work_scale=ev["work_scale"], out_prefix=out_dir / "eval" / out["name"],
            frame_log=(out_dir / ev["frame_log"]) if ev.get("frame_log") else None,
        )
        summary.update({"output": out["name"], "n_frames_without_gt": n_no_gt,
                        "n_frames_dropped": len(meta["frames_dropped"])})
        run_summary = out_dir / "run_summary.json"
        if run_summary.exists():
            summary["method_summary"] = json.loads(run_summary.read_text())
        with open(out_dir / "eval" / (out["name"] + "_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        results[out["name"]] = summary
    return results
