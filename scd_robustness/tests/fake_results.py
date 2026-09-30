"""Synthetic *evaluated* runs with known effects, for testing the analysis without any method or GPU.

Masks, scores, frame logs and images are generated from a quality q(scene, system, stressor,
severity); the real evaluator (scd.evaluate.evaluate_run) then writes the eval files, so the
analysis reads exactly the formats a Sol run produces. Built-in effects:
  * accuracy falls with blur (online fastest, refined slowest), exposure and missing views;
  * online frames lose their pose (no prediction, few inliers) when q is low;
  * scores get noisier as q falls (calibration degrades);
  * no-change pairs: false-alarm area grows with severity.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from scd import SCENES
from scd.config import load_experiment, read_jsonl, seed_from
from scd.evaluate import evaluate_run
from scd.plan import write_manifests
from scd.runner import write_status

REPO = Path(__file__).resolve().parents[1]
H, W, N_FRAMES = 40, 60, 6
MV_SCENES = ["Cantina", "Printing_area", "Meeting_room", "Garden", "Porch"]

EXPERIMENTS = {
    "e1_clean": {"methods": {"oscd": {"seeds": [0, 1, 2]}, "mv3dcd": {"scenes": MV_SCENES, "seeds": [0, 1, 2]}},
                 "sweep": {"clean": [0]}},
    "e2_identity": {"methods": {"oscd": {}}, "sweep": {"identity": [0]}},
    "e4_main": {"methods": {"oscd": {}, "mv3dcd": {"scenes": MV_SCENES}},
                "sweep": {"identity": [0], "blur": [32, 128], "exposure": [-2, 2], "views": [0.5]},
                "trials": {"views": 2}},
    "e5_null": {"methods": {"oscd": {"pairs": ["null:Instance_1:Instance_2", "null:Instance_2:Instance_1"]}},
                "sweep": {"identity": [0], "blur": [32, 128], "exposure": [-2]}},
    "e7_matched": {"methods": {"oscd": {"pairs": ["Instance_1"],
                                        "arms": {"matched": {"refine": False, "steps_per_frame": "auto"}}}},
                   "sweep": {"identity": [0], "views": [0.5], "blur": [128]}, "trials": {"views": 2}},
}
BLUR_K = {"online": 1.3, "matched": 1.1, "refined": 0.6, "final": 0.9}


def quality(scene, role, stressor, severity):
    q = 0.72 + 0.02 * SCENES.index(scene)
    s = float(severity)
    if stressor == "blur":
        q *= math.exp(-BLUR_K[role] * s / 128.0)
    elif stressor == "exposure":
        q *= 1 - 0.12 * abs(s) * (1.3 if role == "online" else 1.0)
    elif stressor == "views":
        q *= 1 - (1 - s) * (0.6 if role in ("online", "matched") else 0.15)
    return float(np.clip(q, 0.05, 1.0))


def _frames(spec):
    names = ["IMG_%03d_test" % i for i in range(N_FRAMES)]
    if spec["stressor"] == "views":
        rng = np.random.default_rng(seed_from("%s|%s" % (spec["scene"], spec["trial"])))
        k = max(1, int(round(N_FRAMES * float(spec["severity"]))))
        keep = sorted(rng.choice(N_FRAMES, k, replace=False))
        return [names[i] for i in keep], [n for i, n in enumerate(names) if i not in keep]
    return names, []


def _gt(i):
    g = np.zeros((H, W), np.uint8)
    h, w = 6 + 2 * i, 8 + 2 * i
    g[4 + i:4 + i + h, 6 + 3 * i:6 + 3 * i + w] = 255
    return g


def _write_variant(spec):
    d = Path(spec["data_dir"])
    if (d / "variant.json").exists():
        return json.loads((d / "variant.json").read_text())
    kept, dropped = _frames(spec)
    img_dir = Path(spec["eval"]["image_dir"])
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed_from(spec["scene"]))
    tex = cv2.resize(rng.integers(0, 256, (H // 4, W // 4, 3), dtype=np.uint8), (W, H),
                     interpolation=cv2.INTER_NEAREST)
    for i in range(N_FRAMES):
        img = tex.astype(np.float32)
        if spec["stressor"] == "blur":
            img = cv2.GaussianBlur(img, (0, 0), float(spec["severity"]) / 32.0)
        elif spec["stressor"] == "exposure":
            img = img * 2.0 ** float(spec["severity"])
        name = "IMG_%03d_test" % i
        if name in kept:
            cv2.imwrite(str(img_dir / (name + ".jpg")), np.clip(img, 0, 255).astype(np.uint8))
    if spec["eval"]["gt_dir"]:
        gt_dir = Path(spec["eval"]["gt_dir"])
        gt_dir.mkdir(parents=True, exist_ok=True)
        for i in range(N_FRAMES):
            cv2.imwrite(str(gt_dir / ("IMG_%03d_test.png" % i)), _gt(i))
    meta = {"frames_kept": kept, "frames_dropped": dropped}
    (d / "variant.json").write_text(json.dumps(meta))
    return meta


def _role(spec, output):
    if spec["method"] == "mv3dcd":
        return "final"
    return "matched" if spec["arm"] == "matched" else output


def _write_outputs(spec, meta):
    out = Path(spec["out_dir"])
    log = []
    n_fail = 0
    for o in spec["eval"]["outputs"]:
        role = _role(spec, o["name"])
        q = quality(spec["scene"], role, spec["stressor"], spec["severity"])
        (out / o["pred"]).mkdir(parents=True, exist_ok=True)
        (out / o["score"]).mkdir(parents=True, exist_ok=True)
        for name in meta["frames_kept"]:
            i = int(name[4:7])
            rng = np.random.default_rng(seed_from("%s|%s|%s" % (spec["run_id"], o["name"], name)))
            missing = role in ("online", "matched") and rng.random() < np.clip(0.75 - q, 0, 0.6)
            if o["name"] in ("online",) and spec["method"] == "oscd":
                inl = 0 if missing else int(max(0, 320 * q + rng.normal(0, 25)))
                log.append({"frame": i, "image_name": name, "pose_ok": not missing, "n_ba_inliers": inl,
                            "n_pnp_inliers": inl + 20, "n_corr": inl + 150,
                            "ssf_loss_frame": float(1 - q + rng.normal(0, 0.05)),
                            "cue_mean": float(0.2 + 0.3 * (1 - q) + rng.normal(0, 0.03))})
                n_fail += missing
            if missing:
                continue
            gt = _gt(i) > 0 if not spec["null"] else np.zeros((H, W), bool)
            shift = int(round((1 - q) * 10))
            pred = np.roll(gt, shift, axis=1)
            fp = int(round((1 - q) * 14))
            if fp:
                y0, x0 = rng.integers(0, H - fp), rng.integers(0, W - fp)
                pred[y0:y0 + fp, x0:x0 + fp] = True
            noise = rng.normal(0, 0.4 + 2.5 * (1 - q), (H, W))
            if spec["method"] == "oscd":
                score = np.where(pred, 2.5, -2.5) + noise
            else:
                score = np.clip(np.where(pred, 0.85, 0.05) + 0.25 * noise, 0, 1)
            cv2.imwrite(str(out / o["pred"] / (name + ".png")), pred.astype(np.uint8) * 255)
            np.save(out / o["score"] / (name + ".npy"), score.astype(np.float16))
    if spec["method"] == "oscd":
        (out / "frame_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
        (out / "run_summary.json").write_text(json.dumps({"n_frames": len(meta["frames_kept"]),
                                                          "n_pose_fail": int(n_fail), "online_fps": 10.5}))


def make_fake_study(root, skip=("e4_main/oscd/default/identity-0_t0/Instance_1/Garden/s0",),
                    fail=("e4_main/oscd/default/identity-0_t0/Instance_1/Zen/s0",)):
    """Write configs + manifests + evaluated runs under root. Returns the analysis config path."""
    root = Path(root)
    cfg = root / "configs"
    (cfg / "experiments").mkdir(parents=True, exist_ok=True)
    paths = yaml.safe_load((REPO / "configs" / "paths.yaml").read_text())
    paths["root"] = str(root / "scd")
    (cfg / "paths.yaml").write_text(yaml.safe_dump(paths))
    (cfg / "perturbations.yaml").write_text((REPO / "configs" / "perturbations.yaml").read_text())
    for name, body in EXPERIMENTS.items():
        methods = {m: dict({"scenes": "all", "pairs": ["Instance_1", "Instance_2"], "seeds": [0]}, **b)
                   for m, b in body["methods"].items()}
        doc = {"name": name, "paths": "../paths.yaml", "perturbations": "../perturbations.yaml",
               "methods": methods, "sweep": body["sweep"], "trials": body.get("trials", {})}
        (cfg / "experiments" / (name + ".yaml")).write_text(yaml.safe_dump(doc))
        exp = load_experiment(cfg / "experiments" / (name + ".yaml"))
        info = write_manifests(exp)
        for key, f in info["files"].items():
            if not key.startswith("runs_") or key.endswith("refcache"):
                continue
            for k, spec in enumerate(read_jsonl(f)):
                if spec["run_id"] in skip:
                    continue
                out = Path(spec["out_dir"])
                out.mkdir(parents=True, exist_ok=True)
                if spec["run_id"] in fail:
                    write_status(out, status="failed", error="synthetic failure")
                    continue
                meta = _write_variant(spec)
                _write_outputs(spec, meta)
                evaluate_run(spec)
                write_status(out, status="ok", seconds=100.0 + (k % 7) * 10, gpu="NVIDIA A100-SXM4-80GB",
                             host="sg000")
    acfg = yaml.safe_load((REPO / "configs" / "analysis.yaml").read_text())
    acfg["stats"]["n_boot"] = 400
    (cfg / "analysis.yaml").write_text(yaml.safe_dump(acfg, sort_keys=False))
    return cfg / "analysis.yaml"
