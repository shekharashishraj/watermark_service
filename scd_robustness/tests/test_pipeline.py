"""End-to-end CPU test: toy data -> plan -> variants -> fake method outputs -> evaluation -> collect."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from scd import colmap_rw
from scd.config import load_experiment, read_jsonl
from scd.evaluate import evaluate_run
from scd.plan import write_manifests
from scd.runner import write_status
from scd.variants import build_variant
from toy_data import make_toy

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def exp(tmp_path, monkeypatch):
    monkeypatch.setenv("SCD_ROOT", str(tmp_path / "scd"))
    online, offline = make_toy(tmp_path / "raw")
    cfg = tmp_path / "configs"
    (cfg / "experiments").mkdir(parents=True)
    paths = yaml.safe_load((REPO / "configs" / "paths.yaml").read_text())
    paths.update(data_online=str(online), data_offline=str(offline),
                 code_oscd=str(tmp_path / "code/O-SCD"), code_mv3dcd=str(tmp_path / "code/MV3DCD"))
    (cfg / "paths.yaml").write_text(yaml.safe_dump(paths))
    shutil.copy(REPO / "configs" / "perturbations.yaml", cfg / "perturbations.yaml")
    exp_cfg = {
        "name": "toy",
        "methods": {
            "oscd": {"scenes": ["Garden"], "pairs": ["Instance_1", "null:Instance_1:Instance_2"], "seeds": [0]},
            "mv3dcd": {"scenes": ["Garden"], "pairs": ["Instance_1"], "seeds": [0]},
            "oscd_matched": {"method": "oscd", "scenes": ["Porch"], "pairs": ["Instance_2"],
                             "arms": {"matched": {"refine": False, "steps_per_frame": "auto"}}},
        },
        "sweep": {"identity": [0], "blur": [5], "exposure": [-1], "views": [0.5], "views_gap": [0.5]},
        "trials": {"views": 2},
    }
    path = cfg / "experiments" / "toy.yaml"
    path.write_text(yaml.safe_dump(exp_cfg))
    return load_experiment(path)


def test_end_to_end(exp, tmp_path):
    info = write_manifests(exp)
    c = info["counts"]
    # oscd: 2 pairs x (identity, blur, exposure, views x2 trials, views_gap) = 12; matched: 1 x 6 = 6
    assert c["runs_oscd"] == 18 and c["runs_mv3dcd"] == 6 and c["runs_mv3dcd_refcache"] == 1
    variants = read_jsonl(info["files"]["variants"])
    assert any(v["stressor"] == "clean" for v in variants)         # added for the MV3DCD reference cache

    metas = {}
    for spec in variants:
        meta = build_variant(spec, exp["_paths"], exp["_perturbations"], workers=1)
        assert meta["_status"] == "built"
        metas[spec["out_dir"]] = meta
    # idempotent
    assert build_variant(variants[0], exp["_paths"], exp["_perturbations"])["_status"] == "exists"

    by = {(v["method"], v["pair"], v["scene"], v["variant_id"]): Path(v["out_dir"]) for v in variants}
    # --- O-SCD blur variant: perturbed images are new files, references are symlinks, GT linked
    d = by[("oscd", "Instance_1", "Garden", "blur-5_t0")]
    img = d / "inference_scene" / "images" / "IMG_000_test.jpg"
    assert img.exists() and not img.is_symlink()
    assert (d / "reference_scene").is_symlink() and (d / "gt_mask").is_symlink()
    assert (d / "reference_reconstruction").is_symlink()
    # --- null pair: revisit = Instance_2 reference images, no GT
    dn = by[("oscd", "null:Instance_1:Instance_2", "Garden", "identity-0_t0")]
    assert sorted(p.name for p in (dn / "inference_scene" / "images").iterdir())[0].startswith("REF2_")
    assert not (dn / "gt_mask").exists()
    assert (dn / "reference_scene").resolve() == (Path(exp["_paths"]["data_online"]) / "Instance_1/Garden/reference_scene").resolve()
    # --- MV3DCD views variant: dropped frames leave both the folder and the COLMAP model
    dv = by[("mv3dcd", "Instance_1", "Garden", "views-0p5_t0")]
    meta = metas[str(dv)]
    assert len(meta["frames_kept"]) == 3 and len(meta["frames_dropped"]) == 3
    _, ims, pts = colmap_rw.read_model(str(dv / "sparse" / "0"))
    names = {Path(i.name).stem for i in ims.values()}
    assert not names & set(meta["frames_dropped"]) and set(meta["frames_kept"]) <= names
    assert all(not (dv / "images" / (f + ".jpg")).exists() for f in meta["frames_dropped"])
    assert (dv / "sparse" / "0" / "points3D.ply").exists()
    kept_ids = set(ims)
    assert all(set(p.image_ids.tolist()) <= kept_ids for p in pts.values())
    # --- sector coverage on the online layout uses offline COLMAP poses
    dg = by[("oscd", "Instance_1", "Garden", "views_gap-0p5_t0")]
    assert len(metas[str(dg)]["frames_dropped"]) == 3
    # --- identity variant re-encodes but leaves pixels (almost) unchanged
    di = by[("oscd", "Instance_1", "Garden", "identity-0_t0")]
    a = cv2.imread(str(di / "inference_scene/images/IMG_001_test.jpg")).astype(int)
    b = cv2.imread(str(Path(exp["_paths"]["data_online"]) / "Instance_1/Garden/inference_scene/images/IMG_001_test.jpg")).astype(int)
    assert np.abs(a - b).mean() < 2.0

    # --- runner commands
    import run_mv3dcd
    import run_oscd
    runs_o = read_jsonl(info["files"]["runs_oscd"])
    spec = next(s for s in runs_o if s["variant_id"] == "views-0p5_t0" and s["pair"] == "Instance_1")
    argv = run_oscd.steps(spec)[0][1]
    assert argv[1] == "oscd.py" and "--refine" in argv and "--save_scores" in argv and "--frame_log" in argv
    matched = next(s for s in runs_o if s["arm"] == "matched" and s["variant_id"] == "identity-0_t0")
    argv_m = run_oscd.steps(matched)[0][1]
    assert argv_m[argv_m.index("--steps_per_frame") + 1] == "500" and "--refine" not in argv_m   # 3000 / 6 frames
    runs_m = read_jsonl(info["files"]["runs_mv3dcd"])
    names_m = [s[0] for s in run_mv3dcd.steps(runs_m[0])]
    assert names_m[0] == "render_reference" and names_m[-1] == "render_final"     # reference comes from cache
    ref = read_jsonl(info["files"]["runs_mv3dcd_refcache"])[0]
    assert [s[0] for s in run_mv3dcd.steps(ref)] == ["train_reference"]

    # --- fake O-SCD outputs: perfect masks for half of the kept frames, the rest "failed pose"
    out = Path(spec["out_dir"])
    kept = metas[spec["data_dir"]]["frames_kept"]
    for sub in ("renders/change_mask", "renders/change_score", "renders/change_mask_refined", "renders/change_score_refined"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    gt_dir = Path(spec["eval"]["gt_dir"])
    log = []
    for i, f in enumerate(kept):
        small = cv2.imread(str(gt_dir / (f + ".png")), cv2.IMREAD_GRAYSCALE)   # "perfect" prediction
        raw = np.where(small > 127, 3.0, -3.0).astype(np.float16)
        for m, s in (("change_mask", "change_score"), ("change_mask_refined", "change_score_refined")):
            if m == "change_mask" and i % 2:
                continue                                   # online: pose failure on odd frames
            cv2.imwrite(str(out / "renders" / m / (f + ".png")), (small > 127).astype(np.uint8) * 255)
            np.save(out / "renders" / s / (f + ".npy"), raw)
        log.append({"frame": i, "image_name": f, "pose_ok": i % 2 == 0, "n_ba_inliers": 150 if i % 2 == 0 else 20})
    (out / "frame_log.jsonl").write_text("\n".join(json.dumps(r) for r in log) + "\n")
    (out / "run_summary.json").write_text(json.dumps({"n_frames": len(kept), "n_pose_fail": 1}))
    res = evaluate_run(spec)
    on, rf = res["online"], res["refined"]
    assert on["n_frames"] == 3 and on["n_pred_missing"] == 1
    assert on["paper_miou"] == pytest.approx(1.0)          # released evaluator would skip the failure...
    assert on["miou"] == pytest.approx(2 / 3)              # ...ours counts it as an empty mask
    assert rf["miou"] == pytest.approx(1.0, abs=1e-6) and rf["auroc"] == pytest.approx(1.0)
    frames_csv = (out / "eval" / "online_frames.csv").read_text()
    assert "log_n_ba_inliers" in frames_csv and "lap_var" in frames_csv
    write_status(out, status="ok", seconds=1.0)

    # --- null evaluation: empty predictions -> no false alarms
    null_spec = next(s for s in runs_o if s["null"] and s["variant_id"] == "identity-0_t0")
    nout = Path(null_spec["out_dir"])
    (nout / "renders/change_mask").mkdir(parents=True)
    for f in metas[null_spec["data_dir"]]["frames_kept"]:
        cv2.imwrite(str(nout / "renders/change_mask" / (f + ".png")), np.zeros((12, 20), np.uint8))
    nres = evaluate_run(null_spec)
    assert nres["online"]["null"] and nres["online"]["far_mean"] == 0.0 and nres["online"]["ffr"] == 0.0

    # --- collect
    cfg_file = exp["_file"]
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "collect_results.py"), "--experiment", cfg_file],
                          capture_output=True, text=True, env=dict(__import__("os").environ))
    assert proc.returncode == 0, proc.stderr
    summary = Path(exp["_paths"]["results"]) / "toy" / "summary.csv"
    text = summary.read_text()
    assert "online" in text and "refined" in text and "not_run" in text
