import numpy as np

from scene_change import calibration as cal
from scene_change import stress
from scene_change.harness.raycast import Lighting
from scene_change.session import Session


def _session(n=6, H=24, W=32, seed=0):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, (n, H, W, 3), dtype=np.uint8)
    depth = np.full((n, H, W), 2.0, np.float32)
    K = np.array([[30.0, 0, W / 2], [0, 30.0, H / 2], [0, 0, 1]])
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = 0.05 * np.arange(n)          # camera moving along +x
    return Session("t", K, rgb, depth, poses)


def test_exposure_is_monotone_and_identity_at_zero():
    img = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (4, 1, 3))
    assert np.array_equal(stress.exposure(img, 0.0), img)
    dark, bright = stress.exposure(img, -1.0), stress.exposure(img, 1.0)
    assert np.all(dark <= img) and np.all(bright >= img)
    assert np.all(np.diff(dark[0, :, 0].astype(int)) >= 0)
    assert bright[0, -1, 0] == 255 and np.mean(bright == 255) > np.mean(img == 255)


def test_motion_blur_kernel_and_direction():
    k = stress.line_kernel(7.0, 0.0)
    assert abs(k.sum() - 1) < 1e-6
    rows = np.nonzero(k.sum(1) > 1e-3)[0]
    assert len(rows) <= 2                          # horizontal kernel spans ~1 row
    s = _session()
    ang = stress.motion_directions(s)
    # camera moves +x, so the scene moves -x in the image: direction ~ pi (or 0 up to sign)
    assert np.all(np.abs(np.sin(ang)) < 0.2)
    b = stress.blur_session(s, 0.2)
    assert b.rgb.shape == s.rgb.shape and not np.array_equal(b.rgb, s.rgb)
    assert np.array_equal(b.depth, s.depth)


def test_drop_views_counts_and_nesting():
    for mode in ("segments", "uniform"):
        prev = None
        for f in (0.0, 0.2, 0.4, 0.6, 0.8):
            keep = stress.drop_views(100, f, mode=mode, seed=3)
            assert len(keep) == 100 - round(100 * f)
            if prev is not None:
                assert set(keep) <= set(prev)
            prev = keep
    keep = stress.drop_views(100, 0.4, mode="segments", segments=2, seed=1)
    gaps = np.sum(np.diff(keep) > 1) + (keep[0] > 0) + (keep[-1] < 99)
    assert gaps <= 2                               # removed frames form at most 2 runs


def test_interpolate_lighting_endpoints():
    base = Lighting(exposure=1.0, white_balance=(1.0, 1.0, 1.0), room_gain={"a": 1.0}, lamps=[([0, 0, 1], 0.35, "a")])
    insp = Lighting(exposure=0.5, white_balance=(1.1, 1.0, 0.9), room_gain={"a": 2.0}, lamps=[([1, 1, 1], 0.35, "a")])
    l0 = stress.interpolate_lighting(base, insp, 0.0)
    l1 = stress.interpolate_lighting(base, insp, 1.0)
    l2 = stress.interpolate_lighting(base, insp, 2.0)
    assert abs(l0.exposure - 1.0) < 1e-9 and abs(l1.exposure - 0.5) < 1e-9 and abs(l2.exposure - 0.25) < 1e-9
    assert abs(l1.room_gain["a"] - 2.0) < 1e-9
    assert [lp[0] for lp in l0.lamps] == [[0, 0, 1]] and [lp[0] for lp in l1.lamps] == [[1, 1, 1]]
    assert abs(l2.lamps[0][1] - 0.7) < 1e-9


def test_calibration_metrics_on_synthetic_scores():
    rng = np.random.default_rng(0)
    s = rng.random(400_000).astype(np.float32)
    y_cal = rng.random(s.size) < s                 # calibrated
    y_mis = rng.random(s.size) < s ** 3            # over-confident scores
    h_cal, h_mis = cal.score_histogram(s, y_cal), cal.score_histogram(s, y_mis)
    assert cal.ece(h_cal) < 0.01 and cal.ece(h_cal, adaptive=True) < 0.01
    assert cal.ece(h_mis) > 0.15
    iso = cal.isotonic_map(h_mis)
    assert np.all(np.diff(iso) >= -1e-12)
    assert cal.ece(cal.recalibrate(h_mis, iso)) < 0.01
    # perfect ranking
    y = s > 0.7
    h = cal.score_histogram(s, y)
    assert cal.auroc(h) > 0.999 and cal.average_precision(h) > 0.99
    # random ranking
    h_r = cal.score_histogram(s, rng.random(s.size) < 0.1)
    assert abs(cal.auroc(h_r) - 0.5) < 0.01


def test_threshold_sweep_and_confident_errors():
    gt = np.zeros((3, 10, 10), bool)
    gt[0, :5] = True
    scores = np.zeros((3, 10, 10), np.float32)
    scores[0, :5] = 0.8                            # true change, fairly confident
    scores[1, :2] = 0.95                           # confident false alarm on a change-free frame
    scores[0, 5:6] = 0.3                           # sub-threshold
    sweep = cal.threshold_sweep(scores, gt, thresholds=(0.2, 0.5, 0.9))
    at = {r["t"]: r for r in sweep}
    assert at[0.5]["F1"] == 1.0 and at[0.5]["false_alarm_rate"] == 20 / 200
    assert at[0.2]["F1"] < 1.0 and at[0.9]["F1"] == 0.0
    ce = cal.confident_errors(scores, gt)
    assert ce["fp"] == 20 and ce["fp_confident"] == 1.0 and ce["fn"] == 0
    rep = cal.calibration_report(scores, gt)
    assert rep["best_t"] is not None and len(rep["hist"]["n"]) == cal.NB
    assert sum(rep["hist"]["n"]) == gt.size


def test_detection_reliability():
    r = cal.detection_reliability([0.9, 0.9, 0.5, 0.3], [True, False, True, False])
    assert r["n"] == 4 and r["precision"] == 0.5
    assert sum(b["n"] for b in r["bins"]) == 4


def _fake_oscd_scene(root, n=12, gt_every=3):
    import cv2
    from pathlib import Path
    root = Path(root)
    for d in ("reference_scene/images", "reference_reconstruction", "inference_scene/images", "gt_mask"):
        (root / d).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    base = (rng.random((48, 64, 3)) * 255).astype(np.uint8)
    for i in range(n):
        img = np.roll(base, 2 * i, axis=1)
        cv2.imwrite(str(root / "inference_scene/images" / f"{i:04d}.png"), img)
        if i % gt_every == 0:
            m = np.zeros((48, 64), np.uint8)
            m[10:20, 10 + i:20 + i] = 255
            cv2.imwrite(str(root / "gt_mask" / f"{i:04d}.png"), m)
    return root


def test_paslcd_perturb_keeps_ground_truth_frames(tmp_path):
    import cv2
    from scene_change.harness import paslcd
    scene = _fake_oscd_scene(tmp_path / "data" / "Instance_1" / "S")
    info = paslcd.make_variant("oscd", scene, tmp_path / "v_cov", "coverage", 0.5)
    kept = sorted(p.name for p in (tmp_path / "v_cov" / "inference_scene" / "images").iterdir())
    assert len(info["removed"]) == 4 and len(kept) == 8          # half of the 8 frames without GT
    assert all(f"{i:04d}.png" in kept for i in range(0, 12, 3))
    assert (tmp_path / "v_cov" / "reference_scene").is_symlink()
    paslcd.make_variant("oscd", scene, tmp_path / "v_dark", "dark", -2.0)
    a = cv2.imread(str(scene / "inference_scene/images/0000.png"))
    b = cv2.imread(str(tmp_path / "v_dark/inference_scene/images/0000.png"))
    assert b.mean() < a.mean() and not (tmp_path / "v_dark/inference_scene/images/0000.png").is_symlink()
    paslcd.make_variant("oscd", scene, tmp_path / "v_blur", "blur", 0.1)
    c = cv2.imread(str(tmp_path / "v_blur/inference_scene/images/0005.png"))
    assert c.shape == a.shape and np.abs(c.astype(int) - cv2.imread(
        str(scene / "inference_scene/images/0005.png")).astype(int)).mean() > 5


def test_paslcd_score_matches_frame_metrics(tmp_path):
    import cv2
    from scene_change.harness import paslcd
    scene = _fake_oscd_scene(tmp_path / "S")
    pred, sc = tmp_path / "pred", tmp_path / "score"
    pred.mkdir()
    sc.mkdir()
    for g in sorted((scene / "gt_mask").iterdir()):
        m = cv2.imread(str(g), 0)
        cv2.imwrite(str(pred / g.name), m)                        # perfect masks
        np.save(sc / (g.stem + ".npy"), (m[None] / 255.0).astype(np.float16))
    rec = paslcd.score_outputs(scene / "gt_mask", pred, sc)
    assert rec["n_frames"] == 4 and rec["frames"]["F1"] == 1.0 and rec["calibration"]["ece"] < 1e-6


def test_official_patches_are_idempotent(tmp_path):
    from scene_change.harness import paslcd
    o = tmp_path / "oscd"
    o.mkdir()
    (o / "oscd.py").write_text(
        "import os\nimport numpy as np\ndef main():\n"
        "    os.makedirs(os.path.join(renders_path, \"change_mask\"), exist_ok=True)\n"
        "    for view in views:\n        with ctx:\n            change_mask = r()\n"
        "            change_mask = change_mask.mean(dim=0)\n            change_mask = (change_mask > 0.5).float()\n"
        "    if refine:\n        for view in views:\n            if True:\n                change_mask = r()\n"
        "                change_mask = change_mask.mean(dim=0)\n                change_mask = (change_mask > 0.5).float()\n")
    assert paslcd.patch_oscd(o) and not paslcd.patch_oscd(o)
    src = (o / "oscd.py").read_text()
    compile(src, "oscd.py", "exec")
    assert src.count("np.save(") == 2 and "change_score_refined" in src
    m = tmp_path / "mv"
    m.mkdir()
    (m / "render_viewpoints.py").write_text(
        "import os\nimport cv2\nimport numpy as np\nfor v in vs:\n    if mask:\n"
        "        final_image_mask = (final_image_mask * 255).astype(np.uint8)\n")
    (m / "train_masks.py").write_text("import sys\ndef training(scene):\n    test_cameras = scene.getTestCameras()\n"
                                      "    return test_cameras\n")
    assert paslcd.patch_mv3dcd(m) and not paslcd.patch_mv3dcd(m)
    for f in ("render_viewpoints.py", "train_masks.py"):
        compile((m / f).read_text(), f, "exec")
    ns = {}
    exec((m / "train_masks.py").read_text(), ns)

    class Cam:
        def __init__(self, n):
            self.image_name = n

    class Scene:
        def getTestCameras(self):
            return [Cam("a_test"), Cam("b_test")]
    import os
    os.environ["SCD_HOLDOUT"] = "a_test"
    try:
        assert [c.image_name for c in ns["training"](Scene())] == ["b_test"]
    finally:
        del os.environ["SCD_HOLDOUT"]
