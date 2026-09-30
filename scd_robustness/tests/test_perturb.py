import numpy as np
import pytest

from scd.config import fmt_severity, seed_from, variant_id
from scd.metrics import auroc, average_precision, confident_errors, ece_equal_mass, f1_at, score_hist
from scd.perturb import apply_to_uint8, linear_to_srgb, motion_kernel, srgb_to_linear, to_uint8
from scd.variants import perturb_positions, select_frames


def test_srgb_roundtrip_is_exact_for_uint8():
    x = np.arange(256, dtype=np.uint8)
    assert np.array_equal(to_uint8(linear_to_srgb(srgb_to_linear(x / 255.0))), x)


def test_identity_and_ev0_are_exact():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)
    out, _ = apply_to_uint8(img, "identity", 0, {}, np.random.default_rng(1))
    assert out is img
    out, _ = apply_to_uint8(img, "exposure", 0, {}, np.random.default_rng(1))
    assert np.array_equal(out, img)


def test_motion_kernel():
    assert motion_kernel(1, 0) is None
    for length, angle in [(2, 0), (16, 37.5), (31, 90), (64, 135)]:
        k = motion_kernel(length, angle)
        assert k.shape[0] % 2 == 1 and k.shape[0] >= length
        assert abs(k.sum() - 1) < 1e-5


@pytest.mark.parametrize("fn,sev,params", [
    ("motion_blur", 9, {"angle_deg": "random"}),
    ("exposure", -2, {}),
    ("white_balance", 0.2, {}),
    ("gain_field", 0.5, {"sigma_frac": 0.125, "min_gain": 0.05}),
    ("shadow", 0.6, {"area_frac": [0.1, 0.3], "feather_px": 5, "n_vertices": 6}),
    ("poisson_gaussian", 2, {"levels": {1: [400, 0.002], 2: [100, 0.005]}}),
    ("compose", 1, {"steps": [{"fn": "motion_blur", "severity": 5}, {"fn": "exposure", "severity": -1}]}),
])
def test_stressors_deterministic_and_valid(fn, sev, params):
    img = np.random.default_rng(3).integers(0, 256, (40, 60, 3), dtype=np.uint8)
    a, info_a = apply_to_uint8(img, fn, sev, params, np.random.default_rng(seed_from("x")))
    b, info_b = apply_to_uint8(img, fn, sev, params, np.random.default_rng(seed_from("x")))
    assert a.dtype == np.uint8 and a.shape == img.shape
    assert np.array_equal(a, b) and info_a == info_b
    assert not np.array_equal(a, img)


def test_exposure_darkens_and_brightens():
    img = np.full((8, 8, 3), 128, np.uint8)
    dark, _ = apply_to_uint8(img, "exposure", -1, {}, np.random.default_rng(0))
    bright, _ = apply_to_uint8(img, "exposure", 1, {}, np.random.default_rng(0))
    assert dark.mean() < 128 < bright.mean()


def test_views_subsets_are_nested_and_ordered():
    frames = ["f%02d.jpg" % i for i in range(25)]
    st = {"kind": "coverage", "mode": "random_nested"}
    kept = {f: select_frames(frames, st, f, 0, "Garden", "inf")[0] for f in (0.6, 0.4, 0.2)}
    assert [len(kept[f]) for f in (0.6, 0.4, 0.2)] == [15, 10, 5]
    assert set(kept[0.2]) <= set(kept[0.4]) <= set(kept[0.6])
    assert kept[0.6] == sorted(kept[0.6])
    other_trial = select_frames(frames, st, 0.4, 1, "Garden", "inf")[0]
    assert other_trial != kept[0.4]


def test_sector_drops_contiguous_block():
    frames = ["f%02d.jpg" % i for i in range(12)]
    centers = {"f%02d" % i: [np.cos(2 * np.pi * i / 12), 0.0, np.sin(2 * np.pi * i / 12)] for i in range(12)}
    kept, dropped, info = select_frames(frames, {"kind": "coverage", "mode": "sector"}, 0.25, 0, "S", "inf", centers)
    assert len(dropped) == 3 and len(kept) == 9
    idx = sorted(int(f[1:3]) for f in dropped)
    gaps = [(b - a) % 12 for a, b in zip(idx, idx[1:] + idx[:1])]
    assert sorted(gaps)[:2] == [1, 1]          # the three dropped views are neighbours on the circle


def test_burst_positions():
    assert perturb_positions(10, None) == list(range(10))
    assert perturb_positions(10, {"position": "start", "count": 3}) == [0, 1, 2]
    assert perturb_positions(10, {"position": "end", "count": 3}) == [7, 8, 9]
    assert perturb_positions(10, {"position": "middle", "count": 3}) == [3, 4, 5]


def test_labels():
    assert fmt_severity(-2) == "m2" and fmt_severity(0.6) == "0p6" and fmt_severity(32.0) == "32"
    assert variant_id("exposure", -1, 0) == "exposure-m1_t0"


def test_histogram_metrics():
    p = np.array([0.95] * 50 + [0.05] * 950)
    y = np.array([True] * 50 + [False] * 950)
    pos, neg = score_hist(p, y)
    assert auroc(pos, neg) == pytest.approx(1.0)
    assert average_precision(pos, neg) == pytest.approx(1.0)
    assert f1_at(pos, neg, 0.5) == pytest.approx(1.0)
    cw_fp, cw_fn = confident_errors(pos, neg)
    assert cw_fp == 0 and cw_fn == 0
    ece, _ = ece_equal_mass(pos, neg)
    assert ece == pytest.approx(0.05, abs=0.01)
