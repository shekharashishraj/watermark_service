import numpy as np
from scipy import sparse

from scene_change.gaussians import GaussianMap
from scene_change.geometry import camera_pose, intrinsics
from scene_change.oscd import (ChangeField, ReferenceIndex, candidate_map, exposure_gain, mutual_matches,
                               ssim_map)
from scene_change.render import render_gaussians
from scene_change.session import Session


def _texture(h, w, seed=0):
    rng = np.random.default_rng(seed)
    import cv2
    x = rng.random((h // 4, w // 4, 3))
    x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(x, 0, 1)


def test_ssim_of_identical_images_is_one():
    img = _texture(40, 50)
    assert np.allclose(ssim_map(img, img), 1.0)


def test_candidate_map_is_zero_for_identical_images_and_peaks_on_change():
    img = _texture(60, 80)
    C, cp, cf = candidate_map(img, img)
    assert np.allclose(C, 0.0) and cf is None
    other = img.copy()
    other[20:40, 30:50] = 1.0 - other[20:40, 30:50]
    C, _, _ = candidate_map(other, img)
    assert C[25:35, 35:45].mean() > 3 * C[:, :20].mean()


def test_change_field_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    n_pix, n_g = 60, 25
    W = sparse.random(n_pix, n_g, density=0.2, random_state=2, format="csr") * 0.4
    C = rng.random(n_pix) * 2
    field = ChangeField(n_g)
    dc = rng.normal(0, 0.3, n_g)
    loss, grad = field.loss_and_grad(W, C, dc)
    eps = 1e-6
    num = np.array([(field.loss_and_grad(W, C, dc + eps * e)[0] - field.loss_and_grad(W, C, dc - eps * e)[0])
                    / (2 * eps) for e in np.eye(n_g)])
    assert np.allclose(grad, num, atol=1e-7)


def test_change_field_keeps_consistent_cues_and_drops_inconsistent_ones():
    # 40 Gaussians seen by 6 views, one Gaussian per pixel per view.
    n_g, rng = 40, np.random.default_rng(3)
    views = []
    for v in range(6):
        W = sparse.identity(n_g, format="csr")
        C = np.full(n_g, 0.2)
        C[:8] = 1.8                                   # a real change: high cue in every view
        C[8 + v * 5: 13 + v * 5] = 1.8                # view-specific distractor
        views.append((W, C))
    field = ChangeField(n_g)
    for it in range(16 * 6):
        W, C = views[int(rng.integers(len(views)))]
        field.step(W, C)
    M = field.render(views[0][0])
    assert (M[:8] > 0.5).all()
    assert (M[8:] > 0.5).mean() < 0.25


def _plane_session(n=3):
    """A textured fronto-parallel wall at 2 m seen by nearby cameras."""
    W, H = 160, 120
    K = intrinsics(W, H, 68.0)
    img = (_texture(H, W, 5) * 255).astype(np.uint8)
    rgb = np.stack([img] * n)
    depth = np.full((n, H, W), 2.0, np.float32)
    poses = np.stack([camera_pose(np.array([0.0, 0.0, 1.5]), 0.0, 0.0) for _ in range(n)])
    return Session("plane", K, rgb, depth, poses)


def test_pnp_localises_a_reference_frame_at_its_own_pose():
    s = _plane_session()
    idx = ReferenceIndex(s, vocab_size=16, shortlist=3)
    T, info = idx.localize(s.rgb[0], max_error=1.5)
    assert T is not None, info
    assert np.linalg.norm(T[:3, 3] - s.poses[0][:3, 3]) < 0.02


def test_mutual_matches_pairs_identical_descriptors():
    rng = np.random.default_rng(0)
    d = rng.random((30, 128)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    a, b = mutual_matches(d, d[::-1], 0.82)
    assert len(a) == 30 and np.all(b == 29 - a)


def test_exposure_gain_recovers_global_scale():
    img = _texture(40, 40) * 0.8 + 0.1
    g = exposure_gain(img * 0.7, img, np.ones((40, 40), bool))
    assert abs(g - 1 / 0.7) < 1e-6


def test_render_weights_reproduce_the_colour_render():
    rng = np.random.default_rng(0)
    n = 400
    means = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(2.5, 3.5, n), rng.uniform(0.5, 2.5, n)])
    q = np.tile([1.0, 0, 0, 0], (n, 1))
    gm = GaussianMap(means.astype(np.float32), np.full((n, 3), 0.06, np.float32), q.astype(np.float32),
                     rng.random((n, 3)).astype(np.float32), np.full(n, 0.9, np.float32),
                     np.tile([0, -1.0, 0], (n, 1)).astype(np.float32), np.ones(n, np.int32))
    K = intrinsics(80, 60, 68.0)
    T = camera_pose(np.array([0.0, 0.0, 1.5]), np.pi / 2, 0.0)
    r = render_gaussians(gm, K, T, 80, 60, return_weights=True)
    Wm = r["weights"]
    assert Wm.shape == (80 * 60, n)
    rgb = (Wm @ gm.colors.astype(np.float64)).reshape(60, 80, 3)
    # identical except for the < 1e-4 transmittance tail the weights drop
    assert np.abs(rgb - r["rgb"]).max() < 2e-3
    assert r["alpha"].max() > 0.5
