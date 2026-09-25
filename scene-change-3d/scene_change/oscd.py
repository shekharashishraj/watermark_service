"""O-SCD: online 3D scene change detection against a reference Gaussian map.

Re-implementation of Galappaththige et al., "Towards Online 3D Scene Change Detection"
(O-SCD, CVPR 2026, arXiv:2511.12370), following the paper and the official code
(github.com/Chumsy0725/O-SCD), on top of this package's CPU Gaussian renderer.

Per inference frame (RGB only):

1. Pose: keypoints are matched (mutual nearest neighbours) against every reference
   keyframe; the 4 keyframes with most matches give 2D-3D correspondences through their
   keypoints' known 3D positions; PnP-RANSAC, then a least-squares refinement.
2. Change cues: the reference is rendered at that pose and compared with the frame,
   C = C_pixel + C_feature, where C_pixel = 0.8 L1 + 0.2 (1 - SSIM) (min-max normalised)
   and C_feature is the channel-mean difference of SAM 2.1 image embeddings.
3. Change field R_change: the reference Gaussians, colours reset to zero, carry a learnable
   change colour. Every frame runs 16 Adam steps of
   L = mean(C (1 - sigmoid(M))) + log(1 + mean(sigmoid(M))^2) on the rendered change M,
   each on the current frame (p = 0.33) or a random earlier one; the frame's mask is M > 0.5.
4. Optionally: offline refinement (to 3000 iterations in total) and a change-guided
   update of the reference map.

Adaptations for CPU and RGB-D captures:

- R_change keeps the reference geometry frozen. Its render is then linear in the change
  colours, M = W c with the blending weights W of the reference, so dL/dc = W^T dL/dM is
  exact and no autograd rasteriser is needed. The official code also adapts positions,
  scales, rotations and opacities, and densifies at every frame's 5th iteration.
- The three change-colour channels receive identical gradients and stay identical, so a
  single channel is optimised (Adam is invariant to the 1/3 gradient scale).
- SIFT (RootSIFT) keypoints replace XFeat, and reference keypoints take their 3D position
  from reference depth instead of triangulation across reference views.
- A frame that PnP cannot localise falls back on the device odometry chained from the
  last localised frame; the official code assumes every frame localises.
- The feature backbone is the SAM 2.1 hiera-tiny image encoder through ONNX Runtime when
  the model file is available, and a weight-free dense-SIFT stand-in otherwise.
- The scene update rebuilds changed regions from the inference RGB-D frames (voxel
  fusion) instead of 5000 photometric iterations, and only prunes reference Gaussians
  that are not behind the observed surface.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage, sparse

from .gaussians import SH_C0, GaussianMap, build_gaussian_map
from .geometry import depth_edge_mask, invert_pose, make_pose, voxel_keys
from .render import render_gaussians
from .session import Session

DEFAULT_SAM2_ONNX = Path.home() / "models" / "sam2.1_hiera_tiny.encoder.onnx"


# ----------------------------------------------------------------------------
# Change cues
# ----------------------------------------------------------------------------

def _gauss1d(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    x = np.arange(size, dtype=np.float64) - size // 2
    g = np.exp(-x * x / (2 * sigma * sigma))
    return g / g.sum()


def ssim_map(a: np.ndarray, b: np.ndarray, window: int = 11) -> np.ndarray:
    """Per-pixel, per-channel SSIM as in the 3DGS code (Gaussian window, sigma 1.5, zero padding)."""
    g = _gauss1d(window)

    def blur(x):
        x = ndimage.correlate1d(x, g, axis=0, mode="constant")
        return ndimage.correlate1d(x, g, axis=1, mode="constant")

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    out = np.empty(a.shape, dtype=np.float64)
    for k in range(a.shape[2]):
        x = a[..., k].astype(np.float64)
        y = b[..., k].astype(np.float64)
        mx, my = blur(x), blur(y)
        sxx = blur(x * x) - mx * mx
        syy = blur(y * y) - my * my
        sxy = blur(x * y) - mx * my
        out[..., k] = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return out


class Sam2Encoder:
    """SAM 2.1 image encoder (ONNX export of hiera-tiny) -> (256, 64, 64) image embedding.

    Matches the official call: nearest-neighbour resize to 1024 x 1024 and [0, 1] RGB input
    without mean/std normalisation (``normalize=True`` applies the SAM preprocessing instead).
    """

    name = "sam2.1-hiera-tiny"

    def __init__(self, path=None, threads: int | None = None, normalize: bool = False):
        import onnxruntime as ort

        path = Path(path or os.environ.get("OSCD_SAM2_ONNX", DEFAULT_SAM2_ONNX))
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or int(os.environ.get("OSCD_THREADS", 4))
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        outs = [o.name for o in self.sess.get_outputs()]
        self.output = "image_embed" if "image_embed" in outs else outs[-1]
        self.normalize = normalize

    def __call__(self, img: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        ys = np.minimum((np.arange(1024) * H) // 1024, H - 1)
        xs = np.minimum((np.arange(1024) * W) // 1024, W - 1)
        x = img[ys][:, xs].astype(np.float32)
        if self.normalize:
            x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        return self.sess.run([self.output], {self.input: x})[0][0]


class DenseSiftEncoder:
    """Weight-free stand-in for the feature backbone: upright RootSIFT on a 64 x 64 grid."""

    name = "dense-sift"

    def __init__(self, grid: int = 64, size: int = 256):
        self.grid, self.size = grid, size
        step = size / grid
        c = (np.arange(grid) + 0.5) * step
        self.kps = [cv2.KeyPoint(float(x), float(y), float(2 * step), 0.0) for y in c for x in c]
        self.sift = cv2.SIFT_create()

    def __call__(self, img: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        kps, d = self.sift.compute(g, self.kps)
        if d is None or len(kps) != len(self.kps):
            raise RuntimeError("dense SIFT dropped grid points")
        d = np.sqrt(d / (d.sum(1, keepdims=True) + 1e-6))
        return d.T.reshape(-1, self.grid, self.grid)


def make_encoder(name: str | None = "auto"):
    """``auto``: SAM 2.1 via ONNX Runtime when its model file exists, else dense SIFT."""
    if name in (None, "none", "pixel"):
        return None
    if name == "dense-sift":
        return DenseSiftEncoder()
    if name in ("sam2", "auto"):
        path = Path(os.environ.get("OSCD_SAM2_ONNX", DEFAULT_SAM2_ONNX))
        ok = path.exists() and importlib.util.find_spec("onnxruntime") is not None
        if ok:
            return Sam2Encoder(path)
        if name == "sam2":
            raise FileNotFoundError(f"SAM 2.1 encoder not found at {path} (set OSCD_SAM2_ONNX)")
        return DenseSiftEncoder()
    raise ValueError(f"unknown feature backbone {name!r}")


def candidate_map(real: np.ndarray, render: np.ndarray, encoder=None, real_feat=None, render_feat=None):
    """Official ``generate_candidate_map``. Images are HxWx3 in [0, 1]; returns (C, C_pixel, C_feature)."""
    s = ssim_map(render, real).mean(axis=2)
    l1 = np.abs(render - real).mean(axis=2)
    delta = 0.8 * l1 + 0.2 * (1.0 - s)
    vmin, vmax = float(delta.min()), float(delta.max())
    c_pix = (delta - vmin) / (vmax - vmin + 1e-8)
    if encoder is None and real_feat is None:
        return c_pix, c_pix, None
    fa = encoder(render) if render_feat is None else render_feat
    fb = encoder(real) if real_feat is None else real_feat
    diff = np.abs(fa.astype(np.float32) - fb.astype(np.float32)).mean(axis=0)
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    H, W = real.shape[:2]
    up = cv2.resize(diff.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    # the official code re-normalises with the pixel cue's range before clamping
    c_feat = np.clip((up - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    return c_pix + c_feat, c_pix, c_feat


# ----------------------------------------------------------------------------
# Localisation against reference keyframes
# ----------------------------------------------------------------------------

def sift_features(rgb: np.ndarray, n_features: int, contrast: float = 0.005, upsample: int = 2):
    """Keypoints (n, 2) and RootSIFT descriptors (n, 128), unit L2 norm.

    Small, low-texture frames need a low contrast threshold and detection on an
    upsampled image to yield enough keypoints for PnP.
    """
    sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=contrast, edgeThreshold=20)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if upsample > 1:
        g = cv2.resize(g, None, fx=upsample, fy=upsample, interpolation=cv2.INTER_CUBIC)
    kps, d = sift.detectAndCompute(g, None)
    if d is None or not kps:
        return np.zeros((0, 2)), np.zeros((0, 128), np.float32)
    d = np.sqrt(d / (d.sum(1, keepdims=True) + 1e-7))
    pts = (np.array([k.pt for k in kps], dtype=np.float64) + 0.5) / upsample - 0.5
    return pts, d.astype(np.float32)


def mutual_matches(a: np.ndarray, b: np.ndarray, min_sim: float):
    """Official ``match``: mutual nearest neighbours with cosine similarity above ``min_sim``."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    S = a @ b.T
    j = S.argmax(1)
    i = S.argmax(0)
    q = np.arange(len(a))
    ok = (i[j] == q) & (S[q, j] > min_sim)
    return q[ok], j[ok]


@dataclass
class RefKeyframe:
    frame: int
    T_wc: np.ndarray
    kpts: np.ndarray       # (n, 2) pixels
    desc: np.ndarray       # (n, 128)
    xyz: np.ndarray        # (n, 3) reference-frame points; NaN without depth


class ReferenceIndex:
    """Reference keyframes with described keypoints and their 3D positions.

    The official code counts matches against every reference keyframe on the GPU to
    pick the best 4. On CPU a bag-of-words shortlist (``shortlist`` keyframes by TF-IDF
    similarity) comes first and the match counting runs on the shortlist only.
    """

    def __init__(self, session: Session, stride: int = 1, n_features: int = 2000, min_sim: float = 0.82,
                 poses=None, vocab_size: int = 256, shortlist: int = 20, seed: int = 0):
        self.K = session.K
        self.min_sim = min_sim
        self.shortlist = shortlist
        self.n_features = n_features
        self.keyframes: list[RefKeyframe] = []
        poses = session.poses if poses is None else poses
        for i in range(0, len(session), stride):
            kp, d = sift_features(session.rgb[i], n_features)
            xyz = np.full((len(kp), 3), np.nan)
            if len(kp):
                depth = session.depth[i]
                H, W = depth.shape
                ui = np.clip(np.round(kp[:, 0]).astype(int), 0, W - 1)
                vi = np.clip(np.round(kp[:, 1]).astype(int), 0, H - 1)
                z = depth[vi, ui].astype(np.float64)
                ok = (z > 0) & ~depth_edge_mask(depth, 0.06)[vi, ui]
                K = session.K
                pc = np.stack([(kp[:, 0] - K[0, 2]) / K[0, 0] * z, (kp[:, 1] - K[1, 2]) / K[1, 1] * z, z], 1)
                pw = pc @ poses[i][:3, :3].T + poses[i][:3, 3]
                xyz[ok] = pw[ok]
            self.keyframes.append(RefKeyframe(i, poses[i], kp, d, xyz))
        # visual vocabulary for the shortlist
        allD = np.concatenate([kf.desc for kf in self.keyframes if len(kf.desc)] or [np.zeros((0, 128), np.float32)])
        k = int(min(vocab_size, max(len(allD) // 20, 1)))
        rng = np.random.default_rng(seed)
        sub = allD[rng.choice(len(allD), min(len(allD), 60_000), replace=False)] if len(allD) else allD
        if len(sub) >= k > 1:
            cv2.setRNGSeed(seed)
            _, _, self.vocab = cv2.kmeans(sub, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1e-4),
                                          1, cv2.KMEANS_PP_CENTERS)
        else:
            self.vocab = np.zeros((1, 128), np.float32)
        tf = np.stack([self._words(kf.desc) for kf in self.keyframes])
        self.idf = np.log((len(tf) + 1) / (1 + np.sum(tf > 0, axis=0)))
        self.bow = self._bow(tf)

    def _words(self, desc: np.ndarray) -> np.ndarray:
        h = np.zeros(len(self.vocab))
        if len(desc):
            d2 = np.sum(self.vocab ** 2, axis=1)[None] - 2 * desc @ self.vocab.T
            h = np.bincount(d2.argmin(1), minlength=len(self.vocab)).astype(np.float64)
        return h

    def _bow(self, tf: np.ndarray) -> np.ndarray:
        v = tf * self.idf
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

    def match_counts(self, desc: np.ndarray, candidates) -> np.ndarray:
        """Official ``evaluate_match``: number of mutual matches with each candidate keyframe."""
        return np.array([len(mutual_matches(desc, self.keyframes[c].desc, self.min_sim)[0]) for c in candidates])

    def localize(self, rgb: np.ndarray, n_ref: int = 4, max_error: float = 1.5, ransac_iters: int = 2000,
                 min_inliers: int = 15, refine_iters: int = 20):
        """Camera-to-reference pose of an RGB frame, or (None, info)."""
        kp, d = sift_features(rgb, self.n_features)
        info = {"keypoints": len(kp), "matches": 0, "inliers": 0, "keyframes": []}
        if len(kp) < 6:
            return None, info
        sim = self.bow @ self._bow(self._words(d))
        cand = np.argsort(-sim, kind="stable")[:self.shortlist]
        top = cand[np.argsort(-self.match_counts(d, cand), kind="stable")[:n_ref]]
        info["keyframes"] = [int(self.keyframes[t].frame) for t in top]
        uv, xyz = [], []
        for t in top:
            kf = self.keyframes[t]
            ia, ib = mutual_matches(d, kf.desc, self.min_sim)
            if len(ia) >= 8:
                # outlier removal with the fundamental matrix (official remove_outliers=True)
                _, m = cv2.findFundamentalMat(kp[ia], kf.kpts[ib], cv2.FM_RANSAC, max_error, 0.999)
                if m is not None:
                    keep = m.ravel().astype(bool)
                    ia, ib = ia[keep], ib[keep]
            has = np.isfinite(kf.xyz[ib, 0])
            uv.append(kp[ia[has]])
            xyz.append(kf.xyz[ib[has]])
        uv = np.concatenate(uv) if uv else np.zeros((0, 2))
        xyz = np.concatenate(xyz) if xyz else np.zeros((0, 3))
        info["matches"] = len(uv)
        if len(uv) < max(6, min_inliers):
            return None, info
        ok, rvec, tvec, inl = cv2.solvePnPRansac(xyz, uv, self.K, None, iterationsCount=ransac_iters,
                                                 reprojectionError=max_error, confidence=0.9999,
                                                 flags=cv2.SOLVEPNP_AP3P)
        if not ok or inl is None or len(inl) < min_inliers:
            info["inliers"] = 0 if inl is None else len(inl)
            return None, info
        inl = inl[:, 0]
        # "mini-BA": pose-only Levenberg-Marquardt on the inliers
        rvec, tvec = cv2.solvePnPRefineLM(xyz[inl], uv[inl], self.K, None, rvec, tvec,
                                          (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, refine_iters, 1e-10))
        info["inliers"] = len(inl)
        T_cw = make_pose(cv2.Rodrigues(rvec)[0], tvec.ravel())
        return invert_pose(T_cw), info


# ----------------------------------------------------------------------------
# Change field R_change
# ----------------------------------------------------------------------------

class ChangeField:
    """Per-Gaussian change colour (SH DC, starting at 0) trained with Adam as in 3DGS."""

    def __init__(self, n: int, lr: float = 0.0025, betas=(0.9, 0.999), eps: float = 1e-15):
        self.dc = np.zeros(n)
        self.m = np.zeros(n)
        self.v = np.zeros(n)
        self.t = 0
        self.lr, self.betas, self.eps = lr, betas, eps

    @property
    def colors(self) -> np.ndarray:
        return np.maximum(SH_C0 * self.dc + 0.5, 0.0)

    def render(self, W: sparse.csr_matrix) -> np.ndarray:
        """Rendered change (clamped to [0, 1] like ``render_change``), black background."""
        return np.clip(W @ self.colors, 0.0, 1.0)

    def loss_and_grad(self, W: sparse.csr_matrix, C: np.ndarray, dc: np.ndarray | None = None):
        """L_SSF = mean(C (1 - sigmoid(M))) + log(1 + mean(sigmoid(M))^2) and its gradient w.r.t. dc."""
        dc = self.dc if dc is None else dc
        raw = W @ np.maximum(SH_C0 * dc + 0.5, 0.0)
        M = np.clip(raw, 0.0, 1.0)
        s = 1.0 / (1.0 + np.exp(-M))
        sbar = float(s.mean())
        loss = float(np.mean(C * (1.0 - s)) + np.log(sbar * sbar + 1.0))
        g = s * (1.0 - s) * (2.0 * sbar / (sbar * sbar + 1.0) - C) / len(M)
        g *= (raw >= 0.0) & (raw <= 1.0)
        grad = SH_C0 * (W.T @ g)
        grad *= (SH_C0 * dc + 0.5) >= 0.0
        return loss, grad

    def step(self, W: sparse.csr_matrix, C: np.ndarray) -> float:
        """One Adam step on L_SSF for a view with blending weights W and candidate map C (flat)."""
        loss, grad = self.loss_and_grad(W, C)
        b1, b2 = self.betas
        self.t += 1
        self.m = b1 * self.m + (1 - b1) * grad
        self.v = b2 * self.v + (1 - b2) * grad * grad
        mh = self.m / (1 - b1 ** self.t)
        vh = self.v / (1 - b2 ** self.t)
        self.dc -= self.lr * mh / (np.sqrt(vh) + self.eps)
        return loss


# ----------------------------------------------------------------------------
# Online pipeline
# ----------------------------------------------------------------------------

@dataclass
class OSCDConfig:
    backbone: str | None = "auto"      # auto | sam2 | dense-sift | none
    iters_per_frame: int = 16
    p_current: float = 0.33            # probability of training on the newest frame
    lr: float = 0.0025                 # feature_lr of 3DGS
    threshold: float = 0.5
    refine_until: int = 3000           # offline refinement: total iterations (0 disables)
    n_ref_keyframes: int = 4
    ref_stride: int = 1                # every reference frame is a keyframe, as in the official code
    n_features: int = 2000
    min_sim: float = 0.82
    max_error_frac: float = 2e-3       # PnP / F-matrix threshold: max(frac * width, 1.5) px
    min_inliers: int = 15
    match_exposure: bool = False       # not in the paper: fit a global gain of the frame to the render
    seed: int = 0


@dataclass
class OSCDResult:
    masks: np.ndarray                  # (N, H, W) bool: online mask, rendered right after the frame
    masks_refined: np.ndarray | None   # (N, H, W) bool: final field rendered at every frame
    poses: np.ndarray                  # (N, 4, 4) camera -> reference
    localized: np.ndarray              # (N,) PnP succeeded (else odometry fallback / unusable)
    usable: np.ndarray                 # (N,) frame has a pose and was processed
    inliers: np.ndarray                # (N,)
    change_colors: np.ndarray          # per-Gaussian change value, 0.5 = unchanged prior
    candidates: np.ndarray             # (N, H, W) float16 candidate maps C
    losses: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    backbone: str = "none"
    iterations: int = 0
    scores: np.ndarray | None = None          # (N, H, W) float16 rendered change M behind ``masks``
    scores_refined: np.ndarray | None = None  # (N, H, W) float16 rendered change M behind ``masks_refined``


class _FeatureCache:
    """On-disk cache of image embeddings keyed by the image bytes (the encoder is deterministic)."""

    def __init__(self, encoder, cache_dir=None):
        self.encoder = encoder
        self.dir = None if cache_dir is None else Path(cache_dir)
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.seconds = 0.0
        self.calls = 0

    def __call__(self, img: np.ndarray) -> np.ndarray:
        key = None
        if self.dir is not None:
            key = hashlib.sha1(np.ascontiguousarray(img, dtype=np.float32).tobytes()).hexdigest()[:20]
            f = self.dir / f"{self.encoder.name}_{key}.npy"
            if f.exists():
                return np.load(f).astype(np.float32)
        t = time.time()
        e = self.encoder(img)
        self.seconds += time.time() - t
        self.calls += 1
        if key is not None:
            e = e.astype(np.float16)
            np.save(self.dir / f"{self.encoder.name}_{key}.npy", e)
            e = e.astype(np.float32)       # same values as a later cache hit
        return e


def exposure_gain(real: np.ndarray, render: np.ndarray, valid: np.ndarray) -> float:
    """Robust global gain g minimising |g * real - render| over ``valid`` pixels (median of ratios)."""
    a = real.mean(axis=2)[valid]
    b = render.mean(axis=2)[valid]
    ok = a > 0.05
    return float(np.median(b[ok] / a[ok])) if ok.sum() > 50 else 1.0


def run_oscd(reference: GaussianMap, ref_session: Session, session: Session, cfg: OSCDConfig | None = None,
             index: ReferenceIndex | None = None, encoder=None, cache_dir=None, poses=None, render_fn=None,
             verbose: bool = False) -> OSCDResult:
    """Online change detection of ``session`` (RGB used; depth only via the reference keyframes).

    ``poses`` (camera -> reference) skips localisation, e.g. to isolate the change stage.
    ``render_fn(i, T_wc) -> HxWx3`` replaces the reference render used for the change cues
    (the change field still uses the reference Gaussians), e.g. an ideal render in a harness.
    """
    cfg = cfg or OSCDConfig()
    rng = np.random.default_rng(cfg.seed)
    t_all = time.time()
    n = len(session)
    H, W = session.depth.shape[1:]
    K = session.K
    encoder = make_encoder(cfg.backbone) if encoder is None else encoder
    feats = None if encoder is None else _FeatureCache(encoder, cache_dir)
    t0 = time.time()
    if poses is None and index is None:
        index = ReferenceIndex(ref_session, stride=cfg.ref_stride, n_features=cfg.n_features, min_sim=cfg.min_sim)
    t_index = time.time() - t0
    max_error = max(cfg.max_error_frac * W, 1.5)

    field_ = ChangeField(len(reference), lr=cfg.lr)
    est = np.tile(np.eye(4), (n, 1, 1))
    localized = np.zeros(n, bool)
    usable = np.zeros(n, bool)
    inliers = np.zeros(n, int)
    masks = np.zeros((n, H, W), bool)
    scores = np.zeros((n, H, W), np.float16)
    cands = np.zeros((n, H, W), np.float16)
    views: list[tuple[int, sparse.csr_matrix, np.ndarray]] = []
    losses = []
    t_loc = t_render = t_cue = t_opt = 0.0
    last = None
    for i in range(n):
        t0 = time.time()
        if poses is not None:
            T, ok = poses[i], True
        else:
            T, info = index.localize(session.rgb[i], cfg.n_ref_keyframes, max_error, min_inliers=cfg.min_inliers)
            inliers[i] = info["inliers"]
            ok = T is not None
            if not ok and last is not None:
                # chain the device odometry from the last localised frame
                T = est[last] @ invert_pose(session.poses[last]) @ session.poses[i]
        t_loc += time.time() - t0
        if T is None:
            continue
        est[i] = T
        localized[i] = ok
        usable[i] = True
        if ok:
            last = i
        t0 = time.time()
        rend = render_gaussians(reference, K, T, W, H, return_weights=True)
        t_render += time.time() - t0
        t0 = time.time()
        real = session.rgb[i].astype(np.float64) / 255.0
        img = np.clip(rend["rgb"] if render_fn is None else render_fn(i, T), 0.0, 1.0)
        if cfg.match_exposure:
            real = np.clip(real * exposure_gain(real, img, rend["alpha"] > 0.5), 0.0, 1.0)
        if feats is None:
            C, _, _ = candidate_map(real, img)
        else:
            C, _, _ = candidate_map(real, img, real_feat=feats(real), render_feat=feats(img))
        cands[i] = C
        t_cue += time.time() - t0
        views.append((i, rend["weights"], C.ravel()))
        t0 = time.time()
        for _ in range(cfg.iters_per_frame):
            k = int(rng.integers(0, len(views))) if rng.random() > cfg.p_current else len(views) - 1
            losses.append(field_.step(views[k][1], views[k][2]))
        M = field_.render(rend["weights"]).reshape(H, W)
        scores[i] = M
        masks[i] = M > cfg.threshold
        t_opt += time.time() - t0
        if verbose and i % 25 == 0:
            print(f"frame {i}: loc={ok} inl={inliers[i]} loss={losses[-1]:.4f} mask={masks[i].mean():.3f}", flush=True)

    refined = scores_refined = None
    if cfg.refine_until and views:
        t0 = time.time()
        for _ in range(field_.t, cfg.refine_until):
            k = int(rng.integers(0, len(views)))
            losses.append(field_.step(views[k][1], views[k][2]))
        refined = np.zeros((n, H, W), bool)
        scores_refined = np.zeros((n, H, W), np.float16)
        for i, Wv, _ in views:
            M = field_.render(Wv).reshape(H, W)
            scores_refined[i] = M
            refined[i] = M > cfg.threshold
        t_opt += time.time() - t0
    timings = {"index_s": round(t_index, 2), "localize_s": round(t_loc, 2), "render_s": round(t_render, 2),
               "cues_s": round(t_cue, 2), "features_s": round(0.0 if feats is None else feats.seconds, 2),
               "optimize_s": round(t_opt, 2), "total_s": round(time.time() - t_all, 2), "frames": n,
               "fps": round(n / max(time.time() - t_all - t_index, 1e-9), 3)}
    return OSCDResult(masks, refined, est, localized, usable, inliers, field_.colors, cands, losses, timings,
                      "none" if encoder is None else encoder.name, field_.t, scores, scores_refined)


# ----------------------------------------------------------------------------
# Change-guided scene update
# ----------------------------------------------------------------------------

def concat_maps(a: GaussianMap, b: GaussianMap) -> GaussianMap:
    keys = ("means", "scales", "quats", "colors", "opacity", "normals", "counts")
    sh = None
    if a.sh_rest is not None and b.sh_rest is not None:
        sh = np.concatenate([a.sh_rest, b.sh_rest])
    return GaussianMap(*(np.concatenate([getattr(a, k), getattr(b, k)]) for k in keys), voxel=a.voxel, sh_rest=sh)


def update_scene(reference: GaussianMap, result: OSCDResult, session: Session, use_refined: bool = True,
                 radius: int | None = None, depth_aware: bool = True, min_frames: int = 2):
    """Change-guided update of the reference map (paper Sec. 3.4, adapted to RGB-D input).

    1. Prune reference Gaussians blended into eroded change pixels in more than one
       pixel-view (official ``score > 1``); with ``depth_aware`` only those in front of
       or at the observed surface, so surfaces hidden behind a new object survive.
    2. Rebuild the dilated change regions from the inference RGB-D frames and add the new
       Gaussians where the pruned reference has none.
    Returns (updated map, stats).
    """
    masks = result.masks_refined if (use_refined and result.masks_refined is not None) else result.masks
    H, W = masks.shape[1:]
    r = radius if radius is not None else max(1, int(round(11 * W / 1200)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    counts = np.zeros(len(reference))
    means = reference.means.astype(np.float64)
    masked_depth = np.zeros_like(session.depth)
    for i in np.nonzero(result.usable)[0]:
        m = masks[i].astype(np.uint8)
        if not m.any():
            continue
        dil = cv2.dilate(m, kernel) > 0
        masked_depth[i][dil] = session.depth[i][dil]
        er = (cv2.erode(m, kernel) > 0).ravel()
        if not er.any():
            continue
        Wv = render_gaussians(reference, session.K, result.poses[i], W, H, return_weights=True)["weights"]
        sub = Wv[np.nonzero(er)[0]].tocoo()
        rows = np.nonzero(er)[0][sub.row]
        g = sub.col
        keep = np.ones(len(g), bool)
        if depth_aware:
            T_cw = invert_pose(result.poses[i])
            z = means[g] @ T_cw[2, :3] + T_cw[2, 3]
            d = session.depth[i].ravel()[rows].astype(np.float64)
            keep = (d <= 0) | (z < d + 0.05 + 0.02 * d)
        counts += np.bincount(g[keep], minlength=len(reference))
    pruned = counts > 1
    kept = reference.subset(np.nonzero(~pruned)[0])
    stats = {"pruned": int(pruned.sum()), "kept": len(kept)}
    new = None
    if masked_depth.any():
        masked = Session(session.name + "_changes", session.K, session.rgb, masked_depth, session.poses,
                         session.timestamps, dict(session.meta))
        try:
            new = build_gaussian_map(masked, voxel=reference.voxel, poses=result.poses, min_frames=min_frames)
        except ValueError:
            new = None
    if new is not None and len(new):
        occupied = set(voxel_keys(kept.means.astype(np.float64), reference.voxel).tolist())
        fresh = np.array([k not in occupied for k in voxel_keys(new.means.astype(np.float64), reference.voxel)])
        new = new.subset(np.nonzero(fresh)[0])
        stats["added"] = len(new)
        return concat_maps(kept, new), stats
    stats["added"] = 0
    return kept, stats


# ----------------------------------------------------------------------------
# Reference colour fit (frozen geometry)
# ----------------------------------------------------------------------------

def refit_colors(gm: GaussianMap, session: Session, poses=None, frames=None, iters: int = 40, lam: float = 0.02,
                 min_alpha: float = 0.5) -> GaussianMap:
    """Photometric least-squares fit of the colours of a frozen-geometry map.

    Minimises sum_v ||W_v c - I_v||^2 + lam ||c - c0||^2 with conjugate gradients, which
    stands in for the photometric training of the reference 3DGS. Pixels the map does not
    cover (accumulated alpha < ``min_alpha``) are ignored.
    """
    poses = session.poses if poses is None else poses
    frames = range(len(session)) if frames is None else frames
    H, W = session.depth.shape[1:]
    mats, rhs = [], np.zeros((len(gm), 3))
    for i in frames:
        r = render_gaussians(gm, session.K, poses[i], W, H, return_weights=True)
        rows = np.nonzero(r["alpha"].ravel() >= min_alpha)[0]
        Wv = r["weights"][rows]
        mats.append(Wv)
        rhs += Wv.T @ (session.rgb[i].reshape(-1, 3)[rows].astype(np.float64) / 255.0)
    c0 = gm.colors.astype(np.float64)
    b = rhs + lam * c0

    def A(x):
        out = lam * x
        for Wv in mats:
            out += Wv.T @ (Wv @ x)
        return out

    x = c0.copy()
    r_ = b - A(x)
    p = r_.copy()
    rs = np.sum(r_ * r_, axis=0)
    for _ in range(iters):
        Ap = A(p)
        alpha = rs / np.maximum(np.sum(p * Ap, axis=0), 1e-30)
        x += p * alpha
        r_ -= Ap * alpha
        rs_new = np.sum(r_ * r_, axis=0)
        p = r_ + p * (rs_new / np.maximum(rs, 1e-30))
        rs = rs_new
    out = gm.subset(np.arange(len(gm)))
    out.colors = np.clip(x, 0.0, 1.0).astype(gm.colors.dtype)
    return out
