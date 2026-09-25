"""2D video-comparison baseline: frame retrieval, homography alignment, image differencing.

What an inspector (or a simple tool) does without a 3D model: for every inspection
frame, find the most similar baseline frame, warp it onto the inspection frame with a
homography, match brightness, and threshold the difference. Parallax breaks the
homography for anything off the dominant plane, which is the weakness the 3D methods
address.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

from .oscd import mutual_matches, sift_features
from .session import Session


@dataclass
class Video2DConfig:
    n_features: int = 2000
    vocab_size: int = 256
    shortlist: int = 8
    min_sim: float = 0.82
    ransac_px: float = 3.0
    min_inliers: int = 12
    blur_sigma: float = 1.0
    abs_threshold: float = 0.12        # mean absolute RGB difference in [0, 1]
    rel_threshold: float = 4.0         # ... and this many times the frame's median difference
    border: int = 3                    # ignore pixels this close to the warped image border
    min_area: int = 12


@dataclass
class Video2DResult:
    masks: np.ndarray                  # (N, H, W) bool
    valid: np.ndarray                  # (N, H, W) bool: pixels covered by the aligned baseline frame
    matched: np.ndarray                # (N,) index of the baseline frame used, -1 if none
    inliers: np.ndarray                # (N,)
    timings: dict = field(default_factory=dict)
    scores: np.ndarray | None = None   # (N, H, W) float16: difference / (2 x threshold), clipped to [0, 1]


def _affine_colour(src: np.ndarray, dst: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-channel gain/offset mapping src to dst, refit once on the better half of pixels."""
    out = src.copy()
    for c in range(3):
        a, b = src[..., c][valid], dst[..., c][valid]
        if len(a) < 50:
            continue
        sel = np.ones(len(a), bool)
        for _ in range(2):
            A = np.stack([a[sel], np.ones(sel.sum())], 1)
            g, o = np.linalg.lstsq(A, b[sel], rcond=None)[0]
            r = np.abs(g * a + o - b)
            sel = r <= np.median(r) * 2 + 1e-6
        out[..., c] = g * src[..., c] + o
    return np.clip(out, 0.0, 1.0)


class _Retrieval:
    def __init__(self, session: Session, cfg: Video2DConfig):
        self.cfg = cfg
        self.feats = [sift_features(session.rgb[i], cfg.n_features) for i in range(len(session))]
        allD = np.concatenate([d for _, d in self.feats if len(d)] or [np.zeros((0, 128), np.float32)])
        rng = np.random.default_rng(0)
        sub = allD[rng.choice(len(allD), min(len(allD), 60_000), replace=False)] if len(allD) else allD
        k = int(min(cfg.vocab_size, max(len(sub) // 20, 1)))
        if k > 1:
            cv2.setRNGSeed(0)
            _, _, self.vocab = cv2.kmeans(sub, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1e-4),
                                          1, cv2.KMEANS_PP_CENTERS)
        else:
            self.vocab = np.zeros((1, 128), np.float32)
        tf = np.stack([self._words(d) for _, d in self.feats])
        self.idf = np.log((len(tf) + 1) / (1 + np.sum(tf > 0, axis=0)))
        self.bow = self._bow(tf)

    def _words(self, d):
        h = np.zeros(len(self.vocab))
        if len(d):
            d2 = np.sum(self.vocab ** 2, axis=1)[None] - 2 * d @ self.vocab.T
            h = np.bincount(d2.argmin(1), minlength=len(self.vocab)).astype(np.float64)
        return h

    def _bow(self, tf):
        v = tf * self.idf
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

    def best_match(self, kp, d):
        """(frame index, homography mapping baseline pixels to inspection pixels, inliers)."""
        cfg = self.cfg
        if len(d) < 8:
            return -1, None, 0
        cand = np.argsort(-(self.bow @ self._bow(self._words(d))), kind="stable")[:cfg.shortlist]
        best = (-1, None, 0)
        for c in cand:
            kp_r, d_r = self.feats[c]
            ia, ib = mutual_matches(d, d_r, cfg.min_sim)
            if len(ia) < max(cfg.min_inliers, 8):
                continue
            Hm, m = cv2.findHomography(kp_r[ib], kp[ia], cv2.RANSAC, cfg.ransac_px)
            n = 0 if m is None else int(m.sum())
            if Hm is not None and n > best[2]:
                best = (int(c), Hm, n)
        return best if best[2] >= cfg.min_inliers else (-1, None, best[2])


def build_retrieval(reference: Session, cfg: Video2DConfig | None = None) -> _Retrieval:
    """Index of the baseline frames, reusable across inspections of the same baseline."""
    return _Retrieval(reference, cfg or Video2DConfig())


def compare_videos(reference: Session, inspection: Session, cfg: Video2DConfig | None = None,
                   retrieval: _Retrieval | None = None) -> Video2DResult:
    """Per-frame change masks; ``scores`` puts the frame's decision threshold at 0.5."""
    cfg = cfg or Video2DConfig()
    t0 = time.time()
    ret = retrieval if retrieval is not None else _Retrieval(reference, cfg)
    t_index = time.time() - t0
    n = len(inspection)
    H, W = inspection.rgb.shape[1:3]
    masks = np.zeros((n, H, W), bool)
    scores = np.zeros((n, H, W), np.float16)
    valid = np.zeros((n, H, W), bool)
    matched = np.full(n, -1)
    inliers = np.zeros(n, int)
    ones = np.ones((H, W), np.uint8)
    kernel = np.ones((3, 3), bool)
    for i in range(n):
        kp, d = sift_features(inspection.rgb[i], cfg.n_features)
        j, Hm, ninl = ret.best_match(kp, d)
        matched[i], inliers[i] = j, ninl
        if j < 0:
            continue
        ref = cv2.warpPerspective(reference.rgb[j], Hm, (W, H), flags=cv2.INTER_LINEAR).astype(np.float64) / 255.0
        cov = cv2.warpPerspective(ones, Hm, (W, H), flags=cv2.INTER_NEAREST) > 0
        cov = ndimage.binary_erosion(cov, iterations=cfg.border)
        cur = inspection.rgb[i].astype(np.float64) / 255.0
        ref = _affine_colour(ref, cur, cov)
        if cfg.blur_sigma > 0:
            ref = ndimage.gaussian_filter(ref, (cfg.blur_sigma, cfg.blur_sigma, 0))
            cur = ndimage.gaussian_filter(cur, (cfg.blur_sigma, cfg.blur_sigma, 0))
        diff = np.abs(cur - ref).mean(axis=2)
        thr = max(cfg.abs_threshold, cfg.rel_threshold * float(np.median(diff[cov])) if cov.any() else 1.0)
        scores[i] = np.where(cov, np.clip(0.5 * diff / thr, 0.0, 1.0), 0.0)
        m = (diff > thr) & cov
        m = ndimage.binary_opening(m, structure=kernel)
        lab, k = ndimage.label(m)
        if k:
            sizes = ndimage.sum(m, lab, index=np.arange(1, k + 1))
            m = np.isin(lab, np.nonzero(sizes >= cfg.min_area)[0] + 1)
        masks[i], valid[i] = m, cov
    total = time.time() - t0
    return Video2DResult(masks, valid, matched, inliers,
                         {"index_s": round(t_index, 2), "total_s": round(total, 2), "frames": n,
                          "fps": round(n / max(total - t_index, 1e-9), 2)}, scores)
