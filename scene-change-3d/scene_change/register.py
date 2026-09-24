"""Localize an inspection walkthrough inside the baseline model.

1. Global, gravity-aligned 4-DoF search: phones report gravity, so only yaw and
   translation are unknown. Vertical structure (walls, furniture sides) is
   rasterised into a floor-plan image for both sessions and cross-correlated with
   FFTs for every candidate yaw; the floor height fixes z.
2. Robust point-to-plane ICP against the baseline Gaussians refines all 6 DoF.
   Changed objects are outliers and are down-weighted.
3. Drift correction: the walkthrough is split into short chunks that are each
   re-aligned; well-conditioned corrections are interpolated per frame and kept
   only where they improve the local fit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .gaussians import frame_points
from .geometry import make_pose, rot_z, so3_exp, so3_log, voxel_keys
from .model import BaselineModel
from .session import Session


def session_cloud(session: Session, frames=None, stride: int = 4, voxel: float = 0.05, max_depth: float = 4.5,
                  poses=None):
    """Voxel-downsampled points + mean normals + colours of (a subset of) a session, in the pose frame."""
    frames = range(len(session)) if frames is None else frames
    P, N, C = [], [], []
    for i in frames:
        p, c, n, cos_inc, _ = frame_points(session, i, stride, max_depth, None if poses is None else poses[i])
        ok = (np.linalg.norm(n, axis=1) > 0.5) & (cos_inc > 0.2)
        P.append(p[ok])
        N.append(n[ok])
        C.append(c[ok])
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))
    P = np.concatenate(P).astype(np.float64)
    N = np.concatenate(N).astype(np.float64)
    C = np.concatenate(C).astype(np.float64)
    if len(P) == 0:
        return P, N, C
    keys = voxel_keys(P, voxel)
    uk, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    Pm = np.stack([np.bincount(inv, P[:, k]) for k in range(3)], 1) / cnt[:, None]
    Nm = np.stack([np.bincount(inv, N[:, k]) for k in range(3)], 1)
    Nm /= np.maximum(np.linalg.norm(Nm, axis=1, keepdims=True), 1e-9)
    Cm = np.stack([np.bincount(inv, C[:, k]) for k in range(3)], 1) / cnt[:, None]
    good = np.linalg.norm(Nm, axis=1) > 0.5
    return Pm[good], Nm[good], Cm[good]


def floor_height(P: np.ndarray, N: np.ndarray) -> float:
    z = P[N[:, 2] > 0.9, 2]
    if len(z) < 20:
        return float(np.percentile(P[:, 2], 2)) if len(P) else 0.0
    bins = np.arange(z.min() - 0.02, z.max() + 0.04, 0.02)
    h, e = np.histogram(z, bins)
    peak = h.max()
    for i in range(len(h)):
        if h[i] >= 0.25 * peak:
            j = i + int(np.argmax(h[i:i + 5]))
            return float((e[j] + e[j + 1]) / 2)
    return float(z.min())


def _structure_points(P, N, floor):
    m = (np.abs(N[:, 2]) < 0.3) & (P[:, 2] > floor + 0.25) & (P[:, 2] < floor + 2.1)
    return P[m, :2]


def _raster(xy, origin, shape, cell):
    ij = np.floor((xy - origin) / cell).astype(np.int64)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[1])
    g = np.zeros(shape)
    np.add.at(g, (ij[ok, 0], ij[ok, 1]), 1.0)
    g = np.log1p(g)
    return ndimage.gaussian_filter(g, 1.0)


def global_candidates(src_xy, dst_xy, cell=0.1, yaw_step_deg=2.0, top=6):
    """Yaw/translation hypotheses (yaw, tx, ty, score) mapping src floor-plan onto dst."""
    dmin = dst_xy.min(0) - 0.5
    dshape = tuple((np.ceil((dst_xy.max(0) + 0.5 - dmin) / cell)).astype(int))
    D = _raster(dst_xy, dmin, dshape, cell)
    c_src = src_xy.mean(0)
    rel = src_xy - c_src
    ext = np.linalg.norm(rel, axis=1).max() + 0.5
    sshape = (int(np.ceil(2 * ext / cell)),) * 2
    N0, N1 = dshape[0] + sshape[0], dshape[1] + sshape[1]
    FD = np.fft.rfft2(D, s=(N0, N1))
    results = []
    for yaw in np.radians(np.arange(0, 360, yaw_step_deg)):
        c, s = np.cos(yaw), np.sin(yaw)
        r = rel @ np.array([[c, s], [-s, c]])
        S = _raster(r, np.array([-ext, -ext]), sshape, cell)
        corr = np.fft.irfft2(FD * np.conj(np.fft.rfft2(S, s=(N0, N1))), s=(N0, N1))
        k = np.unravel_index(int(np.argmax(corr)), corr.shape)
        kk = [k[0] if k[0] <= N0 - sshape[0] else k[0] - N0, k[1] if k[1] <= N1 - sshape[1] else k[1] - N1]
        # src rotated point r maps to dst: r - (-ext) + dmin + kk*cell ... plus the source centroid removed earlier
        t = dmin + np.array(kk) * cell + ext
        tx, ty = t - np.array([c * c_src[0] - s * c_src[1], s * c_src[0] + c * c_src[1]])
        results.append((yaw, tx, ty, float(corr[k])))
    results.sort(key=lambda r: -r[3])
    chosen = []
    for r in results:
        if all(abs((np.degrees(r[0] - q[0]) + 180) % 360 - 180) > 8 for q in chosen):
            chosen.append(r)
        if len(chosen) >= top:
            break
    return chosen


@dataclass
class ICPResult:
    T: np.ndarray
    inlier_ratio: float
    rmse: float
    hessian: np.ndarray
    iterations: int


def icp(src_P, src_N, tree: cKDTree, dst_P, dst_N, T0, schedule=(0.5, 0.3, 0.15, 0.08, 0.05), iters=6,
        damping=1e-6, inlier_tol=0.05):
    T = T0.copy()
    it_total = 0
    H = np.zeros((6, 6))
    for dmax in schedule:
        for _ in range(iters):
            it_total += 1
            Q = src_P @ T[:3, :3].T + T[:3, 3]
            NQ = src_N @ T[:3, :3].T
            dist, j = tree.query(Q, distance_upper_bound=dmax)
            ok = np.isfinite(dist)
            if ok.sum() < 30:
                break
            q, nq, p, n = Q[ok], NQ[ok], dst_P[j[ok]], dst_N[j[ok]]
            compat = np.sum(nq * n, axis=1) > 0.5
            q, p, n = q[compat], p[compat], n[compat]
            if len(q) < 30:
                break
            r = np.sum(n * (q - p), axis=1)
            c = max(dmax / 3.0, 0.01)
            w = 1.0 / (1.0 + (r / c) ** 2)
            ctr = q.mean(0)
            A = np.hstack([np.cross(q - ctr, n), n])
            Aw = A * w[:, None]
            H = A.T @ Aw
            g = -(Aw.T @ r)
            lam = damping * np.trace(H) / 6 + 1e-9
            x = np.linalg.solve(H + lam * np.eye(6), g)
            dT = make_pose(np.eye(3), ctr) @ make_pose(so3_exp(x[:3]), x[3:]) @ make_pose(np.eye(3), -ctr)
            T = dT @ T
            if np.linalg.norm(x[:3]) < 1e-5 and np.linalg.norm(x[3:]) < 1e-5:
                break
    Q = src_P @ T[:3, :3].T + T[:3, 3]
    NQ = src_N @ T[:3, :3].T
    dist, j = tree.query(Q, distance_upper_bound=0.3)
    ok = np.isfinite(dist)
    res = np.full(len(Q), np.inf)
    if ok.any():
        compat = np.sum(NQ[ok] * dst_N[j[ok]], axis=1) > 0.5
        rr = np.abs(np.sum(dst_N[j[ok]] * (Q[ok] - dst_P[j[ok]]), axis=1))
        rr[~compat] = np.inf
        res[ok] = rr
    inl = res < inlier_tol
    rmse = float(np.sqrt(np.mean(res[inl] ** 2))) if inl.any() else float("inf")
    return ICPResult(T, float(inl.mean()) if len(res) else 0.0, rmse, H, it_total)


def score_alignment(src_P, src_N, tree: cKDTree, dst_P, dst_N, T, tol=0.05):
    """Fraction of source points within ``tol`` (point-to-plane) of a compatible target surfel."""
    if len(src_P) == 0:
        return 0.0
    Q = src_P @ T[:3, :3].T + T[:3, 3]
    NQ = src_N @ T[:3, :3].T
    dist, j = tree.query(Q, distance_upper_bound=0.3)
    ok = np.isfinite(dist)
    if not ok.any():
        return 0.0
    jj = j[ok]
    compat = np.sum(NQ[ok] * dst_N[jj], axis=1) > 0.5
    rr = np.abs(np.sum(dst_N[jj] * (Q[ok] - dst_P[jj]), axis=1))
    return float(np.sum(compat & (rr < tol)) / len(Q))


@dataclass
class Registration:
    T_bs: np.ndarray                    # session frame -> baseline frame
    poses: np.ndarray                   # (N,4,4) camera -> baseline, drift-corrected
    inlier_ratio: float
    rmse: float
    candidates: list = field(default_factory=list)
    chunk_stats: dict = field(default_factory=dict)
    seconds: float = 0.0
    confident: bool = True


def _interp_twists(centers, corrs, n_frames):
    """Per-frame correction twists (rotation vector, translation) interpolated between chunk centres."""
    rv = np.zeros((n_frames, 3))
    tv = np.zeros((n_frames, 3))
    if not centers:
        return rv, tv
    centers = np.asarray(centers, dtype=np.float64)
    R = np.stack([so3_log(C[:3, :3]) for C in corrs])
    t = np.stack([C[:3, 3] for C in corrs])
    f = np.arange(n_frames)
    for k in range(3):
        rv[:, k] = np.interp(f, centers, R[:, k])
        tv[:, k] = np.interp(f, centers, t[:, k])
    return rv, tv


def _twists_to_poses(rv, tv):
    return np.stack([make_pose(so3_exp(r), t) for r, t in zip(rv, tv)])


def _window_gain(session, frames, poses_a, poses_b, tree, dst_P, dst_N):
    aP, aN, _ = session_cloud(session, frames=frames, stride=4, voxel=0.05, poses=poses_a)
    if len(aP) < 200:
        return None
    bP, bN, _ = session_cloud(session, frames=frames, stride=4, voxel=0.05, poses=poses_b)
    return (score_alignment(bP, bN, tree, dst_P, dst_N, np.eye(4))
            - score_alignment(aP, aN, tree, dst_P, dst_N, np.eye(4)))


def _local_gate(session, poses, new_poses, tree, dst_P, dst_N, chunk, margin=0.005):
    """Per-frame weight in [0, 1]: keep a correction only where it does not hurt the local fit.

    Interpolation carries corrections across (and beyond) chunks that could not be
    re-aligned themselves; a room at the end of a walkthrough can otherwise inherit
    the drift of its neighbour.
    """
    n = len(session)
    gsum = np.zeros(n)
    gcnt = np.zeros(n)
    half = max(chunk // 2, 4)
    for s0 in range(0, n, half):
        fr = list(range(s0, min(n, s0 + half)))
        if len(fr) < 3:
            continue
        g = _window_gain(session, fr, poses, new_poses, tree, dst_P, dst_N)
        if g is None:
            continue
        gsum[fr] += g
        gcnt[fr] += 1
    gain = np.where(gcnt > 0, gsum / np.maximum(gcnt, 1), 0.0)
    w = (gain >= -margin).astype(np.float64)
    return ndimage.uniform_filter1d(w, size=5, mode="nearest"), gain


def localize(model: BaselineModel, session: Session, chunk: int = 20, refine_chunks: bool = True,
             yaw_step_deg: float = 2.0, verbose: bool = False) -> Registration:
    t0 = time.time()
    g = model.gaussians
    dst_P = g.means.astype(np.float64)
    dst_N = g.normals.astype(np.float64)
    tree = cKDTree(dst_P)
    src_P, src_N, _ = session_cloud(session, stride=4, voxel=0.06)
    f_src = floor_height(src_P, src_N)
    f_dst = model.floor_z
    s_xy = _structure_points(src_P, src_N, f_src)
    d_mask = (np.abs(dst_N[:, 2]) < 0.3) & (dst_P[:, 2] > f_dst + 0.25) & (dst_P[:, 2] < f_dst + 2.1)
    d_xy = dst_P[d_mask, :2]
    cands = global_candidates(s_xy, d_xy, yaw_step_deg=yaw_step_deg)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(src_P), size=min(len(src_P), 25_000), replace=False)
    sP, sN = src_P[sub], src_N[sub]
    best = None
    tried = []
    for yaw, tx, ty, score in cands:
        T0 = make_pose(rot_z(yaw), np.array([tx, ty, f_dst - f_src]))
        res = icp(sP, sN, tree, dst_P, dst_N, T0)
        tried.append({"yaw_deg": round(float(np.degrees(yaw)), 1), "corr": round(score, 1),
                      "inlier_ratio": round(res.inlier_ratio, 3), "rmse": round(res.rmse, 4)})
        if verbose:
            print("candidate", tried[-1])
        if best is None or res.inlier_ratio > best.inlier_ratio:
            best = res
    T_bs = best.T
    poses = np.einsum("ij,njk->nik", T_bs, session.poses)
    stats = {"chunks": 0, "accepted": 0, "mean_shift_cm": 0.0, "max_shift_cm": 0.0, "reverted": False}
    if refine_chunks and len(session) > chunk:
        cands_c = []
        for s0 in range(0, len(session), chunk // 2):
            fr = list(range(s0, min(len(session), s0 + chunk)))
            if len(fr) < 6:
                continue
            stats["chunks"] += 1
            cP, cN, _ = session_cloud(session, frames=fr, stride=3, voxel=0.05, poses=poses)
            if len(cP) < 400:
                continue
            before = score_alignment(cP, cN, tree, dst_P, dst_N, np.eye(4))
            res = icp(cP, cN, tree, dst_P, dst_N, np.eye(4), schedule=(0.12, 0.07, 0.04), iters=5, damping=1e-3)
            Ht = np.linalg.eigvalsh(res.hessian[3:, 3:])
            Hr = np.linalg.eigvalsh(res.hessian[:3, :3])
            cond_ok = Ht[0] > 0.04 * Ht[-1] and Hr[0] > 0.01 * Hr[-1]
            shift = float(np.linalg.norm(res.T[:3, 3]))
            ang = float(np.degrees(np.linalg.norm(so3_log(res.T[:3, :3]))))
            if cond_ok and shift < 0.2 and ang < 3 and res.inlier_ratio >= before + 0.02:
                cands_c.append((float(np.mean(fr)), res.T, shift))
        # drift is smooth: drop corrections that disagree with their neighbours
        kept = []
        for k, (cen, Tc, sh) in enumerate(cands_c):
            nb = [cands_c[j][1][:3, 3] for j in range(max(0, k - 2), min(len(cands_c), k + 3)) if j != k]
            if nb and np.linalg.norm(Tc[:3, 3] - np.median(nb, axis=0)) > 0.06:
                continue
            kept.append((cen, Tc, sh))
        if kept:
            rv, tv = _interp_twists([k[0] for k in kept], [k[1] for k in kept], len(session))
            new_poses = np.einsum("nij,njk->nik", _twists_to_poses(rv, tv), poses)
            w, _ = _local_gate(session, poses, new_poses, tree, dst_P, dst_N, chunk)
            stats["gated_frames"] = int(np.sum(w < 0.5))
            new_poses = np.einsum("nij,njk->nik", _twists_to_poses(rv * w[:, None], tv * w[:, None]), poses)
            aP, aN, _ = session_cloud(session, stride=5, voxel=0.06, poses=poses)
            bP, bN, _ = session_cloud(session, stride=5, voxel=0.06, poses=new_poses)
            if score_alignment(bP, bN, tree, dst_P, dst_N, np.eye(4)) >= score_alignment(aP, aN, tree, dst_P, dst_N, np.eye(4)):
                poses = new_poses
                stats["accepted"] = len(kept)
                stats["mean_shift_cm"] = round(100 * float(np.mean([k[2] for k in kept])), 2)
                stats["max_shift_cm"] = round(100 * float(np.max([k[2] for k in kept])), 2)
            else:
                stats["reverted"] = True
    # final quality on all points with corrected poses
    fP, fN, _ = session_cloud(session, stride=4, voxel=0.06, poses=poses)
    final_ratio = score_alignment(fP, fN, tree, dst_P, dst_N, np.eye(4))
    ratios = sorted((t["inlier_ratio"] for t in tried), reverse=True)
    ambiguous = len(ratios) > 1 and ratios[1] > 0.9 * ratios[0] and ratios[0] < 0.5
    confident = best.inlier_ratio > 0.3 and not ambiguous
    return Registration(T_bs, poses, final_ratio, best.rmse, tried, stats, time.time() - t0, bool(confident))
