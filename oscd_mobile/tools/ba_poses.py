"""Prototype: global bundle adjustment of a scan's ARKit poses on XFeat reprojection error (what Voxelio
and PocketGS run before training). Unlike refine_poses.py (LiDAR alignment), this fits the *images*:
tracks of mutual-nearest XFeat matches across overlapping keyframes, 3D points initialised from LiDAR
depth, then Levenberg–Marquardt over every camera (frame 0 fixed) and point, Huber on pixel residuals,
cameras solved through the Schur complement (points are independent 3×3 blocks). A weak prior keeps
camera centres near ARKit's (fixes the scale gauge).

Usage:
    oscd-harness features <session> Models/XFeat_480x352.mlpackage <feats-dir>
    python3 tools/ba_poses.py <session> <feats-dir> <out-session>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine_poses import FLIP, Frame, match_features, overlap_pairs, rodrigues, skew  # noqa: E402

SIGMA_PX = 1.0          # expected reprojection noise at working resolution
PRIOR_SIGMA_M = 0.10    # camera-centre prior (gauge + drift bound)
HUBER_K = 2.0


class View(Frame):
    """Frame plus the keypoints that have no LiDAR depth (BA can still use them once triangulated)."""
    def __init__(self, root, feats_dir, kf, man):
        super().__init__(root, feats_dir, kf, man)
        K = man["intrinsics"]
        self.fpx, self.cxp, self.cyp = K["f"], K["cx"], K["cy"]
        kp = np.load(os.path.join(feats_dir, f"{kf['id']:06d}_kpts.npy"))
        ft = np.load(os.path.join(feats_dir, f"{kf['id']:06d}_feats.npy"))
        ud = np.clip(np.round((kp[:, 0] + 0.5) * self.s - 0.5).astype(int), 0, self.dw - 1)
        vd = np.clip(np.round((kp[:, 1] + 0.5) * self.s - 0.5).astype(int), 0, self.dh - 1)
        p3 = self.P[vd, ud]
        keep = np.abs(ft).sum(-1) > 0
        self.kp2 = kp[keep]
        self.kp3 = np.where(np.isfinite(p3[keep]).all(-1, keepdims=True), p3[keep], np.nan)
        self.feats = ft[keep] / np.linalg.norm(ft[keep], axis=-1, keepdims=True)


class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_tracks(views, T):
    """Observations (cam, kp index, point id) from pairwise mutual matches, merged by union-find."""
    offs = np.cumsum([0] + [len(v.kp2) for v in views])
    uf = UnionFind(offs[-1])
    pairs = [(i, j) for (i, j) in overlap_pairs(views, T) if i < j]
    n_match = 0
    n_raw = 0
    for (i, j) in pairs:
        a, b = match_features(views[i], views[j])
        n_raw += len(a)
        # Only matches whose LiDAR-lifted positions agree (3 % of depth, ≥ 3 cm) join a track: a wrong
        # match would otherwise chain two tracks into one and lose both.
        pa = views[i].kp3[a]; pb = views[j].kp3[b]
        wa = pa @ T[i][:3, :3].T + T[i][:3, 3]; wb = pb @ T[j][:3, :3].T + T[j][:3, 3]
        tol = np.maximum(0.03, 0.03 * np.minimum(pa[:, 2], pb[:, 2]))
        ok = np.isfinite(wa).all(-1) & np.isfinite(wb).all(-1) & (np.linalg.norm(wa - wb, axis=-1) < tol)
        for x, y in zip(a[ok], b[ok]):
            uf.union(offs[i] + x, offs[j] + y)
        n_match += int(ok.sum())
    roots = np.array([uf.find(k) for k in range(offs[-1])])
    cam = np.concatenate([np.full(len(v.kp2), i) for i, v in enumerate(views)])
    kp = np.concatenate([np.arange(len(v.kp2)) for v in views])
    # Keep tracks seen from ≥ 2 cameras, one observation per camera: XFeat often fires twice on one
    # corner, so a camera that appears twice keeps its first keypoint rather than losing the track.
    order = np.argsort(roots, kind="stable")
    roots, cam, kp = roots[order], cam[order], kp[order]
    _, start, count = np.unique(roots, return_index=True, return_counts=True)
    obs_cam, obs_kp, obs_pt, pt_id = [], [], [], 0
    for s, c in zip(start, count):
        cams, kps = cam[s:s + c], kp[s:s + c]
        ucams, first = np.unique(cams, return_index=True)
        if len(ucams) < 2:
            continue
        obs_cam.append(ucams)
        obs_kp.append(kps[first])
        obs_pt.append(np.full(len(ucams), pt_id))
        pt_id += 1
    print(f"{len(pairs)} overlapping pairs, {n_raw} pairwise matches, {n_match} 3D-consistent → {pt_id} tracks, "
          f"{sum(len(o) for o in obs_cam)} observations")
    return np.concatenate(obs_cam), np.concatenate(obs_kp), np.concatenate(obs_pt), pt_id


def init_points(views, T, obs_cam, obs_kp, obs_pt, n_pts):
    """Each point = median of its LiDAR-lifted observations; tracks whose lifts disagree > 10 cm are
    wrong matches and are dropped."""
    X = np.full((n_pts, 3), np.nan)
    lifted = np.full((len(obs_cam), 3), np.nan)
    for i, v in enumerate(views):
        sel = obs_cam == i
        p = v.kp3[obs_kp[sel]]
        ok = np.isfinite(p).all(-1)
        w = np.full_like(p, np.nan)
        w[ok] = p[ok] @ T[i][:3, :3].T + T[i][:3, 3]
        lifted[sel] = w
    order = np.argsort(obs_pt, kind="stable")
    pt_sorted = obs_pt[order]
    _, start, count = np.unique(pt_sorted, return_index=True, return_counts=True)
    keep_pt = np.zeros(n_pts, bool)
    for s, c in zip(start, count):
        idx = order[s:s + c]
        L = lifted[idx]
        L = L[np.isfinite(L).all(-1)]
        if len(L) == 0:
            continue
        med = np.median(L, 0)
        if len(L) > 1 and np.linalg.norm(L - med, axis=-1).max() > 0.10:
            continue
        X[pt_sorted[s]] = med
        keep_pt[pt_sorted[s]] = True
    return X, keep_pt


def residuals_and_jacobians(views, T, X, obs_cam, obs_kp, obs_pt):
    """Per observation: residual (2,), J_cam (2×6, world-frame left twist of c2w), J_pt (2×3)."""
    M = len(obs_cam)
    r = np.zeros((M, 2)); Jc = np.zeros((M, 2, 6)); Jp = np.zeros((M, 2, 3)); valid = np.zeros(M, bool)
    for i, v in enumerate(views):
        sel = np.nonzero(obs_cam == i)[0]
        if len(sel) == 0:
            continue
        R = T[i][:3, :3]; c = T[i][:3, 3]
        Xw = X[obs_pt[sel]]
        Xc = (Xw - c) @ R
        z = Xc[:, 2]
        ok = z > 0.1
        f = v.fpx
        px = np.stack([Xc[:, 0] / np.where(ok, z, 1) * f + v.cxp, Xc[:, 1] / np.where(ok, z, 1) * f + v.cyp], -1)
        r[sel] = px - v.kp2[obs_kp[sel]]
        dpi = np.zeros((len(sel), 2, 3))
        dpi[:, 0, 0] = f / z; dpi[:, 0, 2] = -f * Xc[:, 0] / z ** 2
        dpi[:, 1, 1] = f / z; dpi[:, 1, 2] = -f * Xc[:, 1] / z ** 2
        dXc_dX = np.broadcast_to(R.T, (len(sel), 3, 3))
        Jp[sel] = dpi @ dXc_dX
        # X_cam = Rᵀ(exp(−ξ)·X − c) ≈ X_cam − Rᵀ([I, −[X]×] ξ)
        G = np.concatenate([np.broadcast_to(np.eye(3), (len(sel), 3, 3)), -skew(Xw)], -1)
        Jc[sel] = -(dpi @ dXc_dX @ G)
        valid[sel] = ok
    return r, Jc, Jp, valid


def bundle_adjust(views, T0, X, obs_cam, obs_kp, obs_pt, iters=15):
    n = len(views)
    T = [t.copy() for t in T0]
    A = [t.copy() for t in T0]
    X = X.copy()
    n_pts = len(X)
    lam = 1e-3
    gate = 20.0
    for it in range(iters):
        r, Jc, Jp, valid = residuals_and_jacobians(views, T, X, obs_cam, obs_kp, obs_pt)
        rn = np.linalg.norm(r, axis=-1)
        use = valid & (rn < gate)
        w = np.where(rn > HUBER_K * SIGMA_PX, HUBER_K * SIGMA_PX / np.maximum(rn, 1e-9), 1.0) / SIGMA_PX ** 2
        w[~use] = 0
        # Normal equations. Camera 0 is fixed: its block is dropped.
        Hcc = np.zeros((6 * n, 6 * n)); bc = np.zeros(6 * n)
        Hpp = np.zeros((n_pts, 3, 3)); bp = np.zeros((n_pts, 3))
        wJc = w[:, None, None] * Jc
        np.add.at(Hpp, obs_pt, np.einsum("mdk,mdl->mkl", w[:, None, None] * Jp, Jp))
        np.add.at(bp, obs_pt, np.einsum("mdk,md->mk", w[:, None, None] * Jp, r))
        for i in range(n):
            sel = obs_cam == i
            Hcc[6 * i:6 * i + 6, 6 * i:6 * i + 6] += np.einsum("mdk,mdl->kl", wJc[sel], Jc[sel])
            bc[6 * i:6 * i + 6] += np.einsum("mdk,md->k", wJc[sel], r[sel])
        # Camera-centre prior toward ARKit: residual (c − c_arkit), J = [I, −[c]×].
        for i in range(1, n):
            c = T[i][:3, 3]
            Jpr = np.concatenate([np.eye(3), -skew(c)], -1)
            Hcc[6 * i:6 * i + 6, 6 * i:6 * i + 6] += Jpr.T @ Jpr / PRIOR_SIGMA_M ** 2
            bc[6 * i:6 * i + 6] += Jpr.T @ (c - A[i][:3, 3]) / PRIOR_SIGMA_M ** 2
        # Schur complement: S = Hcc − Σ_j W_ij Hpp_j⁻¹ W_kjᵀ, with W_ij = Σ_obs wJcᵀ Jp.
        Hpp_inv = np.linalg.inv(Hpp + 1e-9 * np.eye(3) + lam * np.einsum("jkk->jk", Hpp)[:, :, None] * np.eye(3))
        W = np.einsum("mdk,mdl->mkl", wJc, Jp)                        # per observation 6×3
        S = Hcc.copy(); bs = bc.copy()
        # group observations by point to combine pairs (i, k) of cameras seeing the same point
        order = np.argsort(obs_pt, kind="stable")
        pts_sorted = obs_pt[order]
        _, start, count = np.unique(pts_sorted, return_index=True, return_counts=True)
        y = np.einsum("jkl,jl->jk", Hpp_inv, bp)                       # Hpp⁻¹ bp
        for s, c in zip(start, count):
            idx = order[s:s + c]
            j = pts_sorted[s]
            cams = obs_cam[idx]
            Wi = W[idx]                                               # c × 6 × 3
            WHinv = Wi @ Hpp_inv[j]                                   # c × 6 × 3
            for a in range(c):
                ia = cams[a]
                bs[6 * ia:6 * ia + 6] -= WHinv[a] @ bp[j]
                for b in range(c):
                    ib = cams[b]
                    S[6 * ia:6 * ia + 6, 6 * ib:6 * ib + 6] -= WHinv[a] @ Wi[b].T
        Sr = S[6:, 6:] + lam * np.diag(np.diag(S[6:, 6:])) + 1e-9 * np.eye(6 * n - 6)
        dc = np.concatenate([np.zeros(6), np.linalg.solve(Sr, -bs[6:])])
        # Back-substitute the points: Δp_j = Hpp_j⁻¹ (−bp_j − Σ_i W_ijᵀ Δc_i)
        acc = np.zeros((n_pts, 3))
        np.add.at(acc, obs_pt, np.einsum("mkl,mk->ml", W, dc.reshape(n, 6)[obs_cam]))
        dX = np.einsum("jkl,jl->jk", Hpp_inv, -bp - acc)
        # Apply
        Tn = [t.copy() for t in T]
        for k in range(1, n):
            t, om = dc[6 * k:6 * k + 3], dc[6 * k + 3:6 * k + 6]
            Rk = rodrigues(om)
            Tn[k][:3, :3] = Rk @ T[k][:3, :3]
            Tn[k][:3, 3] = Rk @ T[k][:3, 3] + t
        Xn = X + dX
        r2, _, _, v2 = residuals_and_jacobians(views, Tn, Xn, obs_cam, obs_kp, obs_pt)
        cost0 = np.sum(np.minimum(rn[use] ** 2, 2 * HUBER_K * SIGMA_PX * rn[use] - (HUBER_K * SIGMA_PX) ** 2))
        rn2 = np.linalg.norm(r2, axis=-1)
        use2 = v2 & (rn2 < gate)
        cost1 = np.sum(np.minimum(rn2[use2] ** 2, 2 * HUBER_K * SIGMA_PX * rn2[use2] - (HUBER_K * SIGMA_PX) ** 2))
        accepted = cost1 < cost0
        if accepted:
            T, X, lam = Tn, Xn, max(lam / 3, 1e-6)
        else:
            lam = min(lam * 5, 1e3)
        med_before = np.median(rn[use]); med_after = np.median(rn2[use2]) if accepted else med_before
        print(f"iter {it:2d}  gate {gate:4.1f} px  median |r| {med_before:5.2f} → {med_after:5.2f} px  "
              f"p90 {np.percentile(rn2[use2] if accepted else rn[use], 90):5.2f}  used {use.sum()}  "
              f"{'accepted' if accepted else 'rejected'}  λ {lam:.1e}  max cam step {np.abs(dc.reshape(n, 6)[:, :3]).max() * 1000:5.1f} mm")
        gate = max(gate * 0.75, 4.0)
    return T, X


def main():
    root, feats_dir, out = sys.argv[1:4]
    man = json.load(open(os.path.join(root, "session.json")))
    views = [View(root, feats_dir, kf, man) for kf in man["keyframes"]]
    T0 = [v.c2w0.copy() for v in views]
    obs_cam, obs_kp, obs_pt, n_pts = build_tracks(views, T0)
    X, keep = init_points(views, T0, obs_cam, obs_kp, obs_pt, n_pts)
    sel = keep[obs_pt]
    obs_cam, obs_kp, obs_pt = obs_cam[sel], obs_kp[sel], obs_pt[sel]
    print(f"{keep.sum()} tracks with a consistent LiDAR position, {len(obs_cam)} observations")
    r, _, _, v = residuals_and_jacobians(views, T0, X, obs_cam, obs_kp, obs_pt)
    rn = np.linalg.norm(r[v], axis=-1)
    print(f"ARKit poses + LiDAR points: reprojection error median {np.median(rn):.2f} px, p90 {np.percentile(rn, 90):.2f} px "
          f"(working image {views[0].W}×{views[0].H})")
    T, X = bundle_adjust(views, T0, X, obs_cam, obs_kp, obs_pt)
    moved = [np.linalg.norm(T[k][:3, 3] - T0[k][:3, 3]) * 1000 for k in range(len(T))]
    rot = [np.degrees(np.arccos(np.clip((np.trace(T[k][:3, :3].T @ T0[k][:3, :3]) - 1) / 2, -1, 1))) for k in range(len(T))]
    print(f"cameras moved: median {np.median(moved):.1f} mm (max {np.max(moved):.1f}), rotated median {np.median(rot):.2f}° (max {np.max(rot):.2f}°)")
    os.makedirs(out, exist_ok=True)
    for kf, t in zip(man["keyframes"], T):
        kf["arkitCameraToWorld"] = [float(x) for x in (t @ FLIP).T.reshape(-1)]
    json.dump(man, open(os.path.join(out, "session.json"), "w"), indent=1)
    link = os.path.join(out, "keyframes")
    if not os.path.exists(link):
        os.symlink(os.path.abspath(os.path.join(root, "keyframes")), link)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
