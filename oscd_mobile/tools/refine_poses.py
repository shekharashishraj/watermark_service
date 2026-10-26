"""Prototype: refine a scan's ARKit keyframe poses before training (LiDAR alignment + XFeat matches).

ARKit drifts over a room-sized walk: keyframes seconds apart place the same wall centimetres apart, and the
splat then fits each view on its own. This solves one least-squares problem over every keyframe pose
(frame 0 fixed) with three kinds of terms, re-associated each outer iteration:

- LiDAR: point-to-plane distance between overlapping keyframes' depth maps (projective association);
- XFeat: mutual-nearest-neighbour matches, lifted to 3D with LiDAR depth, must meet in the world;
- odometry: consecutive keyframes keep ARKit's relative pose (accurate over short spans).

Usage:
    oscd-harness features <session> Models/XFeat_480x352.mlpackage <feats-dir>
    python3 tools/refine_poses.py <session> <feats-dir> <out-session>

The output session links the input's keyframes and carries the refined `arkitCameraToWorld` poses.
"""
import json
import os
import sys

import numpy as np

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])     # ARKit camera axes ↔ OpenCV camera axes
SIGMA_ICP, SIGMA_FEAT, SIGMA_ODO = 0.01, 0.02, 0.01
ICP_STRIDE = 3                              # depth-map pixels between ICP samples


def skew(v):
    z = np.zeros(v.shape[:-1])
    return np.stack([np.stack([z, -v[..., 2], v[..., 1]], -1),
                     np.stack([v[..., 2], z, -v[..., 0]], -1),
                     np.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = skew(w / th)
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * k @ k


def apply(T, R_pts):
    return R_pts @ T[:3, :3].T + T[:3, 3]


class Frame:
    def __init__(self, root, feats_dir, kf, man):
        K = man["intrinsics"]
        self.W, self.H = K["width"], K["height"]
        self.dw, self.dh = man["depthWidth"], man["depthHeight"]
        self.s = self.dw / self.W
        self.f = K["f"] * self.s
        self.cx, self.cy = (K["cx"] + 0.5) * self.s - 0.5, (K["cy"] + 0.5) * self.s - 0.5
        stem = os.path.join(root, "keyframes", f"{kf['id']:06d}")
        depth, conf = np.load(stem + "_depth.npy").astype(np.float64), np.load(stem + "_conf.npy")
        ok = (conf >= 2) & np.isfinite(depth) & (depth > 0.1) & (depth < 6)
        v, u = np.mgrid[0:self.dh, 0:self.dw]
        P = np.stack([(u - self.cx) / self.f * depth, (v - self.cy) / self.f * depth, depth], -1)
        P[~ok] = np.nan
        # Normals from central differences; dropped across depth jumps.
        dx = P[1:-1, 2:] - P[1:-1, :-2]
        dy = P[2:, 1:-1] - P[:-2, 1:-1]
        n = np.cross(dx, dy)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        jump = (np.linalg.norm(dx, axis=-1) > 0.05) | (np.linalg.norm(dy, axis=-1) > 0.05)
        n[jump] = np.nan
        N = np.full_like(P, np.nan)
        N[1:-1, 1:-1] = n
        N[np.sum(N * P, -1) > 0] *= -1                  # face the camera
        self.P, self.N = P, N
        good = np.isfinite(P).all(-1) & np.isfinite(N).all(-1)
        g = np.zeros_like(good)
        g[::ICP_STRIDE, ::ICP_STRIDE] = True
        self.src = P[good & g]
        self.src_n = N[good & g]
        # XFeat keypoints with LiDAR depth.
        kp = np.load(os.path.join(feats_dir, f"{kf['id']:06d}_kpts.npy"))
        ft = np.load(os.path.join(feats_dir, f"{kf['id']:06d}_feats.npy"))
        ud = np.clip(np.round((kp[:, 0] + 0.5) * self.s - 0.5).astype(int), 0, self.dw - 1)
        vd = np.clip(np.round((kp[:, 1] + 0.5) * self.s - 0.5).astype(int), 0, self.dh - 1)
        p3 = P[vd, ud]
        keep = np.isfinite(p3).all(-1) & (np.abs(ft).sum(-1) > 0)
        self.kp3, self.feats = p3[keep], ft[keep] / np.linalg.norm(ft[keep], axis=-1, keepdims=True)
        self.c2w0 = np.array(kf["arkitCameraToWorld"], float).reshape(4, 4).T @ FLIP

    def project(self, Xc):
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.round(Xc[:, 0] / z * self.f + self.cx)
            v = np.round(Xc[:, 1] / z * self.f + self.cy)
        ok = (z > 0.1) & (u >= 0) & (u < self.dw) & (v >= 0) & (v < self.dh)
        return np.where(ok, u, 0).astype(int), np.where(ok, v, 0).astype(int), ok


def overlap_pairs(frames, T, min_overlap=0.2):
    pairs = []
    for i, fi in enumerate(frames):
        xw = apply(T[i], fi.src[::4])
        for j, fj in enumerate(frames):
            if i == j:
                continue
            _, _, ok = fj.project(apply(np.linalg.inv(T[j]), xw))
            if ok.mean() >= min_overlap:
                pairs.append((i, j))
    return pairs


def match_features(fi, fj, min_sim=0.82):
    sim = fi.feats @ fj.feats.T
    a = sim.argmax(1)
    b = sim.argmax(0)
    i = np.arange(len(a))
    mutual = (b[a] == i) & (sim[i, a] >= min_sim)
    return i[mutual], a[mutual]


def accumulate(H, bvec, i, j, Ji, Jj, r, w):
    """Adds Σ w·Jᵀ J and Σ w·Jᵀ r for residual rows r (M × d) with Jacobians Ji, Jj (M × d × 6)."""
    wJi = w[:, None, None] * Ji
    wJj = w[:, None, None] * Jj
    si, sj = slice(6 * i, 6 * i + 6), slice(6 * j, 6 * j + 6)
    H[si, si] += np.einsum("mdk,mdl->kl", wJi, Ji)
    H[sj, sj] += np.einsum("mdk,mdl->kl", wJj, Jj)
    Hij = np.einsum("mdk,mdl->kl", wJi, Jj)
    H[si, sj] += Hij
    H[sj, si] += Hij.T
    bvec[si] += np.einsum("mdk,md->k", wJi, r)
    bvec[sj] += np.einsum("mdk,md->k", wJj, r)


def point_jac(x):
    """d(exp(ξ)·x)/dξ for ξ = (t, ω) applied on the left in the world: [I, −[x]×]."""
    return np.concatenate([np.broadcast_to(np.eye(3), x.shape[:-1] + (3, 3)), -skew(x)], -1)


def huber(r_norm, sigma, k=1.5):
    w = np.ones_like(r_norm)
    big = r_norm > k * sigma
    w[big] = k * sigma / r_norm[big]
    return w / sigma ** 2


def refine(frames, iters=12):
    n = len(frames)
    T = [f.c2w0.copy() for f in frames]
    A = [f.c2w0 for f in frames]
    pairs = overlap_pairs(frames, T)
    fmatch = {}
    for (i, j) in pairs:
        if i < j:
            fmatch[(i, j)] = match_features(frames[i], frames[j])
    print(f"{n} keyframes, {len(pairs)} overlapping ordered pairs, "
          f"{sum(len(m[0]) for m in fmatch.values())} XFeat matches before gating")
    V = np.array([[-0.5, -0.5, 1.5], [0.5, -0.5, 1.5], [-0.5, 0.5, 1.5], [0.5, 0.5, 1.5], [0, 0, 0.5]])
    for it in range(iters):
        gate = max(0.2 * 0.7 ** it, 0.03)
        H = np.zeros((6 * n, 6 * n))
        bvec = np.zeros(6 * n)
        stats = {"icp": [], "feat": [], "odo": []}
        Tinv = [np.linalg.inv(t) for t in T]
        for (i, j) in pairs:
            fi, fj = frames[i], frames[j]
            xw = apply(T[i], fi.src)
            u, v, ok = fj.project(apply(Tinv[j], xw))
            q = fj.P[v, u]
            nq = fj.N[v, u]
            ok &= np.isfinite(q).all(-1) & np.isfinite(nq).all(-1)
            qw = apply(T[j], q)
            nw = nq @ T[j][:3, :3].T
            ns = fi.src_n @ T[i][:3, :3].T
            d = np.linalg.norm(xw - qw, axis=-1)
            ok &= (d < gate) & (np.sum(nw * ns, -1) > 0.8)
            if ok.sum() < 50:
                continue
            x, qq, nn = xw[ok], qw[ok], nw[ok]
            r = np.sum(nn * (x - qq), -1)[:, None]
            Ji = np.concatenate([nn, np.cross(x, nn)], -1)[:, None, :]
            Jj = -np.concatenate([nn, np.cross(qq, nn)], -1)[:, None, :]
            # Each depth sample is not independent: share one pair's worth of weight among them.
            w = huber(np.abs(r[:, 0]), SIGMA_ICP) * min(1.0, 200.0 / len(r))
            accumulate(H, bvec, i, j, Ji, Jj, r, w)
            stats["icp"].append(np.abs(r[:, 0]))
        for (i, j), (a, b) in fmatch.items():
            xi = apply(T[i], frames[i].kp3[a])
            xj = apply(T[j], frames[j].kp3[b])
            r = xi - xj
            rn = np.linalg.norm(r, axis=-1)
            ok = rn < max(3 * gate, 0.1)
            if ok.sum() < 8:
                continue
            accumulate(H, bvec, i, j, point_jac(xi[ok]), -point_jac(xj[ok]), r[ok], huber(rn[ok], SIGMA_FEAT))
            stats["feat"].append(rn[ok])
        for k in range(n - 1):
            rel = np.linalg.inv(A[k + 1]) @ A[k]                       # ARKit: camera k → camera k+1
            xk = apply(T[k], V)
            xk1 = apply(T[k + 1], apply(rel, V))
            r = xk - xk1
            accumulate(H, bvec, k, k + 1, point_jac(xk), -point_jac(xk1), r,
                       np.full(len(V), 1 / SIGMA_ODO ** 2))
            stats["odo"].append(np.linalg.norm(r, axis=-1))
        Hs = H[6:, 6:] + 1e-6 * np.eye(6 * n - 6) + 1e-3 * np.diag(np.diag(H[6:, 6:]))
        dx = np.linalg.solve(Hs, -bvec[6:])
        for k in range(1, n):
            t, w = dx[6 * (k - 1):6 * (k - 1) + 3], dx[6 * (k - 1) + 3:6 * k]
            R = rodrigues(w)
            T[k][:3, :3] = R @ T[k][:3, :3]
            T[k][:3, 3] = R @ T[k][:3, 3] + t
        med = {k: 1000 * float(np.median(np.concatenate(v))) if v else float("nan") for k, v in stats.items()}
        step = np.abs(dx.reshape(-1, 6)[:, :3]).max() * 1000
        print(f"iter {it:2d}  gate {gate * 100:4.1f} cm  median |r| mm: lidar {med['icp']:5.1f}  xfeat {med['feat']:5.1f}"
              f"  odometry {med['odo']:5.1f}  | {sum(len(v) for v in stats['feat'])} matches  max step {step:5.1f} mm")
    return T


def main():
    root, feats_dir, out = sys.argv[1:4]
    man = json.load(open(os.path.join(root, "session.json")))
    frames = [Frame(root, feats_dir, kf, man) for kf in man["keyframes"]]
    T = refine(frames)
    moved = [np.linalg.norm(T[k][:3, 3] - frames[k].c2w0[:3, 3]) * 1000 for k in range(len(T))]
    print(f"camera centres moved: median {np.median(moved):.1f} mm, max {np.max(moved):.1f} mm")
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
