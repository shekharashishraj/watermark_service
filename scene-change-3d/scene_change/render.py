"""EWA splatting renderer for 3D Gaussians (NumPy, CPU).

Implements the 3D Gaussian Splatting forward model: project each Gaussian's
covariance to the image (J W Sigma W^T J^T + 0.3 I), evaluate its 2D footprint,
sort contributions per pixel by depth and alpha-composite front to back. It is
slow compared with a CUDA rasteriser but exact enough for thumbnails, depth for
coverage bookkeeping, and tests.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .gaussians import GaussianMap
from .geometry import invert_pose

SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
         1.445305721320277, -0.5900435899266435)


def eval_sh_colors(gm: GaussianMap, cam_center: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """View-dependent colour (DC + optional higher bands) for the Gaussians in ``idx``."""
    base = gm.colors[idx].astype(np.float64)
    if gm.sh_rest is None:
        return base
    rest = gm.sh_rest[idx]
    k = rest.shape[1]
    d = gm.means[idx].astype(np.float64) - cam_center[None, :]
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    x, y, z = d[:, 0:1], d[:, 1:2], d[:, 2:3]
    res = base - 0.5  # = SH_C0 * dc
    if k >= 3:
        res = res - SH_C1 * y * rest[:, 0] + SH_C1 * z * rest[:, 1] - SH_C1 * x * rest[:, 2]
    if k >= 8:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        res = (res + SH_C2[0] * xy * rest[:, 3] + SH_C2[1] * yz * rest[:, 4] + SH_C2[2] * (2 * zz - xx - yy) * rest[:, 5]
               + SH_C2[3] * xz * rest[:, 6] + SH_C2[4] * (xx - yy) * rest[:, 7])
        if k >= 15:
            res = (res + SH_C3[0] * y * (3 * xx - yy) * rest[:, 8] + SH_C3[1] * xy * z * rest[:, 9]
                   + SH_C3[2] * y * (4 * zz - xx - yy) * rest[:, 10] + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * rest[:, 11]
                   + SH_C3[4] * x * (4 * zz - xx - yy) * rest[:, 12] + SH_C3[5] * z * (xx - yy) * rest[:, 13]
                   + SH_C3[6] * x * (xx - 3 * yy) * rest[:, 14])
    return np.clip(res + 0.5, 0, 1)


def render_gaussians(gm: GaussianMap, K: np.ndarray, T_wc: np.ndarray, width: int, height: int, near: float = 0.08,
                     far: float = 40.0, max_radius: int = 48, colors: np.ndarray | None = None, return_ids=False,
                     return_weights=False):
    """Render RGB, expected depth, median depth, accumulated alpha (and dominant Gaussian ids).

    ``colors`` overrides per-Gaussian colours (e.g. to paint a change mask).
    ``return_weights`` adds ``weights``: a sparse (H*W, len(gm)) matrix of blending
    weights alpha_i * T_i, so that any per-Gaussian quantity q renders as ``weights @ q``.
    Like the CUDA rasteriser, blending stops once transmittance would fall below 1e-4.
    """
    T_cw = invert_pose(T_wc)
    Rcw, tcw = T_cw[:3, :3], T_cw[:3, 3]
    mc = gm.means.astype(np.float64) @ Rcw.T + tcw
    z = mc[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    smax = gm.scales.max(axis=1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * mc[:, 0] / z + cx
        v = fy * mc[:, 1] / z + cy
        rad_guess = 3 * smax * fx / z
    vis = (z > near) & (z < far) & (u + rad_guess > -1) & (u - rad_guess < width) & (v + rad_guess > -1) & \
          (v - rad_guess < height)
    idx = np.nonzero(vis)[0]
    H, W = height, width
    out_rgb = np.zeros((H * W, 3))
    out_alpha = np.zeros(H * W)
    out_depth = np.zeros(H * W)
    out_med = np.zeros(H * W)
    out_ids = np.full(H * W, -1, dtype=np.int64)
    if len(idx) == 0:
        res = {"rgb": out_rgb.reshape(H, W, 3), "alpha": out_alpha.reshape(H, W), "depth": out_depth.reshape(H, W),
               "median_depth": out_med.reshape(H, W)}
        if return_ids:
            res["ids"] = out_ids.reshape(H, W)
        if return_weights:
            res["weights"] = sparse.csr_matrix((H * W, len(gm)), dtype=np.float32)
        return res
    mcv, zv, uv, vv = mc[idx], z[idx], u[idx], v[idx]
    cov = gm.covariances()[idx]
    cov_c = Rcw[None] @ cov @ Rcw.T[None]
    J = np.zeros((len(idx), 2, 3))
    J[:, 0, 0] = fx / zv
    J[:, 0, 2] = -fx * mcv[:, 0] / zv ** 2
    J[:, 1, 1] = fy / zv
    J[:, 1, 2] = -fy * mcv[:, 1] / zv ** 2
    S2 = J @ cov_c @ np.swapaxes(J, 1, 2)
    a = S2[:, 0, 0] + 0.3
    b = S2[:, 0, 1]
    c = S2[:, 1, 1] + 0.3
    det = a * c - b * b
    ok = det > 1e-12
    inv_a, inv_b, inv_c = c / det, -b / det, a / det
    mid = 0.5 * (a + c)
    lam = mid + np.sqrt(np.maximum(0.1, mid * mid - det))
    radius = np.ceil(3.0 * np.sqrt(lam)).astype(np.int64)
    ok &= radius <= max_radius
    if colors is None:
        col = eval_sh_colors(gm, T_wc[:3, 3], idx)
    else:
        col = np.asarray(colors, dtype=np.float64)[idx]
    op = gm.opacity[idx].astype(np.float64)

    pix_list, alpha_list, z_list, g_list = [], [], [], []
    pu, pv = np.round(uv).astype(np.int64), np.round(vv).astype(np.int64)
    for r in np.unique(radius[ok]):
        sel = np.nonzero(ok & (radius == r))[0]
        oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
        ox, oy = ox.ravel(), oy.ravel()
        # chunk to bound memory
        step = max(1, 400_000 // len(ox))
        for s0 in range(0, len(sel), step):
            ss = sel[s0:s0 + step]
            px = pu[ss, None] + ox[None, :]
            py = pv[ss, None] + oy[None, :]
            dx = px - uv[ss, None]
            dy = py - vv[ss, None]
            power = -0.5 * (inv_a[ss, None] * dx * dx + 2 * inv_b[ss, None] * dx * dy + inv_c[ss, None] * dy * dy)
            alpha = np.minimum(0.99, op[ss, None] * np.exp(np.minimum(power, 0.0)))
            m = (alpha >= 1.0 / 255.0) & (px >= 0) & (px < W) & (py >= 0) & (py < H) & (power <= 0)
            if not m.any():
                continue
            pix_list.append((py * W + px)[m])
            alpha_list.append(alpha[m])
            z_list.append(np.broadcast_to(zv[ss, None], m.shape)[m])
            g_list.append(np.broadcast_to(ss[:, None], m.shape)[m])
    if not pix_list:
        res = {"rgb": out_rgb.reshape(H, W, 3), "alpha": out_alpha.reshape(H, W), "depth": out_depth.reshape(H, W),
               "median_depth": out_med.reshape(H, W)}
        if return_ids:
            res["ids"] = out_ids.reshape(H, W)
        if return_weights:
            res["weights"] = sparse.csr_matrix((H * W, len(gm)), dtype=np.float32)
        return res
    pix = np.concatenate(pix_list)
    alpha = np.concatenate(alpha_list)
    zz = np.concatenate(z_list)
    gg = np.concatenate(g_list)
    zq = np.clip(zz * 1e4, 0, 2 ** 31 - 1).astype(np.int64)
    order = np.argsort((pix << 32) | zq, kind="stable")
    pix, alpha, zz, gg = pix[order], alpha[order], zz[order], gg[order]
    starts = np.r_[True, pix[1:] != pix[:-1]]
    seg = np.cumsum(starts) - 1
    sidx = np.nonzero(starts)[0]
    L = np.log1p(-alpha)
    cs = np.cumsum(L)
    base = cs[sidx] - L[sidx]
    T = np.exp(cs - L - base[seg])
    w = alpha * T
    upix = pix[sidx]
    wsum = np.bincount(seg, weights=w)
    out_alpha[upix] = wsum
    for k in range(3):
        out_rgb[upix, k] = np.bincount(seg, weights=w * col[gg, k])
    zsum = np.bincount(seg, weights=w * zz)
    out_depth[upix] = zsum / np.maximum(wsum, 1e-9)
    after = T * (1 - alpha)
    cross = (T >= 0.5) & (after < 0.5)
    out_med[pix[cross]] = zz[cross]
    if return_ids:
        wmax = np.maximum.reduceat(w, sidx)
        is_max = w >= wmax[seg] - 1e-12
        first = np.nonzero(is_max)[0]
        # keep the first max per segment
        fseg = seg[first]
        keep = np.r_[True, fseg[1:] != fseg[:-1]]
        out_ids[pix[first[keep]]] = idx[gg[first[keep]]]
    res = {"rgb": out_rgb.reshape(H, W, 3), "alpha": out_alpha.reshape(H, W), "depth": out_depth.reshape(H, W),
           "median_depth": out_med.reshape(H, W)}
    if return_ids:
        res["ids"] = out_ids.reshape(H, W)
    if return_weights:
        keep = after >= 1e-4
        res["weights"] = sparse.csr_matrix((w[keep].astype(np.float32), (pix[keep], idx[gg[keep]])),
                                           shape=(H * W, len(gm)))
    return res
