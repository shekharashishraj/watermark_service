"""Change detection: compare the localized inspection against the 3D baseline.

Two channels, both evaluated in 3D and accumulated over every inspection view:

Geometric channel
    * Each baseline Gaussian is projected into every inspection frame. If the
      observed depth lands on it, it is confirmed. If the ray passes clearly
      *through* it (free-space violation), that is evidence it is missing. If
      something is in front of it, the view says nothing.
    * Each inspection surface point that is far from every baseline surface is
      checked against the baseline keyframes: if the baseline saw that space as
      empty, it is evidence of an added object; if the baseline never observed
      the space, it is simply a newly seen area, not a change.

Appearance channel
    * For confirmed Gaussians, the observed colour (averaged over the Gaussian's
      footprint) is compared with the baseline colour after a robust per-frame
      colour normalisation that absorbs exposure and white-balance shifts.

Evidence is clustered into change instances, missing/added pairs are matched as
moved objects, and each change gets a score. Low scores and cases where the
channels disagree are flagged for human review. Baseline surfaces that no
inspection view observed are reported as unverified.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .geometry import depth_edge_mask, invert_pose, voxel_keys, voxel_centers
from .model import BaselineModel, depth_tolerance, free_space_votes, min_pool_depth
from .register import Registration
from .session import Session

STATUS = {"verified": 0, "missing": 1, "appearance": 2, "unverified": 3, "excluded": 4}


@dataclass
class DetectConfig:
    max_range: float = 4.5
    min_range: float = 0.25
    min_cos: float = 0.2
    added_stride: int = 2
    tau_add: float = 0.07
    viol_min_views: int = 2
    viol_ratio: float = 0.6
    app_lum: float = 0.45          # |log luminance ratio| that counts as one unit of appearance difference
    app_chroma: float = 0.07       # chromaticity L1 distance that counts as one unit
    app_min_views: int = 2
    app_ratio: float = 0.6
    app_coherence: float = 0.6     # signed colour shift must agree across views
    added_min_frames: int = 2
    cluster_link: float = 0.09
    min_missing: int = 10
    min_added: int = 8
    min_appearance: int = 8
    min_unverified: int = 60
    review_score: float = 0.55
    disagree_app: float = 0.2
    moved_max_dist: float = 12.0
    coverage_min_room: float = 0.6
    exclude_ceiling: bool = True


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def appearance_score(c_obs: np.ndarray, c_pred: np.ndarray, cfg: DetectConfig, sig_l=0.0, sig_c=0.0) -> np.ndarray:
    """Appearance difference in units of the (texture-adapted) tolerance; > 1 counts as different."""
    c_pred = np.clip(c_pred, 0.0, None)
    lo, lp = c_obs.mean(1), c_pred.mean(1)
    dl = np.abs(np.log((lo + 0.06) / (lp + 0.06)))
    co = c_obs / (c_obs.sum(1, keepdims=True) + 0.12)
    cp = c_pred / (c_pred.sum(1, keepdims=True) + 0.12)
    dc = np.abs(co - cp).sum(1)
    return np.maximum(dl / (cfg.app_lum + 1.5 * sig_l), dc / (cfg.app_chroma + 1.0 * sig_c))


def fit_color_affine(c_ref: np.ndarray, c_obs: np.ndarray, iters: int = 3):
    """Robust per-channel c_obs ~ a * c_ref + b."""
    a, b = np.ones(3), np.zeros(3)
    if len(c_ref) < 40:
        return a, b, False
    w = np.ones(len(c_ref))
    for _ in range(iters):
        for ch in range(3):
            x, y = c_ref[:, ch], c_obs[:, ch]
            sw, sx, sy = w.sum(), (w * x).sum(), (w * y).sum()
            sxx, sxy = (w * x * x).sum(), (w * x * y).sum()
            den = sw * sxx - sx * sx
            if den <= 1e-9:
                continue
            a[ch] = (sw * sxy - sx * sy) / den
            b[ch] = (sy - a[ch] * sx) / sw
        a = np.clip(a, 0.3, 3.0)
        b = np.clip(b, -0.3, 0.3)
        r = np.linalg.norm(c_obs - (c_ref * a + b), axis=1)
        s = 1.4826 * np.median(r) + 1e-3
        w = 1.0 / (1.0 + (r / (2.5 * s)) ** 2)
    return a, b, True


def neighborhood_stats(gm):
    """Per Gaussian: luminance and chromaticity variability of its same-surface voxel neighbourhood."""
    from .geometry import ijk_to_keys, keys_to_ijk
    keys = voxel_keys(gm.means, gm.voxel)
    order = np.argsort(keys)
    sk = keys[order]
    ijk = keys_to_ijk(keys)
    col = gm.colors.astype(np.float64)
    nrm = gm.normals.astype(np.float64)
    L = col.mean(1)
    ch = col / (col.sum(1, keepdims=True) + 0.06)
    sL, sL2, sC, cnt = np.zeros(len(col)), np.zeros(len(col)), np.zeros((len(col), 3)), np.zeros(len(col))
    sC2 = np.zeros(len(col))
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                q = ijk_to_keys(ijk + np.array([di, dj, dk]))
                pos = np.clip(np.searchsorted(sk, q), 0, len(sk) - 1)
                hit = sk[pos] == q
                j = order[pos]
                hit &= np.sum(nrm * nrm[j], axis=1) > 0.7
                w = hit.astype(np.float64)
                sL += w * L[j]
                sL2 += w * L[j] ** 2
                sC += w[:, None] * ch[j]
                sC2 += w * np.sum(ch[j] ** 2, axis=1)
                cnt += w
    cnt = np.maximum(cnt, 1)
    mL = sL / cnt
    varL = np.maximum(sL2 / cnt - mL ** 2, 0)
    mC = sC / cnt[:, None]
    varC = np.maximum(sC2 / cnt - np.sum(mC ** 2, axis=1), 0)
    sig_l = np.sqrt(varL) / (mL + 0.05)
    sig_c = np.sqrt(varC) * 1.7
    return sig_l, sig_c


def _integral(img):
    S = img.astype(np.float64).cumsum(0).cumsum(1)
    return np.pad(S, ((1, 0), (1, 0), (0, 0)))


def _box_mean(S, u, v, r, W, H):
    u0, u1 = np.clip(u - r, 0, W - 1), np.clip(u + r, 0, W - 1)
    v0, v1 = np.clip(v - r, 0, H - 1), np.clip(v + r, 0, H - 1)
    tot = S[v1 + 1, u1 + 1] - S[v0, u1 + 1] - S[v1 + 1, u0] + S[v0, u0]
    area = ((u1 - u0 + 1) * (v1 - v0 + 1)).astype(np.float64)
    return tot / area[:, None]


def cluster_points(P: np.ndarray, link: float):
    """Connected components of points within ``link`` of each other."""
    n = len(P)
    if n == 0:
        return np.zeros(0, dtype=np.int64), 0
    tree = cKDTree(P)
    pairs = tree.query_pairs(link, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(n), n
    A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    k, lab = connected_components(A, directed=False)
    return lab, k


def _bbox(P):
    return P.min(0), P.max(0)


# ----------------------------------------------------------------------------
# Evidence accumulation
# ----------------------------------------------------------------------------

@dataclass
class Evidence:
    n_conf: np.ndarray
    n_viol: np.ndarray
    n_view: np.ndarray
    n_app: np.ndarray
    n_app_hi: np.ndarray
    app_sum: np.ndarray
    n_viol_app_hi: np.ndarray
    app_dsum: np.ndarray     # sum over views of the signed, normalised colour difference
    app_dabs: np.ndarray     # sum over views of its magnitude
    add_P: np.ndarray        # candidate new-surface points (baseline frame)
    add_C: np.ndarray
    add_F: np.ndarray
    color_fits: list
    seconds: float


def accumulate_evidence(model: BaselineModel, session: Session, poses: np.ndarray, cfg: DetectConfig,
                        tree: cKDTree | None = None) -> Evidence:
    t0 = time.time()
    g = model.gaussians
    mu = g.means.astype(np.float64)
    nrm = g.normals.astype(np.float64)
    colg = g.colors.astype(np.float64)
    N = len(mu)
    n_conf = np.zeros(N, np.int32)
    n_viol = np.zeros(N, np.int32)
    n_view = np.zeros(N, np.int32)
    n_app = np.zeros(N, np.int32)
    n_app_hi = np.zeros(N, np.int32)
    app_sum = np.zeros(N)
    n_viol_app_hi = np.zeros(N, np.int32)
    app_dsum = np.zeros((N, 3))
    app_dabs = np.zeros(N)
    sig_l, sig_c = neighborhood_stats(g)
    tree = tree or cKDTree(mu)
    K = session.K
    H, W = session.height, session.width
    fx = K[0, 0]
    add_P, add_C, add_F, fits = [], [], [], []
    offs = [(dv, du) for dv in (-1, 0, 1) for du in (-1, 0, 1)]
    for i in range(len(session)):
        T_wc = poses[i]
        T_cw = invert_pose(T_wc)
        cam = T_wc[:3, 3]
        depth = session.depth[i].astype(np.float64)
        rgb = session.rgb[i].astype(np.float64) / 255.0
        pc = mu @ T_cw[:3, :3].T + T_cw[:3, 3]
        z = pc[:, 2]
        cand = (z > cfg.min_range) & (z < cfg.max_range)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.round(fx * pc[:, 0] / z + K[0, 2])
            v = np.round(K[1, 1] * pc[:, 1] / z + K[1, 2])
        cand &= (u >= 1) & (u < W - 1) & (v >= 1) & (v < H - 1)
        idx = np.nonzero(cand)[0]
        if len(idx) == 0:
            continue
        vd = cam[None, :] - mu[idx]
        dist = np.linalg.norm(vd, axis=1)
        cosv = np.sum(nrm[idx] * vd, axis=1) / np.maximum(dist, 1e-9)
        front = cosv > cfg.min_cos
        idx = idx[front]
        if len(idx) == 0:
            continue
        ui, vi, zi = u[idx].astype(np.int64), v[idx].astype(np.int64), z[idx]
        edge = depth_edge_mask(session.depth[i], 0.06)
        win = np.stack([depth[vi + dv, ui + du] for dv, du in offs], axis=1)
        wedge = np.stack([edge[vi + dv, ui + du] for dv, du in offs], axis=1)
        valid = win > 0
        tau = depth_tolerance(zi)
        diff = np.where(valid, np.abs(win - zi[:, None]), np.inf)
        confirm = diff.min(1) <= tau
        solid = valid & ~wedge                      # depth-edge pixels are unreliable evidence of free space
        closest = np.where(valid, win, np.inf).min(1)
        viol = (~confirm) & (solid.sum(1) >= 6) & (closest > zi + tau)
        center = win[:, 4]
        seen = confirm | viol
        n_view[idx[seen | ((center > 0) & (center < zi - tau))]] += 1
        n_conf[idx[confirm]] += 1
        n_viol[idx[viol]] += 1
        # appearance: footprint-averaged observed colour vs normalised baseline colour
        S = _integral(rgb)
        r = np.clip(np.round(0.55 * g.voxel * fx / zi), 0, 4).astype(np.int64)
        near_edge = wedge.any(1)
        ci = np.nonzero(confirm & ~near_edge)[0]
        vi_ = np.nonzero(viol)[0]
        if len(ci) >= 40:
            c_obs = _box_mean(S, ui[ci], vi[ci], r[ci], W, H)
            a, b, ok = fit_color_affine(colg[idx[ci]], c_obs)
            fits.append((i, a.tolist(), b.tolist()))
            if ok:
                gi = idx[ci]
                pred = np.clip(colg[gi] * a + b, 0.0, None)
                sc = appearance_score(c_obs, pred, cfg, sig_l[gi], sig_c[gi])
                n_app[gi] += 1
                app_sum[gi] += np.minimum(sc, 5.0)
                n_app_hi[gi] += sc > 1.0
                dr = (c_obs - pred) / (pred.mean(1, keepdims=True) + 0.05)
                np.add.at(app_dsum, gi, dr)
                app_dabs[gi] += np.linalg.norm(dr, axis=1)
                if len(vi_):
                    c_v = _box_mean(S, ui[vi_], vi[vi_], r[vi_], W, H)
                    gv = idx[vi_]
                    sv = appearance_score(c_v, np.clip(colg[gv] * a + b, 0, None), cfg, sig_l[gv], sig_c[gv])
                    n_viol_app_hi[gv] += sv > 1.0
        # new-surface candidates
        m = (session.depth[i] > cfg.min_range) & (session.depth[i] < cfg.max_range) & ~depth_edge_mask(session.depth[i], 0.06)
        vs, us = np.mgrid[0:H:cfg.added_stride, 0:W:cfg.added_stride]
        sel = m[vs, us]
        us, vs = us[sel], vs[sel]
        d = session.depth[i][vs, us].astype(np.float64)
        Pc = np.stack([(us - K[0, 2]) / fx * d, (vs - K[1, 2]) / K[1, 1] * d, d], 1)
        Pw = Pc @ T_wc[:3, :3].T + T_wc[:3, 3]
        dist_s, _ = tree.query(Pw, distance_upper_bound=0.5)
        far = dist_s > np.maximum(cfg.tau_add, depth_tolerance(d))
        if far.any():
            add_P.append(Pw[far])
            add_C.append(rgb[vs[far], us[far]])
            add_F.append(np.full(int(far.sum()), i, np.int32))
    cat = (lambda L, shape: np.concatenate(L) if L else np.zeros(shape))
    return Evidence(n_conf, n_viol, n_view, n_app, n_app_hi, app_sum, n_viol_app_hi, app_dsum, app_dabs,
                    cat(add_P, (0, 3)), cat(add_C, (0, 3)), cat(add_F, (0,)).astype(np.int32), fits,
                    time.time() - t0)


# ----------------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------------

@dataclass
class Change:
    id: int
    type: str                  # missing | added | moved | appearance
    room: str
    surface: str
    centroid: list
    bbox_min: list
    bbox_max: list
    area_m2: float
    score: float
    geometry_score: float
    appearance_score: float
    review: bool
    reasons: list
    n_views: int
    best_frames: list = field(default_factory=list)
    moved_to: list | None = None
    moved_distance: float | None = None
    points: np.ndarray | None = field(default=None, repr=False)        # 3D support (for projection/viewer)
    points_to: np.ndarray | None = field(default=None, repr=False)     # moved: destination support

    def summary(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ("points", "points_to")}
        return d


@dataclass
class InspectionResult:
    changes: list
    unverified: list
    new_areas: list
    rooms: list
    verdict: str
    verdict_reasons: list
    status: np.ndarray               # per baseline Gaussian (STATUS codes)
    added_points: np.ndarray         # voxel centres of confirmed added evidence
    added_labels: np.ndarray         # change id per added voxel (-1 = unclustered)
    new_area_points: np.ndarray
    registration: dict
    timings: dict
    config: dict
    poses: np.ndarray                # inspection camera poses in the baseline frame

    def summary(self) -> dict:
        return {
            "verdict": self.verdict,
            "verdict_reasons": self.verdict_reasons,
            "counts": {
                "confirmed": sum(1 for c in self.changes if not c.review),
                "needs_review": sum(1 for c in self.changes if c.review),
                "by_type": {t: sum(1 for c in self.changes if c.type == t) for t in ("missing", "added", "moved", "appearance")},
                "unverified_regions": len(self.unverified),
            },
            "rooms": self.rooms,
            "changes": [c.summary() for c in self.changes],
            "unverified": self.unverified,
            "new_areas": self.new_areas,
            "registration": self.registration,
            "timings": self.timings,
        }


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------

def _surface_labels(model: BaselineModel) -> np.ndarray:
    """floor / wall / ceiling / object label per baseline Gaussian."""
    g = model.gaussians
    P, Nn = g.means.astype(np.float64), g.normals.astype(np.float64)
    lab = np.full(len(P), "object", dtype=object)
    lab[(Nn[:, 2] > 0.85) & (np.abs(P[:, 2] - model.floor_z) < 0.05)] = "floor"
    lab[model.ceiling_mask()] = "ceiling"
    vertical = np.abs(Nn[:, 2]) < 0.25
    if model.rooms:
        near = np.zeros(len(P), dtype=bool)
        for r in model.rooms:
            poly = np.asarray(r["polygon"], dtype=np.float64)
            for k in range(len(poly)):
                a, b = poly[k], poly[(k + 1) % len(poly)]
                ab = b - a
                t = np.clip(((P[:, :2] - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
                d = np.linalg.norm(P[:, :2] - (a + t[:, None] * ab), axis=1)
                near |= d < 0.16
        lab[vertical & near] = "wall"
    else:
        lab[vertical & (P[:, 2] > model.floor_z + 2.0)] = "wall"
    return lab


def detect_changes(model: BaselineModel, session: Session, reg: Registration, cfg: DetectConfig | None = None,
                   verbose: bool = False) -> InspectionResult:
    cfg = cfg or DetectConfig()
    t_all = time.time()
    g = model.gaussians
    mu = g.means.astype(np.float64)
    tree = cKDTree(mu)
    ev = accumulate_evidence(model, session, reg.poses, cfg, tree)
    timings = {"evidence_s": round(ev.seconds, 2)}

    # ---------------- missing / appearance / unverified per Gaussian ----------
    t0 = time.time()
    tot = ev.n_conf + ev.n_viol
    p_viol = np.where(tot > 0, ev.n_viol / np.maximum(tot, 1), 0.0)
    missing = (ev.n_viol >= cfg.viol_min_views) & (p_viol >= cfg.viol_ratio)
    app_ratio = np.where(ev.n_app > 0, ev.n_app_hi / np.maximum(ev.n_app, 1), 0.0)
    coherence = np.linalg.norm(ev.app_dsum, axis=1) / np.maximum(ev.app_dabs, 1e-9)
    appear = ((~missing) & (ev.n_app_hi >= cfg.app_min_views) & (app_ratio >= cfg.app_ratio)
              & (coherence >= cfg.app_coherence))
    excluded = model.ceiling_mask() if cfg.exclude_ceiling else np.zeros(len(mu), bool)
    unverified = (tot == 0) & ~excluded
    status = np.zeros(len(mu), np.int8)
    status[unverified] = STATUS["unverified"]
    status[excluded & (tot == 0)] = STATUS["excluded"]
    status[appear] = STATUS["appearance"]
    status[missing] = STATUS["missing"]
    surface = _surface_labels(model)

    # ---------------- added evidence --------------------------------------
    added_vox = np.zeros((0, 3))
    added_col = np.zeros((0, 3))
    added_nfr = np.zeros(0, np.int32)
    new_vox = np.zeros((0, 3))
    if len(ev.add_P):
        nf_b, ns_b = model.free_space_votes(ev.add_P, max_range=cfg.max_range + 0.5)
        is_add = (nf_b >= 2) & (nf_b >= 2 * ns_b)
        is_new = (nf_b + ns_b) == 0
        vox = g.voxel
        if is_add.any():
            P, C, F = ev.add_P[is_add], ev.add_C[is_add], ev.add_F[is_add]
            keys = voxel_keys(P, vox)
            uk, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
            fr = np.unique(np.stack([inv, F], 1), axis=0)
            nfr = np.bincount(fr[:, 0], minlength=len(uk))
            pm = np.stack([np.bincount(inv, P[:, k]) for k in range(3)], 1) / cnt[:, None]
            cm = np.stack([np.bincount(inv, C[:, k]) for k in range(3)], 1) / cnt[:, None]
            keep = (nfr >= cfg.added_min_frames) & (cnt >= 3)
            pm, cm, nfr = pm[keep], cm[keep], nfr[keep]
            if len(pm):
                # consistency inside the inspection itself: surface seen at least as often as seen-through
                idepth = np.stack([min_pool_depth(d, 2) for d in session.depth])
                Kh = session.K.copy()
                Kh[0, 0] /= 2
                Kh[1, 1] /= 2
                Kh[0, 2] = (Kh[0, 2] - 0.5) / 2
                Kh[1, 2] = (Kh[1, 2] - 0.5) / 2
                nf_i, ns_i = free_space_votes(pm, reg.poses, idepth, Kh, cfg.max_range)
                ok = (ns_i >= 2) & (ns_i >= nf_i)
                added_vox, added_col, added_nfr = pm[ok], cm[ok], nfr[ok]
        if is_new.any():
            kn = np.unique(voxel_keys(ev.add_P[is_new], 0.1))
            new_vox = voxel_centers(kn, 0.1)
    timings["classify_s"] = round(time.time() - t0, 2)

    # ---------------- clustering -------------------------------------------
    t0 = time.time()
    changes = []
    room_names = model.room_of

    def make_change(kind, pts, members=None, n_views=0, geo=0.0, app=0.0, surf="object"):
        lo, hi = _bbox(pts)
        cen = pts.mean(0)
        room = str(room_names(cen[None])[0])
        return Change(len(changes), kind, room, surf, cen.round(3).tolist(), lo.round(3).tolist(), hi.round(3).tolist(),
                      round(float(len(pts) * g.voxel ** 2), 3), 0.0, round(float(geo), 3), round(float(app), 3), False, [],
                      int(n_views), points=pts)

    def majority(labels):
        if len(labels) == 0:
            return "object"
        vals, cnt = np.unique(labels, return_counts=True)
        return str(vals[np.argmax(cnt)])

    miss_idx = np.nonzero(missing)[0]
    lab, k = cluster_points(mu[miss_idx], cfg.cluster_link)
    miss_clusters = []
    for c in range(k):
        mem = miss_idx[lab == c]
        if len(mem) < cfg.min_missing:
            continue
        geo = float(np.mean(p_viol[mem]) * min(1.0, np.mean(ev.n_viol[mem]) / 3.0))
        app = float(np.mean(ev.n_viol_app_hi[mem] / np.maximum(ev.n_viol[mem], 1)))
        ch = make_change("missing", mu[mem], n_views=int(np.median(ev.n_viol[mem])), geo=geo, app=app,
                         surf=majority(surface[mem]))
        ch._members = mem
        miss_clusters.append(ch)

    add_clusters = []
    added_labels = np.full(len(added_vox), -1)
    if len(added_vox):
        lab, k = cluster_points(added_vox, cfg.cluster_link * 1.3)
        for c in range(k):
            mem = np.nonzero(lab == c)[0]
            if len(mem) < cfg.min_added:
                continue
            geo = float(min(1.0, np.mean(added_nfr[mem]) / 4.0))
            ch = make_change("added", added_vox[mem], n_views=int(np.median(added_nfr[mem])), geo=geo, app=0.0)
            # surface under/behind the new object
            _, j = tree.query(added_vox[mem], k=1)
            ch.surface = "object"
            ch._members = mem
            ch._color = added_col[mem].mean(0)
            add_clusters.append(ch)

    app_idx = np.nonzero(appear)[0]
    lab, k = cluster_points(mu[app_idx], cfg.cluster_link)
    app_clusters = []
    for c in range(k):
        mem = app_idx[lab == c]
        if len(mem) < cfg.min_appearance:
            continue
        app = float(np.mean(app_ratio[mem]) * min(1.0, np.mean(ev.n_app_hi[mem]) / 3.0))
        ch = make_change("appearance", mu[mem], n_views=int(np.median(ev.n_app[mem])), geo=0.0, app=app,
                         surf=majority(surface[mem]))
        ch._members = mem
        app_clusters.append(ch)

    # ---------------- moved: match missing <-> added ------------------------
    moved = []
    if miss_clusters and add_clusters:
        cost = np.full((len(miss_clusters), len(add_clusters)), 1e6)
        for a, m in enumerate(miss_clusters):
            mcol = g.colors[m._members].mean(0)
            mext = np.sort(np.asarray(m.bbox_max) - np.asarray(m.bbox_min))
            for b, ad in enumerate(add_clusters):
                dist = np.linalg.norm(np.asarray(m.centroid) - np.asarray(ad.centroid))
                if dist > cfg.moved_max_dist or dist < 0.3:
                    continue
                ratio = max(m.area_m2, ad.area_m2) / max(1e-6, min(m.area_m2, ad.area_m2))
                aext = np.sort(np.asarray(ad.bbox_max) - np.asarray(ad.bbox_min))
                chroma_m = mcol / (mcol.sum() + 0.06)
                chroma_a = ad._color / (ad._color.sum() + 0.06)
                dchroma = float(np.abs(chroma_m - chroma_a).sum())
                dh = abs(float(m.bbox_max[2] - ad.bbox_max[2]))
                if ratio > 3.0 or dchroma > 0.12 or dh > 0.35:
                    continue
                cost[a, b] = np.log(ratio) + dchroma / 0.06 + np.abs(mext - aext).sum() / 0.4 + dh / 0.15 + 0.02 * dist
        ri, ci = linear_sum_assignment(cost)
        used_m, used_a = set(), set()
        for a, b in zip(ri, ci):
            if cost[a, b] >= 1e5:
                continue
            m, ad = miss_clusters[a], add_clusters[b]
            ch = make_change("moved", m.points, n_views=min(m.n_views, ad.n_views),
                             geo=min(m.geometry_score, ad.geometry_score), app=max(m.appearance_score, 0.0),
                             surf="object")
            ch.moved_to = list(ad.centroid)
            ch.moved_distance = round(float(np.linalg.norm(np.asarray(m.centroid) - np.asarray(ad.centroid))), 2)
            ch.points_to = ad.points
            ch._members = m._members
            ch._to_members = ad._members
            moved.append(ch)
            used_m.add(a)
            used_a.add(b)
        miss_clusters = [m for a, m in enumerate(miss_clusters) if a not in used_m]
        add_clusters = [c for b, c in enumerate(add_clusters) if b not in used_a]

    # ---------------- scoring & review flags ----------------------------------
    final = []
    for ch in miss_clusters + add_clusters + moved + app_clusters:
        reasons = []
        if ch.type == "appearance":
            ch.score = round(ch.appearance_score, 3)
            reasons.append("appearance changed but geometry did not (channels disagree)")
            review = True
        else:
            ch.score = round(0.75 * ch.geometry_score + 0.25 * max(ch.appearance_score, 0.5 if ch.type == "added" else 0.0), 3)
            review = False
            if ch.type in ("missing", "moved") and ch.appearance_score < cfg.disagree_app:
                reasons.append("geometry changed but colour looks unchanged (channels disagree)")
                review = True
        if ch.score < cfg.review_score:
            reasons.append(f"change score {ch.score:.2f} below {cfg.review_score:.2f}")
            review = True
        ch.review = review
        ch.reasons = reasons
        final.append(ch)
    order = {"missing": 0, "moved": 1, "added": 2, "appearance": 3}
    final.sort(key=lambda c: (c.review, order[c.type], -c.score))
    for k, ch in enumerate(final):
        ch.id = k
    # per-voxel labels for the viewer
    for ch in final:
        if ch.type == "added":
            added_labels[ch._members] = ch.id
        elif ch.type == "moved":
            added_labels[ch._to_members] = ch.id

    # ---------------- unverified regions & coverage -------------------------
    unv_idx = np.nonzero(unverified)[0]
    lab, k = cluster_points(mu[unv_idx], g.voxel * 1.8)
    unv_regions = []
    for c in range(k):
        mem = unv_idx[lab == c]
        if len(mem) < cfg.min_unverified:
            continue
        lo, hi = _bbox(mu[mem])
        cen = mu[mem].mean(0)
        unv_regions.append({"id": len(unv_regions), "room": str(room_names(cen[None])[0]),
                            "centroid": cen.round(3).tolist(), "bbox_min": lo.round(3).tolist(),
                            "bbox_max": hi.round(3).tolist(), "area_m2": round(float(len(mem) * g.voxel ** 2), 2),
                            "surface": majority(surface[mem])})
    unv_regions.sort(key=lambda r: -r["area_m2"])
    for k2, r in enumerate(unv_regions):
        r["id"] = k2

    rooms_out = []
    gro = room_names(mu)
    for rn in [r["name"] for r in model.rooms] or ["property"]:
        m = (gro == rn) & ~excluded if model.rooms else ~excluded
        n = int(m.sum())
        if n == 0:
            continue
        cov = float(1.0 - (unverified & m).sum() / n)
        rooms_out.append({"name": rn, "coverage": round(cov, 3), "surface_m2": round(n * g.voxel ** 2, 1),
                          "changes": sum(1 for c in final if c.room == rn and not c.review),
                          "review": sum(1 for c in final if c.room == rn and c.review)})

    new_areas = []
    if len(new_vox):
        lab, k = cluster_points(new_vox, 0.15)
        for c in range(k):
            mem = lab == c
            if mem.sum() < 30:
                continue
            cen = new_vox[mem].mean(0)
            new_areas.append({"room": str(room_names(cen[None])[0]), "centroid": cen.round(3).tolist(),
                              "area_m2": round(float(mem.sum() * 0.01), 2)})

    # ---------------- verdict --------------------------------------------------
    confirmed = [c for c in final if not c.review]
    review = [c for c in final if c.review]
    low_cov = [r for r in rooms_out if r["coverage"] < cfg.coverage_min_room]
    reasons = []
    if confirmed:
        verdict = "Needs attention"
        reasons.append(f"{len(confirmed)} confirmed change(s)")
    elif review:
        verdict = "Review needed"
    else:
        verdict = "Guest-ready"
    if review:
        reasons.append(f"{len(review)} item(s) need human review")
    if low_cov:
        verdict += " (incomplete walkthrough)"
        reasons.append("low coverage in " + ", ".join(f"{r['name']} ({100 * r['coverage']:.0f}%)" for r in low_cov))
    if not reg.confident:
        reasons.append("localization confidence is low")
    timings["cluster_s"] = round(time.time() - t0, 2)
    timings["detect_total_s"] = round(time.time() - t_all, 2)

    regd = {"inlier_ratio": round(reg.inlier_ratio, 3), "rmse_m": round(reg.rmse, 4), "confident": reg.confident,
            "seconds": round(reg.seconds, 2), "chunks": reg.chunk_stats, "candidates": reg.candidates,
            "T_baseline_from_inspection": np.round(reg.T_bs, 5).tolist()}
    return InspectionResult(final, unv_regions, new_areas, rooms_out, verdict, reasons, status, added_vox,
                            added_labels, new_vox, regd, timings, asdict(cfg), reg.poses)
