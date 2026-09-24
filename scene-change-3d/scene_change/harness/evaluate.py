"""Ground truth and metrics for harness scenarios.

Change localization is scored the way multi-view change benchmarks do it: on
the inspection images. For every inspection frame the ground-truth change mask
marks pixels showing a changed object in either scene state (where a removed
object used to be, where an added object is now, both ends of a move, and
restyled or stained surfaces). Predicted 3D changes are projected into the same
frames with an occlusion test against the observed depth.

Reported metrics
    mIoU            mean of change-class IoU and no-change IoU over all pixels
    F1              pixel-level F1 of the change class
    PASLCD protocol change-class IoU and F1 per frame, averaged over frames that
                    show a change (as O-SCD's utils/evaluate.py with torchmetrics'
                    binary JaccardIndex/F1Score), plus the false-alarm pixel rate
                    on frames without any change
    object recall   share of observable ground-truth changes hit by a detection
    FP/inspection   detections that match no ground-truth change
    unverified      share of never-observed baseline objects reported unverified
"""

from __future__ import annotations

import numpy as np

from ..geometry import invert_pose


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------

def box_signed_distance(P, center, half, yaw):
    o = P - center
    c, s = np.cos(yaw), np.sin(yaw)
    lx = c * o[:, 0] + s * o[:, 1]
    ly = -s * o[:, 0] + c * o[:, 1]
    q = np.abs(np.stack([lx, ly, o[:, 2]], 1)) - half
    outside = np.linalg.norm(np.maximum(q, 0), axis=1)
    inside = np.minimum(q.max(1), 0)
    return outside + inside


def boxes_of(objects_summary_or_scene, oids=None):
    """(center, half, yaw, oid) arrays for the given scene's objects."""
    scene = objects_summary_or_scene
    out = []
    for o in scene.objects.values():
        if oids is not None and o.oid not in oids:
            continue
        for w, h, yaw, _, _ in o.world_boxes():
            out.append((w, h, yaw, o.oid))
    return out


def label_points(scene, P, tol=0.04, oids=None, chunk=100_000):
    """Object id of the nearest box surface within ``tol`` for each point (0 = none)."""
    P = np.asarray(P, dtype=np.float64)
    boxes = boxes_of(scene, oids)
    best = np.full(len(P), np.inf)
    lab = np.zeros(len(P), dtype=np.int64)
    for c, h, yaw, oid in boxes:
        lo = c - np.linalg.norm(h) - tol
        hi = c + np.linalg.norm(h) + tol
        m = np.all((P >= lo) & (P <= hi), axis=1)
        if not m.any():
            continue
        d = np.abs(box_signed_distance(P[m], c, h, yaw))
        idx = np.nonzero(m)[0]
        better = (d < best[idx]) & (d <= tol)
        best[idx[better]] = d[better]
        lab[idx[better]] = oid
    return lab


# ----------------------------------------------------------------------------
# Ground-truth change sets
# ----------------------------------------------------------------------------

def change_id_sets(changes):
    """oids that mark changed pixels in the baseline-state and inspection-state renders."""
    base, insp = set(), set()
    for c in changes:
        t = c["type"]
        if t == "removed":
            base.add(c["oid"])
        elif t == "added":
            insp.add(c["oid"])
        elif t == "moved":
            base.add(c["oid"])
            insp.add(c["oid"])
        elif t == "appearance":
            insp.add(c["oid"])
            if c.get("swap"):
                base.add(c["oid"])
    return base, insp


def gt_masks(gt_arrays, changes):
    base_ids, insp_ids = change_id_sets(changes)
    ob = gt_arrays["insp_oid_basestate"]
    oi = gt_arrays["insp_oid"]
    m = np.zeros(oi.shape, dtype=bool)
    if base_ids:
        m |= np.isin(ob, list(base_ids))
    if insp_ids:
        m |= np.isin(oi, list(insp_ids))
    return m


def observable_changes(gt_arrays, changes, depth=None, max_range=4.5, min_px=25):
    """Per change: number of inspection pixels showing it (within sensor range if depth is given)."""
    ob = gt_arrays["insp_oid_basestate"]
    oi = gt_arrays["insp_oid"]
    out = {}
    rng_ok = None if depth is None else (depth > 0) & (depth < max_range)
    for c in changes:
        cnt = 0
        if c["type"] in ("removed", "moved") or c.get("swap"):
            m = ob == c["oid"]
            cnt += int((m & rng_ok).sum()) if rng_ok is not None else int(m.sum())
        if c["type"] in ("added", "moved", "appearance"):
            m = oi == c["oid"]
            cnt += int((m & rng_ok).sum()) if rng_ok is not None else int(m.sum())
        out[c["id"]] = cnt
    return {k: v for k, v in out.items()}, {k for k, v in out.items() if v >= min_px}


# ----------------------------------------------------------------------------
# Projection of predicted 3D changes into inspection frames
# ----------------------------------------------------------------------------

def splat_points_mask(points, K, T_wc, depth, voxel, tol=0.06):
    """Binary mask of pixels covered by voxel-sized splats at ``points`` that are not occluded."""
    H, W = depth.shape
    mask = np.zeros((H, W), dtype=bool)
    if len(points) == 0:
        return mask
    T = invert_pose(T_wc)
    pc = points @ T[:3, :3].T + T[:3, 3]
    z = pc[:, 2]
    ok = z > 0.15
    if not ok.any():
        return mask
    pc, z = pc[ok], z[ok]
    u = K[0, 0] * pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    r = np.clip(np.ceil(0.5 * voxel * K[0, 0] / z), 1, 12).astype(int)
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    inb = (ui >= -12) & (ui < W + 12) & (vi >= -12) & (vi < H + 12)
    for rr in np.unique(r[inb]):
        sel = np.nonzero(inb & (r == rr))[0]
        oy, ox = np.mgrid[-rr:rr + 1, -rr:rr + 1]
        px = (ui[sel, None] + ox.ravel()[None]).ravel()
        py = (vi[sel, None] + oy.ravel()[None]).ravel()
        zz = np.repeat(z[sel], ox.size)
        m = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        px, py, zz = px[m], py[m], zz[m]
        d = depth[py, px]
        vis = (d <= 0) | (zz <= d + tol + 0.02 * zz)
        mask[py[vis], px[vis]] = True
    return mask


def predicted_masks(result, session, voxel, include_review=True):
    """Project detections into every inspection frame (using the pipeline's own poses)."""
    pts = []
    for ch in result.changes:
        if ch.review and not include_review:
            continue
        if ch.points is not None:
            pts.append(ch.points)
        if ch.points_to is not None:
            pts.append(ch.points_to)
    P = np.concatenate(pts) if pts else np.zeros((0, 3))
    masks = np.zeros(session.depth.shape, dtype=bool)
    for i in range(len(session)):
        masks[i] = splat_points_mask(P, session.K, result.poses[i], session.depth[i], voxel)
    return masks


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def pixel_metrics(pred, gt):
    tp = float(np.sum(pred & gt))
    fp = float(np.sum(pred & ~gt))
    fn = float(np.sum(~pred & gt))
    tn = float(np.sum(~pred & ~gt))
    iou_c = tp / max(tp + fp + fn, 1.0)
    iou_n = tn / max(tn + fp + fn, 1.0)
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"mIoU": (iou_c + iou_n) / 2, "IoU_change": iou_c, "F1": f1, "precision": prec, "recall": rec,
            "gt_pixels": int(tp + fn), "pred_pixels": int(tp + fp)}


def frame_metrics(pred, gt):
    """PASLCD-style scores: per-frame binary IoU/F1 of the change class, averaged over frames with GT change.

    Frames without GT change contribute to ``false_alarm_rate`` (share of their pixels
    predicted as change) instead, since IoU/F1 are undefined there.
    """
    pred = np.asarray(pred, bool)
    gt = np.asarray(gt, bool)
    ax = tuple(range(1, gt.ndim))
    tp = np.sum(pred & gt, axis=ax).astype(np.float64)
    fp = np.sum(pred & ~gt, axis=ax).astype(np.float64)
    fn = np.sum(~pred & gt, axis=ax).astype(np.float64)
    has = (tp + fn) > 0
    iou = np.where(has, tp / np.maximum(tp + fp + fn, 1.0), 0.0)
    f1 = np.where(has, 2 * tp / np.maximum(2 * tp + fp + fn, 1.0), 0.0)
    empty = ~has
    far = float(np.mean(pred[empty])) if empty.any() else None
    return {"IoU": float(iou[has].mean()) if has.any() else None, "F1": float(f1[has].mean()) if has.any() else None,
            "frames_with_change": int(has.sum()), "false_alarm_rate": far,
            "frames_with_false_alarm": int(np.sum(np.any(pred[empty], axis=tuple(range(1, gt.ndim))))) if empty.any() else 0}


def gt_regions(scene_base, scene_insp, changes):
    """Per GT change: list of (center, half, yaw) boxes in world coords covering the changed region."""
    regions = {}
    for c in changes:
        boxes = []
        oid = c["oid"]
        if c["type"] in ("removed", "moved") or c.get("swap"):
            if oid in scene_base.objects:
                boxes += [(w, h, y) for w, h, y, _, _ in scene_base.objects[oid].world_boxes()]
        if c["type"] in ("added", "moved", "appearance"):
            if oid in scene_insp.objects:
                boxes += [(w, h, y) for w, h, y, _, _ in scene_insp.objects[oid].world_boxes()]
        regions[c["id"]] = boxes
    return regions


def match_detections(result, regions, T_world_from_baseline=None, tol=0.15, min_frac=0.3):
    """Greedy object-level matching of detections to GT changes via 3D support overlap."""
    det_hits = {}
    for ch in result.changes:
        pts = [ch.points] + ([ch.points_to] if ch.points_to is not None else [])
        P = np.concatenate([p for p in pts if p is not None])
        if T_world_from_baseline is not None:
            P = P @ T_world_from_baseline[:3, :3].T + T_world_from_baseline[:3, 3]
        hits = []
        for gid, boxes in regions.items():
            if not boxes:
                continue
            d = np.min([np.abs(box_signed_distance(P, c, h, y)) for c, h, y in boxes], axis=0)
            frac = float(np.mean(d < tol))
            if frac >= min_frac:
                hits.append((frac, gid))
        det_hits[ch.id] = sorted(hits, reverse=True)
    return det_hits


def unverified_recall(result, model, scene_base, gt_arrays, changes, voxel):
    """Share of baseline objects never seen during the inspection that the system marked unverified."""
    oi = gt_arrays["insp_oid"]
    ob = gt_arrays["insp_oid_basestate"]
    seen = set(np.unique(oi).tolist()) | set(np.unique(ob).tolist())
    cand = [o for o in scene_base.objects.values() if o.category in ("furniture", "item", "wall_item")
            and o.oid not in seen]
    if not cand:
        return None, 0
    mu = model.gaussians.means.astype(np.float64)
    labels = label_points(scene_base, mu, tol=0.04, oids={o.oid for o in cand})
    hit = 0
    total = 0
    for o in cand:
        m = labels == o.oid
        if m.sum() < 5:
            continue      # also never seen in the baseline: nothing to verify
        total += 1
        if np.mean(result.status[m] == 3) >= 0.5:
            hit += 1
    return (hit / total if total else None), total


# ----------------------------------------------------------------------------
# Scenario-level evaluation of the 3D pipeline
# ----------------------------------------------------------------------------

TYPE_MATCH = {"removed": {"missing"}, "added": {"added"}, "moved": {"moved"}, "appearance": {"appearance"}}


def registration_errors(result_poses, gt_poses_world, T_world_from_baseline=None):
    from ..geometry import pose_error
    T = np.eye(4) if T_world_from_baseline is None else T_world_from_baseline
    errs = np.array([pose_error(T @ result_poses[i], gt_poses_world[i]) for i in range(len(gt_poses_world))])
    return {"rot_deg_median": float(np.median(errs[:, 0])), "trans_cm_median": float(100 * np.median(errs[:, 1])),
            "trans_cm_p95": float(100 * np.percentile(errs[:, 1], 95))}


def evaluate_3d(sc, model, result, voxel=None):
    """All harness metrics for one inspection result."""
    voxel = voxel or model.gaussians.voxel
    session = sc["inspection"]
    changes = sc["gt"]["changes"]
    arrays = sc["gt_arrays"]
    T_bw = np.asarray(sc["gt"].get("T_baseline_world", np.eye(4)))
    T_wb = invert_pose(T_bw)
    gt = gt_masks(arrays, changes)
    out = {}
    for tag, inc in (("all", True), ("confirmed", False)):
        pred = predicted_masks(result, session, voxel, include_review=inc)
        out[f"pixel_{tag}"] = pixel_metrics(pred, gt)
    counts, observable = observable_changes(arrays, changes, session.depth)
    regions = gt_regions(sc["scene"], sc["inspection_scene"], changes)
    hits = match_detections(result, regions, T_wb)
    hit_gt = {}
    fp = []
    type_ok = 0
    matched = 0
    for ch in result.changes:
        h = hits.get(ch.id, [])
        if not h:
            fp.append(ch)
            continue
        matched += 1
        gid = h[0][1]
        hit_gt.setdefault(gid, []).append(ch.id)
        gtype = next(c["type"] for c in changes if c["id"] == gid)
        if ch.type in TYPE_MATCH[gtype]:
            type_ok += 1
    obs = [c for c in changes if c["id"] in observable]
    rec = [c for c in obs if c["id"] in hit_gt]
    out["objects"] = {
        "gt_changes": len(changes),
        "observable": len(obs),
        "detected": len(rec),
        "recall": (len(rec) / len(obs)) if obs else None,
        "detections": len(result.changes),
        "false_positives": len(fp),
        "false_positives_confirmed": sum(1 for c in fp if not c.review),
        "type_accuracy": (type_ok / matched) if matched else None,
        "per_type_recall": {t: _ratio([c for c in obs if c["type"] == t], hit_gt) for t in TYPE_MATCH},
        "missed": [f'{c["type"]}:{c["name"]}@{c["room"]}' for c in obs if c["id"] not in hit_gt],
        "unobservable": [f'{c["type"]}:{c["name"]}@{c["room"]}' for c in changes if c["id"] not in observable],
        "fp_list": [f"{c.type}@{c.room}{' (review)' if c.review else ''}" for c in fp],
    }
    ur, n_unobs = unverified_recall(result, model, sc["scene"], arrays, changes, voxel)
    out["unverified"] = {"objects_never_seen": n_unobs, "reported_unverified": ur}
    out["registration"] = registration_errors(result.poses, arrays["insp_poses_world"], T_wb)
    out["timings"] = dict(result.timings)
    out["verdict"] = result.verdict
    return out


def _ratio(items, hit):
    if not items:
        return None
    return sum(1 for c in items if c["id"] in hit) / len(items)
