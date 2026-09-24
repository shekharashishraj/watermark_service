"""The persistent 3D baseline: Gaussian map + keyframe depth record + room annotations.

The keyframe depth record is what lets the system tell "this space was seen
empty in the baseline" apart from "this space was never observed", which is the
basis for reporting unverified regions instead of false changes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .gaussians import GaussianMap, build_gaussian_map, estimate_floor
from .geometry import invert_pose
from .session import Session


def depth_tolerance(z):
    """Depth agreement tolerance (m) as a function of range: sensor noise + pose error."""
    return 0.045 + 0.02 * np.asarray(z)


def min_pool_depth(depth: np.ndarray, f: int) -> np.ndarray:
    H, W = depth.shape
    h, w = H // f, W // f
    d = depth[: h * f, : w * f].reshape(h, f, w, f).astype(np.float32)
    d = np.where(d > 0, d, np.inf).min(axis=(1, 3))
    return np.where(np.isfinite(d), d, 0.0).astype(np.float32)


def points_in_polygon(xy: np.ndarray, poly) -> np.ndarray:
    poly = np.asarray(poly, dtype=np.float64)
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xin = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= cond & (x < xin)
    return inside


def free_space_votes(P: np.ndarray, poses: np.ndarray, depths: np.ndarray, K: np.ndarray, max_range: float = 5.0,
                     chunk: int = 200_000):
    """Count, per 3D point, the depth frames whose ray passed through it (free) or ended on it (surface)."""
    P = np.asarray(P, dtype=np.float64)
    n_free = np.zeros(len(P), dtype=np.int32)
    n_surf = np.zeros(len(P), dtype=np.int32)
    if len(P) == 0:
        return n_free, n_surf
    h, w = depths.shape[1:]
    inv = invert_pose(poses)
    for s in range(0, len(P), chunk):
        Q = P[s:s + chunk]
        f_acc = np.zeros(len(Q), dtype=np.int32)
        s_acc = np.zeros(len(Q), dtype=np.int32)
        for k in range(len(poses)):
            T = inv[k]
            pc = Q @ T[:3, :3].T + T[:3, 3]
            z = pc[:, 2]
            ok = (z > 0.1) & (z < max_range)
            if not ok.any():
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                u = np.round(K[0, 0] * pc[:, 0] / z + K[0, 2])
                v = np.round(K[1, 1] * pc[:, 1] / z + K[1, 2])
            ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if not ok.any():
                continue
            ii = np.nonzero(ok)[0]
            d = depths[k, v[ii].astype(np.int64), u[ii].astype(np.int64)]
            good = d > 0
            ii, d = ii[good], d[good]
            tau = depth_tolerance(d)
            f_acc[ii] += (z[ii] < d - tau)
            s_acc[ii] += (np.abs(z[ii] - d) <= tau)
        n_free[s:s + chunk] = f_acc
        n_surf[s:s + chunk] = s_acc
    return n_free, n_surf


@dataclass
class BaselineModel:
    gaussians: GaussianMap
    kf_poses: np.ndarray        # (M,4,4) camera-to-baseline
    kf_depth: np.ndarray        # (M,h,w) float32 metres (min-pooled)
    kf_K: np.ndarray            # (3,3) intrinsics at keyframe depth resolution
    floor_z: float
    rooms: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def free_space_votes(self, P: np.ndarray, max_range: float = 5.0):
        """For world points P, count keyframes that saw through them (free) or saw a surface there."""
        return free_space_votes(P, self.kf_poses, self.kf_depth, self.kf_K, max_range)

    def room_of(self, xy: np.ndarray) -> np.ndarray:
        names = np.full(len(xy), "unassigned", dtype=object)
        for r in self.rooms:
            names[points_in_polygon(np.asarray(xy)[:, :2], r["polygon"])] = r["name"]
        return names

    def ceiling_mask(self) -> np.ndarray:
        g = self.gaussians
        return (g.normals[:, 2] < -0.8) & (g.means[:, 2] > self.floor_z + 1.8)

    # ------------------------------------------------------------------ I/O
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = self.gaussians.to_dict()
        arrays.update(kf_poses=self.kf_poses, kf_depth=np.round(self.kf_depth * 1000).astype(np.uint16),
                      kf_K=self.kf_K, floor_z=np.array(self.floor_z))
        np.savez_compressed(path, **arrays)
        side = {"rooms": self.rooms, "meta": self.meta}
        Path(str(path) + ".json").write_text(json.dumps(side, indent=1))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BaselineModel":
        path = Path(path)
        d = dict(np.load(path))
        side = json.loads(Path(str(path) + ".json").read_text()) if Path(str(path) + ".json").exists() else {}
        return cls(GaussianMap.from_dict(d), d["kf_poses"], d["kf_depth"].astype(np.float32) / 1000.0, d["kf_K"],
                   float(d["floor_z"]), side.get("rooms", []), side.get("meta", {}))


def build_baseline(session: Session, rooms=None, voxel: float = 0.05, kf_pool: int = 2,
                   gaussians: GaussianMap | None = None) -> BaselineModel:
    """Build the baseline model from the first walkthrough.

    ``gaussians`` lets a 3DGS model trained elsewhere replace the fused map; the
    keyframe depth record still comes from the walkthrough.
    """
    t0 = time.time()
    gm = gaussians if gaussians is not None else build_gaussian_map(session, voxel=voxel)
    t_map = time.time() - t0
    kf_depth = np.stack([min_pool_depth(d, kf_pool) for d in session.depth])
    K = session.K.copy()
    K[0, 0] /= kf_pool
    K[1, 1] /= kf_pool
    K[0, 2] = (K[0, 2] - (kf_pool - 1) / 2) / kf_pool
    K[1, 2] = (K[1, 2] - (kf_pool - 1) / 2) / kf_pool
    floor = estimate_floor(gm)
    meta = {"source_session": session.name, "frames": len(session), "voxel": voxel, "gaussians": len(gm),
            "time_map_s": round(t_map, 2), "time_total_s": round(time.time() - t0, 2),
            "image_size": [session.width, session.height]}
    return BaselineModel(gm, session.poses.copy(), kf_depth, K, floor, list(rooms or []), meta)
