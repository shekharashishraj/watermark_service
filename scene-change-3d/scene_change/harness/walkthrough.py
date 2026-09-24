"""Handheld walkthrough simulation: route planning, camera motion, sensor noise.

A walkthrough visits rooms in order. In each room the "cleaner" walks to one or
two standing spots and pans the phone around (a helical sweep that alternates
between eye level and looking down), then walks on. Poses carry VIO-style drift
and jitter, and every session lives in its own gravity-aligned frame, as ARKit
sessions do, so the pipeline has to localize the inspection itself.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from ..geometry import camera_pose, depth_edge_mask, intrinsics, make_pose, rot_z
from ..session import Session
from .raycast import BoxRenderer, Lighting
from .scene import Scene


@dataclass
class WalkConfig:
    width: int = 160
    height: int = 120
    hfov: float = 68.0
    rooms: list | None = None             # visit order (None: BFS from the hallway)
    pan_span: dict = field(default_factory=dict)   # room -> degrees (default 360)
    pan_step: float = 22.0
    walk_step: float = 0.5
    cam_height: float = 1.5
    station_jitter: float = 0.35
    two_stations_area: float = 13.0
    # sensor model
    depth_noise: tuple = (0.0015, 0.0025)  # sigma = a + b z^2
    depth_max: float = 5.0
    depth_min: float = 0.15
    depth_dropout: float = 0.01
    flying_pixel_prob: float = 0.35
    rgb_noise: float = 0.008
    exposure_jitter: float = 0.03
    # pose model
    drift_trans_per_m: float = 0.0        # metres of drift per metre walked (1-sigma, per axis)
    drift_yaw_deg_per_m: float = 0.0
    jitter_trans: float = 0.002
    jitter_rot_deg: float = 0.15
    frame_offset: bool = True             # express poses in a random session frame


def bfs_order(scene: Scene, start="hallway"):
    adj = {r.name: [] for r in scene.rooms}
    for dw in scene.doorways:
        a, b = dw.rooms
        adj[a].append(b)
        adj[b].append(a)
    order, seen, queue = [], {start}, [start]
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        for n in sorted(adj[cur]):
            if n not in seen:
                seen.add(n)
                queue.append(n)
    return order


class FloorGrid:
    """Walkable free space on a 2D grid for route planning."""

    def __init__(self, scene: Scene, res=0.1, clearance=0.3):
        self.res = res
        self.nx = int(np.ceil(scene.width / res)) + 1
        self.ny = int(np.ceil(scene.depth / res)) + 1
        xs = (np.arange(self.nx) + 0.5) * res
        ys = (np.arange(self.ny) + 0.5) * res
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        inside = np.zeros((self.nx, self.ny), dtype=bool)
        room_id = np.full((self.nx, self.ny), -1)
        for k, r in enumerate(scene.rooms):
            x0, y0, x1, y1 = r.interior
            m = (X >= x0) & (X < x1) & (Y >= y0) & (Y < y1)
            inside |= m
            room_id[m] = k
        for dw in scene.doorways:
            c = dw.center_xy()
            hw = dw.width / 2 - 0.05
            if dw.axis == "h":
                m = (np.abs(X - c[0]) < hw) & (np.abs(Y - c[1]) < 0.35)
            else:
                m = (np.abs(Y - c[1]) < hw) & (np.abs(X - c[0]) < 0.35)
            inside |= m
        obst = np.zeros_like(inside)
        for o in scene.objects.values():
            if not o.obstacle or o.category in ("structure", "decal", "wall_item"):
                continue
            lo, hi = o.aabb()
            if lo[2] > 1.2 or hi[2] - lo[2] < 0.05:
                continue
            obst |= (X >= lo[0]) & (X <= hi[0]) & (Y >= lo[1]) & (Y <= hi[1])
        walls = ~inside
        blocked = walls | obst
        # distance (m) to nearest blocked cell
        self.dist = ndimage.distance_transform_edt(~blocked) * res
        self.free = self.dist >= clearance
        # keep doorway cells walkable even though they sit near wall ends
        for dw in scene.doorways:
            c = dw.center_xy()
            if dw.axis == "h":
                m = (np.abs(X - c[0]) < 0.12) & (np.abs(Y - c[1]) < 0.4)
            else:
                m = (np.abs(Y - c[1]) < 0.12) & (np.abs(X - c[0]) < 0.4)
            self.free |= m & ~obst
        self.room_id = room_id
        self.X, self.Y = X, Y

    def cell(self, p):
        return int(np.clip(p[0] / self.res, 0, self.nx - 1)), int(np.clip(p[1] / self.res, 0, self.ny - 1))

    def pos(self, c):
        return np.array([(c[0] + 0.5) * self.res, (c[1] + 0.5) * self.res])

    def nearest_free(self, p):
        i, j = self.cell(p)
        if self.free[i, j]:
            return (i, j)
        fi, fj = np.nonzero(self.free)
        k = np.argmin((fi - i) ** 2 + (fj - j) ** 2)
        return (int(fi[k]), int(fj[k]))

    def astar(self, a, b):
        a, b = self.nearest_free(a), self.nearest_free(b)
        pen = 0.6 / np.maximum(self.dist, 0.05)          # prefer the middle of free space
        openq = [(0.0, a)]
        g = {a: 0.0}
        prev = {}
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414),
                 (-1, -1, 1.414)]
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == b:
                break
            for dx, dy, w in steps:
                n = (cur[0] + dx, cur[1] + dy)
                if not (0 <= n[0] < self.nx and 0 <= n[1] < self.ny) or not self.free[n]:
                    continue
                ng = g[cur] + w * (1 + pen[n])
                if ng < g.get(n, np.inf):
                    g[n] = ng
                    prev[n] = cur
                    h = np.hypot(n[0] - b[0], n[1] - b[1])
                    heapq.heappush(openq, (ng + h, n))
        if b not in g:
            return np.array([self.pos(a), self.pos(b)])
        path = [b]
        while path[-1] != a:
            path.append(prev[path[-1]])
        pts = np.array([self.pos(c) for c in path[::-1]])
        if len(pts) > 4:   # light smoothing
            k = np.ones(5) / 5
            sm = np.stack([np.convolve(np.pad(pts[:, i], 2, mode="edge"), k, "valid") for i in range(2)], axis=1)
            pts = sm
        return pts

    def station(self, scene: Scene, room_name: str, rng, avoid=None, min_sep=1.6, jitter=0.35):
        k = [r.name for r in scene.rooms].index(room_name)
        m = (self.room_id == k) & self.free & (self.dist >= 0.45)
        if not m.any():
            m = (self.room_id == k) & self.free
        if not m.any():
            return None
        ii, jj = np.nonzero(m)
        P = np.stack([(ii + 0.5) * self.res, (jj + 0.5) * self.res], axis=1)
        c = scene.room(room_name).center + rng.uniform(-jitter, jitter, 2)
        score = np.linalg.norm(P - c, axis=1) - 0.8 * np.minimum(self.dist[ii, jj], 1.2)
        if avoid is not None:
            sep = np.linalg.norm(P - avoid, axis=1)
            score = np.where(sep < min_sep, np.inf, -sep - 0.8 * np.minimum(self.dist[ii, jj], 1.2))
            if not np.isfinite(score).any():
                return None
        return P[int(np.argmin(score))]


def plan_walkthrough(scene: Scene, cfg: WalkConfig, rng: np.random.Generator):
    """Return a list of (T_world_camera, kind, room) camera poses."""
    grid = FloorGrid(scene)
    order = cfg.rooms or bfs_order(scene)
    cur = scene.entry[:2].copy()
    heading = float(np.arctan2(*(scene.room(order[0]).center - cur)[::-1]))
    poses = []
    t = 0.0
    hgt = cfg.cam_height

    def emit(p, yaw, pitch, roll, kind, room):
        z = hgt + 0.015 * np.sin(len(poses) * 1.7)
        poses.append((camera_pose(np.array([p[0], p[1], z]), yaw, pitch, roll), kind, room))

    for room in order:
        area = (scene.room(room).x1 - scene.room(room).x0) * (scene.room(room).y1 - scene.room(room).y0)
        st = [grid.station(scene, room, rng, jitter=cfg.station_jitter)]
        if area > cfg.two_stations_area and st[0] is not None:
            s2 = grid.station(scene, room, rng, avoid=st[0])
            if s2 is not None:
                st.append(s2)
        for s in st:
            if s is None:
                continue
            path = grid.astar(cur, s)
            seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
            cum = np.r_[0, np.cumsum(seg)]
            L = cum[-1]
            n = int(L // cfg.walk_step)
            for k in range(1, n + 1):
                d = k * cfg.walk_step
                p = np.array([np.interp(d, cum, path[:, 0]), np.interp(d, cum, path[:, 1])])
                ahead = np.array([np.interp(min(d + 0.8, L), cum, path[:, 0]), np.interp(min(d + 0.8, L), cum, path[:, 1])])
                if np.linalg.norm(ahead - p) > 1e-3:
                    heading = float(np.arctan2(ahead[1] - p[1], ahead[0] - p[0]))
                t += 1
                yaw = heading + np.radians(14) * np.sin(t * 0.9) + rng.normal(0, np.radians(4))
                pitch = np.radians(-17 + rng.normal(0, 5))
                emit(p, yaw, pitch, rng.normal(0, np.radians(1.5)), "walk", scene.room_at(*p) or room)
            cur = s.copy()
            span = cfg.pan_span.get(room, 360.0)
            nsteps = max(2, int(round(span / cfg.pan_step)))
            start = heading - np.radians(span) / 2 if span < 359 else heading + rng.uniform(-0.5, 0.5)
            for k in range(nsteps):
                yaw = start + np.radians(span) * k / nsteps
                phase = 0.5 - 0.5 * np.cos(2 * np.pi * k / 4.0)
                pitch = np.radians(-8 - 26 * phase + rng.normal(0, 2))
                sway = rng.normal(0, 0.03, 2)
                emit(s + sway, yaw + rng.normal(0, np.radians(2)), pitch, rng.normal(0, np.radians(1.5)), "pan", room)
            heading = start + np.radians(span)
    return poses


def _drift(poses_w, cfg: WalkConfig, rng):
    """Apply VIO-like drift (anchored at the first camera) and per-frame jitter."""
    if not poses_w:
        return []
    out = []
    anchor = poses_w[0][:3, 3].copy()
    t_d = np.zeros(3)
    yaw_d = 0.0
    prev = anchor.copy()
    for T in poses_w:
        step = np.linalg.norm(T[:3, 3] - prev) + 0.02   # rotation-only frames still drift a little
        prev = T[:3, 3].copy()
        t_d += rng.normal(0, cfg.drift_trans_per_m * np.sqrt(step), 3) * np.array([1, 1, 0.3])
        yaw_d += rng.normal(0, np.radians(cfg.drift_yaw_deg_per_m) * np.sqrt(step))
        D = make_pose(rot_z(yaw_d), anchor + t_d) @ make_pose(np.eye(3), -anchor)
        J = make_pose(rot_z(rng.normal(0, np.radians(cfg.jitter_rot_deg))), rng.normal(0, cfg.jitter_trans, 3))
        out.append(D @ T @ J)
    return out


def apply_sensor(rgb, depth, cfg: WalkConfig, rng, exposure=1.0):
    z = depth.astype(np.float64)
    valid = z > 0
    sigma = cfg.depth_noise[0] + cfg.depth_noise[1] * z ** 2
    zn = z + rng.normal(0, 1, z.shape) * sigma
    edges = depth_edge_mask(depth, 0.08) & valid
    if cfg.flying_pixel_prob > 0:
        zmax = ndimage.maximum_filter(z, 3)
        zmin = ndimage.minimum_filter(np.where(valid, z, np.inf), 3)
        zmin = np.where(np.isfinite(zmin), zmin, z)
        fly = edges & (rng.random(z.shape) < cfg.flying_pixel_prob)
        zn[fly] = zmin[fly] + (zmax[fly] - zmin[fly]) * rng.random(int(fly.sum()))
    drop = rng.random(z.shape) < cfg.depth_dropout
    zn[~valid | drop | (z > cfg.depth_max) | (z < cfg.depth_min)] = 0.0
    img = rgb.astype(np.float64) * exposure + rng.normal(0, cfg.rgb_noise, rgb.shape)
    img = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return img, zn.astype(np.float32)


def render_walkthrough(scene: Scene, poses, cfg: WalkConfig, lighting: Lighting, rng, renderer=None,
                       extra_scenes=()):
    """Render RGB-D for true world poses. Returns (rgb, depth, oid, extra_oids)."""
    W, H = cfg.width, cfg.height
    K = intrinsics(W, H, cfg.hfov)
    r = renderer or BoxRenderer(scene)
    extras = [BoxRenderer(s) for s in extra_scenes]
    rgbs, depths, oids = [], [], []
    extra_oids = [[] for _ in extras]
    expo = 1.0
    for T in poses:
        out = r.render(K, T, W, H, lighting)
        expo = float(np.clip(expo + rng.normal(0, cfg.exposure_jitter), 0.85, 1.15))
        img, dz = apply_sensor(out["rgb"], out["depth"], cfg, rng, exposure=expo)
        rgbs.append(img)
        depths.append(dz)
        oids.append(out["oid"].astype(np.int16))
        for k, er in enumerate(extras):
            extra_oids[k].append(er.render(K, T, W, H, shade=False)["oid"].astype(np.int16))
    return K, np.stack(rgbs), np.stack(depths), np.stack(oids), [np.stack(e) for e in extra_oids]


def simulate_session(name: str, scene: Scene, cfg: WalkConfig, lighting: Lighting, rng: np.random.Generator,
                     extra_scenes=(), session_frame=None):
    """Plan, render and noise a walkthrough.

    Returns (Session in its own frame, gt dict with true world poses, oid maps and
    T_session_world).
    """
    plan = plan_walkthrough(scene, cfg, rng)
    true_w = [p for p, _, _ in plan]
    kinds = [k for _, k, _ in plan]
    rooms = [rm for _, _, rm in plan]
    K, rgb, depth, oid, extra = render_walkthrough(scene, true_w, cfg, lighting, rng, extra_scenes=extra_scenes)
    noisy_w = _drift(true_w, cfg, rng)
    if session_frame is None:
        if cfg.frame_offset:
            session_frame = make_pose(rot_z(rng.uniform(-np.pi, np.pi)),
                                      np.r_[rng.uniform(-3, 3, 2), rng.uniform(-0.1, 0.1)])
        else:
            session_frame = np.eye(4)
    T_sw = session_frame   # maps world -> session coordinates
    poses_s = np.stack([T_sw @ T for T in noisy_w])
    sess = Session(name, K, rgb, depth, poses_s, np.arange(len(poses_s)) * 0.5,
                   meta={"source": "harness", "kinds": kinds, "rooms": rooms})
    gt = {"poses_world": np.stack(true_w), "oid": oid, "T_session_world": T_sw, "kinds": kinds, "rooms": rooms,
          "extra_oid": extra}
    return sess, gt
