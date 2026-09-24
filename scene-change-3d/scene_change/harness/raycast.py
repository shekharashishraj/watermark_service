"""CPU ray caster for box scenes: RGB, z-depth, object ids and normals.

Boxes are yaw-rotated (rotation about z only), which keeps the slab test cheap
and fully vectorised over (rays x boxes). Textures are procedural functions of
the object-local hit point, so an object's appearance moves with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import invert_pose, pixel_rays
from .scene import Scene


# ----------------------------------------------------------------------------
# Procedural noise
# ----------------------------------------------------------------------------

def _hash(ix, iy, seed):
    h = (np.asarray(ix, dtype=np.int64) * 374761393 + np.asarray(iy, dtype=np.int64) * 668265263
         + (int(seed) & 0x3FFFFFFF) * 1442695041) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return h.astype(np.float64) / 4294967296.0


def vnoise(x, y, seed):
    """Smooth value noise in [0, 1)."""
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    ux, uy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    ix, iy = ix.astype(np.int64), iy.astype(np.int64)
    a = _hash(ix, iy, seed)
    b = _hash(ix + 1, iy, seed)
    c = _hash(ix, iy + 1, seed)
    d = _hash(ix + 1, iy + 1, seed)
    return (a * (1 - ux) + b * ux) * (1 - uy) + (c * (1 - ux) + d * ux) * uy


def fbm(x, y, seed, octaves=3):
    out = np.zeros_like(x, dtype=np.float64)
    amp, tot, f = 1.0, 0.0, 1.0
    for o in range(octaves):
        out += amp * vnoise(x * f, y * f, seed + 17 * o)
        tot += amp
        amp *= 0.5
        f *= 2.0
    return out / tot


def _mix(a, b, w):
    return a * (1 - w[:, None]) + b * w[:, None]


BOOK_COLORS = np.array([[0.62, 0.16, 0.14], [0.14, 0.28, 0.52], [0.86, 0.78, 0.55], [0.18, 0.42, 0.28],
                        [0.12, 0.12, 0.14], [0.78, 0.46, 0.18], [0.55, 0.52, 0.48], [0.42, 0.22, 0.42]])


def albedo(material, s, t, u=None):
    """Evaluate a material pattern at face coordinates (s, t) in metres."""
    c = np.asarray(material.color, dtype=np.float64)
    c2 = np.asarray(material.color2 if material.color2 is not None else c * (1 - material.contrast), dtype=np.float64)
    n = len(s)
    base = np.broadcast_to(c, (n, 3)).astype(np.float64)
    sc = max(material.scale, 1e-3)
    seed = material.seed
    p = material.pattern
    if p == "plain":
        w = fbm(s / 0.6, t / 0.6, seed, 2)
        return base * (1 + material.contrast * (w - 0.5))[:, None]
    if p == "fabric":
        w = vnoise(s / sc, t / sc, seed)
        return base * (1 + 2 * material.contrast * (w - 0.5))[:, None]
    if p == "noise":
        w = fbm(s / sc, t / sc, seed, 3)
        w = np.clip(0.5 + (w - 0.5) * 2.2, 0, 1)
        return _mix(base, c2, w)
    if p == "stripes":
        w = 0.5 + 0.5 * np.tanh(6 * np.sin(2 * np.pi * s / sc))
        return _mix(base, c2, w * min(1.0, material.contrast * 4))
    if p == "tiles":
        gi, gj = np.floor(s / sc), np.floor(t / sc)
        fs, ft = s / sc - gi, t / sc - gj
        g = 0.025 / sc
        grout = (fs < g) | (ft < g)
        jit = _hash(gi.astype(np.int64), gj.astype(np.int64), seed) - 0.5
        col = base * (1 + material.contrast * jit)[:, None]
        col[grout] = c2
        return col
    if p == "planks":
        row = np.floor(t / sc)
        off = _hash(row.astype(np.int64), 7, seed) * 2.0
        seg = np.floor((s + off) / 1.1)
        jit = _hash(row.astype(np.int64), seg.astype(np.int64), seed) - 0.5
        grain = vnoise(s * 6.0, t * 40.0, seed + 3) - 0.5
        col = base * (1 + material.contrast * (1.4 * jit + 0.6 * grain))[:, None]
        gap = ((t / sc - row) < 0.05) | (((s + off) / 1.1 - seg) < 0.01)
        col[gap] *= 0.55
        return col
    if p == "books":
        colw = 0.04
        ci = np.floor(s / colw).astype(np.int64)
        row = np.floor(t / 0.33).astype(np.int64)
        h = _hash(ci, row, seed)
        col = BOOK_COLORS[(h * len(BOOK_COLORS)).astype(np.int64) % len(BOOK_COLORS)]
        ft = t / 0.33 - row
        top = 0.1 + 0.8 * _hash(ci, row + 101, seed)
        shelf = ft < 0.08
        gapm = ft > np.clip(top, 0.5, 0.95)
        col = col.copy()
        col[gapm] = 0.12
        col[shelf] = c
        return col
    if p == "canvas":
        c3 = np.asarray(material.color3 if material.color3 is not None else c2 * 0.5)
        w = fbm(s / sc, t / sc, seed, 3)
        v = fbm(s / (sc * 0.5), t / (sc * 0.5), seed + 5, 2)
        col = np.where((w > 0.52)[:, None], c2, base)
        col = np.where((v > 0.62)[:, None], c3, col)
        return col
    if p == "window":
        w = np.clip(0.5 + t / 1.2, 0, 1)
        col = _mix(np.broadcast_to(c2, (n, 3)), base, w)
        mull = (np.abs(s) < 0.02) | (np.abs(t) < 0.02)
        col = col.copy()
        col[mull] = 0.95
        return col
    if p == "mirror":
        w = 0.5 + 0.5 * np.sin((s + t) / sc * 3.0 + fbm(s / 0.4, t / 0.4, seed, 2) * 3)
        return _mix(base, c2, 0.35 * w)
    if p == "screen":
        w = np.clip(0.5 + (s + t) * 0.4, 0, 1)
        return _mix(base, c2, 0.5 * w)
    return base


# ----------------------------------------------------------------------------
# Lighting
# ----------------------------------------------------------------------------

@dataclass
class Lighting:
    exposure: float = 1.0
    white_balance: tuple = (1.0, 1.0, 1.0)
    ambient: float = 0.5
    ceiling: float = 0.75
    room_gain: dict = field(default_factory=dict)
    lamps: list = field(default_factory=list)   # (position (3,), intensity, room)


# ----------------------------------------------------------------------------
# Ray casting
# ----------------------------------------------------------------------------

class BoxRenderer:
    def __init__(self, scene: Scene):
        a = scene.box_arrays()
        self.scene = scene
        self.center = a["center"]
        self.half = a["half"]
        self.yaw = a["yaw"]
        self.cos = np.cos(self.yaw)
        self.sin = np.sin(self.yaw)
        self.mat = a["material"]
        self.oid = a["oid"]
        self.local = a["local"]
        self.radius = np.linalg.norm(self.half, axis=1)
        self.materials = scene.materials
        self.room_rects = np.array([[r.x0, r.y0, r.x1, r.y1] for r in scene.rooms])
        self.room_names = [r.name for r in scene.rooms]
        self.height = scene.height

    # ------------------------------------------------------------------
    def _cull(self, K, T_cw, W, H, max_range):
        Pc = self.center @ T_cw[:3, :3].T + T_cw[:3, 3]
        r = self.radius
        a = (W / 2 + 1) / K[0, 0]
        b = (H / 2 + 1) / K[1, 1]
        na, nb = np.sqrt(1 + a * a), np.sqrt(1 + b * b)
        x, y, z = Pc[:, 0], Pc[:, 1], Pc[:, 2]
        keep = z > -r
        keep &= (x - a * z) / na < r
        keep &= (-x - a * z) / na < r
        keep &= (y - b * z) / nb < r
        keep &= (-y - b * z) / nb < r
        keep &= z - r < max_range
        return np.nonzero(keep)[0]

    def cast(self, origin, dirs, boxes=None, chunk=24):
        """Nearest hit per ray. Returns (t, box index) with t=inf / -1 on miss."""
        R = dirs.shape[0]
        best_t = np.full(R, np.inf)
        best_b = np.full(R, -1, dtype=np.int64)
        if boxes is None:
            boxes = np.arange(len(self.center))
        dx, dy, dz = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        eps = 1e-6
        for k in range(0, len(boxes), chunk):
            bi = boxes[k:k + chunk]
            c, s = self.cos[bi][:, None], self.sin[bi][:, None]
            o = origin[None, :] - self.center[bi]
            ox = self.cos[bi] * o[:, 0] + self.sin[bi] * o[:, 1]
            oy = -self.sin[bi] * o[:, 0] + self.cos[bi] * o[:, 1]
            oz = o[:, 2]
            lx = c * dx[None, :] + s * dy[None, :]
            ly = -s * dx[None, :] + c * dy[None, :]
            lz = np.broadcast_to(dz[None, :], lx.shape)
            h = self.half[bi]
            tmin = np.full(lx.shape, -np.inf)
            tmax = np.full(lx.shape, np.inf)
            for (L, O, HH) in ((lx, ox, h[:, 0]), (ly, oy, h[:, 1]), (lz, oz, h[:, 2])):
                inv = 1.0 / np.where(np.abs(L) < 1e-12, 1e-12, L)
                t1 = (-HH[:, None] - O[:, None]) * inv
                t2 = (HH[:, None] - O[:, None]) * inv
                tmin = np.maximum(tmin, np.minimum(t1, t2))
                tmax = np.minimum(tmax, np.maximum(t1, t2))
            hit = (tmax >= tmin) & (tmin > eps)
            th = np.where(hit, tmin, np.inf)
            j = np.argmin(th, axis=0)
            tj = th[j, np.arange(R)]
            better = tj < best_t
            best_t[better] = tj[better]
            best_b[better] = bi[j[better]]
        return best_t, best_b

    # ------------------------------------------------------------------
    def render(self, K, T_wc, W, H, lighting: Lighting | None = None, max_range=30.0, shade=True):
        T_cw = invert_pose(T_wc)
        dirs_c = pixel_rays(K, W, H)
        dirs = dirs_c @ T_wc[:3, :3].T
        origin = T_wc[:3, 3]
        boxes = self._cull(K, T_cw, W, H, max_range)
        t, b = self.cast(origin, dirs, boxes)
        hit = b >= 0
        depth = np.where(hit, t, 0.0).reshape(H, W).astype(np.float32)
        oid = np.where(hit, self.oid[np.maximum(b, 0)], 0).reshape(H, W).astype(np.int32)
        out = {"depth": depth, "oid": oid}
        if not shade:
            return out
        rgb = np.zeros((H * W, 3))
        normal = np.zeros((H * W, 3))
        if hit.any():
            hb = b[hit]
            p = origin + dirs[hit] * t[hit, None]
            # local box coordinates
            o = p - self.center[hb]
            c, s = self.cos[hb], self.sin[hb]
            lp = np.stack([c * o[:, 0] + s * o[:, 1], -s * o[:, 0] + c * o[:, 1], o[:, 2]], axis=1)
            rel = np.abs(lp) / np.maximum(self.half[hb], 1e-9)
            axis = np.argmax(rel, axis=1)
            sign = np.sign(lp[np.arange(len(hb)), axis])
            nl = np.zeros_like(lp)
            nl[np.arange(len(hb)), axis] = sign
            nw = np.stack([c * nl[:, 0] - s * nl[:, 1], s * nl[:, 0] + c * nl[:, 1], nl[:, 2]], axis=1)
            normal[hit] = nw
            po = lp + self.local[hb]
            ss = np.where(axis == 0, po[:, 1], po[:, 0])
            tt = np.where(axis == 2, po[:, 1], po[:, 2])
            alb = np.zeros((len(hb), 3))
            mats = self.mat[hb]
            for m in np.unique(mats):
                sel = mats == m
                alb[sel] = albedo(self.materials[m], ss[sel], tt[sel])
            light = lighting or Lighting()
            irr = self._irradiance(p, nw, light)
            col = alb * irr[:, None] * light.exposure * np.asarray(light.white_balance)[None, :]
            rgb[hit] = np.clip(col, 0.0, 1.0)
        out["rgb"] = rgb.reshape(H, W, 3).astype(np.float32)
        out["normal"] = normal.reshape(H, W, 3).astype(np.float32)
        return out

    def _irradiance(self, p, n, light: Lighting):
        irr = light.ambient * (0.78 + 0.22 * n[:, 2])
        q = p + 0.05 * n
        rr = self.room_rects
        inside = ((q[:, 0:1] >= rr[None, :, 0]) & (q[:, 0:1] < rr[None, :, 2]) &
                  (q[:, 1:2] >= rr[None, :, 1]) & (q[:, 1:2] < rr[None, :, 3]))
        room_idx = np.where(inside.any(axis=1), np.argmax(inside, axis=1), -1)
        for k, name in enumerate(self.room_names):
            sel = room_idx == k
            if not sel.any():
                continue
            cx, cy = (rr[k, 0] + rr[k, 2]) / 2, (rr[k, 1] + rr[k, 3]) / 2
            L = np.array([cx, cy, self.height - 0.1])
            gain = light.ceiling * light.room_gain.get(name, 1.0)
            irr[sel] += gain * self._point(p[sel], n[sel], L, 2.8)
            for pos, inten, room in light.lamps:
                if room == name:
                    irr[sel] += inten * self._point(p[sel], n[sel], np.asarray(pos), 1.3)
        return irr

    @staticmethod
    def _point(p, n, L, falloff):
        d = L[None, :] - p
        dist = np.linalg.norm(d, axis=1)
        cosang = np.clip(np.sum(n * d, axis=1) / np.maximum(dist, 1e-6), 0, 1)
        return cosang / (1 + (dist / falloff) ** 2)
