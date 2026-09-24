"""Procedural multi-room apartments for the 3D test harness.

A scene is a set of objects, each made of oriented boxes (yaw-only rotation).
Boxes are enough to model walls, floors, furniture, small items, wall-mounted
items and thin "decals" (stains, scuffs) while keeping CPU ray casting fast.
Every object carries an integer id, which the renderer writes into an id map
so the harness knows exactly which pixels show which object.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------


@dataclass
class Material:
    color: tuple
    pattern: str = "plain"      # plain|noise|stripes|tiles|planks|books|fabric|canvas|window|mirror|screen
    color2: tuple | None = None
    scale: float = 0.2
    contrast: float = 0.12
    seed: int = 0
    color3: tuple | None = None


@dataclass
class Part:
    center: np.ndarray          # object-local
    half: np.ndarray            # half extents
    material: int


@dataclass
class SceneObject:
    oid: int
    name: str
    category: str               # structure|furniture|item|wall_item|decal
    room: str
    parts: list
    position: np.ndarray        # world position of the object origin (footprint centre, bottom)
    yaw: float
    movable: bool = False
    support: int | None = None  # oid of the object it rests on
    obstacle: bool = True       # blocks walking paths
    tags: dict = field(default_factory=dict)
    top: tuple | None = None    # local (x0, y0, x1, y1, z) surface that can hold items

    def world_boxes(self):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        R = np.array([[c, -s], [s, c]])
        out = []
        for p in self.parts:
            ctr = np.array(p.center, dtype=np.float64)
            w = self.position.astype(np.float64).copy()
            w[:2] += R @ ctr[:2]
            w[2] += ctr[2]
            out.append((w, np.asarray(p.half, dtype=np.float64), self.yaw, p.material, ctr))
        return out

    def aabb(self):
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for w, h, yaw, _, _ in self.world_boxes():
            c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
            ext = np.array([c * h[0] + s * h[1], s * h[0] + c * h[1], h[2]])
            lo = np.minimum(lo, w - ext)
            hi = np.maximum(hi, w + ext)
        return lo, hi

    def top_world(self):
        """World-space support surface: (corners (4,2), z) or None."""
        if self.top is None:
            return None
        x0, y0, x1, y1, z = self.top
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        R = np.array([[c, -s], [s, c]])
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) @ R.T + self.position[:2]
        return corners, self.position[2] + z


@dataclass
class Room:
    name: str
    x0: float
    y0: float
    x1: float
    y1: float
    interior: tuple = (0, 0, 0, 0)

    @property
    def center(self):
        return np.array([(self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2])

    def contains(self, x, y):
        return (x >= self.x0) & (x < self.x1) & (y >= self.y0) & (y < self.y1)


@dataclass
class Doorway:
    rooms: tuple
    axis: str          # 'h': wall along x at y=c ; 'v': wall along y at x=c
    c: float
    pos: float
    width: float
    height: float = 2.1

    def center_xy(self):
        return np.array([self.pos, self.c]) if self.axis == "h" else np.array([self.c, self.pos])


@dataclass
class Scene:
    width: float
    depth: float
    height: float
    rooms: list
    doorways: list
    objects: dict
    materials: list
    entry: np.ndarray                    # a standing position just inside the front door
    seed: int = 0

    def copy(self) -> "Scene":
        return copy.deepcopy(self)

    def room(self, name: str) -> Room:
        for r in self.rooms:
            if r.name == name:
                return r
        raise KeyError(name)

    def room_at(self, x: float, y: float) -> str | None:
        for r in self.rooms:
            if r.x0 <= x < r.x1 and r.y0 <= y < r.y1:
                return r.name
        return None

    def next_oid(self) -> int:
        return max(self.objects) + 1 if self.objects else 1

    def add_material(self, m: Material) -> int:
        self.materials.append(m)
        return len(self.materials) - 1

    def box_arrays(self):
        """Flatten all objects into ray-casting arrays."""
        centers, halves, yaws, mats, oids, local = [], [], [], [], [], []
        for obj in self.objects.values():
            for w, h, yaw, mat, ctr in obj.world_boxes():
                centers.append(w)
                halves.append(h)
                yaws.append(yaw)
                mats.append(mat)
                oids.append(obj.oid)
                local.append(ctr)
        return {
            "center": np.asarray(centers, dtype=np.float64),
            "half": np.asarray(halves, dtype=np.float64),
            "yaw": np.asarray(yaws, dtype=np.float64),
            "material": np.asarray(mats, dtype=np.int64),
            "oid": np.asarray(oids, dtype=np.int64),
            "local": np.asarray(local, dtype=np.float64),
        }

    def summary(self) -> list:
        out = []
        for o in self.objects.values():
            lo, hi = o.aabb()
            out.append({"oid": o.oid, "name": o.name, "category": o.category, "room": o.room,
                        "aabb": [lo.round(3).tolist(), hi.round(3).tolist()],
                        "movable": o.movable, "support": o.support})
        return out


# ----------------------------------------------------------------------------
# Palettes
# ----------------------------------------------------------------------------

WOOD = [(0.55, 0.38, 0.22), (0.66, 0.48, 0.30), (0.40, 0.27, 0.16), (0.76, 0.61, 0.43), (0.30, 0.21, 0.14)]
FABRIC = [(0.33, 0.40, 0.56), (0.58, 0.30, 0.28), (0.38, 0.50, 0.40), (0.62, 0.58, 0.50), (0.30, 0.30, 0.33),
          (0.72, 0.62, 0.42), (0.52, 0.44, 0.60), (0.24, 0.42, 0.46)]
PAINT = [(0.90, 0.88, 0.82), (0.80, 0.86, 0.89), (0.89, 0.84, 0.75), (0.79, 0.84, 0.77), (0.92, 0.90, 0.88),
         (0.86, 0.80, 0.82)]
METAL = [(0.84, 0.85, 0.86), (0.26, 0.26, 0.28), (0.70, 0.72, 0.74)]
BRIGHT = [(0.86, 0.26, 0.20), (0.20, 0.45, 0.82), (0.95, 0.76, 0.16), (0.24, 0.66, 0.36), (0.90, 0.50, 0.15),
          (0.60, 0.25, 0.65)]
CERAMIC = (0.93, 0.93, 0.91)


def pick(rng, palette):
    return tuple(float(c) for c in palette[int(rng.integers(len(palette)))])


def jitter(rng, color, amount=0.05):
    c = np.clip(np.asarray(color) * (1 + rng.uniform(-amount, amount, 3)), 0.02, 0.98)
    return tuple(float(x) for x in c)


# ----------------------------------------------------------------------------
# Object catalog (local frame: origin at footprint centre on the support surface,
# x = width, y = depth with the front facing -y, z up)
# ----------------------------------------------------------------------------


def _box(cx, cy, cz, hx, hy, hz, mat):
    return Part(np.array([cx, cy, cz], dtype=np.float64), np.array([hx, hy, hz], dtype=np.float64), mat)


def _table(w, d, h, top_mat, leg_mat, top_th=0.04, leg=0.05):
    parts = [_box(0, 0, h - top_th / 2, w / 2, d / 2, top_th / 2, top_mat)]
    lh = (h - top_th) / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(_box(sx * (w / 2 - leg), sy * (d / 2 - leg), lh, leg / 2, leg / 2, lh, leg_mat))
    return parts


class Catalog:
    """Builds object geometry + materials. All sizes in metres."""

    def __init__(self, scene: Scene, rng: np.random.Generator):
        self.s = scene
        self.rng = rng

    def mat(self, color, pattern="plain", **kw) -> int:
        kw.setdefault("seed", int(self.rng.integers(1 << 30)))
        return self.s.add_material(Material(tuple(color), pattern, **kw))

    # Each builder returns (parts, (w, d, h), top) -------------------------
    def sofa(self):
        r = self.rng
        w, d = r.uniform(1.8, 2.3), 0.92
        m = self.mat(pick(r, FABRIC), "fabric", contrast=0.10, scale=0.05)
        parts = [_box(0, 0, 0.2, w / 2, d / 2, 0.2, m), _box(0, d / 2 - 0.11, 0.6, w / 2, 0.11, 0.2, m)]
        for sx in (-1, 1):
            parts.append(_box(sx * (w / 2 - 0.1), -0.02, 0.5, 0.1, d / 2 - 0.02, 0.1, m))
        return parts, (w, d, 0.8), (-w / 2 + 0.22, -d / 2 + 0.05, w / 2 - 0.22, d / 2 - 0.24, 0.4)

    def armchair(self):
        r = self.rng
        w = d = 0.86
        m = self.mat(pick(r, FABRIC), "fabric", contrast=0.10, scale=0.05)
        parts = [_box(0, 0, 0.21, w / 2, d / 2, 0.21, m), _box(0, d / 2 - 0.1, 0.6, w / 2, 0.1, 0.19, m)]
        for sx in (-1, 1):
            parts.append(_box(sx * (w / 2 - 0.09), -0.02, 0.52, 0.09, d / 2 - 0.02, 0.1, m))
        return parts, (w, d, 0.8), None

    def coffee_table(self):
        r = self.rng
        w, d, h = r.uniform(1.0, 1.25), r.uniform(0.55, 0.7), 0.44
        tm = self.mat(pick(r, WOOD), "planks", scale=0.12, contrast=0.12)
        return _table(w, d, h, tm, tm), (w, d, h), (-w / 2 + 0.05, -d / 2 + 0.05, w / 2 - 0.05, d / 2 - 0.05, h)

    def side_table(self):
        r = self.rng
        w = d = r.uniform(0.42, 0.52)
        h = 0.56
        tm = self.mat(pick(r, WOOD), "planks", scale=0.1, contrast=0.1)
        return _table(w, d, h, tm, tm), (w, d, h), (-w / 2 + 0.04, -d / 2 + 0.04, w / 2 - 0.04, d / 2 - 0.04, h)

    def tv_stand(self):
        r = self.rng
        w, d, h = r.uniform(1.4, 1.8), 0.45, 0.5
        m = self.mat(pick(r, WOOD), "stripes", color2=jitter(r, pick(r, WOOD), 0.2), scale=0.4, contrast=0.2)
        return [_box(0, 0, h / 2, w / 2, d / 2, h / 2, m)], (w, d, h), (-w / 2 + 0.1, -d / 2 + 0.08, w / 2 - 0.1, d / 2 - 0.05, h)

    def tv(self):
        m = self.mat((0.05, 0.05, 0.06), "screen", color2=(0.14, 0.16, 0.2))
        base = self.mat((0.12, 0.12, 0.13))
        w = self.rng.uniform(1.0, 1.3)
        return [_box(0, 0, 0.015, 0.16, 0.1, 0.015, base), _box(0, 0, 0.1, 0.03, 0.03, 0.08, base),
                _box(0, 0, 0.18 + 0.33, w / 2, 0.03, 0.33, m)], (w, 0.2, 0.84), None

    def bookshelf(self):
        r = self.rng
        w, d, h = r.uniform(0.8, 1.1), 0.34, r.uniform(1.6, 1.95)
        m = self.mat(pick(r, WOOD), "books", scale=0.35)
        return [_box(0, 0, h / 2, w / 2, d / 2, h / 2, m)], (w, d, h), None

    def rug(self, w=None, d=None):
        r = self.rng
        w = w or r.uniform(1.8, 2.5)
        d = d or r.uniform(1.3, 1.7)
        m = self.mat(pick(r, FABRIC), r.choice(["stripes", "noise"]), color2=jitter(r, pick(r, FABRIC), 0.1),
                     scale=r.uniform(0.15, 0.35), contrast=0.2)
        return [_box(0, 0, 0.006, w / 2, d / 2, 0.006, m)], (w, d, 0.012), (-w / 2, -d / 2, w / 2, d / 2, 0.012)

    def floor_lamp(self):
        r = self.rng
        metal = self.mat(pick(r, METAL))
        shade = self.mat(jitter(r, (0.92, 0.88, 0.76), 0.04), "fabric", scale=0.03, contrast=0.05)
        return [_box(0, 0, 0.015, 0.15, 0.15, 0.015, metal), _box(0, 0, 0.78, 0.02, 0.02, 0.75, metal),
                _box(0, 0, 1.5, 0.18, 0.18, 0.15, shade)], (0.36, 0.36, 1.65), None

    def plant(self):
        r = self.rng
        pot = self.mat(pick(r, [(0.72, 0.42, 0.28), CERAMIC, (0.2, 0.2, 0.22)]))
        leaf = self.mat((0.18, 0.42, 0.16), "noise", color2=(0.35, 0.62, 0.22), scale=0.06, contrast=0.5)
        s = r.uniform(0.9, 1.15)
        return [_box(0, 0, 0.17 * s, 0.16 * s, 0.16 * s, 0.17 * s, pot),
                _box(0, 0, (0.34 + 0.3) * s, 0.27 * s, 0.27 * s, 0.3 * s, leaf)], (0.55 * s, 0.55 * s, 0.94 * s), None

    def painting(self):
        r = self.rng
        w, h = r.uniform(0.6, 1.0), r.uniform(0.45, 0.8)
        m = self.canvas_material()
        frame = self.mat(pick(r, WOOD + [(0.1, 0.1, 0.1), (0.9, 0.9, 0.88)]))
        return [_box(0, 0, 0, w / 2 + 0.03, 0.012, h / 2 + 0.03, frame), _box(0, -0.006, 0, w / 2, 0.012, h / 2, m)], \
            (w + 0.06, 0.03, h + 0.06), None

    def canvas_material(self):
        r = self.rng
        return self.mat(pick(r, BRIGHT + FABRIC), "canvas", color2=pick(r, BRIGHT + PAINT), color3=pick(r, BRIGHT),
                        scale=r.uniform(0.12, 0.3), contrast=0.6)

    def mirror(self):
        m = self.mat((0.78, 0.82, 0.86), "mirror", color2=(0.6, 0.64, 0.7), scale=0.3)
        frame = self.mat(pick(self.rng, METAL))
        return [_box(0, 0, 0, 0.33, 0.012, 0.48, frame), _box(0, -0.006, 0, 0.3, 0.012, 0.45, m)], (0.66, 0.03, 0.96), None

    def cushion(self):
        m = self.mat(pick(self.rng, BRIGHT + FABRIC), "fabric", scale=0.03, contrast=0.12)
        return [_box(0, 0, 0.2, 0.2, 0.07, 0.2, m)], (0.4, 0.14, 0.4), None

    def vase(self):
        m = self.mat(pick(self.rng, BRIGHT + [CERAMIC]), "plain", contrast=0.05)
        return [_box(0, 0, 0.14, 0.07, 0.07, 0.14, m)], (0.14, 0.14, 0.28), None

    def counter(self, length):
        r = self.rng
        body = self.mat(pick(r, WOOD + [(0.9, 0.9, 0.88), (0.3, 0.36, 0.42)]), "stripes",
                        color2=(0.2, 0.2, 0.2), scale=0.6, contrast=0.08)
        top = self.mat(pick(r, [(0.85, 0.84, 0.8), (0.25, 0.25, 0.26), (0.62, 0.6, 0.56)]), "noise",
                       color2=(0.5, 0.5, 0.5), scale=0.05, contrast=0.12)
        d, h = 0.62, 0.9
        parts = [_box(0, 0.02, 0.43, length / 2, d / 2 - 0.02, 0.43, body), _box(0, 0, 0.88, length / 2, d / 2, 0.02, top)]
        return parts, (length, d, h), (-length / 2 + 0.05, -d / 2 + 0.05, length / 2 - 0.05, d / 2 - 0.1, h)

    def upper_cabinets(self, length):
        m = self.mat(pick(self.rng, WOOD + [(0.9, 0.9, 0.88)]), "stripes", color2=(0.3, 0.3, 0.3), scale=0.6,
                     contrast=0.1)
        return [_box(0, 0, 1.8, length / 2, 0.17, 0.35, m)], (length, 0.34, 2.15), None

    def fridge(self):
        m = self.mat(pick(self.rng, METAL), "stripes", color2=(0.5, 0.5, 0.52), scale=1.2, contrast=0.1)
        return [_box(0, 0, 0.92, 0.4, 0.35, 0.92, m)], (0.8, 0.7, 1.84), None

    def dining_table(self):
        r = self.rng
        w, d, h = r.uniform(1.2, 1.6), r.uniform(0.8, 0.95), 0.75
        tm = self.mat(pick(r, WOOD), "planks", scale=0.14, contrast=0.12)
        return _table(w, d, h, tm, tm), (w, d, h), (-w / 2 + 0.08, -d / 2 + 0.08, w / 2 - 0.08, d / 2 - 0.08, h)

    def chair(self, mat=None):
        m = mat if mat is not None else self.mat(pick(self.rng, WOOD + FABRIC))
        parts = [_box(0, 0, 0.455, 0.225, 0.225, 0.025, m), _box(0, 0.2, 0.7, 0.225, 0.025, 0.22, m)]
        for sx in (-1, 1):
            for sy in (-1, 1):
                parts.append(_box(sx * 0.2, sy * 0.2, 0.215, 0.02, 0.02, 0.215, m))
        return parts, (0.45, 0.45, 0.92), None

    def microwave(self):
        m = self.mat(pick(self.rng, METAL), "screen", color2=(0.1, 0.1, 0.1))
        return [_box(0, 0, 0.15, 0.25, 0.19, 0.15, m)], (0.5, 0.38, 0.3), None

    def kettle(self):
        m = self.mat(pick(self.rng, METAL + BRIGHT))
        return [_box(0, 0, 0.12, 0.09, 0.08, 0.12, m)], (0.18, 0.16, 0.24), None

    def toaster(self):
        m = self.mat(pick(self.rng, METAL + BRIGHT))
        return [_box(0, 0, 0.1, 0.14, 0.09, 0.1, m)], (0.28, 0.18, 0.2), None

    def fruit_bowl(self):
        bowl = self.mat(pick(self.rng, [CERAMIC, (0.3, 0.3, 0.32)]))
        fruit = self.mat((0.9, 0.55, 0.1), "noise", color2=(0.85, 0.15, 0.1), scale=0.04, contrast=0.8)
        return [_box(0, 0, 0.03, 0.15, 0.15, 0.03, bowl), _box(0, 0, 0.1, 0.1, 0.1, 0.04, fruit)], (0.3, 0.3, 0.14), None

    def trash_bin(self):
        m = self.mat(pick(self.rng, METAL + BRIGHT))
        return [_box(0, 0, 0.31, 0.17, 0.17, 0.31, m)], (0.34, 0.34, 0.62), None

    def console_table(self):
        r = self.rng
        w, d, h = r.uniform(0.9, 1.2), 0.34, 0.8
        tm = self.mat(pick(r, WOOD), "planks", scale=0.1)
        return _table(w, d, h, tm, tm), (w, d, h), (-w / 2 + 0.05, -d / 2 + 0.04, w / 2 - 0.05, d / 2 - 0.04, h)

    def key_bowl(self):
        m = self.mat(pick(self.rng, BRIGHT + [CERAMIC]))
        return [_box(0, 0, 0.03, 0.09, 0.09, 0.03, m)], (0.18, 0.18, 0.06), None

    def shoe_rack(self):
        m = self.mat(pick(self.rng, WOOD), "stripes", color2=(0.15, 0.15, 0.15), scale=0.17, contrast=0.3)
        return [_box(0, 0, 0.25, 0.4, 0.15, 0.25, m)], (0.8, 0.3, 0.5), (-0.36, -0.12, 0.36, 0.12, 0.5)

    def coat_stand(self):
        m = self.mat(pick(self.rng, WOOD + METAL))
        coat = self.mat(pick(self.rng, FABRIC), "fabric", scale=0.04)
        return [_box(0, 0, 0.02, 0.2, 0.2, 0.02, m), _box(0, 0, 0.9, 0.025, 0.025, 0.88, m),
                _box(0.08, 0, 1.35, 0.14, 0.12, 0.35, coat)], (0.45, 0.45, 1.8), None

    def umbrella_stand(self):
        m = self.mat(pick(self.rng, METAL + BRIGHT))
        return [_box(0, 0, 0.25, 0.12, 0.12, 0.25, m)], (0.25, 0.25, 0.5), None

    def bed(self):
        r = self.rng
        w, d = r.choice([1.4, 1.6, 1.8]), 2.1
        frame = self.mat(pick(r, WOOD), "planks", scale=0.2)
        sheet = self.mat(jitter(r, (0.93, 0.92, 0.9), 0.03), "fabric", scale=0.03, contrast=0.05)
        blanket = self.mat(pick(r, FABRIC), r.choice(["stripes", "noise", "fabric"]), color2=pick(r, FABRIC),
                           scale=r.uniform(0.1, 0.3), contrast=0.25)
        parts = [_box(0, 0, 0.15, w / 2, d / 2, 0.15, frame),
                 _box(0, -0.02, 0.41, w / 2 - 0.05, d / 2 - 0.07, 0.11, sheet),
                 _box(0, d / 2 - 0.04, 0.55, w / 2, 0.04, 0.55, frame),
                 _box(0, -0.33, 0.53, w / 2 - 0.04, d / 2 - 0.42, 0.012, blanket)]
        return parts, (w, d, 1.1), (-w / 2 + 0.12, d / 2 - 0.62, w / 2 - 0.12, d / 2 - 0.12, 0.52)

    def pillow(self):
        m = self.mat(jitter(self.rng, (0.94, 0.93, 0.9), 0.05), "fabric", scale=0.03, contrast=0.06)
        return [_box(0, 0, 0.07, 0.3, 0.2, 0.07, m)], (0.6, 0.4, 0.14), None

    def nightstand(self):
        m = self.mat(pick(self.rng, WOOD), "stripes", color2=(0.2, 0.15, 0.1), scale=0.28, contrast=0.12)
        return [_box(0, 0, 0.275, 0.24, 0.2, 0.275, m)], (0.48, 0.4, 0.55), (-0.2, -0.16, 0.2, 0.16, 0.55)

    def table_lamp(self):
        base = self.mat(pick(self.rng, [CERAMIC] + BRIGHT + METAL))
        shade = self.mat(jitter(self.rng, (0.93, 0.9, 0.8), 0.04), "fabric", scale=0.03, contrast=0.05)
        return [_box(0, 0, 0.14, 0.06, 0.06, 0.14, base), _box(0, 0, 0.4, 0.15, 0.15, 0.12, shade)], (0.3, 0.3, 0.52), None

    def wardrobe(self):
        r = self.rng
        w = r.uniform(1.0, 1.4)
        m = self.mat(pick(r, WOOD + [(0.9, 0.9, 0.88)]), "stripes", color2=(0.25, 0.2, 0.15), scale=w / 2, contrast=0.08)
        return [_box(0, 0, 1.0, w / 2, 0.3, 1.0, m)], (w, 0.6, 2.0), None

    def desk(self):
        r = self.rng
        w, d, h = r.uniform(1.0, 1.3), 0.6, 0.75
        m = self.mat(pick(r, WOOD + [(0.92, 0.92, 0.9)]), "planks", scale=0.15, contrast=0.08)
        parts = [_box(0, 0, h - 0.02, w / 2, d / 2, 0.02, m), _box(-w / 2 + 0.02, 0, (h - 0.04) / 2, 0.02, d / 2, (h - 0.04) / 2, m),
                 _box(w / 2 - 0.02, 0, (h - 0.04) / 2, 0.02, d / 2, (h - 0.04) / 2, m)]
        return parts, (w, d, h), (-w / 2 + 0.06, -d / 2 + 0.06, w / 2 - 0.06, d / 2 - 0.06, h)

    def laundry_basket(self):
        m = self.mat(pick(self.rng, [(0.8, 0.72, 0.55), (0.3, 0.5, 0.75), (0.9, 0.9, 0.9)]), "stripes",
                     color2=(0.5, 0.45, 0.35), scale=0.05, contrast=0.2)
        return [_box(0, 0, 0.21, 0.25, 0.18, 0.21, m)], (0.5, 0.36, 0.42), None

    def vanity(self):
        body = self.mat(pick(self.rng, WOOD + [(0.92, 0.92, 0.9)]), "stripes", color2=(0.2, 0.2, 0.2), scale=0.4, contrast=0.08)
        top = self.mat(CERAMIC, "noise", color2=(0.8, 0.8, 0.8), scale=0.05, contrast=0.08)
        return [_box(0, 0.02, 0.41, 0.4, 0.23, 0.41, body), _box(0, 0, 0.84, 0.42, 0.25, 0.02, top)], \
            (0.84, 0.5, 0.86), (-0.36, -0.2, 0.36, 0.05, 0.86)

    def toilet(self):
        m = self.mat(CERAMIC)
        return [_box(0, -0.06, 0.2, 0.19, 0.28, 0.2, m), _box(0, 0.22, 0.6, 0.21, 0.1, 0.2, m)], (0.42, 0.66, 0.8), None

    def bathtub(self):
        m = self.mat(CERAMIC, "noise", color2=(0.85, 0.87, 0.88), scale=0.3, contrast=0.05)
        return [_box(0, 0, 0.28, 0.85, 0.37, 0.28, m)], (1.7, 0.74, 0.56), None

    def towel(self):
        m = self.mat(pick(self.rng, BRIGHT + FABRIC), "stripes", color2=(0.95, 0.95, 0.95), scale=0.12, contrast=0.3)
        return [_box(0, 0, 0, 0.25, 0.02, 0.36, m)], (0.5, 0.04, 0.72), None

    def bath_mat(self):
        m = self.mat(pick(self.rng, FABRIC + BRIGHT), "fabric", scale=0.03, contrast=0.12)
        return [_box(0, 0, 0.006, 0.4, 0.26, 0.006, m)], (0.8, 0.52, 0.012), (-0.4, -0.26, 0.4, 0.26, 0.012)

    def toiletry(self):
        m = self.mat(pick(self.rng, BRIGHT + [CERAMIC]))
        return [_box(0, 0, 0.09, 0.035, 0.035, 0.09, m)], (0.07, 0.07, 0.18), None

    # -- things that appear between visits -------------------------------
    def suitcase(self):
        m = self.mat(pick(self.rng, BRIGHT + [(0.15, 0.15, 0.17)]), "stripes", color2=(0.1, 0.1, 0.1), scale=0.12, contrast=0.2)
        return [_box(0, 0, 0.33, 0.23, 0.13, 0.33, m)], (0.46, 0.26, 0.66), None

    def cardboard_box(self):
        m = self.mat((0.66, 0.5, 0.32), "noise", color2=(0.55, 0.4, 0.25), scale=0.1, contrast=0.2)
        return [_box(0, 0, 0.175, 0.25, 0.2, 0.175, m)], (0.5, 0.4, 0.35), (-0.2, -0.15, 0.2, 0.15, 0.35)

    def backpack(self):
        m = self.mat(pick(self.rng, BRIGHT + FABRIC), "fabric", scale=0.04, contrast=0.15)
        return [_box(0, 0, 0.22, 0.16, 0.1, 0.22, m)], (0.32, 0.2, 0.44), None

    def trash_bag(self):
        m = self.mat((0.08, 0.08, 0.09), "noise", color2=(0.2, 0.2, 0.22), scale=0.08, contrast=0.4)
        return [_box(0, 0, 0.3, 0.22, 0.22, 0.3, m)], (0.44, 0.44, 0.6), None

    def shoes(self):
        m = self.mat(pick(self.rng, BRIGHT + [(0.1, 0.1, 0.1), (0.95, 0.95, 0.95)]))
        return [_box(-0.07, 0, 0.055, 0.05, 0.14, 0.055, m), _box(0.07, 0.02, 0.055, 0.05, 0.14, 0.055, m)], (0.26, 0.3, 0.11), None

    def bottle(self):
        m = self.mat(pick(self.rng, [(0.2, 0.55, 0.3), (0.6, 0.35, 0.15), (0.3, 0.45, 0.8)]))
        return [_box(0, 0, 0.14, 0.04, 0.04, 0.14, m)], (0.08, 0.08, 0.28), None

    def pizza_box(self):
        m = self.mat((0.82, 0.74, 0.6), "noise", color2=(0.75, 0.15, 0.1), scale=0.12, contrast=0.3)
        return [_box(0, 0, 0.025, 0.2, 0.2, 0.025, m)], (0.4, 0.4, 0.05), None

    def toy(self):
        m = self.mat(pick(self.rng, BRIGHT), "tiles", color2=pick(self.rng, BRIGHT), scale=0.07, contrast=0.6)
        return [_box(0, 0, 0.1, 0.1, 0.1, 0.1, m)], (0.2, 0.2, 0.2), None

    def jacket(self):
        m = self.mat(pick(self.rng, FABRIC + BRIGHT), "fabric", scale=0.04, contrast=0.15)
        return [_box(0, 0, 0.04, 0.3, 0.24, 0.04, m)], (0.6, 0.48, 0.08), None


FLOOR_ITEMS_ADDED = ["suitcase", "cardboard_box", "backpack", "trash_bag", "shoes", "laundry_basket", "toy"]
SURFACE_ITEMS_ADDED = ["bottle", "pizza_box", "backpack", "toy", "kettle"]
SOFT_ITEMS_ADDED = ["jacket", "backpack"]


# ----------------------------------------------------------------------------
# Placement
# ----------------------------------------------------------------------------

SIDE_YAW = {"S": np.pi, "N": 0.0, "W": np.pi / 2, "E": -np.pi / 2}
SIDE_NORMAL = {"S": np.array([0, 1.0]), "N": np.array([0, -1.0]), "W": np.array([1.0, 0]), "E": np.array([-1.0, 0])}


def _rect_overlap(a, b, pad=0.0):
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


class Planner:
    def __init__(self, scene: Scene, room: Room, rng: np.random.Generator, keepouts, windows):
        self.scene, self.room, self.rng = scene, room, rng
        self.b = room.interior
        self.rects = []                 # floor obstacles (x0,y0,x1,y1)
        self.keepouts = list(keepouts)  # doorway clearances
        self.windows = windows          # side -> list of (lo, hi)
        self.wall_spans = {s: [] for s in "NSEW"}   # side -> list of (lo, hi, zlo, zhi)
        self.cat = Catalog(scene, rng)

    def side_range(self, side):
        x0, y0, x1, y1 = self.b
        return (x0, x1) if side in "SN" else (y0, y1)

    def rect_for(self, x, y, w, d, yaw):
        if abs(np.sin(yaw)) > 0.5:
            w, d = d, w
        return (x - w / 2, y - d / 2, x + w / 2, y + d / 2)

    def is_free(self, rect, tall=False, side=None, pad=0.05):
        x0, y0, x1, y1 = self.b
        if rect[0] < x0 - 1e-6 or rect[1] < y0 - 1e-6 or rect[2] > x1 + 1e-6 or rect[3] > y1 + 1e-6:
            return False
        for r in self.rects:
            if _rect_overlap(rect, r, pad):
                return False
        for r in self.keepouts:
            if _rect_overlap(rect, r):
                return False
        if tall and side is not None:
            lo, hi = (rect[0], rect[2]) if side in "SN" else (rect[1], rect[3])
            for wlo, whi in self.windows.get(side, []):
                if not (hi <= wlo or whi <= lo):
                    return False
        return True

    def wall_pos(self, side, along, d, gap=0.01):
        x0, y0, x1, y1 = self.b
        if side == "S":
            return along, y0 + d / 2 + gap
        if side == "N":
            return along, y1 - d / 2 - gap
        if side == "W":
            return x0 + d / 2 + gap, along
        return x1 - d / 2 - gap, along

    def against_wall(self, size, sides=None, tries=60, along=None, reserve=True):
        w, d, h = size
        sides = list(sides) if sides else list("NSEW")
        for _ in range(tries):
            side = sides[int(self.rng.integers(len(sides)))]
            lo, hi = self.side_range(side)
            if hi - lo < w + 0.02:
                continue
            a = along if along is not None else self.rng.uniform(lo + w / 2, hi - w / 2)
            x, y = self.wall_pos(side, a, d)
            yaw = SIDE_YAW[side]
            rect = self.rect_for(x, y, w, d, yaw)
            if self.is_free(rect, tall=h > 1.0, side=side):
                if reserve:
                    self.rects.append(rect)
                return np.array([x, y, 0.0]), yaw, side
        return None

    def at(self, x, y, size, yaw, reserve=True, pad=0.05):
        w, d, h = size
        rect = self.rect_for(x, y, w, d, yaw)
        if self.is_free(rect, pad=pad):
            if reserve:
                self.rects.append(rect)
            return np.array([x, y, 0.0]), yaw
        return None

    def random_free(self, size, tries=80, margin=0.25, yaw=None):
        w, d, h = size
        x0, y0, x1, y1 = self.b
        for _ in range(tries):
            yw = yaw if yaw is not None else float(self.rng.choice([0, np.pi / 2, np.pi, -np.pi / 2]))
            ww, dd = (d, w) if abs(np.sin(yw)) > 0.5 else (w, d)
            if x1 - x0 < ww + 2 * margin or y1 - y0 < dd + 2 * margin:
                continue
            x = self.rng.uniform(x0 + ww / 2 + margin, x1 - ww / 2 - margin)
            y = self.rng.uniform(y0 + dd / 2 + margin, y1 - dd / 2 - margin)
            r = self.at(x, y, size, yw)
            if r is not None:
                return r
        return None

    def wall_item(self, size, z_center, sides=None, along=None, tries=60):
        """Place a flat item on a wall surface. Returns (position, yaw, side) or None."""
        w, d, h = size
        sides = list(sides) if sides else list("NSEW")
        for _ in range(tries):
            side = sides[int(self.rng.integers(len(sides)))]
            lo, hi = self.side_range(side)
            if hi - lo < w + 0.3:
                continue
            a = along if along is not None else self.rng.uniform(lo + w / 2 + 0.1, hi - w / 2 - 0.1)
            span = (a - w / 2, a + w / 2)
            zlo, zhi = z_center - h / 2, z_center + h / 2
            clash = False
            for (l2, h2, z2l, z2h) in self.wall_spans[side]:
                if not (span[1] + 0.1 <= l2 or h2 + 0.1 <= span[0]) and not (zhi <= z2l or z2h <= zlo):
                    clash = True
            for wlo, whi in self.windows.get(side, []):
                if not (span[1] <= wlo or whi <= span[0]) and zhi > 0.8:
                    clash = True
            for dw in self.room_doorways(side):
                if not (span[1] + 0.05 <= dw[0] or dw[1] + 0.05 <= span[0]):
                    clash = True
            if clash:
                if along is not None:
                    return None
                continue
            x, y = self.wall_pos(side, a, d, gap=0.0)
            self.wall_spans[side].append((span[0], span[1], zlo, zhi))
            return np.array([x, y, z_center]), SIDE_YAW[side], side
        return None

    def room_doorways(self, side):
        out = []
        r = self.room
        for dw in self.scene.doorways:
            if r.name not in dw.rooms:
                continue
            if dw.axis == "h" and side in "SN":
                wall = r.y0 if side == "S" else r.y1
                if abs(wall - dw.c) < 1e-6:
                    out.append((dw.pos - dw.width / 2, dw.pos + dw.width / 2))
            if dw.axis == "v" and side in "WE":
                wall = r.x0 if side == "W" else r.x1
                if abs(wall - dw.c) < 1e-6:
                    out.append((dw.pos - dw.width / 2, dw.pos + dw.width / 2))
        return out


# ----------------------------------------------------------------------------
# Scene generation
# ----------------------------------------------------------------------------


def _add(scene: Scene, name, category, room, built, position, yaw, movable=False, support=None, obstacle=True,
         tags=None) -> SceneObject:
    parts, size, top = built
    oid = scene.next_oid()
    obj = SceneObject(oid, name, category, room, parts, np.asarray(position, dtype=np.float64), float(yaw),
                      movable=movable, support=support, obstacle=obstacle, tags=dict(tags or {}), top=top)
    obj.tags.setdefault("size", [float(s) for s in size])
    scene.objects[oid] = obj
    return obj


def place_on(scene: Scene, support: SceneObject, built, name, rng, room=None, category="item", movable=True,
             tries=40, margin=0.02, yaw=None):
    """Place an item on the support surface of another object."""
    tw = support.top_world()
    if tw is None:
        return None
    parts, size, top = built
    w, d, h = size
    x0, y0, x1, y1, z = support.top
    rad = 0.5 * np.hypot(w, d)
    others = [o for o in scene.objects.values() if o.support == support.oid]
    c, s = np.cos(support.yaw), np.sin(support.yaw)
    R = np.array([[c, -s], [s, c]])
    if (x1 - x0) < min(w, d) or (y1 - y0) < min(w, d):
        return None
    for _ in range(tries):
        lx = rng.uniform(x0 + rad * 0.7, x1 - rad * 0.7) if x1 - x0 > 1.4 * rad else (x0 + x1) / 2
        ly = rng.uniform(y0 + rad * 0.7, y1 - rad * 0.7) if y1 - y0 > 1.4 * rad else (y0 + y1) / 2
        p = support.position[:2] + R @ np.array([lx, ly])
        ok = True
        for o in others:
            if np.linalg.norm(o.position[:2] - p) < rad + 0.5 * np.hypot(*o.tags["size"][:2]) + 0.03:
                ok = False
                break
        if ok:
            yw = support.yaw + (rng.uniform(-0.6, 0.6) if yaw is None else yaw)
            pos = np.array([p[0], p[1], support.position[2] + z])
            return _add(scene, name, category, room or support.room, built, pos, yw, movable=movable,
                        support=support.oid, obstacle=False)
    return None


def _wall_linings(scene: Scene, room: Room, thickness: dict, mat: int):
    H = scene.height
    for side in "NSEW":
        t = thickness[side]
        if side in "SN":
            lo, hi = room.x0, room.x1
            wall = room.y0 if side == "S" else room.y1
        else:
            lo, hi = room.y0, room.y1
            wall = room.x0 if side == "W" else room.x1
        openings = []
        for dw in scene.doorways:
            if room.name not in dw.rooms:
                continue
            if (dw.axis == "h") == (side in "SN") and abs(dw.c - wall) < 1e-6:
                openings.append((dw.pos - dw.width / 2, dw.pos + dw.width / 2, dw.height))
        openings.sort()
        segs = []
        cur = lo
        for a, b, hgt in openings:
            segs.append((cur, a, 0.0, H))
            segs.append((a, b, hgt, H))  # lintel
            cur = b
        segs.append((cur, hi, 0.0, H))
        inward = +1 if side in "SW" else -1
        parts = []
        for a, b, z0, z1 in segs:
            if b - a < 1e-3:
                continue
            mid = (a + b) / 2
            off = wall + inward * t / 2
            if side in "SN":
                c = (mid, off, (z0 + z1) / 2)
                h = ((b - a) / 2, t / 2, (z1 - z0) / 2)
            else:
                c = (off, mid, (z0 + z1) / 2)
                h = (t / 2, (b - a) / 2, (z1 - z0) / 2)
            parts.append(_box(c[0], c[1], c[2], h[0], h[1], h[2], mat))
        _add(scene, f"wall_{room.name}_{side}", "structure", room.name, (parts, (0, 0, H), None),
             np.zeros(3), 0.0, obstacle=False, tags={"surface": "wall", "side": side})


def generate_scene(seed: int, height: float = 2.6) -> Scene:
    rng = np.random.default_rng(seed)
    W = float(rng.uniform(9.0, 11.0))
    D = float(rng.uniform(7.2, 8.4))
    xs = W * float(rng.uniform(0.50, 0.57))
    ysl = D * float(rng.uniform(0.55, 0.62))
    ysr1 = D * float(rng.uniform(0.29, 0.34))
    ysr2 = ysr1 + D * float(rng.uniform(0.38, 0.43))
    rooms = [Room("living", 0, 0, xs, ysl), Room("kitchen", 0, ysl, xs, D), Room("hallway", xs, 0, W, ysr1),
             Room("bedroom", xs, ysr1, W, ysr2), Room("bathroom", xs, ysr2, W, D)]
    doorways = [
        Doorway(("living", "kitchen"), "h", ysl, float(rng.uniform(1.1, xs - 1.1)), 1.2),
        Doorway(("living", "hallway"), "v", xs, float(rng.uniform(0.75, ysr1 - 0.75)), 0.9),
        Doorway(("hallway", "bedroom"), "h", ysr1, float(rng.uniform(xs + 0.8, W - 0.8)), 0.9),
        Doorway(("bedroom", "bathroom"), "h", ysr2, float(rng.uniform(xs + 0.8, W - 0.8)), 0.85),
    ]
    flip_x, flip_y = bool(rng.integers(2)), bool(rng.integers(2))

    def fx(x):
        return W - x if flip_x else x

    def fy(y):
        return D - y if flip_y else y

    for r in rooms:
        r.x0, r.x1 = sorted((fx(r.x0), fx(r.x1)))
        r.y0, r.y1 = sorted((fy(r.y0), fy(r.y1)))
    for dw in doorways:
        if dw.axis == "h":
            dw.c, dw.pos = fy(dw.c), fx(dw.pos)
        else:
            dw.c, dw.pos = fx(dw.c), fy(dw.pos)

    scene = Scene(W, D, height, rooms, doorways, {}, [], np.zeros(3), seed=seed)
    cat = Catalog(scene, rng)

    # --- structure --------------------------------------------------------
    ceiling = cat.mat((0.95, 0.95, 0.94), "plain", contrast=0.03)
    _add(scene, "ceiling", "structure", "all", ([_box(W / 2, D / 2, height + 0.05, W / 2, D / 2, 0.05, ceiling)],
                                                 (W, D, 0.1), None), np.zeros(3), 0.0, obstacle=False,
         tags={"surface": "ceiling"})
    floor_style = {
        "living": ("planks", pick(rng, WOOD), 0.16),
        "kitchen": ("tiles", (0.86, 0.85, 0.82), 0.4),
        "hallway": ("planks", pick(rng, WOOD), 0.14),
        "bedroom": ("fabric", pick(rng, [(0.7, 0.66, 0.6), (0.55, 0.58, 0.62), (0.62, 0.55, 0.5)]), 0.04),
        "bathroom": ("tiles", pick(rng, [(0.8, 0.84, 0.86), (0.9, 0.9, 0.88), (0.45, 0.5, 0.55)]), 0.2),
    }
    thick = {}
    for r in rooms:
        pat, col, sc = floor_style[r.name]
        fm = cat.mat(col, pat, color2=(0.3, 0.3, 0.3) if pat == "tiles" else jitter(rng, col, 0.25), scale=sc,
                     contrast=0.18 if pat != "fabric" else 0.08)
        _add(scene, f"floor_{r.name}", "structure", r.name,
             ([_box((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, -0.05, (r.x1 - r.x0) / 2, (r.y1 - r.y0) / 2, 0.05, fm)],
              (r.x1 - r.x0, r.y1 - r.y0, 0.1), None), np.zeros(3), 0.0, obstacle=False, tags={"surface": "floor"})
        t = {"S": 0.1 if abs(r.y0) < 1e-6 else 0.06, "N": 0.1 if abs(r.y1 - D) < 1e-6 else 0.06,
             "W": 0.1 if abs(r.x0) < 1e-6 else 0.06, "E": 0.1 if abs(r.x1 - W) < 1e-6 else 0.06}
        thick[r.name] = t
        r.interior = (r.x0 + t["W"], r.y0 + t["S"], r.x1 - t["E"], r.y1 - t["N"])
        paint = cat.mat(pick(rng, PAINT), "plain", contrast=0.05)
        _wall_linings(scene, r, t, paint)

    # windows on exterior sides
    windows = {r.name: {} for r in rooms}
    for r in rooms:
        t = thick[r.name]
        x0, y0, x1, y1 = r.interior
        for side in "NSEW":
            if t[side] < 0.09:
                continue
            lo, hi = (x0, x1) if side in "SN" else (y0, y1)
            n = 1 if hi - lo < 3.5 else int(rng.integers(1, 3))
            if r.name == "bathroom":
                n = 1
            for k in range(n):
                ww = float(rng.uniform(0.8, 1.4)) if r.name != "bathroom" else 0.6
                seg = (hi - lo) / n
                a = lo + seg * (k + 0.5) + float(rng.uniform(-0.2, 0.2)) * seg
                a = float(np.clip(a, lo + ww / 2 + 0.3, hi - ww / 2 - 0.3))
                if hi - lo < ww + 0.7:
                    continue
                zc = 1.5 if r.name != "bathroom" else 1.75
                hh = 1.2 if r.name != "bathroom" else 0.6
                gm = cat.mat((0.78, 0.86, 0.95), "window", color2=(0.92, 0.94, 0.97), scale=ww)
                fr = cat.mat((0.95, 0.95, 0.95))
                parts = [_box(0, 0, 0, ww / 2 + 0.05, 0.015, hh / 2 + 0.05, fr), _box(0, -0.01, 0, ww / 2, 0.015, hh / 2, gm)]
                if side in "SN":
                    px, py = a, (y0 + 0.015 if side == "S" else y1 - 0.015)
                else:
                    px, py = (x0 + 0.015 if side == "W" else x1 - 0.015), a
                _add(scene, f"window_{r.name}_{side}{k}", "structure", r.name, (parts, (ww, 0.03, hh), None),
                     np.array([px, py, zc]), SIDE_YAW[side], obstacle=False, tags={"surface": "wall"})
                windows[r.name].setdefault(side, []).append((a - ww / 2 - 0.1, a + ww / 2 + 0.1))

    # front door on an exterior side of the hallway
    hall = scene.room("hallway")
    ht = thick["hallway"]
    ext_sides = [s for s in "NSEW" if ht[s] > 0.09]
    door_side = ext_sides[int(rng.integers(len(ext_sides)))]
    hx0, hy0, hx1, hy1 = hall.interior
    lo, hi = (hx0, hx1) if door_side in "SN" else (hy0, hy1)
    busy = windows["hallway"].get(door_side, [])
    for _ in range(50):
        a = float(rng.uniform(lo + 0.7, hi - 0.7))
        if all(a + 0.6 <= wlo or whi <= a - 0.6 for wlo, whi in busy):
            break
    windows["hallway"].setdefault(door_side, []).append((a - 0.6, a + 0.6))
    dm = cat.mat(pick(rng, WOOD), "planks", scale=0.2, contrast=0.1)
    if door_side in "SN":
        px, py = a, (hy0 + 0.02 if door_side == "S" else hy1 - 0.02)
    else:
        px, py = (hx0 + 0.02 if door_side == "W" else hx1 - 0.02), a
    _add(scene, "front_door", "structure", "hallway", ([_box(0, 0, 1.05, 0.48, 0.02, 1.05, dm)], (0.96, 0.04, 2.1), None),
         np.array([px, py, 0.0]), SIDE_YAW[door_side], obstacle=False, tags={"surface": "wall"})
    n = SIDE_NORMAL[door_side]
    scene.entry = np.array([px + n[0] * 0.9, py + n[1] * 0.9, 0.0])

    # --- furniture --------------------------------------------------------
    for r in rooms:
        keep = []
        for dw in doorways:
            if r.name not in dw.rooms:
                continue
            c = dw.center_xy()
            hw = dw.width / 2 + 0.25
            if dw.axis == "h":
                keep.append((c[0] - hw, c[1] - 0.85, c[0] + hw, c[1] + 0.85))
            else:
                keep.append((c[0] - 0.85, c[1] - hw, c[0] + 0.85, c[1] + hw))
        if r.name == "hallway":
            e = scene.entry
            keep.append((e[0] - 0.8, e[1] - 0.8, e[0] + 0.8, e[1] + 0.8))
        pl = Planner(scene, r, rng, keep, windows[r.name])
        FURNISH[r.name](scene, pl, rng)
    return scene


# ----------------------------------------------------------------------------
# Room furnishing
# ----------------------------------------------------------------------------


def _front(yaw):
    return np.array([np.sin(yaw), -np.cos(yaw)])


def _along(pos, side):
    return float(pos[0] if side in "SN" else pos[1])


def _wall_obj(scene, pl, builder, name, z, room, sides=None, along=None, movable=True):
    b = builder()
    p = pl.wall_item(b[1], z, sides=sides, along=along)
    if p is None and along is not None:
        p = pl.wall_item(b[1], z)
    if p is None:
        return None
    return _add(scene, name, "wall_item", room, b, p[0], p[1], movable=movable, obstacle=False,
                tags={"side": p[2]})


def _inside(pl, rect, margin=0.05):
    x0, y0, x1, y1 = pl.b
    return rect[0] >= x0 + margin and rect[1] >= y0 + margin and rect[2] <= x1 - margin and rect[3] <= y1 - margin


def furnish_living(scene, pl: Planner, rng):
    cat, room = pl.cat, pl.room.name
    x0, y0, x1, y1 = pl.b
    long_sides = "SN" if (x1 - x0) >= (y1 - y0) else "WE"
    sofa_b = cat.sofa()
    got = pl.against_wall(sofa_b[1], sides=long_sides) or pl.against_wall(sofa_b[1])
    if got:
        pos, yaw, side = got
        sofa = _add(scene, "sofa", "furniture", room, sofa_b, pos, yaw)
        f = _front(yaw)
        rug_b = cat.rug()
        rc = pos[:2] + f * (sofa_b[1][1] / 2 + 0.1 + rug_b[1][1] / 2)
        if _inside(pl, pl.rect_for(rc[0], rc[1], rug_b[1][0], rug_b[1][1], yaw)):
            _add(scene, "rug", "furniture", room, rug_b, np.array([rc[0], rc[1], 0.0]), yaw, obstacle=False)
        ct_b = cat.coffee_table()
        cc = pos[:2] + f * (sofa_b[1][1] / 2 + 0.45 + ct_b[1][1] / 2)
        g2 = pl.at(cc[0], cc[1], ct_b[1], yaw)
        if g2:
            ct = _add(scene, "coffee_table", "furniture", room, ct_b, g2[0], yaw, movable=True)
            for item in ("vase", "fruit_bowl"):
                if rng.random() < 0.6:
                    place_on(scene, ct, getattr(cat, item)(), item, rng)
        side_dir = np.array([np.cos(yaw), np.sin(yaw)])
        for sgn in (1, -1):
            st_b = cat.side_table()
            sc = pos[:2] + side_dir * sgn * (sofa_b[1][0] / 2 + 0.35) - f * 0.15
            g3 = pl.at(sc[0], sc[1], st_b[1], yaw)
            if g3:
                st = _add(scene, "side_table", "furniture", room, st_b, g3[0], yaw, movable=True)
                place_on(scene, st, cat.table_lamp(), "table_lamp", rng)
                break
        for _ in range(2):
            place_on(scene, sofa, cat.cushion(), "cushion", rng, yaw=0.0)
        _wall_obj(scene, pl, cat.painting, "painting", 1.6, room, sides=side, along=_along(pos, side))
        opp = {"S": "N", "N": "S", "W": "E", "E": "W"}[side]
        tvs_b = cat.tv_stand()
        g = pl.against_wall(tvs_b[1], sides=opp)
        if g:
            tvs = _add(scene, "tv_stand", "furniture", room, tvs_b, g[0], g[1])
            place_on(scene, tvs, cat.tv(), "tv", rng, yaw=0.0)
    for name in ("bookshelf", "armchair", "floor_lamp", "plant"):
        b = getattr(cat, name)()
        g = pl.against_wall(b[1])
        if g:
            _add(scene, name, "furniture", room, b, g[0], g[1], movable=name != "bookshelf")
    b = cat.plant()
    g = pl.random_free(b[1])
    if g:
        _add(scene, "plant", "furniture", room, b, g[0], g[1], movable=True)
    _wall_obj(scene, pl, cat.painting, "painting", 1.55, room)


def furnish_kitchen(scene, pl: Planner, rng):
    cat, room = pl.cat, pl.room.name
    x0, y0, x1, y1 = pl.b
    sides = sorted("NSEW", key=lambda sd: -(pl.side_range(sd)[1] - pl.side_range(sd)[0]))
    placed = False
    for side in sides:
        lo, hi = pl.side_range(side)
        doors = pl.room_doorways(side)
        # candidate free intervals along this wall (between doorway clearances)
        cuts = sorted([(a - 0.3, b + 0.3) for a, b in doors])
        free, cur = [], lo
        for a, b in cuts:
            if a > cur:
                free.append((cur, a))
            cur = max(cur, b)
        if hi > cur:
            free.append((cur, hi))
        free.sort(key=lambda ab: -(ab[1] - ab[0]))
        for flo, fhi in free:
            span = fhi - flo
            if span < 1.8:
                continue
            with_fridge = span >= 2.7
            length = float(min(3.0, span - (0.85 if with_fridge else 0.05)))
            if rng.random() < 0.5:
                fridge_at, c_lo = flo + 0.41, flo + (0.84 if with_fridge else 0.02)
            else:
                fridge_at, c_lo = fhi - 0.41, fhi - (0.84 if with_fridge else 0.02) - length
            a = c_lo + length / 2
            cb = cat.counter(length)
            g = pl.against_wall(cb[1], sides=side, along=a)
            if not g:
                continue
            counter = _add(scene, "counter", "furniture", room, cb, g[0], g[1])
            if all(a + length / 2 <= w0 or w1 <= a - length / 2 for w0, w1 in pl.windows.get(side, [])):
                ub = cat.upper_cabinets(length)
                _add(scene, "upper_cabinets", "furniture", room, ub, g[0] + np.r_[_front(g[1]) * -0.14, 0.0], g[1],
                     obstacle=False)
            for item in ("microwave", "kettle", "toaster"):
                if item != "toaster" or rng.random() < 0.5:
                    place_on(scene, counter, getattr(cat, item)(), item, rng, yaw=0.0)
            if with_fridge:
                fb = cat.fridge()
                g2 = pl.against_wall(fb[1], sides=side, along=fridge_at)
                if g2:
                    _add(scene, "fridge", "furniture", room, fb, g2[0], g2[1])
                    placed = True
            placed = placed or False
            break
        if any(o.name == "counter" for o in scene.objects.values() if o.room == room):
            break
    if not any(o.name == "fridge" for o in scene.objects.values() if o.room == room):
        fb = cat.fridge()
        g2 = pl.against_wall(fb[1], tries=100)
        if g2:
            _add(scene, "fridge", "furniture", room, fb, g2[0], g2[1])
    tb = cat.dining_table()
    g = None
    for k in range(240):
        if k == 160:
            tb = cat.dining_table()
            tb = (_table(1.0, 0.75, 0.75, tb[0][0].material, tb[0][0].material), (1.0, 0.75, 0.75),
                  (-0.42, -0.3, 0.42, 0.3, 0.75))
        yaw = float(rng.choice([0.0, np.pi / 2]))
        clear = (1.1, 0.8, 0.55)[min(2, k // 60)] if k < 160 else 0.6
        g = pl.at(rng.uniform(x0, x1), rng.uniform(y0, y1), (tb[1][0] + clear, tb[1][1] + clear, 0.75), yaw,
                  reserve=False)
        if g:
            break
    if g:
        pos, yaw = g
        pl.at(pos[0], pos[1], tb[1], yaw)
        table = _add(scene, "dining_table", "furniture", room, tb, pos, yaw)
        place_on(scene, table, cat.fruit_bowl(), "fruit_bowl", rng)
        cm = cat.mat(pick(rng, WOOD + FABRIC))
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        for lx, ly, cyaw in ((-0.28 * tb[1][0], tb[1][1] / 2 + 0.3, 0.0), (0.28 * tb[1][0], tb[1][1] / 2 + 0.3, 0.0),
                             (-0.28 * tb[1][0], -tb[1][1] / 2 - 0.3, np.pi), (0.28 * tb[1][0], -tb[1][1] / 2 - 0.3, np.pi)):
            if rng.random() < 0.8:
                p = pos[:2] + R @ np.array([lx, ly])
                chb = cat.chair(cm)
                g3 = pl.at(p[0], p[1], chb[1], yaw + cyaw, pad=0.0)
                if g3:
                    _add(scene, "chair", "furniture", room, chb, g3[0], yaw + cyaw, movable=True)
    for name in ("trash_bin", "plant"):
        b = getattr(cat, name)()
        g = pl.against_wall(b[1])
        if g:
            _add(scene, name, "furniture", room, b, g[0], g[1], movable=True)
    _wall_obj(scene, pl, cat.painting, "painting", 1.55, room)


def furnish_hallway(scene, pl: Planner, rng):
    cat, room = pl.cat, pl.room.name
    b = cat.console_table()
    g = pl.against_wall(b[1])
    if g:
        ct = _add(scene, "console_table", "furniture", room, b, g[0], g[1])
        place_on(scene, ct, cat.key_bowl(), "key_bowl", rng)
        if rng.random() < 0.7:
            place_on(scene, ct, cat.table_lamp(), "table_lamp", rng)
        _wall_obj(scene, pl, cat.mirror, "mirror", 1.45, room, sides=g[2], along=_along(g[0], g[2]))
    for name in ("shoe_rack", "coat_stand", "umbrella_stand"):
        b = getattr(cat, name)()
        g = pl.against_wall(b[1])
        if g:
            obj = _add(scene, name, "furniture", room, b, g[0], g[1], movable=name != "shoe_rack")
            if name == "shoe_rack" and rng.random() < 0.8:
                place_on(scene, obj, cat.shoes(), "shoes", rng)
    x0, y0, x1, y1 = pl.b
    if (x1 - x0) > 2.2 and (y1 - y0) > 1.8:
        rb = cat.rug(w=min(2.4, x1 - x0 - 1.4), d=0.8)
        _add(scene, "runner_rug", "furniture", room, rb, np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.0]), 0.0,
             obstacle=False)
    _wall_obj(scene, pl, cat.painting, "painting", 1.55, room)


def furnish_bedroom(scene, pl: Planner, rng):
    cat, room = pl.cat, pl.room.name
    bb = cat.bed()
    bw, bd, _ = bb[1]
    g = pl.against_wall((bw + 1.1, bd, bb[1][2]), tries=80, reserve=False) or pl.against_wall(bb[1], tries=80, reserve=False)
    if g:
        pos, yaw, side = g
        pl.at(pos[0], pos[1], bb[1], yaw, pad=0.0)
        bed = _add(scene, "bed", "furniture", room, bb, pos, yaw)
        for _ in range(2):
            place_on(scene, bed, cat.pillow(), "pillow", rng, yaw=0.0)
        right = np.array([np.cos(yaw), np.sin(yaw)])
        f = _front(yaw)
        lamp_done = False
        for sgn in (1, -1):
            nb = cat.nightstand()
            c = pos[:2] + right * sgn * (bw / 2 + 0.3) - f * (bd / 2 - nb[1][1] / 2)
            g2 = pl.at(c[0], c[1], nb[1], yaw, pad=0.02)
            if g2:
                ns = _add(scene, "nightstand", "furniture", room, nb, g2[0], yaw)
                if not lamp_done:
                    place_on(scene, ns, cat.table_lamp(), "table_lamp", rng)
                    lamp_done = True
        _wall_obj(scene, pl, cat.painting, "painting", 1.6, room, sides=side, along=_along(pos, side))
        rug_b = cat.rug(w=1.4, d=0.8)
        rc = pos[:2] + right * (bw / 2 + 0.75) + f * 0.3
        if _inside(pl, pl.rect_for(rc[0], rc[1], 1.4, 0.8, yaw)):
            _add(scene, "rug", "furniture", room, rug_b, np.array([rc[0], rc[1], 0.0]), yaw, obstacle=False)
    for name in ("wardrobe", "desk"):
        b = getattr(cat, name)()
        g = pl.against_wall(b[1])
        if g:
            obj = _add(scene, name, "furniture", room, b, g[0], g[1])
            if name == "desk":
                chb = cat.chair()
                c = g[0][:2] + _front(g[1]) * 0.56
                g5 = pl.at(c[0], c[1], chb[1], g[1] + np.pi, pad=0.0)
                if g5:
                    _add(scene, "desk_chair", "furniture", room, chb, g5[0], g[1] + np.pi, movable=True)
                if rng.random() < 0.6:
                    place_on(scene, obj, cat.table_lamp(), "desk_lamp", rng)
    b = cat.laundry_basket()
    g = pl.against_wall(b[1])
    if g:
        _add(scene, "laundry_basket", "furniture", room, b, g[0], g[1], movable=True)


def furnish_bathroom(scene, pl: Planner, rng):
    cat, room = pl.cat, pl.room.name
    x0, y0, x1, y1 = pl.b
    long_sides = "SN" if (x1 - x0) >= (y1 - y0) else "WE"
    b = cat.bathtub()
    g = pl.against_wall(b[1], sides=long_sides) or pl.against_wall(b[1])
    if g:
        _add(scene, "bathtub", "furniture", room, b, g[0], g[1])
        mb = cat.bath_mat()
        c = g[0][:2] + _front(g[1]) * 0.7
        if pl.at(c[0], c[1], mb[1], g[1], reserve=False):
            _add(scene, "bath_mat", "furniture", room, mb, np.array([c[0], c[1], 0.0]), g[1], movable=True,
                 obstacle=False)
    b = cat.vanity()
    g = pl.against_wall(b[1])
    if g:
        v = _add(scene, "vanity", "furniture", room, b, g[0], g[1])
        for _ in range(int(rng.integers(1, 4))):
            place_on(scene, v, cat.toiletry(), "toiletry", rng)
        _wall_obj(scene, pl, cat.mirror, "mirror", 1.6, room, sides=g[2], along=_along(g[0], g[2]))
    b = cat.toilet()
    g = pl.against_wall(b[1])
    if g:
        _add(scene, "toilet", "furniture", room, b, g[0], g[1])
    _wall_obj(scene, pl, cat.towel, "towel", 1.35, room)
    b = cat.trash_bin()
    g = pl.against_wall(b[1])
    if g:
        _add(scene, "trash_bin", "furniture", room, b, g[0], g[1], movable=True)


FURNISH = {"living": furnish_living, "kitchen": furnish_kitchen, "hallway": furnish_hallway,
           "bedroom": furnish_bedroom, "bathroom": furnish_bathroom}
