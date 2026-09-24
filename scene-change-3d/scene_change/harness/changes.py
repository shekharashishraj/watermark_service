"""Ground-truth scene changes between a baseline visit and an inspection visit."""

from __future__ import annotations

import numpy as np

from .scene import (FLOOR_ITEMS_ADDED, SOFT_ITEMS_ADDED, SURFACE_ITEMS_ADDED, Catalog, Material, Scene,
                    _add, _box, place_on)

STAINS = {
    "wine stain": (0.42, 0.07, 0.12),
    "coffee stain": (0.36, 0.22, 0.11),
    "dirt": (0.30, 0.27, 0.22),
    "ink mark": (0.12, 0.12, 0.38),
}
SCUFFS = {"scuff mark": (0.22, 0.22, 0.24), "wall damage": (0.35, 0.3, 0.26)}
SWAPPABLE = {"painting", "towel", "bath_mat", "pillow", "cushion", "rug", "runner_rug"}
FLAT_SURFACES = {"rug", "runner_rug", "bed", "sofa", "coffee_table", "dining_table", "desk", "bath_mat"}


def _dependents(scene: Scene, oid: int):
    return [o for o in scene.objects.values() if o.support == oid]


def _floor_rects(scene: Scene, exclude=()):
    rects = []
    for o in scene.objects.values():
        if o.oid in exclude or o.category in ("structure", "decal", "wall_item") or o.support is not None:
            continue
        lo, hi = o.aabb()
        if hi[2] - lo[2] < 0.05:        # rugs and mats do not block placement
            continue
        rects.append((lo[0], lo[1], hi[0], hi[1]))
    return rects


def _door_zones(scene: Scene):
    zones = []
    for dw in scene.doorways:
        c = dw.center_xy()
        hw = dw.width / 2 + 0.2
        if dw.axis == "h":
            zones.append((c[0] - hw, c[1] - 0.8, c[0] + hw, c[1] + 0.8))
        else:
            zones.append((c[0] - 0.8, c[1] - hw, c[0] + 0.8, c[1] + hw))
    e = scene.entry
    zones.append((e[0] - 0.7, e[1] - 0.7, e[0] + 0.7, e[1] + 0.7))
    return zones


def _free_floor_spot(scene: Scene, room_name: str, size, rng, exclude=(), tries=150, avoid=None, min_dist=0.0):
    w, d, _ = size
    room = scene.room(room_name)
    x0, y0, x1, y1 = room.interior
    rects = _floor_rects(scene, exclude) + _door_zones(scene)
    r = 0.5 * np.hypot(w, d)
    for _ in range(tries):
        x = rng.uniform(x0 + r + 0.05, x1 - r - 0.05)
        y = rng.uniform(y0 + r + 0.05, y1 - r - 0.05)
        cand = (x - r, y - r, x + r, y + r)
        if any(not (cand[2] + 0.05 <= q[0] or q[2] + 0.05 <= cand[0] or cand[3] + 0.05 <= q[1] or q[3] + 0.05 <= cand[1])
               for q in rects):
            continue
        if avoid is not None and np.hypot(x - avoid[0], y - avoid[1]) < min_dist:
            continue
        return np.array([x, y, 0.0]), float(rng.uniform(-np.pi, np.pi))
    return None


def _record(kind, obj, **kw):
    lo, hi = obj.aabb()
    rec = {"type": kind, "oid": int(obj.oid), "name": obj.name, "room": obj.room}
    rec.update(kw)
    rec.setdefault("aabb_after" if kind != "removed" else "aabb_before", [lo.tolist(), hi.tolist()])
    return rec


def apply_changes(scene: Scene, rng: np.random.Generator, n_removed=2, n_added=2, n_moved=2, n_appearance=2,
                  rooms=None):
    """Return (changed scene copy, list of change records)."""
    s = scene.copy()
    cat = Catalog(s, rng)
    rooms = list(rooms) if rooms else [r.name for r in s.rooms]
    changes = []
    touched = set()

    def candidates(filter_fn):
        return [o for o in s.objects.values() if o.room in rooms and o.oid not in touched and filter_fn(o)]

    # --- removed ---------------------------------------------------------
    rem = candidates(lambda o: o.movable and o.category in ("furniture", "item", "wall_item") and not _dependents(s, o.oid))
    rng.shuffle(rem)
    for o in rem[:n_removed]:
        rec = _record("removed", o, detail=f"{o.name} missing")
        del s.objects[o.oid]
        touched.add(o.oid)
        changes.append(rec)

    # --- moved -----------------------------------------------------------
    mov = candidates(lambda o: o.movable and o.category == "furniture" and o.support is None
                     and not _dependents(s, o.oid) and o.aabb()[1][2] - o.aabb()[0][2] > 0.2)
    rng.shuffle(mov)
    moved = 0
    for o in mov:
        if moved >= n_moved:
            break
        lo, hi = o.aabb()
        size = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
        target_room = o.room
        if rng.random() < 0.3:
            target_room = rooms[int(rng.integers(len(rooms)))]
        spot = _free_floor_spot(s, target_room, size, rng, exclude=(o.oid,), avoid=o.position, min_dist=0.8)
        if spot is None:
            continue
        before = [lo.tolist(), hi.tolist()]
        old_pos, old_yaw = o.position.copy(), o.yaw
        o.position, o.yaw = spot[0], spot[1]
        o.room = target_room
        lo2, hi2 = o.aabb()
        changes.append({"type": "moved", "oid": int(o.oid), "name": o.name, "room": target_room,
                        "aabb_before": before, "aabb_after": [lo2.tolist(), hi2.tolist()],
                        "from": old_pos.tolist(), "from_yaw": old_yaw, "to": o.position.tolist(),
                        "detail": f"{o.name} moved {np.linalg.norm(o.position[:2] - old_pos[:2]):.1f} m"})
        touched.add(o.oid)
        moved += 1

    # --- added -----------------------------------------------------------
    added = 0
    for _ in range(40):
        if added >= n_added:
            break
        room = rooms[int(rng.integers(len(rooms)))]
        mode = rng.random()
        obj = None
        if mode < 0.55:
            name = FLOOR_ITEMS_ADDED[int(rng.integers(len(FLOOR_ITEMS_ADDED)))]
            built = getattr(cat, name)()
            spot = _free_floor_spot(s, room, built[1], rng)
            if spot is not None:
                obj = _add(s, name, "item", room, built, spot[0], spot[1], movable=True, obstacle=True)
        else:
            sups = [o for o in s.objects.values() if o.room == room and o.top is not None
                    and o.category in ("furniture",) and o.oid not in touched]
            if sups:
                sup = sups[int(rng.integers(len(sups)))]
                pool = SOFT_ITEMS_ADDED if sup.name in ("bed", "sofa") else SURFACE_ITEMS_ADDED
                name = pool[int(rng.integers(len(pool)))]
                obj = place_on(s, sup, getattr(cat, name)(), name, rng, room=room)
        if obj is not None:
            changes.append(_record("added", obj, detail=f"{obj.name} left behind"))
            touched.add(obj.oid)
            added += 1

    # --- appearance ------------------------------------------------------
    done = 0
    for _ in range(40):
        if done >= n_appearance:
            break
        if rng.random() < 0.55:
            rec = _add_stain(s, cat, rng, rooms, touched)
        else:
            rec = _swap_look(s, cat, rng, rooms, touched)
        if rec is not None:
            changes.append(rec)
            done += 1

    for k, c in enumerate(changes):
        c["id"] = k
    return s, changes


def _add_stain(s: Scene, cat: Catalog, rng, rooms, touched):
    """A thin decal on a horizontal surface or a wall (appearance-only change)."""
    if rng.random() < 0.3:
        walls = [o for o in s.objects.values() if o.name.startswith("wall_") and o.room in rooms]
        if not walls:
            return None
        wall = walls[int(rng.integers(len(walls)))]
        side = wall.tags["side"]
        room = s.room(wall.room)
        x0, y0, x1, y1 = room.interior
        name, col = list(SCUFFS.items())[int(rng.integers(len(SCUFFS)))]
        w, h = rng.uniform(0.2, 0.4), rng.uniform(0.12, 0.3)
        z = rng.uniform(0.35, 1.3)
        lo, hi = (x0, x1) if side in "SN" else (y0, y1)
        a = rng.uniform(lo + 0.5, hi - 0.5)
        m = cat.mat(col, "noise", color2=tuple(np.clip(np.asarray(col) * 1.5, 0, 1)), scale=0.05, contrast=0.4)
        if side in "SN":
            y = y0 + 0.002 if side == "S" else y1 - 0.002
            parts = [_box(0, 0, 0, w / 2, 0.002, h / 2, m)]
            pos, yaw = np.array([a, y, z]), 0.0
        else:
            x = x0 + 0.002 if side == "W" else x1 - 0.002
            parts = [_box(0, 0, 0, 0.002, w / 2, h / 2, m)]
            pos, yaw = np.array([x, a, z]), 0.0
        # reject if it lands on a doorway, window or wall item
        for o in s.objects.values():
            if o.category in ("wall_item",) or o.name.startswith("window") or o.name == "front_door":
                olo, ohi = o.aabb()
                if (olo[0] - 0.1 <= pos[0] <= ohi[0] + 0.1 and olo[1] - 0.1 <= pos[1] <= ohi[1] + 0.1
                        and olo[2] - 0.2 <= z <= ohi[2] + 0.2):
                    return None
        for dw in s.doorways:
            c = dw.center_xy()
            if np.hypot(pos[0] - c[0], pos[1] - c[1]) < dw.width / 2 + w / 2 + 0.1:
                return None
        # something standing in front of the wall would hide it: require clear space
        n_in = {"S": (0, 1), "N": (0, -1), "W": (1, 0), "E": (-1, 0)}[side]
        probe = pos[:2] + 0.3 * np.asarray(n_in)
        for o in s.objects.values():
            if o.category in ("furniture", "item") and o.oid not in touched:
                olo, ohi = o.aabb()
                if olo[0] - 0.1 <= probe[0] <= ohi[0] + 0.1 and olo[1] - 0.1 <= probe[1] <= ohi[1] + 0.1 and olo[2] <= z <= ohi[2]:
                    return None
        obj = _add(s, name.replace(" ", "_"), "decal", room.name, (parts, (w, 0.004, h), None), pos, yaw,
                   obstacle=False, tags={"surface": "wall", "on": wall.oid})
        touched.add(obj.oid)
        return {"type": "appearance", "oid": int(obj.oid), "name": name, "room": room.name, "on": wall.name,
                "aabb_after": [a_.tolist() for a_ in obj.aabb()], "detail": f"{name} on {room.name} wall"}
    # horizontal surface
    targets = [o for o in s.objects.values() if o.room in rooms and o.oid not in touched
               and (o.name in FLAT_SURFACES or o.name.startswith("floor_"))]
    if not targets:
        return None
    t = targets[int(rng.integers(len(targets)))]
    name, col = list(STAINS.items())[int(rng.integers(len(STAINS)))]
    if t.name.startswith("floor_"):
        room = s.room(t.room)
        spot = _free_floor_spot(s, room.name, (0.4, 0.4, 0.01), rng)
        if spot is None:
            return None
        pos, ztop = spot[0], 0.0
    else:
        tw = t.top_world()
        if tw is None:
            return None
        corners, ztop = tw
        # sample inside the top quad but away from items resting on it
        for _ in range(30):
            u, v = rng.uniform(0.2, 0.8, 2)
            p = corners[0] + u * (corners[1] - corners[0]) + v * (corners[3] - corners[0])
            if all(np.linalg.norm(o.position[:2] - p) > 0.35 for o in s.objects.values() if o.support == t.oid):
                break
        else:
            return None
        pos = np.array([p[0], p[1], 0.0])
    m = cat.mat(col, "noise", color2=tuple(np.clip(np.asarray(col) * 1.4, 0, 1)), scale=0.04, contrast=0.35)
    parts = []
    r = rng.uniform(0.1, 0.18)
    for k in range(3):
        off = rng.uniform(-r * 0.6, r * 0.6, 2)
        parts.append(_box(off[0], off[1], 0.0015, r * rng.uniform(0.5, 0.9), r * rng.uniform(0.4, 0.8), 0.0015, m))
    pos[2] = ztop
    obj = _add(s, name.replace(" ", "_"), "decal", t.room, (parts, (2 * r, 2 * r, 0.003), None), pos,
               float(rng.uniform(-np.pi, np.pi)), obstacle=False, tags={"surface": "floor" if t.name.startswith("floor_") else "object", "on": t.oid})
    touched.add(obj.oid)
    return {"type": "appearance", "oid": int(obj.oid), "name": name, "room": t.room, "on": t.name,
            "aabb_after": [a_.tolist() for a_ in obj.aabb()], "detail": f"{name} on {t.name.replace('_', ' ')}"}


def _swap_look(s: Scene, cat: Catalog, rng, rooms, touched):
    cands = [o for o in s.objects.values() if o.room in rooms and o.oid not in touched and o.name in SWAPPABLE]
    if not cands:
        return None
    o = cands[int(rng.integers(len(cands)))]
    old = [p.material for p in o.parts]
    if o.name == "painting":
        new_mat = cat.canvas_material()
        o.parts[-1].material = new_mat
    else:
        m0 = s.materials[o.parts[-1].material]
        col = np.asarray(m0.color)
        new_col = tuple(float(x) for x in np.clip(1.0 - col * 0.9 + rng.uniform(-0.1, 0.1, 3), 0.05, 0.95))
        new = Material(new_col, m0.pattern, color2=m0.color2, scale=m0.scale, contrast=m0.contrast,
                       seed=int(rng.integers(1 << 30)), color3=m0.color3)
        o.parts[-1].material = s.add_material(new)
    touched.add(o.oid)
    lo, hi = o.aabb()
    return {"type": "appearance", "oid": int(o.oid), "name": o.name, "room": o.room, "swap": True,
            "aabb_before": [lo.tolist(), hi.tolist()], "aabb_after": [lo.tolist(), hi.tolist()],
            "detail": f"{o.name.replace('_', ' ')} looks different", "old_materials": old}
