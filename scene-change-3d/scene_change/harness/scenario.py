"""Build, save and load harness scenarios (baseline visit + inspection visit + ground truth)."""

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..session import Session
from .changes import apply_changes
from .raycast import Lighting
from .scene import Scene, generate_scene
from .walkthrough import WalkConfig, bfs_order, simulate_session

SCENARIOS = {
    # no changes: only lighting, viewpoint, drift and sensor noise differ
    "clean": {"changes": (0, 0, 0, 0), "skip_room": False, "partial_pan": False},
    # every change type, full coverage
    "mixed": {"changes": (2, 2, 2, 2), "skip_room": False, "partial_pan": False},
    # every change type, one room never visited and one room only half panned
    "partial": {"changes": (2, 2, 2, 2), "skip_room": True, "partial_pan": True},
}

LEAF_ROOMS = ("kitchen", "bathroom")


def random_lighting(scene: Scene, rng: np.random.Generator, strength: float) -> Lighting:
    """strength 0 = reference lighting, 1 = a typically different visit."""
    gains = {r.name: float(np.exp(rng.normal(0, 0.18 * strength))) for r in scene.rooms}
    lamps = []
    for o in scene.objects.values():
        if o.name in ("floor_lamp", "table_lamp", "desk_lamp") and rng.random() < 0.5:
            lo, hi = o.aabb()
            lamps.append((((lo + hi) / 2).tolist(), 0.35, o.room))
    wb = np.exp(rng.normal(0, 0.05 * strength, 3))
    wb /= wb.mean()
    return Lighting(exposure=float(np.exp(rng.normal(0, 0.15 * strength))), white_balance=tuple(wb.tolist()),
                    ambient=0.5, ceiling=0.75, room_gain=gains, lamps=lamps)


def build_scenario(seed: int, kind: str = "mixed", width: int = 160, height: int = 120,
                   inspection_drift: float = 0.01, lighting_strength: float = 1.0, fast: bool = False):
    if kind not in SCENARIOS:
        raise ValueError(f"unknown scenario {kind!r}; choose from {sorted(SCENARIOS)}")
    spec = SCENARIOS[kind]
    rng = np.random.default_rng([seed, {"clean": 1, "mixed": 2, "partial": 3}[kind]])
    scene = generate_scene(seed)
    nr, na, nm, nap = spec["changes"]
    insp_scene, changes = apply_changes(scene, rng, nr, na, nm, nap)

    base_cfg = WalkConfig(width=width, height=height, frame_offset=False, station_jitter=0.25,
                          cam_height=float(rng.uniform(1.45, 1.55)))
    rooms = bfs_order(scene)
    skipped, partial = [], []
    insp_rooms = ["hallway"] + [r for r in rng.permutation([r for r in rooms if r != "hallway"]).tolist()]
    if spec["skip_room"]:
        skip = LEAF_ROOMS[int(rng.integers(len(LEAF_ROOMS)))]
        insp_rooms = [r for r in insp_rooms if r != skip]
        skipped.append(skip)
    pan_span = {}
    if spec["partial_pan"]:
        cand = [r for r in insp_rooms if r not in ("hallway",)]
        pr = cand[int(rng.integers(len(cand)))]
        pan_span[pr] = 180.0
        partial.append(pr)
    insp_cfg = WalkConfig(width=width, height=height, rooms=insp_rooms, pan_span=pan_span, station_jitter=0.7,
                          cam_height=float(rng.uniform(1.4, 1.6)), drift_trans_per_m=inspection_drift,
                          drift_yaw_deg_per_m=20 * inspection_drift, pan_step=24.0)
    if fast:
        base_cfg = replace(base_cfg, pan_step=30.0, walk_step=0.8, two_stations_area=1e9)
        insp_cfg = replace(insp_cfg, pan_step=32.0, walk_step=0.8, two_stations_area=1e9)

    base_light = random_lighting(scene, rng, 0.3)
    insp_light = random_lighting(insp_scene, rng, lighting_strength)
    base_sess, base_gt = simulate_session("baseline", scene, base_cfg, base_light, rng)
    insp_sess, insp_gt = simulate_session("inspection", insp_scene, insp_cfg, insp_light, rng, extra_scenes=(scene,))

    rooms_ann = [{"name": r.name, "polygon": [[r.x0, r.y0], [r.x1, r.y0], [r.x1, r.y1], [r.x0, r.y1]]}
                 for r in scene.rooms]
    gt = {
        "seed": seed,
        "kind": kind,
        "changes": changes,
        "skipped_rooms": skipped,
        "partial_rooms": partial,
        "objects_baseline": scene.summary(),
        "objects_inspection": insp_scene.summary(),
        "rooms": rooms_ann,
        "T_baseline_world": base_gt["T_session_world"].tolist(),
        "T_inspection_world": insp_gt["T_session_world"].tolist(),
        "lighting": {"baseline_exposure": base_light.exposure, "inspection_exposure": insp_light.exposure},
    }
    arrays = {
        "base_poses_world": base_gt["poses_world"],
        "base_oid": base_gt["oid"],
        "insp_poses_world": insp_gt["poses_world"],
        "insp_oid": insp_gt["oid"],
        "insp_oid_basestate": insp_gt["extra_oid"][0],
    }
    return {"baseline": base_sess, "inspection": insp_sess, "gt": gt, "gt_arrays": arrays, "scene": scene,
            "inspection_scene": insp_scene, "rooms": rooms_ann,
            "lighting": {"baseline": base_light, "inspection": insp_light}}


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def save_scenario(sc: dict, out: str | Path) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sc["baseline"].save(out / "baseline")
    sc["inspection"].save(out / "inspection")
    (out / "gt.json").write_text(json.dumps(_jsonable(sc["gt"]), indent=1))
    (out / "rooms.json").write_text(json.dumps(_jsonable(sc["rooms"]), indent=1))
    np.savez_compressed(out / "gt.npz", **sc["gt_arrays"])
    with open(out / "scenes.pkl", "wb") as f:
        pickle.dump({"baseline": sc["scene"], "inspection": sc["inspection_scene"],
                     "lighting": sc.get("lighting")}, f)
    return out


def load_scenario(path: str | Path) -> dict:
    path = Path(path)
    arrays = dict(np.load(path / "gt.npz"))
    scenes = {}
    if (path / "scenes.pkl").exists():
        with open(path / "scenes.pkl", "rb") as f:
            scenes = pickle.load(f)
    return {
        "scene": scenes.get("baseline"),
        "inspection_scene": scenes.get("inspection"),
        "baseline": Session.load(path / "baseline"),
        "inspection": Session.load(path / "inspection"),
        "gt": json.loads((path / "gt.json").read_text()),
        "gt_arrays": arrays,
        "rooms": json.loads((path / "rooms.json").read_text()),
        "lighting": scenes.get("lighting"),
    }
