"""Controlled degradations of a walkthrough, for robustness studies.

A stressor changes what the camera recorded, never the scene, so a scenario's ground
truth (change masks, object ids, true poses) stays valid at every severity.

Image-space stressors work on any RGB frames, including PASLCD image folders:

    motion_blur(rgb, length, angle)     linear motion blur, applied in linear light
    exposure(rgb, ev)                   exposure change by ``ev`` stops in linear light,
                                        then 8-bit sRGB clipping and quantisation

Session-level helpers:

    blur_session(session, frac)         blur of ``frac`` x image width along each frame's
                                        apparent camera motion (from the device poses)
    exposure_session(session, ev)
    drop_views(n, frac, mode)           frames kept after removing ``frac`` of them, either as
                                        contiguous segments (lost coverage) or uniformly
                                        (lower frame rate); nested across ``frac``

Harness only (needs the scenario's scene and lighting):

    relight_inspection(sc, s)           re-renders the inspection walkthrough under lighting
                                        interpolated in log space between the baseline visit
                                        (s = 0) and the scenario's inspection visit (s = 1),
                                        extrapolated beyond; depth and poses are untouched

``STRESSORS`` lists the families and severity levels used by
``scene_change.harness.robustness``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import invert_pose
from .session import Session


# ----------------------------------------------------------------------------
# Colour helpers
# ----------------------------------------------------------------------------

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(y: np.ndarray) -> np.ndarray:
    y = np.clip(np.asarray(y, dtype=np.float32), 0.0, 1.0)
    return np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055).astype(np.float32)


def _to_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x) * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# Image-space stressors
# ----------------------------------------------------------------------------

def line_kernel(length: float, angle: float) -> np.ndarray:
    """Normalised linear motion-blur kernel, ``length`` px along ``angle`` (radians, x right, y down)."""
    if length < 1.0:
        return np.ones((1, 1), np.float32)
    r = int(np.ceil(length / 2)) + 1
    k = np.zeros((2 * r + 2, 2 * r + 2), np.float64)
    t = np.linspace(-length / 2, length / 2, max(int(np.ceil(4 * length)), 2))
    xs = r + t * np.cos(angle)
    ys = r + t * np.sin(angle)
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    fx, fy = xs - x0, ys - y0
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        np.add.at(k, (y0 + dy, x0 + dx), w)
    k = k[:2 * r + 1, :2 * r + 1]
    return (k / k.sum()).astype(np.float32)


def motion_blur(rgb: np.ndarray, length: float, angle: float) -> np.ndarray:
    """Blur a uint8 RGB frame by a linear motion of ``length`` px, integrating in linear light."""
    if length < 1.0:
        return rgb.copy()
    lin = srgb_to_linear(rgb.astype(np.float32) / 255.0)
    out = cv2.filter2D(lin, -1, line_kernel(length, angle), borderType=cv2.BORDER_REFLECT)
    return _to_u8(linear_to_srgb(out))


def exposure(rgb: np.ndarray, ev: float) -> np.ndarray:
    """Change exposure by ``ev`` stops in linear light; highlights clip, the result is re-quantised to 8 bits."""
    if ev == 0:
        return rgb.copy()
    lin = srgb_to_linear(rgb.astype(np.float32) / 255.0) * np.float32(2.0 ** ev)
    return _to_u8(linear_to_srgb(lin))


# ----------------------------------------------------------------------------
# Session-level stressors
# ----------------------------------------------------------------------------

def motion_directions(session: Session, default_depth: float = 2.5, grid: int = 12, seed: int = 0) -> np.ndarray:
    """Per-frame direction (radians) of the apparent image motion, from the device poses.

    Points on a grid are back-projected with the frame's depth (``default_depth`` where it
    is missing), moved into the neighbouring frames' cameras (central difference) and
    projected; the median flow gives the direction. Frames without motion get a random one.
    """
    n = len(session)
    rng = np.random.default_rng(seed)
    out = rng.uniform(0, np.pi, n)
    if n < 2:
        return out
    H, W = session.height, session.width
    K = session.K
    us, vs = np.meshgrid((np.arange(grid) + 0.5) * W / grid, (np.arange(grid) + 0.5) * H / grid)
    us, vs = us.ravel(), vs.ravel()
    Kinv = np.linalg.inv(K)
    for i in range(n):
        a, b = max(i - 1, 0), min(i + 1, n - 1)
        z = session.depth[i][np.clip(vs.astype(int), 0, H - 1), np.clip(us.astype(int), 0, W - 1)].astype(np.float64)
        z = np.where(z > 0, z, default_depth)
        Xc = (Kinv @ np.stack([us, vs, np.ones_like(us)])) * z
        Xw = session.poses[i][:3, :3] @ Xc + session.poses[i][:3, 3:4]
        flows = []
        for j in (a, b):
            if j == i:
                continue
            T = invert_pose(session.poses[j])
            Xj = T[:3, :3] @ Xw + T[:3, 3:4]
            ok = Xj[2] > 0.1
            if ok.sum() < 5:
                continue
            u = K[0, 0] * Xj[0, ok] / Xj[2, ok] + K[0, 2]
            v = K[1, 1] * Xj[1, ok] / Xj[2, ok] + K[1, 2]
            f = np.stack([u - us[ok], v - vs[ok]], 1) * (1 if j > i else -1)
            flows.append(np.median(f, axis=0))
        if flows:
            f = np.mean(flows, axis=0)
            if np.hypot(*f) > 0.25:
                out[i] = np.arctan2(f[1], f[0])
    return out


def _replace_rgb(session: Session, rgb: np.ndarray, tag: str) -> Session:
    meta = dict(session.meta)
    meta["stress"] = meta.get("stress", []) + [tag]
    return Session(session.name, session.K, rgb, session.depth, session.poses, session.timestamps, meta)


def blur_session(session: Session, frac: float, seed: int = 0) -> Session:
    """Motion blur of ``frac`` x image width on every frame, along its apparent camera motion."""
    if frac <= 0:
        return session
    length = frac * session.width
    ang = motion_directions(session, seed=seed)
    rgb = np.stack([motion_blur(session.rgb[i], length, ang[i]) for i in range(len(session))])
    return _replace_rgb(session, rgb, f"blur:{frac}")


def exposure_session(session: Session, ev: float) -> Session:
    if ev == 0:
        return session
    rgb = np.stack([exposure(session.rgb[i], ev) for i in range(len(session))])
    return _replace_rgb(session, rgb, f"exposure:{ev}")


def drop_views(n: int, frac: float, mode: str = "segments", segments: int = 2, seed: int = 0) -> np.ndarray:
    """Sorted indices of the frames kept after removing ``round(frac * n)`` of ``n`` frames.

    ``segments``: the removed frames form ``segments`` contiguous runs growing around random
    centres (parts of the property never filmed). ``uniform``: a random subset (a lower
    frame rate). Removals are nested: a larger ``frac`` removes a superset.
    """
    k = int(round(np.clip(frac, 0.0, 1.0) * n))
    if k <= 0:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    if mode == "segments":
        centres = rng.uniform(0, n, segments)
        dist = np.min(np.abs(np.arange(n)[:, None] - centres[None, :]), axis=1)
        order = np.lexsort((rng.random(n), dist))
    elif mode == "uniform":
        order = rng.permutation(n)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    keep = np.ones(n, bool)
    keep[order[:k]] = False
    return np.nonzero(keep)[0]


# ----------------------------------------------------------------------------
# Harness relighting
# ----------------------------------------------------------------------------

def interpolate_lighting(base, insp, s: float):
    """Lighting at severity ``s``: log-space interpolation from ``base`` (0) to ``insp`` (1), extrapolated beyond.

    Lamps on in both visits keep their intensity; lamps switched on for the inspection
    ramp up with ``s`` (and get brighter beyond 1); lamps switched off fade out by ``s = 1``.
    """
    from .harness.raycast import Lighting

    def lerp(a, b):
        return float(np.exp((1 - s) * np.log(max(a, 1e-6)) + s * np.log(max(b, 1e-6))))

    wb = np.exp((1 - s) * np.log(np.asarray(base.white_balance)) + s * np.log(np.asarray(insp.white_balance)))
    wb = wb / wb.mean()
    rooms = set(base.room_gain) | set(insp.room_gain)
    gains = {r: lerp(base.room_gain.get(r, 1.0), insp.room_gain.get(r, 1.0)) for r in sorted(rooms)}

    def key(lamp):
        return lamp[2], tuple(np.round(np.asarray(lamp[0], dtype=np.float64), 3).tolist())

    B = {key(lp): lp for lp in base.lamps}
    I = {key(lp): lp for lp in insp.lamps}
    lamps = []
    for k in sorted(set(B) | set(I)):
        if k in B and k in I:
            pos, inten, room = I[k][0], I[k][1], I[k][2]
        elif k in I:
            pos, inten, room = I[k][0], I[k][1] * s, I[k][2]
        else:
            pos, inten, room = B[k][0], B[k][1] * max(0.0, 1.0 - s), B[k][2]
        if inten > 0:
            lamps.append((list(pos), float(inten), room))
    return Lighting(exposure=lerp(base.exposure, insp.exposure), white_balance=tuple(wb.tolist()),
                    ambient=lerp(base.ambient, insp.ambient), ceiling=lerp(base.ceiling, insp.ceiling),
                    room_gain=gains, lamps=lamps)


def relight_inspection(sc: dict, s: float, frames=None, seed: int = 0) -> Session:
    """The scenario's inspection walkthrough re-rendered under lighting severity ``s``.

    Renders RGB at the true world poses of ``frames`` (default: all) with the harness
    ray caster and its RGB sensor model (per-frame exposure jitter, noise); depth, device
    poses and ground truth are those of the original walkthrough.
    """
    from .harness.raycast import BoxRenderer
    from .harness.walkthrough import WalkConfig

    session = sc["inspection"]
    frames = np.arange(len(session)) if frames is None else np.asarray(frames)
    light = interpolate_lighting(sc["lighting"]["baseline"], sc["lighting"]["inspection"], s)
    renderer = BoxRenderer(sc["inspection_scene"])
    cfg = WalkConfig(width=session.width, height=session.height)
    rng = np.random.default_rng([seed, int(round(1000 * s))])
    poses_w = sc["gt_arrays"]["insp_poses_world"]
    W, H = session.width, session.height
    rgb = np.empty((len(frames), H, W, 3), np.uint8)
    expo = 1.0
    for n, i in enumerate(frames):
        out = renderer.render(session.K, poses_w[i], W, H, light)
        expo = float(np.clip(expo + rng.normal(0, cfg.exposure_jitter), 0.85, 1.15))
        img = out["rgb"].astype(np.float64) * expo + rng.normal(0, cfg.rgb_noise, out["rgb"].shape)
        rgb[n] = _to_u8(img)
    sub = session.subset(frames)
    return _replace_rgb(sub, rgb, f"relight:{s}")


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Stressor:
    name: str
    levels: tuple
    unit: str
    description: str
    harness_only: bool = False
    reference: float = 0.0          # severity of the reference condition
    reference_run: str = "none"     # "none": the reference is the unmodified run; "self": a run of this stressor
    target: str = "inspection"      # what is degraded: the inspection walkthrough or the baseline map


STRESSORS = {
    "blur": Stressor("blur", (0.01, 0.02, 0.04, 0.07, 0.10), "blur length / image width",
                     "linear motion blur along the apparent camera motion"),
    "dark": Stressor("dark", (-0.5, -1.0, -2.0, -3.0, -4.0), "exposure change (stops)",
                     "underexposure: less light reaches the sensor"),
    "bright": Stressor("bright", (0.5, 1.0, 1.5, 2.0), "exposure change (stops)",
                       "overexposure: highlights clip"),
    "relight": Stressor("relight", (0.0, 0.5, 1.0, 2.0, 3.0), "illumination change (x the scenario's)",
                        "room lights, lamps, white balance and exposure moved from the baseline visit's "
                        "lighting (0) past the scenario's inspection lighting (1)", harness_only=True,
                        reference=1.0, reference_run="self"),
    "coverage": Stressor("coverage", (0.2, 0.4, 0.6, 0.8), "share of views removed",
                         "contiguous stretches of the walkthrough removed (areas never filmed)"),
    "sparse": Stressor("sparse", (0.5, 0.75), "share of views removed",
                       "views removed uniformly at random (lower frame rate)"),
    "compress": Stressor("compress", (0.075, 0.10, 0.15, 0.20), "Gaussian voxel size (m)",
                         "coarser baseline map: fewer, larger Gaussians (lower capacity)", harness_only=True,
                         reference=0.05, target="baseline"),
}


def apply_stressor(name: str, level: float, session: Session, sc: dict | None = None, frames=None,
                   seed: int = 0):
    """(stressed session, indices of the kept frames within ``session``).

    ``frames`` maps ``session`` to the scenario's full inspection walkthrough (needed by
    ``relight``, which re-renders those frames).
    """
    n = len(session)
    if name == "none" or (name in STRESSORS and STRESSORS[name].target == "baseline"):
        return session, np.arange(n)
    if name == "blur":
        return blur_session(session, level, seed=seed), np.arange(n)
    if name in ("dark", "bright", "exposure"):
        return exposure_session(session, level), np.arange(n)
    if name == "relight":
        if sc is None:
            raise ValueError("relight needs the harness scenario")
        return relight_inspection(sc, level, frames=frames, seed=seed), np.arange(n)
    if name in ("coverage", "sparse"):
        keep = drop_views(n, level, mode="segments" if name == "coverage" else "uniform", seed=seed)
        return session.subset(keep), keep
    raise ValueError(f"unknown stressor {name!r}")
