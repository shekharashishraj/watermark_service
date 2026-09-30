"""Photometric perturbations, applied in linear light (experiment_plan.md §7).

Every function has the signature fn(lin, severity, params, rng) -> (lin, info), where `lin` is a
float32 HxWx3 linear-RGB image in [0, 1], `rng` is a seeded numpy Generator and `info` records the
random draws so a variant can be audited later.
"""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------------------------
# sRGB <-> linear (IEC 61966-2-1)
# ---------------------------------------------------------------------------------------------
def srgb_to_linear(x):
    x = x.astype(np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(y):
    y = np.clip(y, 0.0, 1.0).astype(np.float32)
    return np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1.0 / 2.4) - 0.055).astype(np.float32)


def to_uint8(srgb):
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------
# stressors
# ---------------------------------------------------------------------------------------------
def identity(lin, severity, params, rng):
    return lin, {}


def motion_kernel(length, angle_deg):
    """Normalised linear motion-blur kernel of odd size; None for length <= 1."""
    size = int(round(float(length)))
    if size <= 1:
        return None
    if size % 2 == 0:
        size += 1
    k = np.zeros((size, size), np.float32)
    k[size // 2, :] = 1.0
    c = (size - 1) / 2.0
    rot = cv2.getRotationMatrix2D((c, c), float(angle_deg), 1.0)
    k = cv2.warpAffine(k, rot, (size, size), flags=cv2.INTER_LINEAR)
    return k / k.sum()


def motion_blur(lin, severity, params, rng):
    angle = params.get("angle_deg", "random")
    if angle == "random":
        angle = float(rng.uniform(0.0, 180.0))
    k = motion_kernel(severity, angle)
    if k is None:
        return lin, {"angle_deg": angle, "kernel_px": 0}
    out = cv2.filter2D(lin, -1, k, borderType=cv2.BORDER_REFLECT)
    return out, {"angle_deg": angle, "kernel_px": int(k.shape[0])}


def exposure(lin, severity, params, rng):
    return np.clip(lin * np.float32(2.0 ** float(severity)), 0.0, 1.0), {"gain": 2.0 ** float(severity)}


def white_balance(lin, severity, params, rng):
    k = float(severity)
    gains = np.array([1.0 + k, 1.0, 1.0 - k], np.float32)  # R, G, B
    return np.clip(lin * gains, 0.0, 1.0), {"gains_rgb": gains.tolist()}


def gain_field(lin, severity, params, rng):
    h, w = lin.shape[:2]
    small_w = 64
    small_h = max(2, int(round(small_w * h / float(w))))
    noise = rng.standard_normal((small_h, small_w)).astype(np.float32)
    sigma = float(params.get("sigma_frac", 0.125)) * small_w
    field = cv2.GaussianBlur(noise, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
    field /= np.abs(field).max() + 1e-8
    gain = np.clip(1.0 + float(severity) * field, float(params.get("min_gain", 0.05)), None)
    return np.clip(lin * gain[..., None], 0.0, 1.0), {"gain_min": float(gain.min()), "gain_max": float(gain.max())}


def shadow(lin, severity, params, rng):
    h, w = lin.shape[:2]
    lo, hi = params.get("area_frac", [0.1, 0.3])
    area = float(rng.uniform(lo, hi)) * h * w
    n = int(params.get("n_vertices", 6))
    radius = np.sqrt(area / np.pi)
    cx, cy = float(rng.uniform(0, w)), float(rng.uniform(0, h))
    ang = np.sort(rng.uniform(0.0, 2.0 * np.pi, n))
    rad = radius * rng.uniform(0.7, 1.3, n)
    pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], axis=1).astype(np.int32)
    mask = np.zeros((h, w), np.float32)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1.0)
    feather = float(params.get("feather_px", 15))
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather / 2.0)
    out = lin * (1.0 - float(severity) * mask)[..., None]
    return np.clip(out, 0.0, 1.0), {"center": [cx, cy], "area_frac": area / (h * w)}


def poisson_gaussian(lin, severity, params, rng):
    levels = {str(k): v for k, v in params["levels"].items()}
    key = str(int(severity)) if float(severity).is_integer() else str(severity)
    if key not in levels:
        raise KeyError("noise level %r not in params.levels %s" % (severity, sorted(levels)))
    photons, read_sigma = levels[key]
    photons = float(photons)
    out = rng.poisson(np.clip(lin, 0.0, 1.0) * photons).astype(np.float32) / photons
    out += rng.normal(0.0, float(read_sigma), lin.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0), {"photons": photons, "read_sigma": float(read_sigma)}


def compose(lin, severity, params, rng):
    info = {"steps": []}
    for step in params["steps"]:
        lin, step_info = FUNCS[step["fn"]](lin, step["severity"], step.get("params", {}), rng)
        info["steps"].append({"fn": step["fn"], "severity": step["severity"], **step_info})
    return lin, info


FUNCS = {
    "identity": identity,
    "motion_blur": motion_blur,
    "exposure": exposure,
    "white_balance": white_balance,
    "gain_field": gain_field,
    "shadow": shadow,
    "poisson_gaussian": poisson_gaussian,
    "compose": compose,
}


def apply_to_uint8(rgb_u8, fn, severity, params, rng):
    """uint8 sRGB in -> uint8 sRGB out. 'identity' skips the colour conversion entirely, so the
    identity variant differs from the original only by the re-encode (what it is meant to measure)."""
    if fn == "identity":
        return rgb_u8, {}
    lin = srgb_to_linear(rgb_u8.astype(np.float32) / 255.0)
    out, info = FUNCS[fn](lin, severity, params or {}, rng)
    return to_uint8(linear_to_srgb(out)), info
