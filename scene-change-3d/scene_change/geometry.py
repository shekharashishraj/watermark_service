"""Rigid transforms, pinhole cameras and depth back-projection.

Conventions used everywhere in this package:

* World / session frames are gravity-aligned with **z up** (metres).
* Cameras use the OpenCV convention: x right, y down, z forward.
* A pose ``T_wc`` is a 4x4 camera-to-world transform.
* Depth maps store **z-depth** in metres; 0 marks an invalid pixel.
* Pixel ``(u, v)`` has its centre at integer coordinates (OpenCV style).
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Rotations and rigid transforms
# ----------------------------------------------------------------------------

def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def make_pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_pose(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform (works on (4,4) or (N,4,4))."""
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


def transform_points(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N,3) points."""
    P = np.asarray(P)
    return P @ T[:3, :3].T.astype(P.dtype, copy=False) + T[:3, 3].astype(P.dtype, copy=False)


def transform_dirs(T: np.ndarray, D: np.ndarray) -> np.ndarray:
    D = np.asarray(D)
    return D @ T[:3, :3].T.astype(D.dtype, copy=False)


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues formula."""
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + skew(w)
    k = w / th
    K = skew(k)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = float(np.arccos(c))
    if th < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    if np.pi - th < 1e-6:
        # Rotation by ~pi: pick the axis from the diagonal.
        A = (R + np.eye(3)) / 2
        axis = np.sqrt(np.clip(np.diag(A), 0, None))
        i = int(np.argmax(axis))
        axis = A[:, i] / max(axis[i], 1e-12)
        axis /= np.linalg.norm(axis)
        return axis * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * th / (2 * np.sin(th))


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Small-motion update: xi = (wx, wy, wz, tx, ty, tz); rotation then translation."""
    return make_pose(so3_exp(xi[:3]), xi[3:6])


def pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> tuple[float, float]:
    """Rotation error (degrees) and translation error (metres) between two poses."""
    dT = invert_pose(T_gt) @ T_est
    rot = float(np.degrees(np.linalg.norm(so3_log(dT[:3, :3]))))
    trans = float(np.linalg.norm(dT[:3, 3]))
    return rot, trans


def yaw_of(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


# ----------------------------------------------------------------------------
# Cameras
# ----------------------------------------------------------------------------

def intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    return np.array([[fx, 0, (width - 1) / 2.0], [0, fx, (height - 1) / 2.0], [0, 0, 1]], dtype=np.float64)


def camera_pose(position: np.ndarray, yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    """Camera-to-world pose for a camera at ``position`` looking along (yaw, pitch).

    yaw is measured from +x towards +y, pitch is positive looking up. The camera
    x axis stays horizontal when roll is zero.
    """
    cp, sp = np.cos(pitch), np.sin(pitch)
    f = np.array([cp * np.cos(yaw), cp * np.sin(yaw), sp])
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    if roll != 0.0:
        c, s = np.cos(roll), np.sin(roll)
        right, down = c * right + s * down, -s * right + c * down
    R = np.stack([right, down, f], axis=1)
    return make_pose(R, np.asarray(position, dtype=np.float64))


def pixel_rays(K: np.ndarray, width: int, height: int) -> np.ndarray:
    """(H*W, 3) camera-frame ray directions with z = 1 (so t equals z-depth)."""
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    return np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)


def project(K: np.ndarray, T_cw: np.ndarray, P: np.ndarray):
    """Project world points. Returns (u, v, z) in pixel coordinates / camera depth."""
    Pc = transform_points(T_cw, P)
    z = Pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * Pc[:, 0] / z + K[0, 2]
        v = K[1, 1] * Pc[:, 1] / z + K[1, 2]
    return u, v, z


def backproject(K: np.ndarray, depth: np.ndarray, stride: int = 1, mask: np.ndarray | None = None):
    """Back-project a depth map to camera-frame points.

    Returns (points (N,3) float32, pixel_index (N,) into the full H*W grid).
    """
    H, W = depth.shape
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    d = depth[vs, us]
    valid = d > 0
    if mask is not None:
        valid &= mask[vs, us]
    us, vs, d = us[valid], vs[valid], d[valid]
    x = (us - K[0, 2]) / K[0, 0] * d
    y = (vs - K[1, 2]) / K[1, 1] * d
    pts = np.stack([x, y, d], axis=-1).astype(np.float32)
    return pts, (vs * W + us).astype(np.int64)


def depth_normals(K: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Per-pixel camera-frame normals (H,W,3) facing the camera; zero where undefined."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    X = (u - K[0, 2]) / K[0, 0] * depth
    Y = (v - K[1, 2]) / K[1, 1] * depth
    P = np.stack([X, Y, depth], axis=-1).astype(np.float32)
    n = np.zeros_like(P)
    dx = P[1:-1, 2:] - P[1:-1, :-2]
    dy = P[2:, 1:-1] - P[:-2, 1:-1]
    cr = np.cross(dy, dx)
    norm = np.linalg.norm(cr, axis=-1, keepdims=True)
    ok = (
        (depth[1:-1, 2:] > 0) & (depth[1:-1, :-2] > 0) & (depth[2:, 1:-1] > 0) & (depth[:-2, 1:-1] > 0)
        & (norm[..., 0] > 1e-12)
    )
    inner = np.where(ok[..., None], cr / np.maximum(norm, 1e-12), 0.0)
    n[1:-1, 1:-1] = inner
    return n


def depth_edge_mask(depth: np.ndarray, rel_jump: float = 0.08) -> np.ndarray:
    """True where a pixel sits on a depth discontinuity (likely flying / mixed pixel)."""
    d = depth
    m = np.zeros(d.shape, dtype=bool)
    for axis in (0, 1):
        a = np.diff(d, axis=axis)
        base = np.minimum(np.delete(d, -1, axis=axis), np.delete(d, 0, axis=axis))
        jump = (np.abs(a) > rel_jump * np.maximum(base, 0.3))
        if axis == 0:
            m[:-1, :] |= jump
            m[1:, :] |= jump
        else:
            m[:, :-1] |= jump
            m[:, 1:] |= jump
    return m


# ----------------------------------------------------------------------------
# Sparse voxel keys
# ----------------------------------------------------------------------------

_KEY_OFF = 1 << 20  # supports +/- 1M voxels per axis


def voxel_keys(P: np.ndarray, size: float) -> np.ndarray:
    """Pack integer voxel coordinates into int64 keys."""
    ijk = np.floor(np.asarray(P, dtype=np.float64) / size).astype(np.int64) + _KEY_OFF
    return (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]


def keys_to_ijk(keys: np.ndarray) -> np.ndarray:
    mask = (1 << 21) - 1
    i = (keys >> 42) & mask
    j = (keys >> 21) & mask
    k = keys & mask
    return np.stack([i, j, k], axis=-1) - _KEY_OFF


def ijk_to_keys(ijk: np.ndarray) -> np.ndarray:
    ijk = np.asarray(ijk, dtype=np.int64) + _KEY_OFF
    return (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]


def voxel_centers(keys: np.ndarray, size: float) -> np.ndarray:
    return (keys_to_ijk(keys).astype(np.float64) + 0.5) * size
