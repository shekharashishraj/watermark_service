"""Gaussian map: surfel-style 3D Gaussians fused from RGB-D keyframes.

Each Gaussian is a flat ellipsoid (a disc) lying on the observed surface, with a
colour, an opacity and an observation count. The map can be exported to and
loaded from the standard 3D Gaussian Splatting ``.ply`` layout, so a model
trained with GPU 3DGS tooling can replace the fused map without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .geometry import backproject, depth_edge_mask, depth_normals, voxel_keys

SH_C0 = 0.28209479177387814


# ----------------------------------------------------------------------------
# Quaternion helpers (w, x, y, z)
# ----------------------------------------------------------------------------

def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    q = np.empty(R.shape[:-2] + (4,))
    # robust branch-free-ish conversion
    w = np.sqrt(np.maximum(0, 1 + tr)) / 2
    x = np.sqrt(np.maximum(0, 1 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2])) / 2
    y = np.sqrt(np.maximum(0, 1 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2])) / 2
    z = np.sqrt(np.maximum(0, 1 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2])) / 2
    x = np.copysign(x, R[..., 2, 1] - R[..., 1, 2])
    y = np.copysign(y, R[..., 0, 2] - R[..., 2, 0])
    z = np.copysign(z, R[..., 1, 0] - R[..., 0, 1])
    q[..., 0], q[..., 1], q[..., 2], q[..., 3] = w, x, y, z
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


# ----------------------------------------------------------------------------
# Map
# ----------------------------------------------------------------------------

@dataclass
class GaussianMap:
    means: np.ndarray              # (N,3)
    scales: np.ndarray             # (N,3) std-devs along local x, y (tangent) and z (normal)
    quats: np.ndarray              # (N,4) local -> world rotation, (w,x,y,z)
    colors: np.ndarray             # (N,3) base colour in [0,1]
    opacity: np.ndarray            # (N,)
    normals: np.ndarray            # (N,3)
    counts: np.ndarray             # (N,) number of keyframes that observed it
    voxel: float = 0.05
    sh_rest: np.ndarray | None = None   # (N, K, 3) higher-order SH (loaded 3DGS models only)
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return int(self.means.shape[0])

    def subset(self, idx) -> "GaussianMap":
        return GaussianMap(self.means[idx], self.scales[idx], self.quats[idx], self.colors[idx], self.opacity[idx],
                           self.normals[idx], self.counts[idx], self.voxel,
                           None if self.sh_rest is None else self.sh_rest[idx], dict(self.meta))

    def rotations(self) -> np.ndarray:
        return quat_to_rotmat(self.quats)

    def covariances(self) -> np.ndarray:
        R = self.rotations()
        S2 = self.scales.astype(np.float64) ** 2
        return np.einsum("nij,nj,nkj->nik", R, S2, R)

    def keys(self) -> np.ndarray:
        return voxel_keys(self.means, self.voxel)

    # ------------------------------------------------------------ npz I/O
    def to_dict(self, prefix="g_"):
        d = {f"{prefix}{k}": getattr(self, k) for k in ("means", "scales", "quats", "colors", "opacity", "normals",
                                                          "counts")}
        d[f"{prefix}voxel"] = np.array(self.voxel)
        if self.sh_rest is not None:
            d[f"{prefix}sh_rest"] = self.sh_rest
        return d

    @classmethod
    def from_dict(cls, d, prefix="g_"):
        return cls(*(np.asarray(d[f"{prefix}{k}"]) for k in ("means", "scales", "quats", "colors", "opacity",
                                                               "normals", "counts")),
                   voxel=float(d[f"{prefix}voxel"]),
                   sh_rest=np.asarray(d[f"{prefix}sh_rest"]) if f"{prefix}sh_rest" in d else None)

    # ------------------------------------------------------------ 3DGS PLY
    def to_ply(self, path: str | Path, sh_degree: int | None = None) -> Path:
        """Write the standard 3DGS layout (log scales, logit opacity, SH DC colours).

        ``sh_degree`` pads the higher SH bands with zeros, for loaders that expect a fixed
        degree (the reference 3DGS code asserts 45 ``f_rest`` fields for degree 3).
        """
        path = Path(path)
        n = len(self)
        rest = self.sh_rest
        if sh_degree is not None:
            k = (sh_degree + 1) ** 2 - 1
            padded = np.zeros((n, k, 3), np.float32)
            if rest is not None:
                m = min(k, rest.shape[1])
                padded[:, :m] = rest[:, :m]
            rest = padded if k else None
        n_rest = 0 if rest is None else rest.shape[1] * 3
        names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
        names += [f"f_rest_{i}" for i in range(n_rest)]
        names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        arr = np.zeros(n, dtype=[(k, "<f4") for k in names])
        arr["x"], arr["y"], arr["z"] = self.means.T
        arr["nx"], arr["ny"], arr["nz"] = self.normals.T
        dc = (np.clip(self.colors, 0, 1) - 0.5) / SH_C0
        arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = dc.T
        if rest is not None:
            flat = np.transpose(rest, (0, 2, 1)).reshape(n, -1)   # channel-major like the reference code
            for i in range(n_rest):
                arr[f"f_rest_{i}"] = flat[:, i]
        op = np.clip(self.opacity, 1e-4, 1 - 1e-4)
        arr["opacity"] = np.log(op / (1 - op))
        ls = np.log(np.maximum(self.scales, 1e-7))
        arr["scale_0"], arr["scale_1"], arr["scale_2"] = ls.T
        arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = self.quats.T
        header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n"
        header += "".join(f"property float {k}\n" for k in names) + "end_header\n"
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(arr.tobytes())
        return path

    @classmethod
    def from_ply(cls, path: str | Path, voxel: float = 0.05, transform: np.ndarray | None = None) -> "GaussianMap":
        """Load a 3DGS ``.ply``. ``transform`` (4x4) maps the file's frame into a z-up metric frame.

        Higher-order SH bands are dropped when a rotation is applied, since
        they would need rotating too.
        """
        path = Path(path)
        with open(path, "rb") as f:
            header = b""
            while not header.endswith(b"end_header\n"):
                line = f.readline()
                if not line:
                    raise ValueError("not a PLY file")
                header += line
            lines = header.decode("ascii").splitlines()
            if "format binary_little_endian 1.0" not in lines:
                raise ValueError("only binary little-endian PLY is supported")
            n = int(next(ln.split()[-1] for ln in lines if ln.startswith("element vertex")))
            props = [ln.split() for ln in lines if ln.startswith("property")]
            typemap = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4", "short": "<i2"}
            dtype = [(p[2], typemap[p[1]]) for p in props]
            arr = np.frombuffer(f.read(n * np.dtype(dtype).itemsize), dtype=dtype, count=n)
        means = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
        colors = np.clip(0.5 + SH_C0 * np.stack([arr[f"f_dc_{i}"] for i in range(3)], axis=1), 0, 1)
        opacity = 1 / (1 + np.exp(-arr["opacity"].astype(np.float64)))
        scales = np.exp(np.stack([arr[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float64))
        quats = np.stack([arr[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        rest_names = sorted([p[2] for p in props if p[2].startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
        sh_rest = None
        if rest_names:
            flat = np.stack([arr[k] for k in rest_names], axis=1).astype(np.float64)
            k = flat.shape[1] // 3
            sh_rest = np.transpose(flat.reshape(n, 3, k), (0, 2, 1))
        R = quat_to_rotmat(quats)
        if transform is not None:
            T = np.asarray(transform, dtype=np.float64)
            s = np.cbrt(abs(np.linalg.det(T[:3, :3])))
            Rt = T[:3, :3] / s
            means = means @ T[:3, :3].T + T[:3, 3]
            scales = scales * s
            R = Rt[None] @ R
            quats = rotmat_to_quat(R)
            sh_rest = None
        # the normal of a flat Gaussian is its shortest axis
        short = np.argmin(scales, axis=1)
        normals = R[np.arange(n), :, short]
        return cls(means, scales, quats, colors, opacity, normals, np.full(n, 99, dtype=np.int32), voxel,
                   sh_rest=sh_rest, meta={"source": str(path)})


# ----------------------------------------------------------------------------
# Building a map from RGB-D keyframes
# ----------------------------------------------------------------------------

def frame_points(session, i: int, stride: int = 1, max_depth: float = 5.0, pose=None):
    """World-space points, colours, normals and view directions for one frame (edges removed)."""
    K = session.K
    depth = session.depth[i]
    mask = (depth > 0) & (depth < max_depth) & ~depth_edge_mask(depth, 0.06)
    pts_c, pix = backproject(K, depth, stride=stride, mask=mask)
    T = session.poses[i] if pose is None else pose
    R, t = T[:3, :3], T[:3, 3]
    pts_w = pts_c @ R.T.astype(np.float32) + t.astype(np.float32)
    n_c = depth_normals(K, depth).reshape(-1, 3)[pix]
    n_w = n_c @ R.T.astype(np.float32)
    rgb = session.rgb[i].reshape(-1, 3)[pix].astype(np.float32) / 255.0
    view = pts_c / np.maximum(np.linalg.norm(pts_c, axis=1, keepdims=True), 1e-6)
    cos_inc = np.abs(np.sum(n_c * view, axis=1))
    return pts_w, rgb, n_w, cos_inc, pts_c[:, 2]


def _accumulate(keys, w, cols):
    uk, inv = np.unique(keys, return_inverse=True)
    out = [np.bincount(inv, weights=w * c if c is not None else w, minlength=len(uk)) for c in cols]
    return uk, inv, out


def build_gaussian_map(session, voxel: float = 0.05, stride: int = 1, min_frames: int = 2,
                       max_depth: float = 5.0, exposure_passes: int = 1, poses=None) -> GaussianMap:
    """Fuse RGB-D keyframes into surfel Gaussians (one per occupied voxel)."""
    n_frames = len(session)
    P, C, N, Wt, F = [], [], [], [], []
    for i in range(n_frames):
        p, c, n, cos_inc, z = frame_points(session, i, stride, max_depth, None if poses is None else poses[i])
        ok = np.linalg.norm(n, axis=1) > 0.5
        p, c, n, cos_inc, z = p[ok], c[ok], n[ok], cos_inc[ok], z[ok]
        w = (np.clip(cos_inc, 0.15, 1.0) / (0.3 + z * z)).astype(np.float64)
        P.append(p)
        C.append(c)
        N.append(n)
        Wt.append(w)
        F.append(np.full(len(p), i, dtype=np.int32))
    P = np.concatenate(P).astype(np.float64)
    C = np.concatenate(C).astype(np.float64)
    N = np.concatenate(N).astype(np.float64)
    Wt = np.concatenate(Wt)
    F = np.concatenate(F)
    keys = voxel_keys(P, voxel)

    # frames per voxel (multi-view support)
    fk = np.unique(np.stack([keys, F.astype(np.int64)], axis=1), axis=0)
    uk_f, nfr = np.unique(fk[:, 0], return_counts=True)

    uk, inv = np.unique(keys, return_inverse=True)
    assert np.array_equal(uk, uk_f)
    sw = np.bincount(inv, weights=Wt)
    mean = np.stack([np.bincount(inv, weights=Wt * P[:, k]) for k in range(3)], axis=1) / sw[:, None]

    # exposure compensation: per-frame gain relative to the fused colour
    gain = np.ones(n_frames)
    lum = C.mean(axis=1)
    for _ in range(exposure_passes + 1):
        Cn = C / gain[F][:, None]
        col = np.stack([np.bincount(inv, weights=Wt * Cn[:, k]) for k in range(3)], axis=1) / sw[:, None]
        if _ == exposure_passes:
            break
        ref = col.mean(axis=1)[inv]
        ratio = lum / np.maximum(ref, 1e-3)
        for f in range(n_frames):
            m = (F == f) & (ref > 0.05)
            if m.sum() > 50:
                gain[f] = float(np.median(ratio[m]))
        gain /= np.median(gain)

    # covariance of points in each voxel -> surface normal
    d = P - mean[inv]
    cov = np.zeros((len(uk), 3, 3))
    for a in range(3):
        for b in range(a, 3):
            v = np.bincount(inv, weights=Wt * d[:, a] * d[:, b]) / sw
            cov[:, a, b] = v
            cov[:, b, a] = v
    nsum = np.stack([np.bincount(inv, weights=Wt * N[:, k]) for k in range(3)], axis=1)
    cnt = np.bincount(inv)
    evals, evecs = np.linalg.eigh(cov + np.eye(3)[None] * 1e-10)
    normal = evecs[:, :, 0]
    # where points are too few or the patch is not flat, trust the depth-map normals
    avg_n = nsum / np.maximum(np.linalg.norm(nsum, axis=1, keepdims=True), 1e-9)
    weak = (cnt < 6) | (evals[:, 1] < 4 * evals[:, 0] + 1e-8)
    normal = np.where(weak[:, None], avg_n, normal)
    flip = np.sum(normal * avg_n, axis=1) < 0
    normal[flip] *= -1

    keep = nfr >= min_frames
    # drop isolated voxels (depth noise): need >= 2 occupied neighbours among 26
    kk = uk[keep]
    from .geometry import ijk_to_keys, keys_to_ijk
    ijk = keys_to_ijk(kk)
    nb = np.zeros(len(kk), dtype=np.int32)
    offs = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
    sk = np.sort(kk)
    for o in offs:
        q = ijk_to_keys(ijk + np.array(o))
        pos = np.searchsorted(sk, q)
        pos = np.clip(pos, 0, len(sk) - 1)
        nb += (sk[pos] == q)
    keep_idx = np.nonzero(keep)[0][nb >= 2]

    mean, normal, col, nfr_k = mean[keep_idx], normal[keep_idx], col[keep_idx], nfr[keep_idx]
    # rotation whose z axis is the normal
    ref = np.where(np.abs(normal[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    t1 = np.cross(ref, normal)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(normal, t1)
    R = np.stack([t1, t2, normal], axis=2)
    quats = rotmat_to_quat(R)
    n = len(mean)
    scales = np.column_stack([np.full(n, 0.55 * voxel), np.full(n, 0.55 * voxel), np.full(n, 0.08 * voxel)])
    opacity = np.clip(0.55 + 0.1 * nfr_k, 0.6, 0.95)
    gm = GaussianMap(mean.astype(np.float32), scales.astype(np.float32), quats.astype(np.float32),
                     np.clip(col, 0, 1).astype(np.float32), opacity.astype(np.float32), normal.astype(np.float32),
                     nfr_k.astype(np.int32), voxel, meta={"exposure_gain": gain.tolist()})
    return gm


def estimate_floor(gm: GaussianMap) -> float:
    """Height of the lowest large horizontal surface."""
    up = gm.normals[:, 2] > 0.9
    z = gm.means[up, 2]
    if len(z) < 20:
        return float(np.percentile(gm.means[:, 2], 2))
    lo, hi = np.percentile(z, 0.5), np.percentile(z, 99.5)
    bins = np.arange(lo - 0.02, hi + 0.04, 0.02)
    h, e = np.histogram(z, bins)
    peak = h.max()
    for i in range(len(h)):
        if h[i] >= 0.25 * peak:
            j = i + int(np.argmax(h[i:i + 5]))
            return float((e[j] + e[j + 1]) / 2)
    return float(lo)


def neighbor_tree(gm: GaussianMap) -> cKDTree:
    return cKDTree(gm.means.astype(np.float64))
