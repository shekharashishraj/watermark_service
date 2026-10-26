#!/usr/bin/env python
"""Export a recorded oscd_mobile session as a COLMAP text model (PINHOLE, ARKit poses, LiDAR points)
so the candidate splat engines can train on it."""
import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image


def rot_to_qvec(R):
    """Rotation matrix → COLMAP (qw, qx, qy, qz)."""
    m = R
    t = np.trace(m)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return np.array([0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
    q = np.zeros(4)
    q[0] = (m[k, j] - m[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (m[j, i] + m[i, j]) / s
    q[1 + k] = (m[k, i] + m[i, k]) / s
    return q


def lidar_normals(depth, conf, fd, cxd, cyd):
    """Unit normals (camera frame, facing the camera) per depth pixel from central differences over the
    LiDAR point grid; zero where a neighbour is missing or across a depth jump (> 5 cm)."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    ok = (conf >= 2) & np.isfinite(depth) & (depth > 0)
    P = np.stack([(u - cxd) / fd * depth, (v - cyd) / fd * depth, depth], -1)
    P[~ok] = np.nan
    N = np.zeros_like(P)
    dx = P[1:-1, 2:] - P[1:-1, :-2]
    dy = P[2:, 1:-1] - P[:-2, 1:-1]
    n = np.cross(dx, dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
    bad = ~np.isfinite(n).all(-1) | (np.linalg.norm(dx, axis=-1) > 0.05) | (np.linalg.norm(dy, axis=-1) > 0.05)
    n[bad] = 0
    N[1:-1, 1:-1] = n
    N[np.sum(N * np.nan_to_num(P), -1) > 0] *= -1
    return N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("out")
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--normals", action="store_true",
                    help="also write sparse/0/normals.txt: the patched splatx then starts each Gaussian as a disc on the "
                         "LiDAR surface (PocketGS). Measured no better than the isotropic start on a room scan, so off by default")
    ap.add_argument("--depth", action="store_true",
                    help="also write depth/<id>.f32 (W×H float32 metres at the training image size, 0 = none; LiDAR confidence "
                         "high, nearest-neighbour upsampled) for splatx --depth-weight")
    ap.add_argument("--full", action="store_true", help="train from the full-size <id>_full.jpg images when every keyframe has one")
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.session, "session.json")))
    K = man["intrinsics"]
    W, H, f, cx, cy = K["width"], K["height"], K["f"], K["cx"], K["cy"]
    dw, dh = man["depthWidth"], man["depthHeight"]
    roi = man.get("roi")
    full = a.full and all(os.path.exists(os.path.join(a.session, "keyframes", f"{kf['id']:06d}_full.jpg"))
                          for kf in man["keyframes"])
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "sparse", "0"), exist_ok=True)
    if a.depth:
        os.makedirs(os.path.join(a.out, "depth"), exist_ok=True)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    pts, cols, nrms, img_lines = [], [], [], []
    for n, kf in enumerate(man["keyframes"]):
        stem = os.path.join(a.session, "keyframes", f"{kf['id']:06d}")
        rgb = np.load(stem + "_rgb.npy")
        depth = np.load(stem + "_depth.npy")
        conf = np.load(stem + "_conf.npy")
        name = f"{kf['id']:06d}.jpg" if full else f"{kf['id']:06d}.png"
        if a.depth:
            tw, th = (man["captureWidth"], man["captureHeight"]) if full else (W, H)
            good = np.where((conf >= 2) & np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
            vv = np.clip(np.round((np.arange(th) + 0.5) * dh / th - 0.5).astype(int), 0, dh - 1)
            uu = np.clip(np.round((np.arange(tw) + 0.5) * dw / tw - 0.5).astype(int), 0, dw - 1)
            good[vv][:, uu].astype("<f4").tofile(os.path.join(a.out, "depth", f"{kf['id']:06d}.f32"))
        if full:
            shutil.copyfile(stem + "_full.jpg", os.path.join(a.out, "images", name))
        else:
            Image.fromarray(rgb).save(os.path.join(a.out, "images", name))
        c2w = np.array(kf["arkitCameraToWorld"], dtype=np.float64).reshape(4, 4).T @ flip   # column-major → OpenCV axes
        w2c = np.linalg.inv(c2w)
        q, t = rot_to_qvec(w2c[:3, :3]), w2c[:3, 3]
        img_lines.append(f"{n + 1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {name}\n\n")
        # LiDAR → world points: depth-map intrinsics are the working ones scaled by dw / W.
        s = dw / W
        fd, cxd, cyd = f * s, (cx + 0.5) * s - 0.5, (cy + 0.5) * s - 0.5
        v, u = np.nonzero((conf >= 2) & np.isfinite(depth) & (depth > 0))
        z = depth[v, u]
        cam = np.stack([(u - cxd) / fd * z, (v - cyd) / fd * z, z, np.ones_like(z)])
        world = (c2w @ cam)[:3].T
        normal = (c2w[:3, :3] @ lidar_normals(depth, conf, fd, cxd, cyd)[v, u].T).T
        uw = np.clip(np.round((u + 0.5) / s - 0.5).astype(int), 0, W - 1)
        vw = np.clip(np.round((v + 0.5) / s - 0.5).astype(int), 0, H - 1)
        pts.append(world)
        cols.append(rgb[vw, uw])
        nrms.append(normal)
    pts, cols, nrms = np.concatenate(pts), np.concatenate(cols), np.concatenate(nrms)
    if roi:
        lo, hi = np.array(roi["min"]), np.array(roi["max"])
        keep = np.all((pts >= lo) & (pts <= hi), axis=1)
        pts, cols, nrms = pts[keep], cols[keep], nrms[keep]
    _, first = np.unique(np.floor(pts / a.voxel).astype(np.int64), axis=0, return_index=True)
    pts, cols, nrms = pts[first], cols[first], nrms[first]
    with open(os.path.join(a.out, "sparse", "0", "cameras.txt"), "w") as fh:
        if full:   # the camera at the capture size; pixel centres scale as (c + 0.5)·s − 0.5
            k = man["captureWidth"] / W
            fh.write(f"1 PINHOLE {man['captureWidth']} {man['captureHeight']} {f * k} {f * k} {(cx + .5) * k - .5} {(cy + .5) * k - .5}\n")
        else:
            fh.write(f"1 PINHOLE {W} {H} {f} {f} {cx} {cy}\n")
    with open(os.path.join(a.out, "sparse", "0", "images.txt"), "w") as fh:
        fh.writelines(img_lines)
    if a.normals:
        with open(os.path.join(a.out, "sparse", "0", "normals.txt"), "w") as fh:
            fh.writelines(f"{n[0]:.5f} {n[1]:.5f} {n[2]:.5f}\n" for n in nrms)
    with open(os.path.join(a.out, "sparse", "0", "points3D.txt"), "w") as fh:
        for i, (p, c) in enumerate(zip(pts, cols)):
            fh.write(f"{i + 1} {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]} 0\n")
    print(f"{len(man['keyframes'])} images, {len(pts)} points → {a.out}")


if __name__ == "__main__":
    main()
