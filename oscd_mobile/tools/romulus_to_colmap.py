"""Convert a Romulus capture (capture.json + frame_*.jpg + depth_*.bin + depthConfidence_*.bin) into the
COLMAP layout splatx trains from, like session_to_colmap.py does for the app's own scans.

Romulus: 1920×1080 JPEGs, per-frame pinhole intrinsics (fx fy cx cy), `pose4x4` column-major camera-to-world
in OpenGL axes (camera looks down −z, y up: the ARKit convention), LiDAR depth 256×144 float32 metres with a
uint8 confidence (0–2). The seed cloud is the high-confidence LiDAR of every frame, one point per 4 mm voxel.

Usage: python3 tools/romulus_to_colmap.py <capture dir> <out dir> [--stride N] [--depth]
"""
import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image


def rot_to_qvec(R):
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1)
        return [0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("out")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--depth", action="store_true", help="also write depth/<stem>.f32 at the image size (for --depth-weight)")
    a = ap.parse_args()
    # A chunked capture (uploaded in parts) has capture_chunk0.json, capture_chunk1.json, … instead of capture.json;
    # the frames continue across chunks in one ARKit world.
    if os.path.exists(os.path.join(a.capture, "capture.json")):
        man = json.load(open(os.path.join(a.capture, "capture.json")))
    else:
        chunks = sorted(f for f in os.listdir(a.capture) if f.startswith("capture_chunk") and f.endswith(".json"))
        if not chunks:
            raise SystemExit("no capture.json or capture_chunk*.json in " + a.capture)
        man = json.load(open(os.path.join(a.capture, chunks[0])))
        for c in chunks[1:]:
            man["frames"] += json.load(open(os.path.join(a.capture, c)))["frames"]
    assert man.get("coordinates", "opengl") == "opengl", man.get("coordinates")
    dw, dh = man["depthResolution"]
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "sparse", "0"), exist_ok=True)
    if a.depth:
        os.makedirs(os.path.join(a.out, "depth"), exist_ok=True)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    cam_lines, img_lines, pts, cols = [], [], [], []
    frames = man["frames"][:: max(a.stride, 1)]
    skipped = 0
    n = -1
    for fr in frames:
        W, H = fr["resolution"]
        fx, fy, cx, cy = fr["intrinsics"][:4]          # newer captures append a 5th value (0)
        name = fr["image"]
        try:                                           # a chunked upload can leave a truncated JPEG behind
            rgb = np.asarray(Image.open(os.path.join(a.capture, name)).convert("RGB"))
        except OSError:
            skipped += 1
            continue
        n += 1
        shutil.copyfile(os.path.join(a.capture, name), os.path.join(a.out, "images", name))
        c2w = np.array(fr["pose4x4"], dtype=np.float64).reshape(4, 4).T @ flip   # column-major → OpenCV axes
        w2c = np.linalg.inv(c2w)
        q, t = rot_to_qvec(w2c[:3, :3]), w2c[:3, 3]
        cam_lines.append(f"{n + 1} PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")
        img_lines.append(f"{n + 1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {n + 1} {name}\n\n")
        depth = np.fromfile(os.path.join(a.capture, fr["depth"]), dtype=np.float32).reshape(dh, dw)
        conf = np.fromfile(os.path.join(a.capture, fr["depthConfidence"]), dtype=np.uint8).reshape(dh, dw)
        # The depth map covers the sensor's full field. When the image has a different aspect (newer captures
        # store a 4:3 centre crop of a 16:9 stream), treat the image as a centre crop of a frame with the depth's
        # aspect: full width = H·dw/dh, cropped by (full − W)/2 on each side.
        full_w = H * dw / dh
        crop_x = (full_w - W) / 2
        s = dw / full_w
        fd, cxd, cyd = fx * s, (cx + crop_x + 0.5) * s - 0.5, (cy + 0.5) * s - 0.5
        v, u = np.nonzero((conf >= 2) & np.isfinite(depth) & (depth > 0))
        z = depth[v, u]
        cam = np.stack([(u - cxd) / fd * z, (v - cyd) / (fy * s) * z, z, np.ones_like(z)])
        world = (c2w @ cam)[:3].T
        uw = np.round((u + 0.5) / s - 0.5 - crop_x).astype(int)
        vw = np.clip(np.round((v + 0.5) / s - 0.5).astype(int), 0, H - 1)
        inside = (uw >= 0) & (uw < W)
        pts.append(world[inside])
        cols.append(rgb[vw[inside], uw[inside]])
        if a.depth:
            good = np.where((conf >= 2) & np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
            vv = np.clip(np.round((np.arange(H) + 0.5) * dh / H - 0.5).astype(int), 0, dh - 1)
            uu = np.clip(np.round((np.arange(W) + crop_x + 0.5) * s - 0.5).astype(int), 0, dw - 1)
            good[vv][:, uu].astype("<f4").tofile(os.path.join(a.out, "depth", os.path.splitext(name)[0] + ".f32"))
    pts, cols = np.concatenate(pts), np.concatenate(cols)
    _, first = np.unique(np.floor(pts / a.voxel).astype(np.int64), axis=0, return_index=True)
    pts, cols = pts[first], cols[first]
    with open(os.path.join(a.out, "sparse", "0", "cameras.txt"), "w") as fh:
        fh.writelines(cam_lines)
    with open(os.path.join(a.out, "sparse", "0", "images.txt"), "w") as fh:
        fh.writelines(img_lines)
    with open(os.path.join(a.out, "sparse", "0", "points3D.txt"), "w") as fh:
        for i, (p, c) in enumerate(zip(pts, cols)):
            fh.write(f"{i + 1} {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]} 0\n")
    print(f"{n + 1} images ({skipped} unreadable skipped), {len(pts)} points → {a.out}")


if __name__ == "__main__":
    main()
