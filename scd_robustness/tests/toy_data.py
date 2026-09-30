"""Tiny synthetic PASLCD look-alike in both layouts (online / offline) for CPU tests."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scd import INSTANCES, colmap_rw


def _texture(rng, h, w):
    base = rng.integers(0, 256, (h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    return cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)


def make_toy(root, scenes=("Garden", "Porch"), n_ref=4, n_inf=6, h=48, w=80):
    root = Path(root)
    online, offline = root / "online" / "PASLCD", root / "offline" / "PASLCD"
    for s_idx, scene in enumerate(scenes):
        rng = np.random.default_rng(100 + s_idx)
        scene_tex = _texture(rng, h, w)
        inf_names = ["IMG_%03d_test" % i for i in range(n_inf)]
        inf_imgs, gts = {}, {}
        for i, name in enumerate(inf_names):          # post-change capture shared by both instances
            img = scene_tex.copy()
            y0, x0 = 8 + i, 10 + 2 * i
            img[y0:y0 + 12, x0:x0 + 16] = (255, 0, 0)    # the "change"
            gt = np.zeros((h, w), np.uint8)
            gt[y0:y0 + 12, x0:x0 + 16] = 255
            inf_imgs[name], gts[name] = img, gt
        for inst_idx, inst in enumerate(INSTANCES):
            gain = 1.0 if inst_idx == 0 else 0.7          # Instance_2 reference: different lighting
            ref_names = ["REF%d_%03d" % (inst_idx + 1, i) for i in range(n_ref)]
            ref_imgs = {n: np.clip(scene_tex * gain, 0, 255).astype(np.uint8) for n in ref_names}

            d = online / inst / scene
            for sub in ("reference_scene/images", "inference_scene/images", "gt_mask",
                        "reference_reconstruction/point_cloud/iteration_30000", "reference_scene/sparse/0"):
                (d / sub).mkdir(parents=True, exist_ok=True)
            for n, img in ref_imgs.items():
                cv2.imwrite(str(d / "reference_scene/images" / (n + ".jpg")), img[..., ::-1])
            for n, img in inf_imgs.items():
                cv2.imwrite(str(d / "inference_scene/images" / (n + ".jpg")), img[..., ::-1])
                cv2.imwrite(str(d / "gt_mask" / (n + ".png")), gts[n])
            (d / "reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply").write_bytes(b"ply\n")

            o = offline / scene / inst
            (o / "images").mkdir(parents=True, exist_ok=True)
            (o / "gt_mask").mkdir(parents=True, exist_ok=True)
            (o / "sparse" / "0").mkdir(parents=True, exist_ok=True)
            for n, img in list(ref_imgs.items()) + list(inf_imgs.items()):
                cv2.imwrite(str(o / "images" / (n + ".jpg")), img[..., ::-1])
            for n in inf_names:
                cv2.imwrite(str(o / "gt_mask" / (n + ".png")), gts[n])
            write_colmap(o / "sparse" / "0", ref_names + inf_names, h, w)
            write_colmap(d / "reference_scene" / "sparse" / "0", ref_names, h, w)
    return online, offline


def write_colmap(path, names, h, w):
    cams = {1: colmap_rw.Camera(id=1, model="PINHOLE", width=w, height=h,
                                params=np.array([50.0, 50.0, w / 2, h / 2]))}
    images = {}
    for i, name in enumerate(names, start=1):
        ang = 2 * np.pi * (i - 1) / len(names)
        centre = np.array([np.cos(ang), 0.1 * i, np.sin(ang)]) * 3
        images[i] = colmap_rw.Image(id=i, qvec=np.array([1.0, 0, 0, 0]), tvec=-centre, camera_id=1,
                                    name=name + ".jpg", xys=np.zeros((0, 2)), point3D_ids=np.zeros(0, dtype=np.int64))
    pts = {}
    for pid in range(1, 6):
        pts[pid] = colmap_rw.Point3D(id=pid, xyz=np.array([0.1 * pid, 0, 0]), rgb=np.array([pid * 40, 0, 0]),
                                     error=0.5, image_ids=np.array(sorted(images)[:4], dtype=np.int64),
                                     point2D_idxs=np.zeros(4, dtype=np.int64))
    colmap_rw.write_model(cams, images, pts, str(path), ext=".bin")
