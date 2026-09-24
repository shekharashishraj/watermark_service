"""Export a harness scenario in the layout of the official O-SCD code, and read its masks back.

Layout (github.com/Chumsy0725/O-SCD, ``--source_path``)::

    reference_scene/images/00000.png ...          baseline walkthrough RGB
    reference_scene/sparse/0/{cameras,images,points3D}.txt   COLMAP text model
    inference_scene/images/00000.png ...          inspection walkthrough RGB
    reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply
    gt_change_masks/00000.png ...                 harness ground truth (for utils/evaluate.py)

The reference poses are the baseline session poses, i.e. the frame of the exported
Gaussian map. After running ``oscd.py`` on a GPU machine, ``load_official_masks`` reads
``<model_path>/renders/change_mask`` (or ``change_mask_refined``) for ``frame_metrics``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..gaussians import GaussianMap, rotmat_to_quat
from ..geometry import invert_pose
from .evaluate import gt_masks


def _write_images(rgb: np.ndarray, folder: Path) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    names = []
    for i, im in enumerate(rgb):
        name = f"{i:05d}.png"
        cv2.imwrite(str(folder / name), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        names.append(name)
    return names


def write_colmap_text(folder: Path, K: np.ndarray, width: int, height: int, poses_wc: np.ndarray, names):
    """COLMAP text model with one PINHOLE camera (COLMAP puts pixel centres at +0.5)."""
    folder.mkdir(parents=True, exist_ok=True)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2] + 0.5, K[1, 2] + 0.5
    (folder / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {width} {height} {fx:.10f} {fy:.10f} {cx:.10f} {cy:.10f}\n")
    lines = ["# Image list with two lines of data per image:",
             "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
             "#   POINTS2D[] as (X, Y, POINT3D_ID)"]
    for i, (T, name) in enumerate(zip(poses_wc, names)):
        T_cw = invert_pose(T)
        q = rotmat_to_quat(T_cw[None, :3, :3])[0]
        t = T_cw[:3, 3]
        lines.append(f"{i + 1} {q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f} {t[0]:.12f} {t[1]:.12f} {t[2]:.12f} 1 {name}")
        lines.append("")
    (folder / "images.txt").write_text("\n".join(lines) + "\n")
    (folder / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")


def export_oscd_dataset(sc: dict, reference: GaussianMap, out: str | Path) -> Path:
    """Write ``sc`` (a loaded harness scenario) and the reference map for the official O-SCD code."""
    out = Path(out)
    base, insp = sc["baseline"], sc["inspection"]
    H, W = base.rgb.shape[1:3]
    ref_names = _write_images(base.rgb, out / "reference_scene" / "images")
    write_colmap_text(out / "reference_scene" / "sparse" / "0", base.K, W, H, base.poses, ref_names)
    inf_names = _write_images(insp.rgb, out / "inference_scene" / "images")
    ply = out / "reference_reconstruction" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    ply.parent.mkdir(parents=True, exist_ok=True)
    reference.to_ply(ply, sh_degree=3)
    gt = gt_masks(sc["gt_arrays"], sc["gt"]["changes"])
    gdir = out / "gt_change_masks"
    gdir.mkdir(parents=True, exist_ok=True)
    for name, m in zip(inf_names, gt):
        cv2.imwrite(str(gdir / name), m.astype(np.uint8) * 255)
    return out


def load_official_masks(folder: str | Path, n_frames: int, shape) -> np.ndarray:
    """Binary masks written by the official ``oscd.py`` (``<name>.png``, 255 = change)."""
    folder = Path(folder)
    masks = np.zeros((n_frames,) + tuple(shape), bool)
    for i in range(n_frames):
        f = folder / f"{i:05d}.png"
        if f.exists():
            m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if m.shape != tuple(shape):
                m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            masks[i] = m > 127
    return masks
