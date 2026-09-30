"""Image read/write that keeps what the method loaders see.

cv2.IMREAD_UNCHANGED ignores EXIF orientation, exactly like O-SCD (cv2 IMREAD_UNCHANGED) and
MV3DCD (PIL without exif_transpose). Writing drops EXIF, so stored pixel order is preserved.
File names and formats are kept because both loaders look images up by name.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import IMAGE_EXTS


def list_images(folder) -> list:
    folder = Path(folder)
    return sorted(p.name for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def find_by_stem(folder, stem, exts=IMAGE_EXTS + (".npy",)):
    folder = Path(folder)
    for ext in exts:
        for cand in (folder / (stem + ext), folder / (stem + ext.upper())):
            if cand.exists():
                return cand
    return None


def read_rgb(path):
    """Return (rgb uint8 HxWx3, alpha uint8 HxW or None)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError("could not read image: %s" % path)
    if img.dtype != np.uint8:
        raise ValueError("only 8-bit images are supported: %s (%s)" % (path, img.dtype))
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), None
    alpha = img[..., 3].copy() if img.shape[2] == 4 else None
    return cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB), alpha


def write_rgb(path, rgb, alpha=None, jpeg_quality=100, png_compression=3):
    path = Path(path)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if alpha is not None:
        bgr = np.dstack([bgr, alpha])
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
        if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR") and hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444"):
            params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)]
    else:
        raise ValueError("unsupported output format: %s" % path)
    if not cv2.imwrite(str(path), bgr, params):
        raise IOError("could not write image: %s" % path)


def read_mask(path):
    """Binary mask (bool) from a PNG/JPG: >127 is foreground, like the released evaluators."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError("could not read mask: %s" % path)
    return m > 127
