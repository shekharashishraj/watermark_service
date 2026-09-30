"""Build perturbed copies ("variants") of PASLCD scenes for O-SCD and MV3DCD.

A variant directory mirrors the original scene folder with symlinks and only writes the images
that change. It contains `variant.json`, which records exactly which frames were kept, dropped
and perturbed (with their random draws); the evaluator reads it.

Layouts (see papers_explained.md §6):
  O-SCD  (online zip):  <data_online>/<Instance_X>/<Scene>/{reference_scene, reference_reconstruction,
                                                           inference_scene/images, gt_mask}
  MV3DCD (offline zip): <data_offline>/<Scene>/<Instance_X>/{images (post-change have 'test' in the
                                                            name), sparse/0, gt_mask}
Output:
  O-SCD : <variants>/oscd/<variant_id>/<pair_dir>/<Scene>/
  MV3DCD: <variants>/mv3dcd/<variant_id>/<Scene>/<pair_dir>/
"""
from __future__ import annotations

import contextlib
import datetime
import io
import json
import os
import shutil
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import colmap_rw
from .config import fmt_severity, seed_from, stable_hash
from .imageio import list_images, read_rgb, write_rgb
from .perturb import apply_to_uint8

LAYOUT = {"oscd": "online", "mv3dcd": "offline"}


# ---------------------------------------------------------------------------------------------
# pairs
# ---------------------------------------------------------------------------------------------
def parse_pair(pair: str):
    """'Instance_1' -> ('real', 'Instance_1', None); 'null:Instance_1:Instance_2' -> ('null', map, revisit)."""
    if pair.startswith("null:"):
        _, map_inst, rev_inst = pair.split(":")
        return "null", map_inst, rev_inst
    return "real", pair, None


def pair_dirname(pair: str) -> str:
    kind, a, b = parse_pair(pair)
    return a if kind == "real" else "null-%s-%s" % (a, b)


def variant_dir(paths: dict, method: str, vid: str, pair: str, scene: str) -> Path:
    root = Path(paths["variants"]) / method / vid
    if LAYOUT[method] == "online":
        return root / pair_dirname(pair) / scene
    return root / scene / pair_dirname(pair)


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _require(*paths):
    for p in paths:
        if not Path(p).exists():
            raise FileNotFoundError(
                "missing %s — check configs/paths.yaml (data_online / data_offline) and the dataset layout" % p
            )


def _mirror(src_dir: Path, dst_dir: Path, skip=()):
    for entry in sorted(src_dir.iterdir()):
        if entry.name in skip:
            continue
        (dst_dir / entry.name).symlink_to(entry.resolve())


def _stem(name: str) -> str:
    return Path(name).stem


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _read_model(path):
    with contextlib.redirect_stdout(io.StringIO()):   # colmap_rw prints the detected format
        model = colmap_rw.read_model(str(path))
    if model is None:
        raise FileNotFoundError("no COLMAP model (cameras/images/points3D .bin or .txt) in %s" % path)
    return model


def camera_centers(sparse_dir: Path) -> dict:
    """{image stem: camera centre (world)} from a COLMAP model."""
    _, images, _ = _read_model(sparse_dir)
    out = {}
    for im in images.values():
        R = im.qvec2rotmat()
        out[_stem(im.name)] = (-R.T @ np.asarray(im.tvec)).tolist()
    return out


def write_points_ply(path: Path, points3D: dict):
    """PLY in the format MV3DCD's storePly/fetchPly use (xyz, normals, uchar rgb)."""
    pts = list(points3D.values())
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.zeros(len(pts), dtype=dtype)
    if pts:
        xyz = np.array([p.xyz for p in pts], np.float32)
        rgb = np.array([p.rgb for p in pts], np.uint8)
        arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(pts)
    header += "".join("property %s %s\n" % ("float" if t == "<f4" else "uchar", n) for n, t in dtype)
    header += "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


# ---------------------------------------------------------------------------------------------
# frame selection (coverage stressors) and burst subsets
# ---------------------------------------------------------------------------------------------
def select_frames(frames, st: dict, severity, trial: int, scene: str, stream: str, centers=None):
    """Return (kept, dropped, info). Frames keep their original order.

    random_nested: keep round(severity * n) frames; the subsets for smaller fractions are contained in
    those for larger fractions because they are prefixes of one seeded permutation.
    sector: drop round(severity * n) frames whose camera centres form one contiguous angular sector
    around the capture (angles in the best-fit plane of the camera centres).
    """
    n = len(frames)
    if st.get("kind") != "coverage":
        return list(frames), [], {}
    mode = st.get("mode", "random_nested")
    if mode == "random_nested":
        k = max(1, int(round(float(severity) * n)))
        rng = np.random.default_rng(seed_from("%s|%s|coverage|t%d" % (scene, stream, trial)))
        keep_idx = sorted(rng.permutation(n)[:k].tolist())
        info = {"mode": mode, "keep_frac": float(severity), "k": k}
    elif mode == "sector":
        if centers is None:
            raise ValueError("sector coverage needs camera centres (COLMAP model of the offline dataset)")
        stems = [_stem(f) for f in frames]
        missing = [s for s in stems if s not in centers]
        if missing:
            raise KeyError("no camera pose for frames %s in the offline COLMAP model" % missing[:5])
        pts = np.array([centers[s] for s in stems], np.float64)
        X = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        uv = X @ vt[:2].T
        order = np.argsort(np.arctan2(uv[:, 1], uv[:, 0]))
        k_drop = min(n - 1, int(round(float(severity) * n)))
        rng = np.random.default_rng(seed_from("%s|%s|sector|t%d" % (scene, stream, trial)))
        start = int(rng.integers(n))
        drop = {int(order[(start + i) % n]) for i in range(k_drop)}
        keep_idx = [i for i in range(n) if i not in drop]
        info = {"mode": mode, "drop_frac": float(severity), "k_drop": k_drop, "start": start}
    else:
        raise ValueError("unknown coverage mode %r" % mode)
    keep = set(keep_idx)
    return [frames[i] for i in keep_idx], [frames[i] for i in range(n) if i not in keep], info


def perturb_positions(n_kept: int, subset) -> list:
    """Indices (into the kept frames) that receive the photometric perturbation."""
    if not subset:
        return list(range(n_kept))
    count = min(int(subset.get("count", n_kept)), n_kept)
    pos = subset.get("position", "start")
    start = {"start": 0, "end": n_kept - count, "middle": (n_kept - count) // 2}[pos]
    return list(range(start, start + count))


# ---------------------------------------------------------------------------------------------
# image processing (runs in worker processes)
# ---------------------------------------------------------------------------------------------
def _init_worker():
    import cv2
    cv2.setNumThreads(1)


def _process_one(job):
    src, dst, fn, severity, params, seed, io_cfg = job
    rgb, alpha = read_rgb(src)
    out, info = apply_to_uint8(rgb, fn, severity, params, np.random.default_rng(seed))
    write_rgb(dst, out, alpha, io_cfg.get("jpeg_quality", 100), io_cfg.get("png_compression", 3))
    info = dict(info, seed=seed)
    return _stem(src.name), info, list(rgb.shape[:2])


# ---------------------------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------------------------
def build_variant(spec: dict, paths: dict, catalog: dict, force=False, workers=1) -> dict:
    """Create the variant described by `spec` (a line of variants.jsonl). Idempotent and atomic."""
    stressors = catalog["stressors"]
    io_cfg = catalog.get("io", {})
    st = stressors[spec["stressor"]]
    method, scene, pair = spec["method"], spec["scene"], spec["pair"]
    severity, trial = spec["severity"], int(spec["trial"])
    layout = LAYOUT[method]

    out = Path(spec["out_dir"])
    core = {k: spec[k] for k in ("method", "scene", "pair", "stressor", "severity", "trial")}
    spec_hash = stable_hash({"core": core, "stressor": st, "io": io_cfg})
    meta_path = out / "variant.json"
    if meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        if meta.get("spec_hash") == spec_hash:
            return dict(meta, _status="exists")
        raise RuntimeError("%s exists but was built from a different config; rebuild with --force" % out)

    tmp = out.parent / (out.name + ".tmp-%d" % os.getpid())
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    kind, a, b = parse_pair(pair)
    centers = None
    try:
        if layout == "online":
            root = Path(paths["data_online"])
            scene_src = root / a / scene
            if kind == "real":
                frames_dir, stream, skip = scene_src / "inference_scene" / "images", "inf", {"inference_scene"}
            else:
                frames_dir, stream, skip = root / b / scene / "reference_scene" / "images", "ref:" + b, \
                    {"inference_scene", "gt_mask"}
            _require(scene_src, frames_dir)
            _mirror(scene_src, tmp, skip)
            inf_out = tmp / "inference_scene"
            inf_out.mkdir()
            if kind == "real":
                _mirror(scene_src / "inference_scene", inf_out, {"images"})
            img_out = inf_out / "images"
            img_out.mkdir()
            frames = list_images(frames_dir)
            if st.get("kind") == "coverage" and st.get("mode") == "sector":
                inst = a if kind == "real" else b
                centers = camera_centers(Path(paths["data_offline"]) / scene / inst / "sparse" / "0")
        else:
            if kind != "real":
                raise NotImplementedError(
                    "no-change pairs for MV3DCD need COLMAP re-registration (pose protocol B, "
                    "experiment_plan.md §8.1–8.2); not implemented yet"
                )
            scene_src = Path(paths["data_offline"]) / scene / a
            frames_dir, stream = scene_src / "images", "inf"
            src_sparse = scene_src / "sparse" / "0"
            _require(scene_src, frames_dir, src_sparse)
            _mirror(scene_src, tmp, {"images", "sparse"})
            img_out = tmp / "images"
            img_out.mkdir()
            all_imgs = list_images(frames_dir)
            frames = [f for f in all_imgs if "test" in _stem(f)]
            for f in all_imgs:
                if f not in frames:
                    (img_out / f).symlink_to((frames_dir / f).resolve())
            if not frames:
                raise FileNotFoundError("no post-change ('test') images in %s" % frames_dir)
            if st.get("kind") == "coverage" and st.get("mode") == "sector":
                centers = camera_centers(src_sparse)

        kept, dropped, sel_info = select_frames(frames, st, severity, trial, scene, stream, centers)
        to_perturb = set()
        if st.get("kind") == "photometric":
            to_perturb = {kept[i] for i in perturb_positions(len(kept), st.get("subset"))}

        jobs = []
        for f in kept:
            src, dst = frames_dir / f, img_out / f
            if f in to_perturb:
                key = "%s|%s|%s|%s|%s|t%d" % (scene, stream, _stem(f), spec["stressor"], fmt_severity(severity), trial)
                jobs.append((src, dst, st["fn"], severity, st.get("params", {}), seed_from(key), io_cfg))
            else:
                dst.symlink_to(src.resolve())

        per_frame, native_size = {}, None
        if jobs:
            if workers > 1:
                with Pool(workers, initializer=_init_worker) as pool:
                    results = pool.map(_process_one, jobs, chunksize=1)
            else:
                results = [_process_one(j) for j in jobs]
            for stem, info, size in results:
                per_frame[stem] = info
                native_size = size

        if layout == "offline":
            _write_sparse(src_sparse, tmp / "sparse" / "0", {_stem(f) for f in dropped})

        meta = {
            "spec": core,
            "spec_hash": spec_hash,
            "variant_id": spec["variant_id"],
            "layout": layout,
            "stressor_cfg": st,
            "io": io_cfg,
            "source_scene": str(scene_src),
            "source_frames_dir": str(frames_dir),
            "stream": stream,
            "frames_all": [_stem(f) for f in frames],
            "frames_kept": [_stem(f) for f in kept],
            "frames_dropped": [_stem(f) for f in dropped],
            "frames_perturbed": sorted(_stem(f) for f in to_perturb),
            "selection": sel_info,
            "per_frame": per_frame,
            "native_size_hw": native_size,
            "scd_git": _git_sha(),
            "created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        (tmp / "variant.json").write_text(json.dumps(meta, indent=1))
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    os.rename(tmp, out)
    return dict(meta, _status="built")


def _write_sparse(src_sparse: Path, dst_sparse: Path, dropped_stems: set):
    """Symlink the COLMAP model, or write a filtered copy without the dropped post-change images.

    MV3DCD iterates the COLMAP image entries and opens every file, so dropped views must leave the
    model, not just the folder. It also converts points3D.bin to points3D.ply inside the source
    folder on first use; we create the .ply here so no job ever writes into shared data.
    """
    dst_sparse.mkdir(parents=True)
    if dropped_stems:
        cams, imgs, pts = _read_model(src_sparse)
        drop_ids = {iid for iid, im in imgs.items() if _stem(im.name) in dropped_stems}
        missing = dropped_stems - {_stem(imgs[i].name) for i in drop_ids}
        if missing:
            raise KeyError("dropped frames not found in COLMAP model: %s" % sorted(missing)[:5])
        imgs = {k: v for k, v in imgs.items() if k not in drop_ids}
        new_pts = {}
        for pid, p in pts.items():
            keep = np.array([iid not in drop_ids for iid in p.image_ids], bool)
            new_pts[pid] = p._replace(image_ids=p.image_ids[keep], point2D_idxs=p.point2D_idxs[keep])
        colmap_rw.write_model(cams, imgs, new_pts, str(dst_sparse), ext=".bin")
        pts = new_pts
    else:
        for entry in sorted(src_sparse.iterdir()):
            (dst_sparse / entry.name).symlink_to(entry.resolve())
        pts = None
    ply = dst_sparse / "points3D.ply"
    src_ply = src_sparse / "points3D.ply"
    if ply.exists() or ply.is_symlink():
        return
    if src_ply.exists():
        ply.symlink_to(src_ply.resolve())
    else:
        if pts is None:
            _, _, pts = _read_model(src_sparse)
        write_points_ply(ply, pts)
