"""PASLCD robustness study for the official implementations (run on a GPU machine).

Two layouts are supported:

``oscd``    PASLCD_online, as read by the official O-SCD code
            (github.com/Chumsy0725/O-SCD): ``<Instance>/<Scene>/`` holding
            ``reference_scene/``, ``reference_reconstruction/``, ``inference_scene/images/``
            and ``gt_mask/``.
``mv3dcd``  PASLCD, as read by the official MV3DCD code (github.com/Chumsy0725/MV3DCD):
            ``<Scene>/<Instance>/`` holding ``images/`` (post-change images carry "test" in
            their name), ``sparse/0/`` and ``gt_mask/``.

Commands::

    # a degraded copy of one scene: only post-change images change, the rest is symlinked
    python -m scene_change.harness.paslcd perturb --layout oscd --scene data/PASLCD/Instance_1/Cantina \
        --out variants/blur_0.04/Instance_1/Cantina --stressor blur --level 0.04

    # score an official output with the harness metrics (PASLCD frame scores, calibration)
    python -m scene_change.harness.paslcd score --gt data/PASLCD/Instance_1/Cantina/gt_mask \
        --pred output/.../renders/change_mask --scores output/.../renders/change_score \
        --method oscd-official-online --scene Instance_1/Cantina --stressor blur --level 0.04 \
        --out runs/paslcd/results_oscd_Instance_1_Cantina_blur_0.04.jsonl

    # insert the continuous-score dumps (and MV3DCD's view hold-out) into an official checkout
    python -m scene_change.harness.paslcd patch --method oscd --repo O-SCD

Stressors are those of ``scene_change.stress`` except ``relight`` (harness only). For
``coverage``/``sparse``, O-SCD simply never sees the removed frames; frames with a ground
truth mask are kept unless ``--allow-gt-drop``, so every level is scored on the same
frames. MV3DCD needs all images on disk (they are registered in its COLMAP model), so the
removed post-change views are listed in ``holdout.txt`` and the patched ``train_masks.py``
leaves them out of change-channel training (``SCD_HOLDOUT``) while they are still rendered
and scored: change localisation for views the method never used.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np

from .. import calibration as cal
from ..stress import STRESSORS, drop_views, exposure, motion_blur
from .evaluate import frame_metrics, pixel_metrics

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


# ----------------------------------------------------------------------------
# Perturbation
# ----------------------------------------------------------------------------

def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix in IMAGE_EXT)


def image_motion_directions(paths: list[Path], seed: int = 0, size: int = 256) -> np.ndarray:
    """Per-image direction of apparent motion from phase correlation with its neighbours (random if unclear)."""
    rng = np.random.default_rng(seed)
    out = rng.uniform(0, np.pi, len(paths))
    grays = []
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        s = size / max(g.shape)
        grays.append(np.float32(cv2.resize(g, (int(g.shape[1] * s), int(g.shape[0] * s)), interpolation=cv2.INTER_AREA)))
    win = None
    for i in range(len(paths)):
        shifts = []
        for j in (i - 1, i + 1):
            if 0 <= j < len(paths) and grays[j].shape == grays[i].shape:
                if win is None or win.shape != grays[i].shape:
                    win = cv2.createHanningWindow(grays[i].shape[::-1], cv2.CV_32F)
                (dx, dy), resp = cv2.phaseCorrelate(grays[i], grays[j], win)
                if resp > 0.05:
                    shifts.append(np.array([dx, dy]) * (1 if j > i else -1))
        if shifts:
            v = np.mean(shifts, axis=0)
            if np.hypot(*v) > 0.5:
                out[i] = np.arctan2(v[1], v[0])
    return out


def _write_like(src: Path, dst: Path, rgb: np.ndarray):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    params = [cv2.IMWRITE_JPEG_QUALITY, 98] if src.suffix.lower() in (".jpg", ".jpeg") else []
    cv2.imwrite(str(dst), bgr, params)


def _link(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(os.path.relpath(src.resolve(), dst.parent.resolve()))


def perturb_images(paths: list[Path], out_dir: Path, stressor: str, level: float, seed: int = 0,
                   directions: np.ndarray | None = None) -> None:
    """Degraded copies of ``paths`` in ``out_dir`` (same file names); level 0 symlinks the originals."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if stressor == "blur" and level > 0 and directions is None:
        directions = image_motion_directions(paths, seed)
    for k, p in enumerate(paths):
        dst = out_dir / p.name
        if stressor in ("none", "coverage", "sparse") or level == 0:
            _link(p, dst)
            continue
        rgb = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if stressor == "blur":
            rgb = motion_blur(rgb, level * rgb.shape[1], directions[k])
        elif stressor in ("dark", "bright"):
            rgb = exposure(rgb, level)
        else:
            raise ValueError(f"stressor {stressor!r} is not available on PASLCD")
        _write_like(p, dst, rgb)


def drop_names(names: list[str], stressor: str, level: float, protected: set, seed: int = 0) -> list[str]:
    """Names removed by ``coverage``/``sparse`` at ``level``; protected names (ground truth) are never removed."""
    cand = [n for n in names if Path(n).stem not in protected]
    if not cand or level <= 0:
        return []
    keep = set(drop_views(len(cand), level, mode="segments" if stressor == "coverage" else "uniform", seed=seed))
    return [n for i, n in enumerate(cand) if i not in keep]


def make_variant(layout: str, scene: Path, out: Path, stressor: str, level: float, seed: int = 0,
                 allow_gt_drop: bool = False) -> dict:
    """Write a degraded copy of one PASLCD scene; returns the variant description (also saved as variant.json)."""
    scene, out = Path(scene), Path(out)
    if stressor not in STRESSORS and stressor != "none":
        raise ValueError(f"unknown stressor {stressor!r}")
    if stressor == "relight":
        raise ValueError("relight needs the harness renderer; it is not available on PASLCD")
    out.mkdir(parents=True, exist_ok=True)
    gt_names = {p.stem for p in list_images(scene / "gt_mask")} if (scene / "gt_mask").exists() else set()
    protected = set() if allow_gt_drop else gt_names
    info = {"layout": layout, "scene": str(scene), "stressor": stressor, "level": level, "seed": seed}
    if layout == "oscd":
        for child in scene.iterdir():
            if child.name != "inference_scene":
                _link(child, out / child.name)
        inf = scene / "inference_scene"
        (out / "inference_scene").mkdir(parents=True, exist_ok=True)
        for child in inf.iterdir():
            if child.name != "images":
                _link(child, out / "inference_scene" / child.name)
        paths = list_images(inf / "images")
        removed = set(drop_names([p.name for p in paths], stressor, level, protected, seed)) \
            if stressor in ("coverage", "sparse") else set()
        kept = [p for p in paths if p.name not in removed]
        perturb_images(kept, out / "inference_scene" / "images", stressor, level, seed)
        info.update(images=len(paths), removed=sorted(removed))
    elif layout == "mv3dcd":
        for child in scene.iterdir():
            if child.name != "images":
                _link(child, out / child.name)
        paths = list_images(scene / "images")
        post = [p for p in paths if "test" in p.name]
        pre = [p for p in paths if "test" not in p.name]
        (out / "images").mkdir(parents=True, exist_ok=True)
        for p in pre:
            _link(p, out / "images" / p.name)
        held = drop_names([p.name for p in post], stressor, level, protected, seed) \
            if stressor in ("coverage", "sparse") else []
        # not a video: blur directions are random per image
        dirs = np.random.default_rng(seed).uniform(0, np.pi, len(post)) if stressor == "blur" else None
        perturb_images(post, out / "images", stressor, level, seed, directions=dirs)
        (out / "holdout.txt").write_text(",".join(Path(n).stem for n in held))
        info.update(images=len(post), removed=sorted(held))
    else:
        raise ValueError(f"unknown layout {layout!r}")
    info["protected_gt"] = sorted(protected)
    (out / "variant.json").write_text(json.dumps(info, indent=1))
    return info


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

def _read_mask(path: Path, shape=None) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if shape is not None and m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


def _read_score(path: Path, shape) -> np.ndarray:
    if path.suffix == ".npy":
        s = np.asarray(np.load(path), np.float32)
        if s.ndim == 3:
            s = s.mean(axis=0) if s.shape[0] in (1, 3) else s.mean(axis=-1)
    else:
        s = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        if s.ndim == 3:
            s = s.mean(axis=-1)
        s /= 65535.0 if s.max() > 255 else 255.0
    if s.shape != shape:
        s = cv2.resize(s, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(s, 0.0, 1.0)


def _find(folder: Path | None, stem: str, exts=(".png", ".npy", ".jpg")) -> Path | None:
    if folder is None:
        return None
    for e in exts:
        p = folder / f"{stem}{e}"
        if p.exists():
            return p
    return None


def score_outputs(gt_dir: Path, pred_dir: Path, score_dir: Path | None = None, threshold: float = 0.5) -> dict:
    """Harness metrics for one official run: GT masks vs predicted binary masks (and continuous scores)."""
    gts = list_images(Path(gt_dir))
    G, P, S, names, missing = [], [], [], [], []
    for g in gts:
        pp = _find(Path(pred_dir), g.stem)
        if pp is None:
            missing.append(g.stem)
            continue
        gm = _read_mask(g)
        G.append(gm)
        P.append(_read_mask(pp, gm.shape))
        sp = _find(score_dir, g.stem, (".npy", ".png"))
        S.append(None if sp is None else _read_score(sp, gm.shape))
        names.append(g.stem)
    if not G:
        raise SystemExit(f"no predictions in {pred_dir} match the masks in {gt_dir}")
    shape = G[0].shape
    same = all(g.shape == shape for g in G)
    rec = {"frames_scored": names, "missing": missing, "n_frames": len(G)}
    if same:
        gt, pred = np.stack(G), np.stack(P)
    else:   # differing resolutions: score frame by frame at a common size
        H, W = shape
        gt = np.stack([cv2.resize(g.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0 for g in G])
        pred = np.stack([cv2.resize(p.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0 for p in P])
    ax = (1, 2)
    rec.update(frames=frame_metrics(pred, gt), pixels=pixel_metrics(pred, gt),
               per_frame={"tp": np.sum(pred & gt, axis=ax).astype(int).tolist(),
                          "fp": np.sum(pred & ~gt, axis=ax).astype(int).tolist(),
                          "fn": np.sum(~pred & gt, axis=ax).astype(int).tolist()})
    fp = pred & ~gt
    has = gt.any(axis=ax)
    rec["errors"] = {"tp_px": int(np.sum(pred & gt)), "fn_px": int(np.sum(~pred & gt)), "fp_px": int(fp.sum()),
                     "fp_near_px": 0, "fp_far_px": int(fp.sum()), "fp_changefree_px": int(fp[~has].sum()),
                     "px": int(gt.size), "changefree_frames": int((~has).sum()),
                     "changefree_frames_flagged": int(np.sum(np.sum(fp[~has], axis=ax) >= 20))}
    if all(s is not None for s in S):
        sc = np.stack([cv2.resize(s, (gt.shape[2], gt.shape[1])) if s.shape != gt.shape[1:] else s for s in S])
        rec["calibration"] = cal.calibration_report(sc, gt, threshold)
    rec["changes"] = []
    return rec


# ----------------------------------------------------------------------------
# Patching the official code
# ----------------------------------------------------------------------------

MARK = "# scd-robustness patch"


def patch_oscd(repo: Path) -> bool:
    """Save O-SCD's continuous change render (before the 0.5 threshold) as float16 .npy, online and refined."""
    f = Path(repo) / "oscd.py"
    s = f.read_text()
    if MARK in s:
        return False
    anchor = 'os.makedirs(os.path.join(renders_path, "change_mask"), exist_ok=True)'
    assert s.count(anchor) == 1, "O-SCD layout changed: makedirs anchor not found"
    s = s.replace(anchor, anchor + f'  {MARK}\n    for _d in ("change_score", "change_score_refined"):\n'
                  f'        os.makedirs(os.path.join(renders_path, _d), exist_ok=True)', 1)
    pat = re.compile(r"\n([ \t]+)change_mask = change_mask\.mean\(dim=0\)\n")
    hits = list(pat.finditer(s))
    assert len(hits) == 2, f"O-SCD layout changed: expected 2 mask renders, found {len(hits)}"
    for m, sub in reversed(list(zip(hits, ("change_score", "change_score_refined")))):
        ind = m.group(1)
        ins = (f"{ind}np.save(os.path.join(renders_path, \"{sub}\", f\"{{view.image_name}}.npy\"), "
               f"change_mask.clamp(0, 1).half().cpu().numpy())  {MARK}\n")
        s = s[:m.end()] + ins + s[m.end():]
    f.write_text(s)
    return True


def patch_mv3dcd(repo: Path) -> bool:
    """MV3DCD: save the continuous change mask next to the binary one, and honour SCD_HOLDOUT in training."""
    repo = Path(repo)
    changed = False
    f = repo / "render_viewpoints.py"
    s = f.read_text()
    if MARK not in s:
        anchor = "final_image_mask = (final_image_mask * 255).astype(np.uint8)"
        assert s.count(anchor) == 1, "MV3DCD layout changed: mask anchor not found"
        i = s.index(anchor)
        ind = s[s.rfind("\n", 0, i) + 1:i]
        ins = (f"\n{ind}os.makedirs(os.path.join(scene.model_path, folder_n, \"score_masks\"), exist_ok=True)  {MARK}"
               f"\n{ind}cv2.imwrite(os.path.join(scene.model_path, folder_n, \"score_masks\", "
               f"\"{{}}.png\".format(test_viewpoint.image_name)), final_image_mask.mean(axis=-1).astype(np.uint8))")
        s = s.replace(anchor, anchor + ins, 1)
        f.write_text(s)
        changed = True
    f = repo / "train_masks.py"
    s = f.read_text()
    if MARK not in s:
        anchor = "test_cameras = scene.getTestCameras()"
        assert s.count(anchor) == 1, "MV3DCD layout changed: test camera anchor not found"
        i = s.index(anchor)
        ind = s[s.rfind("\n", 0, i) + 1:i]
        ins = (f"\n{ind}_hold = set(filter(None, os.environ.get(\"SCD_HOLDOUT\", \"\").split(\",\")))  {MARK}"
               f"\n{ind}test_cameras = [c for c in test_cameras if c.image_name not in _hold]")
        s = s.replace(anchor, anchor + ins, 1)
        if not re.search(r"^import os\b", s, re.M):
            s = "import os  " + MARK + "\n" + s
        f.write_text(s)
        changed = True
    return changed


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("perturb", help="write a degraded copy of one scene")
    p.add_argument("--layout", choices=["oscd", "mv3dcd"], required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stressor", required=True, choices=["none"] + [k for k in STRESSORS if k != "relight"])
    p.add_argument("--level", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-gt-drop", action="store_true")
    s = sub.add_parser("score", help="score one official run with the harness metrics")
    s.add_argument("--gt", required=True)
    s.add_argument("--pred", required=True)
    s.add_argument("--scores", default=None)
    s.add_argument("--method", required=True)
    s.add_argument("--scene", required=True, help="scene id, e.g. Instance_1/Cantina")
    s.add_argument("--stressor", default="none")
    s.add_argument("--level", type=float, default=0.0)
    s.add_argument("--threshold", type=float, default=0.5)
    s.add_argument("--out", required=True, help="results .jsonl to append to")
    q = sub.add_parser("patch", help="add score dumps (and the MV3DCD hold-out) to an official checkout")
    q.add_argument("--method", choices=["oscd", "mv3dcd"], required=True)
    q.add_argument("--repo", required=True)
    lv = sub.add_parser("levels", help="print 'stressor level' lines for the job list")
    lv.add_argument("--stressors", nargs="+", default=["blur", "dark", "bright", "coverage", "sparse"])
    lv.add_argument("--preset", choices=["full", "quick"], default="full",
                    help="quick: the middle and the most severe level of each stressor")
    args = ap.parse_args(argv)
    if args.cmd == "perturb":
        info = make_variant(args.layout, Path(args.scene), Path(args.out), args.stressor, args.level, args.seed,
                            args.allow_gt_drop)
        print(json.dumps({k: v for k, v in info.items() if k not in ("protected_gt",)}))
    elif args.cmd == "score":
        rec = score_outputs(Path(args.gt), Path(args.pred), Path(args.scores) if args.scores else None,
                            args.threshold)
        rec.update(method=args.method, seed=args.scene, kind="paslcd", stressor=args.stressor,
                   level=args.level if args.stressor != "none" else 0.0, dataset="PASLCD")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        fr = rec["frames"]
        print(f"{args.scene} {args.stressor}={args.level} {args.method}: F1 {fr['F1']} IoU {fr['IoU']} "
              f"({rec['n_frames']} frames, {len(rec['missing'])} missing)")
    elif args.cmd == "patch":
        done = patch_oscd(Path(args.repo)) if args.method == "oscd" else patch_mv3dcd(Path(args.repo))
        print("patched" if done else "already patched")
    elif args.cmd == "levels":
        print("none 0")
        for name in args.stressors:
            levels = list(STRESSORS[name].levels)
            if args.preset == "quick":
                levels = sorted({levels[len(levels) // 2], levels[-1]}, key=levels.index)
            for v in levels:
                print(f"{name} {v}")


if __name__ == "__main__":
    main()
