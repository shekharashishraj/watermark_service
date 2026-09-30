#!/usr/bin/env python
"""Run patched MV3DCD (the steps of its run.sh) for manifest line(s), then evaluate (mv3dcd env, GPU).

  # 1) reference checkpoints, once per scene/instance/seed
  python scripts/run_mv3dcd.py --manifest .../runs_mv3dcd_refcache.jsonl --index $SLURM_ARRAY_TASK_ID
  # 2) variant runs, which link the cached reference instead of retraining it
  python scripts/run_mv3dcd.py --manifest .../runs_mv3dcd.jsonl --index $SLURM_ARRAY_TASK_ID

The reference 3DGS is trained on pre-change images only (train.py skips names containing 'test'),
so perturbing or dropping post-change images leaves it unchanged and it can be reused.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scd import runner  # noqa: E402


def steps(spec):
    a = spec["method_args"]
    d, o, inst = spec["data_dir"], spec["out_dir"], spec["pair"]
    py, R, T = sys.executable, str(a["resolution"]), str(a["t"])
    it0, it1, it2 = str(a["ref_iters"]), str(a["change_iters"]), str(a["aug_iters"])
    env = {"SCD_SEED": spec["seed"], "SCD_SAVE_SCORES": "1" if a.get("save_scores") else "0"}
    out = []
    if a.get("reference") == "train":
        out.append(("train_reference", [py, "train.py", "-s", d, "-m", o, "--iterations", it0, "--change", inst,
                                        "--resolution", R, "--checkpoint_iterations", it0, "--save_iterations", it0], env))
    if a.get("refcache_only"):
        return out
    out += [
        ("render_reference", [py, "render_viewpoints.py", "-s", d, "-m", o, "--iterations", it0, "--change", inst,
                              "--resolution", R], env),
        ("candidate_masks", [py, "create_mask.py", "--t", T, "--input_folder", o], env),
        ("train_change", [py, "train_masks.py", "-s", d, "-m", o, "--iterations", it1, "--change", inst,
                          "--checkpoint_iterations", it1, "--start_checkpoint", "%s/chkpnt%s.pth" % (o, it0),
                          "--resolution", R], env),
        ("render_augment", [py, "render_viewpoints.py", "-s", d, "-m", o, "--iterations", it1, "--change", inst,
                            "--resolution", R, "--aug", "True"], env),
        ("candidate_masks_aug", [py, "create_mask.py", "--t", T, "--input_folder", o], env),
        ("train_change_aug", [py, "train_masks.py", "-s", d, "-m", o, "--iterations", it2, "--change", inst,
                              "--checkpoint_iterations", it2, "--start_checkpoint", "%s/chkpnt%s.pth" % (o, it1),
                              "--resolution", R, "--augment", "True"], env),
        ("render_final", [py, "render_viewpoints.py", "-s", d, "-m", o, "--iterations", it2, "--change", inst,
                          "--resolution", R, "--mask", "True"], env),
    ]
    return out


def pre(spec):
    code = Path(spec["code_dir"])
    if "SCD_SAVE_SCORES" not in (code / "render_viewpoints.py").read_text():
        raise RuntimeError("MV3DCD at %s is not patched; run setup/01_fetch_code.sh" % code)
    a = spec["method_args"]
    if a.get("reference") != "cache":
        return
    ref, out, it = Path(spec["reference_cache"]), Path(spec["out_dir"]), a["ref_iters"]
    if not runner.is_done(ref):
        raise RuntimeError("reference cache %s is missing or failed; run runs_mv3dcd_refcache.jsonl first" % ref)
    (out / "point_cloud").mkdir(parents=True, exist_ok=True)
    links = {out / "point_cloud" / ("iteration_%s" % it): ref / "point_cloud" / ("iteration_%s" % it),
             out / ("chkpnt%s.pth" % it): ref / ("chkpnt%s.pth" % it)}
    for dst, src in links.items():
        if not src.exists():
            raise FileNotFoundError("reference cache incomplete: %s" % src)
        if not dst.exists():
            dst.symlink_to(src.resolve())
    # small files are copied, not linked: later steps rewrite them in the run folder
    for name in ("cfg_args", "cameras.json", "input.ply"):
        if (ref / name).exists():
            shutil.copy2(ref / name, out / name)


if __name__ == "__main__":
    runner.main(steps, __doc__, pre_fn=pre)
