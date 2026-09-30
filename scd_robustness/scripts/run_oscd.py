#!/usr/bin/env python
"""Run patched O-SCD for manifest line(s), then evaluate (oscd env, GPU).

  python scripts/run_oscd.py --manifest $SCD_ROOT/manifests/e4_main/runs_oscd.jsonl --index $SLURM_ARRAY_TASK_ID
  python scripts/run_oscd.py --manifest ... --index 0 --dry-run        # print the command only
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scd import IMAGE_EXTS, runner  # noqa: E402


def _n_frames(data_dir):
    d = Path(data_dir) / "inference_scene" / "images"
    return sum(1 for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS) if d.exists() else None


def steps(spec):
    a = spec["method_args"]
    spf = a["steps_per_frame"]
    if spf == "auto":   # iteration-matched arm: total online steps ~= refine budget
        n = _n_frames(spec["data_dir"])
        spf = max(1, int(round(a["refine_total"] / float(n)))) if n else "auto"
    argv = [sys.executable, "oscd.py", "-s", spec["data_dir"], "-m", spec["out_dir"],
            "--resolution", str(a["resolution"]), "--test_hold", str(a["test_hold"]),
            "--seed", str(spec["seed"]), "--steps_per_frame", str(spf), "--refine_total", str(a["refine_total"])]
    for flag in ("refine", "save_scores", "frame_log"):
        if a.get(flag):
            argv.append("--" + flag)
    return [("oscd", argv, {})]


def check_patched(spec):
    cfg = Path(spec["code_dir"]) / "arguments" / "config_args.py"
    if "--steps_per_frame" not in cfg.read_text():
        raise RuntimeError("O-SCD at %s is not patched; run setup/01_fetch_code.sh" % spec["code_dir"])


if __name__ == "__main__":
    runner.main(steps, __doc__, pre_fn=check_patched)
