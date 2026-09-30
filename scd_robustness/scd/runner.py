"""Shared runner logic: one manifest line -> method subprocess steps -> evaluation -> status file.

Standard library only (the evaluator is imported lazily), so it works in the O-SCD (3.12) and
MV3DCD (3.8) environments. Re-running a finished run is a no-op unless --force is given.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

STATUS = "_scd_status.json"


def cli(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--manifest", required=True, help="runs_<method>.jsonl written by scripts/scd_plan.py")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--index", type=int, help="0-based line index (use $SLURM_ARRAY_TASK_ID)")
    sel.add_argument("--run-id", help="run_id of the line to execute")
    ap.add_argument("--chunk", type=int, default=1,
                    help="process lines [index*chunk, (index+1)*chunk) sequentially in this job")
    ap.add_argument("--force", action="store_true", help="re-run even if the status file says ok")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--no-eval", action="store_true", help="skip evaluation after the method finishes")
    return ap.parse_args()


def load_specs(args):
    with open(args.manifest) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    if args.run_id:
        specs = [s for s in lines if s["run_id"] == args.run_id]
        if not specs:
            sys.exit("run_id not found in manifest: %s" % args.run_id)
        return specs
    start = args.index * args.chunk
    if start >= len(lines):
        sys.exit("index %d (chunk %d) is beyond the manifest (%d lines)" % (args.index, args.chunk, len(lines)))
    return lines[start:start + args.chunk]


def _now():
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _git_sha(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return out.splitlines()[0] if out else "unknown"
    except Exception:
        return "unavailable"


def is_done(out_dir):
    p = Path(out_dir) / STATUS
    return p.exists() and json.loads(p.read_text()).get("status") == "ok"


def write_status(out_dir, **fields):
    p = Path(out_dir) / STATUS
    data = json.loads(p.read_text()) if p.exists() else {}
    data.update(fields)
    p.write_text(json.dumps(data, indent=2))


def run_step(name, argv, cwd, env, log_dir):
    log_path = Path(log_dir) / ("%s.log" % name)
    t0 = time.time()
    with open(log_path, "w") as log:
        log.write("$ (cd %s && %s)\n\n" % (cwd, " ".join(argv)))
        log.flush()
        proc = subprocess.run(argv, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
    rec = {"step": name, "returncode": proc.returncode, "seconds": round(time.time() - t0, 2), "log": str(log_path)}
    if proc.returncode != 0:
        tail = log_path.read_text().splitlines()[-30:]
        raise RuntimeError("step %s failed (exit %d); last lines of %s:\n%s"
                           % (name, proc.returncode, log_path, "\n".join(tail)))
    return rec


def execute(spec, steps_fn, args, pre_fn=None, post_fn=None):
    """Run one manifest line. steps_fn(spec) -> list of (name, argv, env_overrides)."""
    out_dir = Path(spec["out_dir"])
    if is_done(out_dir) and not args.force:
        print("[scd] done already, skipping: %s" % spec["run_id"])
        return True
    steps = steps_fn(spec)
    if args.dry_run:
        print("[scd] %s" % spec["run_id"])
        for name, argv, extra_env in steps:
            env_txt = " ".join("%s=%s" % kv for kv in sorted(extra_env.items()))
            print("  [%s] (cd %s && %s %s)" % (name, spec["code_dir"], env_txt, " ".join(argv)))
        return True
    if args.force and out_dir.exists():
        shutil.rmtree(out_dir)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    scd_repo = Path(__file__).resolve().parents[1]
    write_status(out_dir, status="running", run_id=spec["run_id"], start=_now(), host=socket.gethostname(),
                 slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
                 gpu=_gpu_name(), python=sys.version.split()[0], method_git=_git_sha(spec["code_dir"]),
                 scd_git=_git_sha(scd_repo), spec=spec)
    records = []
    try:
        if pre_fn:
            pre_fn(spec)
        for name, argv, extra_env in steps:
            env = dict(os.environ)
            env.update({k: str(v) for k, v in extra_env.items()})
            records.append(run_step(name, argv, spec["code_dir"], env, log_dir))
            write_status(out_dir, steps=records)
        if post_fn:
            post_fn(spec)
        if spec.get("eval") and not args.no_eval:
            from .evaluate import evaluate_run
            t0 = time.time()
            results = evaluate_run(spec)
            records.append({"step": "evaluate", "returncode": 0, "seconds": round(time.time() - t0, 2)})
            write_status(out_dir, eval={k: {m: v.get(m) for m in ("miou", "mf1", "paper_miou", "far_mean",
                                                                    "n_pred_missing", "auprc")}
                                        for k, v in results.items()})
        write_status(out_dir, status="ok", end=_now(), steps=records,
                     seconds=round(sum(r["seconds"] for r in records), 2))
        print("[scd] ok: %s" % spec["run_id"])
        return True
    except Exception as e:
        write_status(out_dir, status="failed", end=_now(), steps=records, error=str(e),
                     traceback=traceback.format_exc())
        print("[scd] FAILED: %s\n%s" % (spec["run_id"], e), file=sys.stderr)
        return False


def main(steps_fn, description, pre_fn=None, post_fn=None):
    args = cli(description)
    ok = True
    for spec in load_specs(args):
        ok = execute(spec, steps_fn, args, pre_fn, post_fn) and ok
    sys.exit(0 if ok else 1)
