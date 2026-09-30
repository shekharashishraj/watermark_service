#!/usr/bin/env python
"""(Re-)evaluate finished runs without re-running the method (any env with NumPy + OpenCV).

  python scripts/evaluate_run.py --manifest .../runs_oscd.jsonl --all
  python scripts/evaluate_run.py --manifest .../runs_oscd.jsonl --index 3
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scd.config import read_jsonl  # noqa: E402
from scd.evaluate import evaluate_run  # noqa: E402
from scd.runner import is_done  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--all", action="store_true")
    sel.add_argument("--index", type=int)
    args = ap.parse_args()
    specs = read_jsonl(args.manifest)
    if not args.all:
        specs = [specs[args.index]]
    for spec in specs:
        if not spec.get("eval"):
            continue
        if not is_done(spec["out_dir"]):
            print("[skip, not finished] %s" % spec["run_id"])
            continue
        res = evaluate_run(spec)
        print(spec["run_id"], json.dumps({k: {m: v.get(m) for m in ("miou", "mf1", "far_mean", "auprc")}
                                          for k, v in res.items()}))


if __name__ == "__main__":
    main()
