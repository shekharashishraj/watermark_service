#!/usr/bin/env python
"""Expand an experiment config into manifests (scd-tools env).

  python scripts/scd_plan.py --experiment configs/experiments/e4_main.yaml

Writes <manifests>/<name>/variants.jsonl, runs_<method>.jsonl (and runs_mv3dcd_refcache.jsonl when
MV3DCD reuses reference checkpoints) and prints the SLURM commands to submit them.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scd.config import load_experiment  # noqa: E402
from scd.plan import write_manifests  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True)
    args = ap.parse_args()
    exp = load_experiment(args.experiment)
    info = write_manifests(exp)
    print(json.dumps(info, indent=2))
    print("\nNext (on Sol):  bash slurm/submit_experiment.sh %s" % args.experiment)


if __name__ == "__main__":
    main()
