#!/usr/bin/env python
"""Build the perturbed dataset variants listed in an experiment's variants.jsonl (scd-tools env).

  python scripts/make_variants.py --experiment configs/experiments/e4_main.yaml --all --workers 8
  python scripts/make_variants.py --experiment ... --index $SLURM_ARRAY_TASK_ID --chunk 10

Idempotent: an existing variant built from the same config is skipped; a variant built from a
different config is an error unless --force is given.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scd.config import load_experiment, read_jsonl  # noqa: E402
from scd.variants import build_variant  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--manifest", help="default: <manifests>/<name>/variants.jsonl")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--all", action="store_true")
    sel.add_argument("--index", type=int)
    ap.add_argument("--chunk", type=int, default=1, help="lines per array task")
    ap.add_argument("--workers", type=int, default=None, help="image workers (default: perturbations.yaml io.workers)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    exp = load_experiment(args.experiment)
    manifest = Path(args.manifest or Path(exp["_paths"]["manifests"]) / exp["name"] / "variants.jsonl")
    specs = read_jsonl(manifest)
    if not args.all:
        specs = specs[args.index * args.chunk:(args.index + 1) * args.chunk]
    workers = args.workers or int(exp["_perturbations"].get("io", {}).get("workers", 1))

    failed = 0
    for spec in specs:
        label = "%s  %s/%s  %s" % (spec["variant_id"], spec["method"], spec["pair"], spec["scene"])
        if args.dry_run:
            print("[plan] %s -> %s" % (label, spec["out_dir"]))
            continue
        t0 = time.time()
        try:
            meta = build_variant(spec, exp["_paths"], exp["_perturbations"], force=args.force, workers=workers)
            print("[%s] %s  kept=%d dropped=%d perturbed=%d  (%.1fs)" % (
                meta["_status"], label, len(meta["frames_kept"]), len(meta["frames_dropped"]),
                len(meta["frames_perturbed"]), time.time() - t0))
        except Exception as e:  # keep going; report at the end
            failed += 1
            print("[FAILED] %s: %s" % (label, e), file=sys.stderr)
    if failed:
        sys.exit("%d variant(s) failed" % failed)


if __name__ == "__main__":
    main()
