#!/usr/bin/env python
"""Collect every run's status and evaluation summaries into one CSV per experiment (scd-tools env).

  python scripts/collect_results.py --experiment configs/experiments/e4_main.yaml
  -> <results>/<name>/summary.csv   (one row per run x output, including failed runs)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scd.config import load_experiment, read_jsonl  # noqa: E402

KEYS = ("run_id", "method", "arm", "scene", "pair", "null", "stressor", "severity", "trial", "seed", "variant_id")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True)
    args = ap.parse_args()
    exp = load_experiment(args.experiment)
    mdir = Path(exp["_paths"]["manifests"]) / exp["name"]
    rows = []
    for manifest in sorted(mdir.glob("runs_*.jsonl")):
        for spec in read_jsonl(manifest):
            out = Path(spec["out_dir"])
            status_p = out / "_scd_status.json"
            status = json.loads(status_p.read_text()) if status_p.exists() else {}
            base = {k: spec.get(k) for k in KEYS}
            base.update({"status": status.get("status", "not_run"), "seconds": status.get("seconds"),
                         "gpu": status.get("gpu"), "host": status.get("host")})
            summaries = sorted((out / "eval").glob("*_summary.json")) if (out / "eval").exists() else []
            if not summaries:
                rows.append(base)
                continue
            for s in summaries:
                data = json.loads(s.read_text())
                method_summary = data.pop("method_summary", {}) or {}
                row = dict(base)
                row.update(data)
                row.setdefault("output", s.name[:-len("_summary.json")])
                row.update({"m_" + k: v for k, v in method_summary.items()})
                rows.append(row)
    out_csv = Path(exp["_paths"]["results"]) / exp["name"] / "summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    done = sum(r["status"] == "ok" for r in rows)
    print("%d rows (%d ok) -> %s" % (len(rows), done, out_csv))


if __name__ == "__main__":
    main()
