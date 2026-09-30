#!/usr/bin/env python
"""Statistics for the whole study (scd-tools env): gates, degradation curves, H1-H5, Holm.

  python scripts/analyze.py --config configs/analysis.yaml
  python scripts/analyze.py --config configs/analysis.yaml --only gates,h1 --n-boot 2000   # quick look
  -> <results>/analysis/<name>/{report.md, results.json, tables/*.csv, data/}

Works on partial results: whatever has finished and been evaluated is analysed, the rest is listed
in the report's coverage table. Figures: scripts/make_figures.py (reads tables/).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scd.analysis.data import load_config  # noqa: E402
from scd.analysis.report import SECTIONS, run_analysis  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "analysis.yaml"))
    ap.add_argument("--out", help="output directory (default: <results>/analysis/<name>)")
    ap.add_argument("--only", help="comma-separated subset of: %s" % ",".join(SECTIONS))
    ap.add_argument("--n-boot", type=int, help="override stats.n_boot (e.g. 1000 for a quick look)")
    ap.add_argument("--reuse-data", action="store_true",
                    help="reuse <out>/data from the previous call instead of re-reading every run")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out) if args.out else Path(cfg["_paths"]["results"]) / "analysis" / cfg["name"]
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    bad = set(only or []) - set(SECTIONS)
    if bad:
        ap.error("unknown sections %s" % sorted(bad))
    t0 = time.time()
    run_analysis(cfg, out, only=only, n_boot=args.n_boot, reuse_data=args.reuse_data)
    print("[analysis] done in %.0f s -> %s" % (time.time() - t0, out / "report.md"))


if __name__ == "__main__":
    main()
