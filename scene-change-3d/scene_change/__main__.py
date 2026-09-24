"""Command line: generate harness scenarios and inspect them.

    python -m scene_change generate --seed 3 --kind mixed --out runs/s003_mixed
    python -m scene_change inspect runs/s003_mixed --out runs/s003_mixed/result [--oscd]

``inspect`` builds the baseline from the scenario's first walkthrough, inspects the
second one, and writes ``summary.json``, ``metrics.json`` (harness ground truth),
``report.html`` and the baseline as a 3DGS ``baseline.ply``. ``--oscd`` also runs the
O-SCD re-implementation and adds its scores and change field to the outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _generate(args):
    from .harness.scenario import build_scenario, save_scenario
    sc = build_scenario(args.seed, args.kind)
    out = save_scenario(sc, args.out)
    print(f"wrote {out}: {len(sc['baseline'])} baseline frames, {len(sc['inspection'])} inspection frames, "
          f"{len(sc['gt']['changes'])} changes")


def _inspect(args):
    from .harness.evaluate import evaluate_3d, frame_metrics, gt_masks, predicted_masks
    from .harness.scenario import load_scenario
    from .pipeline import build, inspect
    from .report import write_report
    sc = load_scenario(args.scenario)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = build(sc["baseline"], rooms=sc["rooms"])
    model.gaussians.to_ply(out / "baseline.ply")
    result = inspect(model, sc["inspection"])
    (out / "summary.json").write_text(json.dumps(result.summary(), indent=1))
    metrics = evaluate_3d(sc, model, result)
    gt = gt_masks(sc["gt_arrays"], sc["gt"]["changes"])
    metrics["frames"] = frame_metrics(predicted_masks(result, sc["inspection"], model.gaussians.voxel), gt)
    oscd = None
    if args.oscd:
        from .harness.evaluate import registration_errors
        from .oscd import OSCDConfig, run_oscd
        oscd = run_oscd(model.gaussians, sc["baseline"], sc["inspection"], OSCDConfig(backbone=args.backbone),
                        cache_dir=out / "featcache")
        T_wb = np.linalg.inv(np.asarray(sc["gt"]["T_baseline_world"]))
        metrics["oscd"] = {"frames": frame_metrics(oscd.masks, gt),
                           "frames_refined": frame_metrics(oscd.masks_refined, gt),
                           "registration": registration_errors(oscd.poses, sc["gt_arrays"]["insp_poses_world"], T_wb),
                           "backbone": oscd.backbone, "timings": oscd.timings}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    write_report(model, sc["inspection"], result, out / "report.html", title=args.title or Path(args.scenario).name,
                 oscd=oscd)
    f = metrics["frames"]
    print(f"{result.verdict}: {len(result.changes)} changes; frame IoU {f['IoU']}, F1 {f['F1']}; "
          f"object recall {metrics['objects']['recall']}; report {out / 'report.html'}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m scene_change", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="render a harness scenario (baseline + inspection walkthroughs)")
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--kind", default="mixed", choices=["clean", "mixed", "partial"])
    g.add_argument("--out", required=True)
    g.set_defaults(fn=_generate)
    i = sub.add_parser("inspect", help="build the baseline, inspect, score against ground truth, write a report")
    i.add_argument("scenario")
    i.add_argument("--out", required=True)
    i.add_argument("--title", default=None)
    i.add_argument("--oscd", action="store_true", help="also run the O-SCD re-implementation")
    i.add_argument("--backbone", default="auto", help="O-SCD feature backbone: auto | sam2 | dense-sift | none")
    i.set_defaults(fn=_inspect)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
