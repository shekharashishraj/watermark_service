"""Build the project overview page (single self-contained HTML file)."""
import base64
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/watermark_service/scene-change-3d")
import numpy as np  # noqa: E402

from scene_change.harness.scenario import load_scenario  # noqa: E402
from scene_change.pipeline import build, inspect  # noqa: E402
from scene_change.report import report_data  # noqa: E402

SP = Path("/tmp/claude-0/-home-user-watermark-service/36501a67-21c0-53c2-a496-4630204028bb/scratchpad")
REPO = Path("/home/user/watermark_service")
OUT = SP / "dashboard" / "scene_change_3d.html"


def jsonl(p):
    p = Path(p)
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


# ---------------------------------------------------------------- 3D viewer data
sc = load_scenario(SP / "dev" / "s100_mixed")
model = build(sc["baseline"], rooms=sc["rooms"])
res = inspect(model, sc["inspection"])
viewer = report_data(model, sc["inspection"], res, max_points=70_000)

# ---------------------------------------------------------------- results
variants = [r for r in jsonl(SP / "oscd_results.jsonl") if r["scenario"] == "s100_mixed"]
dev_methods = {r["variant"]: r for r in variants}
bench = jsonl(REPO / "scene-change-3d" / "runs" / "bench" / "results.jsonl")

VARIANT_LABEL = {
    "pnp-map": "As published: own PnP poses",
    "pnp-map-exp": "Own PnP poses + exposure matching",
    "ours-map": "Our localization poses",
    "gt-map": "True poses",
    "gt-map-exp": "True poses + exposure matching",
    "gt-oracle": "True poses + perfect reference render",
    "gt-oracle-exp": "True poses + perfect render + exposure matching",
}
oscd_rows = []
for key in ("pnp-map", "pnp-map-exp", "ours-map", "gt-map", "gt-map-exp", "gt-oracle", "gt-oracle-exp"):
    r = dev_methods.get(key)
    oscd_rows.append({"key": key, "label": VARIANT_LABEL[key], "done": r is not None,
                      "iou": r["online"]["IoU"] if r else None, "f1": r["online"]["F1"] if r else None,
                      "far": r["online"]["false_alarm_rate"] if r else None,
                      "pose_cm": r["registration"]["trans_cm_median"] if r else None,
                      "localized": r.get("localized") if r else None, "frames": r.get("frames") if r else None,
                      "fps": r["timings"]["fps"] if r else None})

ours = dev_methods["ours"]
v2d = dev_methods.get("video2d")
done_oscd = [o for o in oscd_rows if o["done"]]
faithful = next((o for o in oscd_rows if o["key"] == "pnp-map" and o["done"]), None)
best_oscd = max(done_oscd, key=lambda o: o["f1"]) if done_oscd else None
method_rows = [
    {"label": "Ours: 3D geometry + appearance", "f1": ours["all"]["F1"], "iou": ours["all"]["IoU"],
     "far": ours["all"]["false_alarm_rate"], "hi": True},
    {"label": "Ours: confirmed items only", "f1": ours["confirmed"]["F1"], "iou": ours["confirmed"]["IoU"],
     "far": ours["confirmed"]["false_alarm_rate"], "hi": True},
    {"label": "2D video comparison", "f1": v2d["all"]["F1"], "iou": v2d["all"]["IoU"],
     "far": v2d["all"]["false_alarm_rate"], "hi": False},
]
if faithful:
    method_rows.append({"label": "O-SCD as published (own poses)", "f1": faithful["f1"], "iou": faithful["iou"],
                        "far": faithful["far"], "hi": False})
if best_oscd and (not faithful or best_oscd["key"] != "pnp-map"):
    method_rows.append({"label": f"O-SCD best variant ({best_oscd['label'].lower()})", "f1": best_oscd["f1"],
                        "iou": best_oscd["iou"], "far": best_oscd["far"], "hi": False})

def bench_list(rows):
    out = []
    for r in rows:
        f = r["frames"]
        out.append({"scene": f"s{r['seed']:03d} {r['kind']}", "kind": r["kind"], "method": r["method"],
                    "iou": f["IoU"], "f1": f["F1"], "far": f["false_alarm_rate"],
                    "recall": r.get("objects", {}).get("recall"),
                    "fp_conf": r.get("objects", {}).get("false_positives_confirmed"),
                    "fp_all": r.get("objects", {}).get("false_positives"),
                    "reg_cm": r.get("registration", {}).get("trans_cm_median"),
                    "seconds": r["seconds"]})
    return out


bench_v1 = bench_list(jsonl(REPO / "scene-change-3d" / "runs" / "bench" / "results_ours_v1.jsonl"))
bench_v2 = bench_list(jsonl(REPO / "scene-change-3d" / "runs" / "bench" / "results_ours_v2.jsonl"))
bench_rows = []
for r in bench:
    f = r["frames"]
    bench_rows.append({"scene": f"s{r['seed']:03d} {r['kind']}", "kind": r["kind"], "method": r["method"],
                       "iou": f["IoU"], "f1": f["F1"], "far": f["false_alarm_rate"],
                       "recall": r.get("objects", {}).get("recall"),
                       "fp_conf": r.get("objects", {}).get("false_positives_confirmed"),
                       "fp_all": r.get("objects", {}).get("false_positives"),
                       "reg_cm": r.get("registration", {}).get("trans_cm_median"),
                       "seconds": r["seconds"]})

# ---------------------------------------------------------------- code map
def lines(p):
    return sum(1 for _ in open(p))

MODULES = [
    ("scene_change/harness/scene.py", "Procedural 5-room apartments from ~40 object builders"),
    ("scene_change/harness/raycast.py", "CPU ray caster: textures, room lighting, lamps"),
    ("scene_change/harness/changes.py", "Removed, added, moved, restyled items; stains and scuffs"),
    ("scene_change/harness/walkthrough.py", "Walk planning through doorways, sensor noise, VIO drift"),
    ("scene_change/harness/scenario.py", "Scenario kinds: mixed, clean, partial"),
    ("scene_change/harness/evaluate.py", "Ground-truth masks and every metric"),
    ("scene_change/harness/benchmark.py", "Multi-scene benchmark CLI"),
    ("scene_change/harness/oscd_export.py", "Export to the official O-SCD data layout"),
    ("scene_change/gaussians.py", "RGB-D fusion into surfel Gaussians, 3DGS .ply I/O"),
    ("scene_change/render.py", "EWA splatting renderer with sparse blend weights"),
    ("scene_change/model.py", "Baseline model: Gaussians, keyframe depth, rooms"),
    ("scene_change/register.py", "Floor-plan FFT search, ICP, gated drift correction"),
    ("scene_change/detect.py", "Missing / added / moved / restyled, unverified areas, verdict"),
    ("scene_change/report.py", "HTML inspection report with 3D viewer"),
    ("scene_change/baseline2d.py", "2D video-comparison baseline"),
    ("scene_change/oscd.py", "O-SCD re-implementation (paper arXiv:2511.12370)"),
    ("scene_change/__main__.py", "CLI: generate, inspect, export-oscd"),
    ("scene_change/stress.py", "Stressors: motion blur, exposure, relighting, coverage loss, map compression"),
    ("scene_change/calibration.py", "ECE, reliability, AP/AUROC, threshold sweeps, isotonic recalibration"),
    ("scene_change/harness/robustness.py", "Severity sweeps over held-out scenes, all methods"),
    ("scene_change/harness/robustness_report.py", "Severity curves, failure boundaries, calibration, offline vs online"),
    ("scene_change/harness/paslcd.py", "PASLCD: degrade scenes, patch and score the official code"),
    ("tests/test_oscd.py", "Unit tests: gradients, change field, PnP, renderer"),
    ("tests/test_pipeline.py", "Metric definitions and an end-to-end harness run"),
    ("tests/test_robustness.py", "Stressors, calibration metrics, PASLCD tools, official-code patches"),
]
root = REPO / "scene-change-3d"
code = [{"path": p, "lines": lines(root / p), "what": w} for p, w in MODULES]
commits = subprocess.run(["git", "-C", str(REPO), "log", "--format=%h %s", "-2"], capture_output=True,
                         text=True).stdout.strip().splitlines()

data = {
    "viewer": viewer,
    "methods": method_rows,
    "oscd": oscd_rows,
    "bench": bench_rows,
    "bench_v1": bench_v1,
    "bench_v2": bench_v2,
    "code": code,
    "commits": commits,
    "total_lines": sum(c["lines"] for c in code),
    "registration": [  # s100_clean, median translation error per room (cm)
        {"room": "Living room", "before": 4.5, "ungated": 1.7, "after": 1.4},
        {"room": "Kitchen", "before": 4.7, "ungated": 1.1, "after": 1.1},
        {"room": "Hallway", "before": 3.3, "ungated": 1.8, "after": 1.8},
        {"room": "Bedroom", "before": 8.0, "ungated": 0.8, "after": 0.8},
        {"room": "Bathroom", "before": 1.3, "ungated": 9.1, "after": 1.3},
    ],
    "dev_objects": [  # dev scenes after the drift fix
        {"scene": "s100 mixed", "found": 7, "visible": 7, "fp_conf": 0, "fp_review": 6, "f1": 0.829},
        {"scene": "s100 partial", "found": 7, "visible": 8, "fp_conf": 0, "fp_review": 4, "f1": 0.745},
        {"scene": "s100 clean", "found": 0, "visible": 0, "fp_conf": 0, "fp_review": 5, "f1": None},
        {"scene": "s101 mixed", "found": 8, "visible": 8, "fp_conf": 0, "fp_review": 8, "f1": 0.804},
        {"scene": "s101 partial", "found": 8, "visible": 8, "fp_conf": 0, "fp_review": 10, "f1": 0.666},
        {"scene": "s101 clean", "found": 0, "visible": 0, "fp_conf": 0, "fp_review": 6, "f1": None},
    ],
    "update": [
        {"label": "Reference map, unchanged", "changed": 12.9, "unchanged": 15.8, "note": ""},
        {"label": "Updated with true change masks", "changed": 21.2, "unchanged": 15.7, "note": "11k Gaussians removed, 5.7k added"},
        {"label": "Updated with O-SCD masks", "changed": 21.7, "unchanged": 15.5, "note": "58% of the map rebuilt"},
    ],
}
# ---------------------------------------------------------------- robustness study
from scene_change.harness.robustness_report import Records, load_records, summary_data  # noqa: E402

RUNS = REPO / "scene-change-3d" / "runs" / "robust"
recs = load_records(RUNS)
robust = summary_data(Records(recs)) if recs else None
expected = {"fast": 5 * 26 * 3, "oscd": 5 * 30 * 2, "compress": 5 * 4 * 2, "depth": 5 * 4 * 2}


def unique_runs(names):
    return len({(r["seed"], r["stressor"], r["level"], r["method"]) for n in names for r in jsonl(RUNS / n)
                if "error" not in r})


counts = {"fast": unique_runs(["results_fast.jsonl"]), "oscd": unique_runs(["results_oscd.jsonl", "results_oscd2.jsonl"]),
          "compress": unique_runs(["results_compress.jsonl"]), "depth": unique_runs(["results_depth.jsonl"])}
done_all = all(counts[k] >= expected[k] for k in expected)
data["robust"] = robust
data["robust_progress"] = {"counts": counts, "expected": expected, "done": done_all,
                           "records": len(recs), "seeds_oscd": sorted({r["seed"] for r in recs if r["method"].startswith("oscd")})}
notes_file = SP / "dashboard" / "robust_notes.json"
notes = json.loads(notes_file.read_text()) if notes_file.exists() else {}
data["robust_findings"] = notes.get("findings", [])
gates = {}
for k in ("depth", "exposure"):
    f = REPO / "scene-change-3d" / "docs" / "robustness" / f"{k}_gate.json"
    if f.exists():
        gates[k] = json.loads(f.read_text())
data["robust_gates"] = gates
data["robust_notes"] = notes.get("notes", {})
share = sum(min(counts[k], expected[k]) for k in expected) / sum(expected.values())

cues = "data:image/jpeg;base64," + base64.b64encode((SP / "dashboard" / "cues.jpg").read_bytes()).decode()
template = (SP / "dashboard" / "template.html").read_text()
page = (template.replace("__CUES__", cues)
        .replace("__ROBUST_PILL_CLASS__", "ok" if done_all else "run")
        .replace("__ROBUST_PILL__", "Robustness sweeps finished" if done_all else f"Robustness sweeps running ({round(100 * share)}%)")
        .replace("__DATA__", json.dumps(data, separators=(",", ":"), default=float).replace("</", "<\\/")))
OUT.write_text(page)
print(OUT, round(OUT.stat().st_size / 1e6, 2), "MB")
