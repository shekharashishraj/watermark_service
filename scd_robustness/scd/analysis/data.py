"""Load evaluated runs for the analysis and put every run on a common severity axis.

One row of `runs` = one evaluated output of one run (e.g. O-SCD online masks of run X).
`frames` / `components` rows point back to it through the column `row`, and so do the per-run
score histograms `pos[row]`, `neg[row]` (256 bins over p in [0, 1], summed over frames).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import interpolate, load_experiment, load_yaml, read_jsonl
from ..metrics import NBINS

SPEC_KEYS = ["run_id", "method", "arm", "scene", "pair", "null", "stressor", "severity", "trial", "seed",
             "variant_id"]
RUN_COLS = ["exp", "system", "scene", "pair", "null", "stressor", "severity", "trial", "seed", "axis", "group",
            "magnitude", "is_base"]


def load_config(path):
    path = Path(path).resolve()
    cfg = load_yaml(path)
    cfg["_file"] = str(path)
    cfg["_paths"] = interpolate(load_yaml((path.parent / cfg.get("paths", "paths.yaml")).resolve()))
    cfg["_experiments"] = {k: load_experiment((path.parent / v).resolve()) for k, v in cfg["experiments"].items()}
    cfg.setdefault("name", path.stem)
    return cfg


def system_key(cfg, method, arm, output):
    for key, s in cfg["systems"].items():
        if (s["method"], s.get("arm", "default"), s["output"]) == (method, arm, output):
            return key
    return "%s:%s:%s" % (method, arm, output)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


class Results:
    """Tables of one analysis. Build with load_results(); save()/load() cache them as files."""

    def __init__(self, status, runs, frames, components, pos, neg):
        self.status, self.runs, self.frames, self.components = status, runs, frames, components
        self.pos, self.neg = pos, neg

    def save(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.status.to_csv(out / "status.csv", index=False)
        self.runs.to_csv(out / "runs.csv", index=False)
        self.frames.to_csv(out / "frames.csv.gz", index=False)
        self.components.to_csv(out / "components.csv.gz", index=False)
        np.savez_compressed(out / "hists.npz", pos=self.pos, neg=self.neg)

    @classmethod
    def load(cls, out_dir):
        d = Path(out_dir)
        h = np.load(d / "hists.npz")
        runs = pd.read_csv(d / "runs.csv")
        runs["axis"] = runs["axis"].astype(object).where(runs["axis"].notna(), None)
        return cls(pd.read_csv(d / "status.csv"), runs, pd.read_csv(d / "frames.csv.gz", low_memory=False),
                   pd.read_csv(d / "components.csv.gz"), h["pos"], h["neg"])

    def with_run_cols(self, df, cols=RUN_COLS):
        cols = [c for c in cols if c in self.runs and c not in df]
        return df.merge(self.runs[["row"] + cols], on="row", how="left")


def load_results(cfg, only=None, log=print):
    """Read manifests and evaluation outputs of every experiment in the analysis config."""
    missing_as_zero = cfg.get("calibration", {}).get("missing_as_zero", True)
    status, runs, frames, comps, pos, neg = [], [], [], [], [], []
    for key, exp in cfg["_experiments"].items():
        if only and key not in only:
            continue
        mdir = Path(exp["_paths"]["manifests"]) / exp["name"]
        manifests = [m for m in sorted(mdir.glob("runs_*.jsonl")) if not m.stem.endswith("_refcache")]
        if not manifests:
            log("[analysis] %s: no manifests in %s (run scripts/scd_plan.py)" % (key, mdir))
        for manifest in manifests:
            for spec in read_jsonl(manifest):
                out = Path(spec["out_dir"])
                st = _read_json(out / "_scd_status.json")
                base = {"exp": key, "experiment": exp["name"], **{k: spec.get(k) for k in SPEC_KEYS}}
                srow = dict(base, status=st.get("status", "not_run"), seconds=st.get("seconds"),
                            gpu=st.get("gpu"), host=st.get("host"), n_eval=0)
                status.append(srow)
                if srow["status"] != "ok" or not spec.get("eval"):
                    continue
                for o in spec["eval"]["outputs"]:
                    prefix = str(out / "eval" / o["name"])
                    summ = _read_json(prefix + "_summary.json")
                    if not summ:
                        continue
                    srow["n_eval"] += 1
                    msum = summ.pop("method_summary", None) or {}
                    row_id = len(runs)
                    row = dict(base, row=row_id, output=o["name"],
                               system=system_key(cfg, spec["method"], spec["arm"], o["name"]),
                               work_scale=spec["eval"].get("work_scale"))
                    row.update({k: v for k, v in summ.items() if not isinstance(v, (dict, list))})
                    row.update({"m_" + k: v for k, v in msum.items() if not isinstance(v, (dict, list))})
                    runs.append(row)
                    fr = pd.read_csv(prefix + "_frames.csv") if Path(prefix + "_frames.csv").exists() else pd.DataFrame()
                    fr["row"] = row_id
                    fr["position"] = np.arange(len(fr))
                    frames.append(fr)
                    cp = Path(prefix + "_components.csv")
                    if cp.exists():
                        c = pd.read_csv(cp)
                        c["row"] = row_id
                        comps.append(c)
                    p, n = np.zeros(NBINS, np.int64), np.zeros(NBINS, np.int64)
                    hp = Path(prefix + "_hists.npz")
                    if hp.exists():
                        h = np.load(hp)
                        if len(h["pos"]):
                            p, n = h["pos"].sum(0).astype(np.int64), h["neg"].sum(0).astype(np.int64)
                    if (missing_as_zero and not spec.get("null") and len(fr)
                            and {"tp", "pred_missing", "score_missing"} <= set(fr.columns)):
                        miss = _truthy(fr.get("pred_missing")) & _truthy(fr.get("score_missing"))
                        p[0] += int((fr.loc[miss, "tp"] + fr.loc[miss, "fn"]).sum())
                        n[0] += int((fr.loc[miss, "fp"] + fr.loc[miss, "tn"]).sum())
                    pos.append(p)
                    neg.append(n)

    status = pd.DataFrame(status)
    runs = pd.DataFrame(runs)
    frames = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["row"])
    comps = pd.concat(comps, ignore_index=True) if comps else pd.DataFrame(
        columns=["row", "frame", "component", "area_frac", "detected_frac"])
    pos = np.stack(pos) if pos else np.zeros((0, NBINS), np.int64)
    neg = np.stack(neg) if neg else np.zeros((0, NBINS), np.int64)
    if len(runs):
        runs = add_derived(runs)
        runs = add_axes(runs, cfg)
    for col in ("pred_missing", "score_missing", "false_alarm_frame", "log_pose_ok"):
        if col in frames:
            frames[col] = _truthy(frames[col]).astype(float).where(frames[col].notna())
    return Results(status, runs, frames, comps, pos, neg)


def _truthy(series):
    if series is None:
        return pd.Series(dtype=bool)
    return series.map(lambda v: v is True or str(v).strip().lower() in ("true", "1", "1.0"))


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(np.nan, index=df.index)


def add_derived(runs):
    runs = runs.copy()
    runs["missing_rate"] = _num(runs, "n_pred_missing") / _num(runs, "n_frames")
    runs["pose_fail_rate"] = _num(runs, "m_n_pose_fail") / _num(runs, "m_n_frames")
    runs["thr_gap"] = _num(runs, "best_f1") - _num(runs, "f1_at_decision")
    runs["null"] = _truthy(runs["null"])
    return runs


def axis_of(stressor, severity, stressors_cfg):
    """(axis, group, magnitude) of a non-baseline run."""
    c = stressors_cfg.get(stressor, {})
    s = float(severity)
    kind = c.get("magnitude", "value")
    mag = abs(s) if kind == "abs" else (1.0 - s if kind == "one_minus" else s)
    axis = stressor
    if c.get("split_sign"):
        axis = stressor + ("_under" if s < 0 else "_over")
    return axis, stressor, round(mag, 6)


def add_axes(runs, cfg):
    """Mark baseline rows (first of cfg['baseline'] present per experiment and system; magnitude 0)
    and give every other row its axis / group / magnitude."""
    runs = runs.copy()
    order = cfg.get("baseline", ["identity", "clean"])
    scfg = cfg.get("stressors", {})
    runs["is_base"] = False
    for (exp, system), idx in runs.groupby(["exp", "system"]).groups.items():
        present = set(runs.loc[idx, "stressor"])
        base = next((b for b in order if b in present), None)
        if base is not None:
            runs.loc[idx[(runs.loc[idx, "stressor"] == base).to_numpy()], "is_base"] = True
    axes = [axis_of(st, sev, scfg) if (not b and st not in order) else (None, None, 0.0)
            for st, sev, b in zip(runs["stressor"], runs["severity"], runs["is_base"])]
    runs["axis"] = [a[0] for a in axes]
    runs["group"] = [a[1] for a in axes]
    runs["magnitude"] = [a[2] for a in axes]
    return runs


def axes_of(runs, exp, system):
    sel = runs[(runs["exp"] == exp) & (runs["system"] == system) & runs["axis"].notna()]
    return sorted(set(sel["axis"]), key=lambda a: _axis_sort_key(a))


def _axis_sort_key(axis):
    return (axis.split("_")[0], axis)


def curve_rows(runs, exp, system, axis=None, group=None):
    """Runs on one severity axis (or all axes of a group), with the baseline runs at magnitude 0."""
    sel = (runs["exp"] == exp) & (runs["system"] == system)
    on = (runs["axis"] == axis) if axis is not None else (runs["group"] == group)
    return runs[sel & (on | runs["is_base"])]


def severity_labels(rows):
    """magnitude -> label of the original severity (e.g. views magnitude 0.4 -> '0.6')."""
    out = {}
    for mag, sev, base in zip(rows["magnitude"], rows["severity"], rows["is_base"]):
        out.setdefault(mag, "base" if base else ("%g" % float(sev)))
    return out
