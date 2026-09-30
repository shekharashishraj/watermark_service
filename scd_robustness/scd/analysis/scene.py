"""Scene-level values (the statistical unit, experiment_plan.md §10) and degradation curves.

Run-summary metrics: mean over trials and seeds within an instance, then mean over the instances.
Histogram metrics (AUPRC, CW_FP, ...): computed once on the histograms pooled over the scene's runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import stats
from ..metrics import summarize_hists
from .data import axes_of, curve_rows, severity_labels

LONG_KEYS = ["exp", "system", "axis", "group", "magnitude", "severity", "scene", "metric", "value"]


def scene_values(res, rows, by, summary_metrics=(), hist_metrics=()):
    """One row per `by` group (by must contain 'scene') with the requested metrics."""
    by = list(by)
    out = None
    sm = [m for m in summary_metrics if m in rows.columns]
    if sm and len(rows):
        num = rows[by + ["pair"]].copy()
        for m in sm:
            num[m] = pd.to_numeric(rows[m], errors="coerce")
        out = num.groupby(by + ["pair"], dropna=False)[sm].mean().groupby(level=by).mean()
    if hist_metrics and len(rows):
        recs = []
        for key, g in rows.groupby(by, dropna=False):
            rid = g["row"].to_numpy()
            p, n = res.pos[rid].sum(0), res.neg[rid].sum(0)
            if p.sum() + n.sum() == 0:
                continue
            dec = (pd.to_numeric(g["decision_threshold"], errors="coerce").dropna()
                   if "decision_threshold" in g else pd.Series(dtype=float))
            s = summarize_hists(p, n, float(dec.iloc[0]) if len(dec) else 0.5)
            s["thr_gap"] = s["best_f1"] - s["f1_at_decision"]
            key = key if isinstance(key, tuple) else (key,)
            recs.append(dict(zip(by, key), **{m: s.get(m, np.nan) for m in hist_metrics}))
        if recs:
            h = pd.DataFrame(recs).set_index(by)
            out = h if out is None else out.join(h, how="outer")
    if out is None:
        return pd.DataFrame(columns=by)
    return out.reset_index()


def scene_long(res, cfg, experiments, summary_metrics, hist_metrics):
    """Long table of scene values for every (experiment, system, axis, magnitude, scene, metric)."""
    parts = []
    runs = res.runs
    for exp in experiments:
        for system in sorted(set(runs.loc[runs["exp"] == exp, "system"])):
            for axis in axes_of(runs, exp, system):
                rows = curve_rows(runs, exp, system, axis=axis)
                sv = scene_values(res, rows, ["magnitude", "scene"], summary_metrics, hist_metrics)
                if not len(sv):
                    continue
                labels = severity_labels(rows)
                metrics = [c for c in sv.columns if c not in ("magnitude", "scene")]
                long = sv.melt(id_vars=["magnitude", "scene"], value_vars=metrics, var_name="metric")
                long = long.assign(exp=exp, system=system, axis=axis,
                                   group=rows.loc[~rows["is_base"], "group"].iloc[0],
                                   severity=long["magnitude"].map(labels))
                parts.append(long[LONG_KEYS])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=LONG_KEYS)


def pivot(long, exp, system, axis, metric):
    """scene x magnitude matrix (columns sorted, baseline 0 first)."""
    sel = long[(long["exp"] == exp) & (long["system"] == system) & (long["axis"] == axis)
               & (long["metric"] == metric)]
    if not len(sel):
        return pd.DataFrame()
    m = sel.pivot_table(index="scene", columns="magnitude", values="value", aggfunc="mean")
    return m[sorted(m.columns)]


def curve_table(long, st, noise=None):
    """Per level: scene-mean with bootstrap CI, paired change and relative drop vs the baseline."""
    noise = noise or {}
    nb, ci, seed, k = st["n_boot"], st["ci"], st["seed"], st.get("noise_k", 2)
    recs = []
    keys = long[["exp", "system", "axis", "group", "metric"]].drop_duplicates().itertuples(index=False)
    for exp, system, axis, group, metric in keys:
        m = pivot(long, exp, system, axis, metric)
        if m.empty:
            continue
        labels = dict(long.loc[(long["exp"] == exp) & (long["axis"] == axis), ["magnitude", "severity"]]
                      .drop_duplicates("magnitude").itertuples(index=False))
        base = m[0.0] if 0.0 in m.columns else None
        nsd = noise.get((system, metric), np.nan)
        for level, mag in enumerate(m.columns):
            mean, lo, hi, n = stats.boot_mean(m[mag], nb, ci, seed)
            rec = {"exp": exp, "system": system, "axis": axis, "group": group, "metric": metric,
                   "level": level, "magnitude": mag, "severity": labels.get(mag), "n": n,
                   "mean": mean, "lo": lo, "hi": hi, "noise_sd": nsd}
            if base is not None and mag != 0.0:
                pair = m[[0.0, mag]].dropna()
                d = pair[mag] - pair[0.0]
                rec["delta"], rec["delta_lo"], rec["delta_hi"], _ = stats.boot_mean(d, nb, ci, seed)
                ok = pair[0.0] > 0
                r = 1.0 - pair.loc[ok, mag] / pair.loc[ok, 0.0]
                rec["rel_drop"], rec["rel_lo"], rec["rel_hi"], _ = stats.boot_mean(r, nb, ci, seed)
                excl0 = rec["delta_lo"] > 0 or rec["delta_hi"] < 0
                big = abs(rec["delta"]) > k * nsd if np.isfinite(nsd) else True
                rec["exceeds_noise"] = bool(excl0 and big)
            recs.append(rec)
    return pd.DataFrame(recs)


def noise_floor(res, cfg):
    """Per system and metric: RMS over scenes of the SD across seeds of the clean scene value (E1)."""
    rc = cfg.get("reproduction", {})
    exp = rc.get("experiment")
    metrics = rc.get("noise_metrics", [])
    hist_m = [m for m in metrics if m in cfg["calibration"]["hist_metrics"]]
    sum_m = [m for m in metrics if m not in hist_m]
    runs = res.runs
    per_scene, summary = [], []
    if exp is None or not len(runs):
        return pd.DataFrame(), pd.DataFrame(), {}
    for system in sorted(set(runs.loc[runs["exp"] == exp, "system"])):
        rows = runs[(runs["exp"] == exp) & (runs["system"] == system) & runs["is_base"]]
        sv = scene_values(res, rows, ["seed", "scene"], sum_m, hist_m)
        for metric in [m for m in metrics if m in sv.columns]:
            g = sv.groupby("scene")[metric]
            sd = g.std(ddof=1)
            cnt = g.count()
            sd = sd[cnt >= 2]
            for scene, v in sd.items():
                per_scene.append({"system": system, "metric": metric, "scene": scene, "sd": v,
                                  "n_seeds": int(cnt[scene])})
            summary.append({"system": system, "metric": metric, "n_scenes": int(sd.notna().sum()),
                            "noise_sd": float(np.sqrt(np.nanmean(sd.to_numpy() ** 2))) if sd.notna().any() else np.nan})
    summary = pd.DataFrame(summary)
    lookup = {(r.system, r.metric): r.noise_sd for r in summary.itertuples()} if len(summary) else {}
    return summary, pd.DataFrame(per_scene), lookup


def component_recall(res, cfg, st):
    """Change-object recall by size bin (§9.3): a GT component is detected if >= detected_frac of its
    pixels are predicted. Pooled within a scene, then scene mean with a bootstrap CI."""
    cc = cfg.get("components") or {}
    comps = res.components
    if not cc or not len(comps):
        return pd.DataFrame()
    bins = cc.get("size_bins", [0.0, 0.005, 0.02, 0.05, 1.0])
    c = res.with_run_cols(comps)
    c = c[c["exp"].isin(cc.get("experiments", [])) & (c["axis"].notna() | c["is_base"].astype(bool))]
    if not len(c):
        return pd.DataFrame()
    c = c.assign(detected=(pd.to_numeric(c["detected_frac"], errors="coerce") >= cc.get("detected_frac", 0.5))
                 .astype(float),
                 size_bin=pd.cut(pd.to_numeric(c["area_frac"], errors="coerce"), bins, include_lowest=True)
                 .astype(str))
    recs = []
    runs = res.runs
    for (exp, system), g in c.groupby(["exp", "system"]):
        for axis in axes_of(runs, exp, system):
            sel = g[(g["axis"] == axis) | g["is_base"].astype(bool)]
            labels = severity_labels(runs[(runs["exp"] == exp) & (runs["system"] == system)
                                          & ((runs["axis"] == axis) | runs["is_base"])])
            per = sel.groupby(["magnitude", "size_bin", "scene"])["detected"].agg(["mean", "count"]).reset_index()
            for (mag, sb), h in per.groupby(["magnitude", "size_bin"]):
                mean, lo, hi, n = stats.boot_mean(h["mean"], st["n_boot"], st["ci"], st["seed"])
                recs.append({"exp": exp, "system": system, "axis": axis, "magnitude": mag,
                             "severity": labels.get(mag), "size_bin": sb, "n_scenes": n,
                             "n_components": int(h["count"].sum()), "recall": mean, "lo": lo, "hi": hi})
    return pd.DataFrame(recs)
