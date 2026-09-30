"""Gates (G1 reproduction, noise floor, identity check) and pre-registered tests H1-H5 (plan §1, §10).

Every test returns records with a p-value; primary ones form the Holm-Bonferroni family.
Trend tests: per-scene Spearman rho(severity magnitude, metric) -> exact Wilcoxon on the rhos.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .. import stats
from .data import RUN_COLS, severity_labels
from .scene import pivot, scene_values

SIGN = {"positive": 1, "negative": -1}


def _test(hyp, system, stressor, metric, w, expected, primary, **extra):
    effect = w.get("mean", np.nan)
    ok = (np.sign(effect) == SIGN[expected]) if np.isfinite(effect) else False
    rec = {"hypothesis": hyp, "system": system, "stressor": stressor, "metric": metric, "primary": primary,
           "n": w.get("n", 0), "effect": effect, "expected": expected, "direction_ok": bool(ok),
           "p": w.get("p", np.nan)}
    rec.update(extra)
    return rec


# ----------------------------------------------------------------------------- gates

def reproduction(res, cfg, st):
    """G1: paper-comparable numbers on clean data (mean over frames within an instance, then over
    the 20 instances; seeds averaged within an instance)."""
    rc = cfg.get("reproduction") or {}
    runs = res.runs
    if not len(runs):
        return pd.DataFrame(), pd.DataFrame()
    rows = runs[(runs["exp"] == rc.get("experiment")) & runs["is_base"]]
    table, scenes = [], []
    for system, targets in rc.get("targets", {}).items():
        r = rows[rows["system"] == system]
        if not len(r):
            continue
        metrics = sorted(set(targets) | {"miou", "mf1", "paper_miou", "paper_f1"})
        metrics = [m for m in metrics if m in r]
        inst = r.groupby(["scene", "pair"])[metrics].mean(numeric_only=True)
        sc = inst.groupby(level="scene").mean()
        for scene, vals in sc.iterrows():
            scenes.append(dict({"system": system, "scene": scene}, **vals.to_dict()))
        for metric in metrics:
            value = float(inst[metric].mean())
            target = targets.get(metric)
            rec = {"system": system, "metric": metric, "value": value, "n_instances": int(inst[metric].count()),
                   "n_runs": len(r), "target": target}
            if target is not None:
                rec["diff"] = value - target
                rec["pass"] = bool(abs(value - target) <= rc.get("tolerance", 0.03))
            table.append(rec)
    return pd.DataFrame(table), pd.DataFrame(scenes)


def identity_check(res, cfg, st, noise):
    ic = cfg.get("identity_check") or {}
    runs = res.runs
    recs = []
    if not ic or not len(runs):
        return pd.DataFrame()
    for system in ic.get("systems", []):
        a = runs[(runs["exp"] == ic["experiment"]) & (runs["system"] == system) & runs["is_base"]]
        b = runs[(runs["exp"] == ic["clean_experiment"]) & (runs["system"] == system) & runs["is_base"]]
        if not len(a) or not len(b):
            continue
        va = scene_values(res, a, ["scene"], ic["metrics"]).set_index("scene")
        vb = scene_values(res, b, ["scene"], ic["metrics"]).set_index("scene")
        for metric in ic["metrics"]:
            if metric not in va or metric not in vb:
                continue
            d = (va[metric] - vb[metric]).dropna()
            mean, lo, hi, n = stats.boot_mean(d, st["n_boot"], st["ci"], st["seed"])
            nsd = noise.get((system, metric), np.nan)
            within = abs(mean) <= st.get("noise_k", 2) * nsd if np.isfinite(nsd) else (lo <= 0 <= hi)
            recs.append({"system": system, "metric": metric, "n": n, "delta": mean, "lo": lo, "hi": hi,
                         "noise_sd": nsd, "p": stats.wilcoxon(d)["p"], "pass": bool(within)})
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------------- helpers

def scene_rhos(long, exp, system, axes, metric):
    """Per-scene Spearman rho between magnitude and metric over the given axes (baseline once)."""
    sel = long[(long["exp"] == exp) & (long["system"] == system) & long["axis"].isin(axes)
               & (long["metric"] == metric)]
    if not len(sel):
        return pd.Series(dtype=float)
    per = sel.groupby(["scene", "magnitude"])["value"].mean().reset_index()
    # two levels (baseline + one severity) still give a sign: rho = +-1
    return pd.Series({s: stats.spearman(g["magnitude"], g["value"], min_n=2) for s, g in per.groupby("scene")},
                     dtype=float)


def group_axes(long, exp, system, group):
    sel = long[(long["exp"] == exp) & (long["system"] == system) & (long["group"] == group)]
    return sorted(set(sel["axis"]))


def _trend(long, exp, system, group, metric, st):
    rhos = scene_rhos(long, exp, system, group_axes(long, exp, system, group), metric)
    w = stats.wilcoxon(rhos.to_numpy())
    mean, lo, hi, n = stats.boot_mean(rhos.to_numpy(), st["n_boot"], st["ci"], st["seed"])
    w.update({"mean": mean, "n": n})
    return w, (lo, hi), rhos


# ----------------------------------------------------------------------------- H1

def h1(long, curves, cfg, st):
    c = cfg["h1"]
    exp, metric, drop = c["experiment"], c["metric"], c.get("boundary_drop", 0.10)
    tests, bounds = [], []
    for system in c["systems"] + c.get("also_report", []):
        primary = system in c["systems"]
        for stressor in c["stressors"]:
            w, (lo, hi), rhos = _trend(long, exp, system, stressor, metric, st)
            tests.append(_test("H1", system, stressor, metric, w, c.get("expected", "negative"), primary,
                               effect_lo=lo, effect_hi=hi, effect_name="mean Spearman rho"))
            for axis in group_axes(long, exp, system, stressor):
                b = _boundary(long, curves, exp, system, axis, metric, drop, st)
                bounds.append(dict(b, group=stressor, primary=primary))
    return tests, pd.DataFrame(bounds)


def _boundary(long, curves, exp, system, axis, metric, drop, st):
    cv = curves[(curves["exp"] == exp) & (curves["system"] == system) & (curves["axis"] == axis)]
    main = cv[(cv["metric"] == metric) & (cv["magnitude"] > 0)].sort_values("magnitude")
    rec = {"system": system, "axis": axis, "metric": metric, "boundary_drop": drop, "s_star": np.nan,
           "s_star_severity": "none in grid", "rel_drop": np.nan, "rel_lo": np.nan, "rel_hi": np.nan}
    hit = main[main["rel_lo"] > drop]
    if len(hit):
        h = hit.iloc[0]
        rec.update(s_star=h["magnitude"], s_star_severity=h["severity"], rel_drop=h["rel_drop"],
                   rel_lo=h["rel_lo"], rel_hi=h["rel_hi"])
        for extra in ("missing_rate", "pose_fail_rate"):
            e = cv[(cv["metric"] == extra) & (cv["magnitude"] == h["magnitude"])]
            rec[extra + "_at_s_star"] = float(e["mean"].iloc[0]) if len(e) else np.nan
    if len(main):
        top = main.iloc[-1]
        rec.update(max_severity=top["severity"], rel_drop_max=top["rel_drop"], rel_lo_max=top["rel_lo"],
                   rel_hi_max=top["rel_hi"])
    # secondary: knee of the relative-drop curve (hinge fit), with a scene-bootstrap CI
    m = pivot(long, exp, system, axis, metric)
    if 0.0 in m.columns and m.shape[1] >= 3:
        m = m[m[0.0] > 0]
        r = (1.0 - m.div(m[0.0], axis=0)).to_numpy()
        mags = np.asarray(m.columns, float)
        if len(r) >= 2:
            k0, _ = stats.hinge_knee(np.nanmean(r, 0))
            idx = stats.boot_indices(len(r), st["n_boot"], st["seed"])
            kb, _ = stats.hinge_knee(np.nanmean(r[idx], 1))
            kb = kb[np.isfinite(kb)]
            a = (1 - st["ci"]) / 2
            to_mag = lambda k: float(np.interp(k, np.arange(len(mags)), mags))  # noqa: E731
            rec["knee"] = to_mag(k0[0]) if np.isfinite(k0[0]) else np.nan
            if len(kb):
                rec["knee_lo"], rec["knee_hi"] = to_mag(np.quantile(kb, a)), to_mag(np.quantile(kb, 1 - a))
            rec["knee_defined_frac"] = len(kb) / float(st["n_boot"])
    return rec


# ----------------------------------------------------------------------------- H2

def h2(long, curves, cfg, st):
    c = cfg["h2"]
    exp = c["experiment"]
    tests = []
    for system in c["systems"] + c.get("also_report", []):
        primary = system in c["systems"]
        for stressor in c["stressors"]:
            for metric in [c["metric"]] + c.get("secondary", []):
                w, (lo, hi), _ = _trend(long, exp, system, stressor, metric, st)
                cv = curves[(curves["exp"] == exp) & (curves["system"] == system) & (curves["group"] == stressor)
                            & (curves["metric"] == metric) & (curves["magnitude"] > 0)]
                tests.append(_test("H2", system, stressor, metric, w, c.get("expected", "positive"),
                                   primary and metric == c["metric"], effect_lo=lo, effect_hi=hi,
                                   effect_name="mean Spearman rho",
                                   any_level_increase=bool((cv["delta_lo"] > 0).any()) if len(cv) else False,
                                   max_delta=float(cv["delta"].max()) if len(cv) else np.nan))
    return tests


# ----------------------------------------------------------------------------- H3

def h3(long, cfg, st):
    c = cfg["h3"]
    exp = c["experiment"]
    tests = []
    for system in c["systems"] + c.get("also_report", []):
        primary = system in c["systems"]
        for metric, expected in c["metrics"].items():
            per = {}
            for stressor in c["stressors"]:
                w, (lo, hi), rhos = _trend(long, exp, system, stressor, metric, st)
                per[stressor] = rhos
                tests.append(_test("H3", system, stressor, metric, w, expected, False, effect_lo=lo,
                                   effect_hi=hi, effect_name="mean Spearman rho"))
            if not per:
                continue
            combined = pd.DataFrame(per).mean(axis=1, skipna=True) if c.get("combine", "mean") == "mean" else \
                pd.DataFrame(per).median(axis=1, skipna=True)
            w = stats.wilcoxon(combined.to_numpy())
            mean, lo, hi, n = stats.boot_mean(combined.to_numpy(), st["n_boot"], st["ci"], st["seed"])
            w.update({"mean": mean, "n": n})
            tests.append(_test("H3", system, "+".join(c["stressors"]), metric, w, expected, primary,
                               effect_lo=lo, effect_hi=hi, effect_name="mean over stressors of Spearman rho"))
    return tests


# ----------------------------------------------------------------------------- H4

def _audc_by_scene(m):
    if m.empty or 0.0 not in m.columns:
        return pd.Series(dtype=float)
    m = m.dropna()
    return pd.Series({s: stats.audc(row.to_numpy()) for s, row in m.iterrows()}, dtype=float)


def h4(long, cfg, st):
    c = cfg["h4"]
    exp, metric = c["experiment"], c["metric"]
    tests, per_scene = [], []
    for stressor in c["stressors"]:
        vals = {}
        for role in ("online", "refined"):
            system = c[role]
            axes = group_axes(long, exp, system, stressor)
            a = [_audc_by_scene(pivot(long, exp, system, ax, metric)) for ax in axes]
            vals[role] = pd.concat(a, axis=1).mean(axis=1) if a else pd.Series(dtype=float)
        both = pd.DataFrame(vals).dropna()
        for scene, r in both.iterrows():
            per_scene.append({"stressor": stressor, "scene": scene, "audc_online": r["online"],
                              "audc_refined": r["refined"]})
        d = (both["refined"] - both["online"]).to_numpy() if len(both) else np.array([])
        w = stats.wilcoxon(d)
        mean, lo, hi, n = stats.boot_mean(d, st["n_boot"], st["ci"], st["seed"])
        w.update({"mean": mean, "n": n})
        tests.append(_test("H4", "%s vs %s" % (c["refined"], c["online"]), stressor, "AUDC(%s)" % metric, w,
                           c.get("expected", "positive"), True, effect_lo=lo, effect_hi=hi,
                           effect_name="mean AUDC(refined) - AUDC(online)",
                           audc_online=float(both["online"].mean()) if len(both) else np.nan,
                           audc_refined=float(both["refined"].mean()) if len(both) else np.nan))
    return tests, pd.DataFrame(per_scene)


def attribution(res, cfg, st):
    """E7: online vs refined vs iteration-matched on the grid all three share (exploratory)."""
    c = cfg["h4"]
    a = c.get("attribution") or {}
    runs = res.runs
    systems = {"online": c["online"], "refined": c["refined"], "matched": a.get("matched")}
    if not a or not len(runs):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    cell = ["scene", "pair", "stressor", "severity", "trial"]
    sub = runs[runs["exp"].isin(a["experiments"]) & runs["system"].isin(systems.values())]
    cells = None
    for s in systems.values():
        cs = set(map(tuple, sub.loc[sub["system"] == s, cell].astype(str).to_numpy()))
        cells = cs if cells is None else cells & cs
    if not cells:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    keep = sub[cell].astype(str).apply(tuple, axis=1).isin(cells)
    sub = sub[keep]
    table, scenes, warm = [], [], []
    frames = res.with_run_cols(res.frames[res.frames["row"].isin(sub["row"])], RUN_COLS)
    for stressor in c["stressors"]:
        axes = sorted(set(sub.loc[sub["group"] == stressor, "axis"].dropna()))
        per = {}
        for role, system in systems.items():
            rows = sub[(sub["system"] == system) & (sub["axis"].isin(axes) | sub["is_base"])]
            sv = scene_values(res, rows, ["magnitude", "scene"], [c["metric"]])
            if not len(sv):
                continue
            m = sv.pivot_table(index="scene", columns="magnitude", values=c["metric"])
            m = m[sorted(m.columns)]
            per[role] = pd.DataFrame({"grid_mean": m.mean(axis=1), "audc": _audc_by_scene(m)})
            fr = frames[(frames["system"] == system) & (frames["axis"].isin(axes) | frames["is_base"])]
            labels = severity_labels(rows)
            if len(fr) and "iou" in fr:
                w = fr.assign(iou=pd.to_numeric(fr["iou"], errors="coerce").fillna(1.0))
                sc = w.groupby(["magnitude", "position", "scene"])["iou"].mean().reset_index()
                agg = sc.groupby(["magnitude", "position"])["iou"].agg(["mean", "count"]).reset_index()
                for r in agg.itertuples(index=False):
                    warm.append({"stressor": stressor, "system": system, "role": role, "magnitude": r.magnitude,
                                 "severity": labels.get(r.magnitude), "position": r.position, "iou": r.mean,
                                 "n_scenes": r.count})
        if len(per) < 3:
            continue
        both = pd.concat(per, axis=1).dropna()
        for scene, r in both.iterrows():
            scenes.append(dict({"stressor": stressor, "scene": scene},
                               **{"%s_%s" % (role, k): r[(role, k)] for role in per for k in ("grid_mean", "audc")}))
        for k in ("grid_mean", "audc"):
            rec = {"stressor": stressor, "quantity": k, "n": len(both)}
            for role in per:
                rec[role], rec[role + "_lo"], rec[role + "_hi"], _ = stats.boot_mean(
                    both[(role, k)], st["n_boot"], st["ci"], st["seed"])
            for name, (x, y) in {"matched_minus_online": ("matched", "online"),
                                 "refined_minus_matched": ("refined", "matched"),
                                 "refined_minus_online": ("refined", "online")}.items():
                d = both[(x, k)] - both[(y, k)]
                rec[name], rec[name + "_lo"], rec[name + "_hi"], _ = stats.boot_mean(
                    d, st["n_boot"], st["ci"], st["seed"])
                rec[name + "_p"] = stats.wilcoxon(d.to_numpy())["p"]
            gap = rec["refined_minus_online"]
            rec["share_matched"] = rec["matched_minus_online"] / gap if gap and np.isfinite(gap) else np.nan
            sh = rec["share_matched"]
            rec["verdict"] = ("no offline advantage" if not (gap > 0) else
                              "hindsight (future frames)" if sh < a.get("hindsight_below", 0.33) else
                              "optimisation budget" if sh > a.get("budget_above", 0.67) else "both")
            table.append(rec)
    return pd.DataFrame(table), pd.DataFrame(scenes), pd.DataFrame(warm)


# ----------------------------------------------------------------------------- H5

def _signal_matrix(frames, sig_cfg, population="real"):
    X, raw_missing = {}, {}
    for name, sc in sig_cfg.items():
        if population == "null" and sc.get("null", True) is False:
            continue
        col = sc.get("column", name)
        if col not in frames:
            continue
        v = pd.to_numeric(frames[col], errors="coerce")
        raw_missing[name] = int(v.isna().sum())
        if "fill" in sc:
            v = v.fillna(sc["fill"])
        t = sc.get("transform")
        if t == "log1p":
            v = np.log1p(v.clip(lower=0))
        elif t == "absdev":
            v = (v - sc.get("center", 0.5)).abs()
        X[name] = v if sc.get("risk", "high") == "high" else -v
    return pd.DataFrame(X, index=frames.index), raw_missing


def _inverse_text(name, sc, thr):
    """Human-readable rejection condition for a risk threshold."""
    if not np.isfinite(thr):
        return "%s: not used" % name
    col = sc.get("column", name)
    t = sc.get("transform")
    low = sc.get("risk", "high") == "low"
    v = -thr if low else thr
    if t == "log1p":
        v = float(np.expm1(v))
    elif t == "absdev":
        return "|%s - %g| > %.4g" % (col, sc.get("center", 0.5), v)
    return "%s %s %.4g" % (col, "<" if low else ">", v)


def fit_rule(R, fail, grid, min_cov):
    """Thresholds (risk units) minimising the failure rate on kept frames with coverage >= min_cov."""
    cands = []
    for j in range(R.shape[1]):
        col = R[:, j]
        q = np.nanquantile(col, np.linspace(0.5, 1.0, grid)) if np.isfinite(col).any() else []
        cands.append(np.unique(np.concatenate([[np.inf], q])))
    best, best_key = tuple(np.inf for _ in cands), None
    for combo in itertools.product(*cands):
        reject = (R > np.asarray(combo)).any(1)
        cov = 1.0 - reject.mean()
        if cov < min_cov or reject.all():
            continue
        key = (fail[~reject].mean(), -cov)
        if best_key is None or key < best_key:
            best, best_key = combo, key
    return np.asarray(best, float)


def _loso_model(X, fail, scene):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.full(len(X), np.nan)
    coefs = []
    for s in np.unique(scene):
        tr, te = scene != s, scene == s
        if len(np.unique(fail[tr])) < 2:
            continue
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(max_iter=2000))
        model.fit(X[tr], fail[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
        coefs.append(model[-1].coef_[0])
    return oof, (np.mean(coefs, 0) if coefs else None)


def h5(res, cfg, st):
    c = cfg["h5"]
    nb, ci, seed = st["n_boot"], st["ci"], st["seed"]
    runs = res.runs
    out = {"tests": [], "signals": pd.DataFrame(), "risk_coverage": pd.DataFrame(), "rule": pd.DataFrame(),
           "rule_folds": pd.DataFrame(), "coef": pd.DataFrame()}
    if not len(runs):
        return out
    sig_rows, rc_rows = [], []

    def single(frames, fail, population):
        X, raw_missing = _signal_matrix(frames, c["signals"], population)
        scene = frames["scene"].to_numpy()
        for name in X:
            ca = stats.ClusterAUROC(X[name].to_numpy(), fail, scene)
            auc, lo, hi, _ = ca.bootstrap(nb, ci, seed)
            cov, risk, aurc = stats.risk_coverage(X[name].to_numpy(), fail)
            sig_rows.append({"population": population, "signal": name, "auroc": auc, "lo": lo, "hi": hi,
                             "aurc": aurc, "n_frames": int(np.isfinite(X[name]).sum()),
                             "n_missing_raw": raw_missing.get(name, 0), "risk": c["signals"][name].get("risk")})
            rc_rows.extend({"population": population, "curve": name, "coverage": a, "risk": b}
                           for a, b in zip(cov, risk))
        return X

    # real pairs: failure = IoU < fail_iou (NaN IoU = both empty = correct)
    rows = runs[(runs["exp"] == c["experiment"]) & (runs["system"] == c["system"])]
    frames = res.with_run_cols(res.frames[res.frames["row"].isin(rows["row"])])
    if len(frames) and "iou" in frames:
        iou = pd.to_numeric(frames["iou"], errors="coerce")
        fail = (iou < c.get("fail_iou", 0.3)).to_numpy()
        X = single(frames, fail, "real")
        scene = frames["scene"].to_numpy()
        base_rate = float(fail.mean())
        for name, risk_vec in (("oracle", fail + 1e-9 * np.arange(len(fail))), ("random", None)):
            if risk_vec is None:
                rc_rows.extend({"population": "real", "curve": "random", "coverage": x, "risk": base_rate}
                               for x in np.linspace(0, 1, 101)[1:])
            else:
                cov, risk, aurc = stats.risk_coverage(risk_vec, fail)
                rc_rows.extend({"population": "real", "curve": name, "coverage": a, "risk": b}
                               for a, b in zip(cov, risk))
        # leave-one-scene-out logistic model on all signals (primary H5 test)
        if X.shape[1] and len(np.unique(scene)) >= 3 and 0 < fail.sum() < len(fail):
            oof, coef = _loso_model(X.to_numpy(), fail, scene)
            ok = np.isfinite(oof)
            ca = stats.ClusterAUROC(oof[ok], fail[ok], scene[ok])
            auc, lo, hi, boots = ca.bootstrap(nb, ci, seed)
            p = min(1.0, 2.0 * (1 + np.sum(boots <= 0.5)) / (len(boots) + 1)) if len(boots) else np.nan
            out["tests"].append({"hypothesis": "H5", "system": c["system"], "stressor": "all", "primary": True,
                                 "metric": "LOSO AUROC (frame failure)", "n": int(len(np.unique(scene[ok]))),
                                 "effect": auc, "effect_lo": lo, "effect_hi": hi, "effect_name": "AUROC",
                                 "expected": "positive", "direction_ok": bool(auc > 0.5), "p": p,
                                 "base_failure_rate": base_rate, "n_frames": int(ok.sum())})
            cov, risk, aurc = stats.risk_coverage(oof, fail)
            sig_rows.append({"population": "real", "signal": "LOSO model", "auroc": auc, "lo": lo, "hi": hi,
                             "aurc": aurc, "n_frames": int(ok.sum()), "n_missing_raw": 0, "risk": "model"})
            rc_rows.extend({"population": "real", "curve": "LOSO model", "coverage": a, "risk": b}
                           for a, b in zip(cov, risk))
            if coef is not None:
                out["coef"] = pd.DataFrame({"signal": list(X.columns), "mean_std_coef": coef})
        # trust rule: thresholds chosen on 9 scenes, evaluated on the held-out one
        tr = c.get("trust_rule") or {}
        names = [n for n in tr.get("signals", []) if n in X]
        if names and len(np.unique(scene)) >= 3:
            R = X[names].to_numpy()
            folds, kept_all = [], np.zeros(len(R), bool)
            for s in np.unique(scene):
                trn, te = scene != s, scene == s
                Rtr = np.where(np.isfinite(R[trn]), R[trn], np.nanmedian(R[trn], 0))
                thr = fit_rule(Rtr, fail[trn], tr.get("grid", 41), tr.get("min_coverage", 0.7))
                Rte = np.where(np.isfinite(R[te]), R[te], np.nanmedian(R[trn], 0))
                keep = ~(Rte > thr).any(1)
                kept_all[te] = keep
                folds.append(dict({"held_out": s, "coverage": keep.mean(),
                                   "risk_kept": fail[te][keep].mean() if keep.any() else np.nan,
                                   "risk_all": fail[te].mean()},
                                  **{"thr_" + n: t for n, t in zip(names, thr)}))
            Rall = np.where(np.isfinite(R), R, np.nanmedian(R, 0))
            thr_all = fit_rule(Rall, fail, tr.get("grid", 41), tr.get("min_coverage", 0.7))
            rej = ~kept_all
            out["rule_folds"] = pd.DataFrame(folds)
            out["rule"] = pd.DataFrame([{
                "rule": "reject if " + " or ".join(_inverse_text(n, c["signals"][n], t)
                                                   for n, t in zip(names, thr_all)),
                "heldout_coverage": kept_all.mean(), "heldout_risk_kept": fail[kept_all].mean(),
                "risk_all": base_rate,
                "heldout_rejection_precision": fail[rej].mean() if rej.any() else np.nan,
                "heldout_failure_recall": fail[rej].sum() / fail.sum() if fail.sum() else np.nan,
                "min_coverage": tr.get("min_coverage", 0.7)}])

    # secondary: no-change frames, failure = false-alarm pixel rate > fail_far
    nc = c.get("null") or {}
    if nc:
        nrows = runs[(runs["exp"] == nc["experiment"]) & (runs["system"] == c["system"])]
        nfr = res.with_run_cols(res.frames[res.frames["row"].isin(nrows["row"])])
        if len(nfr) and "far" in nfr:
            nfail = (pd.to_numeric(nfr["far"], errors="coerce") > nc.get("fail_far", 0.01)).to_numpy()
            if 0 < nfail.sum() < len(nfail):
                single(nfr, nfail, "null")
    out["signals"] = pd.DataFrame(sig_rows)
    out["risk_coverage"] = pd.DataFrame(rc_rows)
    return out


# ----------------------------------------------------------------------------- family

def holm_family(tests, st):
    df = pd.DataFrame(tests)
    if not len(df):
        return df
    df["p_holm"] = np.nan
    prim = df["primary"].astype(bool)
    if prim.any():
        df.loc[prim, "p_holm"] = stats.holm(df.loc[prim, "p"].to_numpy(), m=st.get("family_size"))
    alpha = st.get("alpha", 0.05)
    df["significant"] = prim & (df["p_holm"] < alpha) & df["direction_ok"]
    df["min_attainable_p"] = [stats.min_wilcoxon_p(int(n)) if h != "H5" else np.nan
                              for n, h in zip(df["n"].fillna(0), df["hypothesis"])]
    return df


def decisions(family, h1_bounds, cfg, st):
    """One line per primary test, in words."""
    out = []
    alpha = st.get("alpha", 0.05)
    m = st.get("family_size")
    for r in family[family["primary"].astype(bool)].itertuples(index=False):
        if not r.n or (r.hypothesis != "H5" and r.n < st.get("min_scenes", 3)):
            verdict = "insufficient data (n = %d)" % (r.n or 0)
        elif r.significant:
            verdict = "supported"
        else:
            verdict = "not supported"
        note = ""
        if r.hypothesis == "H1" and len(h1_bounds):
            b = h1_bounds[(h1_bounds["system"] == r.system) & (h1_bounds["group"] == r.stressor)]
            found = b[b["s_star"].notna()]
            note = "; ".join("%s s* = %s" % (x.axis, x.s_star_severity) for x in b.itertuples())
            if verdict == "supported" and not len(found):
                verdict = "trend supported, no failure boundary within grid"
        if r.hypothesis == "H2" and verdict == "supported" and not getattr(r, "any_level_increase", True):
            verdict = "trend supported, no single level exceeds the clean null"
        if r.hypothesis != "H5" and r.n and stats.min_wilcoxon_p(int(r.n)) > alpha / (m or 1):
            note += (" [n = %d scenes: smallest attainable p = %.3g, cannot pass the first Holm step"
                     " (alpha/m = %.3g)]" % (r.n, stats.min_wilcoxon_p(int(r.n)), alpha / (m or 1)))
        out.append({"hypothesis": r.hypothesis, "system": r.system, "stressor": r.stressor, "metric": r.metric,
                    "n": r.n, "effect": r.effect, "p": r.p, "p_holm": r.p_holm, "verdict": verdict,
                    "note": note.strip()})
    return pd.DataFrame(out)
