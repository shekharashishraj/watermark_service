"""Calibration analyses from pooled score histograms (experiment_plan.md §9.5-9.6, E6).

reliability      equal-mass reliability-diagram points per system x stressor level (all scenes pooled)
transfer         fit on the CLEAN runs of one scene fold, test on the other fold at every severity:
                   - isotonic recalibration  -> ECE after transfer (vs raw, vs refitting in-condition)
                   - F1-optimal threshold    -> F1 after transfer (vs the method's own threshold, vs oracle)
                 folds are swapped and the two directions averaged; split by scene, never by instance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics import NBINS, best_f1, brier, f1_at
from .data import axes_of, curve_rows, severity_labels

CENTRES = (np.arange(NBINS) + 0.5) / NBINS


def ece_conf(conf, pos, neg, n_bins=15):
    """Equal-mass ECE when histogram bin i carries confidence conf[i] (e.g. after recalibration).
    Returns (ece, table of (conf, acc, weight))."""
    order = np.argsort(conf, kind="mergesort")
    conf, pos, neg = np.asarray(conf, float)[order], pos[order], neg[order]
    tot = pos + neg
    total = tot.sum()
    if not total:
        return float("nan"), []
    edges = np.searchsorted(np.cumsum(tot), np.linspace(0, total, n_bins + 1)[1:-1], side="left")
    ece, table = 0.0, []
    for g in np.split(np.arange(NBINS), np.unique(edges + 1)):
        n = tot[g].sum()
        if not n:
            continue
        c = float((tot[g] * conf[g]).sum() / n)
        a = float(pos[g].sum() / n)
        ece += (n / total) * abs(a - c)
        table.append((c, a, float(n / total)))
    return float(ece), table


def isotonic_map(pos, neg):
    """Monotone map (per histogram bin) from score to observed change frequency."""
    from sklearn.isotonic import IsotonicRegression

    w = (pos + neg).astype(float)
    ok = w > 0
    if ok.sum() < 2:
        return CENTRES.copy()
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    ir.fit(CENTRES[ok], pos[ok] / w[ok], sample_weight=w[ok])
    return ir.predict(CENTRES)


def _pooled(res, rows):
    rid = rows["row"].to_numpy()
    return res.pos[rid].sum(0), res.neg[rid].sum(0)


def _decision(rows):
    d = pd.to_numeric(rows["decision_threshold"], errors="coerce").dropna() if "decision_threshold" in rows else []
    return float(d.iloc[0]) if len(d) else 0.5


def reliability(res, cfg, experiments):
    n_bins = cfg["calibration"].get("reliability_bins", 15)
    recs = []
    runs = res.runs
    for exp in experiments:
        for system in sorted(set(runs.loc[(runs["exp"] == exp) & ~runs["null"], "system"])):
            for axis in axes_of(runs, exp, system):
                rows = curve_rows(runs, exp, system, axis=axis)
                labels = severity_labels(rows)
                for mag, g in rows.groupby("magnitude"):
                    p, n = _pooled(res, g)
                    if not p.sum():
                        continue
                    ece, table = ece_conf(CENTRES, p, n, n_bins)
                    for b, (c, a, w) in enumerate(table):
                        recs.append({"exp": exp, "system": system, "axis": axis, "magnitude": mag,
                                     "severity": labels.get(mag), "bin": b, "conf": c, "acc": a, "weight": w,
                                     "ece": ece})
    return pd.DataFrame(recs)


def transfer(res, cfg, experiments):
    folds = cfg["calibration"]["folds"]
    fold_names = list(folds)
    if len(fold_names) != 2:
        raise ValueError("calibration.folds must define exactly two scene folds")
    n_bins = cfg["calibration"].get("reliability_bins", 15)
    recs = []
    runs = res.runs
    for exp in experiments:
        for system in sorted(set(runs.loc[(runs["exp"] == exp) & ~runs["null"], "system"])):
            for axis in axes_of(runs, exp, system):
                rows = curve_rows(runs, exp, system, axis=axis)
                dec = _decision(rows)
                labels = severity_labels(rows)
                for mag in sorted(set(rows["magnitude"])):
                    per_fold = []
                    for test_fold in fold_names:
                        train_scenes = folds[fold_names[1 - fold_names.index(test_fold)]]
                        train_base = rows[rows["is_base"] & rows["scene"].isin(train_scenes)]
                        train_same = rows[(rows["magnitude"] == mag) & rows["scene"].isin(train_scenes)]
                        test = rows[(rows["magnitude"] == mag) & rows["scene"].isin(folds[test_fold])]
                        if not len(train_base) or not len(test):
                            continue
                        pb, nb = _pooled(res, train_base)
                        pt, nt = _pooled(res, test)
                        if not pb.sum() or not pt.sum():
                            continue
                        g_clean = isotonic_map(pb, nb)
                        tau = best_f1(pb, nb)[1]
                        ps, ns = _pooled(res, train_same)
                        g_same = isotonic_map(ps, ns) if ps.sum() else np.full(NBINS, np.nan)
                        per_fold.append({
                            "ece_raw": ece_conf(CENTRES, pt, nt, n_bins)[0],
                            "ece_transfer": ece_conf(g_clean, pt, nt, n_bins)[0],
                            "ece_refit": ece_conf(g_same, pt, nt, n_bins)[0] if ps.sum() else np.nan,
                            "brier_raw": brier(pt, nt),
                            "f1_decision": f1_at(pt, nt, dec),
                            "f1_transfer": f1_at(pt, nt, tau) if np.isfinite(tau) else np.nan,
                            "f1_oracle": best_f1(pt, nt)[0],
                            "tau_clean": tau,
                        })
                    if per_fold:
                        avg = pd.DataFrame(per_fold).mean().to_dict()
                        recs.append(dict({"exp": exp, "system": system, "axis": axis, "magnitude": mag,
                                          "severity": labels.get(mag), "n_folds": len(per_fold),
                                          "decision_threshold": dec}, **avg))
    return pd.DataFrame(recs)
