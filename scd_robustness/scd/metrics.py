"""Pixel metrics and histogram-based score metrics (experiment_plan.md §9). NumPy only.

Scores are summarised per frame as two 256-bin histograms over [0, 1] (changed / unchanged pixels).
Summing histograms over frames, scenes or folds gives exact-enough AUROC, AUPRC, Brier, ECE,
confident-error rates and threshold sweeps without storing full score maps.
Threshold semantics: "predict change if p >= t", resolved at bin edges t = i / NBINS.
"""
from __future__ import annotations

import math

import numpy as np

NBINS = 256
_trapz = getattr(np, "trapezoid", None) or np.trapz


def score_hist(p, positive=None):
    """(pos_hist, neg_hist) of scores p in [0, 1]; positive is a bool mask or None (all negative)."""
    idx = np.clip((np.asarray(p, np.float32) * NBINS).astype(np.int64), 0, NBINS - 1).ravel()
    if positive is None:
        return np.zeros(NBINS, np.int64), np.bincount(idx, minlength=NBINS).astype(np.int64)
    pm = np.asarray(positive, bool).ravel()
    return (np.bincount(idx[pm], minlength=NBINS).astype(np.int64),
            np.bincount(idx[~pm], minlength=NBINS).astype(np.int64))


def counts(gt, pred) -> dict:
    gt = np.asarray(gt, bool)
    pred = np.asarray(pred, bool)
    tp = int(np.count_nonzero(gt & pred))
    fp = int(np.count_nonzero(~gt & pred))
    fn = int(np.count_nonzero(gt & ~pred))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": int(gt.size - tp - fp - fn)}


def _div(a, b):
    return a / b if b else float("nan")


def rates(tp, fp, fn, tn=0) -> dict:
    return {
        "iou": _div(tp, tp + fp + fn),
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "precision": _div(tp, tp + fp),
        "recall": _div(tp, tp + fn),
    }


def _cum_from_top(h):
    return np.cumsum(h[::-1])[::-1]


def auroc(pos, neg):
    P, N = pos.sum(), neg.sum()
    if not P or not N:
        return float("nan")
    tpr = np.concatenate([[0.0], _cum_from_top(pos)[::-1] / P])
    fpr = np.concatenate([[0.0], _cum_from_top(neg)[::-1] / N])
    return float(_trapz(tpr, fpr))


def average_precision(pos, neg):
    P = pos.sum()
    if not P:
        return float("nan")
    tp = _cum_from_top(pos)[::-1].astype(np.float64)   # thresholds from high to low
    fp = _cum_from_top(neg)[::-1].astype(np.float64)
    ap, prev_recall = 0.0, 0.0
    for t, f in zip(tp, fp):
        if t + f == 0:
            continue
        recall = t / P
        ap += (recall - prev_recall) * (t / (t + f))
        prev_recall = recall
    return float(ap)


def f1_curve(pos, neg):
    """F1 for every bin-edge threshold t_i = i / NBINS (i = 0..NBINS-1)."""
    P = pos.sum()
    tp = _cum_from_top(pos).astype(np.float64)
    fp = _cum_from_top(neg).astype(np.float64)
    fn = P - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)


def f1_at(pos, neg, threshold):
    i = min(NBINS - 1, max(0, int(math.ceil(threshold * NBINS))))
    return float(f1_curve(pos, neg)[i])


def best_f1(pos, neg):
    curve = f1_curve(pos, neg)
    if np.all(np.isnan(curve)):
        return float("nan"), float("nan")
    i = int(np.nanargmax(curve))
    return float(curve[i]), i / NBINS


def confident_errors(pos, neg, hi=0.9, lo=0.1):
    """CW_FP = P(p >= hi | unchanged), CW_FN = P(p < lo | changed)."""
    P, N = pos.sum(), neg.sum()
    return (_div(float(neg[int(hi * NBINS):].sum()), float(N)),
            _div(float(pos[:int(lo * NBINS)].sum()), float(P)))


def brier(pos, neg):
    c = (np.arange(NBINS) + 0.5) / NBINS
    total = pos.sum() + neg.sum()
    return _div(float((pos * (1 - c) ** 2).sum() + (neg * c ** 2).sum()), float(total))


def ece_equal_mass(pos, neg, n_bins=15):
    """Adaptive (equal-mass) ECE from histograms; returns (ece, list of (conf, acc, weight))."""
    c = (np.arange(NBINS) + 0.5) / NBINS
    tot = pos + neg
    total = tot.sum()
    if not total:
        return float("nan"), []
    edges = np.searchsorted(np.cumsum(tot), np.linspace(0, total, n_bins + 1)[1:-1], side="left")
    groups = np.split(np.arange(NBINS), np.unique(edges + 1))
    ece, table = 0.0, []
    for g in groups:
        n = tot[g].sum()
        if not n:
            continue
        conf = float((tot[g] * c[g]).sum() / n)
        acc = float(pos[g].sum() / n)
        w = float(n / total)
        ece += w * abs(acc - conf)
        table.append((conf, acc, w))
    return float(ece), table


def summarize_hists(pos, neg, decision_threshold):
    cw_fp, cw_fn = confident_errors(pos, neg)
    bf1, bthr = best_f1(pos, neg)
    ece, _ = ece_equal_mass(pos, neg)
    return {
        "auroc": auroc(pos, neg),
        "auprc": average_precision(pos, neg),
        "brier": brier(pos, neg),
        "ece15": ece,
        "cw_fp": cw_fp,
        "cw_fn": cw_fn,
        "f1_at_decision": f1_at(pos, neg, decision_threshold),
        "best_f1": bf1,
        "best_threshold": bthr,
        "decision_threshold": decision_threshold,
    }
