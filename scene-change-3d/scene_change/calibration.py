"""Calibration and threshold analysis of continuous change scores.

A score map is read as P(change) per pixel. Most quantities are computed from a fine
histogram of the scores (``NB`` equal-width bins, holding the number of pixels, of
changed pixels and the score sum per bin), so runs can be pooled across scenes, and a
recalibration fitted on one set of runs can be scored on another without keeping the
score maps. Bin k covers scores in (k/NB, (k+1)/NB] (bin 0 also holds 0), so "score > t"
for t = k/NB is exactly bins k and up.

    score_histogram(scores, gt)            fine histogram (poolable: add the arrays)
    ece(hist, bins, adaptive)              expected calibration error, equal-width or equal-mass bins
    reliability(hist, bins)                reliability-diagram points
    average_precision(hist), auroc(hist)   threshold-free ranking quality
    brier(scores, gt)
    threshold_sweep(scores, gt)            PASLCD frame IoU/F1, false alarms and pixel P/R per threshold
    confident_errors(scores, gt)           share of errors made with high confidence
    isotonic_map(hist), recalibrate(hist, mapping)
    detection_reliability(scores, correct) the same for per-detection scores
    calibration_report(scores, gt)         everything above in one JSON-able dict
"""

from __future__ import annotations

import numpy as np

NB = 300                                    # fine bins (divisible by 10, 15, 20, 50, 100)
THRESHOLDS = (0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
              0.85, 0.9, 0.95)


# ----------------------------------------------------------------------------
# Fine histogram
# ----------------------------------------------------------------------------

def _bin_index(s: np.ndarray, nb: int = NB) -> np.ndarray:
    return np.clip(np.ceil(np.asarray(s, np.float64) * nb).astype(np.int64) - 1, 0, nb - 1)


def score_histogram(scores, gt, valid=None, nb: int = NB) -> dict:
    s = np.asarray(scores, np.float32).ravel()
    y = np.asarray(gt, bool).ravel()
    if valid is not None:
        m = np.asarray(valid, bool).ravel()
        s, y = s[m], y[m]
    b = _bin_index(s, nb)
    return {"n": np.bincount(b, minlength=nb).astype(np.int64),
            "pos": np.bincount(b, weights=y, minlength=nb).astype(np.int64),
            "sum": np.bincount(b, weights=s, minlength=nb)}


def pool(hists) -> dict:
    hists = [h for h in hists if h is not None]
    return {k: np.sum([np.asarray(h[k]) for h in hists], axis=0) for k in ("n", "pos", "sum")}


def hist_to_json(h: dict) -> dict:
    return {"n": np.asarray(h["n"]).astype(int).tolist(), "pos": np.asarray(h["pos"]).astype(int).tolist(),
            "sum": np.round(np.asarray(h["sum"], np.float64), 3).tolist()}


def hist_from_json(d: dict) -> dict:
    return {"n": np.asarray(d["n"], np.int64), "pos": np.asarray(d["pos"], np.int64),
            "sum": np.asarray(d["sum"], np.float64)}


def _groups(h: dict, bins: int, adaptive: bool) -> list[np.ndarray]:
    n = np.asarray(h["n"], np.float64)
    nb = len(n)
    if not adaptive:
        edges = np.round(np.linspace(0, nb, bins + 1)).astype(int)
        return [np.arange(edges[i], edges[i + 1]) for i in range(bins)]
    # equal-mass groups of consecutive fine bins; a fine bin is never split, so a bin holding
    # more than its share (e.g. all the zero scores) forms one group and the rest is re-split
    remaining, left = n.sum(), bins
    groups, cur, acc = [], [], 0.0
    for k in range(nb):
        cur.append(k)
        acc += n[k]
        if left > 1 and acc > 0 and acc >= remaining / left - 1e-9:
            groups.append(np.array(cur))
            remaining -= acc
            left -= 1
            cur, acc = [], 0.0
    if cur:
        groups.append(np.array(cur))
    return groups


def reliability(h: dict, bins: int = 10, adaptive: bool = False) -> list[dict]:
    """Per bin: score range, mean score, observed change frequency, pixel count."""
    out = []
    n, pos, ssum = (np.asarray(h[k], np.float64) for k in ("n", "pos", "sum"))
    nb = len(n)
    for g in _groups(h, bins, adaptive):
        if len(g) == 0:
            continue
        cnt = n[g].sum()
        out.append({"lo": g[0] / nb, "hi": (g[-1] + 1) / nb, "n": int(cnt),
                    "conf": float(ssum[g].sum() / cnt) if cnt else None,
                    "freq": float(pos[g].sum() / cnt) if cnt else None})
    return out


def ece(h: dict, bins: int = 15, adaptive: bool = False) -> float | None:
    """Expected calibration error of P(change): sum over bins of weight x |mean score - change frequency|."""
    rel = reliability(h, bins, adaptive)
    total = sum(r["n"] for r in rel)
    if not total:
        return None
    return float(sum(r["n"] / total * abs(r["conf"] - r["freq"]) for r in rel if r["n"]))


def _curve(h: dict):
    n = np.asarray(h["n"], np.float64)
    pos = np.asarray(h["pos"], np.float64)
    tp = np.cumsum(pos[::-1])[::-1]           # predicted positive at threshold k/NB: bins >= k
    fp = np.cumsum((n - pos)[::-1])[::-1]
    return tp, fp, pos.sum(), (n - pos).sum()


def average_precision(h: dict) -> float | None:
    tp, fp, P, _ = _curve(h)
    if P == 0:
        return None
    prec = tp / np.maximum(tp + fp, 1.0)
    rec = tp / P
    r = np.r_[rec, 0.0]
    return float(np.sum((r[:-1] - r[1:]) * prec))


def auroc(h: dict) -> float | None:
    tp, fp, P, N = _curve(h)
    if P == 0 or N == 0:
        return None
    tpr = np.r_[tp / P, 0.0]
    fpr = np.r_[fp / N, 0.0]
    return float(np.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + tpr[1:]) / 2))


def brier(scores, gt) -> float:
    s = np.asarray(scores, np.float32)
    return float(np.mean((s - np.asarray(gt, np.float32)) ** 2))


# ----------------------------------------------------------------------------
# Thresholds and errors
# ----------------------------------------------------------------------------

def threshold_sweep(scores, gt, thresholds=THRESHOLDS) -> list[dict]:
    """PASLCD frame scores (IoU/F1 averaged over frames with change, false-alarm rate) and pixel P/R per threshold."""
    from .harness.evaluate import frame_metrics

    s = np.asarray(scores, np.float32)
    y = np.asarray(gt, bool)
    P = float(y.sum())
    out = []
    for t in thresholds:
        pred = s > t
        fm = frame_metrics(pred, y)
        tp = float(np.sum(pred & y))
        pp = float(pred.sum())
        out.append({"t": t, "IoU": fm["IoU"], "F1": fm["F1"], "false_alarm_rate": fm["false_alarm_rate"],
                    "precision": tp / pp if pp else None, "recall": tp / P if P else None})
    return out


def best_threshold(sweep: list[dict], key: str = "F1") -> dict | None:
    rows = [r for r in sweep if r[key] is not None]
    return max(rows, key=lambda r: r[key]) if rows else None


def confident_errors(scores, gt, threshold: float = 0.5, hi: float = 0.9, lo: float = 0.1) -> dict:
    """Errors at ``threshold``, and the share of them made with high confidence.

    A false positive is confident when its score is at least ``hi``; a false negative when
    its score is at most ``lo``.
    """
    s = np.asarray(scores, np.float32)
    y = np.asarray(gt, bool)
    fp = (s > threshold) & ~y
    fn = (s <= threshold) & y
    nfp, nfn = int(fp.sum()), int(fn.sum())
    return {"fp": nfp, "fn": nfn,
            "fp_confident": float(np.sum(fp & (s >= hi)) / nfp) if nfp else None,
            "fn_confident": float(np.sum(fn & (s <= lo)) / nfn) if nfn else None}


# ----------------------------------------------------------------------------
# Recalibration
# ----------------------------------------------------------------------------

def isotonic_map(h: dict) -> np.ndarray:
    """Non-decreasing map from fine bin to P(change), fitted by pool-adjacent-violators."""
    n = np.asarray(h["n"], np.float64)
    pos = np.asarray(h["pos"], np.float64)
    nb = len(n)
    prior = pos.sum() / max(n.sum(), 1.0)
    blocks = []                                  # [weight, positives, first bin, last bin]
    for k in range(nb):
        if n[k] == 0:
            continue
        blocks.append([n[k], pos[k], k, k])
        while len(blocks) > 1 and blocks[-2][1] / blocks[-2][0] > blocks[-1][1] / blocks[-1][0]:
            w, p, _, e = blocks.pop()
            blocks[-1][0] += w
            blocks[-1][1] += p
            blocks[-1][3] = e
    mapping = np.full(nb, np.nan)
    for w, p, a, b in blocks:
        mapping[a:b + 1] = p / w
    # empty bins take the value of the nearest fitted bin below (or above)
    idx = np.arange(nb)
    ok = ~np.isnan(mapping)
    if not ok.any():
        return np.full(nb, prior)
    below = np.maximum.accumulate(np.where(ok, idx, -1))
    above = np.minimum.accumulate(np.where(ok, idx, nb)[::-1])[::-1]
    fill = np.where(below >= 0, mapping[np.maximum(below, 0)], mapping[np.minimum(above, nb - 1)])
    return np.where(ok, mapping, fill)


def recalibrate(h: dict, mapping: np.ndarray) -> dict:
    """Fine histogram of the scores after applying ``mapping`` (fine bin -> probability)."""
    n = np.asarray(h["n"], np.float64)
    pos = np.asarray(h["pos"], np.float64)
    nb = len(n)
    tgt = _bin_index(mapping, nb)
    return {"n": np.bincount(tgt, weights=n, minlength=nb).astype(np.int64),
            "pos": np.bincount(tgt, weights=pos, minlength=nb).astype(np.int64),
            "sum": np.bincount(tgt, weights=n * mapping, minlength=nb)}


# ----------------------------------------------------------------------------
# Detection-level calibration
# ----------------------------------------------------------------------------

def detection_reliability(scores, correct, bins=(0.0, 0.4, 0.55, 0.7, 0.85, 1.0001)) -> dict:
    """Precision per score bin for per-detection scores, and the detection-level ECE."""
    s = np.asarray(scores, np.float64)
    c = np.asarray(correct, bool)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (s >= lo) & (s < hi)
        if m.any():
            rows.append({"lo": lo, "hi": min(hi, 1.0), "n": int(m.sum()), "conf": float(s[m].mean()),
                         "precision": float(c[m].mean())})
    total = sum(r["n"] for r in rows)
    e = float(sum(r["n"] / total * abs(r["conf"] - r["precision"]) for r in rows)) if total else None
    return {"bins": rows, "ece": e, "n": int(len(s)), "precision": float(c.mean()) if len(c) else None}


# ----------------------------------------------------------------------------
# One-stop report
# ----------------------------------------------------------------------------

def calibration_report(scores, gt, threshold: float = 0.5) -> dict:
    h = score_histogram(scores, gt)
    sweep = threshold_sweep(scores, gt)
    best = best_threshold(sweep)
    return {"ece": ece(h, 15), "ace": ece(h, 15, adaptive=True), "brier": brier(scores, gt),
            "ap": average_precision(h), "auroc": auroc(h), "sweep": sweep,
            "best_t": None if best is None else best["t"], "best_F1": None if best is None else best["F1"],
            "confident": confident_errors(scores, gt, threshold), "hist": hist_to_json(h)}
