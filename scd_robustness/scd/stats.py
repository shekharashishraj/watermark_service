"""Statistics for the analysis (experiment_plan.md §10). NumPy only.

The scene is the statistical unit (n = 10), so most functions take one value per scene:
  * percentile bootstrap over scenes;
  * exact Wilcoxon signed-rank test (handles tied ranks; zeros dropped);
  * per-scene Spearman rho, then Wilcoxon on the rhos (trend test);
  * Holm-Bonferroni over the pre-registered family;
  * area under the normalised degradation curve (AUDC) and a hinge "knee" fit.
Frame-level AUROC with a scene-cluster bootstrap is used for the trust signals (H5).
"""
from __future__ import annotations

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz


def finite(x):
    x = np.asarray(x, float).ravel()
    return x[np.isfinite(x)]


def rankdata(x):
    """1-based ranks; ties get their average rank."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    _, first, cnt = np.unique(x[order], return_index=True, return_counts=True)
    ranks = np.empty(len(x))
    ranks[order] = np.repeat(first + (cnt + 1) / 2.0, cnt)
    return ranks


def spearman(x, y, min_n=3):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < max(2, min_n):
        return float("nan")
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def wilcoxon(d, alternative="two-sided"):
    """Exact Wilcoxon signed-rank test of median(d) = 0. Zeros are dropped; tied |d| get average
    ranks and the exact null distribution is computed for those ranks (dynamic programming)."""
    d = finite(d)
    d = d[d != 0]
    n = len(d)
    out = {"n": n, "mean": float(d.mean()) if n else float("nan"),
           "median": float(np.median(d)) if n else float("nan")}
    if n == 0:
        return dict(out, w_plus=float("nan"), p=1.0)
    r2 = np.round(2 * rankdata(np.abs(d))).astype(np.int64)      # doubled ranks are integers
    w2 = int(r2[d > 0].sum())
    dist = np.zeros(int(r2.sum()) + 1)
    dist[0] = 1.0
    for v in r2:                                                   # P(sum of a random subset of ranks)
        new = 0.5 * dist
        new[v:] += 0.5 * dist[:len(dist) - v]
        dist = new
    k = np.arange(len(dist))
    centre = r2.sum() / 2.0
    if alternative == "greater":
        p = dist[k >= w2].sum()
    elif alternative == "less":
        p = dist[k <= w2].sum()
    else:
        p = dist[np.abs(k - centre) >= abs(w2 - centre) - 1e-9].sum()
    return dict(out, w_plus=w2 / 2.0, p=float(min(1.0, p)))


def min_wilcoxon_p(n, alternative="two-sided"):
    """Smallest attainable p with n non-zero, untied differences (2 / 2^n two-sided)."""
    return (2.0 if alternative == "two-sided" else 1.0) / 2.0 ** n if n else 1.0


def boot_indices(n, n_boot, seed):
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def boot_mean(x, n_boot=10000, ci=0.95, seed=0):
    """Mean and percentile-bootstrap CI over the finite values of x (one per scene)."""
    x = finite(x)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    if n == 1:
        return float(x[0]), float("nan"), float("nan"), 1
    means = x[boot_indices(n, n_boot, seed)].mean(1)
    a = (1 - ci) / 2
    return float(x.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a)), n


def holm(pvals, m=None):
    """Holm-Bonferroni adjusted p-values. m = family size; tests that were not run count as p = 1."""
    p = np.asarray(pvals, float)
    p = np.where(np.isfinite(p), p, 1.0)
    m = max(len(p), m or 0)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(np.argsort(p, kind="mergesort")):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def audc(values):
    """Area under the normalised degradation curve. values[0] is the baseline (severity 0), the rest
    are the levels in increasing severity; x is an evenly spaced index of levels scaled to [0, 1].
    1.0 = no degradation, lower = worse."""
    v = np.asarray(values, float)
    if len(v) < 2 or not np.all(np.isfinite(v)) or v[0] <= 0:
        return float("nan")
    return float(_trapz(v / v[0], np.linspace(0.0, 1.0, len(v))))


def hinge_knee(Y, grid_step=0.05):
    """Knee of relative-drop curves by a hinge fit  y = a + b * max(0, x - k)  on x = 0..L-1.
    Y: (B, L) curves (one per bootstrap sample) or (L,). Returns (k, b) arrays; k = NaN when b <= 0."""
    Y = np.atleast_2d(np.asarray(Y, float))
    L = Y.shape[1]
    x = np.arange(L, dtype=float)
    ks = np.arange(0.0, L - 1 + 1e-9, grid_step)
    best_sse = np.full(Y.shape[0], np.inf)
    best_k = np.full(Y.shape[0], np.nan)
    best_b = np.full(Y.shape[0], np.nan)
    for k in ks:
        h = np.maximum(0.0, x - k)
        if not h.any():
            continue
        X = np.stack([np.ones(L), h], 1)
        coef, *_ = np.linalg.lstsq(X, Y.T, rcond=None)              # (2, B)
        sse = ((X @ coef - Y.T) ** 2).sum(0)
        better = sse < best_sse - 1e-12
        best_sse = np.where(better, sse, best_sse)
        best_k = np.where(better, k, best_k)
        best_b = np.where(better, coef[1], best_b)
    best_k = np.where(best_b > 0, best_k, np.nan)
    return best_k, best_b


# ----------------------------------------------------------------------------- frame-level (H5)

class ClusterAUROC:
    """AUROC of `score` for predicting `label`, with a bootstrap that resamples clusters (scenes).
    Ties count one half. Weighted form: a resampled cluster that appears w times has weight w."""

    def __init__(self, score, label, cluster):
        score, label = np.asarray(score, float), np.asarray(label, bool)
        ok = np.isfinite(score)
        self.score, self.label = score[ok], label[ok]
        self.clusters, self.cidx = np.unique(np.asarray(cluster)[ok], return_inverse=True)
        order = np.argsort(self.score, kind="mergesort")
        self.order = order
        _, self.group = np.unique(self.score[order], return_inverse=True)

    def value(self, cluster_weights=None):
        w = np.ones(len(self.clusters)) if cluster_weights is None else cluster_weights
        fw = w[self.cidx][self.order]
        y = self.label[self.order]
        wp, wn = fw * y, fw * ~y
        P, N = wp.sum(), wn.sum()
        if P <= 0 or N <= 0:
            return float("nan")
        gn = np.bincount(self.group, weights=wn)
        below = np.cumsum(gn) - gn
        return float((wp * (below[self.group] + 0.5 * gn[self.group])).sum() / (P * N))

    def bootstrap(self, n_boot=10000, ci=0.95, seed=0):
        k = len(self.clusters)
        idx = boot_indices(k, n_boot, seed)
        vals = np.array([self.value(np.bincount(row, minlength=k).astype(float)) for row in idx])
        vals = vals[np.isfinite(vals)]
        a = (1 - ci) / 2
        point = self.value()
        if not len(vals):
            return point, float("nan"), float("nan"), vals
        return point, float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a)), vals


def auroc(score, label):
    return ClusterAUROC(score, label, np.zeros(len(label))).value()


def risk_coverage(risk, fail, n_points=101):
    """Keep the most trusted frames first (lowest risk). Returns (coverage, risk_on_kept, AURC)."""
    risk, fail = np.asarray(risk, float), np.asarray(fail, float)
    ok = np.isfinite(risk)
    risk, fail = risk[ok], fail[ok]
    if not len(risk):
        return np.array([]), np.array([]), float("nan")
    order = np.argsort(risk, kind="mergesort")
    kept = np.arange(1, len(risk) + 1)
    curve = np.cumsum(fail[order]) / kept
    cov = kept / float(len(risk))
    grid = np.linspace(1.0 / len(risk), 1.0, n_points)
    return grid, np.interp(grid, cov, curve), float(curve.mean())
