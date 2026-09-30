"""Statistics primitives against brute force / textbook values."""
import itertools

import numpy as np
import pytest

from scd import stats
from scd.analysis.calibration import CENTRES, ece_conf, isotonic_map
from scd.metrics import ece_equal_mass


def _brute_wilcoxon(d):
    d = d[d != 0]
    r = stats.rankdata(np.abs(d))
    obs, centre = r[d > 0].sum(), r.sum() / 2
    hits = [abs(r[np.array(s, bool)].sum() - centre) >= abs(obs - centre) - 1e-9
            for s in itertools.product([0, 1], repeat=len(d))]
    return float(np.mean(hits))


@pytest.mark.parametrize("d", [
    np.array([0.3, -0.1, 0.5, 0.2, 0.9, -0.4, 0.7]),
    np.array([1, 1, 2, -2, 3, 3, 3, -1, 4, 5.0]),            # tied |d|
    np.array([0.0, 0.2, 0.4, -0.1, 0.6]),                     # a zero (dropped)
])
def test_wilcoxon_exact(d):
    assert stats.wilcoxon(d)["p"] == pytest.approx(_brute_wilcoxon(d))


def test_wilcoxon_extremes():
    assert stats.wilcoxon(np.arange(1, 11))["p"] == pytest.approx(2 / 1024)
    assert stats.min_wilcoxon_p(5) == pytest.approx(0.0625)
    assert stats.wilcoxon([])["p"] == 1.0


def test_spearman_and_ranks():
    assert list(stats.rankdata([3, 1, 3, 2])) == [3.5, 1, 3.5, 2]
    assert stats.spearman([1, 2, 3, 4], [10, 20, 25, 40]) == pytest.approx(1.0)
    assert stats.spearman([0, 1], [5, 3], min_n=2) == pytest.approx(-1.0)
    assert np.isnan(stats.spearman([1, 2, 3], [1, 1, 1]))


def test_holm_counts_untested_as_p1():
    adj = stats.holm([0.01, 0.04, 0.03, 0.2], m=6)
    assert list(np.round(adj, 3)) == [0.06, 0.16, 0.15, 0.6]
    assert stats.holm([np.nan, 0.001])[0] == 1.0


def test_audc_and_knee():
    assert stats.audc([0.5, 0.5, 0.5]) == pytest.approx(1.0)
    assert stats.audc([0.5, 0.5, 0.25]) == pytest.approx(0.875)
    k, b = stats.hinge_knee([0.0, 0.0, 0.0, 0.2, 0.4])
    assert k[0] == pytest.approx(2.0, abs=0.06) and b[0] > 0
    assert np.isnan(stats.hinge_knee([0.1, 0.1, 0.1])[0][0])      # flat curve: no knee


def test_boot_mean():
    mean, lo, hi, n = stats.boot_mean([1.0, 2.0, 3.0, np.nan], n_boot=2000, seed=1)
    assert n == 3 and mean == 2.0 and 1.0 <= lo < 2.0 < hi <= 3.0


def test_cluster_auroc_matches_pairwise_and_weights():
    rng = np.random.default_rng(0)
    s = np.round(rng.normal(size=200), 1)
    y = rng.random(200) < 1 / (1 + np.exp(-2 * s))
    pos, neg = s[y], s[~y]
    brute = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg))
    assert stats.auroc(s, y) == pytest.approx(brute)
    cl = rng.integers(0, 5, 200)
    ca = stats.ClusterAUROC(s, y, cl)
    w = np.array([2.0, 0, 1, 1, 0])                  # weights = duplicating clusters
    keep = np.concatenate([np.flatnonzero(cl == c) for c in range(5) for _ in range(int(w[c]))])
    assert ca.value(w) == pytest.approx(stats.auroc(s[keep], y[keep]))


def test_risk_coverage():
    cov, risk, aurc = stats.risk_coverage([0.1, 0.2, 0.9, 0.8], [0, 0, 1, 1])
    assert risk[0] == 0.0 and risk[-1] == pytest.approx(0.5) and aurc == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)


def test_ece_conf_equals_histogram_ece_and_isotonic_calibrates():
    rng = np.random.default_rng(3)
    pos = rng.integers(0, 50, 256) * (np.arange(256) > 100)
    neg = rng.integers(0, 500, 256)
    assert ece_conf(CENTRES, pos, neg)[0] == pytest.approx(ece_equal_mass(pos, neg)[0])
    g = isotonic_map(pos, neg)
    assert np.all(np.diff(g) >= -1e-12)
    assert ece_conf(g, pos, neg)[0] < ece_conf(CENTRES, pos, neg)[0]
