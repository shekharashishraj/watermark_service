"""End-to-end analysis on a synthetic study with known effects (fake_results.py)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from fake_results import make_fake_study
from scd.analysis.data import load_config
from scd.analysis.report import run_analysis

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("study")
    cfg_path = make_fake_study(root)
    cfg = load_config(cfg_path)
    out = root / "analysis"
    tables = run_analysis(cfg, out, n_boot=300, log=lambda *a: None)
    return {"cfg_path": cfg_path, "cfg": cfg, "out": out, "T": tables}


def _decision(T, hyp, system, stressor):
    d = T["decisions"]
    return d[(d["hypothesis"] == hyp) & (d["system"] == system) & (d["stressor"] == stressor)].iloc[0]


def test_coverage_counts_failed_and_missing(study):
    cov = study["T"]["coverage"].set_index(["exp", "method"])
    assert cov.loc[("e4", "oscd"), "failed"] == 1 and cov.loc[("e4", "oscd"), "not_run"] == 1
    assert cov.loc[("e1", "oscd"), "ok"] == 60


def test_gates(study):
    T = study["T"]
    rep = T["reproduction"]
    assert set(rep["system"]) == {"oscd_online", "oscd_refined", "mv3dcd"}
    assert rep.loc[(rep["system"] == "oscd_online") & (rep["metric"] == "paper_miou"), "n_instances"].iloc[0] == 20
    nf = T["noise_floor"].set_index(["system", "metric"])
    assert nf.loc[("oscd_online", "miou"), "n_scenes"] == 10 and nf.loc[("mv3dcd", "miou"), "n_scenes"] == 5
    assert set(T["identity_check"]["system"]) == {"oscd_online", "oscd_refined"}


def test_primary_family(study):
    T = study["T"]
    fam = T["tests"]
    assert fam["primary"].sum() == 15                  # H1 6 + H2 2 + H3 4 + H4 2 + H5 1
    assert _decision(T, "H1", "oscd_online", "blur")["verdict"] == "supported"
    assert _decision(T, "H2", "oscd_online", "blur")["verdict"] == "supported"
    assert _decision(T, "H4", "oscd_refined vs oscd_online", "blur")["verdict"] == "supported"
    assert _decision(T, "H5", "oscd_online", "all")["verdict"] == "supported"
    mv = _decision(T, "H1", "mv3dcd", "blur")               # 5 scenes cannot reach alpha
    assert mv["verdict"] == "not supported" and "smallest attainable p" in mv["note"]
    auprc = fam[(fam["hypothesis"] == "H3") & (fam["system"] == "oscd_online") & (fam["metric"] == "auprc")
                & fam["primary"]].iloc[0]
    assert auprc["effect"] < 0 and auprc["significant"]


def test_boundaries_and_curves(study):
    T = study["T"]
    b = T["h1_boundaries"].set_index(["system", "axis"])
    assert b.loc[("oscd_online", "blur"), "s_star_severity"] == "32"
    assert set(b.index.get_level_values("axis")) >= {"blur", "exposure_under", "exposure_over", "views"}
    cv = T["curves"]
    base = cv[(cv["exp"] == "e4") & (cv["metric"] == "miou") & (cv["magnitude"] == 0)]
    assert (base["n"] >= 5).all() and base["delta"].isna().all()
    far = cv[(cv["exp"] == "e5") & (cv["system"] == "oscd_online") & (cv["metric"] == "far_mean")
             & (cv["axis"] == "blur")].sort_values("magnitude")
    assert far["mean"].is_monotonic_increasing


def test_attribution_calibration_trust(study):
    T = study["T"]
    att = T["h4_attribution"]
    assert set(att["stressor"]) == {"views", "blur"} and (att["refined_minus_online"] > 0).all()
    assert set(T["warmup"]["role"]) == {"online", "refined", "matched"}
    tr = T["calibration_transfer"]
    blur = tr[(tr["system"] == "oscd_online") & (tr["axis"] == "blur")].sort_values("magnitude")
    assert blur["ece_transfer"].iloc[-1] > blur["ece_transfer"].iloc[0]
    rule = T["h5_rule"].iloc[0]
    assert rule["rule"].startswith("reject if") and rule["heldout_coverage"] >= 0.6
    assert rule["heldout_risk_kept"] < rule["risk_all"]
    assert len(T["components"])


def test_outputs_and_figures(study):
    out = study["out"]
    assert (out / "report.md").read_text().startswith("# Analysis report")
    res = json.loads((out / "results.json").read_text())
    assert len(res["decisions"]) == 15
    assert (out / "tables" / "curves.csv").exists() and (out / "data" / "runs.csv").exists()
    env = dict(os.environ, MPLCONFIGDIR=str(out / "mpl"))
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "make_figures.py"), "--config",
                           str(study["cfg_path"]), "--analysis-dir", str(out)], capture_output=True, text=True,
                          env=env)
    assert proc.returncode == 0, proc.stderr
    names = {p.stem for p in (out / "figures").glob("*.png")}
    assert names == {"f1_degradation", "f2_fp_fn", "f3_null_false_alarms", "f4_reliability", "f5_calibration",
                     "f6_online_vs_offline", "f7_trust_signals", "f8_component_recall"}


def test_reuse_data_gives_same_decisions(study):
    again = run_analysis(study["cfg"], study["out"], n_boot=300, reuse_data=True, log=lambda *a: None)
    a = study["T"]["decisions"][["hypothesis", "system", "stressor", "verdict"]]
    b = again["decisions"][["hypothesis", "system", "stressor", "verdict"]]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
