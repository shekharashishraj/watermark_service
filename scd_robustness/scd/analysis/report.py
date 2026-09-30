"""Run the whole analysis and write tables/*.csv, results.json and report.md."""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibration, hypotheses, scene
from .data import Results, load_results

SECTIONS = ("gates", "curves", "h1", "h2", "h3", "h4", "h5", "calibration", "components")


def run_analysis(cfg, out_dir, only=None, n_boot=None, reuse_data=False, log=print):
    out = Path(out_dir)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    st = dict(cfg["stats"])
    if n_boot:
        st["n_boot"] = int(n_boot)
    want = set(only or SECTIONS)

    if reuse_data and (out / "data" / "runs.csv").exists():
        res = Results.load(out / "data")
        log("[analysis] reusing %s" % (out / "data"))
    else:
        res = load_results(cfg, log=log)
        res.save(out / "data")
    runs = res.runs
    log("[analysis] %d run outputs, %d frames, %d components" % (len(runs), len(res.frames), len(res.components)))
    T = {"coverage": coverage(res), "compute": compute(res)}
    if not len(runs):
        _write(T, tables)
        write_report(cfg, st, T, [], out, warnings=["No evaluated runs found. Nothing to analyse yet."])
        return T

    noise_sum, noise_scene, noise = scene.noise_floor(res, cfg)
    T.update(noise_floor=noise_sum, noise_floor_scenes=noise_scene)
    if "gates" in want:
        T["reproduction"], T["reproduction_scenes"] = hypotheses.reproduction(res, cfg, st)
        T["identity_check"] = hypotheses.identity_check(res, cfg, st, noise)

    exps = sorted(set(cfg["curves"]["experiments"]) | {cfg[h]["experiment"] for h in ("h1", "h2", "h3", "h4")})
    exps = [e for e in exps if e in set(runs["exp"])]
    long = scene.scene_long(res, cfg, exps, cfg["curves"]["metrics"], cfg["calibration"]["hist_metrics"])
    curves = scene.curve_table(long, st, noise)
    T.update(scene_values=long, curves=curves)

    tests = []
    if "h1" in want:
        t, T["h1_boundaries"] = hypotheses.h1(long, curves, cfg, st)
        tests += t
    if "h2" in want:
        tests += hypotheses.h2(long, curves, cfg, st)
    if "h3" in want:
        tests += hypotheses.h3(long, cfg, st)
    if "h4" in want:
        t, T["h4_audc_scenes"] = hypotheses.h4(long, cfg, st)
        tests += t
        T["h4_attribution"], T["h4_attribution_scenes"], T["warmup"] = hypotheses.attribution(res, cfg, st)
    if "h5" in want:
        h5 = hypotheses.h5(res, cfg, st)
        tests += h5["tests"]
        T.update(h5_signals=h5["signals"], h5_risk_coverage=h5["risk_coverage"], h5_rule=h5["rule"],
                 h5_rule_folds=h5["rule_folds"], h5_model_coef=h5["coef"])
    if "calibration" in want:
        cal_exps = [e for e in {cfg["h3"]["experiment"]} if e in exps]
        T["reliability"] = calibration.reliability(res, cfg, cal_exps)
        T["calibration_transfer"] = calibration.transfer(res, cfg, cal_exps)
    if "components" in want:
        T["components"] = scene.component_recall(res, cfg, st)

    family = hypotheses.holm_family(tests, st)
    T["tests"] = family
    T["decisions"] = hypotheses.decisions(family, T.get("h1_boundaries", pd.DataFrame()), cfg, st) \
        if len(family) else pd.DataFrame()
    _write(T, tables)
    write_report(cfg, st, T, tests, out, warnings=_warnings(T, cfg, st))
    return T


def coverage(res):
    s = res.status
    if not len(s):
        return pd.DataFrame()
    s = s.assign(ok=s["status"] == "ok", failed=s["status"] == "failed", running=s["status"] == "running",
                 not_run=s["status"] == "not_run", ok_without_eval=(s["status"] == "ok") & (s["n_eval"] == 0))
    cols = ["ok", "failed", "running", "not_run", "ok_without_eval"]
    t = s.groupby(["exp", "method"])[cols].sum().astype(int)
    t.insert(0, "planned", s.groupby(["exp", "method"]).size())
    return t.reset_index()


def compute(res):
    """T5: GPU time actually used, per experiment and method (runs that finished)."""
    s = res.status
    if not len(s):
        return pd.DataFrame()
    ok = s[s["status"] == "ok"].copy()
    if not len(ok):
        return pd.DataFrame()
    ok["seconds"] = pd.to_numeric(ok["seconds"], errors="coerce")
    g = ok.groupby(["exp", "method", "arm"])
    t = pd.DataFrame({"runs": g.size(), "gpu_hours": g["seconds"].sum() / 3600.0,
                      "median_minutes": g["seconds"].median() / 60.0,
                      "gpus": g["gpu"].apply(lambda x: ", ".join(sorted(set(map(str, x.dropna())))))}).reset_index()
    r = res.runs
    if len(r) and "m_online_fps" in r:
        fps = r[r["output"] == "online"].groupby(["exp", "method", "arm"])["m_online_fps"].median()
        t = t.merge(fps.rename("median_online_fps").reset_index(), how="left")
    return t


def _write(T, tables):
    for name, df in T.items():
        if isinstance(df, pd.DataFrame) and len(df.columns):
            suffix = ".csv.gz" if name == "scene_values" else ".csv"
            df.to_csv(tables / (name + suffix), index=False)


def _warnings(T, cfg, st):
    w = []
    cov = T.get("coverage")
    if cov is not None and len(cov):
        for r in cov.itertuples(index=False):
            if r.failed or r.not_run or r.running or r.ok_without_eval:
                w.append("%s/%s: %d planned, %d ok, %d failed, %d running, %d not run, %d ok without evaluation."
                         % (r.exp, r.method, r.planned, r.ok, r.failed, r.running, r.not_run, r.ok_without_eval))
    fam = T.get("tests")
    if fam is not None and len(fam):
        prim = fam[fam["primary"].astype(bool)]
        expected = st.get("family_size")
        if expected and len(prim) < expected:
            w.append("Only %d of the %d pre-registered primary tests could be computed; the missing ones count "
                     "as p = 1 in Holm (conservative)." % (len(prim), expected))
    rep = T.get("reproduction")
    if rep is not None and len(rep) and "pass" in rep:
        bad = rep[rep["pass"].eq(False)]
        for r in bad.itertuples(index=False):
            w.append("Gate G1: %s %s = %.3f vs paper %.3f (tolerance %.2f). Robustness conclusions use relative "
                     "drops; document the gap." % (r.system, r.metric, r.value, r.target,
                                                    cfg["reproduction"].get("tolerance", 0.03)))
    ic = T.get("identity_check")
    if ic is not None and len(ic) and (~ic["pass"]).any():
        w.append("E2 identity check failed for: %s. Re-encoding changes the results; investigate before E4."
                 % ", ".join("%s/%s" % (r.system, r.metric) for r in ic[~ic["pass"]].itertuples()))
    return w


# ----------------------------------------------------------------------------- markdown

def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if v != 0 and (abs(v) < 10 ** -nd or abs(v) >= 1e5):
            return "%.2e" % v
        return ("%." + str(nd) + "f") % v
    return str(v)


def md_table(df, cols=None, headers=None, nd=3):
    if df is None or not len(df):
        return "_no data_\n"
    cols = [c for c in (cols or list(df.columns)) if c in df.columns]
    headers = headers or {}
    lines = ["| " + " | ".join(headers.get(c, c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in df[cols].itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v, nd) for v in r) + " |")
    return "\n".join(lines) + "\n"


def _ci(df, m, lo, hi, name):
    if df is None or not len(df) or m not in df:
        return df
    df = df.copy()
    df[name] = ["%s [%s, %s]" % (_fmt(a), _fmt(b), _fmt(c)) for a, b, c in zip(df[m], df[lo], df[hi])]
    return df


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(Path(__file__).parent),
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_report(cfg, st, T, tests, out, warnings):
    cfg_bytes = Path(cfg["_file"]).read_bytes()
    meta = {"name": cfg["name"], "generated": datetime.datetime.now().isoformat(timespec="seconds"),
            "config": cfg["_file"], "config_sha1": hashlib.sha1(cfg_bytes).hexdigest()[:12],
            "code_version": _git_sha(), "stats": st}
    L = ["# Analysis report: %s" % cfg["name"], "",
         "Generated %s · config `%s` (sha1 %s) · code %s  " % (meta["generated"], Path(cfg["_file"]).name,
                                                            meta["config_sha1"], meta["code_version"]),
         "Scene = statistical unit · bootstrap %d resamples, %d%% percentile CIs · Holm-Bonferroni over %s "
         "primary tests at alpha = %s." % (st["n_boot"], round(100 * st["ci"]), st.get("family_size"),
                                           st.get("alpha")), ""]
    if warnings:
        L += ["> **Warnings**", ">"] + ["> - " + w for w in warnings] + [""]
    L += ["## 1. Data coverage", "", md_table(T.get("coverage"))]
    L += ["## 2. Gate G1: reproduction (E1)", "",
          "Paper-mode numbers skip frames without a prediction, like the released evaluators; `miou` scores "
          "them as empty masks.", "",
          md_table(T.get("reproduction"), ["system", "metric", "value", "target", "diff", "pass", "n_instances"])]
    L += ["## 3. Noise floor (E1 seeds)", "",
          "RMS over scenes of the SD across seeds. A change counts only if |mean| > %s x noise SD and its CI "
          "excludes 0." % st.get("noise_k", 2), "", md_table(T.get("noise_floor"), nd=4)]
    L += ["## 4. E2 identity check (identity re-encode minus clean)", "",
          md_table(_ci(T.get("identity_check"), "delta", "lo", "hi", "delta [CI]"),
                   ["system", "metric", "n", "delta [CI]", "noise_sd", "p", "pass"], nd=4)]
    L += ["## 5. Primary hypotheses", "",
          md_table(T.get("decisions"), ["hypothesis", "system", "stressor", "metric", "n", "effect", "p", "p_holm",
                                        "verdict", "note"])]
    fam = T.get("tests")
    L += ["All tests (primary and exploratory):", "",
          md_table(_ci(fam, "effect", "effect_lo", "effect_hi", "effect [CI]"),
                   ["hypothesis", "system", "stressor", "metric", "primary", "n", "effect_name", "effect [CI]", "p",
                    "p_holm", "significant"])]
    b = T.get("h1_boundaries")
    L += ["## 6. H1: failure boundaries s* (first level whose CI lower bound of the mean relative drop > %s)"
          % _fmt(cfg["h1"].get("boundary_drop", 0.1), 2), "",
          md_table(_ci(b, "rel_drop", "rel_lo", "rel_hi", "drop at s* [CI]"),
                   ["system", "axis", "s_star_severity", "drop at s* [CI]", "missing_rate_at_s_star",
                    "pose_fail_rate_at_s_star", "max_severity", "rel_drop_max", "knee", "knee_lo", "knee_hi"])]
    cv = T.get("curves")
    if cv is not None and len(cv):
        e5 = cv[(cv["exp"] == cfg["h2"]["experiment"]) & cv["metric"].isin([cfg["h2"]["metric"]] +
                                                                            cfg["h2"].get("secondary", []))]
        L += ["## 7. H2: no-change false alarms", "",
              md_table(_ci(e5, "mean", "lo", "hi", "mean [CI]"),
                       ["system", "axis", "metric", "severity", "n", "mean [CI]", "delta", "delta_lo", "delta_hi",
                        "exceeds_noise"], nd=4)]
    ct = T.get("calibration_transfer")
    L += ["## 8. H3 / E6: calibration under shift", "",
          "ECE after an isotonic map fitted on the clean runs of the other scene fold (`ece_transfer`), vs raw and "
          "vs refitting in-condition; F1 with the clean-tuned threshold vs the method's own and the oracle.", "",
          md_table(ct, ["system", "axis", "severity", "ece_raw", "ece_transfer", "ece_refit", "f1_decision",
                        "f1_transfer", "f1_oracle"])]
    L += ["## 9. H4 / E7: online vs refined vs iteration-matched", "",
          md_table(T.get("h4_attribution"), ["stressor", "quantity", "n", "online", "matched", "refined",
                                             "matched_minus_online", "refined_minus_matched", "share_matched",
                                             "verdict"])]
    L += ["## 10. H5: frame-level trust signals", "",
          md_table(_ci(T.get("h5_signals"), "auroc", "lo", "hi", "AUROC [CI]"),
                   ["population", "signal", "risk", "AUROC [CI]", "aurc", "n_frames", "n_missing_raw"]),
          "Trust rule (thresholds chosen on 9 scenes, tested on the held-out scene, rotated):", "",
          md_table(T.get("h5_rule"))]
    L += ["## 11. Compute used (T5)", "", md_table(T.get("compute"))]
    L += ["## Files", "", "`tables/` holds every table as CSV (inputs of `scripts/make_figures.py`); "
          "`data/` holds the loaded runs, frames, components and histograms.", ""]
    (Path(out) / "report.md").write_text("\n".join(L))
    results = {"meta": meta, "warnings": warnings,
               "decisions": _records(T.get("decisions")), "tests": _records(T.get("tests")),
               "reproduction": _records(T.get("reproduction")), "h1_boundaries": _records(T.get("h1_boundaries")),
               "h4_attribution": _records(T.get("h4_attribution")), "h5_rule": _records(T.get("h5_rule"))}
    (Path(out) / "results.json").write_text(json.dumps(results, indent=2, default=_json_default))


def _records(df):
    if df is None or not len(df):
        return []
    return json.loads(df.to_json(orient="records"))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)
