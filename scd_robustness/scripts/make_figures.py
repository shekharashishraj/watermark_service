#!/usr/bin/env python
"""Report figures F1-F8 (experiment_plan.md §14) from the tables written by scripts/analyze.py.

  python scripts/make_figures.py --config configs/analysis.yaml
  -> <results>/analysis/<name>/figures/f*.{png,pdf}

F1 degradation curves (mIoU, F1)      F5 AUPRC / confident FP / threshold gap / ECE vs severity
F2 FP vs FN pixels                    F6 online vs refined: AUDC per scene, warm-up curves
F3 no-change false alarms             F7 trust signals: risk-coverage, AUROC per signal
F4 reliability diagrams               F8 change-object recall by size
Every figure shows scene-mean +- 95% bootstrap CI and the per-scene values (n = 10 dots, not bars).
Colours follow the system (configs/analysis.yaml), markers are the second channel.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from scd.analysis.data import load_config  # noqa: E402

SURFACE, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
ORDINAL = ["#86b6ef", "#2a78d6", "#104281"]              # one hue, light -> dark = mild -> severe
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # validated categorical order (non-system series)
METRIC_LABELS = {
    "miou": "mIoU", "mf1": "mean F1", "pooled_iou": "pooled IoU", "miou_btol": "mIoU (boundary-tolerant)",
    "fp_frac_mean": "FP pixels (fraction of frame)", "fn_frac_mean": "FN pixels (fraction of frame)",
    "far_mean": "false-alarm pixel rate", "ffr": "frames with a false alarm", "auprc": "AUPRC",
    "cw_fp": "confident FP rate, P(p≥0.9 | unchanged)", "thr_gap": "F1 gap: oracle − own threshold",
    "ece15": "ECE (15 equal-mass bins)", "missing_rate": "frames without prediction",
    "pose_fail_rate": "pose failure rate",
}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 8.5, "axes.titlesize": 9,
        "axes.labelsize": 8.5, "legend.frameon": False, "lines.linewidth": 1.5, "lines.markersize": 5,
    })


class Figs:
    def __init__(self, cfg, tables, out):
        self.cfg, self.T, self.out = cfg, tables, out
        self.formats = cfg.get("figures", {}).get("format", ["png", "pdf"])
        self.dpi = cfg.get("figures", {}).get("dpi", 200)
        self.systems = cfg["systems"]
        self.stress_order = list(cfg.get("stressors", {}))
        self.written = []

    def t(self, name):
        return self.T.get(name, pd.DataFrame())

    def save(self, fig, name):
        for fmt in self.formats:
            p = self.out / ("%s.%s" % (name, fmt))
            fig.savefig(p, dpi=self.dpi, bbox_inches="tight")
            self.written.append(p)
        plt.close(fig)

    def sys_style(self, key):
        s = self.systems.get(key, {})
        return {"color": s.get("color", NEUTRAL), "marker": s.get("marker", "o"), "label": s.get("label", key)}

    def axis_title(self, axis):
        group = axis.split("_under")[0].split("_over")[0]
        label = self.cfg.get("stressors", {}).get(group, {}).get("label", group)
        if axis.endswith("_under"):
            return "%s: %s (darker)" % (group, label)
        if axis.endswith("_over"):
            return "%s: %s (brighter)" % (group, label)
        return "%s: %s" % (group, label)

    def axis_key(self, axis):
        group = axis.split("_under")[0].split("_over")[0]
        return (self.stress_order.index(group) if group in self.stress_order else 99, axis)

    def legend(self, fig, keys):
        handles = [Line2D([], [], color=self.sys_style(k)["color"], marker=self.sys_style(k)["marker"],
                          label=self.sys_style(k)["label"]) for k in keys]
        if len(handles) >= 2:          # call after tight_layout; sits above the panels
            fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0))

    # ------------------------------------------------------------------ curves (F1, F2, F3, F5)
    def curve_grid(self, name, exp, metrics, boundaries=None):
        cv, sv = self.t("curves"), self.t("scene_values")
        if not len(cv):
            return
        cv = cv[(cv["exp"] == exp) & cv["metric"].isin(metrics)]
        metrics = [m for m in metrics if m in set(cv["metric"])]
        if not metrics:
            return
        axes_list = sorted(set(cv["axis"]), key=self.axis_key)
        systems = [s for s in self.systems if s in set(cv["system"])] + \
            sorted(set(cv["system"]) - set(self.systems))
        fig, axs = plt.subplots(len(metrics), len(axes_list), squeeze=False,
                                figsize=(2.7 * len(axes_list) + 0.6, 2.3 * len(metrics) + 0.5))
        for j, axis in enumerate(axes_list):
            ca = cv[cv["axis"] == axis]
            mags = sorted(set(ca["magnitude"]))
            xpos = {m: i for i, m in enumerate(mags)}
            ticks = ca.drop_duplicates("magnitude").set_index("magnitude")["severity"].to_dict()
            for i, metric in enumerate(metrics):
                ax = axs[i, j]
                for k, system in enumerate(systems):
                    d = ca[(ca["system"] == system) & (ca["metric"] == metric)].sort_values("magnitude")
                    if not len(d):
                        continue
                    stl = self.sys_style(system)
                    x = d["magnitude"].map(xpos).to_numpy(float)
                    off = (k - (len(systems) - 1) / 2) * 0.06
                    if len(sv):
                        pts = sv[(sv["exp"] == exp) & (sv["system"] == system) & (sv["axis"] == axis)
                                 & (sv["metric"] == metric)]
                        ax.scatter(pts["magnitude"].map(xpos) + off, pts["value"], s=7, color=stl["color"],
                                   alpha=0.35, linewidths=0, zorder=2)
                    ax.fill_between(x + off, d["lo"], d["hi"], color=stl["color"], alpha=0.15, linewidth=0,
                                    zorder=1)
                    ax.plot(x + off, d["mean"], color=stl["color"], marker=stl["marker"], zorder=3,
                            markeredgecolor=SURFACE, markeredgewidth=0.8)
                    if boundaries is not None and len(boundaries) and metric == boundaries["metric"].iloc[0]:
                        b = boundaries[(boundaries["system"] == system) & (boundaries["axis"] == axis)
                                       & boundaries["s_star"].notna()]
                        for s_star in b["s_star"]:
                            ax.plot([xpos.get(s_star, np.nan) + off], [1.0], marker="v", color=stl["color"],
                                    transform=ax.get_xaxis_transform(), clip_on=False, markersize=6)
                ax.set_xticks(range(len(mags)))
                ax.set_xticklabels([("clean" if ticks.get(m) == "base" else ticks.get(m)) for m in mags])
                if i == 0:
                    ax.set_title(self.axis_title(axis), loc="left")
                if j == 0:
                    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        fig.tight_layout()
        self.legend(fig, systems)
        self.save(fig, name)

    # ------------------------------------------------------------------ F4
    def reliability(self, exp, stressor):
        rel = self.t("reliability")
        if not len(rel):
            return
        rel = rel[(rel["exp"] == exp) & rel["axis"].astype(str).str.startswith(stressor)]
        axes_list = sorted(set(rel["axis"]), key=self.axis_key)[:1]
        systems = [s for s in self.systems if s in set(rel["system"])]
        if not axes_list or not systems:
            return
        rel = rel[rel["axis"] == axes_list[0]]
        fig, axs = plt.subplots(1, len(systems), squeeze=False, figsize=(2.8 * len(systems), 2.9))
        for ax, system in zip(axs[0], systems):
            d = rel[rel["system"] == system]
            mags = sorted(set(d["magnitude"]))
            pick = [mags[0], mags[len(mags) // 2], mags[-1]] if len(mags) >= 3 else mags
            pick = list(dict.fromkeys(pick))
            ax.plot([0, 1], [0, 1], color=NEUTRAL, linestyle="--", linewidth=1)
            for c, m in zip(ORDINAL[-len(pick):] if len(pick) < 3 else ORDINAL, pick):
                r = d[d["magnitude"] == m].sort_values("conf")
                sev = r["severity"].iloc[0]
                ax.plot(r["conf"], r["acc"], color=c, marker="o",
                        label="clean (ECE %.3f)" % r["ece"].iloc[0] if sev == "base"
                        else "%s %s (ECE %.3f)" % (stressor, sev, r["ece"].iloc[0]))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_title(self.sys_style(system)["label"], loc="left")
            ax.set_xlabel("mean score in bin")
            ax.legend(loc="upper left", fontsize=7)
        axs[0, 0].set_ylabel("observed change rate")
        fig.tight_layout()
        self.save(fig, "f4_reliability")

    # ------------------------------------------------------------------ F6
    def online_offline(self):
        a, att, warm = self.t("h4_audc_scenes"), self.t("h4_attribution_scenes"), self.t("warmup")
        c = self.cfg["h4"]
        if not len(a):
            return
        stressors = [s for s in c["stressors"] if s in set(a["stressor"])]
        roles = [("online", c["online"]), ("refined", c["refined"])]
        n_warm = len(stressors) + 1 if len(warm) else 0
        fig, axs = plt.subplots(1, 1 + n_warm, squeeze=False, figsize=(3.2 + 2.6 * n_warm, 2.9))
        ax = axs[0, 0]
        for i, s in enumerate(stressors):
            d = a[a["stressor"] == s]
            xs = {role: i + (-0.15 if role == "online" else 0.15) for role, _ in roles}
            for r in d.itertuples(index=False):
                ax.plot([xs["online"], xs["refined"]], [r.audc_online, r.audc_refined], color=GRID, zorder=1)
            for role, system in roles:
                stl = self.sys_style(system)
                v = d["audc_" + role].to_numpy(float)
                ax.scatter(np.full(len(v), xs[role]), v, s=12, color=stl["color"], marker=stl["marker"],
                           zorder=2, alpha=0.8, linewidths=0)
                if len(v) > 1:
                    boots = np.random.default_rng(0).integers(0, len(v), (2000, len(v)))
                    lo, hi = np.quantile(v[boots].mean(1), [0.025, 0.975])
                    ax.errorbar([xs[role] + (0.09 if role == "refined" else -0.09)], [v.mean()],
                                yerr=[[v.mean() - lo], [hi - v.mean()]], color=INK, marker=stl["marker"],
                                capsize=2, markersize=5, zorder=3)
        ax.set_xticks(range(len(stressors)))
        ax.set_xticklabels(stressors)
        ax.set_ylabel("AUDC (1 = no degradation)")
        ax.set_title("Area under normalised mIoU curve", loc="left")
        if len(warm):
            panels = [("base", None)] + [(s, s) for s in stressors]
            for k, (label, s) in enumerate(panels):
                ax = axs[0, 1 + k]
                w = warm if s is None else warm[warm["stressor"] == s]
                if s is None:
                    w = w[w["magnitude"] == 0]
                    w = w[w["stressor"] == w["stressor"].iloc[0]] if len(w) else w
                    title = "Warm-up, clean"
                else:
                    top = w["magnitude"].max()
                    w = w[w["magnitude"] == top]
                    title = "Warm-up, %s %s" % (s, w["severity"].iloc[0] if len(w) else "")
                for role in ("online", "matched", "refined"):
                    d = w[w["role"] == role].sort_values("position")
                    if len(d):
                        stl = self.sys_style(d["system"].iloc[0])
                        ax.plot(d["position"], d["iou"], color=stl["color"], marker=stl["marker"], markersize=3)
                ax.set_title(title, loc="left")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_xlabel("frame index in the revisit stream")
                if k == 0:
                    ax.set_ylabel("IoU (scene mean)")
        keys = [c["online"], c.get("attribution", {}).get("matched"), c["refined"]]
        present = set(warm["system"]) if len(warm) else set()
        fig.tight_layout()
        self.legend(fig, [k for k in keys if k and (k in present or k in (c["online"], c["refined"]))])
        self.save(fig, "f6_online_vs_offline")

    # ------------------------------------------------------------------ F7
    def trust(self):
        rc, sig = self.t("h5_risk_coverage"), self.t("h5_signals")
        if not len(sig):
            return
        rule = self.cfg["h5"].get("trust_rule", {}).get("signals", [])
        fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2), gridspec_kw={"width_ratios": [1.1, 1]})
        ax = axs[0]
        r = rc[rc["population"] == "real"] if len(rc) else rc
        styles = [("LOSO model", SLOTS[0], "-")] + [(n, SLOTS[1 + i], "-") for i, n in enumerate(rule[:3])] + \
            [("random", NEUTRAL, "--"), ("oracle", NEUTRAL, ":")]
        for curve, color, ls in styles:
            d = r[r["curve"] == curve] if len(r) else r
            if len(d):
                ax.plot(d["coverage"], d["risk"], color=color, linestyle=ls, label=curve)
        ax.set_xlabel("coverage (fraction of frames kept, most trusted first)")
        ax.set_ylabel("failure rate on kept frames")
        ax.set_title("Risk-coverage (lower is better)", loc="left")
        ax.legend(fontsize=7)
        ax = axs[1]
        order = list(dict.fromkeys(["LOSO model"] + list(self.cfg["h5"]["signals"])))
        order = [o for o in order if o in set(sig["signal"])]
        y = {o: i for i, o in enumerate(reversed(order))}
        for pop, face in (("real", True), ("null", False)):
            d = sig[sig["population"] == pop]
            for rr in d.itertuples(index=False):
                color = SLOTS[0] if rr.signal == "LOSO model" else INK2
                yy = y[rr.signal] + (0.15 if pop == "real" else -0.15)
                ax.errorbar([rr.auroc], [yy], xerr=[[rr.auroc - rr.lo], [rr.hi - rr.auroc]] if np.isfinite(rr.lo)
                            else None, color=color, marker="o", markerfacecolor=color if face else SURFACE,
                            capsize=2, linewidth=1)
        ax.axvline(0.5, color=NEUTRAL, linestyle="--", linewidth=1)
        ax.set_yticks(list(y.values()))
        ax.set_yticklabels(list(y.keys()))
        ax.set_xlabel("AUROC for frame failure (filled: real pairs, hollow: no-change)")
        ax.set_title("Single signals vs the model", loc="left")
        fig.tight_layout()
        self.save(fig, "f7_trust_signals")

    # ------------------------------------------------------------------ F8
    def components(self, exp):
        c = self.t("components")
        if not len(c):
            return
        c = c[c["exp"] == exp]
        bins = [b for b in c["size_bin"].drop_duplicates()]
        bins = sorted(bins, key=lambda b: float(str(b).strip("([").split(",")[0]))
        axes_list = sorted(set(c["axis"]), key=self.axis_key)
        systems = [s for s in self.systems if s in set(c["system"])]
        fig, axs = plt.subplots(len(bins), len(axes_list), squeeze=False,
                                figsize=(2.7 * len(axes_list) + 0.6, 1.9 * len(bins) + 0.5), sharey=True)
        for j, axis in enumerate(axes_list):
            ca = c[c["axis"] == axis]
            mags = sorted(set(ca["magnitude"]))
            xpos = {m: i for i, m in enumerate(mags)}
            ticks = ca.drop_duplicates("magnitude").set_index("magnitude")["severity"].to_dict()
            for i, b in enumerate(bins):
                ax = axs[i, j]
                for system in systems:
                    d = ca[(ca["system"] == system) & (ca["size_bin"] == b)].sort_values("magnitude")
                    if len(d):
                        stl = self.sys_style(system)
                        x = d["magnitude"].map(xpos)
                        ax.fill_between(x, d["lo"], d["hi"], color=stl["color"], alpha=0.15, linewidth=0)
                        ax.plot(x, d["recall"], color=stl["color"], marker=stl["marker"])
                ax.set_xticks(range(len(mags)))
                ax.set_xticklabels([("clean" if ticks.get(m) == "base" else ticks.get(m)) for m in mags])
                ax.set_ylim(-0.02, 1.02)
                if i == 0:
                    ax.set_title(self.axis_title(axis), loc="left")
                if j == 0:
                    ax.set_ylabel("recall, area %s" % b)
        fig.tight_layout()
        self.legend(fig, systems)
        self.save(fig, "f8_component_recall")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "analysis.yaml"))
    ap.add_argument("--analysis-dir", help="default: <results>/analysis/<name>")
    ap.add_argument("--out", help="default: <analysis-dir>/figures")
    args = ap.parse_args()
    cfg = load_config(args.config)
    adir = Path(args.analysis_dir) if args.analysis_dir else Path(cfg["_paths"]["results"]) / "analysis" / cfg["name"]
    out = Path(args.out) if args.out else adir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    tables = {}
    for p in sorted((adir / "tables").glob("*.csv*")):
        tables[p.name.split(".")[0]] = pd.read_csv(p)
    if not tables:
        sys.exit("no tables in %s; run scripts/analyze.py first" % (adir / "tables"))
    style()
    f = Figs(cfg, tables, out)
    h1e, h2e, h3e = cfg["h1"]["experiment"], cfg["h2"]["experiment"], cfg["h3"]["experiment"]
    f.curve_grid("f1_degradation", h1e, ["miou", "mf1"], boundaries=f.t("h1_boundaries"))
    f.curve_grid("f2_fp_fn", h1e, ["fp_frac_mean", "fn_frac_mean"])
    f.curve_grid("f3_null_false_alarms", h2e, ["far_mean", "ffr"])
    f.reliability(h3e, cfg.get("figures", {}).get("reliability_stressor", "blur"))
    f.curve_grid("f5_calibration", h3e, ["auprc", "cw_fp", "thr_gap", "ece15"])
    f.online_offline()
    f.trust()
    for e in cfg.get("components", {}).get("experiments", [])[:1]:
        f.components(e)
    for p in f.written:
        print(p)


if __name__ == "__main__":
    main()
