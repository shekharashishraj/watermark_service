"""Report on the robustness sweeps: severity curves, failure boundaries, calibration, offline vs online.

    python -m scene_change.harness.robustness_report --runs runs/robust --out ROBUSTNESS.md \
        --figures docs/robustness

Reads every ``results*.jsonl`` written by ``scene_change.harness.robustness`` and writes a
Markdown report with figures, plus ``summary.json`` next to the figures for other tools.

Conventions
    reference   the unmodified walkthrough (level "none") for every stressor except
                ``relight``, whose reference is s = 1 (the scenario's own lighting)
    curves      mean over scenes with a 95% t-interval
    boundary    first severity whose mean frame F1 is at least 20% below the reference,
                with a paired 95% interval over scenes that excludes no change
    coverage    ``coverage`` and ``sparse`` remove views, so their frame scores are computed
                on the frames every level kept (the most severe level's frames)
    recalibrated ECE  isotonic map fitted on the reference runs of the *other* scenes,
                applied to this run: does calibration learned on clean captures still hold?
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import calibration as cal
from ..stress import STRESSORS

PLOTTED = ["ours", "oscd-online", "oscd-offline", "video2d",                     # harness sweeps
           "oscd-official-online", "oscd-official-offline", "mv3dcd-official"]    # PASLCD, official code
ALL_METHODS = ["ours", "ours-confirmed", "oscd-online", "oscd-offline", "video2d", "oscd-official-online",
               "oscd-official-offline", "mv3dcd-official"]
LABEL = {"ours": "Ours (3D, offline)", "ours-confirmed": "Ours, confirmed only", "oscd-online": "O-SCD online",
         "oscd-offline": "O-SCD offline (refined)", "video2d": "2D video comparison",
         "oscd-official-online": "O-SCD official, online", "oscd-official-offline": "O-SCD official, refined",
         "mv3dcd-official": "MV3DCD official (offline)"}
# colour follows the method, never its rank; the official O-SCD runs share their re-implementation's hues
COLOR = {"ours": "#2a78d6", "oscd-online": "#eb6834", "oscd-offline": "#1baf7a", "video2d": "#eda100",
         "ours-confirmed": "#e87ba4", "oscd-official-online": "#eb6834", "oscd-official-offline": "#1baf7a",
         "mv3dcd-official": "#008300"}
MARKER = {"ours": "o", "oscd-online": "s", "oscd-offline": "^", "video2d": "D", "ours-confirmed": "v",
          "oscd-official-online": "s", "oscd-official-offline": "^", "mv3dcd-official": "o"}
ONLINE_OFFLINE = (("oscd-online", "oscd-offline"), ("oscd-official-online", "oscd-official-offline"))
STRESS_ORDER = ["blur", "dark", "bright", "relight", "coverage", "sparse", "compress"]
STRESS_TITLE = {"blur": "Motion blur", "dark": "Underexposure", "bright": "Overexposure",
                "relight": "Illumination change", "coverage": "Lost coverage", "sparse": "Fewer views",
                "compress": "Map compression"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DROP = 0.20          # relative F1 drop that counts as failure
T975 = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


# ----------------------------------------------------------------------------
# Loading and per-record metrics
# ----------------------------------------------------------------------------

def load_records(runs: Path) -> list[dict]:
    recs = {}
    for f in sorted(Path(runs).glob("results*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            recs[(r["seed"], r["kind"], r["stressor"], r["level"], r["method"])] = r
    return list(recs.values())


def reference_level(stressor: str) -> float:
    return STRESSORS[stressor].reference if stressor in STRESSORS else 0.0


def reference_is_none(stressor: str) -> bool:
    """True when the stressor's reference level is the unmodified run (level "none")."""
    return stressor not in STRESSORS or STRESSORS[stressor].reference_run == "none"


def severity_levels(stressor: str) -> list[float]:
    """Levels in increasing severity, the reference included."""
    lv = [float(v) for v in STRESSORS[stressor].levels]
    return [float(STRESSORS[stressor].reference)] + lv if reference_is_none(stressor) else lv


def frame_scores(pf: dict, idx, px_per_frame: float) -> dict:
    tp, fp, fn = (np.asarray(pf[k], np.float64)[idx] for k in ("tp", "fp", "fn"))
    has = (tp + fn) > 0
    iou = tp / np.maximum(tp + fp + fn, 1.0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1.0)
    return {"F1": float(f1[has].mean()) if has.any() else None, "IoU": float(iou[has].mean()) if has.any() else None,
            "FA": float(fp[~has].sum() / (px_per_frame * (~has).sum())) if (~has).any() else None,
            "precision": float(tp.sum() / (tp.sum() + fp.sum())) if tp.sum() + fp.sum() > 0 else None,
            "recall_px": float(tp.sum() / (tp.sum() + fn.sum())) if tp.sum() + fn.sum() > 0 else None}


class Records:
    """Index of sweep records with the derived metrics used by the report."""

    def __init__(self, recs: list[dict]):
        self.recs = recs
        self.by = {}
        for r in recs:
            self.by[(r["seed"], r["stressor"], float(r["level"]), r["method"])] = r
        self.seeds = sorted({r["seed"] for r in recs})
        self.methods = [m for m in ALL_METHODS if any(r["method"] == m for r in recs)]
        self.plotted = [m for m in PLOTTED if m in self.methods]
        self.dataset = "PASLCD" if any(r.get("dataset") == "PASLCD" for r in recs) else "harness"
        self.stressors = [s for s in STRESS_ORDER if any(r["stressor"] == s for r in recs)]
        self._common = {}

    def get(self, seed, stressor, level, method):
        if stressor == "none" or (reference_is_none(stressor) and float(level) == reference_level(stressor)):
            return self.by.get((seed, "none", 0.0, method))
        return self.by.get((seed, stressor, float(level), method))

    def common_frames(self, seed, stressor):
        """Frame indices kept at every level of a view-removing stressor (those of its most severe level)."""
        key = (seed, stressor)
        if key not in self._common:
            kept = None
            for lv in sorted(STRESSORS[stressor].levels, reverse=True):
                for m in ALL_METHODS:
                    r = self.by.get((seed, stressor, float(lv), m))
                    if r is not None and "kept" in r:
                        kept = np.asarray(r["kept"])
                        break
                if kept is not None:
                    break
            self._common[key] = kept
        return self._common[key]

    def metrics(self, r: dict, stressor: str) -> dict:
        out = {"F1": r["frames"]["F1"], "IoU": r["frames"]["IoU"], "FA": r["frames"]["false_alarm_rate"],
               "precision": r["pixels"]["precision"] if r["pixels"]["pred_pixels"] else None,
               "recall_px": r["pixels"]["recall"] if r["pixels"]["gt_pixels"] else None}
        if stressor in ("coverage", "sparse"):
            common = self.common_frames(r["seed"], stressor)
            if common is not None:
                kept = np.asarray(r["kept"]) if "kept" in r else np.arange(r["n_frames"])
                pos = np.searchsorted(kept, common)
                if np.all(pos < len(kept)) and np.array_equal(kept[np.minimum(pos, len(kept) - 1)], common):
                    out.update(frame_scores(r["per_frame"], pos, r["errors"]["px"] / r["n_frames"]))
        ch = [c for c in r["changes"] if c["gt_px"] > 0]
        out["recall"] = float(np.mean([c["detected"] for c in ch])) if ch else None
        c = r.get("calibration") or {}
        for k in ("ece", "ace", "brier", "ap", "auroc", "best_F1", "best_t"):
            out[k] = c.get(k)
        conf = c.get("confident") or {}
        out["fp_conf"], out["fn_conf"] = conf.get("fp_confident"), conf.get("fn_confident")
        e = r["errors"]
        out["fp_far_share"] = e["fp_far_px"] / e["fp_px"] if e["fp_px"] else None
        out["fa_frames"] = e["changefree_frames_flagged"] / e["changefree_frames"] if e["changefree_frames"] else None
        reg = r.get("registration") or {}
        out["reg_cm"] = reg.get("trans_cm_median")
        if "localized" in r:
            out["localized"] = r["localized"] / max(r["n_frames"], 1)
        if r.get("fps") is not None:
            out["fps"] = r["fps"]
        return out


def ci(values):
    v = np.asarray([x for x in values if x is not None], np.float64)
    if len(v) == 0:
        return None
    m = float(v.mean())
    if len(v) == 1:
        return m, m, m, 1
    h = T975.get(len(v) - 1, 1.96) * float(v.std(ddof=1)) / np.sqrt(len(v))
    return m, m - h, m + h, len(v)


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------

def curve(R: Records, method: str, stressor: str, key: str):
    """[(level, mean, lo, hi, n, {seed: value})] over severity levels."""
    out = []
    for lv in severity_levels(stressor):
        vals = {}
        for s in R.seeds:
            r = R.get(s, stressor, lv, method)
            if r is not None:
                vals[s] = R.metrics(r, stressor).get(key)
        c = ci(vals.values())
        if c is not None:
            out.append((lv, *c, vals))
    return out


def paired(ref: dict, cur: dict):
    d = [cur[s] - ref[s] for s in cur if s in ref and cur[s] is not None and ref[s] is not None]
    return ci(d)


def boundary(R: Records, method: str, stressor: str, key: str = "F1", drop: float = DROP):
    """(first failing level, relative change there, knee interval) for a metric where higher is better."""
    cv = curve(R, method, stressor, key)
    ref_lv = reference_level(stressor)
    ref = next((c for c in cv if c[0] == ref_lv), None)
    if ref is None or not ref[1]:
        return None
    order = severity_levels(stressor)
    beyond = set(order[order.index(ref_lv) + 1:])          # levels more severe than the reference
    fail = None
    for lv, m, lo, hi, n, vals in cv:
        if lv not in beyond:
            continue
        d = paired(ref[5], vals)
        rel = m / ref[1] - 1.0
        if d is not None and rel <= -drop and (d[3] == 1 or d[2] < 0):
            fail = (lv, rel)
            break
    pts = [c for c in cv if c[0] == ref_lv or c[0] in beyond]
    knee = None
    for a, b in zip(pts[:-1], pts[1:]):
        step = b[1] - a[1]
        if knee is None or step < knee[2]:
            knee = (a[0], b[0], step)
    return {"fail": fail, "knee": knee, "ref": ref[1]}


def recalibrated_ece(R: Records, method: str, stressor: str, level: float, seed: int):
    """ECE after an isotonic map fitted on the reference runs of the other scenes."""
    train = [R.get(s, "none", 0.0, method) for s in R.seeds if s != seed]
    train = [cal.hist_from_json(r["calibration"]["hist"]) for r in train if r is not None and r.get("calibration")]
    r = R.get(seed, stressor, level, method)
    if not train or r is None or not r.get("calibration"):
        return None
    mapping = cal.isotonic_map(cal.pool(train))
    return cal.ece(cal.recalibrate(cal.hist_from_json(r["calibration"]["hist"]), mapping))


def pooled_hist(R: Records, method: str, stressor: str, level: float):
    hs = []
    for s in R.seeds:
        r = R.get(s, stressor, level, method)
        if r is not None and r.get("calibration"):
            hs.append(cal.hist_from_json(r["calibration"]["hist"]))
    return cal.pool(hs) if hs else None


def worst_level(stressor: str) -> float:
    return severity_levels(stressor)[-1]


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def _style(ax, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=8.5)


def _xpos(stressor, levels):
    if stressor == "dark":
        return [-v for v in levels]
    if stressor == "compress":
        return [100 * v for v in levels]
    if stressor in ("coverage", "sparse"):
        return [100 * v for v in levels]
    if stressor == "blur":
        return [100 * v for v in levels]
    return levels


XLABEL = {"blur": "blur length (% of image width)", "dark": "exposure reduction (stops)",
          "bright": "exposure increase (stops)", "relight": "illumination change (x scenario's)",
          "coverage": "views removed in stretches (%)", "sparse": "views removed at random (%)",
          "compress": "baseline Gaussian voxel size (cm)"}


def fig_curves(R: Records, key: str, ylabel: str, path: Path, scale: float = 1.0, methods=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = methods or R.plotted
    st = R.stressors
    ncol = 3
    nrow = int(np.ceil(len(st) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.3 * nrow), dpi=144, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    handles = {}
    for k, s in enumerate(st):
        ax = axes[k // ncol][k % ncol]
        _style(ax, XLABEL[s], ylabel if k % ncol == 0 else None)
        ax.set_title(STRESS_TITLE[s], color=INK, fontsize=10, loc="left")
        for m in methods:
            cv = curve(R, m, s, key)
            if not cv:
                continue
            x = _xpos(s, [c[0] for c in cv])
            y = np.array([c[1] for c in cv]) * scale
            lo = np.array([c[2] for c in cv]) * scale
            hi = np.array([c[3] for c in cv]) * scale
            ax.fill_between(x, lo, hi, color=COLOR[m], alpha=0.10, linewidth=0)
            h, = ax.plot(x, y, color=COLOR[m], lw=1.8, marker=MARKER[m], ms=5.5, mec=SURFACE, mew=1.2,
                         solid_capstyle="round", solid_joinstyle="round", label=LABEL[m])
            handles[m] = h
        if s == "relight":
            ax.axvline(1.0, color=AXIS, lw=0.8)
            ax.text(1.02, 0.98, "scenario lighting", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=7,
                    va="top")
        ax.set_ylim(bottom=0)
    for k in range(len(st), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.legend([handles[m] for m in methods if m in handles], [LABEL[m] for m in methods if m in handles],
               loc="lower center", ncol=4, frameon=False, fontsize=8.5, labelcolor=INK2,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_reliability(R: Records, path: Path, methods=None, stressors=("blur", "dark", "coverage")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [m for m in (methods or R.plotted) if pooled_hist(R, m, "none", 0.0) is not None]
    if not methods:
        return False
    shades = ["#0b0b0b", "#2a78d6", "#eb6834", "#1baf7a"]
    fig, axes = plt.subplots(1, len(methods), figsize=(11, 3.2), dpi=144, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for k, m in enumerate(methods):
        ax = axes[0][k]
        _style(ax, "predicted change score", "observed change frequency" if k == 0 else None)
        ax.set_title(LABEL[m], color=INK, fontsize=9.5, loc="left")
        ax.plot([0, 1], [0, 1], color=AXIS, lw=0.8)
        conds = [("reference", "none", 0.0)] + [(f"{STRESS_TITLE[s].lower()}, worst", s, worst_level(s))
                                               for s in stressors if s in R.stressors]
        for j, (name, s, lv) in enumerate(conds):
            h = pooled_hist(R, m, s, lv)
            if h is None:
                continue
            rel = [b for b in cal.reliability(h, bins=10) if b["n"] >= 200]
            ax.plot([b["conf"] for b in rel], [b["freq"] for b in rel], color=shades[j], lw=1.6, marker="o", ms=4,
                    mec=SURFACE, mew=1.0, label=name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if k == 0:
            ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return True


def fig_thresholds(R: Records, path: Path, methods=None, stressors=("blur", "dark")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [m for m in (methods or [m for m in R.plotted if m not in ("ours",)])
               if pooled_hist(R, m, "none", 0.0) is not None]
    if not methods:
        return False
    shades = ["#0b0b0b", "#2a78d6", "#eb6834"]
    fig, axes = plt.subplots(1, len(methods), figsize=(11, 3.2), dpi=144, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for k, m in enumerate(methods):
        ax = axes[0][k]
        _style(ax, "decision threshold", "frame F1" if k == 0 else None)
        ax.set_title(LABEL[m], color=INK, fontsize=9.5, loc="left")
        conds = [("reference", "none", 0.0)] + [(f"{STRESS_TITLE[s].lower()}, worst", s, worst_level(s))
                                               for s in stressors if s in R.stressors]
        for j, (name, s, lv) in enumerate(conds):
            rows = defaultdict(list)
            for sd in R.seeds:
                r = R.get(sd, s, lv, m)
                if r is None or not r.get("calibration"):
                    continue
                for row in r["calibration"]["sweep"]:
                    if row["F1"] is not None:
                        rows[row["t"]].append(row["F1"])
            if not rows:
                continue
            ts = sorted(rows)
            ax.plot(ts, [np.mean(rows[t]) for t in ts], color=shades[j], lw=1.6, label=name)
        ax.axvline(0.5, color=AXIS, lw=0.8)
        ax.set_ylim(bottom=0)
        if k == 0:
            ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return True


def temporal(R: Records, method: str, stressor: str, level: float, parts: int = 3):
    """Mean frame F1 per part of the walkthrough (early ... late), pooled over scenes."""
    acc = [[] for _ in range(parts)]
    for s in R.seeds:
        r = R.get(s, stressor, level, method)
        if r is None:
            continue
        pf = r["per_frame"]
        n = len(pf["tp"])
        for p in range(parts):
            idx = np.arange(p * n // parts, (p + 1) * n // parts)
            f = frame_scores(pf, idx, r["errors"]["px"] / r["n_frames"])["F1"]
            if f is not None:
                acc[p].append(f)
    return [float(np.mean(a)) if a else None for a in acc]


# ----------------------------------------------------------------------------
# Markdown
# ----------------------------------------------------------------------------

def fmt(v, pct=False, digits=3):
    if v is None:
        return "-"
    if pct:
        return f"{100 * v:.2f}%"
    return f"{v:.{digits}f}"


def lv_label(stressor, lv):
    if stressor == "none" or (reference_is_none(stressor) and lv == reference_level(stressor)):
        return "reference"
    if stressor == "compress":
        return f"{100 * lv:g} cm"
    if stressor == "blur":
        return f"{100 * lv:g}%"
    if stressor in ("dark", "bright"):
        return f"{lv:+g} EV"
    if stressor in ("coverage", "sparse"):
        return f"{100 * lv:g}% removed"
    if stressor == "relight":
        return f"s={lv:g}"
    return f"{lv:g}"


def build_report(R: Records, fig_dir: Path, rel_fig: str, preamble: str | None = None) -> tuple[str, dict]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = {"seeds": R.seeds, "methods": R.methods, "stressors": R.stressors, "curves": {}, "boundaries": {}}
    L = []
    if R.dataset == "PASLCD":
        L.append("# Robustness of 3D change detection on PASLCD (official implementations)\n")
        L.append("Generated by `python -m scene_change.harness.robustness_report` from runs scored with "
                 "`python -m scene_change.harness.paslcd score`. Scenes: "
                 f"{', '.join(str(s) for s in R.seeds)}. Only the post-change images are degraded; frames with a "
                 "ground-truth mask are never removed, so every level is scored on the same frames.\n")
    else:
        L.append("# Robustness of 3D change detection under capture degradation\n")
        L.append("Generated by `python -m scene_change.harness.robustness_report` from the sweeps of "
                 "`scene_change.harness.robustness`. Scenes: held-out harness seeds "
                 f"{', '.join(str(s) for s in R.seeds)} (`mixed`, 8 changes each), every second inspection frame. "
                 "The reference capture is never degraded; each stressor degrades the inspection walkthrough only, "
                 "and ground truth is unchanged by construction.\n")

    if preamble:
        L.append(preamble.strip() + "\n")

    # reference performance
    L.append("## Reference performance\n")
    L.append("Unmodified walkthroughs. Frame scores follow the PASLCD protocol at each method's own threshold; "
             "change recall counts a change as found when at least a quarter of its pixels are flagged. ECE and AP "
             "use the continuous change scores (ours: the highest detection score covering a pixel).\n")
    L.append("| Method | Frame F1 | Frame IoU | False-alarm px | Change recall | ECE | AP | Frames/s |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m in R.methods:
        vals = [R.metrics(R.get(s, "none", 0.0, m), "none") for s in R.seeds if R.get(s, "none", 0.0, m)]
        if not vals:
            continue

        def mean(k):
            c = ci([v.get(k) for v in vals])
            return None if c is None else c[0]
        L.append(f"| {LABEL[m]} | {fmt(mean('F1'))} | {fmt(mean('IoU'))} | {fmt(mean('FA'), pct=True)} | "
                 f"{fmt(mean('recall'))} | {fmt(mean('ece'))} | {fmt(mean('ap'))} | {fmt(mean('fps'), digits=2)} |")
    L.append("")

    # curves
    fig_curves(R, "F1", "frame F1", fig_dir / "f1_vs_severity.png")
    fig_curves(R, "FA", "false-alarm pixels (%)", fig_dir / "false_alarms_vs_severity.png", scale=100)
    fig_curves(R, "recall", "changes found", fig_dir / "recall_vs_severity.png")
    fig_curves(R, "ece", "ECE", fig_dir / "ece_vs_severity.png")
    L.append("## Performance against severity\n")
    L.append(f"![Frame F1 against severity]({rel_fig}/f1_vs_severity.png)\n")
    L.append(f"![False alarms against severity]({rel_fig}/false_alarms_vs_severity.png)\n")
    L.append(f"![Changes found against severity]({rel_fig}/recall_vs_severity.png)\n")
    L.append("Bands are 95% intervals over scenes. Frame scores for lost coverage and fewer views use the frames "
             "every level kept.\n")

    # false positives and misses
    fig_curves(R, "precision", "pixel precision", fig_dir / "precision_vs_severity.png")
    fig_curves(R, "recall_px", "pixel recall", fig_dir / "pixel_recall_vs_severity.png")
    L.append("## False positives and misses\n")
    L.append("Pixel precision falls when false positives grow; pixel recall falls when changes are missed.\n")
    L.append(f"![Pixel precision against severity]({rel_fig}/precision_vs_severity.png)\n")
    L.append(f"![Pixel recall against severity]({rel_fig}/pixel_recall_vs_severity.png)\n")
    L.append("Where the false positives are, at the reference and at each stressor's most severe level: share of "
             "false-positive pixels farther than 3 px from any real change (the rest hug real changes: boundary "
             "and alignment errors), and share of change-free frames with a false alarm of at least 20 px.\n")
    L.append("| Method | Condition | Pixel precision | Pixel recall | False positives away from changes | "
             "Change-free frames flagged |")
    L.append("|---|---|---|---|---|---|")
    for m in R.methods:
        conds = [("reference", "none", 0.0)] + [(f"{STRESS_TITLE[s]}, {lv_label(s, worst_level(s))}", s,
                                                 worst_level(s)) for s in R.stressors]
        for name, s, lv in conds:
            vals = [R.metrics(R.get(sd, s, lv, m), s) for sd in R.seeds if R.get(sd, s, lv, m)]
            if not vals:
                continue

            def mean(k):
                c = ci([v.get(k) for v in vals])
                return None if c is None else c[0]
            L.append(f"| {LABEL[m]} | {name} | {fmt(mean('precision'))} | {fmt(mean('recall_px'))} | "
                     f"{fmt(mean('fp_far_share'), pct=True)} | {fmt(mean('fa_frames'), pct=True)} |")
    L.append("")

    # boundaries
    L.append("## Failure boundaries\n")
    L.append(f"First severity at which mean frame F1 falls at least {int(DROP * 100)}% below the reference, "
             "with a paired 95% interval over scenes that excludes no change; `none` means no such level in "
             "the tested range. The steepest step is the pair of neighbouring levels with the largest F1 drop.\n")
    L.append("| Stressor | " + " | ".join(LABEL[m] for m in R.methods) + " |")
    L.append("|---|" + "---|" * len(R.methods))
    for s in R.stressors:
        cells = []
        for m in R.methods:
            b = boundary(R, m, s)
            summary["boundaries"][f"{m}|{s}"] = b
            if b is None:
                cells.append("-")
                continue
            fail = "none" if b["fail"] is None else f"{lv_label(s, b['fail'][0])} ({100 * b['fail'][1]:+.0f}%)"
            knee = "" if b["knee"] is None or b["knee"][2] >= 0 else (
                f"; steepest {lv_label(s, b['knee'][0])} to {lv_label(s, b['knee'][1])} ({b['knee'][2]:+.3f})")
            cells.append(fail + knee)
        L.append(f"| {STRESS_TITLE[s]} | " + " | ".join(cells) + " |")
    L.append("")

    # per stressor tables
    L.append("## Per-level scores\n")
    for s in R.stressors:
        L.append(f"### {STRESS_TITLE[s]} ({STRESSORS[s].description})\n")
        levels = severity_levels(s)
        L.append("| Method | Metric | " + " | ".join(lv_label(s, lv) for lv in levels) + " |")
        L.append("|---|---|" + "---|" * len(levels))
        for m in R.methods:
            for key, name, pct in (("F1", "frame F1", False), ("FA", "false alarms", True), ("ece", "ECE", False)):
                cv = {c[0]: c for c in curve(R, m, s, key)}
                summary["curves"][f"{m}|{s}|{key}"] = [[c[0], c[1], c[2], c[3], c[4]] for c in cv.values()]
                ref = cv.get(reference_level(s))
                cells = []
                for lv in levels:
                    c = cv.get(lv)
                    if c is None:
                        cells.append("-")
                        continue
                    mark = ""
                    if ref is not None and lv != reference_level(s):
                        d = paired(ref[5], c[5])
                        if d is not None and d[3] > 1 and (d[1] > 0 or d[2] < 0):
                            mark = "*"
                    cells.append(fmt(c[1], pct=pct) + mark)
                L.append(f"| {LABEL[m]} | {name} | " + " | ".join(cells) + " |")
        L.append("\n`*` paired 95% interval over scenes excludes no change from the reference.\n")

    # calibration
    has_rel = fig_reliability(R, fig_dir / "reliability.png")
    has_thr = fig_thresholds(R, fig_dir / "thresholds.png")
    L.append("## Calibration\n")
    L.append(f"![Expected calibration error against severity]({rel_fig}/ece_vs_severity.png)\n")
    if has_rel:
        L.append(f"![Reliability diagrams]({rel_fig}/reliability.png)\n")
        L.append("Reliability diagrams pool all scenes (bins with fewer than 200 pixels hidden). The diagonal is "
                 "perfect calibration.\n")
    if has_thr:
        L.append(f"![Frame F1 against the decision threshold]({rel_fig}/thresholds.png)\n")
    L.append("### Does calibration learned on clean captures survive degradation?\n")
    L.append("ECE of the raw scores and after an isotonic map fitted on the reference runs of the other scenes, "
             "at the reference and at each stressor's most severe level. Confident errors: share of false-positive "
             "pixels scored at least 0.9, and of missed change pixels scored at most 0.1.\n")
    L.append("| Method | Condition | ECE raw | ECE recalibrated on clean | AP | Best threshold | "
             "Confident false positives | Confident misses |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m in R.methods:
        conds = [("reference", "none", 0.0)] + [(f"{STRESS_TITLE[s]}, {lv_label(s, worst_level(s))}", s,
                                                 worst_level(s)) for s in R.stressors]
        for name, s, lv in conds:
            vals = [R.metrics(R.get(sd, s, lv, m), s) for sd in R.seeds if R.get(sd, s, lv, m)]
            if not vals:
                continue
            rec = [recalibrated_ece(R, m, s, lv, sd) for sd in R.seeds]

            def mean(k):
                c = ci([v.get(k) for v in vals])
                return None if c is None else c[0]
            rc = ci(rec)
            L.append(f"| {LABEL[m]} | {name} | {fmt(mean('ece'))} | {fmt(None if rc is None else rc[0])} | "
                     f"{fmt(mean('ap'))} | {fmt(mean('best_t'), digits=2)} | {fmt(mean('fp_conf'), pct=True)} | "
                     f"{fmt(mean('fn_conf'), pct=True)} |")
    L.append("")

    # offline vs online
    for on_m, off_m in ONLINE_OFFLINE:
        if on_m not in R.methods or off_m not in R.methods:
            continue
        L.append(f"## Offline against online ({LABEL[on_m]} vs {LABEL[off_m]})\n")
        L.append("O-SCD's online masks are rendered right after each frame; the offline masks come from the same "
                 "change field after refinement over all frames. Mean frame F1 by walkthrough third:\n")
        L.append("| Condition | Online: early / middle / late | Offline: early / middle / late |")
        L.append("|---|---|---|")
        conds = [("reference", "none", 0.0)] + [(f"{STRESS_TITLE[s]}, {lv_label(s, worst_level(s))}", s,
                                                 worst_level(s)) for s in R.stressors]
        for name, s, lv in conds:
            on = temporal(R, on_m, s, lv)
            off = temporal(R, off_m, s, lv)
            if all(v is None for v in on + off):
                continue
            L.append(f"| {name} | {' / '.join(fmt(v) for v in on)} | {' / '.join(fmt(v) for v in off)} |")
        L.append("")
        L.append("Relative F1 change from the reference at the most severe level:\n")
        third = [m for m in ("ours", "mv3dcd-official") if m in R.methods]
        cols = [on_m, off_m] + third
        L.append("| Stressor | " + " | ".join(LABEL[m] for m in cols) + " |")
        L.append("|---|" + "---|" * len(cols))
        for s in R.stressors:
            cells = []
            for m in cols:
                cv = {c[0]: c for c in curve(R, m, s, "F1")}
                ref, w = cv.get(reference_level(s)), cv.get(worst_level(s))
                cells.append("-" if ref is None or w is None or not ref[1] else f"{100 * (w[1] / ref[1] - 1):+.0f}%")
            L.append(f"| {STRESS_TITLE[s]} | " + " | ".join(cells) + " |")
        L.append("")

    # coverage: known unknowns
    if "coverage" in R.stressors:
        L.append("## Changes the walkthrough never saw\n")
        L.append("Under lost coverage, some changes are not visible in any kept frame. Only our method reports "
                 "areas it could not check; the other methods return no change there.\n")
        L.append("| Views removed | Changes not visible | Reported as not checked (ours) |")
        L.append("|---|---|---|")
        for lv in STRESSORS["coverage"].levels:
            lost = flagged = 0
            for sd in R.seeds:
                r0, r = R.get(sd, "none", 0.0, "ours"), R.get(sd, "coverage", lv, "ours")
                if r0 is None or r is None:
                    continue
                seen0 = {c["id"] for c in r0["changes"] if c["gt_px"] > 0}
                for c in r["changes"]:
                    if c["id"] in seen0 and c["gt_px"] == 0:
                        lost += 1
                        flagged += bool(c.get("unverified_flag"))
            L.append(f"| {100 * lv:g}% | {lost} | {flagged} ({fmt(flagged / lost if lost else None, pct=True)}) |")
        L.append("")

    vols = sorted(c["volume"] for r in R.recs for c in r["changes"] if c.get("volume") is not None)
    med = float(np.median(vols)) if vols else 0.0

    # compression: small against large changes
    if "compress" in R.stressors:
        L.append("## Map compression: do small changes fail first?\n")
        L.append("The baseline map is fused at coarser voxel sizes (fewer, larger Gaussians); the inspection "
                 "walkthrough is unchanged. Share of visible changes found, split at the median box volume "
                 f"({med:.3f} m^3), and frame F1.\n")
        L.append("| Method | Voxel | Gaussians | Small changes found | Large changes found | Frame F1 |")
        L.append("|---|---|---|---|---|---|")
        for m in R.methods:
            if not any(R.get(sd, "compress", lv, m) for sd in R.seeds for lv in STRESSORS["compress"].levels):
                continue
            for lv in severity_levels("compress"):
                small, large, gauss, f1 = [], [], [], []
                for sd in R.seeds:
                    r = R.get(sd, "compress", lv, m)
                    if r is None:
                        continue
                    f1.append(r["frames"]["F1"])
                    if r.get("gaussians"):
                        gauss.append(r["gaussians"])
                    for c in r["changes"]:
                        if c["gt_px"] > 0 and c.get("volume") is not None:
                            (small if c["volume"] < med else large).append(c["detected"])
                if not f1:
                    continue
                L.append(f"| {LABEL[m]} | {100 * lv:g} cm | {int(np.mean(gauss)) if gauss else '-'} | "
                         f"{fmt(float(np.mean(small)) if small else None, digits=2)} | "
                         f"{fmt(float(np.mean(large)) if large else None, digits=2)} | {fmt(ci(f1)[0])} |")
        L.append("")

    # errors by change type and size
    L.append("## Which changes are missed\n")
    L.append(f"Share of visible changes found, by type and size (small: box volume below the median, "
             f"{med:.3f} m^3), at the reference and pooled over the most severe level of every stressor.\n")
    groups = ["removed", "added", "moved", "appearance", "small", "large"]
    L.append("| Method | Condition | " + " | ".join(groups) + " |")
    L.append("|---|---|" + "---|" * len(groups))
    for m in R.methods:
        for name, conds in (("reference", [("none", 0.0)]),
                            ("most severe", [(s, worst_level(s)) for s in R.stressors])):
            acc = defaultdict(list)
            for s, lv in conds:
                for sd in R.seeds:
                    r = R.get(sd, s, lv, m)
                    if r is None:
                        continue
                    for c in r["changes"]:
                        if c["gt_px"] <= 0:
                            continue
                        acc[c["type"]].append(c["detected"])
                        if c.get("volume") is not None:
                            acc["small" if c["volume"] < med else "large"].append(c["detected"])
            L.append(f"| {LABEL[m]} | {name} | " + " | ".join(
                fmt(float(np.mean(acc[g])) if acc[g] else None, digits=2) for g in groups) + " |")
    L.append("")
    return "\n".join(L) + "\n", summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/robust")
    ap.add_argument("--out", default="ROBUSTNESS.md")
    ap.add_argument("--figures", default="docs/robustness")
    ap.add_argument("--preamble", default=None, help="Markdown inserted after the introduction (findings)")
    args = ap.parse_args(argv)
    R = Records(load_records(Path(args.runs)))
    if not R.recs:
        raise SystemExit(f"no results in {args.runs}")
    out = Path(args.out)
    fig_dir = Path(args.figures)
    try:
        rel = fig_dir.resolve().relative_to(out.resolve().parent)
    except ValueError:
        rel = fig_dir.resolve()
    pre = Path(args.preamble).read_text() if args.preamble and Path(args.preamble).exists() else None
    text, summary = build_report(R, fig_dir, rel.as_posix(), pre)
    out.write_text(text)
    (fig_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"wrote {out} and figures in {fig_dir} ({len(R.recs)} records, seeds {R.seeds})")


if __name__ == "__main__":
    main()
