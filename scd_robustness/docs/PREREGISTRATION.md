# Pre-registration: E4 main robustness sweep

**CSE 598 Group 13.** Ashish Raj Shekhar, Yashwardhan Ramchaware, Rohit Shinde.
**Frozen on 2026-10-18**, before any E4 run (Decision G2, `experiment_plan.md` §11).

The commit that adds this file freezes, together with it:
- `configs/experiments/e4_main.yaml` (the grid below),
- `configs/analysis.yaml` and `scripts/analyze.py` (the analysis),
- `patches/oscd.patch` and `patches/mv3dcd.patch` (the method code).

Anything decided or changed after E4 results are seen is labelled *exploratory* in the report.

## 1. Gates passed before the freeze

| Gate | Result (ASU Sol, A100 80 GB) | Pass |
|---|---|---|
| G1, E1 clean reproduction | O-SCD online 0.487, refined 0.556 mIoU (target 0.486 / 0.552 ± 0.03); MV3DCD 0.472 on seed 0 (target 0.46–0.48) | yes |
| E2 re-encode check | identity − clean, O-SCD seed 0, 20 pairs: online +0.0007, refined −0.0037 mIoU, both within the noise floor | yes |
| E3 pilot | Cantina and Porch × 2 instances, extended grids; used only to choose the levels below | done |

Clean noise floor (E1, mean seed-to-seed SD of mIoU per pair): O-SCD online 0.0038, refined 0.0119, MV3DCD 0.0034. The `noise_k` rule in `configs/analysis.yaml` uses these runs.

## 2. Frozen E4 grid

Rule (G2): 4 non-zero levels per stressor, spanning from no measurable effect to a ≥ 30% relative mIoU drop or ≥ 50% pose failure. E3 evidence is the relative mIoU drop from identity (MV3DCD / O-SCD online / O-SCD refined, mean of 4 pairs).

| Stressor | E4 levels | E3 evidence |
|---|---|---|
| Motion blur (native px) | 16, 32, 64, 128 | 16 px −1 / −1 / −3%; 64 px −24 / −36 / −48%; 128 px −67 / −87 / −91%. 256 px dropped: O-SCD is already at 0 mIoU. |
| Exposure (EV) | −4, −2, +1, +2 | −2 EV −3 / −4 / −10%; −4 EV −18 / −45 / −55%; +1 EV −3 / −11 / −15%; +2 EV −30 / −37 / −48%. Analysed as two curves (darker, brighter), per `analysis.yaml`. |
| Views kept (of 25) | 60%, 40%, 20%, 12%; 3 random nested subsets each | Worst E3 drop −21 / −8 / −18%; no level reaches 30%. 12% (3 views) added to reach the smallest coverage; 3 subsets average out the subset noise seen in E3. |

Plus the identity variant (severity 0) as each experiment's baseline. 21 variants per pair.

## 3. Scope and the MV3DCD power choice

- **O-SCD:** 10 scenes × 2 instances × 21 variants = **420 runs**, seed 0, `--refine` (online and refined masks from each run).
- **MV3DCD, pose protocol A (oracle COLMAP poses):** **all 10 scenes** × 2 instances × 21 = **420 runs**, plus 20 reference caches (reused from E1).

`experiment_plan.md` §10 required one of two fixes, because with 5 scenes the smallest two-sided Wilcoxon p is 0.0625 and the MV3DCD primary tests could never reject. **Chosen: run MV3DCD on all 10 scenes.** The MV3DCD tests therefore stay in the primary family, which keeps its 15 tests.

## 4. Hypotheses and decision rules

Unchanged from `experiment_plan.md` §1; restated here as frozen.

| ID | Hypothesis | Decision rule |
|---|---|---|
| H1 | Scene-level mIoU falls with severity, and a failure boundary s\* exists in the grid (per stressor and method) | s\* = smallest severity whose scene-bootstrap 95% CI lower bound of mean relative drop `r(s) = 1 − mIoU(s)/mIoU(0)` exceeds 10%; otherwise "no boundary within grid" |
| H2 | On no-change revisits, false alarms rise with blur and exposure (E5) | Paired increase of FAR / FFR vs the clean null across scenes; bootstrap CI of Δ excludes 0 |
| H3 | Scores become less informative and more confidently wrong with severity | Per-scene Spearman ρ(severity, CW_FP) and ρ(severity, AUPRC); mean ρ ≠ 0 by two-sided Wilcoxon |
| H4 | O-SCD online degrades faster than refined under coverage loss and blur | Wilcoxon signed-rank on per-scene AUDC differences (online − refined) |
| H5 | A cheap per-frame signal predicts frame failure (IoU < 0.3) | Leave-one-scene-out AUROC; bootstrap CI above 0.5 |

## 5. Analysis

- **Unit:** the scene (n = 10). Scene value = mean over its 2 instances after averaging views trials.
- **Settings** (`configs/analysis.yaml`): 10,000 scene-bootstrap resamples, percentile 95% CI, seed 598; Holm–Bonferroni at α = 0.05 over the 15 primary tests (H1 6, H2 2, H3 4, H4 2, H5 1); a test that cannot run counts as p = 1; `noise_k = 2`; `min_scenes = 3`.
- **Trend tests:** per-scene Spearman ρ over the baseline plus every level, then exact Wilcoxon on the ρ values.
- **Effect sizes** with CIs are always reported, with per-scene points on every plot.

## 6. Known run failures and how they are handled

- **O-SCD start-up error** (about 1 in 25 runs in E1–E3, all seed 0): a CUDA error while O-SCD records a CUDA graph for its triangulator on dummy inputs, before any data is loaded. Every retry passed. Failed E4 runs are re-submitted until they finish; a run still failing after 3 attempts is listed in the report and left out of its scene's mean.
- **O-SCD renderer crash on no-change scenes** (E3: 13 of 52 no-change runs, at the mildest levels): the FastGS rasterizer launches a zero-block CUDA kernel when no change Gaussian is visible. E4 has no no-change pairs. The fix (skip that launch when it has no work) is applied before E5; it does not change any run that completes without it.
