# scd_robustness — robustness & calibration study of 3DGS change detection (CSE 598 Group 13)

Config-driven pipeline to degrade PASLCD revisit captures (blur, exposure, lighting, missing views,
no-change pairs), run **patched O-SCD** and **MV3DCD** on ASU Sol in isolated conda/mamba environments,
and evaluate every run the same way.

- The plan: [`docs/experiment_plan.md`](docs/experiment_plan.md).
- The papers: [`docs/papers_explained.md`](docs/papers_explained.md).

```
configs/      paths.yaml, perturbations.yaml (stressor catalog), experiments/e*.yaml (one per plan stage),
              analysis.yaml (pre-registered analysis)
envs/         scd-tools.yml (CPU), oscd.yml (py3.12, CUDA 12.8), mv3dcd.yml (py3.8, CUDA 12.4)
patches/      oscd.patch, mv3dcd.patch — applied to the pinned upstream commits
setup/        sol.env + numbered one-time setup scripts + smoke-test job
scd/          library: perturbations, variant builder, planner, runner, evaluator, metrics, stats, analysis/
scripts/      CLIs: scd_plan, make_variants, run_oscd, run_mv3dcd, evaluate_run, collect_results, check_dataset,
              analyze, make_figures
slurm/        array-job scripts + submit_experiment.sh (dependency chain) + analyze.sbatch
tests/        CPU tests on a synthetic dataset (pytest)
```

## How it fits together
```
experiment.yaml ──scd_plan──▶ manifests/<exp>/variants.jsonl ──make_variants──▶ data/variants/…  (CPU)
                           └▶ runs_oscd.jsonl / runs_mv3dcd.jsonl (+ refcache) ──run_*──▶ runs/…/eval/  (GPU)
                                                                        collect_results ──▶ results/<exp>/summary.csv
```
Every step is **idempotent**:
- Variants built from the same config are skipped.
- Runs whose `_scd_status.json` says `ok` are skipped.
- Re-submitting an experiment therefore only redoes what failed.

## One-time setup on Sol
Everything goes under `SCD_ROOT` (default `/scratch/$USER/scd`): envs, caches, code, data, outputs. Nothing goes in `$HOME`.

```bash
git clone <this repo> && cd <repo>/scd_robustness
source setup/sol.env                    # sets SCD_ROOT, caches, helpers (scd_activate, scd_offline)

bash setup/01_fetch_code.sh             # clone O-SCD@3abfeaa, MV3DCD@b45606c, apply patches/
bash setup/02_create_envs.sh all        # prefix envs in $SCD_ROOT/envs (scd-tools, oscd, mv3dcd)
bash setup/03_fetch_data.sh             # both PASLCD zips + scripts/check_dataset.py report
bash setup/04_prestage_models.sh        # SAM2, XFeat, DINOv2 into caches -> jobs run offline
sbatch -p <partition> -q <qos> setup/05_smoke_test.sbatch    # E0 on one GPU (submit from scd_robustness/)
```

Notes:
- **Where to run the scripts.** Steps 1–4 need internet, so run them on a login node or in an interactive session.
- **No module needed.** `02_create_envs.sh` compiles the CUDA extensions for `TORCH_CUDA_ARCH_LIST` (default `8.0;8.6;8.9;9.0`: A100, A30, A40, L40S, H100), so no GPU is needed while building. The CUDA toolkit and GCC come from conda-forge inside each env, so no cluster CUDA module is required.
- **Cluster names.** `module avail mamba` shows the conda module (default `mamba/latest`; override with `SCD_MAMBA_MODULE`). Use `sinfo` / `sacctmgr show qos` or the Sol docs for partition and QOS names.
- **After step 3, fix two paths if needed.** Open `$SCD_ROOT/results/dataset_facts.json`. If the unzipped layout differs from `configs/paths.yaml`, edit `data_online` / `data_offline`. Record the answers in `docs/dataset_facts.md`.
- **Smoke-test output.** `05_smoke_test.sbatch` prints the timings you need to replace the plan's GPU-hour estimate.

## Running an experiment
```bash
source setup/sol.env
export SCD_SBATCH_ARGS="-p <partition> -q <qos>"          # + "-A <account>" if you have one
export SCD_GPU_ARGS="--gres=gpu:a100:1"                   # optional: pin a GPU type
bash slurm/submit_experiment.sh configs/experiments/e1_clean.yaml
# ... when the jobs finish:
scd_activate scd-tools && python scripts/collect_results.py --experiment configs/experiments/e1_clean.yaml
```

Order:

| Stage | Config | Purpose |
|---|---|---|
| E0 | `e0_smoke` | Smoke test |
| E1 | `e1_clean` | Clean reproduction (gate G1) |
| E2 | `e2_identity` | Re-encode check |
| E3 | `e3_pilot` | Pilot — then **freeze the grid in `e4_main.yaml` and commit `docs/PREREGISTRATION.md`** |
| E4 | `e4_main` | Main sweep |
| E5 | `e5_null` | No-change false alarms |
| E7 | `e7_matched` | Online vs offline attribution |

**Sizes** (full dataset):

| Config | O-SCD runs | MV3DCD runs | MV3DCD reference caches |
|---|---|---|---|
| e1 | 60 | 40 | 40 (one per seed) |
| e2 | 20 | — | — |
| e3 | 68 + 52 null | 68 | 4 |
| e4 | 360 | 180 | 10 |
| e5 | 200 (null) | — | — |
| e7 | 120 | — | — |

Useful knobs:
- `SCD_RUN_CHUNK=4` runs 4 lines per GPU task, which amortises O-SCD start-up. Raise `-t` with it.
- `SCD_MAX_PARALLEL` limits concurrently running tasks per array (default 16).
- `DRY=1` prints the `sbatch` commands without submitting.

Debugging a single run by hand:
```bash
scd_activate oscd; scd_offline
python scripts/run_oscd.py --manifest $SCD_ROOT/manifests/e4_main/runs_oscd.jsonl --index 12 --dry-run   # show command
python scripts/run_oscd.py --manifest $SCD_ROOT/manifests/e4_main/runs_oscd.jsonl --index 12 --force     # rerun
```
Each run folder contains:
- `logs/<step>.log`
- `_scd_status.json` (GPU, host, git SHAs, timings, errors)
- the method outputs
- `eval/<output>_{frames.csv,components.csv,hists.npz,summary.json}`

## Configs
**`configs/perturbations.yaml`** is the stressor catalog. Each entry is `kind: clean | photometric | coverage`.
- `photometric` entries name an `fn` in `scd/perturb.py` plus `params`. They can also take a `subset` (`position` start/middle/end and `count`), which limits the stressor to a burst of consecutive frames.
- Perturbations run in linear light at native resolution and keep file names and formats.
- Randomness is seeded from `md5(scene|stream|image|stressor|severity|trial)`. The same image therefore gets the same perturbation in both instances and for both methods.
- **To add a stressor:** write `fn(lin, severity, params, rng) -> (lin, info)` in `scd/perturb.py`, register it in `FUNCS`, and add a catalog entry.

**`configs/experiments/*.yaml`**:
```yaml
name: e4_main
methods:
  oscd:   {scenes: all, pairs: [Instance_1, Instance_2], seeds: [0]}
  mv3dcd: {scenes: [Cantina, Garden], pairs: [Instance_1], seeds: [0], arms: {default: {reference: cache}}}
  extra:  {method: oscd, pairs: ["null:Instance_1:Instance_2"], sweep: {blur: [32]}}   # labelled block
sweep:  {identity: [0], blur: [16, 32], views: [0.6, 0.4]}     # stressor -> severities
trials: {views: 3}                                             # repeats with different random draws
```

Other fields:
- **pairs:** `Instance_X` is a real pair. `null:<map>:<revisit>` is a no-change pair: the revisit is the other *pre-change* capture of the scene (O-SCD only).
- **arms:** override method arguments (`scd/plan.py: METHOD_DEFAULTS`).
  - O-SCD: `refine`, `steps_per_frame` (an int, or `auto` to match the refine budget), `refine_total`, `resolution`.
  - MV3DCD: `reference: cache|train`, `resolution`, `t`, iteration counts.

## What the patches change (defaults reproduce the released behaviour)

**`oscd.patch`**
- **Pose-rejection crash.** The released code crashes when a frame cannot be localised (`init_inference_pose` returns `None`). The patch now logs and skips the frame.
- **New flags:** `--seed`, `--steps_per_frame`, `--refine_total`, `--save_scores` and `--frame_log`.
  - `--save_scores` saves the raw change score before the `> 0.5` threshold.
  - `--frame_log` writes per-frame matches, inliers, pose ok, timings, SSF loss and cue mean.
- **Run summary.** A `run_summary.json` records the pose-failure count and FPS.
- **Offline XFeat.** `SCD_XFEAT_REPO` loads XFeat from a local clone, for offline nodes.

**`mv3dcd.patch`**
- `SCD_SAVE_SCORES=1` saves the continuous change map before the `> 127` threshold.
- `SCD_SEED` sets the seed in `safe_state`.
- `SCD_DINOV2_REPO` loads DINOv2 from a local clone.

**Evaluator (`scd/evaluate.py`).** It replaces the released evaluators, which silently skip frames that have no prediction. It scores such frames as empty masks and still reports the released number as `paper_miou`. It also adds:
- pooled, boundary-tolerant and object-level metrics;
- score histograms (AUROC / AUPRC / ECE / confident errors);
- false-alarm rates for null pairs.

MV3DCD's reference 3DGS is trained on pre-change images only. It is therefore built **once** per scene/instance/seed (`runs_mv3dcd_refcache.jsonl`) and linked into every variant run.

## Analysis and statistics
**`configs/analysis.yaml`** is the pre-registered analysis. It holds:
- which experiments and systems to compare;
- how each severity maps to a magnitude;
- the bootstrap and Holm settings;
- H1–H5 with their thresholds, trust signals and scene folds.

Freeze it with `docs/PREREGISTRATION.md` before E4.

```bash
scd_activate scd-tools
python scripts/analyze.py --config configs/analysis.yaml          # -> $SCD_ROOT/results/analysis/main/
python scripts/make_figures.py --config configs/analysis.yaml     # -> .../main/figures/f1…f8 (png + pdf)
python scripts/analyze.py --only gates,h1 --n-boot 1000          # quick look while runs are still going
sbatch -p <partition> -q <qos> -o $SCD_ROOT/logs/slurm/analysis-%j.out slurm/analyze.sbatch   # as a CPU job
```

It works on partial results: whatever has finished is analysed, and the rest is listed in the report's coverage table.

**Outputs:**
- `report.md`: coverage, then gate G1, noise floor, E2 identity check, the primary-test decisions with Holm, s\* table, false alarms, calibration transfer, online/refined/matched attribution, trust rule and compute used.
- `results.json`
- `tables/*.csv`: every table, including per-scene values.
- `data/`: the loaded runs, frames, components and histograms, for custom analysis in pandas.

| Module | What it does |
|---|---|
| `scd/stats.py` | Exact Wilcoxon (tied ranks), Spearman, scene bootstrap, Holm, AUDC, hinge knee, scene-cluster AUROC, risk–coverage |
| `scd/analysis/scene.py` | Scene values (the statistical unit), degradation curves with Δ and relative drop, noise floor, component recall |
| `scd/analysis/calibration.py` | Reliability diagrams, clean-fitted isotonic recalibration and threshold transfer across scene folds |
| `scd/analysis/hypotheses.py` | G1, identity check, H1 (trend + s\* + knee), H2, H3, H4 (AUDC) + E7 attribution and warm-up, H5 (LOSO model, trust rule) |

**Power caveat.** MV3DCD runs on 5 scenes in `e4_main`, where the smallest attainable Wilcoxon p-value is 0.0625, so its primary tests can never pass. The report says so. Before E4, decide between running all 10 scenes and reporting MV3DCD as CIs only (`docs/experiment_plan.md` §10).

## Testing locally (no GPU)
```bash
scd_activate scd-tools && python -m pytest -q tests      # synthetic dataset and synthetic study, end to end
```

## Troubleshooting
- **An extension fails to build with the conda CUDA toolkit.**
  1. Find a cluster CUDA module: `module avail cuda`.
  2. Recreate the env with it: `FORCE=1 SCD_CUDA_MODULE=<cuda-12.8 module> bash setup/02_create_envs.sh oscd`. Use a 12.4 module for `mv3dcd`.
- **Jobs hang or fail downloading models.** Run `setup/04_prestage_models.sh` on a login node. The array jobs call `scd_offline` and use the local XFeat and DINOv2 clones.
- **Downstream jobs stay `DependencyNeverSatisfied`.**
  1. A variant task failed; its log is under `$SCD_ROOT/logs/slurm`.
  2. Fix the cause.
  3. Cancel the pending jobs (`scancel`) and re-run `submit_experiment.sh`. Finished work is skipped.
- **CuPy cannot find NVRTC or headers.** The smoke test checks this first. Make sure the env was activated through `scd_activate`, which sets `CUDA_PATH` and `LD_LIBRARY_PATH` via the env's activation hook.
