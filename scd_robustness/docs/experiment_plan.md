# Experiment plan: robustness and calibration of 3DGS scene change detection

**CSE 598 Group 13.** Ashish Raj Shekhar, Yashwardhan Ramchaware, Rohit Shinde.

Background on the methods and dataset is in [`papers_explained.md`](papers_explained.md). Section numbers such as §9.6 refer to sections of this document.

**Status: plan only.** Nothing has been run yet. Every number marked *(estimate)* must be replaced by a measurement from step E0.

---

## 0. Summary

We will degrade the **revisit (post-change) images** of the PASLCD benchmark in controlled steps: motion blur, exposure shifts, local illumination and fewer views. At each step we measure how two released 3DGS change detectors behave:
- **O-SCD** (online, CVPR 2026), and its refined (offline) output;
- **MV3DCD** (offline, CVPR 2025).

We measure four things:
1. accuracy and where it breaks (failure boundary);
2. **false alarms when nothing changed** (no-change control);
3. whether change scores stay **calibrated / informative** as inputs degrade;
4. whether cheap per-frame signals predict failures (a **trust rule**).

We also compare **online vs offline** behavior with a control that separates "access to future frames" from "more optimization".

**What changed from the proposal**, with reasons in the linked sections:
- Added a real **no-change control** (§8.2).
- Reframed "confidence" as score **separability and calibration** (§9.6).
- Made the **scene** (n = 10) the statistical unit (§10).
- Coverage levels now match the dataset's 25 post-change views (§7.5).
- Added **two pose protocols** for MV3DCD (§8.1) and an **iteration-matched** online arm (§8.3).
- Fixed citations: added O-SCD; GS-DIFF is discussed but not evaluated (§16).

---

## 1. Research questions and pre-registered hypotheses

**Primary question.** How does the accuracy of current 3DGS-based change detectors degrade when the revisit capture is degraded (motion blur, exposure shift, fewer views)? Do they raise false alarms when only the observation changed? And do their continuous scores, or cheap per-frame signals, indicate when predictions are unreliable?

**Rules.**
- Write these hypotheses, thresholds and analysis choices into `PREREGISTRATION.md` **before** the main sweep (E4). Commit it.
- Anything decided after seeing E4 results is labelled *exploratory*.

| ID | Hypothesis (H1) — tested against H0: "no effect beyond the clean noise floor" | Primary measurement | Decision rule | Serves success criterion |
|---|---|---|---|---|
| **H1** | For each stressor and method, scene-level mIoU falls as severity grows, and a **failure boundary** s\* exists inside the grid | Relative drop `r(s) = 1 − mIoU(s)/mIoU(0)` per scene | s\* = smallest severity whose scene-bootstrap 95% CI lower bound of mean `r(s)` exceeds **10%**. 10% is above MV3DCD's real-lighting drop (7.2%). If no severity qualifies, report "no boundary within grid". | (1) failure boundary |
| **H2** | On revisits with **no real change**, the false-alarm rate rises with blur and exposure severity | False-alarm pixel rate FAR, and frame false-alarm rate FFR (§9.2) | Paired increase vs the clean null across scenes; bootstrap CI of Δ excludes 0 | (1) + the proposal's "change reported when only the observation changed" |
| **H3** | As severity grows, scores become **less informative and more confidently wrong** | Primary: confident-FP rate `CW_FP = P(p ≥ 0.9 \| y = 0)` and AUPRC. Secondary: ECE after clean-fitted recalibration; fixed-vs-oracle F1 gap | Per-scene Spearman ρ(severity, metric); mean ρ ≠ 0 by Wilcoxon (two-sided) | (2) calibration |
| **H4** | O-SCD **online** masks degrade faster than **refined** masks under coverage loss and blur | Per-scene area under the normalized degradation curve (AUDC, §10) | Wilcoxon signed-rank on per-scene AUDC differences. The iteration-matched arm then attributes the gap (§8.3). | (3) online vs offline |
| **H5** | A cheap per-frame signal predicts frame failure | Frame failure = IoU < 0.3 (or FAR > 1% on null frames); signals in §9.7 | Leave-one-scene-out AUROC with bootstrap CI above 0.5. Report the resulting trust rule and its precision / coverage. | "Practical recommendation" |

**Predictions** (for the discussion, not decision rules):
- Blur hits O-SCD earlier than MV3DCD at the same native blur. Reasons: O-SCD runs at 2× the resolution and relies on XFeat for pose.
- Global exposure shifts are partly cancelled by per-frame normalization.
- Coverage loss hurts online masks more than refined masks.
- The PnP inlier count is the best single trust signal for O-SCD.

---

## 2. Scope

### 2.1 Methods evaluated
| Arm | What | Why |
|---|---|---|
| **O-SCD online** | `change_mask/`: each frame's mask uses only earlier frames | The only released online 3DGS SCD method; state of the art (0.486 / 0.638) |
| **O-SCD refined** | `change_mask_refined/` from the same run with `--refine` | Offline-style re-optimization over all frames; same poses and cues |
| **O-SCD iteration-matched** | Online, with the per-frame step count raised so total steps ≈ refine budget (§8.3) | Separates "future frames" from "more optimization" |
| **MV3DCD** | Released pipeline (`run.sh`) | Independent offline method with different cues (DINOv2 + SSIM, hard thresholds) and COLMAP poses |

### 2.2 Not evaluated, and why
- **GS-DIFF.** Code and annotations are unreleased; a public issue reports failed reproduction. Discuss its observability idea only.
- **3DGS-CD.** 0.209 mIoU on PASLCD; built for object rearrangement. Cite it as related work only.

### 2.3 Data
- **PASLCD**: 10 scenes × 2 instances = 20.
- 25 annotated post-change images per scene.
- Both downloads are needed:
  - `PASLCD_online.zip` for O-SCD;
  - `PASLCD.zip` for MV3DCD, which also supplies the joint COLMAP poses for pose analysis.

---

## 3. Design principles (read before doing anything)

1. **Degrade only the revisit.** The reference capture is the robot's map; keep it clean. This keeps ground-truth masks valid, and lets O-SCD's prebuilt reference and MV3DCD's reference checkpoint be reused. Degrading the reference is future work (E11).
2. **The scene is the statistical unit.** The two instances of a scene appear to share the same post-change images (verify in S3), so they are not independent.
   - Average the two instances into one per-scene value.
   - Bootstrap and test over the **10 scenes**.
3. **Score every method with the same patched evaluator** (§6, P7). Also report the paper-style number so reproduction can be checked.
4. **Perturb at native resolution.** Then report the *effective* severity at each method's working resolution: O-SCD runs at 1/4, MV3DCD at 1/8.
5. **Label pose sources.** O-SCD localizes itself on degraded frames. MV3DCD is run under **oracle poses** (A) and, where feasible, **re-registered poses** (B).
6. **Pilot, then freeze.** Tune the severity grid on the pilot (E3). Then freeze the grid and the analysis code before E4.
7. **Never tune on test results.** Use fixed thresholds (0.5 as released). Any fitted quantity (a recalibration map, a trust-rule threshold) is fitted on some scenes and tested on the others.
8. **Log everything per run and per frame** (§5.3). Missing frames count as failures, never as "skipped".

---

## 4. Environment setup (step by step)

Commands are templates for ASU Sol (SLURM). Module names vary: check `module avail cuda` and `module avail mamba`. Build CUDA extensions **on a GPU node** matching the one you'll run on.

### S1. Accounts, storage, layout
```
$SCRATCH/scd/
  code/{O-SCD, MV3DCD}          # pinned clones
  data/raw/{PASLCD_online, PASLCD}
  data/variants/                # generated perturbed datasets (symlinks + perturbed images)
  runs/                         # method outputs
  results/                      # runs.jsonl, per-frame parquet, figures
  cache/{hf, torch, inductor}   # model weights + compile cache
```
Keep code and docs in this git repo under `scd_robustness/`. Keep data and outputs on scratch only.

### S2. Code, pinned to the commits we read
```bash
git clone https://github.com/Chumsy0725/O-SCD code/O-SCD && git -C code/O-SCD checkout 3abfeaa
git clone https://github.com/Chumsy0725/MV3DCD code/MV3DCD && git -C code/MV3DCD checkout b45606c
```

### S3. Data (login node, which has internet) and a sanity check
```bash
cd data/raw
wget https://huggingface.co/datasets/ChamudithaJay/PASLCD/resolve/main/PASLCD_online.zip && unzip PASLCD_online.zip -d PASLCD_online
wget https://huggingface.co/datasets/ChamudithaJay/PASLCD/resolve/main/PASLCD.zip        && unzip PASLCD.zip -d PASLCD
sha256sum *.zip > checksums.txt
```
Then answer these four questions and record the answers in `docs/dataset_facts.md`:
1. **Do the instances share post-change images?** Compare
   `md5sum PASLCD_online/.../Instance_1/Cantina/inference_scene/images/*`
   with the same files under `Instance_2`.
2. **How many frames are in each inference stream**, and how many have a GT mask? The online zip may contain more frames than the 25 annotated ones.
3. **Does every instance contain `reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply`?** O-SCD needs it.
4. **What are the native image size and file format** (JPEG or PNG)? This matters for blur units and for re-encoding (§7.0).

### S4. Environment A — O-SCD
```bash
module load cuda/12.8          # name varies; must match the cu128 wheels
conda create -n oscd python=3.12 -y && conda activate oscd
pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu128
pip install cupy-cuda12x
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"   # A100 / A30 / RTX 40xx — set to your GPU
cd code/O-SCD && pip install -r requirements.txt   # builds the rasterizers, simple-knn, fused-ssim
# pre-stage SAM2 weights on the login node; compute nodes may be offline
export HF_HOME=$SCRATCH/scd/cache/hf
huggingface-cli download facebook/sam2.1-hiera-tiny
```
In every job script, set:
- `HF_HOME` (as above) and `HF_HUB_OFFLINE=1`;
- `TORCHINDUCTOR_CACHE_DIR=$SCRATCH/scd/cache/inductor`, because O-SCD calls `torch.compile(mode='max-autotune')` and would otherwise recompile in every job.

### S5. Environment B — MV3DCD
```bash
conda create -n mv3dcd python=3.8 -y && conda activate mv3dcd
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0
pip install plyfile opencv-python timm matplotlib scikit-learn torchmetrics
pip install code/MV3DCD/submodules/simple-knn
# pre-cache DINOv2 ViT-L/14 (loaded via torch.hub) on the login node
export TORCH_HOME=$SCRATCH/scd/cache/torch
python -c "import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vitl14')"
```
Watch the dataset folder order: MV3DCD expects `data/PASLCD/<Scene>/<Instance_X>`.

### S6. Environment C — analysis and perturbation (CPU)
Python 3.11 with `numpy scipy pandas pyarrow opencv-python scikit-image scikit-learn matplotlib pycolmap`. Add COLMAP (module or conda-forge `colmap`) for pose protocol B.

### S7. Smoke tests (this is E0)
- **O-SCD**: `Instance_1/Garden`, with `--refine`, then its evaluator.
- **MV3DCD**: `Garden/Instance_1`, via a one-scene copy of `run.sh`.

Record for each: wall-clock time per stage, peak GPU memory, and output folder sizes. **These numbers replace every (estimate) in §12.**

---

## 5. Repository layout, conventions, logging

### 5.1 Layout (in this repo)
```
scd_robustness/
  docs/            papers_explained.md, experiment_plan.md, PREREGISTRATION.md, dataset_facts.md
  patches/         oscd.patch, mv3dcd.patch       (applied to the pinned commits)
  configs/         sweep_mvp.yaml, sweep_stretch.yaml, scenes.yaml
  perturb/         io.py, blur.py, exposure.py, illumination.py, coverage.py, make_variant.py, null_pairs.py
  run/             run_oscd.py, run_mv3dcd.py, colmap_reregister.py, manifest.py, slurm_array.sh
  eval/            evaluate.py (counts + histograms), calibration.py, components.py, signals.py
  analysis/        aggregate.py, stats.py, figures.py, report_tables.py
  tests/           test_perturb.py (identity + determinism), test_eval.py (toy masks)
```

### 5.2 Variant naming
Each variant is identified as `{method}/{scene}/{instance}/{stressor}-{severity}/t{trial}-s{seed}`. Examples: `oscd/Cantina/Instance_1/blur-32/t0-s0` and `mv3dcd/Porch/Instance_2/views-10/t2-s0`.

Variant datasets **symlink** every unchanged file (reference images, reference reconstruction, `sparse/`, `gt_mask/`) and write only the perturbed inference images.

### 5.3 What every run records
One line per run in `runs.jsonl`:
- identity: `run_id`, method, arm, scene, instance, stressor, severity, trial, seed, pose protocol, resolution, git SHAs (this repo and the method repo), GPU type;
- timing: start and end time, runtime per stage;
- frames: total frames, frames with a pose, frames with a prediction;
- status and output paths.

One row per frame in `frames.parquet`:
- counts and metrics: TP, FP, FN, TN, IoU, F1, precision, recall, GT area, predicted area;
- score histograms (§9.4) and boundary-band variants;
- signals: PnP match count, inliers, pose accepted, registration success, Laplacian variance, mean brightness, fraction of clipped pixels, mean score, score entropy, reference-render alpha coverage, SSF loss at the frame.

---

## 6. Patches to the released code (Ashish, week 2)

Apply these as a patch file to the pinned commits. Keep each change behind a flag so the defaults still reproduce the papers.

| ID | Target | Change | Why |
|---|---|---|---|
| **P1** | O-SCD `oscd.py` (the call to `init_inference_pose`) and `poses/pose_initializer.py` | If the pose is rejected (returns `None`) or there are too few matches for PnP, log the reason, skip the frame, and write no mask | The released code crashes (`Rt, _ = None`) when a frame has ≤ 100 inliers. Heavy blur or darkness will trigger this. |
| **P2** | O-SCD, online and refined outputs | Save the continuous map (raw rendered change value, float16 `.npy`) before the `> 0.5` threshold | Calibration needs scores |
| **P3** | MV3DCD `render_viewpoints.py` | Save `final_image_mask` (float) before the `> 127` threshold | Same |
| **P4** | O-SCD | Per-frame JSON log: matches, PnP inliers, accepted flag, per-stage time, SSF loss | Trust signals (H5) |
| **P5** | O-SCD | Add `--seed`; the code hardcodes seed 0 | Noise floor |
| **P6** | O-SCD | Add `--steps_per_frame` (default 16) and `--refine_total` (default 3000) | Iteration-matched arm (§8.3) |
| **P7** | New evaluator `eval/evaluate.py` | Details below | Same scoring for every method |
| **P8** | MV3DCD, optional | A small `render_heldout.py`: load the trained change-aware checkpoint and render at poses of held-out views | Coverage "unseen view" evaluation, as in MV3DCD Fig. 4 |

**P7 details.** The released `utils/evaluate.py` silently skips ground-truth frames that have no prediction file, so failed frames would vanish from the average. The new evaluator:
- (a) treats a missing prediction as an **all-zero mask** and flags it;
- (b) outputs TP / FP / FN / TN per frame, not just per-image IoU and F1;
- (c) reads continuous maps and builds histograms;
- (d) reproduces the paper metric exactly (per-image mean) as a check;
- (e) supports "null mode" (§8.2).

**Acceptance test.** On clean data with defaults, the patched code must reproduce the unpatched masks bit-for-bit (same seed), apart from the extra files. Run once, then compare.

---

## 7. Perturbation specification (Yashwardhan, weeks 2–3)

### 7.0 Rules shared by all perturbations
- **Deterministic randomness.** Each image's random parameters come from `seed = int(md5(f"{scene}|{instance}|{image}|{stressor}|{severity}|{trial}").hexdigest()[:8], 16)`. Do **not** use Python's `hash()`, which changes between processes.
- **Work in linear light.** Convert sRGB to linear before blur and exposure, then convert back:
  - `lin = x/12.92` if `x ≤ 0.04045`, else `((x+0.055)/1.055)^2.4`;
  - the inverse is `12.92·y` if `y ≤ 0.0031308`, else `1.055·y^(1/2.4) − 0.055`.
- **Keep file names and formats**, because the loaders look images up by name.
  - JPEG sources: re-encode with quality 98–100 and 4:4:4 chroma.
  - **Also pass the clean images through the same decode → encode pipeline as severity 0** (the "identity" variant). Any effect of re-encoding then sits inside the baseline.
- **Never touch GT masks.** All perturbations are photometric or remove whole frames, so GT pixel alignment is preserved.
- **Resolution.** Perturb native-resolution images. In `frames.parquet`, record the **effective** severity at each working resolution (blur length ÷4 for O-SCD, ÷8 for MV3DCD) and the variance of the Laplacian after downscaling.

### 7.1 Motion blur (MVP)
- **Kernel.** A straight line of length `L` pixels (odd; if even, use L+1) through the center of an L×L kernel, rotated by an angle θ ~ U[0°, 180°) drawn per image, normalized to sum to 1.
- **Apply** with `cv2.filter2D(..., borderType=cv2.BORDER_REFLECT)` in linear space.
- **Grid (native px): L ∈ {0, 16, 32, 64, 128}.** That is 0–32 px at O-SCD's 1008×560 and 0–16 px at MV3DCD's resolution. The pilot may add 256.
- **Stretch.** Trajectory-consistent blur: interpolate between neighbouring camera poses and average the corresponding homography-warped copies of the image.

### 7.2 Global exposure (MVP)
- `lin' = clip(lin · 2^EV, 0, 1)`, then back to sRGB and quantize to 8 bits.
- **Grid: EV ∈ {−3, −2, −1, 0, +1, +2}.** +2 clips highlights; −3 crushes shadows.
- **Stretch.** A white-balance shift: per-channel gains of (1+k, 1, 1−k), k ∈ {0.1, 0.2, 0.3}.

### 7.3 Local illumination / cast shadows (tier 2)
- **Smooth gain field.**
  1. Draw white noise.
  2. Apply a Gaussian blur with σ = image width / 8.
  3. Rescale to [−1, 1].
  4. `gain = clip(1 + a·field, 0.05, ∞)`, applied in linear space.
- **Grid: a ∈ {0, 0.25, 0.5, 0.75}.**
- **Shadow variant.** A random convex polygon covering 10–30% of the image, darkened by a factor `(1 − a)` with a 15-px feathered edge.
- These are exactly the "distractors" the methods claim to suppress, so this directly tests that claim.

### 7.4 Sensor noise (stretch)
Poisson–Gaussian noise in linear space: `y = Poisson(lin·k)/k + N(0, σ_r²)`. Use three levels, e.g. `(k, σ_r) = (400, 0.002), (100, 0.005), (25, 0.01)`, roughly like rising ISO.

### 7.5 View coverage (MVP)
- **Keep a fraction of the post-change views: 100%, 60%, 40%, 20%.** With 25 frames that is 25 / 15 / 10 / 5 (same levels as MV3DCD Fig. 4). If S3 shows a different frame count, keep the fractions.
- **Nested subsets:** the 5-view set ⊂ the 10-view set ⊂ the 15-view set. Draw **3 random trials** (seeded); average trials within a scene before statistics.
- **Evaluation.**
  - O-SCD: kept views only; it has no poses for frames it never saw.
  - MV3DCD: kept views, and additionally the held-out views via P8 (tier 2).
  - Report both the kept-views metric and, for MV3DCD, the all-views metric.
- **Implementation.**
  - O-SCD variant: `inference_scene/images` contains only the kept images (symlinks); the order is preserved.
  - MV3DCD variant: delete the dropped "test" images **from the COLMAP model** (`pycolmap`: `rec.deregister_image(id)` then write) as well as from `images/`. The loader iterates the COLMAP entries and would fail on a missing file.
- **Gap variant (tier 2).** Drop the views whose camera centers fall in a contiguous angular sector around the scene (30% or 50% of views). This models a region the robot never revisited, rather than sparse sampling.

### 7.6 Combined and burst (stretch)
- **Combined:** blur 32 + EV −2 ("night motion").
- **Burst (O-SCD only):** 3 consecutive frames at L = 128, placed at the start, middle or end of the stream. Tests whether early bad frames poison the online memory.

### 7.7 Checks (E2)
- Unit tests:
  - identity perturbation = decode/encode only;
  - same seed → byte-identical output;
  - kernel sums to 1;
  - exposure EV 0 = identity after quantization.
- Contact sheets: one image per stressor × severity, for the report.
- Distribution plots: Laplacian variance and mean brightness vs severity, to check the grid is monotone and well spread.

---

## 8. Protocols

### 8.1 Pose protocols
- **O-SCD: self-localization (end-to-end).** Poses come from its own XFeat + PnP on the degraded frames.
  - Rejected frames produce no mask; they are scored as empty masks and counted in "pose failure rate".
  - Optional analysis: align O-SCD's estimated camera centers to the offline COLMAP poses (similarity transform on camera centers) and report rotation/translation error per frame.
- **MV3DCD protocol A: oracle poses.** Use the joint COLMAP model shipped in `PASLCD.zip` (computed from clean images). This isolates the change-detection stage. **Label every result "oracle poses".**
- **MV3DCD protocol B: re-registration (tier 2; blur and exposure only).** Reproduces the paper's actual pipeline, where degraded after-photos must be registered to the reference.
  1. Fresh COLMAP on reference images + degraded test images:
     - `feature_extractor` with intrinsics fixed to the original camera parameters;
     - `exhaustive_matcher`;
     - `mapper` with focal and principal-point refinement off.
  2. `model_aligner`: align the new model to the original frame using the **reference image camera centers**, which exist in both models (`--alignment_type custom`, robust alignment on).
  3. Copy the aligned poses of the registered test images into a copy of the original model. Keep the original reference poses and points, so the reference checkpoint stays valid. Remove unregistered test images.
  4. Record the registration rate and the pose error vs oracle.

  Unregistered frames are scored as empty masks.
- **Reference checkpoint reuse (MV3DCD).** `train.py` trains the reference only on non-"test" images, so `chkpnt7000.pth` depends only on the reference images, poses and points. Train it **once per instance** and copy it into each variant's output folder; start the pipeline at the `render_viewpoints.py` step. **Check once:** a variant built from a reused checkpoint must match a from-scratch run within the noise floor.

### 8.2 No-change control (null pairs)
**Idea.** Instance 1's and Instance 2's reference captures are both **pre-change** captures of the same scene, under different lighting. Using one as the map and the other as the "revisit" gives a pair where **nothing changed**, so every detection is a false alarm.

**Construction for O-SCD** (easy; it localizes itself):
```
null/{scene}/map-I1_revisit-I2/
  reference_scene          -> Instance_1/{scene}/reference_scene          (symlink)
  reference_reconstruction -> Instance_1/{scene}/reference_reconstruction (symlink)
  inference_scene/images   -> Instance_2/{scene}/reference_scene/images   (perturbed per severity)
```
Also build the reverse direction (map I2, revisit I1). That gives 2 null pairs per scene (20 in total); cluster by scene in the statistics.

**MV3DCD** needs the Instance-2 reference images registered into the Instance-1 model (protocol B tooling). This is tier 2.

**Metrics** (no GT needed, since every pixel is negative):
- **FAR** = flagged pixels / all pixels;
- **FFR** = fraction of frames whose flagged area > 0.5% of the image (below that, a detection is too small to matter; the smallest real change in PASLCD is 0.17%);
- mean score;
- FAR split by reference-render alpha (≥ 0.5 = the map has seen this region, < 0.5 = unseen).

**Grid:** clean + blur {16, 32, 64, 128} + EV {−3, −2, −1, +1, +2}.

**Caveats.**
- Check visually that the two reference captures show the same scene state. Drop any scene where they don't.
- Revisit views may see regions the map never saw. The alpha split above keeps this separate from true false alarms.

### 8.3 Online vs offline (O-SCD arms)
For a stream of `n` frames:

| Arm | Steps | Masks produced |
|---|---|---|
| Online (default) | 16 per frame → 16n total (≈ 400 for n = 25) | After each frame |
| Refined | Online, then refine to 3,000 total over all frames | After the visit |
| **Iteration-matched online** | `k = round(3000/n)` per frame (≈ 120) → ≈ 3,000 total | After each frame |

**Interpretation.**
- Matched ≈ Online < Refined → the offline advantage comes from **access to future frames (hindsight)**.
- Matched ≈ Refined > Online → the advantage comes from the **optimization budget**, not hindsight.
- Matched in between → both contribute.

Refine also uses a standard densify/prune schedule, whereas the online arms densify at step 4 of each frame. Note this residual difference.

Also report the **warm-up curve**: mean IoU vs frame index, and how quickly online masks approach refined quality. Check whether degradation lengthens it.

**MV3DCD vs O-SCD** differs in cues, pose source and resolution. Describe their failure patterns side by side, but do **not** attribute the differences to "offline vs online".

### 8.4 Resolution
- Run both methods at their released resolutions (O-SCD `--resolution 4`, MV3DCD `RESOLUTION=8`).
- Plot degradation curves against **both** native and effective severity.
- Optional sensitivity check: MV3DCD at resolution 4 on clean data only, to show how much resolution alone matters.

### 8.5 Seeds and noise floor
- O-SCD: seeds {0, 1, 2} via P5.
- MV3DCD: three repeated runs; CUDA rasterization is non-deterministic even with fixed seeds.
- Noise floor = per-scene SD of the clean metric across repeats. A degradation effect counts only if the mean drop is > 2 × noise SD **and** its CI excludes 0.

---

## 9. Metrics (Rohit, with Yashwardhan for FP/FN)

### 9.1 Detection metrics
- Per frame: TP, FP, FN, TN on the change class.
  - `IoU = TP/(TP+FP+FN)`, `F1 = 2TP/(2TP+FP+FN)`, `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`.
  - Resize predictions to GT size with nearest-neighbour, as the released evaluator does.
- **Paper-comparable number:** mean over frames within an instance, then over instances.
- **Primary number for our statistics:** per-scene value = mean over frames, then mean over the scene's two instances. Also report **pooled** (micro) versions: sum the counts, then compute the metric.
- **Missing prediction** (pose rejected, registration failed, or frame dropped by P1) = all-zero mask. Report the failure rate alongside.
- **Boundary-tolerant variant** (sensitivity check for blur): ignore a band of ±3 px (at working resolution) around GT boundaries. Build it by morphological dilation minus erosion.

### 9.2 Error decomposition
- **FP vs FN** pixels per frame (MV3DCD's Table 5 reports these too) as functions of severity.
- **Edge FPs:** the share of FP pixels within 3 px of Canny edges of the reference render. This tests the "blur → false positives along edges" prediction.
- **Null-pair metrics:** FAR and FFR (§8.2).

### 9.3 Change-object recall by size and type
- Split GT into connected components (8-connectivity; merge components under 50 px into their nearest neighbour). A component is **detected** if ≥ 50% of its pixels are predicted as change.
- **Size bins** by fraction of image area: < 0.5%, 0.5–2%, 2–5%, > 5%.
- **Type (tier 2):** label each component structural or surface using the dataset's per-scene change descriptions. That is 91 changes across 10 scenes, a few hours of work. It enables "which change types fail first".
- This also answers the proposal's compression extension (do small changes degrade earlier?).

### 9.4 Score histograms: storage-cheap and exact enough
For every frame, store two 256-bin histograms of the score `p` over [0, 1]: one for GT-changed pixels and one for GT-unchanged pixels.

Everything in §9.5–9.6 is computed from histograms summed over frames, scenes or folds: AUROC, AUPRC, Brier score, ECE, reliability curves and threshold sweeps.

Keep full-resolution float16 score maps only for the pilot, plus 2 scenes of the main sweep for figures. Full maps cost about 1.1 MB per frame at O-SCD resolution.

**Which quantity is `p`:**
- **O-SCD:** `p = sigmoid(raw)` (what the loss trains). Its decision rule `raw > 0.5` equals `p > 0.622`. Report the decision threshold alongside.
- **MV3DCD:** `p = clip(rendered change × alpha mask, 0, 1)`, decision `p > 0.5`.
- Neither is trained as a probability. That is the point of the analysis, not a flaw in it.

### 9.5 Threshold-free and threshold metrics
- **AUROC and AUPRC** (average precision) over pixels. AUPRC is the primary one, because only 3.5% of pixels are positive.
- **Threshold sweep:** F1(τ) for τ ∈ {0, 0.01, …, 1}.
  - Oracle threshold τ\* per scene; **threshold gap** `Δτ = F1(τ*) − F1(τ_dec)`.
  - **Clean-tuned threshold transfer:** pick τ on the clean runs of fold A scenes and apply it to fold B at each severity. This asks whether a threshold tuned in good conditions stays good. Then swap the folds.

### 9.6 Calibration
- **Reliability diagrams**, 15 equal-mass bins, clean vs each severity: the observed change rate per bin plotted against mean `p`.
- **ECE (adaptive):** `Σ_b (n_b/N)·|ȳ_b − p̄_b|`. Report it, but read it cautiously: with 96.5% of pixels negative, ECE mostly measures the background.
- **Class-conditional confident errors** (primary for H3; independent of prevalence):
  - `CW_FP = P(p ≥ 0.9 | y = 0)`;
  - `CW_FN = P(p ≤ 0.1 | y = 1)`;
  - the share of all errors that are confident.
- **Brier score** with its reliability / resolution decomposition.
- **Calibration under shift** (the deployment question):
  1. Fit an isotonic map `p → P(change)` on **clean** runs of 5 scenes (fold A).
  2. Apply it to fold-B scenes at every severity and compute ECE.
  3. Swap the folds and average.

  If ECE rises with severity, a calibration learned in good conditions does not transfer to bad ones. Split by scene, never by instance.

### 9.7 Frame-level trust signals (H5)
- **Signals per frame:**
  - O-SCD: PnP inliers and matches; pose accepted; SSF loss at the frame.
  - MV3DCD: registration success under protocol B.
  - Both methods: Laplacian variance of the input (blur), mean brightness and clipped-pixel fraction (exposure), predicted change area, mean score entropy, and reference-render alpha coverage (unseen regions).
- **Failure label:** frame IoU < 0.3 on real pairs; FAR > 1% on null pairs.
- **Analysis.**
  - AUROC of each single signal.
  - A leave-one-scene-out logistic regression on all signals.
  - **Risk–coverage curves:** sort frames by signal, keep the top c%, and plot the error on kept frames. Report AURC.
- **Output: a trust rule.** For example: "reject a frame if inliers < X or Laplacian variance < Y". Pick X and Y on 9 scenes and test on the 10th (rotate). Report precision and coverage.

### 9.8 Runtime
Wall-clock time per stage and per frame (O-SCD FPS), measured on the same GPU type; report which.

---

## 10. Statistics (Rohit)

- **Unit.** The scene (n = 10). Per-scene value = mean over its two instances, after averaging coverage trials and seeds.
- **Degradation curve.**
  - Absolute drop `Δ(s) = m(s) − m(0)` and relative drop `r(s) = 1 − m(s)/m(0)`.
  - 95% CI from 10,000 bootstrap resamples of scenes (percentile method).
- **Failure boundary s\*.** As defined in H1 (10% rule).
  - Secondary: a knee from a piecewise-linear fit, with the breakpoint's bootstrap CI.
  - Report the pose-failure rate at s\* next to it.
- **AUDC** (area under the normalized degradation curve): integrate `m(s)/m(0)` over the severity levels, via the trapezoid rule on an evenly spaced index of levels. It is one number per scene per stressor, used for comparing methods and arms.
- **Tests.**
  - Wilcoxon signed-rank on per-scene paired values (exact; with n = 10 the smallest two-sided p is about 0.002).
  - Trends: per-scene Spearman ρ, then Wilcoxon on the ρ values.
- **Multiple comparisons.** The primary family is:
  - H1: 3 stressors × 2 methods;
  - H2: 2 stressors;
  - H3: 2 metrics × 2 methods;
  - H4: 2 stressors;
  - H5: 1.

  That is 15 tests. Apply **Holm–Bonferroni** at α = 0.05; everything else is exploratory.
- **Always report effect sizes with CIs.** Show per-scene points on every plot; 10 dots are more honest than a bar.

---

## 11. Experiment cards

Each card lists: goal · inputs · procedure · runs · outputs · pass / decision · owner.

### E0 — Environment and smoke test (week 1)
- **Goal:** both pipelines run end to end on one instance.
- **Procedure:** steps S1–S7.
- **Outputs:** timing table, disk usage, `dataset_facts.md`.
- **Pass:** masks produced; the released evaluator gives sensible numbers for Garden.
- **Owner:** Ashish (O-SCD), Rohit (MV3DCD).

### E1 — Clean reproduction and noise floor (week 2)
- **Runs:**
  - O-SCD online + refined: 20 instances × 3 seeds (60 runs).
  - MV3DCD: all 20 instances × 1, plus 5 scenes × 2 instances × 2 extra repeats (40 runs).
- **Gate G1:**
  - O-SCD online mIoU within **0.486 ± 0.03**; refined within **0.552 ± 0.03**. The README says the FastGS version runs slightly *better*.
  - MV3DCD within **0.46–0.48 ± 0.03**.
  - The patched evaluator's paper-mode number equals the released evaluator's.
- **If G1 fails:** check resolution, versions, seeds and GPU type; compare per-scene numbers with the per-scene tables. Robustness conclusions use *relative* drops, so a small absolute gap is acceptable if documented.
- **Owner:** Ashish.

### E2 — Perturbation pipeline validation (week 3)
- **Procedure:** unit tests (§7.7); identity variant run for O-SCD on all 20 instances.
- **Pass:** identity metrics equal clean within the noise floor; contact sheets reviewed by all three members.
- **Owner:** Yashwardhan.

### E3 — Pilot (week 4)
- **Scope:** Cantina (indoor, FF) and Porch (outdoor, 360°) × 2 instances; seed 0.
- **Grid:** every MVP stressor over an **extended** grid (blur up to 256, EV −4…+3, views down to 3), both methods, plus the O-SCD null pairs for these 2 scenes.
- **Decision G2 (freeze):** for each stressor, choose 4 non-zero levels that span from "no measurable effect" to "≥ 30% relative drop or ≥ 50% pose failure". Write them into `PREREGISTRATION.md`, together with every hypothesis threshold (§1) and the analysis code version. Commit before E4.
- **Owner:** all three.

### E4 — Main robustness sweep (weeks 5–6) → H1
- **O-SCD:** 20 instances × (blur 4 + exposure 4 + views 3 levels × 3 trials) = 20 × 17 = **340 runs**. Each run yields online **and** refined masks.
- **MV3DCD, protocol A:** 5 scenes × 2 instances × 17 = **170 runs**. The 5 scenes: Cantina, Printing Area, Meeting Room, Garden, Porch (indoor/outdoor, FF/360° balanced).
- **MV3DCD, protocol B (tier 2):** the same 10 instances × 8 blur/exposure levels = 80 runs, plus COLMAP jobs.
- **Tier 2:** local illumination, 3 levels × 20 instances (O-SCD) = 60 runs.
- **Outputs:** degradation curves with CIs, s\* table, FP/FN decomposition, edge-FP share, component recall by size.
- **Owners:** Ashish (runs), Yashwardhan (FP/FN analysis).

### E5 — No-change false alarms (week 6) → H2
- **O-SCD:** 10 scenes × 2 directions × (1 clean + 4 blur + 5 exposure) = **200 runs**. These are short: inference only, no GT.
- **MV3DCD (tier 2):** 5 scenes × 1 direction × 5 levels via protocol B.
- **Outputs:** FAR and FFR vs severity; FAR split by alpha coverage; example frames.
- **Owner:** Yashwardhan.

### E6 — Calibration and score informativeness (weeks 6–7) → H3
- **Inputs:** histograms from E1 and E4 (no new runs).
- **Outputs:**
  - reliability diagrams (clean vs mid vs severe);
  - `CW_FP` / `CW_FN`, AUPRC, Brier and ECE vs severity;
  - recalibration-transfer ECE;
  - threshold gap, and clean-tuned threshold transfer.
- **Owner:** Rohit.

### E7 — Online vs offline (week 7) → H4
- **Runs:** iteration-matched arm, 10 scenes × Instance 1 × (clean + views 3 levels × 3 trials + blur 2 levels) = **120 runs**. The online and refined arms come free from E4.
- **Outputs:**
  - AUDC per arm with paired tests;
  - the attribution table (§8.3);
  - warm-up curves (IoU vs frame index);
  - burst experiment if time allows (E10).
- **Owner:** Ashish.

### E8 — Trust rule (week 7) → H5
- **Inputs:** per-frame signals from E4 and E5.
- **Outputs:**
  - single-signal AUROC;
  - leave-one-scene-out model;
  - risk–coverage curves;
  - the final **trust rule**, with its held-out precision and coverage.
- **Owner:** Rohit.

### E9 — Compression extension (stretch, week 8)
- **Procedure.** Prune the O-SCD reference `.ply` to keep 50%, 25% or 10% of Gaussians, ranked by opacity × volume. Optionally also quantize (float16 positions/scales, 8-bit colors).
- **Record:** reference-render PSNR against the clean reference at reference views, to quantify how much capacity was removed.
- **Runs:** 20 instances × 3 levels = 60 runs (clean inputs).
- **Output:** component recall by size bin vs capacity. Does a small change fail first?

### E10 — Combined stressors and burst (stretch, week 8)
Night-motion combination; burst at start / middle / end (O-SCD, 10 scenes × Instance 1).

### E11 — Reference-side degradation (future work)
Degrade the map instead of the revisit. This needs new reconstructions, so it is out of scope.

---

## 12. Run budget (estimate; replace after E0)

| Experiment | O-SCD runs | MV3DCD runs | Notes |
|---|---|---|---|
| E1 | 60 | 40 | |
| E2 | 20 | — | Identity variant |
| E3 | ~70 | ~70 | 2 scenes × 2 instances × extended grid |
| E4 | 340 (+60 tier 2) | 170 (+80 tier 2) | |
| E5 | 200 (short) | (25 tier 2) | |
| E7 | 120 | — | |
| E9 / E10 | 60 + 30 | — | Stretch |
| **Total MVP** | **≈ 810** | **≈ 280** | |

**Per-run cost (estimate).**
- **O-SCD: 2–5 min.**
  - The online loop is only 2–3 s for 25 frames at 11 FPS; the refine loop and the startup dominate.
  - Startup covers loading SAM2, the `torch.compile` warm-up, and XFeat + exhaustive matching on the reference images.
  - Two cheap speed-ups:
    - (a) cache the reference keyframes and their 3D anchors per instance (pickle);
    - (b) process several variants per job to amortize startup.
- **MV3DCD: 5–8 min** with the reference checkpoint reused (the paper reports 479 s including the reference).

**Total ≈ 810 × 4 min + 280 × 7 min ≈ 54 + 33 ≈ 90 GPU-hours** for the MVP, well within a semester's cluster allocation if split into SLURM arrays.

**Storage.** With histograms plus binary masks, a few GB. With full float16 maps, about 1.1 MB per frame, so keep them for the pilot plus 2 scenes only.

**SLURM array template:**
```bash
#!/bin/bash
#SBATCH -J scd-sweep
#SBATCH -p general            # partition name varies
#SBATCH --gres=gpu:1 -c 8 --mem=48G -t 00:30:00
#SBATCH --array=0-339%16      # 340 runs, at most 16 at a time
source activate oscd
export HF_HOME=$SCRATCH/scd/cache/hf HF_HUB_OFFLINE=1 TORCHINDUCTOR_CACHE_DIR=$SCRATCH/scd/cache/inductor
python run/run_oscd.py --manifest configs/e4_oscd.jsonl --index $SLURM_ARRAY_TASK_ID --skip-if-done
```
The manifest has one JSON line per run (variant path, arm, seed, output path). `--skip-if-done` makes reruns safe.

---

## 13. Timeline (10 weeks) and gates

| Week | Ashish (design, benchmarking) | Yashwardhan (perturbations, failure analysis) | Rohit (calibration, evaluation) | Gate |
|---|---|---|---|---|
| 1 | S1–S4, O-SCD smoke test | Read papers; perturbation library skeleton; sRGB/linear helpers | S5, MV3DCD smoke test; evaluator skeleton | E0 done |
| 2 | Patches P1–P6; E1 runs | Blur + exposure + coverage generators; unit tests | P7 evaluator; histogram pipeline; paper-mode check | **G1** reproduction |
| 3 | Manifest + SLURM tooling; reference-checkpoint reuse check | E2 validation; null-pair builder; contact sheets | Calibration and statistics code on E1 data (dry run) | E2 pass |
| 4 | E3 pilot runs | Pilot FP/FN look | Pilot calibration look | **G2** freeze → `PREREGISTRATION.md` |
| 5 | E4 O-SCD sweep | Local-illumination generator (tier 2); component labelling | Aggregation scripts | |
| 6 | E4 MV3DCD sweep (A); protocol B tooling | E5 null runs + analysis | E6 calibration | |
| 7 | E7 online vs offline | E4 failure analysis (edge FPs, size recall) | E8 trust rule | |
| 8 | Stretch: E10 / protocol B | Stretch: E9 compression | Statistics finalization (Holm, CIs) | Results freeze |
| 9 | Figures and tables | Qualitative figure strips | Calibration figures | Draft report |
| 10 | Report + presentation | Report + presentation | Report + presentation | Submit |

**Cut order if behind:** E10 → E9 → protocol B → local illumination → MV3DCD null → iteration-matched arm (keep online vs refined).

**Never cut:** E1, E4 blur/exposure/views for O-SCD, E5 for O-SCD, E6, E8.

---

## 14. Deliverables: figures and tables for the report

**Figures.**
- **F1.** Degradation curves (mIoU and F1 vs severity), one panel per stressor; lines for O-SCD online, O-SCD refined and MV3DCD (A); per-scene dots and bootstrap bands; s\* marked.
- **F2.** FP vs FN pixels per frame vs severity, plus the edge-FP share.
- **F3.** Null-pair false-alarm curves (FAR, FFR), with the real-lighting null marked.
- **F4.** Reliability diagrams: clean vs mid vs severe, for each method.
- **F5.** AUPRC, `CW_FP` and threshold gap vs severity.
- **F6.** Online vs refined vs iteration-matched (AUDC bars with per-scene dots) and warm-up curves.
- **F7.** Risk–coverage curves for the trust signals, with the trust rule marked.
- **F8.** Change-object recall by size bin vs severity (and vs capacity, if E9 runs).
- **F9.** Qualitative strips: input | reference render | score map | mask | GT at three severities.

**Tables.**
- **T1.** Reproduction vs paper (per scene).
- **T2.** Failure boundaries s\* with CIs and the pose-failure rate at s\*.
- **T3.** Calibration summary.
- **T4.** Trust rule with held-out precision and coverage.
- **T5.** Compute used.

---

## 15. Risks and fallbacks

| Risk | Early warning | Fallback |
|---|---|---|
| The online zip lacks the prebuilt reference `.ply` | S3 check | Train the reference with the 3DGS/FastGS code on `reference_scene` (30k iterations); document it |
| CUDA extension builds fail | S4/S5 | Match the CUDA module to the wheel version; build on a GPU node; set `TORCH_CUDA_ARCH_LIST` |
| Compute nodes are offline | Smoke test hangs on downloads | Pre-stage HF and torch.hub caches; `HF_HUB_OFFLINE=1` |
| Reproduction off by > 0.03 | G1 | Compare per scene; ask in the repos' issues; continue with relative metrics, documented |
| MV3DCD too slow | E0 timing | Keep the 5-scene subset; drop protocol B |
| COLMAP protocol B eats time | Week 6 | Report oracle-pose results only, clearly labelled |
| Instances do not share the post-change set | S3 hashes | The scene is still the unit (conservative); treat instances as repeated measures |
| The two reference captures differ in content | Visual check before E5 | Drop that scene from E5 |
| Knee outside the grid | E3 | Extend the grid before freezing (that is what the pilot is for) |
| Too many hypotheses → nothing significant | — | Report effect sizes and CIs; a "robust up to max severity" result is a valid finding |

---

## 16. Suggested edits to the proposal text

- **§1, first paragraph.** Replace "Recent 3D methods, including Gaussian-Splatting-based approaches Kerbl et al. [2023], Lu et al. [2025], localize…" with:
  > "Recent change-detection methods built on 3D Gaussian Splatting (3DGS) [Kerbl et al. 2023] — including 3DGS-CD [Lu et al. 2025], MV3DCD [Galappaththige et al. 2025] and the online O-SCD [Galappaththige et al. 2026a] — localize additions, removals, displacements and surface changes by comparing post-change observations with a 3D representation of the pre-change scene."
- **§1, question.** Replace with:
  > "How does the accuracy of current 3DGS-based change detectors (offline MV3DCD; online O-SCD) degrade under controlled motion blur, exposure shifts and reduced view coverage of the revisit capture; do they report changes when only the observation changed; and do their change scores, or cheap per-frame signals, indicate when predictions are unreliable?"
- **§3, add a bullet:**
  > "No-change control: we pair the two pre-change captures of each PASLCD scene (different lighting, no physical change) and measure false alarms under the same stressors."
- **§3, calibration bullet.** Add:
  > "These methods' scores are not trained as probabilities (thresholds are fixed at 0.5 for lack of a validation set), so we evaluate threshold-free separability (AUPRC), class-conditional confident-error rates, reliability diagrams, and whether a calibration map fitted on clean data remains valid under degradation."
- **§3, poses.** Add:
  > "Offline results are reported with oracle poses and, where feasible, with poses re-registered from degraded images; the online method estimates its own poses."
- **§6, success criteria.** Replace "or" with a specific target:
  > "We will report a failure boundary (with scene-level confidence intervals) for each stressor, a false-alarm curve, a calibration-drift result and an online/offline comparison. The project succeeds if at least one failure boundary is established and a trust rule is validated on held-out scenes."
- **References.**
  - Add O-SCD: Galappaththige et al., "Changes in Real Time: Online Scene Change Detection with Multi-View Fusion", CVPR 2026, arXiv:2511.12370.
  - Mark GS-DIFF as a 2026 preprint (arXiv:2605.07203) that is **discussed, not evaluated**.

---

## 17. Checklists

**Before any sweep**
- [ ] S3 facts recorded (instance sharing, frame counts, reference `.ply`, resolution, format)
- [ ] G1 passed or deviation documented
- [ ] Patches reproduce clean masks bit-for-bit with defaults
- [ ] Identity variant equals clean within the noise floor
- [ ] `PREREGISTRATION.md` committed (grid, hypotheses, thresholds, analysis code SHA)

**Every run**
- [ ] Manifest line → one output folder, `runs.jsonl` line, `frames.parquet` rows
- [ ] Pose / registration failures logged; missing frames scored as empty masks
- [ ] Git SHAs and GPU type recorded

**Before writing**
- [ ] Every claim has a figure or table with scene-level CIs
- [ ] Oracle-pose results labelled as such
- [ ] Exploratory analyses labelled as exploratory
- [ ] Limitations: synthetic perturbations, one dataset, 10 scenes, reference kept clean, GS-DIFF not runnable
