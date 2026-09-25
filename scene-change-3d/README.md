# scene-change-3d

Prototype for 3D scene change detection between property walkthroughs. A first
walkthrough (RGB-D video with device poses) becomes a baseline 3D Gaussian map; each
later walkthrough is localized inside it and compared, and the result is a list of
missing, added, moved and restyled items, the areas that were not checked this visit,
and an HTML report with a 3D viewer.

Everything runs on CPU with NumPy, SciPy and OpenCV. A synthetic harness generates
apartments, walkthroughs and ground truth, so every stage can be measured.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# render a scenario: baseline walkthrough + a later visit with changes
python -m scene_change generate --seed 3 --kind mixed --out runs/s003_mixed

# build the baseline, inspect, score against ground truth, write report.html
python -m scene_change inspect runs/s003_mixed --out runs/s003_mixed/result

# compare methods over many scenarios (ours, 2D video comparison, O-SCD)
python -m scene_change.harness.benchmark --out runs/bench --seeds 0 1 2 --kinds mixed clean partial

pytest              # fast tests; `pytest -m slow` adds an end-to-end harness run
```

Scenario kinds: `mixed` (8 changes of all types), `clean` (lighting, viewpoint, drift
and sensor noise differ, nothing else) and `partial` (a skipped room and a room only
half panned, to test unverified-area reporting).

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| Baseline | `gaussians.py`, `model.py` | Fuses RGB-D keyframes into one surfel Gaussian per 5 cm voxel (exposure-compensated colour, PCA normals), keeps pooled keyframe depth for free-space tests. Exports standard 3DGS `.ply`. |
| Rendering | `render.py` | EWA splatting of the Gaussians (NumPy); can also return the sparse per-pixel blending weights. |
| Localization | `register.py` | Gravity-aligned 4-DoF floor-plan correlation (FFT over yaw), robust point-to-plane ICP, then drift correction on overlapping chunks, kept only where it improves the local fit. |
| Detection | `detect.py` | Missing: baseline surfaces seen through (free-space violation). Added: new surfaces where the baseline saw free space. Moved: missing/added pairs matched by shape and colour. Restyled: colour change after a per-frame colour fit, with texture-aware thresholds and cross-view agreement. Anything never observed again is reported as unverified, per room. |
| Report | `report.py` | Self-contained HTML: verdict, change list with baseline/visit image pairs, room coverage, unchecked areas, 3D viewer. |
| 2D baseline | `baseline2d.py` | Frame retrieval, homography alignment, brightness matching and differencing. |
| O-SCD | `oscd.py` | Re-implementation of the O-SCD paper (below). |

Conventions: z-up world, OpenCV cameras, `T_wc` camera-to-world poses, z-depth in metres.

## Harness

`scene_change/harness/` generates five-room apartments (living room, kitchen,
hallway, bedroom, bathroom) from about 40 object builders, applies changes (removed,
added, moved, restyled items, stains and scuffs), plans walkthroughs through
doorways, and ray-casts RGB-D frames with lighting variation, exposure jitter, sensor
noise, depth dropout, flying pixels and VIO drift. Ground truth includes true poses,
per-pixel object ids in both scene states and the list of changes.

Scoring (`harness/evaluate.py`):

- **frame IoU / F1**: change-class IoU and F1 per inspection frame, averaged over frames
  that show a change. This is the PASLCD protocol O-SCD reports, so numbers are
  comparable in kind (not in value: different data).
- **false-alarm rate**: share of pixels flagged on frames with no change.
- **object recall / false positives per inspection**, **unverified recall**,
  **registration error** and timings for the 3D method.

## O-SCD re-implementation

`scene_change/oscd.py` re-implements *Towards Online 3D Scene Change Detection*
(Galappaththige et al., CVPR 2026, [arXiv:2511.12370](https://arxiv.org/abs/2511.12370)),
following the paper and its [official code](https://github.com/Chumsy0725/O-SCD):

1. **Pose**: RootSIFT keypoints matched (mutual nearest neighbours, cosine > 0.82)
   against reference keyframes, the 4 best keyframes by match count, fundamental-matrix
   outlier removal, PnP-RANSAC and a pose-only Levenberg-Marquardt refinement.
2. **Change cues**: C = C_pixel + C_feature, with C_pixel = 0.8·L1 + 0.2·(1 − SSIM)
   (11×11 Gaussian window, min-max normalised) and C_feature the channel-mean difference
   of **SAM 2.1 hiera-tiny** image embeddings (256×64×64), including the official code's
   renormalisation of C_feature by the pixel cue's range.
3. **Change field**: the reference Gaussians with colours reset to zero carry a
   learnable change colour; per frame, 16 Adam steps (lr 0.0025, eps 1e-15) of
   L = mean(C·(1 − σ(M))) + log(1 + mean(σ(M))²), each on the newest frame (p = 0.33) or
   a random earlier one; mask = M > 0.5; optional refinement to 3000 iterations.
4. **Scene update** (`update_scene`): prune Gaussians blended into eroded change pixels
   in more than one pixel-view, then rebuild the dilated change regions.

Adaptations, all documented in the module docstring:

- The change field keeps the reference geometry frozen. Rendering it is then linear in
  the change colours (M = W·c with the fixed blending weights W from `render.py`), so
  the colour gradient is exact without an autograd rasteriser. The official code also
  adapts positions, scales, rotations and opacities, and densifies at every frame's 5th
  iteration.
- SIFT replaces XFeat. Reference keypoints get 3D positions from reference depth
  instead of triangulation. A bag-of-words shortlist precedes exhaustive match counting.
- Frames PnP cannot localise fall back on odometry chained from the last localised frame.
- The scene update rebuilds changed regions from RGB-D (voxel fusion) instead of 5000
  photometric iterations, and does not prune surfaces behind the observed depth.

SAM 2.1 runs through ONNX Runtime. Put the encoder at
`~/models/sam2.1_hiera_tiny.encoder.onnx` or point `OSCD_SAM2_ONNX` at it. One source
is the [X-AnyLabeling release](https://github.com/CVHub520/X-AnyLabeling/releases/download/v2.5.0/sam2.1_hiera_tiny.encoder.onnx);
for production, export it yourself from Meta's checkpoint. Without the file,
`backbone="auto"` falls back to a weight-free dense-SIFT descriptor.

To run the **official** code on harness data (GPU machine),
`harness/oscd_export.py:export_oscd_dataset` writes a scenario in its layout (COLMAP
text model, reference 3DGS `.ply` padded to SH degree 3, ground-truth masks), and
`load_official_masks` reads its output back for scoring with the same metrics.

Licensing: the official O-SCD repository has no license file and several of its files
carry Inria's non-commercial research header; its rasterizer is under the
Gaussian-Splatting License. `oscd.py` is written from the paper and uses none of that
code, but check before shipping anything derived from the method commercially.

## Robustness study

How robust are 3D change detectors to degraded captures, and do their change scores
say when they are wrong? The study runs every method on degraded copies of the same
inspection walkthroughs (the ground truth stays valid by construction) and measures
accuracy, errors and calibration against severity.

```bash
# harness sweeps (CPU): held-out scenes x stressor x severity x method
python -m scene_change.harness.robustness --bench runs/bench --out runs/robust --methods ours video2d oscd
# severity curves, failure boundaries, calibration, offline vs online -> ROBUSTNESS.md
python -m scene_change.harness.robustness_report --runs runs/robust --out ROBUSTNESS.md --figures docs/robustness
```

| Module | What it does |
|---|---|
| `stress.py` | Stressors with severity levels, each applied to the inspection walkthrough: motion blur along the apparent camera motion and exposure shifts in linear light (work on any images); relighting by re-rendering under lighting moved from the baseline visit's toward and past the inspection's (harness); depth noise and dropout (only methods that read depth are affected); coverage loss as contiguous or uniform view removal. Map compression instead coarsens the baseline map. |
| `calibration.py` | ECE (equal-width and equal-mass), reliability diagrams, Brier, AP, AUROC, threshold sweeps, share of confidently wrong pixels, isotonic recalibration; computed from poolable score histograms. |
| `harness/robustness.py` | Resumable sweeps; records PASLCD frame scores, false-positive and miss breakdowns per change, per-frame counts and calibration for each method variant (ours, ours confirmed-only, 2D, O-SCD online, O-SCD refined). |
| `harness/robustness_report.py` | Curves with 95% intervals over scenes, failure boundaries (first level with a significant 20% F1 drop), calibration transfer (isotonic maps fitted on clean scenes, applied to degraded ones), offline vs online, unobserved changes, compression by change size. |
| `harness/paslcd.py`, `gpu/` | The same stressors and metrics on PASLCD with the official O-SCD and MV3DCD code (see `gpu/README.md`). |

## Results

See `RESULTS.md` for the benchmark and `ROBUSTNESS.md` for the robustness study.

## Layout

```
scene_change/
  geometry.py session.py gaussians.py render.py model.py
  register.py detect.py pipeline.py report.py baseline2d.py oscd.py __main__.py
  stress.py calibration.py
  harness/ scene.py raycast.py changes.py walkthrough.py scenario.py evaluate.py
           benchmark.py oscd_export.py robustness.py robustness_report.py paslcd.py
gpu/        official O-SCD and MV3DCD on PASLCD (setup, jobs, SLURM array, report)
tests/
```
