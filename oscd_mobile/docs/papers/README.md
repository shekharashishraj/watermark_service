# Related work: GS-Pool and GOLDILOCS

Both PDFs are in this folder. These are notes for the project report: what each method does, its numbers, and how it relates to our on-phone O-SCD (Method A) and the planned Method B.

## GOLDILOCS (ICLR 2026)

Friedlander, Shamir, Fried (Reichman University). `GOLDILOCS_ICLR2026.pdf` · [project page](https://goldilocs.github.io/) · code "to be released" ([GitHub](https://github.com/almogfri/GOLDILOCS))

**Task:** zero-shot, pose-free, object-level change detection from **one image pair**. It extends to image sets, including a single after-image. No training, no calibration.

**Pipeline (§4):**
1. **MASt3R stereo reconstruction:** pointmaps P0, P1 in a shared frame plus camera intrinsics and extrinsics (§4.1).
2. **Depth-conflict filtering (§4.2):** a point of Pi that projects *in front of* the surface seen in view j (z_i→j < D_j) is an occluder or a changed object and is removed. The union of the cleaned pointmaps, P*, is the "farthest visible surfaces": a canonical static scene.
3. **Rendering pointmaps as images:** R_i,j renders P_i from camera j, and R_*,j renders the clean scene. These are reference images with the changed content removed.
4. **SAM2 masks and tracking (§4.3):**
   - Segment R_0,1 and track the masks into R_*,1. Masks that fail to track are "changed".
   - Track the changed masks into I1: masks that track are *moved*, the rest are *removed*.
   - The symmetric pass from I1 gives *added*.
   - A visibility filter discards masks that mostly overlap occluded or out-of-view gaps (threshold α per dataset).
5. **Non-rigid detection (§4.4):** a dense SSIM map between R_0,1 and I1. A static mask whose mean dissimilarity is more than 1 SD above the mean over all masks is **warped**. The paper's own example is "a tablecloth that appears smooth in T0 but wrinkled in T1".
6. **Per-pixel label** by priority: Warped > Moved > Removed > Added (§4.5).
7. **Multi-view (§4.6):** each after-image is paired with its top-1 before-image by MASt3R match count, then labels are made consistent by cross-view majority voting.

**Taxonomy:** removed, added, moved (full or partial), warped. Occluded or out-of-frame regions are deliberately left unclassified.

**Results:**

| Dataset | Metric | GOLDILOCS | Best baseline |
|---|---|---|---|
| 3DGS-CD (5 tabletop/indoor scenes) | F1 / IoU | **97.72 / 95.30** with 0 auxiliary after-views | 3DGS-CD 97.51 / 95.16 with 4 auxiliary |
| NeRFCD | F1 / IoU | **91.97 / 87.62** | Gaussian Difference 91.90 / 85.74 (10+ views per time) |
| ChangeSim | binary mIoU / multi-class mIoU | **64.9 / 33.6** | ZSSCD 57.2 / 26.7 |
| RC-3D | mAP | **0.531** RGB-only | CYWS-3D 0.50 with RGB-D |
| VL-CMU-CD | F1 | 61.8 % (best zero-shot) | supervised in-domain methods are higher |

**Runtime (NeRFCD Potting scene, Table 5):**

| Method | 3D reconstruction | Change detection | Rendering | Total |
|---|---|---|---|---|
| GOLDILOCS | 4.7 min | 5.2 min | 0.6 min | **10.6 min** |
| Gaussian Difference | | | | 148.2 min |
| C-NeRF | | | | 1056 min |

The paper says inference takes under 5 min per image on 3DGS-CD. It does not state the hardware.

**Ablations:**
- Without 3D reconstruction, ChangeSim binary mIoU drops from 65.0 to 46.3.
- Performance falls as the paired before-image gets less similar, and reconstruction fails below the 20th percentile of matches.
- About 90 % of mIoU is kept with only 5 % of the reconstruction steps.

**Stated limitations (§7):**
- SAM2 can mistake shadows, reflections, texture patterns or lens artefacts for objects.
- The segmentation parameters need tuning per scene scale.
- Lighting changes affect SSIM and the reconstruction.
- The SSIM threshold is statistical.
- Nested changes are suppressed by the label priority, e.g. a key removed from a moved keyboard.

## GS-Pool (arXiv 2610.06688, 5 Oct 2026)

Keren-Gil, Gain, Marais (University of Cape Town). `GS-Pool_arXiv2610.06688.pdf` · per-scene outputs on Hugging Face (`boazkg/gs-pool-paslcd`, `boazkg/gs-pool-cl-splats`)

**Task:** object-level change detection between **two separately trained 3DGS reconstructions**, one per visit. The visits must share one camera frame:
- *joint:* COLMAP over both visits;
- *anchored:* the after-visit is registered into the before-visit's frame.

**Method:**
1. **Object pool.** SAM2 (hiera-large, automatic masks) is run on keyframes and lifted onto Gaussians. A mask claims a Gaussian if the Gaussian is on the rendered surface and at least half its rendering weight lies inside the mask. Tracks with IoU ≥ 0.5 are merged. Objects holding more than half of all Gaussians are treated as **background**.
2. **Four carriers per Gaussian:**
   - geometry (from GS-Diff);
   - colour (from GS-Diff);
   - semantics: DINOv3 ViT-L/16, PCA to 32 dimensions, distilled onto the Gaussians;
   - **photographic:** each model is rendered at the other visit's cameras, and the L1 + DSSIM loss is back-propagated onto the Gaussians.
3. **Per-scene threshold.** An empirical null from objects present in both visits; per-carrier rank tests are combined with Fisher's method and thresholded with a fitted χ².
4. **Decision rule.** Quorum and photographic clauses decide per object. The extent of a changed object is its Gaussians above the null's 95th percentile on at least one carrier.

**Results:**

| Benchmark | GS-Pool | GS-Diff | **O-SCD** | PlenoCI | MV-3DCD |
|---|---|---|---|---|---|
| PASLCD mIoU / F1 | **0.751 / 0.846** (FastPGSR, joint) | 0.644 / 0.758 | **0.552 / 0.694** | 0.538 / 0.688 | 0.478 / 0.628 |
| CL-Splats mIoU (GS-Pool's own masks) | **0.865** | – | 0.622 (4 scenes) | – | 0.641 |
| False alarms on unchanged scenes (% px, mean) | 1.03 | – | 1.47 | – | 1.72 |

**Ablation (mIoU):**

| Configuration | mIoU |
|---|---|
| Photographic carrier alone | 0.468 |
| + semantics | 0.590 |
| Four carriers averaged | 0.609 |
| Core rule | 0.703 |
| Full rule | 0.751 |

The photographic carrier matters most.

**Runtime:** RTX 4090, 1.7–5.7 min per scene, excluding reconstruction, DINOv3 distillation and SAM2.

**Stated limitations:**
- Bounded by capture and reconstruction quality.
- A change SAM2 doesn't segment can't be found.
- Lighting-only false alarms remain.
- Costlier than comparing the Gaussians alone.
- Non-rigid changes are not modelled.

## How they relate to our app

| | GS-Pool | GOLDILOCS | Ours (Method A on iPhone) |
|---|---|---|---|
| Input | Two full 3DGS reconstructions | Image pair or sets | One before 3DGS model plus a stream of after-frames |
| Camera positions | Shared frame required | MASt3R, pose-free | XFeat + PnP per frame against the before-scan (exact O-SCD) |
| When it runs | Offline, minutes on an RTX 4090 | Offline, ≈ 5–10 min | **Online, 0.67 s per frame on an iPhone 17 Pro** (Detect, 2026-10-16) |
| Output | Changed objects (Gaussians) | Labelled objects: add / remove / move / warp | Per-pixel masks from a fused change model |
| Non-rigid | Not handled; large objects treated as background | **"Warped" via SSIM** | Fails: the camera can't be placed, and the cue fires (re-laid sheet, 2026-10-16) |
| PASLCD mIoU | 0.751 | – | inherits O-SCD's 0.552 |

**What we can claim:** the on-device, online part. Neither paper runs online or on a phone; both are offline desktop pipelines taking minutes. Method A copies O-SCD, so it inherits O-SCD's accuracy, which GS-Pool beats by about 20 mIoU points on PASLCD.

**Ideas for Method B:**
1. **Per-scene threshold (GS-Pool):** learn "normal difference" from regions unchanged in both scans, instead of the fixed 0.5 cutoff.
2. **Object-level decisions (GS-Pool, GOLDILOCS):** group the change mask into SAM2 objects. On the phone, prompt SAM2 inside the change regions instead of running the full automatic mask generator.
3. **"Warped" from LiDAR depth (our angle):** both papers infer geometry from images (GS-Pool from reconstructions, GOLDILOCS from MASt3R). The phone has metric depth. A re-laid sheet changes depth by millimetres over a large area; a moved object changes it by centimetres over a small area.
4. **Depth-conflict filtering for pose (GOLDILOCS §4.2):** keep only before/after geometry that agrees, so changed or deformed regions drop out instead of corrupting the camera estimate.

**Evaluation gap:** both papers report public benchmarks. GOLDILOCS reports on 3DGS-CD, a tabletop rearrangement dataset close to our setting. To compare, our pipeline needs at least one benchmark number. The 3DGS-CD dataset is the natural first one.
