# Understanding the papers: 3D scene change detection with Gaussian Splatting

Companion to [`experiment_plan.md`](experiment_plan.md).

**Sources.** MV3DCD paper (arXiv 2412.03911v2, CVPR 2025) and code (commit `b45606c`); O-SCD paper (arXiv 2511.12370, CVPR 2026) and code (commit `3abfeaa`); GS-DIFF preprint (arXiv 2605.07203). Numbers are quoted from these papers.

Tags used below:
- **(code)** — read in the released code.
- **(prediction)** — our expectation, not a published result.

---

## 0. The one-paragraph version

All of these methods answer one question. You have photos of a place taken **before** and photos taken **after** something changed. Which pixels of each "after" photo show a **real change** (an object added, removed, moved or recolored), as opposed to a lighting difference, shadow or reflection?

The before and after photos come from different, uncontrolled viewpoints, and no labelled change data is used. The shared trick is to build a **3D Gaussian Splatting (3DGS)** model of the "before" scene. You can then render what the scene *used to look like* from exactly the viewpoint of each "after" photo, and compare the two.

- **MV3DCD** (2025) does this **offline**. It fuses evidence across views by storing a learned "change value" inside the 3D Gaussians.
- **O-SCD** (2026) does the same **online**: frame by frame at about 11 FPS, with a better fusion loss.
- **GS-DIFF** (2026 preprint) skips rendering altogether and compares two 3D models Gaussian by Gaussian.

Your project asks what happens to these methods when the "after" photos are bad (blurred, badly exposed, too few), and whether their scores reveal when they are wrong.

---

## 1. Background concepts you need

### 1.1 The task: scene change detection (SCD)
- **Inputs.** Reference (pre-change) images `I_ref` and inference (post-change) images `I_inf`.
- **Output.** A binary change mask for every inference image.
- **Relevant changes.**
  - *Structural*: objects added, removed or moved.
  - *Surface*: color or texture changes, spills, stickers.
- **Distractors (must be ignored).** Illumination changes, shadows, reflections.
- **Pose-agnostic.** The after-photos need not match the before-photos' viewpoints.
- **Label-free.** No change annotations are used for training. Pretrained foundation-model features are allowed, but they never see change labels.
- **Offline vs online.**
  - Offline methods see *all* after-photos before predicting.
  - Online methods must output a mask for each after-photo when it arrives, using only it and earlier frames.
- **Multi-view consistency.** A real change is visible consistently from many viewpoints; a reflection or shadow usually is not. Combining evidence across views suppresses distractors. This is the central idea of MV3DCD and O-SCD.

### 1.2 Camera poses: SfM, registration, PnP, RANSAC
- **Pose.** A camera's rotation and position (plus intrinsics such as focal length). Rendering a comparison view requires the pose of the after-photo *in the coordinate frame of the before-model*.
- **Structure-from-Motion (SfM, COLMAP).** Detect keypoints in all images and match them, then jointly estimate every camera pose and a sparse 3D point cloud. The joint refinement step is *bundle adjustment*.
- **Registration.** Adding new images to an existing SfM model: match their keypoints to the model's 3D points and solve for each new pose.
- **PnP (Perspective-n-Point).** Given 2D keypoints in a new image matched to known 3D points, compute the camera pose.
- **RANSAC.** Solve PnP from many small random subsets and keep the pose that the most matches agree with (the *inliers*). This makes it robust to wrong matches. A small bundle adjustment ("miniBA") then refines the pose on the inliers.
- **Why it matters for you.** Blur and darkness destroy keypoints, which leads to fewer inliers. The result is either a failed pose (no mask at all) or a slightly wrong pose. A wrong pose gives a misaligned render, and misalignment looks like change along every edge.

### 1.3 3D Gaussian Splatting (Kerbl et al., 2023): the substrate
Kerbl et al. is a reconstruction and rendering method, **not** a change detector.

**What a Gaussian stores.** A scene is roughly 10⁵–10⁶ 3D Gaussians. Each Gaussian has:
- a center `μ`;
- a covariance `Σ = R S Sᵀ Rᵀ` (a rotation plus three scales, i.e. an oriented ellipsoid);
- an opacity `α`;
- a color stored as **spherical-harmonic (SH)** coefficients. Degree 0 means one color from every viewpoint; degree 3 means color that varies with viewing direction, which is how shine and reflections are modeled.

**Rendering.** Project every Gaussian to a 2D ellipse, sort by depth, and **alpha-composite** front to back:

`pixel = Σ_i value_i · α_i · Π_{j<i} (1 − α_j)`

"value" is normally color. It can be **any per-Gaussian number**, such as a learned "change" scalar. That is exactly how MV3DCD and O-SCD render change maps: the change information lives in 3D, so every rendered view is automatically consistent with every other.

**Training.**
- Start from the SfM points.
- Render the training views and minimize `(1−λ)·L1 + λ·(1 − SSIM)` with `λ = 0.2`.
- Periodically clone or split Gaussians where gradients are large, and prune almost-transparent ones ("adaptive density control").

**Non-uniqueness.** Many different sets of Gaussians render the same photos. Two reconstructions of the same unchanged room end up with different Gaussians. This is the "drift" GS-DIFF has to model.

**What bad input does to a 3DGS** (general knowledge):
- Blurry training images give smeared geometry.
- Exposure differences between images get baked into colors inconsistently.
- Too few views give "floaters" (spurious Gaussians in empty space) and holes.
- Wrong poses give ghosting.

**In your project** the reference model is built from clean images and stays clean; only the after-photos are degraded. Degradation therefore enters through three routes:
1. pose estimation;
2. the comparison between a degraded photo and a clean render;
3. for MV3DCD only, the retraining of its change-aware 3DGS on the degraded photos.

### 1.4 Comparing a photo with a render
- **L1.** Per-pixel absolute color difference. It reacts to *any* photometric difference (exposure, shadows), so it produces many false positives under lighting change.
- **SSIM (structural similarity).** Compares local mean brightness, contrast and structure in 11×11 windows; 1 means identical. It is partly insensitive to a uniform brightness change but very sensitive to edges. Blur therefore lowers SSIM along every edge in the image.
- **Foundation-model features.** DINOv2 (a self-supervised Vision Transformer) and the image encoder of SAM2 (Segment Anything 2) turn an image into a grid of feature vectors, one per patch (14×14 or 16×16 pixels). Comparing the two grids detects *semantic* differences (is an object there or not?) while ignoring pixel noise. Two known weaknesses:
  1. **Patch granularity.** Masks come out blobby, overestimating change around boundaries.
  2. **Recolored objects barely change the features.** A red cup and a blue cup look almost the same to DINOv2. This causes the surface-change misses both MV3DCD and O-SCD report.
- **Per-frame min-max normalization (code, both methods).** Each difference map is rescaled so its own minimum is 0 and its own maximum is 1. The map is therefore *relative*: even when nothing changed, some spot in every frame scores 1.0. Keep this in mind for the no-change experiment in the plan.

### 1.5 Metrics used by these papers
- **Per image.** TP, FP and FN counted over "change" pixels.
  - `IoU = TP / (TP + FP + FN)`
  - `F1 = 2TP / (2TP + FP + FN)`
- **"mIoU" here** means the IoU of the change class, computed **per image** and then averaged over images and instances (code: `torchmetrics` binary Jaccard per image, then a mean). It is not a mean over classes.
- **Threshold.** Continuous change maps are binarized at **0.5**. MV3DCD explains why: without labels there is no validation set to tune a threshold, so the midpoint is used. Consequently, none of these scores was designed as a calibrated probability, which is the gap your calibration analysis addresses.
- **AUROC** (threshold-free) appears only in MV3DCD's MAD-Real table.

---

## 2. MV3DCD — "Multi-View Pose-Agnostic Change Localization with Zero Labels" (CVPR 2025)

### 2.1 Problem and key idea
Earlier pose-agnostic methods (OmniPoseAD, SplatPose) compared each after-photo with a rendered before-view **one image at a time**. As a result:
- reflections, shadows and unseen regions became false positives;
- their optimization-based pose estimation often failed in large scenes.

**Key idea:** turn per-image change masks into a **3D quantity**. Each Gaussian gets a learned change value, trained to explain the masks of *all* views at once. A value that must agree with every view cannot follow one view's reflection, so view-specific false positives average out. The paper also contributes the **PASLCD** dataset.

### 2.2 The pipeline, step by step
1. **Reference model.** Run COLMAP on the before-photos to get poses, then build `3DGS_ref` (7,000 iterations). (code) The released script trains at 1/8 image resolution.
2. **Poses of the after-photos.** Register the after-photos into the reference SfM model with COLMAP, so both sets share a coordinate frame.
   - The paper states the assumption openly: registration must succeed, and it can fail when the after-scene is, for example, "extremely dark".
   - (code) The released `PASLCD.zip` ships **one COLMAP model containing both sets**, computed from clean images. After-photos have "test" in their file names.
3. **Render comparison views.** Render `3DGS_ref` at every after-photo pose to get `I_ren`.
4. **Two change masks per after-photo.**
   - **Feature-aware.** (code) DINOv2 ViT-L/14; both images are resized to 518×518, giving a 37×37 patch grid.
     1. Compute `D = Σ_channels |f_ren − f_inf|`.
     2. Upsample (bicubic) to image size.
     3. Min-max normalize to [0, 1].
     4. Zero every value ≤ 0.5. The result is `M_F`.
   - **Structure-aware.** `M_S = 1` where `SSIM(I_ren, I_inf) < 0.5`, else 0.
   - **Candidate mask.** `M_{F,S} = M_F · M_S`: both must agree. Values are 0, or in (0.5, 1].
5. **Change-aware 3DGS.** Copy `3DGS_ref` and add two parameters to every Gaussian:
   - a change magnitude `c̃`, stored as **SH degree 0**, so it is the same from every viewpoint;
   - a change opacity `α̃`.

   Train 3,000 more iterations on the after-photos with two losses (code):
   - an RGB loss, so the model becomes the *after* scene;
   - an L1 + D-SSIM loss between the rendered change map and `M_{F,S}`.
6. **Final masks.** Render the change channel at each after-photo pose, or at any other pose. Multiply by the binarized rendered **alpha** (keep pixels with α ≥ 0.5). Regions the reference never saw render as empty background and would otherwise look like change.
7. **Augmentation (reverse comparison).** Render the change-aware after-model from the **before-photo poses**, compute candidate masks against the real before-photos, and fine-tune the change channel for 3,000 more iterations on both sets of masks.
8. **Threshold at 0.5.**

### 2.3 Why each design choice
- **SH degree 0 for change.** Real changes look the same from every angle; view-dependent "change" is mostly reflections, shadows or misalignment. Table 5 (indoor scenes) moves from SH degree 3 to degree 0:

  | SH degree | FP pixels per image | FN pixels per image | mIoU |
  |---|---|---|---|
  | 3 | 1,257 | 416 | 0.442 |
  | 0 | 879 | 534 | 0.474 |

- **Separate change opacity.** Gaussians of a *removed* object become transparent in RGB (the object is absent from the after-photos) and would be pruned. Yet they are exactly where the change is. Pruning only when both opacities are low keeps them.
- **Initialize from the reference.** This retains structures that disappeared and needs fewer after-photos.
- **Product of feature and structure masks** (Table 6, mIoU):

  | Masks used | mIoU |
  |---|---|
  | Feature only | 0.311 |
  | Structure only | 0.324 |
  | Combined | 0.449 |
  | + augmentation | 0.457 |
  | + alpha filtering | 0.461 |

  The two cues fail in different places (features give blobs and miss recolorings; SSIM fires on fine edges and reflections), so requiring both removes false positives. The price: if either cue misses a change, the product misses it too.

### 2.4 Results
- **PASLCD, all 20 instances: 0.461 mIoU / 0.612 F1.**
  - Instance 1 (similar lighting): 0.478 / 0.626.
  - Instance 2 (different lighting): 0.444 / 0.598.
  - Best competitor, CYWS-2D: 0.273 / 0.398.
- **Per scene.** Best is Printing Area (0.588); worst is Playground (0.249), a large 360° outdoor scene with small changes.
- **Reproduction target.** O-SCD's and GS-DIFF's tables list MV3DCD at 0.478 / 0.628. That matches MV3DCD's own Instance-1 number, or possibly a re-run of the released code (its README says results come out slightly better than reported). **Use 0.46–0.48 as your reproduction band.**
- **Lighting robustness (Table 7).** Relative drop from similar to different lighting:

  | Method | ΔmIoU | ΔF1 |
  |---|---|---|
  | MV3DCD | 7.2% | 4.5% |
  | CYWS-2D | 16.1% | 10.0% |
  | Feature Difference | 17.2% | 12.6% |

  This is your real-world anchor for the illumination stressor.
- **Few views (Fig. 4, indoor scenes).** Using 5, 10, 15 or 25 after-photos, mIoU rises with more views. Even 5 views give about 1.8× the per-image baseline. Masks rendered for 10 **never-seen** poses reach 0.36–0.45 mIoU. Your coverage experiment should reproduce this curve first.
- **Runtime.** About **479 s per instance** on an RTX 4090 (as reported in O-SCD's Table 1).
- **Other datasets.**
  - MAD-Real (LEGO objects): 0.132 mIoU, AUROC 0.953.
  - ChangeSim (change class): 0.407.
- **Failures the authors report.**
  1. Color-only changes, which the features don't register.
  2. Tiny changes in big scenes.
  3. Blobby, over-sized masks from upsampling the patch grid.

### 2.5 What it outputs
- A continuous rendered change map (alpha-masked, clipped to [0, 1]) and a binary mask at 0.5.
- (code) Only the binary mask is saved; the plan patches it to save the continuous map too.
- The continuous value is trained to reproduce candidate masks that are either 0 or above 0.5. Expect scores clustered near 0 and above 0.5 — **not a probability**.

### 2.6 How your stressors should hit it (prediction)
- **Motion blur.**
  - COLMAP registration gets harder, but the shipped oracle poses hide this. That is why the plan uses two pose protocols.
  - SSIM drops along every edge, so `M_S` fires on edges. DINOv2 is fairly robust to mild blur, so the product may suppress much of it.
  - Blur also enters the RGB retraining of the change-aware model.
  - MV3DCD runs at 1/8 resolution, so the same physical blur is **half as strong** for it as for O-SCD (1/4 resolution).
- **Exposure shift.**
  - SSIM and DINOv2 are partly invariant to it, and the product helps further.
  - Severe under- or over-exposure creates black or saturated regions where both masks fire.
  - At extreme darkness, registration fails, which is the paper's own stated assumption.
- **Fewer views.** Less multi-view averaging means more false positives survive; Fig. 4 already shows the trend on clean data.

---

## 3. O-SCD — "Changes in Real Time: Online Scene Change Detection with Multi-View Fusion" (CVPR 2026)

### 3.1 Problem and key idea
**Online SCD:** a robot revisits a scene and must flag changes **as each frame arrives**. It has no future frames, no known poses and no labels. Previous online methods were far less accurate than offline ones and usually slow.

O-SCD contributes three things:
1. **Fast, drift-free pose estimation** against the reference.
2. A **self-supervised fusion (SSF) loss** that soft-merges pixel and feature change cues from *all frames seen so far* into a 3D change representation. This replaces MV3DCD's hard thresholds and mask product.
3. A **change-guided map update** after the visit.

The paper assumes the scene does not change *during* a visit.

### 3.2 Offline preparation (before the revisit)
- **Reference 3DGS** from the before-photos. The paper uses Speedy-Splat; the released code integrates FastGS. (code) The online dataset is expected to ship a prebuilt model at `reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply`. Verify this after download.
- **Reference anchors.** For every reference image:
  1. extract XFeat keypoints (code: 6,144 per image);
  2. match exhaustively against all other reference images;
  3. triangulate with the known poses.

  Every reference keypoint that triangulates gets a 3D coordinate.

### 3.3 Per incoming frame (the online loop)
1. **Pose.**
   1. Extract XFeat keypoints.
   2. Retrieve the **4 reference images with the most matches**.
   3. Turn matches into 2D–3D correspondences via those images' anchors.
   4. Solve PnP + RANSAC, then refine with a GPU mini bundle adjustment.

   Details:
   - Only fixed reference frames are used, so the pose cannot drift and each frame costs constant time.
   - (code) If **100 or fewer inliers** remain, the pose is rejected. The released `oscd.py` does not handle that case and **crashes**; the plan patches it (P1).
2. **Render** the reference model from that pose to get `I_ren`.
3. **Change cues** (code):
   - **Pixel cue.** `C_pixel = 0.8·|I_inf − I_ren| + 0.2·(1 − SSIM)`, averaged over color channels and min-max normalized per frame.
   - **Feature cue.**
     1. Encode both images with **SAM2.1-Hiera-tiny**, each resized to 1024×1024, giving 64×64×256 feature grids.
     2. Take the mean absolute difference over channels.
     3. Min-max normalize and upsample.
     4. (code quirk) The map is then re-scaled with the pixel cue's raw min/max and clamped to [0, 1]. The feature cue's strength therefore depends on how large the raw pixel differences were in that frame.
   - **Combined.** `C = C_pixel + C_feature` (range 0–2). There are **no thresholds**, unlike MV3DCD.
4. **Fuse into the change representation `R_change`.**
   - `R_change` is the reference Gaussians with colors discarded and one learnable change value per Gaussian. (code) It starts at 0 everywhere.
   - Rendering `R_change` from a pose gives a change map `M`; `σ(M)` is its sigmoid.
   - Run **16 optimization steps** per frame. Each step picks a frame `i` — about 1/3 of the time the newest frame, otherwise a random frame seen so far — and minimizes:

     `L_SSF = mean( C_i ⊙ (1 − σ(M_i)) ) + log( 1 + mean(σ(M_i))² )`

   How to read the loss:
   - **First term.** Wherever the cue says "different" (`C` high), the loss is large unless the model predicts change there. It pushes change *up* at strong-cue pixels.
   - **Second term.** It penalizes the total predicted change area, which stops the model from predicting "change everywhere". Remove it and training collapses to all-change (ablation).
   - **Why this is multi-view.** One Gaussian projects into many frames. Its change value only rises if the cue is strong at its projection in most of those frames. A reflection that is strong in one frame and absent in the others gets outvoted.
   - **Why it is fast.** `R_change` persists between frames, so each new frame starts from everything learned so far and needs only a few steps.
   - (code) At step 4 of each frame, high-gradient Gaussians are cloned or split so change boundaries can sharpen.
5. **Mask for this frame.** Render `R_change` at this pose. (code) Mask = raw rendered value > 0.5. The loss used `σ(value)`, and σ > 0.5 corresponds to raw > 0, so the decision threshold actually corresponds to σ ≈ 0.62.
6. **After the visit (`--refine`)** (code). Keep optimizing over **all** frames up to 3,000 total steps, with regular densify-and-prune, then re-render every mask. These are the "refined" masks. The paper's **offline** setting similarly optimizes `R_change` with all views jointly for 3k iterations.

### 3.4 Scene update (after detection)
1. Mask each after-photo with its refined change mask.
2. Reconstruct **only the changed regions** as new Gaussians.
3. Delete reference Gaussians that contributed to changed pixels, and merge.
4. Run a short global optimization with densification only around changes. This also corrects global lighting differences and seams.

Result on PASLCD: an updated map in **42 s**, versus **550 s** to rebuild from scratch, at higher PSNR (23.70 vs 22.21) because unchanged regions keep the well-trained reference Gaussians.

### 3.5 Results
- **PASLCD (20 instances).**

  | Setting | mIoU | F1 | Speed |
  |---|---|---|---|
  | O-SCD online | 0.486 | 0.638 | 11.2 FPS |
  | O-SCD offline | 0.552 | 0.694 | 156 s total, incl. pose + reference |
  | MV3DCD (offline) | 0.478 | 0.628 | 479 s |
  | Best previous online (ChangeSim retrieval + CYWS-2D) | 0.243 | 0.360 | — |

- **Runtime per frame (Table 2):** 89.6 ms total.

  | Stage | ms |
  |---|---|
  | Fusion | 58.2 (65%) |
  | Pose | 16.5 |
  | Retrieval | 11.5 |
  | Change cues | 1.7 |
  | Descriptors | 1.3 |
  | Mask | 0.5 |

  Fewer steps per frame give 11–20 FPS at a cost of about 3.6% relative F1 (Fig. 4).
- **Ablation (Table 3, mIoU).**

  | Variant | mIoU |
  |---|---|
  | Full | 0.486 |
  | Without L1 | 0.320 |
  | Without D-SSIM | 0.447 |
  | Pixel cue only | fails to converge |
  | Feature cue only | fails to converge |
  | No regularizer | collapses to all-change |
  | MV3DCD-style thresholds + product instead of SSF | 0.350 |

  **Take-away:** the soft fusion of both cues across views is what makes it work.
- **Hardware.** RTX 4090, images at 1008×560 (code: `--resolution 4`).

### 3.6 Outputs and score semantics
- A continuous, unbounded rendered change value is binarized at 0.5. (code) Only binary masks are saved.
- The GS-DIFF authors note O-SCD outputs are driven toward 0 or 1 (bimodal). Expect **over-confidence**. The scores are not calibrated by design.

### 3.7 How your stressors should hit it (prediction)
- **Motion blur.**
  - XFeat keypoints degrade, giving fewer inliers. The result is a rejected pose (a crash without patch P1, and never a mask for that frame) or a slightly wrong pose.
  - A misaligned render makes L1/SSIM fire along all edges, and per-frame normalization keeps the strongest edges near 1. Expect **structured false positives along edges**.
  - The SAM2 cue is somewhat robust to blur.
- **Exposure shift.**
  - L1 rises everywhere by a roughly uniform amount. Min-max normalization turns a uniform offset into relative contrast, so it is **partly cancelled**.
  - Clipped black or white regions and texture-dependent effects remain.
  - Very dark frames starve XFeat, so the pose fails.
- **Fewer views.**
  - Fewer frames are available to outvote distractors.
  - Early frames' online masks are based on very little evidence; refined masks should recover more.
- **Frame order.** Online masks depend on which frames came first; the refine pass removes most of this order dependence.

---

## 4. GS-DIFF — "From Pixels to Primitives: Scene Change Detection in 3D Gaussian Splatting" (preprint, 2026)

### 4.1 Key idea
Instead of rendering and comparing images, build **two independent 3DGS models** (before and after) and compare them **Gaussian by Gaussian**.

The obstacle is **drift**: even with no change, two reconstructions differ in the number, placement, shape and color of their Gaussians. GS-DIFF models how much drift to expect and only flags differences beyond that.

Claimed benefits:
1. Multi-view consistency comes for free, because a Gaussian is a single 3D object.
2. It can separate **structural** from **surface-only** changes.

### 4.2 Pipeline
0. **Setup.**
   - Get COLMAP poses for both image sets in one frame.
   - Build two geometry-accurate reconstructions with PGSR.
   - Discard Gaussians seen by only one set's cameras.
1. **Representation-ambiguity drift.**
   1. For every Gaussian, find its nearest neighbor in the other model.
   2. Split the displacement into an *along-normal* part and an *along-surface* part.
   3. Take the 75th percentile of each over the scene to get `u_n` and `u_t`.
   4. Inflate every Gaussian's covariance by `U_i = u_t²·I + (u_n² − u_t²)·n nᵀ`.
2. **Observation uncertainty (Fisher information).**
   - Compute `H_i = Σ_cameras (1/d²)(I − v vᵀ)`. A camera pins a point across its viewing ray but not along it, and far cameras pin it less.
   - A point seen from a single direction has a nearly singular `H_i`.
   - Add `s·H_i⁺` to the covariance, with the scale `s` matched by medians. Under-observed Gaussians then tolerate larger displacements before being called "changed".
3. **Neighbor search.** Consider Gaussians of the other model within `3·sqrt(largest eigenvalue of the inflated covariance)`.
4. **Geometric kernel.**
   - `k_geo = exp(−½ Δμᵀ (Σ̃_i + Σ̃_j)⁻¹ Δμ)`, an **unnormalized** Mahalanobis RBF.
   - `δ_geo = 1 − max over neighbors`.
   - Dropping the usual normalization constant matters: 0.096 vs 0.034 mIoU in the ablation.
5. **Appearance kernel** on the SH degree-0 (diffuse) color.
   - Bandwidth `σ_c` is the weighted median color difference between well-matched Gaussians, widened for large Gaussians. This absorbs global lighting and exposure offsets **by design**.
   - `δ_app = 1 − max over neighbors of exp(−‖c_i − c_j‖² / 2σ²)`.
6. **Confidence weight.** `ω_i = sigmoid(log tr H_i − log Q25(tr H))`. The least-observed quarter of Gaussians gets weight below 0.5.
7. **Score and render.**
   - `δ_i = ω_i · min(δ_geo + δ_app, 1)`.
   - Render `δ` from both models at the query view via alpha compositing and take the pixel-wise max.
   - Threshold at 0.5.
8. **Change type.** A change is structural if `δ_geo` dominates, and surface-only if the residual `max(δ_app − δ_geo, 0)` dominates.

### 4.3 Results
- **PASLCD: 0.644 mIoU / 0.758 F1**; with a per-scene best ("oracle") threshold, 0.669 / 0.779. This is +17% over O-SCD offline (0.552 / 0.694).
- **Ablation ladder** (fixed-threshold mIoU). Drift modelling is what makes Gaussian-to-Gaussian comparison work:

  | Step | mIoU |
  |---|---|
  | Naive nearest neighbor | 0.110 |
  | Kernels | 0.102 |
  | + ambiguity inflation | 0.258 |
  | + Fisher observability | 0.465 |
  | + data-driven color bandwidth | 0.537 |
  | + confidence weighting | 0.644 |

- **Fixed-vs-oracle threshold gap** of only 0.025 mIoU. The authors present this as "well-calibrated" scores; it is the same idea as the threshold-gap metric in your plan.
- **Structural vs surface classification.** Balanced accuracy 0.868, using their own per-pixel type annotations (unreleased).
- **Quantile choices** (75th / 50th / 25th percentiles) sit on stable plateaus.

### 4.4 Limitations and status
- Accuracy is bounded by reconstruction quality and COLMAP pose accuracy. The color bandwidth could hide subtle color changes.
- **Offline only**: it needs a full after-reconstruction.
- **Code and annotations "will be released upon acceptance".** The repository currently contains only a README, and a GitHub issue reports trouble reproducing (0.2–0.3 mIoU). **It is not runnable for your project.**

### 4.5 Why it still matters for you
- The **observability term** is an explicit, principled uncertainty signal: how well each 3D point is constrained by the cameras. That is exactly the kind of "should I trust this?" quantity your calibration question is about, and missing coverage lowers it directly. Use it in your discussion.
- The plan includes a cheap camera-coverage proxy you can compute for any method.
- (prediction) Under your stressors:
  - Blur degrades the after-reconstruction's geometry, which inflates the drift estimates and lowers sensitivity.
  - Global exposure shifts are absorbed by `σ_c`.
  - Missing views lower observability, which down-weights scores: fewer false alarms, more misses.

---

## 5. 3DGS-CD (Lu, Ye, Leonard, IEEE RA-L 2025) — brief
- **What it does** (per its abstract). Detects **physically rearranged objects** between a pre-change 3DGS and a *sparse* set of post-change images (as few as one). It uses a promptable segmentation model (EfficientSAM) to get object-level 2D change masks, associates them across views into 3D, and estimates each moved object's transformation.
- **What it is built for.** Object rearrangement in clutter, not scene-level surface changes.
- **On PASLCD** (run by the O-SCD authors): 0.209 mIoU / 0.339 F1, 824 s.
- **Priority for you:** low. Cite it as related work.

---

## 6. The PASLCD dataset
- **10 scenes.**
  - Indoor: Cantina (FF), Lounge (FF), Printing Area (FF), Lunch Room (360°), Meeting Room (360°).
  - Outdoor: Garden (FF), Pots (FF), Zen (FF), Playground (360°), Porch (360°).
  - FF means a front-facing capture; 360° means the camera circled the area.
- **Instances.** Each scene has **one post-change capture** and **two pre-change captures**:
  - **Instance 1**: lighting similar to the post-change capture.
  - **Instance 2**: different lighting.

  The post-change set is annotated relative to each, giving 2 instances per scene and 20 in total. The supplement's wording implies both instances **share the same post-change images**; verify by hashing files after download (plan, step S3).
- **Size.** 25 post-change images per scene (MV3DCD Sec. 5.3), giving 50 annotated masks per scene and 500 in total. The online zip may be organized differently, so count its frames after download.
- **Changes.** 91 in total, 5–17 per scene.
  - Structural, 70%: added 24%, removed 27%, moved 18%.
  - Surface, 30%: spills and stickers 19%, color swaps 12%.
  - Includes transparent glass, thin cutlery and benches.
- **Changed pixels per image.** 0.17%–20.12%, mean **3.51%**. This is heavy class imbalance, which matters for calibration metrics. Every annotated image contains some change.
- **Capture.** iPhone at 16:9, handheld random trajectories at random heights and orientations.
- **Two downloads** (Hugging Face `ChamudithaJay/PASLCD`). Note that the folder order is flipped between them.

  | Download | Used by | Layout |
  |---|---|---|
  | `PASLCD.zip` (offline) | MV3DCD | `Scene/Instance_X/` containing `images/` (post-change images have "test" in the name), `sparse/0` (joint COLMAP) and `gt_mask/` |
  | `PASLCD_online.zip` | O-SCD | `Instance_X/Scene/` containing `reference_scene/{images,sparse/0}`, `inference_scene/images`, `reference_reconstruction/…ply` and `gt_mask/` |

---

## 7. Side-by-side

| | MV3DCD | O-SCD | GS-DIFF | 3DGS-CD |
|---|---|---|---|---|
| Setting | Offline | **Online** (+ optional offline refine) | Offline | Offline |
| After-photo poses | COLMAP registration (shipped as clean oracle poses) | Self-localizes: XFeat + PnP vs reference | COLMAP, both sets | Its own pipeline |
| Change evidence | DINOv2 ViT-L/14 × SSIM, hard thresholds | L1/SSIM + SAM2 features, soft | Gaussian geometry + diffuse color | Segmentation-model masks |
| Multi-view fusion | Change channel trained on masks | SSF loss over all frames so far | By construction (3D comparison) | Cross-view association |
| Output | Continuous map → 0.5 | Continuous (raw) → 0.5 | Continuous → 0.5 | Masks |
| Code | ✅ | ✅ | ❌ (unreleased) | ✅ (per paper) |
| PASLCD mIoU / F1 | 0.461 / 0.612 (0.478 Inst. 1) | 0.486 / 0.638 online; 0.552 / 0.694 offline | 0.644 / 0.758 | 0.209 / 0.339 |
| Runtime | ~479 s | 11 FPS; 156 s offline | n/a | 824 s |
| Weak spot under your stressors (prediction) | Registration under blur/dark; edge FPs from SSIM | Pose rejection; edge FPs; early-frame online masks | Reconstruction quality | — |

---

## 8. Glossary
- **Alpha compositing.** Blending sorted semi-transparent layers front to back; how 3DGS turns Gaussians into pixels.
- **AUROC / AUPRC.** Threshold-free ranking quality. AUPRC is more informative when positives are rare, as here (3.5% of pixels).
- **Brier score.** Mean squared error between a probability and the 0/1 label; lower is better; rewards calibration and sharpness together.
- **Calibration.** Among pixels scored 0.8, about 80% should truly be change. ECE measures the average gap.
- **COLMAP.** Standard SfM software.
- **D-SSIM.** `1 − SSIM`, a dissimilarity.
- **Densification.** 3DGS cloning or splitting Gaussians to add detail.
- **FF / 360°.** Front-facing capture vs circling capture.
- **Instance (PASLCD).** One pairing of a reference capture with the post-change capture.
- **Inlier.** A match consistent with the estimated pose (RANSAC).
- **Observability.** How well cameras constrain a point's 3D position (GS-DIFF's Fisher-information term).
- **PnP.** Pose from 2D–3D correspondences.
- **Pose-agnostic.** Works from arbitrary viewpoints.
- **SH (spherical harmonics).** How 3DGS stores view-dependent color; degree 0 means view-independent.
- **SSF loss.** O-SCD's self-supervised fusion loss.
- **XFeat.** A fast lightweight keypoint detector and descriptor.
