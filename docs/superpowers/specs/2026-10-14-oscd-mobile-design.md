# O-SCD Mobile: live on-device scene change detection demo

- **Date:** 2026-10-14
- **Status:** design approved in conversation; written spec awaiting review
- **Owner:** shekharashishraj (ASU applied project)
- **Related:** `scd_robustness/docs/papers_explained.md` §3 (O-SCD notes), `scd_robustness/patches/oscd.patch` (P1: skip a frame whose pose is rejected)

---

## 1. Purpose

### 1.1 What was asked
- A **live demo in front of course instructors**. A tabletop with blocks is scanned, the blocks are changed, and the table is rescanned. The phone highlights what changed as you sweep it, the way O-SCD / GS-CD-style methods do.
- **Everything runs on the phone**: capture, inference and interface. The device is an iPhone Pro with LiDAR.
- The **reference ("before") model is built live on the phone** during the demo.
- Changes to show: **added, removed, moved and recolored** blocks.
- **Two methods, one button:**
  - **A**: an **exact replica of O-SCD**, including its XFeat + PnP pose estimation, its SAM2 feature cue and its SSF fusion.
  - **B**: O-SCD **plus a LiDAR depth cue and a color cue**.
- **Timeline:** 2–3 weeks. The plan below is about 21 working days, with agreed cuts (§2.2).
- Building the reference on a Mac is acceptable only as a **last-resort fallback**.

### 1.2 Assumptions
- One prepared scene: a table region of up to about 1 × 1 m, indoor lighting, blocks at least 2–3 cm in size.
- A **patterned mat** under the blocks gives XFeat and ARKit enough texture.
- The scene does not change *during* a sweep (O-SCD's assumption). Hands passing through the view are transient, and multi-view fusion outvotes them.
- Performance targets assume an A17 Pro-class iPhone (15 Pro). The demo device's model and its measured numbers go into the spike report (§11).

### 1.3 Success criteria
1. **Reference build:** from the end of capture to "ready" (3DGS + anchors) takes 2 minutes or less on the phone.
2. **Detection:**
   - **Method A**, live: a removed or moved block is highlighted within about 5 s of sweeping past it.
   - **Method B**, after replay and refine: each of an added, removed, moved and recolored block is highlighted.
   - Method A is *expected* to be weaker on recolored and added blocks. That gap is part of the demo story.
3. **False positives:** after refine, no unchanged block or table area is persistently flagged, apart from thin edge halos (about 1 cm or less).
4. **Live performance:**
   - The overlay follows the camera at ≥ 30 FPS.
   - The mask contents update at ≥ 3 Hz.
   - At least 90% of processed frames pass O-SCD's pose check (more than 50 inlier points after mini-BA) on the prepared table.
5. **Demo length:** the full run (scan → build → change → rescan → refine → replay with B) takes 6 minutes or less, and has been rehearsed 3 times without restarting the app.
6. **Parity:** pose, cues and fusion match the real O-SCD on golden fixtures within the tolerances in §12.1.

---

## 2. Scope

### 2.1 In scope (v1)
- iOS app: SwiftUI + ARKit + Metal + Core ML, iOS 17+.
- Reference scan, on-device 3DGS reference build and O-SCD reference anchors.
- Method A (O-SCD exact) running live, followed by refine.
- Method B as a **replay** of the stored sweep (same frames, same poses), plus an A/B overlay toggle.
- Debug HUD and split view (live / rendered reference / cue / mask) for explaining the method.
- Session recorder, a macOS harness (replay and parity tests), and the Mac fallback reference server.

### 2.2 Stretch goals (in priority order; agreed cuts from v1)
1. **S1:** run A and B live in parallel with an instant toggle, if the spike shows enough GPU headroom.
2. **S2:** LiDAR-guided insertion of Gaussians for added objects in Method B (§8.5).
3. **S3:** a 3D orbit view of the change model.

### 2.3 Out of scope
- O-SCD's post-visit map update (its paper's §3.4).
- Android, multi-room scenes, and capturing the VeriPhEye homes.
- Benchmarking accuracy beyond the parity tests and a small hand-labeled IoU check.

---

## 3. Background: O-SCD in one paragraph
A reference 3DGS model is built from "before" images with known poses. Reference anchors are XFeat keypoints that are triangulated across reference images.

Each incoming "after" frame is processed online:
1. **Pose:** XFeat keypoints are matched to the 4 best-matching reference keyframes, giving 2D–3D correspondences. A RANSAC over 4-point LM pose solves and a mini bundle adjustment estimate the pose. The frame is rejected if 50 or fewer inlier points remain.
2. **Render** the reference from that pose.
3. **Change cue:** `C = C_pixel + C_feature`. The pixel cue is L1 + SSIM; the feature cue is a SAM2 feature difference.
4. **Fusion:** a copy of the reference Gaussians, `R_change`, has its colors reset and re-learns color, opacity, scale and rotation (positions frozen). It is trained for 16 steps per frame with the self-supervised fusion (SSF) loss, over the newest and past frames. Its rendered intensity is the change score.
5. **Mask:** `R_change` is rendered at the frame's pose and thresholded.

After the visit, a refine loop continues optimizing over all frames. Full notes are in `scd_robustness/docs/papers_explained.md` §3. Where those notes and the code disagree, §6 (read from the code) is authoritative.

---

## 4. System overview

### 4.1 Demo flow (app states)
```
Idle → ReferenceScan → ReferenceBuild → Ready → ChangeScan(A, live) → Refine(A)
     → Review ──[button "Replay with depth + color"]──► ReplayB → Refine(B) → Review(A|B toggle)
```
- **ReferenceScan (about 30 s):**
  - The user taps the table to place the region-of-interest (ROI) box.
  - Sweeping the phone captures keyframes with ARKit poses and LiDAR depth.
  - A coverage indicator fills in as viewing directions are covered.
- **ReferenceBuild (≤ 2 min):**
  - LiDAR-initialized 3DGS training (§7).
  - O-SCD reference anchors are built in parallel (§6.1).
  - The screen shows progress and a live render preview.
- **ChangeScan:** the user changes the blocks, presses **Detect** and sweeps. Method A runs live (§6). Every processed frame is written to the SweepStore (§9.3).
- **Refine:** O-SCD's refine loop runs over all stored frames (§6.5).
- **ReplayB:** Method B's fusion re-runs over the same stored frames and poses with the extra cues (§8), then refines. The **A | B** toggle switches which change model is overlaid.

### 4.2 Coordinate frames
- **World frame `W`:** the ARKit world frame of the session in which the reference was captured. It is metric and gravity-aligned. The reference 3DGS, anchors, `R_change` and all PnP poses live in `W`.
- **Camera convention:** OpenCV (x right, y down, z forward). ARKit camera-to-world poses are converted with `T_cv = T_arkit · diag(1, −1, −1, 1)`.
- **Intrinsics:**
  - Taken from `ARFrame.camera.intrinsics` for the 1920×1440 capture and scaled to each working resolution. Method A uses one focal for all images, as O-SCD does (§6.6).
  - Autofocus is disabled during both scans so the intrinsics stay stable.
- **The method's pose comes only from XFeat + PnP.** ARKit is used only for:
  1. reference keyframe poses (standing in for COLMAP);
  2. LiDAR depth and intrinsics;
  3. keeping the display overlay smooth between pose updates (§9.2);
  4. a HUD sanity check of PnP vs ARKit pose.

  Because the anchors fix the world frame, detection still works after an ARKit session restart. No `ARWorldMap` relocalization is needed.

### 4.3 Resolutions
| Data | Resolution |
|---|---|
| Camera capture | 1920×1440 (ARKit, YCbCr → RGB) |
| LiDAR depth and confidence | 256×192 |
| Working resolution: reference training, rendering, cues, fusion | **480×360** (1/4, O-SCD's downscale factor). The fallback is 240×180 if the spike requires it (deviation D9). |
| XFeat input | working resolution floored to a multiple of 32 (§6.1) |
| SAM2 input | 1024×1024, as in O-SCD |

---

## 5. Components

All of the core lives in a Swift package, **`OSCDCore`**, shared by the iOS app and the macOS tools. Metal kernels live in the package.

| Unit | Responsibility | Interface (in → out) |
|---|---|---|
| `CaptureSession` | Owns the `ARSession` (world tracking + `sceneDepth`). Locks exposure, white balance and focus. Selects keyframes. | AR frames → `Keyframe {rgb, depth, confidence, K, T_wc, timestamp}` and `LiveFrame` |
| `RegionOfInterest` | A gravity-aligned box on the detected table plane, placed by a tap and resizable by pinch. | plane + tap → `AABB` |
| `SplatEngine` | Adapter over the chosen Metal 3DGS engine (msplat-ios or splatx-metal, picked in the spike). Provides training with a custom-loss hook and per-parameter learning rates (including frozen xyz), rendering of RGB / expected depth / alpha, densify, clone/split, prune and opacity reset. | model + camera → textures; `step(lossKernel)` |
| `ReferenceBuilder` | LiDAR initialization, reference training and the depth-quality check (§7). | keyframes, AABB → `GaussianModel` |
| `XFeatExtractor` | XFeat in Core ML plus post-processing (NMS, top-k, descriptor sampling) on the GPU. | grayscale image → `Keypoints {uv, desc, score}` |
| `Matcher` | Mutual-nearest-neighbor matching (cos > 0.82) on the GPU, batched matching against all reference keyframes, and 8-point F-RANSAC filtering. | descriptors × descriptors → matches |
| `AnchorBuilder` | O-SCD reference anchors: XFeat on each keyframe, exhaustive pairwise matching, triangulation with keyframe poses (§6.1). | keyframes → `AnchorSet` |
| `PoseEstimator` | O-SCD online pose: retrieval → 2D–3D → P4P-RANSAC → mini-BA → accept or reject (§6.2). | `Keypoints`, `AnchorSet` → `T_cw` or rejection reason |
| `SAM2Encoder` | Apple's Core ML SAM2.1-tiny image encoder, run on the Neural Engine. | 1024² RGB → 64×64×256 features |
| `ChangeCues` | Method A cues (pixel + feature) and the Method B extra cues (depth + color), with validity masks. | live, render, features, depth → `C` map + valid mask |
| `ChangeFusion` | `R_change` and the SSF online loop: frame sampling, densification and refine. One instance per method. | `C_i`, pose → updated `R_change`; masks |
| `SweepStore` | Disk-backed per-frame store for replay and recording (§9.3). | frames ↔ disk |
| `DemoUI` | The state machine screens, overlay, A/B toggle, HUD and split view. | view models |
| `MacHarness` (macOS CLI) | Replays recorded sessions, runs parity tests, exports masks and metrics. | session folder → masks, report |
| `RefServer` (macOS CLI) | Mac fallback: trains the reference and builds anchors from uploaded keyframes (§7.4). | HTTP upload → model + anchors |

---

## 6. Method A: O-SCD, exact replica

The behavior is ported directly from the O-SCD release at commit `3abfeaa` (github.com/Chumsy0725/O-SCD). All values below were read from that code; citations are in the extraction report (§6.8).

Three findings correct our `papers_explained.md` notes:
- **Pose:** there is no closed-form PnP. Each RANSAC hypothesis is a 20-step LM solve on 4 points.
- **`R_change`:** it learns RGB, opacity, scale and rotation, not a single scalar.
- **Rejection rule:** it counts *residual components*, so "> 100" means **> 50 inlier points**.

### 6.1 Reference anchors (during ReferenceBuild)
1. **XFeat** on each reference keyframe:
   - `top_k = 6144`, at working resolution floored to a multiple of 32 (bilinear resize; keypoints are scaled back);
   - fp16, 5×5 NMS over 1.5·top_k candidates, XFeat's own detection threshold;
   - score = heatmap × reliability; descriptors are L2-normalized.
2. **Exhaustive pairwise matching** of all pairs i < j:
   1. Mutual nearest neighbor on cosine similarity, keeping pairs with **cos > 0.82** (no ratio test).
   2. Filter with **8-point F-RANSAC**: 2,000 hypotheses, Sampson error < `max_err²`, where `max_err = max(2e-3·W, 1.5) px`.
   3. Store the inliers only.
3. **Triangulation:**
   - Each keypoint is tested against up to **8** other keyframes (those with the most stored matches) by intersecting two rays.
   - A candidate is valid if:
     - z > 1e-6;
     - the reprojection error is < `max_err`;
     - its disambiguation is the best so far. Disambiguation is the pixel distance in the other view between the ray's points at depth 1 and depth 10.
   - The point is accepted if the best disambiguation is > `30·max_err`.
   - There is no angle threshold, and conf = 1.
4. **Stored per keyframe:** keypoints, descriptors, a has-3D flag, the 3D points in `W`, and the pose (6D rotation + translation). O-SCD's unused dense XFeat feature map is not built.

### 6.2 Online pose (per processed frame)
1. **XFeat** as in §6.1.
2. **Retrieval:** score every reference keyframe by its mutual-NN (cos > 0.82) match count, with no geometric check. Keep the **top 4**.
3. **2D–3D correspondences:**
   - Match the frame against each of the 4 keyframes with F-RANSAC filtering.
   - Keep matches whose reference keypoint has a 3D point. There is no deduplication across keyframes.
   - If there are more than 4,000, sample 2,000 uniformly without replacement.
4. **PnP-RANSAC ("P4P"):**
   - 2,000 hypotheses, each using 4 random correspondences.
   - Each hypothesis is a pose-only LM solve: **20 iterations**, Huber δ = 1 px, fixed focal, started from the **top-retrieved keyframe's pose**.
   - Inlier = reprojection error < `max_err` px (no cheirality check); score = Σ inlier·conf; the best hypothesis wins.
5. **Mini-BA** on up to **2,000** inliers (a random subset):
   - pose only, **20 LM iterations**, Huber δ = 1 (weight `δ/√|r|`);
   - MAD rejection: `c = median + 4·MAD`, with quantile `1 − 0.5·valid fraction` and `c ≥ 5 px`;
   - LM settings: λ₀ = 1e-5, diagonal floor 1e-2, factor 2;
   - an update is accepted only if the weighted MSE drops, and the rotation is re-orthonormalized every iteration.
   - The final inlier mask uses the last *tentative* parameters (code quirk, reproduced).
6. **Accept** if the mask sum (counted per x and y residual) is **> 100**, i.e. more than 50 inlier points. Otherwise **skip the frame**: no fusion and no mask (P1). The ≤ 4-reference-keyframe crash is impossible here, because the reference scan requires at least 20 keyframes.
7. **Log** `n_corr`, `n_pnp_inliers`, `n_ba_inliers`, the per-stage time and the acceptance flag (P4 format).

RANSAC (2,000 tiny LM solves) runs batched on the GPU, and mini-BA runs on the CPU (D5).

### 6.3 Change cues (`generate_candidate_map`)
- **Pixel cue:**
  - `C_pixel = 0.8·mean_c|I_ren − I_inf| + 0.2·(1 − mean_c SSIM)`.
  - SSIM uses an 11×11 Gaussian window (σ = 1.5), zero padding, C1 = 0.01², C2 = 0.03².
  - Then per-frame min-max with ε = 1e-8.
- **Feature cue:**
  1. **Nearest-neighbor** resize of both images to 1024×1024. The aspect ratio is ignored, and **no ImageNet normalization** is applied: raw values in [0, 1] go in.
  2. Run SAM2.1-hiera-tiny (fp16) and take the last FPN level plus `no_memory_embedding`, giving 256×64×64.
  3. Take the channel mean of |diff|, min-max it, and upsample bilinearly (align_corners = false) to working resolution.
  4. Re-normalize with the pixel cue's **raw** min/max and clamp to [0, 1].
- **Combined:** `C_A = C_pixel + C_feature`, in [0, 2]. There is no blur and no masking in O-SCD. Our validity mask (D3) is the only addition.
- **Core ML match:** Apple's Core ML encoder must be checked to output the same tensor, i.e. the last FPN level plus the no-memory embedding. If its preprocessing differs (e.g. ImageNet normalization built in), we feed it so that its effective input matches O-SCD's. This is checked by the parity test.

### 6.4 Fusion (`R_change`)
- **Initialization:** copy the reference Gaussians (xyz, opacity, scale, rotation) and set **SH-DC = 0**. Rendering at SH degree 0 then gives a color of 0.5 per channel. The background is black and the render is clamped to [0, 1].
- **Learnable parameters** (Adam, ε = 1e-15):

  | Parameter | Learning rate |
  |---|---|
  | f_dc | 0.0025 |
  | opacity | 0.025 |
  | scaling | 0.005 |
  | rotation | 0.001 |
  | xyz | 0 (frozen) |

  This is ordinary 3DGS training with a custom loss and frozen positions, so the engine needs **no scalar-channel render**.
- **Per accepted frame,** 16 steps:
  1. Pick a frame: with probability 0.67, a uniformly random seen frame (the newest included); otherwise the newest. So P(newest) = 0.33 + 0.67/n.
  2. Render, then compute `m = sigmoid(render.mean over channels)`, which lies in [0.5, 0.731].
  3. Minimize `loss = mean(C·(1 − m)) + log(mean(m)² + 1)`.
- **Densification at step index 4** of each frame:
  - grad = mean view-space ‖∇xy‖ accumulated since the last reset, with NDC scaling by 0.5·W; threshold 1e-3.
  - `extent` = 1.1 × the maximum distance of all camera centres (reference + live) from their mean.
  - **Clone** if the max scale ≤ 0.01·extent. Otherwise **split**: 2 children, sampled from the scale std, scale ÷ 1.6, parent deleted.
  - No pruning online.
- **Mask:** the raw render's channel mean **> 0.5** (no sigmoid).
- **Rasterizer constants to match:**
  - α ≤ 0.99, skip α < 1/255, stop at T < 1e-4;
  - +0.3 low-pass on the 2D covariance, 3σ radius;
  - znear / zfar = 0.01 / 100.
  - The engine is checked against these in the spike.
- **Seeds:** 0 for all RNGs.

### 6.5 Refine
- **Budget:** continue from the current total step count up to **3,000 total steps**, including the online steps. It does nothing if the online steps already reached 3,000 (≥ 188 frames).
- **Sampling and loss:** frames are sampled uniformly. The loss is the same with `log(… + 1.000000001)`.
- **Every 100 total steps:** `densify_and_prune(grad 2e-4, min_opacity 0.4)`.
- **At total step 2,000:** reset opacities to ≤ 0.01 and zero the Adam moments (the step count is kept).
- **Final:** re-render every stored frame and binarize at raw > 0.5.
- **Note:** a long live sweep (more than 187 frames) gets no refine, exactly as in O-SCD.

### 6.6 Intrinsics
- **O-SCD's convention:** the focal is `mean(fx, fy)` and the principal point is fixed at the image centre. Query frames reuse the last reference image's focal.
- **Ours:** we use the same single focal (from ARKit, locked focus) for every image, but keep ARKit's calibrated principal point (D10).

### 6.7 Documented deviations from O-SCD
| # | O-SCD | Ours | Why |
|---|---|---|---|
| D1 | Reference poses from COLMAP | ARKit poses from the reference scan | No SfM on the phone. O-SCD only needs posed reference images, whatever their source. |
| D2 | Prebuilt FastGS reference: 30k iterations, SH degree 3 | On-device splatx trainer: LiDAR initialization, 3,000 steps, SH 3, cropped to the ROI (§7). Measured: 41–58 s on the iPhone 17 Pro. The §7.3 depth-quality loss is not yet in (median depth gap 1.1 cm). | Live on-device build. SH 3 keeps the settings of the validated Mac model. |
| D3 | Cues and loss computed over all pixels | Valid pixels only (inside the ROI, rendered reference alpha ≥ 0.5) | Our reference covers only the table, so the unreconstructed room must not count as "change" |
| D4 | PyTorch SAM2.1-tiny | Apple Core ML fp16 SAM2.1-tiny (same weights) | On-device; the gap is bounded by the parity tests |
| D5 | CUDA F-RANSAC and P4P RANSAC, GPU mini-BA | The same algorithms in Metal (RANSAC) and on the CPU (mini-BA) | Platform |
| D6 | Every frame of a recorded sequence is processed | Only the latest frame is processed; frames are dropped while busy | Live input. Method B replays exactly the frames A processed. |
| D7 | Crashes on a rejected pose | The frame is skipped (P1) | Robustness |
| D8 | Post-visit map update | Not implemented | Scope |
| D9 | 6,144 keypoints, 1/4 resolution | Reduced only if the spike requires it, and recorded in the spike report | Performance |
| D10 | Principal point at the image centre | ARKit's calibrated principal point | The iPhone's principal point is off-centre by a few pixels; using the centre would add pose error |
| D12 | XFeat NMS takes the top 1.5·top_k heatmap values; when there are fewer real local maxima, the rest are zero-valued "junk" positions whose positive scores can push low-scoring real keypoints out of the final top_k | Only real local maxima compete for the top_k, so a few more tail keypoints survive (3,668 vs 3,636 on a test image) | Which junk positions CUDA `topk` returns is implementation-defined and can't be reproduced. The top-scoring keypoints, the ones matching relies on, are identical. |
| D13 | (Method B, not O-SCD) `C_depth` thresholds `|D_live − D_ren|` directly | The frame's median `D_live − D_ren` over judged pixels is subtracted first | LiDAR reads about 1 cm deeper than the reference's expected depth everywhere (median 0.6–1.2 cm on Scans 4 and 5). With τ = 1 cm, that bias gave a small positive depth cue on most of the table, and fusion turned it into 84.6 % changed. With alignment: 37.1 % (Scan 4 → Scan 5) and 17.3 % on an unchanged pair (A: 11.7 %). |

### 6.8 Source
The full extraction report, with file:line citations, is in [`oscd_mobile/docs/oscd_code_reference.md`](../../../oscd_mobile/docs/oscd_code_reference.md).

---

## 7. Reference scan and build

### 7.1 Capture
- **Session configuration:**
  - `ARWorldTrackingConfiguration` with `.sceneDepth` and horizontal plane detection.
  - Exposure, white balance and focus are locked via `configurableCaptureDeviceForPrimaryCamera` when the scan starts, and stay locked through the change scan.
- **Keyframe rules:**
  - Take a keyframe when the camera has moved ≥ 5 cm or turned ≥ 10° since the last one, while tracking is `normal`.
  - Reject the frame if the camera is moving faster than 0.3 m/s or 30°/s (estimated from consecutive poses).
  - Target: 60–100 keyframes. The build can't start below 20.
- **Coverage UI:** 8 azimuth sectors × 2 elevation bands around the ROI center, each filled once a keyframe covers it.

### 7.2 Initialization
1. Back-project high-confidence LiDAR depth from every keyframe into `W` and keep the points inside the ROI.
2. Voxel-downsample at 4 mm and color the points from RGB.
3. Each point becomes a Gaussian:
   - position: the point;
   - color: SH DC from the RGB color;
   - scale: isotropic, the mean distance to the 3 nearest neighbors;
   - opacity: the engine's default initialization.

### 7.3 Training
- **Setup:** 480×360, SH degree 0, `0.8·L1 + 0.2·D-SSIM`, 3,000 steps.
- **Densification:** every 100 steps from step 300 to step 2,000, capped at 200k Gaussians.
- **Held-out keyframes:** 10% of keyframes are held out to report PSNR and the depth error.
- **Depth-quality check:**
  - Compute the median `|D_ren − D_lidar|` over high-confidence pixels in the ROI on the held-out keyframes.
  - If it exceeds 1 cm, enable an L1 LiDAR depth loss with weight 0.1. This uses the same custom-loss hook as SSF.
  - If held-out PSNR is below 24 dB, try SH degree 1.
- **Failure handling:** a NaN loss restarts training once with half the learning rate. Training longer than 3 minutes, or a second failure, offers the Mac fallback.

### 7.4 Mac fallback (last resort)
- **Upload:** the **Build on Mac** button sends keyframes over HTTP to `RefServer` on the Mac, which is discovered via Bonjour. Each keyframe includes its RGB (JPEG), depth (16-bit PNG), confidence, pose and intrinsics, plus the ROI.
- **Build:** `RefServer` uses the same `OSCDCore` engine and returns the model (`.ply`) and the anchors.
- **Labeling:** the HUD shows "reference built on Mac".

---

## 8. Method B: O-SCD + depth + color

Method B is identical to A (same poses, same fusion, same schedule and seed) except for the cue: `C_B = clamp(C_A + λ_d·C_depth + λ_c·C_color, 0, 2)`.

### 8.1 Depth cue
- **Inputs:**
  - `D_live`: ARKit `sceneDepth` (high confidence only), upsampled to working resolution with nearest-neighbor.
  - `D_ren`: the reference expected depth (`Σ wᵢzᵢ / Σ wᵢ`) rendered at the PnP pose.
- **Cue:** with `τ(z) = max(1 cm, 0.02·z)`, `C_depth = clamp((|D_live − D_ren| − τ) / 2 cm, 0, 1)`.
- **Sign:** `D_live < D_ren` means something was added in front of the reference; `D_live > D_ren` means a reference surface was removed. The sign is logged and used by S2.

### 8.2 Color cue
1. **Gain alignment:** compute a per-channel gain `g_c = median(live_c / render_c)` over valid pixels where `|D_live − D_ren| < 1 cm`. If there are fewer than 500 such pixels, use `g = 1`. Apply the gain to the render.
2. **Blur:** Gaussian blur with σ = 1 px on both images.
3. **Lab difference:** convert sRGB to CIELAB (D65) and compute `ΔE_w = √((0.25·ΔL*)² + Δa*² + Δb*²)`.
4. **Ramp:** `C_color = clamp((ΔE_w − 10) / 15, 0, 1)`, which is 0 below 10 and 1 above 25.
5. **Masking:** set the cue to 0 where `L* < 15` or `L* > 95` in either image.

A lightness-only swap (for example white ↔ gray) is deliberately left to the pixel cue.

### 8.3 Weights
- **Defaults:** `λ_d = λ_c = 1`.
- **Tuning:** grid search over {0.5, 1, 1.5}² on the recorded sessions of §12.3, using hand-labeled masks. Pick the best mean IoU and record it in the spike report.

### 8.4 Execution as a replay
- **Trigger:** pressing **Replay with depth + color** initializes a fresh `R_change_B` from the reference.
- **Online pass:** it iterates over the SweepStore frames in their original order. For each frame it re-renders the reference and computes `C_B` (reusing the stored `C_A`), then runs the same `steps_per_frame` schedule with the same RNG seed as A.
- **Refine and display:** it then refines with the same budget. Pose and SAM2 are not recomputed, so replay is expected to run faster than the live sweep.
- **Toggle:** the A | B toggle renders `R_change_A` or `R_change_B` at the display pose.

### 8.5 S2 (stretch): Gaussian insertion for added objects
- **Problem:** `R_change` only has Gaussians where the reference had geometry, so an added block's change lands on the table behind it.
- **Insertion:** where `D_live < D_ren − 3 cm` in at least 2 frames, back-project the pixels into `W` and voxel-hash them at 5 mm.
- **New Gaussians:** add them to `R_change_B` only, with isotropic scale 5 mm, opacity 0.8 and SH-DC set so they render at 1.0 (high change).

---

## 9. Runtime design

### 9.1 Live pipeline (ChangeScan)
- **Frame intake:** a latest-frame slot; any frame arriving while the slot is busy replaces the waiting one.
- **Three stages,** each on its own queue, so Neural Engine and CPU work overlaps GPU work:
  1. **Pose:** XFeat (Neural Engine) → retrieval matching and RANSAC (GPU) → mini-BA (CPU).
  2. **Render and encode:** reference RGB and depth render (GPU); SAM2 on the live and rendered images (Neural Engine).
  3. **Cues and fusion:** cues (GPU), push to the buffer, `steps_per_frame` SSF steps (GPU).
- **Rate:** the mask update rate is set by the slowest stage. The HUD shows per-stage milliseconds.

### 9.2 Display path (every display frame)
- **Display pose:** `T_wc_disp(now) = T_wc_pnp(k) · T_wc_ar(k)⁻¹ · T_wc_ar(now)`, where `k` is the last accepted frame. This carries the last PnP pose forward with ARKit's relative motion.
- **Overlay:** render the active `R_change` at that pose (480×360, upsampled) and overlay pixels whose channel mean is > 0.5 in red at 50% alpha on the camera image.
- **Staleness:** the overlay is marked stale (dimmed) if the last accepted pose is more than 2 s old.

### 9.3 SweepStore (per processed frame)
- **Contents:** `T_cw` (PnP), K, live RGB at 480×360, LiDAR depth and confidence (256×192), `C_A` (fp16), pose statistics and timestamps. That's about 1 MB per frame.
- **Storage:** written to the session folder on disk. The same folder format is what the session recorder exports and the Mac harness reads.

### 9.4 Error handling
| Situation | Detected by | Response |
|---|---|---|
| Pose rejected (≤ 50 inlier points or too few matches) | `PoseEstimator` | Skip the frame for fusion and count it in the HUD. After 5 consecutive rejections, show an "aim at the table" hint. The overlay keeps bridging from the last good pose. |
| ARKit tracking limited during the reference scan | `trackingState` | Pause keyframe capture and show a "slow down / more texture" banner |
| ARKit tracking limited during the change scan | `trackingState` | Detection continues on PnP poses. Only display bridging and LiDAR depth are affected. |
| Session interrupted | `ARSessionObserver` | The reference, anchors and SweepStore are already on disk. Restart the session and resume, since PnP re-anchors to `W`. Display bridging restarts at the first accepted frame of the new session. |
| PnP vs ARKit pose disagreement | HUD (difference in mm and degrees) | Informational. Over 2 cm or 2° shows a warning. |
| Thermal state `.serious` or worse | `ProcessInfo.thermalState` | Algorithm parameters stay as they are (exactness); the rate drops and the HUD warns the presenter |
| Memory warning | Caps: reference ≤ 200k Gaussians, `R_change` ≤ 300k; SweepStore on disk | Flush caches |
| Reference build NaN or too slow | §7.3 | Retry, then the Mac fallback |
| Model files missing or failing to load | App launch check | Blocking error screen |

---

## 10. Repository layout
```
oscd_mobile/
  README.md                 build, run and demo instructions
  project.yml               XcodeGen spec (iOS app + macOS tools)
  Packages/OSCDCore/        Swift package: Capture, Splat, Pose, Cues, Fusion, Store (+ Metal)
    Tests/OSCDCoreTests/    unit and parity tests
  App/                      SwiftUI iOS app (DemoUI)
  MacHarness/               macOS CLI: replay, parity, export
  RefServer/                macOS CLI: Mac fallback reference build
  tools/                    Python: convert_xfeat_coreml.py, fetch_models.sh,
                            dump_oscd_fixtures.py (+ Sol job), reference_cues.py
  docs/                     oscd_code_reference.md, spike_report.md, demo_runbook.md
```
- **Model files:** the XFeat `.mlpackage` and SAM2.1 encoder `.mlpackage` are fetched or converted by `tools/` into `App/Models/`, which is gitignored.
- **Fixtures:** stored in `oscd_mobile/fixtures/` (gitignored) and pulled from Sol by a script.

---

## 11. Spike (days 1–3) and go/no-go
The spike measures, on the demo iPhone:

| Item | Target | If missed |
|---|---|---|
| Reference training: 3k steps at 480×360, ≤ 150k Gaussians | ≤ 90 s | ≤ 180 s: use fewer steps. Otherwise the Mac fallback becomes the default. |
| XFeat (Core ML), per frame | ≤ 15 ms | Lower XFeat input size (D9) |
| Retrieval matching against all keyframes | ≤ 60 ms | Fewer keypoints (D9) |
| P4P-RANSAC (2,000 × 20-step LM, GPU) + mini-BA | ≤ 20 ms | Optimize the batched kernel |
| Rasterizer constants (§6.4) match the engine | yes | Patch the engine |
| SAM2 encoder, per image | ≤ 120 ms | Accept a lower rate if ≥ 3 Hz still holds |
| SSF: 16 steps at 480×360 | ≤ 250 ms | 240×180 working resolution (D9) |
| End-to-end mask update rate | ≥ 3 Hz | The fixes above, in that order |

**Engine choice:** msplat-ios or splatx-metal. The engine must train on the device, render expected depth, accept a custom loss kernel and per-parameter learning rates (frozen xyz), and match O-SCD's rasterizer constants.

**Records:** every result and decision goes into `oscd_mobile/docs/spike_report.md`.

---

## 12. Testing and verification

### 12.1 Parity with the real O-SCD (golden fixtures)
**Fixture source:** `tools/dump_oscd_fixtures.py` plus a Sol job runs the patched O-SCD on one PASLCD instance for 10 frames. It dumps:
- XFeat keypoints and descriptors;
- retrieval sets;
- 2D–3D correspondences;
- poses and inlier counts after PnP and BA;
- `I_inf`, `I_ren`, `C_pixel`, `C_feature` and `C`;
- `R_change` parameters and its rendered change map before and after one frame's 16 steps, with the sampled frame indices;
- `L_SSF` values and the final masks.

| Check | Tolerance |
|---|---|
| XFeat (Core ML vs PyTorch) | ≥ 95% of the top 1,000 keypoints within 1 px; descriptor cosine ≥ 0.99 |
| Retrieval | Same top-`n` reference set |
| RANSAC + mini-BA on the dumped correspondences (same sampled indices injected) | Rotation ≤ 0.1°; translation ≤ 0.5% of scene scale; inlier counts within 2% |
| `C_pixel` (identical inputs) | Max absolute error ≤ 1e-3 |
| `C_feature` (Core ML SAM2) | Pearson r ≥ 0.98; mean absolute error ≤ 0.02 |
| SSF loss and gradients (f_dc, opacity, scale, rotation) on fixed inputs | Relative error ≤ 1e-3 |
| One frame of fusion steps, including step-4 densification (injected sampling sequence) | Gaussian count equal; rendered change map within 1e-3 |

### 12.2 Method B cue unit tests
- **Reference implementations:** NumPy in `tools/reference_cues.py`.
- **Synthetic cases:**
  - depth offsets of 0.5, 1.5, 3 and 5 cm at 0.5 m;
  - red → blue swap (should fire);
  - white → gray swap (`C_color` ≈ 0);
  - global exposure ×1.3 (≈ 0 after gain alignment).

### 12.3 Integration (Mac harness on recorded tabletop sessions)
Three sessions: no change; add + remove; move + recolor.
- **No-change session:** after refine, < 1% of valid pixels are flagged (both methods).
- **Change sessions:** mean IoU ≥ 0.4 for Method B on 10 hand-labeled frames per session. Method A is reported for comparison.

### 12.4 On-device
- Per-stage timing goes to a CSV for each session.
- XCTest performance tests run for each stage.
- `docs/demo_runbook.md` holds the rehearsal checklist: 3 full runs on the real table under classroom lighting, plus a recorded backup video.

---

## 13. Schedule (about 21 working days)
| Days | Milestone | Demoable at the end |
|---|---|---|
| 1–3 | Spike (§11); session recorder; ROI | Go/no-go decisions |
| 4–7 | Reference capture + on-device 3DGS build + render; anchors | "Scan → 3DGS of the table on the phone" |
| 8–10 | Online pose (XFeat → retrieval → P4P-RANSAC → BA → reject); HUD; pose parity tests | Live pose vs ARKit check |
| 11–13 | O-SCD cues (pixel + SAM2); per-frame cue overlay | **Fallback demo:** live change cues on the phone; tagged build |
| 14–17 | `R_change` + SSF online fusion + refine; parity tests | Full Method A |
| 18–19 | Method B cues, SweepStore replay, A/B toggle | Full demo |
| 20–21 | Tuning on the real table, rehearsals, backup video | Demo-ready |

Stretch items S1–S3 are attempted only after day 19 if everything else passes.

---

## 14. Risks
| Risk | Mitigation |
|---|---|
| On-device training too slow or out of memory | Spike gate; fewer steps or Gaussians; Mac fallback |
| XFeat Core ML conversion problems | Fixed-size export (as O-SCD does); post-processing in Metal/Swift; the LiteRT port as a reference |
| Retrieval matching cost on the phone | Batched GPU matmul; keypoint reduction (D9) |
| SAM2 latency on the Neural Engine unknown | Pipelining; a lower update rate is acceptable if ≥ 3 Hz holds |
| Too few PnP inliers on a plain table | Patterned mat; HUD hint |
| D3 masking changes the SSF balance | Parity tests on unmasked fixtures; the no-change session check |
| Thermal throttling during the demo | Short sessions, phone out of its case, HUD warning |
