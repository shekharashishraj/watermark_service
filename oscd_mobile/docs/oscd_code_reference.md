# O-SCD @ `3abfeaa`: code reference for the mobile port

These notes come from reading the code of github.com/Chumsy0725/O-SCD at commit `3abfeaa` (2026-10-14); nothing was run. Citations are `file:line` in that commit. The FastGS rasterizer (`forward.cu` / `backward.cu`) and HF transformers v4.56.1 `modeling_sam2.py` were also checked.

This file is the source of truth for the "exact replica" behavior in `docs/superpowers/specs/2026-10-14-oscd-mobile-design.md` §6.

## A. Arguments (`arguments/config_args.py`, abbreviated CA)

| Arg | Default | Line | Used by `oscd.py` |
|---|---|---|---|
| --refine | False | 29 | ✔ (`run_oscd.sh` sets it) |
| --pyr_levels | 1 | 31 | ✔ (no pyramid) |
| --num_kpts | int(4096·1.5) = 6144 | 35 | ✔ |
| --match_max_error | 2e-3 (fraction of width) | 37 | ✔ |
| --fundmat_samples | 2000 | 39 | ✔ |
| --min_num_inliers | 100 | 41 | ✔ |
| --num_keyframes_for_triangulation | 8 | 51 | ✔ |
| --num_reference_keyframes | 4 | 53 | ✔ |
| --pnpransac_samples | 2000 | 55 | ✔ |
| --num_pts_miniba_incr | 2000 | 57 | ✔ |
| --iters_miniba_incr | 20 | 59 | ✔ |
| --test_hold | -1 | 23 | Only sets `is_test`, which is unused (`oscd.py:183` passes False) |
| --use_colmap_poses, --eval_poses, --fix_focal, --init_focal, --init_fov, viewer and port args | – | – | ✘ |

- **Inherited from `arguments/__init__.py`:**
  - `sh_degree` 3 (:49); `-r/--resolution` -1 (:54).
  - Learning rates: position 1.6e-4, feature 0.0025, opacity 0.025, scaling 0.005, rotation 0.001 (:77-84).
  - `percent_dense` 0.01; `densify_grad_threshold` 2e-4 (:89-95).
- **`run_oscd.sh:26-28`** overrides only `--resolution 4 --test_hold 5`, plus `--refine` (REFINE=true, :15).
- There is no `iters_miniba_init` argument.

## B. Reference anchors

**Poses and intrinsics**
- Reference poses and intrinsics come from COLMAP `reference_scene/sparse/0` (image_dataset.py:91-93, 165-203).
- Focal = mean(fx, fy), scaled to the working width (:186-187).
- The principal point is always ((W-1)/2, (H-1)/2) (keyframe.py:46, pose_initializer.py:31).

**XFeat** (feature_detector.py)
- top_k 6144 at working resolution, fp16 (:94-100, 183).
- Input is resized bilinearly to floor(H/32)·32 × floor(W/32)·32, and keypoints are scaled back (:106-112, 166).
- NMS: 5×5 kernel, 1.5·top_k candidates, XFeat's own `detection_threshold` (:116-141).
- Score = heatmap × reliability; descriptors are L2-normalized; features with score ≤ 0 are zeroed (:146-168).

**Matching** (every pair i<j, oscd.py:136-142)
- Mutual nearest neighbor on cosine similarity, keeping pairs with cos > 0.82; no ratio test (matcher.py:29-41).
- Then 8-point F-RANSAC: 2000 hypotheses, Sampson error < max_err² (ransac.cu:43-148; ransac.py:132). Only inliers are stored.
- `max_err = max(2e-3·W, 1.5)` px (oscd.py:91).

**Triangulation** (triangulator.py, keyframe.py:120-155)
- Each keypoint is tested by two-ray intersection against up to 8 other keyframes, chosen by the most stored matches (:110-125).
- A candidate is valid if:
  - z > 1e-6;
  - reprojection error < max_err;
  - the other uv is valid;
  - its disambiguation is the best so far (:79-83).
- Disambiguation is the pixel distance, in the other view, between the ray's points at depth 1 and depth 10 (:43-47).
- The point is accepted if the best disambiguation is > 30·max_err (:88, 102).
- The angle is computed but unused (:37). The minimum track is 2 views. conf = 1 (keyframe.py:154).

**Stored per keyframe** (keyframe.py:41-74)
- kpts, feats, valid, has_pt3d, pts3d (world), depth, pts_conf, matches
- `rW2C` (6D rotation) and `tW2C`
- an unused dense stride-4 feature map

## C. Online pose (`pose_initializer.py`, `ransac.py`, `mini_ba.py`)

1. **XFeat:** as in B (oscd.py:178).
2. **Retrieval:**
   - Score each reference keyframe by its mutual-NN (cos > 0.82) match count, with no geometric check. Take the top 4 (pose_initializer.py:138-149).
   - Quirk: with ≤ 4 reference keyframes the code raises NameError.
3. **2D–3D correspondences** (:65-77):
   - Match against each of the 4 keyframes with F-RANSAC filtering. Keep matches whose reference keypoint has a 3D point. No deduplication.
   - If there are more than 4000 (2·2000), `torch.multinomial(confs, 2000)` takes 2000 without replacement; the weights are uniform (:34, 80-85).
4. **P4P-RANSAC** (ransac.py:57-69, 102-111, 137-198):
   - 2000 hypotheses, each using 4 random correspondences.
   - Each hypothesis is fit by MiniBA: pose only, fixed focal, 20 LM iterations, Huber δ=1, no MAD.
   - Every hypothesis is initialized from the top-retrieved keyframe's pose (pose_initializer.py:88-90).
   - Inlier = ‖reprojection‖ < max_err px, with no cheirality check. Score = Σ inlier·conf.
5. **Mini-BA** (pose_initializer.py:39-41, 100-113; mini_ba.py):
   - Optimizes the 6D rotation + translation only, for 20 iterations, on 2000 points: a random subset of the inliers, or padded with uv=-1 (masked).
   - Huber δ=1 px, applied as weight δ/sqrt|r| (:118-126).
   - MAD rejection: c = median + 4·MAD, quantile q = 1 − 0.5·(valid fraction), c ≥ 5 px (:97-114).
   - LM: λ0 = 1e-5, diagonal floor 1e-2, factor 2. An update is accepted only if the weighted MSE drops (:243-252).
   - The rotation is re-orthonormalized every iteration.
   - Quirk: the final mask uses the last *tentative* parameters (:256-259).
6. **Rejection:** `mask.sum() > 100` (pose_initializer.py:117).
   - The mask is expanded to the x and y residuals (mini_ba.py:113), so this means more than 50 inlier points.
   - Otherwise the function returns None, and `oscd.py:182` crashes (our patch P1 skips the frame).

## D. Change cues (`generate_candidate_map`, oscd.py:30-67)

**Pixel cue**
- SSIM: 11×11 Gaussian window (σ=1.5), zero padding, C1=0.01², C2=0.03² (loss_utils.py:46-81).
- `0.8·mean_c|R−I| + 0.2·(1 − mean_c SSIM)`, then min-max with ε=1e-8 (oscd.py:32-36).

**Feature cue**
- Model: `facebook/sam2.1-hiera-tiny`, fp16, `torch.compile` (:154-155).
- **Nearest** resize to 1024×1024. Aspect ratio is ignored, and there is **no ImageNet normalization**: raw [0,1] values go in (:38-45).
- Output: `get_image_embeddings(...)[-1]`, shape (2,256,64,64). This is the last FPN level plus `no_memory_embedding` (HF modeling_sam2.py:1346-1355).
- Channel mean of |diff| → min-max → bilinear upsample to H×W (align_corners=False) → re-normalize with the pixel cue's **raw** vmin/vmax → clamp to [0,1] (:52-63).

**Combined:** `C = pixel + feature`, range [0,2] (:65). No blur, no masking.

## E. Fusion (`R_change`)

**Initialization** (gaussian_model.py:353-395)
- Copied from the reference PLY: xyz, opacity, scale and rotation. f_rest is copied but unused. **f_dc is set to 0.**
- Rendered at SH degree 0 (gaussian_renderer:145): color = max(0, C0·dc + 0.5), so it starts at 0.5 per channel (forward.cu:35-75).
- Black background (oscd.py:163); the render is clamped to [0,1] (renderer:202).

**Learnable parameters** (Adam, eps 1e-15; gaussian_model.py:202-221)

| Parameter | Learning rate |
|---|---|
| f_dc | 0.0025 |
| f_rest | 1.25e-4 (no gradient) |
| opacity | 0.025 |
| scaling | 0.005 |
| rotation | 0.001 |
| xyz | **0** (`spatial_lr_scale` stays 0, :61) |

**Training loop**
- 16 steps per frame (oscd.py:207).
- Frame choice: if `np.random.rand() > 0.33`, a uniformly random seen frame (newest included); otherwise the newest. So P(newest) = 0.33 + 0.67/n (:209-212).
- Loss (:221-226):
  ```
  m = sigmoid(render.mean(0))          # in [0.5, 0.731]
  loss = mean(C*(1-m)) + log(mean(m)**2 + 1.0)
  ```

**Densification** at in-frame step index 4 (:242-255)
- grad = mean view-space ‖∇xy‖, accumulated since the last reset. The NDC-scaled gradient uses 0.5·W (backward.cu:489). Threshold = 5×2e-4 = 1e-3.
- extent = 1.1 × the max distance of all camera centres (reference + inference) from their mean.
- Clone if max scale ≤ 0.01·extent. Otherwise split: N=2, sampled from the scale std, scale ÷1.6, parent deleted (gaussian_model.py:497-535).
- No online pruning.

**Online mask:** raw render, mean over channels, > 0.5. No sigmoid (oscd.py:258-263).

## F. Refine (`--refine`)
- Runs from the current total step count to 3000. It does nothing if there are ≥ 188 frames (oscd.py:268).
- Frames are sampled uniformly (:272). Same loss, except the log constant is 1.000000001 (:281).
- Every 100 total steps: `densify_and_prune(2e-4, min_opacity=0.4, extent, None)`. The size threshold never fires (:297-302; gaussian_model.py:537-549).
- At total step 2000: reset opacity to ≤ 0.01 and zero the Adam moments; the step count is kept (:304-305; gaussian_model.py:397-410).
- Finally, every view is re-rendered and binarized at raw > 0.5 (:312-319).

## G. Resolutions

| Data | Resolution |
|---|---|
| Working images | original ÷ 4, INTER_AREA (image_dataset.py:132-139; run_oscd.sh:26) |
| Rendering, pixel cue, change render | working H×W |
| XFeat | working resolution floored to multiples of 32 |
| SAM2 | 1024×1024 → 64×64 |

## H. Reference 3DGS
- Loaded prebuilt from `<source>/reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply` (oscd.py:77). There is no trainer in the tree.
- The FastGS rasterizer is used, with Speedy-Splat tile culling `mult=0.5` (forward.cu:246). SH degree 3.

## I. Other details to reproduce
- **fp16:** XFeat and SAM2 only; everything else is fp32.
- **Seeds:** 0 everywhere (oscd.py:70-74).
- **Intrinsics:** inference cameras reuse the *last* reference image's focal and FoV (oscd.py:117-119, 135, 191). znear/zfar = 0.01/100.
- **Rasterizer:** α ≤ 0.99; skip α < 1/255; stop at T < 1e-4; +0.3 low-pass on the 2D covariance; 3σ radius (forward.cu:115, 387-391).
- **Warm-up:** 10 dummy cue computations before the loop (oscd.py:167-170).
- **Per-frame order:** pose → reference render → cue → 16 fusion steps → mask.

## Where `scd_robustness/docs/papers_explained.md` §3 disagrees with the code
1. **Rejection rule:** "≤ 100 inliers" is really ≤ 100 residual components, i.e. ≤ 50 points. Our P4 `n_ba_inliers` is likewise 2× the point count.
2. **Pose solver:** there is no closed-form PnP. Each RANSAC hypothesis is 20 LM steps from the top reference pose.
3. **Change representation:** it is a 3-channel SH-DC color, not one value starting at 0. Opacity, scale and rotation also train; xyz is frozen through lr = 0.
4. **σ(M):** σ is applied to a value already clamped to [0,1], so it lies in [0.5, 0.731].
5. **Frame sampling:** P(newest) = 0.33 + 0.67/n.
6. **Anchors:** the notes lack the matching details (mutual NN with cos > 0.82, then F-RANSAC) and the triangulation gates (reprojection error and the 30× disambiguation threshold).
7. **Refine:** the 3000 steps include the online steps. Refine also prunes opacity < 0.4 every 100 steps and resets opacity at step 2000.
