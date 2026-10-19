# O-SCD Mobile, phase 2: on-device reference build, faster live view, Method B

**Spec:** `docs/superpowers/specs/2026-10-14-oscd-mobile-design.md` (§7 reference build, §8 Method B, §9.1 live pipeline).
**Starting point:** branch `oscd-mobile/phase1` at `1fcb08b9`. Method A runs on the phone (Detect and Live view); the before-model is trained on the Mac (`tools/build_reference.sh`).
**Build quirks:** see `oscd_mobile/README.md`:
- the repo is in iCloud;
- splatx is a symlinked clone in `~/Library/Caches/oscd-mobile/splatx-metal`;
- build with `--scratch-path` / `-derivedDataPath` under `~/Library/Caches`.

## Global constraints
- Method A stays exact O-SCD. Every change to its behaviour is a numbered deviation in the spec.
- The phone is an iPhone 17 Pro on iOS 26 (device `00008150-0011681A3430C01C`, bundle `com.veripheye.changes3d`). splatx kernels compile on the phone (`kernel_src_dir`).
- The phone's reference training uses the settings of the Mac model that worked:
  - 3,000 iterations, `max_points` 200k, factor 1 (480×360);
  - SH degree 3, SSIM weight 0.2;
  - resident on, MCMC off, balanced thermal mode;
  - LiDAR points cropped to the table region, voxel 4 mm.
- Verify on the phone with launch arguments and pulled files, not just a build.

## Task 1: Export a scan as a COLMAP dataset in Swift (OSCDCore)
- **Files:**
  - create `Packages/OSCDCore/Sources/OSCDCore/Store/ColmapExport.swift`;
  - test in `Tests/OSCDCoreTests/ColmapExportTests.swift`.
- **Port of `tools/session_to_colmap.py`:**
  - one PINHOLE camera;
  - each keyframe's world-to-camera pose as COLMAP qvec/tvec (ARKit camera-to-world × diag(1, −1, −1, 1), inverted);
  - PNG images;
  - LiDAR points with confidence ≥ 2, back-projected with the depth-map intrinsics scaled by depthWidth / W, cropped to the table region, deduplicated at 4 mm keeping the first point per voxel;
  - colour from the nearest working-resolution pixel.
- **Tests (red first):**
  - the qvec/tvec of a known pose round-trips to the same world-to-camera matrix;
  - a synthetic flat depth plane back-projects onto that plane in world coordinates;
  - the region crop and the 4 mm voxel deduplication give the expected counts.
- **Check:** export Scan 4 (`2026-10-14T15-45-15Z`) with both the Swift port and the Python script, and compare:
  - `cameras.txt` and `images.txt` should be equal to within float printing;
  - `points3D.txt` should have the same point count.

## Task 2: Build the reference on the phone (app)
- **Files:** `SpikeApp/Sources/ReferenceBuilder.swift` (new); scan detail screen; Settings text.
- **ReferenceBuilder,** on a dedicated thread:
  - `ColmapExport` into `Caches/reference-build/<scan>/`;
  - `sx_train_run_args` with the global settings, `kernel_src_dir` and a progress callback;
  - thermal state injected from `ProcessInfo.thermalState` (`sx_thermal_set_state`);
  - `reference.ply` written atomically into the scan folder;
  - a log in `Documents/build-log.txt`.
- **UI:** "Build 3D model on this phone" on the scan detail screen, with a Romulus-style stage list (Export / Train i of 3000, Gaussians, time left / Save) and Cancel (`sx_train_request_stop`).
- **Launch argument:** `-buildReference <scan folder>` for headless checks.
- **Check on the phone:**
  - build Scan 4's model on the phone: record the time, Gaussian count and final loss against the Mac's 47 s / 95k;
  - run Detect Scan 4 → Scan 5 with the phone-built model and compare with the Mac model: 28/47 frames placed, 16.8 % changed.

## Task 3: Faster live view (exact)
- Move fusion and the change-mask render to a second splatx thread. The live view's mask comes from the latest fused model, so its update rate is max(pose + render + SAM2, fusion) rather than their sum.
- **Check:** median live update interval before and after, on the same before-scan.

## Task 4: Method B (spec §8)
- Expected-depth render of the reference at the pose. splatx's render session returns RGBA only, so add a depth output to the patch: Σ wᵢzᵢ / Σ wᵢ.
- `CuesB` already exists in OSCDCore with tests: depth cue, colour cue, combine.
- **Detect "Method B":** a replay over the same frames and poses as A, with `C_B = clamp(C_A + λd·C_depth + λc·C_color, 0, 2)`, fused into a second change model with the same seed.
- **Results:** an A | B toggle.
- **Check:** Detect Scan 4 → Scan 5 with A and with B; compare changed area and masks.

## Ledger
(One line per finished task: commits, checks run, results.)
- Task 1: done. `ColmapExport` + 4 tests (66/66 pass, 11 skipped without Sol fixtures). Parity with `session_to_colmap.py` on Scan 4: cameras identical, 47 poses within 1.8e-7, 100,003 points within 3e-8 m with identical colours, PNGs identical. Ruling: pose test tolerance 1e-6, not 1e-9, because ARKit poses are Float.
