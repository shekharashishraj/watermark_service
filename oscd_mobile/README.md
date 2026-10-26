# oscd_mobile

On-device port of O-SCD for a live tabletop change-detection demo on an iPhone Pro.
Design: `../docs/superpowers/specs/2026-10-14-oscd-mobile-design.md`.
O-SCD code reference: `docs/oscd_code_reference.md`.

## Layout
- `Packages/OSCDCore`: Swift package with the algorithms that need no splatx: pose, cues, scans and their COLMAP export, the orbit camera (iOS 17 / macOS 14)
- `Packages/OSCDFusion`: O-SCD change fusion and the reference's expected depth on the patched splatx (macOS 15 / iOS 26)
- `tools/`: Python model conversion, Sol fixture dump, COLMAP export, `build_reference.sh` (Mac build of a scan's model)
- `SpikeApp/`: the iOS app "3D Changes"
- `docs/`: spec deviations and results (`spike_report.md`), run reports (`2026-10-16_demo_app_runs.md`, `2026-10-19_phase2_runs.md`), related papers (`papers/`)
- `fixtures/`, `Models/`, `third_party/`: generated or downloaded, not in git

## Tests
    cd Packages/OSCDCore
    swift test                                  # fast tests; fixture tests skip if fixtures are absent
    swift test -c release --filter Parity       # O-SCD parity tests (need fixtures/ and Models/)

If `swift test` fails at a CodeSign step with "resource fork, Finder information, or similar
detritus not allowed", the checkout sits in an iCloud-synced folder (e.g. Desktop) that stamps
Finder metadata onto build products. Build outside it:

    swift test --scratch-path ~/Library/Caches/oscd-mobile/OSCDCore-build

## Models (not in git)
    python3 -m venv tools/.venv && tools/.venv/bin/pip install -r tools/requirements-mac.txt
    git clone https://github.com/verlab/accelerated_features third_party/accelerated_features
    tools/.venv/bin/python tools/convert_xfeat_coreml.py     # → Models/XFeat_480x352.mlpackage
    tools/.venv/bin/python tools/convert_sam2_coreml.py      # → Models/SAM2_1Tiny_1x1024.mlpackage (batch 1)

If torch.hub fails with CERTIFICATE_VERIFY_FAILED (python.org Python without its certificates),
copy the weights the checkout already has into torch's cache:
`mkdir -p ~/.cache/torch/hub/checkpoints && cp third_party/accelerated_features/weights/xfeat.pt ~/.cache/torch/hub/checkpoints/`

## App ("3D Changes")
    cd SpikeApp && xcodegen generate && open OSCDSpike.xcodeproj
- **Scans:** every recorded scan (`Documents/sessions/<date>/`), newest first, named Scan 1, 2, … by capture order. A scan's detail screen has:
  - **Build 3D model on this phone** (Settings → Model quality: Quick ≈ 30 s, Standard ≈ 2 min (default), High ≈ 4 min; the old model is kept as `reference.previous.ply`);
  - **View 3D model** (drag to orbit, pinch to zoom, double-tap to reset);
  - **Use as the before-scan**;
  - the scan's saved results.
- **Capture:** Record, move slowly over the scene (a tabletop or a room), Stop. Each keyframe is the sharpest frame of a 0.25 s window (up to 8 frames, PocketGS gradient energy, saved as `stats.sharpness`); nothing is saved while the phone moves faster than 0.6 m/s or turns faster than 30°/s (averaged over 0.25 s), and the screen says so after 1.5 s of it; the scanned area (`roi`) is worked out from the scan's LiDAR at Stop (`SceneRegion`).
- **Detect:**
  - choose the before-scan (it needs a 3D model) and the after-scan, then Run;
  - Method A goes over every after-scan frame; "Also B" adds the LiDAR depth and colour cues over the same frames;
  - Results show the change masks (A | B toggle) and each stage's time;
  - after the final masks, "Find objects" runs the object tracker over the placed frames with their final masks (B's when B ran, else A's), reusing each frame's SAM2 features from a half-precision cache in `Library/Caches/detect/<after>_<before>/` that the run deletes when it ends, and rendering the before model again at each frame's pose (`MethodA.render(pose:)`); changes lying within 1 cm of the table (its height from the before-scan's LiDAR, logged as `semantic: support y`) are ignored;
  - results are saved in `<after-scan>/detections/<before-scan>/` (summary.json + one JPEG per frame), with B under `B/`; the object events go to `objects.json` (an `ObjectReport`) in B's folder when B ran, else in A's.
- **Settings:** Live view (Method A on the camera feed against the chosen before-scan) and the stage benchmark. The Live view also feeds each fused frame (A's mask, LiDAR depth and the before model's depth) to an object tracker on its own thread, and rewrites the object events to `Documents/preview/objects.json` (an `ObjectReport`) after every tracker update.

Capture and Live view share one ARView (`SharedARView.swift`). With a second ARView, RealityKit drew the camera feed in only one of them, and Capture went black.

**Building a scan's model on the Mac instead** (with the phone connected):

    tools/build_reference.sh <scan folder>

The script copies the scan off the phone, exports it, trains splatx (8,000 iterations, ≤ 400k Gaussians: the phone's Standard quality), and copies `reference.ply` back into the scan's folder. It takes about 2 minutes. Python runs from a venv in `~/Library/Caches`, because iCloud offloads `tools/.venv`.

**Checks without tapping:** launch arguments (folder names under `Documents/sessions`):

    xcrun devicectl device process launch --device <id> --terminate-existing com.veripheye.changes3d -- -detectBefore <before> -detectAfter <after> -detectB YES

| Argument | Effect |
|---|---|
| `-tab capture` | Open Capture |
| `-live YES` | Open the Live view and start it |
| `-buildReference <scan>` | Build that scan's model on the phone |
| `-viewModel <scan>` | Open the 3D viewer; `-viewYawDeg` / `-viewPitchDeg` turn the first view |
| `-beforeScanID <scan>` | Set the before-scan |
| `-detectObjects YES` | With `-detectBefore`/`-detectAfter`: also write the object report to `app-log.txt` as one `objects:` JSON line, plus diagnostics: `objects-roi:` (the ROI box), per frame `objects-frame:` (changed pixels by depth sign and \|d\| band, the frame's LiDAR − model depth offset, components, created/merged/dropped), `objects-comp:` (each signed component's points, how many lie within the edge margin of the ROI's vertical faces and how many below the support gate) and `objects-gone:` (instances merged or dropped), and at the end one `objects-instance:` line per tracked instance and one `objects-score:` line per left × arrived pair |

**Capture checks:** at Stop, `app-log.txt` gets one `capture stats:` line:
- frame gap (median, max) and `didUpdate` time (median, p95, max): does keyframe selection slow the camera down?
- sharpness and copy time: the cost of scoring frames and copying the window's sharpest one;
- keyframe ÷ first-frame sharpness, and how many windows had a first frame under 70 % as sharp: what the window gains over keeping the first frame past the gate;
- candidates per window: 1–2 means the window rarely had a choice.

Each keyframe in `session.json` keeps `stats.sharpness`, `stats.firstSharpness` and `stats.candidates`, and the line `capture: scanned area …` gives the box's size.

**On-phone logs** (pull with `xcrun devicectl device copy from ... --source Documents/<file>`):
- `Documents/app-log.txt`: camera, screen and build events, including each session's first-frame brightness; after each Detect run's "Find objects", one `semantic: support y` line and one `detect: find objects` line (frames ingested, time, median ingest ms, ICP fits, each event's summary);
- `Documents/preview-log.txt`: one line per Live view frame; at start one `semantic: support y` line (the table height the object layer gates on); per tracked frame a `semantic:` line (frame, ingest and report time in ms, components, instances) and an `objects:` line (event count and each event's summary); at Stop one `semantic:` line with frames ingested, frames dropped and the median ingest time.

When building the app, check the build log for warnings as well as errors: the Live view bug of 2026-10-19 was flagged only by a warning.

## splatx-metal (reference rendering + O-SCD fusion)
The app renders the reference splat and runs O-SCD's change fusion with a patched splatx-metal:

    git clone https://github.com/PoSTMEDIA-AI/splatx-metal third_party/splatx-metal
    git -C third_party/splatx-metal checkout 9ae5f66
    git -C third_party/splatx-metal am ../../patches/splatx-oscd-fusion.patch

The patch adds:
- `Trainer::backward_custom`: custom-loss backward on the host path;
- `splatx::oscd::ChangeFusion`: R_change, SSF loss, torch-style Adam, O-SCD densification, refine;
- `splatx::oscd::DepthRenderer`: the reference's expected depth Σwz / Σw, for Method B;
- a C API for all of these (`oscd_fusion_c.h`).
 `Packages/OSCDFusion` wraps it for Swift; its tests run on the Mac:

    cd Packages/OSCDFusion && swift test --disable-swift-testing --scratch-path ~/Library/Caches/oscd-mobile/OSCDFusion-build

splatx kernels are compiled on the phone at first use (Xcode 27's offline metallib format 2.9 does not
load on iOS 26).

If a build hangs at "Resolve Package Graph", iCloud has offloaded files in the checkout: Xcode's `git
status` and the compiler wait on files that never download (`find third_party -flags +dataless`). The
fix is to keep the splatx clone outside iCloud and symlink it:

    git clone https://github.com/PoSTMEDIA-AI/splatx-metal ~/Library/Caches/oscd-mobile/splatx-metal
    git -C ~/Library/Caches/oscd-mobile/splatx-metal checkout -b oscd-fusion 9ae5f66
    git -C ~/Library/Caches/oscd-mobile/splatx-metal am "$PWD/patches/splatx-oscd-fusion.patch"
    ln -s ~/Library/Caches/oscd-mobile/splatx-metal third_party/splatx-metal
