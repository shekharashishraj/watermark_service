# oscd_mobile

On-device port of O-SCD for a live tabletop change-detection demo on an iPhone Pro.
Design: `../docs/superpowers/specs/2026-10-14-oscd-mobile-design.md`.
O-SCD code reference: `docs/oscd_code_reference.md`.

## Layout
- `Packages/OSCDCore`: Swift package with all algorithms (iOS 17 / macOS 14)
- `tools/`: Python model conversion, Sol fixture dump, COLMAP export
- `SpikeApp/`: iOS app for recording sessions and on-device benchmarks
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
    tools/.venv/bin/python tools/convert_sam2_coreml.py      # → Models/SAM2_1Tiny_2x1024.mlpackage

If torch.hub fails with CERTIFICATE_VERIFY_FAILED (python.org Python without its certificates),
copy the weights the checkout already has into torch's cache:
`mkdir -p ~/.cache/torch/hub/checkpoints && cp third_party/accelerated_features/weights/xfeat.pt ~/.cache/torch/hub/checkpoints/`

## App ("3D Changes")
    cd SpikeApp && xcodegen generate && open OSCDSpike.xcodeproj
- Scans: every recorded scan (Documents/sessions/<date>/), newest first, named Scan 1, 2, … by capture order.
- Capture: tap the table, Record, sweep, Stop.
- Detect: choose the before-scan and the after-scan, then Run. Method A goes over every after-scan frame,
  then shows the change masks and each stage's time. Results are saved in
  `<after-scan>/detections/<before-scan>/` (summary.json + one JPEG per frame).
- Settings: Live view (Method A on the camera feed) and the stage benchmark.

The before-scan needs its 3D model on the phone. With the phone connected, build it on the Mac:

    tools/build_reference.sh <scan folder>

The script copies the scan off the phone, exports it, trains splatx (3,000 iterations), and copies
`reference.ply` into the scan's folder (about 1 minute; Python runs from a venv in ~/Library/Caches because iCloud offloads `tools/.venv`).

For a rehearsal without tapping, launch the app with a pair (scan folder names):

    xcrun devicectl device process launch --device <id> --terminate-existing com.veripheye.changes3d -- -detectBefore <before> -detectAfter <after>

## splatx-metal (reference rendering + O-SCD fusion)
The app renders the reference splat and runs O-SCD's change fusion with a patched splatx-metal:

    git clone https://github.com/PoSTMEDIA-AI/splatx-metal third_party/splatx-metal
    git -C third_party/splatx-metal checkout 9ae5f66
    git -C third_party/splatx-metal am ../../patches/splatx-oscd-fusion.patch

The patch adds `Trainer::backward_custom` (custom-loss backward on the host path) and
`splatx::oscd::ChangeFusion` (R_change, SSF loss, torch-style Adam, O-SCD densification, refine) with a
C API (`oscd_fusion_c.h`). `Packages/OSCDFusion` wraps it for Swift; its tests run on the Mac:

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
