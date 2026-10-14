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

## Spike app
    cd SpikeApp && xcodegen generate && open OSCDSpike.xcodeproj
Capture tab: tap the table, Record, sweep, Stop → Documents/sessions/<date>/ (Files app).
Benchmark tab: times each stage on the device → Documents/benchmarks/bench_<date>.csv.
