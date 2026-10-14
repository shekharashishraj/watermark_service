# Spike report: O-SCD Mobile (day 3)

Device: not measured: pending human step (Task 16 Step 4) · iOS: pending · date 2026-10-14 · fixtures scene: pending Sol run (Task 4 Step 3) · sessions: pending device capture

Status: **partially filled.** Everything below that can be measured on the development Mac (Apple M4) is
filled in. Every cell that needs the iPhone, Sol, or the splat engines on device says "pending human step".
Fill those in from `Documents/benchmarks/bench_<date>.csv` and Task 17 Steps 3–4, then make the go/no-go.

## On-device timings (median ms, from Documents/benchmarks/bench_<date>.csv)
| Stage | Target (spec §11) | Measured | Pass |
|---|---|---|---|
| XFeat per frame | ≤ 15 | not measured: pending human step | – |
| Retrieval vs 80 keyframes | ≤ 60 | not measured: pending human step | – |
| P4P-RANSAC + mini-BA | ≤ 20 | not measured: pending human step | – |
| SAM2 encoder (pair) | ≤ 240 | not measured: pending human step | – |
| Pixel cue (SSIM) | n/a | not measured: pending human step | – |
| XFeat on all keyframes + anchor build | ≤ 30 s total | not measured on device; Mac (M4): 0.9 s anchor build for 24 synthetic keyframes | – |

Notes for reading the device numbers:
- XFeat ships as an **FP32** Core ML model, so on the iPhone it runs on the GPU rather than the Neural Engine. Every FP16 build tested gave wrong outputs on some compute unit (ledger, Task 5). If `xfeat_detect` misses its target, revisit FP16 with on-device correctness checks.
- SAM2.1-tiny ships as FP16. On the M4 it matches PyTorch to r ≥ 0.9996 on CPU, GPU and Neural Engine alike.

## Anchor yield (D11)
Measured with `oscd-harness anchors` on a **synthetic** tabletop session: 24 ARKit poses on a 0.6 m arc at 0.5 m height, rendered with the real session format. The real-session rows are pending the device capture.

| Depths | Yield (synthetic) | Anchors (synthetic) | Yield (real session) | Decision |
|---|---|---|---|---|
| (1, 10) m, O-SCD | 0.638 | 36,491 | pending human step | keep, unless the real-session yield is < 0.15 |
| (0.1, 1.0) m | 0.891 | 50,938 | pending human step | adopt only per the Task 17 decision rule |

On the CPU parity fixture (4 cameras on a 0.9 m arc), O-SCD's (1, 10) m gate kept 2,841 of 3,000 points (median parallax 174 px against the 45 px gate).

## Splat engines
| Check | splatx-metal | msplat-ios |
|---|---|---|
| 3k iters at 480×360 on iPhone (s) | pending human step | pending human step |
| Peak memory (MB) | pending human step | pending human step |
| Mac 3k iters (s) | pending human step | pending human step |
| 1 Custom loss hook (file:line) | pending | pending |
| 2 Per-parameter LR incl. xyz = 0 | pending | pending |
| 3 Expected-depth render | pending | pending |
| 4 External densify/prune/reset | pending | pending |
| 5 Rasterizer constants match O-SCD | pending | pending |
| 6 SH degree 0 | pending | pending |
| 7 Init from our points | pending | pending |

## Verified on the Mac (no device or Sol needed)
| Component | Reference it was checked against | Result |
|---|---|---|
| XFeat post-processing + Core ML detector | XFeat's own PyTorch `detectAndCompute` on a real 480×360 photo | both parity tests pass |
| Mini-BA | O-SCD `poses/mini_ba.py` MiniBAInternal on CPU | parity test passes |
| P4P-RANSAC | O-SCD P4P path on CPU (injected samples) | passes; contenders 829/832 within 0.05°, best score identical |
| Triangulation | O-SCD `matches_to_points` on CPU | parity test passes |
| Pixel + feature cues | O-SCD `generate_candidate_map` + `ssim` on CPU with real SAM2.1-tiny | parity test passes |
| SAM2 Core ML | HF SAM2.1-tiny (O-SCD's input path) | r = 0.9999 |
| Full pose pipeline | synthetic 5-view anchors, held-out view | accepted; 0.0°, 0.065 mm error |
| ARKit conventions + session format + COLMAP export | synthetic rendered session | yield 0.638; LiDAR points on the table plane; table centre projects to the principal point at the expected depth |

Still to verify against the real O-SCD on Sol: everything above, at fp16 CUDA on real PASLCD data, plus F-RANSAC (a CUDA-only kernel) and the end-to-end pose and anchor parity.

## Decisions
- Engine: pending human step (Task 17 Step 4).
- Reference build: pending (depends on the engine's on-device timing).
- D9 (fewer keypoints or lower resolution): pending device timings.
- D11 (disambiguation depths): provisionally O-SCD (1, 10). The synthetic yield of 0.638 is far above the 0.15 switch threshold; confirm on a real session.
- Go / no-go for Phase 2: pending the human steps above.
