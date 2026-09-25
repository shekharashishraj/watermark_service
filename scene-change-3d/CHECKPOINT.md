# Checkpoint: robustness study (25 Sep 2026, 22:00 UTC; all jobs stopped)

Branch `claude/elegant-shannon-gjj5hd`, folder `scene-change-3d/`. The submitted proposal
("Robustness in 3D Scene Change Detection") asks how 3D change detectors hold up under
capture degradation and whether their scores say when they are wrong. The system from
the applied proposal (baseline, localization, detection, report, benchmark) was already
built. This checkpoint covers the robustness and calibration study built on top of it.

Findings so far: `docs/robustness/findings.md` (also the top of `ROBUSTNESS.md`).
Overview page (private): https://claude.ai/artifact/N8soP8ZxWd3ggseDoVjDzC (version 5, interim).

## Done

| Piece | Where | State |
|---|---|---|
| Stressors: blur, under- and overexposure, relighting, depth noise, lost coverage, fewer views, map compression | `scene_change/stress.py` | done, tested |
| Calibration metrics: ECE, reliability, AP/AUROC, threshold sweeps, confident errors, isotonic recalibration | `scene_change/calibration.py` | done, tested |
| Sweep runner (resumable) and report generator | `scene_change/harness/robustness.py`, `robustness_report.py` | done |
| GPU kit: official O-SCD and MV3DCD on PASLCD, same stressors and scoring | `gpu/`, `scene_change/harness/paslcd.py` | done; tested on mock data only, never run on a GPU |
| Sweeps: ours, ours confirmed-only, 2D (5 held-out scenes x 26 levels) | `results/robustness/results_fast.jsonl.gz` | **complete** (390 runs) |
| Sweeps: depth noise, map compression (ours) | `results_depth`, `results_compress` | **complete** (80 runs) |
| Sweeps: O-SCD online and refined (5 scenes x 30 levels) | `results_oscd`, `results_oscd2` | **96 of 150 levels**, stopped (scenes 0 and 3 complete, scene 1 at 26 of 30, scene 4 at 10 of 30, scene 2 not started) |
| Depth safeguard (no all-clear from noisy depth), cut-off set on dev scenes | `detect.py` (`min_depth_agreement`) | done; held-out: wrong all-clears 6 -> 0, no false triggers (`docs/robustness/depth_gate.json`) |
| Exposure safeguard (no restyle flood from bad exposure) | `detect.py` (`app_exposure_gate`) | done; held-out: -4 EV F1 0.18 -> 0.56, false alarms 39% -> 0.4% (`docs/robustness/exposure_gate.json`) |
| Benchmark re-run with both safeguards | `results/robustness/benchmark_current_code.jsonl.gz`, note in `RESULTS.md` | same scores to 0.001 |
| Interim report | `ROBUSTNESS.md`, figures in `docs/robustness/` | generated from 638 runs |
| Tests | `tests/` | 21 pass (`pytest`) |

Raw results in `results/robustness/` (gzipped JSON lines; one line per run and method
variant):

| File | Contents |
|---|---|
| `results_fast` | ours, ours-confirmed and 2D, 5 scenes x 26 levels |
| `results_oscd`, `results_oscd2` | O-SCD online and refined (two processes: scenes 0-2 and 3-4), 96 of 150 levels |
| `results_depth`, `results_compress` | ours under depth noise and map compression |
| `results_ref` | ours at the reference, with self-reported depth agreement |
| `robust_gate_*` | held-out depth-safeguard evaluation (mixed and change-free scenes) |
| `robust_expo_*` | held-out exposure-safeguard evaluation (safeguard on) |
| `devrobust_*` | development scenes under depth noise (sets the depth cut-off) |
| `benchmark_current_code` | the 15-scene benchmark with both safeguards |

## Headline numbers

Reference, held-out scenes, every second frame (frame F1 / false-alarm pixels):
ours 0.635 / 0.49%, ours confirmed-only 0.555 / 0.07%, 2D 0.231 / 1.65%,
O-SCD online 0.239 / 23.1% and refined 0.271 / 27.6% (O-SCD interim).

Failure boundaries (first significant 20% F1 drop):

| Stressor | Ours | Ours, confirmed | O-SCD (interim) | 2D |
|---|---|---|---|---|
| Motion blur | 10% | none | 7% (no output) | 1% |
| Underexposure | -3 EV | none | -4 EV | -2 EV |
| Overexposure | none | none | none | none |
| Lighting change | none | none | none | 2x |
| Depth noise | x1 | x1 | n/a | n/a |
| Lost coverage | none | none | none | none |
| Fewer views | 50% removed | 50% removed | none | none |
| Map compression | 10 cm | 10 cm | none | n/a |

## Resume

```bash
cd scene-change-3d
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

# 1. restore the raw results the runner resumes from
mkdir -p runs/robust
for f in results/robustness/results_*.jsonl.gz; do gunzip -c "$f" > runs/robust/$(basename "${f%.gz}"); done

# 2. SAM 2.1 encoder for O-SCD (see README for the source); scenarios regenerate on their own
export OSCD_SAM2_ONNX=~/models/sam2.1_hiera_tiny.encoder.onnx

# 3. finish O-SCD: both commands skip finished levels (about 7 min per level each on 4 CPU cores)
OSCD_THREADS=2 OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 python -m scene_change.harness.robustness \
    --bench runs/bench --out runs/robust --seeds 0 1 2 --methods oscd --tag _oscd --feature-cache &
OSCD_THREADS=2 OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 python -m scene_change.harness.robustness \
    --bench runs/bench --out runs/robust --seeds 3 4 --methods oscd --tag _oscd2 --feature-cache &

# 4. final report: first update docs/robustness/findings.md (drop the interim note, final O-SCD numbers)
python -m scene_change.harness.robustness_report --runs runs/robust --out ROBUSTNESS.md \
    --figures docs/robustness --preamble docs/robustness/findings.md

# 5. overview page: docs/overview/build.py still points at the old session's scratch folder
#    (SP, REPO); set them, regenerate dev scenes s100/s101 if missing
#    (python -m scene_change generate --seed 100 --kind mixed --out <SP>/dev/s100_mixed, ...),
#    build, then republish to the same artifact URL.
```

All jobs were stopped at this checkpoint; nothing is running. After step 3 finishes,
gzip `runs/robust/results_oscd*.jsonl` into `results/robustness/` again and commit.

## Next

1. Finish the O-SCD sweep (54 levels left, about 3.5 hours on 4 CPU cores), then finalise
   `findings.md`, `ROBUSTNESS.md` and the overview page.
2. Check the online error-accumulation reading: O-SCD with true poses under blur and a
   longer walk, to separate localisation failure from change-cue failure.
3. On a GPU machine: `gpu/README.md` runs the authors' O-SCD and MV3DCD on PASLCD under
   the same stressors (the proposal's reproduction step). This session cannot reach a GPU
   over SSH. Run `claude remote-control` on the GPU machine to drive it from the Claude
   Code app.
4. Re-set the two safeguard thresholds on real captures before relying on them. They
   were set on synthetic sensor noise.
