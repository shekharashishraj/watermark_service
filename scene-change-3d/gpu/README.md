# Robustness study on PASLCD with the official implementations (GPU)

The CPU harness study (`ROBUSTNESS.md`) uses re-implementations on synthetic scenes. This
folder runs the same stressors on the real PASLCD dataset with the authors' own code:

| Method | Code | Mode | Continuous scores |
|---|---|---|---|
| O-SCD (CVPR 2026) | [Chumsy0725/O-SCD](https://github.com/Chumsy0725/O-SCD) | online; `--refine` adds the offline refinement | rendered change before the 0.5 threshold, saved by our patch |
| MV3DCD (CVPR 2025) | [Chumsy0725/MV3DCD](https://github.com/Chumsy0725/MV3DCD) | offline | change channel before binarisation, saved by our patch |

Our code never modifies the methods beyond two small, idempotent patches applied by
`python -m scene_change.harness.paslcd patch` (grep for `scd-robustness patch`):

- save the continuous change map next to each binary mask, for calibration;
- MV3DCD only: honour `SCD_HOLDOUT`, a list of post-change views left out of
  change-channel training (they are still rendered and scored).

Neither repository has a license file, and parts carry Inria's non-commercial research
header: fine for this course project, not for a product.

## Steps

```bash
cd scene-change-3d
export WORK=/scratch/$USER/scd                 # scratch space for data, outputs and results

bash gpu/setup.sh                              # conda envs, clone + patch, download PASLCD (both zips)
bash gpu/make_jobs.sh > "$WORK/jobs.txt"       # method x instance x scene x stressor x level

# smoke test: one scene, reference + one level
WORK=$WORK bash gpu/run_job.sh oscd Instance_1 Cantina none 0
WORK=$WORK bash gpu/run_job.sh oscd Instance_1 Cantina blur 0.04

# everything: SLURM array (8 at a time) or one machine
sbatch --array=1-$(wc -l < "$WORK/jobs.txt")%8 --export=ALL,WORK=$WORK gpu/array.sbatch
# or: bash gpu/run_all.sh

bash gpu/report.sh                             # -> $WORK/ROBUSTNESS_PASLCD.md + figures
```

`make_jobs.sh` takes `METHODS`, `INSTANCES`, `SCENES` and `PRESET=quick` (middle and most
severe level per stressor) to cut the grid. The full grid is 20 scene instances x 21
levels per method. O-SCD takes a few minutes per run; MV3DCD retrains 3DGS three times per
run, so start it with `PRESET=quick INSTANCES=Instance_1`.

## What each job does

1. `paslcd perturb` writes a degraded copy of one scene under `$WORK/variants/`. Only the
   post-change images change; everything else is symlinked. The stressors come from
   `scene_change/stress.py`:
   - motion blur: along the image motion between neighbouring frames (phase correlation),
     or in a random direction for MV3DCD's unordered views;
   - over- or underexposure in linear light;
   - lost coverage or fewer views: for O-SCD, frames are removed from the stream, but never
     frames with a ground-truth mask, so every level is scored on the same frames; for
     MV3DCD, views are held out of training instead (`holdout.txt`).

   `relight` needs the harness renderer and is not available on PASLCD.
2. The official method runs on the degraded copy (commands as in the authors' `run*.sh`).
3. `paslcd score` compares ground-truth masks with the binary outputs using the harness
   metrics: PASLCD frame IoU and F1, false alarms, and calibration statistics from the
   continuous maps (ECE, reliability, AP, threshold sweeps, confident errors). It writes one
   JSON line per method variant.

The report is the same one used for the harness. It covers:
- severity curves with 95% intervals over scenes
- failure boundaries
- calibration drift, including isotonic maps fitted on clean scenes and applied to degraded ones
- O-SCD online against its refined offline masks

## Notes

- PASLCD already has lighting differences between visits, so the exposure stressors add
  to a shift that is already there.
- Each job deletes its degraded copy and MV3DCD's checkpoints when it finishes
  (`KEEP=1` keeps them). Masks, scores and `variant.json` stay under `$WORK/output/`.
- If a run fails, `run_all.sh` records it in `$WORK/failed.txt`. Finished jobs are
  skipped on a rerun.
