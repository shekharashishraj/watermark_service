#!/bin/bash
# GPU array job: one (or SCD_CHUNK) MV3DCD run(s) per task — reference-cache runs or variant runs.
# Submitted by slurm/submit_experiment.sh (needs SCD_REPO, SCD_MANIFEST, SCD_CHUNK in the environment).
#SBATCH -J scd-mv3dcd
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 01:30:00
set -eo pipefail
source "$SCD_REPO/setup/sol.env"
scd_offline
scd_activate mv3dcd
python "$SCD_REPO/scripts/run_mv3dcd.py" --manifest "$SCD_MANIFEST" \
    --index "$SLURM_ARRAY_TASK_ID" --chunk "${SCD_CHUNK:-1}"
