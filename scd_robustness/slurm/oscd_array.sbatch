#!/bin/bash
# GPU array job: one (or SCD_CHUNK) O-SCD run(s) per task. Submitted by slurm/submit_experiment.sh
# (needs SCD_REPO, SCD_MANIFEST, SCD_CHUNK in the environment).
#SBATCH -J scd-oscd
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
set -eo pipefail
source "$SCD_REPO/setup/sol.env"
scd_offline
scd_activate oscd
python "$SCD_REPO/scripts/run_oscd.py" --manifest "$SCD_MANIFEST" \
    --index "$SLURM_ARRAY_TASK_ID" --chunk "${SCD_CHUNK:-1}"
