#!/bin/bash
# CPU array job: build dataset variants. Submitted by slurm/submit_experiment.sh
# (needs SCD_REPO, SCD_EXP, SCD_CHUNK in the environment).
#SBATCH -J scd-variants
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
set -eo pipefail
source "$SCD_REPO/setup/sol.env"
scd_activate scd-tools
python "$SCD_REPO/scripts/make_variants.py" --experiment "$SCD_EXP" \
    --index "$SLURM_ARRAY_TASK_ID" --chunk "${SCD_CHUNK:-1}" --workers "${SLURM_CPUS_PER_TASK:-8}"
