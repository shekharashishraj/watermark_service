#!/usr/bin/env bash
# SLURM array over the job list; each task runs one line of $WORK/jobs.txt.
#
#   N=$(wc -l < "$WORK/jobs.txt")
#   sbatch --array=1-$N%8 --export=ALL,WORK=$WORK gpu/array.sbatch
#
# Adjust partition / QOS / GPU type to your cluster.
#SBATCH --job-name=scd-robust
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%A_%a.out
##SBATCH --partition=general
##SBATCH --qos=public
set -euo pipefail
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$WORK/jobs.txt")
bash "$SLURM_SUBMIT_DIR/gpu/run_job.sh" $LINE
