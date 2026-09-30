#!/bin/bash
# CPU job: statistics + figures for the whole study (scd-tools env). Safe to rerun at any time;
# it analyses whatever has finished and lists the rest. Submit from scd_robustness/:
#   sbatch -p <partition> -q <qos> -o $SCD_ROOT/logs/slurm/analysis-%j.out slurm/analyze.sbatch [config]
#SBATCH -J scd-analysis
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
set -eo pipefail
source "${SCD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}/setup/sol.env"
scd_activate scd-tools
CFG="${1:-$SCD_REPO/configs/analysis.yaml}"
python "$SCD_REPO/scripts/analyze.py" --config "$CFG"
python "$SCD_REPO/scripts/make_figures.py" --config "$CFG"
