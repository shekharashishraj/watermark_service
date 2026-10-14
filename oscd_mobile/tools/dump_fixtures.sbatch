#!/bin/bash
# Golden fixtures for oscd_mobile. Submit from oscd_mobile/tools on Sol:
#   SCD_REPO=/path/to/scd_robustness sbatch -p <partition> -q <qos> dump_fixtures.sbatch
#SBATCH -J oscd-fixtures
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o slurm-oscd-fixtures-%j.out
set -eo pipefail
: "${SCD_REPO:?set SCD_REPO to the scd_robustness checkout on Sol}"
source "$SCD_REPO/setup/sol.env"
scd_offline
scd_activate oscd
SCENE="${SCENE:-Garden}"
OUT="${OUT:-$SCD_ROOT/fixtures/oscd_mobile}"
python "$SLURM_SUBMIT_DIR/dump_oscd_fixtures.py" \
  --oscd-root "$SCD_ROOT/code/O-SCD" \
  --instance "$SCD_ROOT/data/raw/online/PASLCD/Instance_1/$SCENE" \
  --out "$OUT" --frames 3
du -sh "$OUT"
