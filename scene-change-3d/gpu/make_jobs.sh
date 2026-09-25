#!/usr/bin/env bash
# Write the job list: one "method instance scene stressor level" line per run.
#
#   WORK=... bash gpu/make_jobs.sh > "$WORK/jobs.txt"                  # full grid, both methods
#   PRESET=quick METHODS=mv3dcd INSTANCES=Instance_1 bash gpu/make_jobs.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHODS="${METHODS:-oscd mv3dcd}"
PRESET="${PRESET:-full}"
INSTANCES="${INSTANCES:-Instance_1 Instance_2}"
SCENES="${SCENES:-Cantina Garden Lounge Lunch_room Meeting_room Playground Porch Pots Printing_area Zen}"
source "$(conda info --base)/etc/profile.d/conda.sh"
LEVELS="$(cd "$HERE" && conda run -n scd python -m scene_change.harness.paslcd levels --preset "$PRESET")"
for m in $METHODS; do
  for i in $INSTANCES; do
    for s in $SCENES; do
      while read -r stressor level; do
        echo "$m $i $s $stressor $level"
      done <<< "$LEVELS"
    done
  done
done
