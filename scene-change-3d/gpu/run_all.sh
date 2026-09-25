#!/usr/bin/env bash
# Run every job in $WORK/jobs.txt one after another on this machine, then write the report.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${WORK:?set WORK}"
while read -r line; do
  bash "$HERE/gpu/run_job.sh" $line || echo "FAILED: $line" | tee -a "$WORK/failed.txt"
done < "$WORK/jobs.txt"
bash "$HERE/gpu/report.sh"
