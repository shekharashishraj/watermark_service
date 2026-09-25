#!/usr/bin/env bash
# Severity curves, failure boundaries and calibration for everything scored so far.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${WORK:?set WORK}"
source "$(conda info --base)/etc/profile.d/conda.sh"
(cd "$HERE" && conda run -n scd python -m scene_change.harness.robustness_report --runs "$WORK/runs" \
   --out "$WORK/ROBUSTNESS_PASLCD.md" --figures "$WORK/figures")
echo "report: $WORK/ROBUSTNESS_PASLCD.md"
