#!/bin/bash
# Copy fixtures from Sol into oscd_mobile/fixtures/.
# Usage: tools/fetch_fixtures.sh <user>@<sol-login-host> [remote_dir]
set -euo pipefail
REMOTE="${1:?usage: fetch_fixtures.sh user@host [remote_dir]}"
DIR="${2:-/scratch/\$USER/scd/fixtures/oscd_mobile}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$HERE/fixtures"
rsync -av --delete --exclude "_oscd_model" "$REMOTE:$DIR/" "$HERE/fixtures/"
