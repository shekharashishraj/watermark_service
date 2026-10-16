#!/bin/sh
# Build a scan's 3D model on the Mac and put it on the phone, so the scan can be the before-scan.
#
# Usage: tools/build_reference.sh <scan folder> [device id]
#   <scan folder>  the folder name under Documents/sessions on the phone (the scan's detail screen shows it)
#   device id      default: $OSCD_DEVICE, else the only connected iPhone
#
# Steps: copy the scan off the phone → COLMAP export (tools/session_to_colmap.py) → splatx train
# (3000 iterations, ≤ 200k Gaussians, as in docs/spike_report.md) → copy reference.ply into the scan.
set -eu

SCAN=${1:?usage: tools/build_reference.sh <scan folder> [device id]}
HERE=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
APP=com.veripheye.changes3d
DEVICE=${2:-${OSCD_DEVICE:-}}
if [ -z "$DEVICE" ]; then
    DEVICE=$(xcrun devicectl list devices 2>/dev/null | awk '/iPhone/ && /connected|available/ {print $3; exit}')
fi
[ -n "$DEVICE" ] || { echo "error: no iPhone found; pass the device id or set OSCD_DEVICE" >&2; exit 1; }

CACHE="$HOME/Library/Caches/oscd-mobile"
WORK="$CACHE/references/$SCAN"
SPLATX="$CACHE/splatx-build/release/splatx"
METALLIB="$HERE/third_party/splatx-metal/build/splatx.metallib"
# numpy + Pillow in a venv outside iCloud (iCloud offloads tools/.venv, and its Python then hangs).
PY="$CACHE/tools-venv/bin/python"
if [ ! -x "$PY" ]; then
    python3 -m venv "$CACHE/tools-venv" && "$CACHE/tools-venv/bin/pip" install -q numpy pillow
fi

if [ ! -x "$SPLATX" ]; then
    echo "Building the splatx command-line trainer…"
    swift build -c release --product splatx --package-path "$HERE/third_party/splatx-metal" --scratch-path "$CACHE/splatx-build"
fi
[ -f "$METALLIB" ] || "$HERE/third_party/splatx-metal/tools/build_metallib.sh"

rm -rf "$WORK"; mkdir -p "$WORK"
echo "1/4 Copying $SCAN off the phone…"
xcrun devicectl device copy from --device "$DEVICE" --domain-type appDataContainer --domain-identifier "$APP" \
    --source "Documents/sessions/$SCAN" --destination "$WORK/session" >/dev/null
echo "2/4 Exporting it for training…"
"$PY" "$HERE/tools/session_to_colmap.py" "$WORK/session" "$WORK/colmap" > "$WORK/export.log" 2>&1 || { tail -20 "$WORK/export.log" >&2; exit 1; }
echo "3/4 Training the 3D model (about a minute)…"
"$SPLATX" train --data "$WORK/colmap" --factor 1 --iters 3000 --max-points 200000 --metallib "$METALLIB" \
    --out "$WORK/reference.ply" > "$WORK/train.log" 2>&1 || { tail -20 "$WORK/train.log" >&2; exit 1; }
echo "4/4 Copying the model to the phone…"
xcrun devicectl device copy to --device "$DEVICE" --domain-type appDataContainer --domain-identifier "$APP" \
    --source "$WORK/reference.ply" --destination "Documents/sessions/$SCAN/reference.ply" >/dev/null
echo "Done. In the app, pull Scans to refresh: $SCAN shows \"3D model ready\"."
