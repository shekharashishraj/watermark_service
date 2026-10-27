#!/bin/sh
# Build a scan's 3D model on the Mac and put it on the phone, so the scan can be the before-scan.
#
# Usage: tools/build_reference.sh <scan folder> [device id]
#   <scan folder>  the folder name under Documents/sessions on the phone (the scan's detail screen shows it)
#   device id      default: $OSCD_DEVICE, else the only connected iPhone
#
# Environment:
#   TRAINER=brush (default) | splatx
#   STEPS=7000        Brush steps (dinovi's "Good" tier is 7,000; 30,000 is Brush's default)
#   MAX_RES=1280      Brush training image size (the longer side)
#   BRUSH=<path>      the Brush binary; default: ~/Library/Caches/oscd-mobile/brush/brush_app (the v0.3.0 macOS release,
#                     https://github.com/ArthurBrussee/brush/releases), else the remus checkout's release build
#
# Steps: copy the scan off the phone → COLMAP export at full capture resolution (tools/session_to_colmap.py
# --full) → Brush (adaptive density control + mip-splatting: the trainer that gives scanner apps their clean
# walls; ~2 min per 1,000 steps on an M4 for a 130-frame room) → copy reference.ply into the scan.
# TRAINER=splatx keeps the old path (8000 iterations, ≤ 400k Gaussians, 480×360).
set -eu

SCAN=${1:?usage: tools/build_reference.sh <scan folder> [device id]}
HERE=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
APP=com.veripheye.changes3d
DEVICE=${2:-${OSCD_DEVICE:-}}
if [ -z "$DEVICE" ]; then
    DEVICE=$(xcrun devicectl list devices 2>/dev/null | awk '/iPhone/ && /connected|available/ {print $3; exit}')
fi
[ -n "$DEVICE" ] || { echo "error: no iPhone found; pass the device id or set OSCD_DEVICE" >&2; exit 1; }

TRAINER=${TRAINER:-brush}
STEPS=${STEPS:-7000}
MAX_RES=${MAX_RES:-1280}
CACHE="$HOME/Library/Caches/oscd-mobile"
WORK="$CACHE/references/$SCAN"
SPLATX="$CACHE/splatx-build/release/splatx"
METALLIB="$HERE/third_party/splatx-metal/build/splatx.metallib"
BRUSH=${BRUSH:-}
if [ -z "$BRUSH" ]; then
    for c in "$CACHE/brush/brush_app" "$HOME/projects/remus/engine/brush/target/release/brush"; do
        [ -x "$c" ] && BRUSH=$c && break
    done
fi
# numpy + Pillow in a venv outside iCloud (iCloud offloads tools/.venv, and its Python then hangs).
PY="$CACHE/tools-venv/bin/python"
if [ ! -x "$PY" ]; then
    python3 -m venv "$CACHE/tools-venv" && "$CACHE/tools-venv/bin/pip" install -q numpy pillow
fi

if [ "$TRAINER" = brush ]; then
    [ -n "$BRUSH" ] && [ -x "$BRUSH" ] || {
        echo "error: no Brush binary. Build it: cd ~/projects/remus/engine/brush && cargo build --release -p brush-app" >&2
        echo "       or download brush-app-aarch64-apple-darwin from https://github.com/ArthurBrussee/brush/releases into $CACHE/brush/" >&2
        exit 1
    }
else
    if [ ! -x "$SPLATX" ]; then
        echo "Building the splatx command-line trainer…"
        swift build -c release --product splatx --package-path "$HERE/third_party/splatx-metal" --scratch-path "$CACHE/splatx-build"
    fi
    [ -f "$METALLIB" ] || "$HERE/third_party/splatx-metal/tools/build_metallib.sh"
fi

rm -rf "$WORK"; mkdir -p "$WORK"
echo "1/4 Copying $SCAN off the phone…"
xcrun devicectl device copy from --device "$DEVICE" --domain-type appDataContainer --domain-identifier "$APP" \
    --source "Documents/sessions/$SCAN" --destination "$WORK/session" >/dev/null
if [ "$TRAINER" = brush ]; then
    echo "2/4 Exporting it for training (full resolution)…"
    "$PY" "$HERE/tools/session_to_colmap.py" --full "$WORK/session" "$WORK/colmap" > "$WORK/export.log" 2>&1 || { tail -20 "$WORK/export.log" >&2; exit 1; }
    N_PTS=$(wc -l < "$WORK/colmap/sparse/0/points3D.txt" | tr -d ' ')
    SUB=$(( N_PTS / 150000 )); [ "$SUB" -lt 1 ] && SUB=1          # Brush grows from ~150k points; a denser start only slows it
    echo "3/4 Training with Brush ($STEPS steps, $MAX_RES px, about $(( STEPS / 500 )) minutes)…"
    mkdir -p "$WORK/brush"
    "$BRUSH" "$WORK/colmap" --total-steps "$STEPS" --export-every "$STEPS" --export-path "$WORK/brush" \
        --sh-degree 2 --max-resolution "$MAX_RES" --subsample-points "$SUB" \
        > "$WORK/train.log" 2>&1 || { tail -20 "$WORK/train.log" >&2; exit 1; }
    PLY=$(ls -t "$WORK"/brush/*.ply 2>/dev/null | head -1)
    [ -n "$PLY" ] || { echo "error: Brush wrote no model (see $WORK/train.log)" >&2; exit 1; }
    cp "$PLY" "$WORK/reference.ply"
else
    echo "2/4 Exporting it for training…"
    "$PY" "$HERE/tools/session_to_colmap.py" "$WORK/session" "$WORK/colmap" > "$WORK/export.log" 2>&1 || { tail -20 "$WORK/export.log" >&2; exit 1; }
    echo "3/4 Training the 3D model with splatx (about 2 minutes)…"
    "$SPLATX" train --data "$WORK/colmap" --factor 1 --iters 8000 --max-points 400000 --metallib "$METALLIB" \
        --out "$WORK/reference.ply" > "$WORK/train.log" 2>&1 || { tail -20 "$WORK/train.log" >&2; exit 1; }
fi
echo "4/4 Copying the model to the phone…"
xcrun devicectl device copy to --device "$DEVICE" --domain-type appDataContainer --domain-identifier "$APP" \
    --source "$WORK/reference.ply" --destination "Documents/sessions/$SCAN/reference.ply" >/dev/null
echo "Done. In the app, pull Scans to refresh: $SCAN shows \"3D model ready\". Open View 3D model (Walk)."
