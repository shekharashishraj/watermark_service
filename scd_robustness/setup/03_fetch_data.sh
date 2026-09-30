#!/bin/bash
# Download and unpack both PASLCD zips (login node, needs internet), then record the dataset facts.
#   bash setup/03_fetch_data.sh
# Read the dataset card (license) first: https://huggingface.co/datasets/ChamudithaJay/PASLCD
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/sol.env"
RAW="$SCD_ROOT/data/raw"
HF=https://huggingface.co/datasets/ChamudithaJay/PASLCD/resolve/main

get() {  # file dest_dir
    local file="$1" dest="$2"
    mkdir -p "$dest"
    [ -f "$RAW/$file" ] || wget -c -O "$RAW/$file" "$HF/$file"
    [ -f "$dest/.unzipped" ] || { unzip -q -o "$RAW/$file" -d "$dest" && touch "$dest/.unzipped"; }
}

get PASLCD_online.zip "$RAW/online"     # O-SCD layout: PASLCD/Instance_X/<Scene>/
get PASLCD.zip        "$RAW/offline"    # MV3DCD layout: PASLCD/<Scene>/Instance_X/
(cd "$RAW" && sha256sum PASLCD_online.zip PASLCD.zip | tee checksums.sha256)

scd_activate scd-tools
python "$SCD_REPO/scripts/check_dataset.py" --out "$SCD_ROOT/results/dataset_facts.json" | tail -n 25
echo "[scd] full report: $SCD_ROOT/results/dataset_facts.json  (copy the answers into docs/dataset_facts.md)"
