#!/bin/bash
# Download every model the methods fetch at runtime, so jobs can run with no internet access.
# Run on a login node:   bash setup/04_prestage_models.sh
#   O-SCD : SAM2.1-Hiera-tiny (Hugging Face), XFeat (torch.hub verlab/accelerated_features)
#   MV3DCD: DINOv2 ViT-L/14 (torch.hub facebookresearch/dinov2)
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/sol.env"

clone() {  # url dir [ref]
    [ -d "$2/.git" ] || git clone "$1" "$2"
    if [ -n "$3" ]; then git -C "$2" checkout --quiet "$3"; fi
    echo "[scd] $2 @ $(git -C "$2" rev-parse --short HEAD)"
}
# Pin with SCD_XFEAT_REF / SCD_DINOV2_REF (commit or tag) if a newer upstream breaks Python 3.8 (MV3DCD env).
clone https://github.com/verlab/accelerated_features "$SCD_XFEAT_REPO" "$SCD_XFEAT_REF"
clone https://github.com/facebookresearch/dinov2 "$SCD_DINOV2_REPO" "$SCD_DINOV2_REF"

scd_activate oscd
python - <<'EOF'
import os, torch
from transformers import Sam2Model
Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")          # -> $HF_HOME
torch.hub.load(os.environ["SCD_XFEAT_REPO"], "XFeat", source="local", pretrained=True, top_k=4096)  # weights -> $TORCH_HOME
print("[scd] O-SCD models cached")
EOF
conda deactivate

scd_activate mv3dcd
python - <<'EOF'
import os, torch
torch.hub.load(os.environ["SCD_DINOV2_REPO"], "dinov2_vitl14", source="local", pretrained=True)  # weights -> $TORCH_HOME
print("[scd] MV3DCD models cached")
EOF
conda deactivate
echo "[scd] done. Next: sbatch setup/05_smoke_test.sbatch (see README for partition/QOS flags)"
