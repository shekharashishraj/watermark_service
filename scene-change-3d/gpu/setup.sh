#!/usr/bin/env bash
# One-time setup on the GPU machine: environments, official code (patched), PASLCD.
#
#   WORK=/scratch/$USER/scd bash gpu/setup.sh            # both methods
#   METHODS="oscd" WORK=... bash gpu/setup.sh            # O-SCD only
#
# Needs conda and a CUDA 12.x driver. Environments follow the official READMEs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"      # scene-change-3d
WORK="${WORK:?set WORK to a scratch directory}"
METHODS="${METHODS:-oscd mv3dcd}"
mkdir -p "$WORK"/{repos,data,variants,output,runs}
source "$(conda info --base)/etc/profile.d/conda.sh"

# harness tools (perturb / score / report): numpy, scipy, opencv
if ! conda env list | grep -q "^scd "; then
  conda create -y -n scd python=3.11
  conda run -n scd pip install -r "$HERE/requirements.txt" matplotlib "huggingface_hub[cli]"
fi

if [[ " $METHODS " == *" oscd "* ]]; then
  [ -d "$WORK/repos/O-SCD" ] || git clone --recursive https://github.com/Chumsy0725/O-SCD "$WORK/repos/O-SCD"
  if ! conda env list | grep -q "^oscd "; then
    conda create -y -n oscd python=3.12
    conda run -n oscd pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu128
    conda run -n oscd pip install cupy-cuda12x
    (cd "$WORK/repos/O-SCD" && conda run -n oscd pip install -r requirements.txt)
  fi
  (cd "$HERE" && conda run -n scd python -m scene_change.harness.paslcd patch --method oscd --repo "$WORK/repos/O-SCD")
  if [ ! -d "$WORK/data/online/PASLCD" ]; then
    conda run -n scd huggingface-cli download ChamudithaJay/PASLCD PASLCD_online.zip --repo-type dataset \
      --local-dir "$WORK/data/online"
    unzip -q "$WORK/data/online/PASLCD_online.zip" -d "$WORK/data/online"
  fi
fi

if [[ " $METHODS " == *" mv3dcd "* ]]; then
  [ -d "$WORK/repos/MV3DCD" ] || git clone --recursive https://github.com/Chumsy0725/MV3DCD "$WORK/repos/MV3DCD"
  if ! conda env list | grep -q "^mv3dcd "; then
    conda create -y -n mv3dcd python=3.8
    conda run -n mv3dcd pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
      --index-url https://download.pytorch.org/whl/cu124
    conda run -n mv3dcd pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0
    conda run -n mv3dcd pip install plyfile opencv-python timm matplotlib scikit-learn torchmetrics
    (cd "$WORK/repos/MV3DCD" && conda run -n mv3dcd pip install submodules/simple-knn)
  fi
  (cd "$HERE" && conda run -n scd python -m scene_change.harness.paslcd patch --method mv3dcd --repo "$WORK/repos/MV3DCD")
  if [ ! -d "$WORK/data/offline/PASLCD" ]; then
    conda run -n scd huggingface-cli download ChamudithaJay/PASLCD PASLCD.zip --repo-type dataset \
      --local-dir "$WORK/data/offline"
    unzip -q "$WORK/data/offline/PASLCD.zip" -d "$WORK/data/offline"
  fi
fi
echo "setup done in $WORK"
