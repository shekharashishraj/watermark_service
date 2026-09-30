#!/bin/bash
# Create the three isolated prefix environments under $SCD_ENVS and build the CUDA extensions.
# Needs internet (login node, or an interactive session on a node with internet). No GPU needed:
# extensions are compiled for $TORCH_CUDA_ARCH_LIST. GPU checks happen in setup/05_smoke_test.sbatch.
#
#   bash setup/02_create_envs.sh all            # or: scd-tools | oscd | mv3dcd
#   FORCE=1 bash setup/02_create_envs.sh oscd   # delete and recreate
#   SCD_CUDA_MODULE=cuda-12.8.1-gcc-12.1.0 bash setup/02_create_envs.sh oscd
#       fallback: use a cluster CUDA module instead of the conda CUDA toolkit (see README)
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/sol.env"
command -v conda >/dev/null 2>&1 || module load "$SCD_MAMBA_MODULE"
eval "$(conda shell.bash hook)"
SOLVER=$(command -v mamba || command -v conda)
TARGET="${1:-all}"

make_env() {  # name yml
    local name="$1" yml="$2" prefix="$SCD_ENVS/$1"
    if [ -d "$prefix" ] && [ "${FORCE:-0}" = "1" ]; then rm -rf "$prefix"; fi
    if [ -d "$prefix" ]; then
        echo "[scd] $prefix exists (FORCE=1 to recreate); updating packages only"
        "$SOLVER" env update -p "$prefix" -f "$yml"
    else
        "$SOLVER" env create -p "$prefix" -f "$yml"
    fi
    conda activate "$prefix"
}

cuda_activation_hook() {  # make the conda CUDA toolkit visible to torch.utils.cpp_extension and CuPy
    mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
    cat > "$CONDA_PREFIX/etc/conda/activate.d/scd_cuda.sh" <<'EOF'
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$CONDA_PREFIX/lib"
EOF
    if [ -n "${SCD_CUDA_MODULE:-}" ]; then   # fallback: cluster CUDA instead of conda CUDA
        cat > "$CONDA_PREFIX/etc/conda/activate.d/scd_cuda.sh" <<EOF
module load $SCD_CUDA_MODULE
export CUDA_PATH="\$CUDA_HOME"
EOF
    fi
    conda deactivate; conda activate "$1"
    echo "[scd] nvcc: $(nvcc --version | tail -n 1)"
}

build_tools() {
    make_env scd-tools "$SCD_REPO/envs/scd-tools.yml"
    python -c "import numpy, cv2, yaml, scipy, pandas; print('[scd] scd-tools ok')"
    (cd "$SCD_REPO" && python -m pytest -q tests)
    conda deactivate
}

build_oscd() {
    make_env oscd "$SCD_REPO/envs/oscd.yml"
    cuda_activation_hook "$SCD_ENVS/oscd"
    # O-SCD README: torch/torchvision/xformers from the cu128 index (unpinned upstream; frozen below)
    pip install ${OSCD_TORCH_PKGS:-torch torchvision xformers} --index-url https://download.pytorch.org/whl/cu128
    pip install cupy-cuda12x
    # requirements.txt lists the local CUDA extensions; they need torch at build time -> no build isolation
    (cd "$SCD_ROOT/code/O-SCD" && pip install --no-build-isolation -r requirements.txt)
    pip freeze > "$SCD_ENVS/oscd.freeze.txt"
    python - <<'EOF'
import torch, cupy, transformers
import diff_gaussian_rasterization_fastgs, simple_knn, fused_ssim  # built extensions
from transformers import Sam2Model  # noqa: F401  (needs transformers>=4.56)
print("[scd] oscd ok: torch", torch.__version__, "cuda", torch.version.cuda, "| cupy", cupy.__version__,
      "| transformers", transformers.__version__)
EOF
    conda deactivate
}

build_mv3dcd() {
    make_env mv3dcd "$SCD_REPO/envs/mv3dcd.yml"
    cuda_activation_hook "$SCD_ENVS/mv3dcd"
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
    pip install plyfile opencv-python timm matplotlib scikit-learn torchmetrics tqdm
    pip install --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0"
    pip install --no-build-isolation "$SCD_ROOT/code/MV3DCD/submodules/simple-knn"
    pip freeze > "$SCD_ENVS/mv3dcd.freeze.txt"
    python - <<'EOF'
import torch, gsplat, simple_knn  # noqa: F401
print("[scd] mv3dcd ok: torch", torch.__version__, "cuda", torch.version.cuda, "| gsplat", gsplat.__version__)
EOF
    conda deactivate
}

case "$TARGET" in
    scd-tools) build_tools ;;
    oscd) build_oscd ;;
    mv3dcd) build_mv3dcd ;;
    all) build_tools; build_oscd; build_mv3dcd ;;
    *) echo "usage: $0 [all|scd-tools|oscd|mv3dcd]"; exit 2 ;;
esac
echo "[scd] done. Next: bash setup/03_fetch_data.sh"
