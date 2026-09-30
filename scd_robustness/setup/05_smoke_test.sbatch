#!/bin/bash
# E0 smoke test on one GPU: runs O-SCD and MV3DCD on Garden/Instance_1 (clean) and evaluates both.
#   sbatch -p <partition> -q <qos> setup/05_smoke_test.sbatch     (partition/QOS names: see README)
#SBATCH -J scd-smoke
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o slurm-scd-smoke-%j.out
set -eo pipefail
source "${SCD_REPO:-$SLURM_SUBMIT_DIR}/setup/sol.env"
scd_offline
EXP="$SCD_REPO/configs/experiments/e0_smoke.yaml"
M="$SCD_ROOT/manifests/e0_smoke"
nvidia-smi

scd_activate scd-tools
python "$SCD_REPO/scripts/scd_plan.py" --experiment "$EXP"
python "$SCD_REPO/scripts/make_variants.py" --experiment "$EXP" --all
conda deactivate

scd_activate oscd
python - <<'EOF'
import cupy, torch
# the RANSAC kernel is compiled at runtime with NVRTC; check the headers it needs resolve
cupy.RawModule(code="#include <cuda/std/utility>\n#include <cuda_fp16.h>\nextern \"C\" __global__ void k(float* x){x[0]=1.f;}",
               options=("--std=c++17",)).get_function("k")
print("[scd] cupy NVRTC ok | torch sees", torch.cuda.get_device_name(0))
EOF
time python "$SCD_REPO/scripts/run_oscd.py" --manifest "$M/runs_oscd.jsonl" --index 0
conda deactivate

scd_activate mv3dcd
time python "$SCD_REPO/scripts/run_mv3dcd.py" --manifest "$M/runs_mv3dcd.jsonl" --index 0
conda deactivate

scd_activate scd-tools
python "$SCD_REPO/scripts/collect_results.py" --experiment "$EXP"
(column -s, -t < "$SCD_ROOT/results/e0_smoke/summary.csv" || cat "$SCD_ROOT/results/e0_smoke/summary.csv") | cut -c1-200
