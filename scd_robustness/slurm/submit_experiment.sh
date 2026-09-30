#!/bin/bash
# Plan an experiment and submit it as dependent SLURM array jobs:
#   variants (CPU) -> MV3DCD reference caches (GPU) -> O-SCD / MV3DCD runs (GPU)
#
#   bash slurm/submit_experiment.sh configs/experiments/e4_main.yaml
#
# Knobs (environment variables):
#   SCD_SBATCH_ARGS   extra sbatch flags for every job, e.g. "-p general -q public -A <account>"
#   SCD_GPU_ARGS      extra flags for GPU jobs, e.g. "--gres=gpu:a100:1" to pin a GPU type
#   SCD_MAX_PARALLEL  max simultaneously running tasks per array (default 16)
#   SCD_VARIANT_CHUNK variants per CPU task (default 10)
#   SCD_RUN_CHUNK     runs per GPU task (default 1; raise it to amortise start-up, and raise -t too)
#   DRY=1             print the sbatch commands instead of submitting
set -eo pipefail
[ -n "$1" ] || { echo "usage: $0 <experiment.yaml>"; exit 2; }
EXP="$(realpath "$1")"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/setup/sol.env"
scd_activate scd-tools

python "$SCD_REPO/scripts/scd_plan.py" --experiment "$EXP" >/dev/null
NAME=$(python -c "import sys; sys.path.insert(0, '$SCD_REPO'); from scd.config import load_experiment; print(load_experiment('$EXP')['name'])")
M="$SCD_ROOT/manifests/$NAME"
MAXP="${SCD_MAX_PARALLEL:-16}"
VCHUNK="${SCD_VARIANT_CHUNK:-10}"
RCHUNK="${SCD_RUN_CHUNK:-1}"
LOG="$SCD_ROOT/logs/slurm"

lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
tasks() { echo $(( ($1 + $2 - 1) / $2 )); }
submit() {  # prints the job id
    if [ "${DRY:-0}" = "1" ]; then echo "sbatch $*" >&2; echo "DRY"; else sbatch --parsable "$@"; fi
}
array_job() {  # manifest chunk dependency script jobname extra_export
    local manifest="$1" chunk="$2" dep="$3" script="$4" name="$5" n t gpu_args=""
    n=$(lines "$manifest"); [ "$n" -gt 0 ] || { echo ""; return; }
    t=$(tasks "$n" "$chunk")
    [ "$script" = "variants.sbatch" ] || gpu_args="${SCD_GPU_ARGS:-}"
    # shellcheck disable=SC2086
    submit ${SCD_SBATCH_ARGS:-} $gpu_args $dep \
        -J "$name" --array=0-$((t - 1))%"$MAXP" -o "$LOG/%x-%A_%a.out" \
        --export=ALL,SCD_REPO="$SCD_REPO",SCD_EXP="$EXP",SCD_MANIFEST="$manifest",SCD_CHUNK="$chunk" \
        "$SCD_REPO/slurm/$script"
}

echo "[scd] $NAME: $(lines "$M/variants.jsonl") variants, $(lines "$M/runs_oscd.jsonl") O-SCD runs," \
     "$(lines "$M/runs_mv3dcd.jsonl") MV3DCD runs, $(lines "$M/runs_mv3dcd_refcache.jsonl") reference caches"

JV=$(array_job "$M/variants.jsonl" "$VCHUNK" "" variants.sbatch "scd-var-$NAME")
DEP=${JV:+--dependency=afterok:$JV}
JR=$(array_job "$M/runs_mv3dcd_refcache.jsonl" 1 "$DEP" mv3dcd_array.sbatch "scd-ref-$NAME")
DEP_MV=${JR:+--dependency=afterok:${JV:+$JV:}$JR}
DEP_MV=${DEP_MV:-$DEP}
JO=$(array_job "$M/runs_oscd.jsonl" "$RCHUNK" "$DEP" oscd_array.sbatch "scd-oscd-$NAME")
JM=$(array_job "$M/runs_mv3dcd.jsonl" "$RCHUNK" "$DEP_MV" mv3dcd_array.sbatch "scd-mv-$NAME")

echo "[scd] submitted: variants=${JV:-none} refcache=${JR:-none} oscd=${JO:-none} mv3dcd=${JM:-none}"
echo "[scd] logs: $LOG    progress: squeue -u $USER"
echo "[scd] when finished: python scripts/collect_results.py --experiment $EXP"
echo "[scd] failed runs are re-submitted by running this script again (finished runs are skipped)."
