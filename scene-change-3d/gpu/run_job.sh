#!/usr/bin/env bash
# Run one job: degrade the scene, run the official method on it, score the output.
#
#   WORK=... bash gpu/run_job.sh oscd Instance_1 Cantina blur 0.04
#
# Results go to $WORK/runs/results_<method>_<instance>_<scene>_<stressor>_<level>.jsonl (one file
# per job, so array jobs never write to the same file). Finished jobs are skipped.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${WORK:?set WORK}"
METHOD=$1 INSTANCE=$2 SCENE=$3 STRESSOR=$4 LEVEL=$5
TAG="${METHOD}_${INSTANCE}_${SCENE}_${STRESSOR}_${LEVEL}"
RES="$WORK/runs/results_${TAG}.jsonl"
[ -s "$RES" ] && { echo "done: $TAG"; exit 0; }
source "$(conda info --base)/etc/profile.d/conda.sh"
VAR="$WORK/variants/$METHOD/${STRESSOR}_${LEVEL}/$INSTANCE/$SCENE"
OUT="$WORK/output/$METHOD/${STRESSOR}_${LEVEL}/$INSTANCE/$SCENE"
score() {  # pred-dir score-dir method-name
  (cd "$HERE" && conda run -n scd python -m scene_change.harness.paslcd score --gt "$VAR/gt_mask" \
     --pred "$1" --scores "$2" --method "$3" --scene "$INSTANCE/$SCENE" --stressor "$STRESSOR" --level "$LEVEL" \
     --out "$RES.tmp")
}
rm -f "$RES.tmp"

if [ "$METHOD" = oscd ]; then
  SRC="$WORK/data/online/PASLCD/$INSTANCE/$SCENE"
  (cd "$HERE" && conda run -n scd python -m scene_change.harness.paslcd perturb --layout oscd --scene "$SRC" \
     --out "$VAR" --stressor "$STRESSOR" --level "$LEVEL")
  conda activate oscd
  (cd "$WORK/repos/O-SCD" && python oscd.py -s "$VAR/" -m "$OUT/" --resolution 4 --test_hold 5 --refine)
  conda deactivate
  score "$OUT/renders/change_mask" "$OUT/renders/change_score" oscd-official-online
  score "$OUT/renders/change_mask_refined" "$OUT/renders/change_score_refined" oscd-official-offline

elif [ "$METHOD" = mv3dcd ]; then
  SRC="$WORK/data/offline/PASLCD/$SCENE/$INSTANCE"
  # removed post-change views stay on disk (they are in the COLMAP model); the patched
  # train_masks.py leaves them out of training, and they are still rendered and scored
  (cd "$HERE" && conda run -n scd python -m scene_change.harness.paslcd perturb --layout mv3dcd --scene "$SRC" \
     --out "$VAR" --stressor "$STRESSOR" --level "$LEVEL" --allow-gt-drop)
  export SCD_HOLDOUT="$(cat "$VAR/holdout.txt" 2>/dev/null || true)"
  R=8; T=0.5
  conda activate mv3dcd
  cd "$WORK/repos/MV3DCD"
  python train.py -s "$VAR" -m "$OUT" --iterations 7000 --change "$INSTANCE" --resolution $R \
    --checkpoint_iterations 7000 --save_iterations 7000
  python render_viewpoints.py -s "$VAR" -m "$OUT" --iterations 7000 --change "$INSTANCE" --resolution $R
  python create_mask.py --t $T --input_folder "$OUT"
  python train_masks.py -s "$VAR" -m "$OUT" --iterations 10000 --change "$INSTANCE" --checkpoint_iterations 10000 \
    --start_checkpoint "$OUT/chkpnt7000.pth" --resolution $R
  python render_viewpoints.py -s "$VAR" -m "$OUT" --iterations 10000 --change "$INSTANCE" --resolution $R --aug True
  python create_mask.py --t $T --input_folder "$OUT"
  python train_masks.py -s "$VAR" -m "$OUT" --iterations 13000 --change "$INSTANCE" --checkpoint_iterations 13000 \
    --start_checkpoint "$OUT/chkpnt10000.pth" --resolution $R --augment True
  python render_viewpoints.py -s "$VAR" -m "$OUT" --iterations 13000 --change "$INSTANCE" --resolution $R --mask True
  cd - >/dev/null
  conda deactivate
  score "$OUT/renders/binary_masks" "$OUT/renders/score_masks" mv3dcd-official
else
  echo "unknown method $METHOD" >&2; exit 2
fi
mv "$RES.tmp" "$RES"
# degraded copies and checkpoints are large; keep only masks, scores and the variant description
if [ "${KEEP:-0}" != 1 ]; then
  cp "$VAR/variant.json" "$OUT/" 2>/dev/null || true
  rm -rf "$VAR"
  rm -f "$OUT"/chkpnt*.pth
fi
echo "finished: $TAG"
