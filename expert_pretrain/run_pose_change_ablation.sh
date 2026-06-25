#!/usr/bin/env bash
# Root-current vs self-pose-change multi-expert ablation.
#
# This writes a fresh unified dataset with both target families, trains both expert
# variants with identical hyperparameters, then compares all-expert full-pose
# outputs using MobilePoser-style FK metrics.
set -euo pipefail

PY=${PY:-python}
ED=${ED:-expert_pretrain}
DATA=${DATA:-data60hz}
W=${W:-weights}
L=${L:-logs}

cd "$ED"
mkdir -p "$W" "$L"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(stamp)] [0/5] build unified data with self pose-change targets -> $DATA"
$PY build_unified_pose_data.py \
  --output "$DATA" \
  --target-fps 60 \
  --sources mobileposer,dynaip,vidimu,ultra,ceti \
  > "$L/posechange_build.log" 2>&1

echo "[$(stamp)] [1/5] make leak-free sequence splits"
$PY make_splits.py --data "$DATA" --out "$DATA/splits.json" \
  > "$L/posechange_splits.log" 2>&1

COMMON=(
  --data "$DATA"
  --manifest "$DATA/splits.json"
  --epochs 100
  --patience 15
  --loss per_joint
  --hidden 64
  --layers 4
  --loss-balance adaptive
  --xyz-weight 2.0
  --acc-noise 0.2
  --ori-noise-deg 8
  --preload
  --num-workers 6
)

echo "[$(stamp)] [2/5] train root-current baseline"
$PY train_multiexpert.py "${COMMON[@]}" \
  --target joint_orient_r6d,joint_delta \
  --save "$W/multiexpert_root_current_ablate.pt" \
  > "$L/multiexpert_root_current_ablate.log" 2>&1

echo "[$(stamp)] [3/5] train self pose-change model"
$PY train_multiexpert.py "${COMMON[@]}" \
  --target joint_rot_delta_r6d,joint_delta_local \
  --save "$W/multiexpert_self_change_ablate.pt" \
  > "$L/multiexpert_self_change_ablate.log" 2>&1

echo "[$(stamp)] [4/5] compare all-expert full-pose outputs with MobilePoser metrics"
$PY compare_expert_pose_metrics.py \
  --data "$DATA" \
  --manifest "$DATA/splits.json" \
  --root-ckpt "$W/multiexpert_root_current_ablate_best.pt" \
  --self-ckpt "$W/multiexpert_self_change_ablate_best.pt" \
  --ks 2,3,4,5 \
  > "$L/posechange_mobileposer_compare.log" 2>&1

echo "[$(stamp)] [5/5] done"
echo "  data:    $DATA"
echo "  root:    $W/multiexpert_root_current_ablate_best.pt"
echo "  self:    $W/multiexpert_self_change_ablate_best.pt"
echo "  compare: $L/posechange_mobileposer_compare.log"
