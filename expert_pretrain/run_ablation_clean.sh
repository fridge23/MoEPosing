#!/usr/bin/env bash
# Root-relative vs self-relative MultiExpert ablation on the CLEAN rebuilt data.
# Both models retrained with matched config + RATIO-matched fixed loss weights
# (xyz:orient ~1.2 at start): SELF w_xyz=0.15, ROOT w_xyz=120. Train SELF then ROOT,
# then compare with MobilePoser metrics. Each warm-started from its own pre-fix best.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # GB10 unified-memory safety
PY=python
cd expert_pretrain
D=data60hz
MAN=$D/splits.json
W=weights
L=logs
SMPL=smpl_models/basicmodel_m.pkl

# matched everything except --target and the ratio-matched xyz weight; num-workers low for GB10
COMMON="--data $D --manifest $MAN --loss per_joint --loss-balance fixed \
  --lambda-orientation 1.0 --early-stop-metric total_loss --min-delta 1e-6 --patience 15 \
  --epochs 100 --hidden 64 --layers 4 --batch-size 64 --num-workers 2 --lr 3e-4 --weight-decay 1e-4 \
  --amp bf16 --min-k 2 --max-k 5 --acc-noise 0.2 --ori-noise-deg 8 \
  --resume-reset-best --resume-reset-epoch"

echo "==== [1/3] TRAIN self-change (rot_delta_r6d,delta_local) w_xyz=0.15 ===="
$PY train_multiexpert.py $COMMON --target joint_rot_delta_r6d,joint_delta_local --lambda-motion-delta 0.15 \
  --resume $W/multiexpert_self_change_9d64_best.pt \
  --save $W/multiexpert_selfchange_clean.pt \
  --per-joint-log $L/selfchange_clean_perjoint.jsonl \
  > $L/selfchange_clean_train.log 2>&1
echo "     self done -> $W/multiexpert_selfchange_clean_best.pt"

echo "==== [2/3] TRAIN root-relative (orient_r6d,joint_delta) w_xyz=120 ===="
$PY train_multiexpert.py $COMMON --target joint_orient_r6d,joint_delta --lambda-motion-delta 120 \
  --resume $W/multiexpert_9d64_best.pt \
  --save $W/multiexpert_root_clean.pt \
  --per-joint-log $L/root_clean_perjoint.jsonl \
  > $L/root_clean_train.log 2>&1
echo "     root done -> $W/multiexpert_root_clean_best.pt"

echo "==== [3/3] TEST: MobilePoser metrics, per-dataset + all, k=2,3,4,5 ===="
for KW in totalcapture dip imuposer __all__; do
  INC=""; [ "$KW" != "__all__" ] && INC="--include-kw $KW"
  $PY compare_expert_pose_metrics.py \
    --data $D --manifest $MAN --split test \
    --root-ckpt $W/multiexpert_root_clean_best.pt \
    --self-ckpt $W/multiexpert_selfchange_clean_best.pt \
    --ks 2,3,4,5 $INC --mobileposer-smpl $SMPL \
    > $L/compare_${KW}.log 2>&1
  echo "     compare done: $KW -> $L/compare_${KW}.log"
done
echo "==== ALL DONE ===="
