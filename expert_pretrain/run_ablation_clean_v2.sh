#!/usr/bin/env bash
# v2: same ablation, but num_workers=8 (the GB10 OOM cause — the concurrent rebuild —
# is gone, 91GB free). SELF RESUMES from its epoch-15 state (no reset: keeps per-joint
# best/convergence + LR schedule); ROOT starts fresh from its pre-fix best; then tests.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/pengfei/Downloads/.codex-data-env/bin/python
cd /home/pengfei/Downloads/dynaip/expert_pretrain
D=/home/pengfei/Downloads/poser_mle_posechange_fixed_60hz
MAN=$D/splits.json
W=/home/pengfei/Downloads/dynaip/weights
L=/home/pengfei/Downloads/logs
SMPL=/home/pengfei/Downloads/mobileposer_official/mobileposer/smpl/basicmodel_m.pkl

BASE="--data $D --manifest $MAN --loss per_joint --loss-balance fixed \
  --lambda-orientation 1.0 --early-stop-metric total_loss --min-delta 1e-6 --patience 15 \
  --epochs 100 --hidden 64 --layers 4 --batch-size 64 --num-workers 12 --lr 3e-4 --weight-decay 1e-4 \
  --amp bf16 --min-k 2 --max-k 5 --acc-noise 0.2 --ori-noise-deg 8"

echo "==== [1/3] CONTINUE self-change from epoch 15 (12 workers) w_xyz=0.15 ===="
$PY train_multiexpert.py $BASE --target joint_rot_delta_r6d,joint_delta_local --lambda-motion-delta 0.15 \
  --resume $W/multiexpert_selfchange_clean.pt \
  --save $W/multiexpert_selfchange_clean.pt \
  --per-joint-log $L/selfchange_clean_perjoint.jsonl \
  >> $L/selfchange_clean_train.log 2>&1
echo "     self done -> $W/multiexpert_selfchange_clean_best.pt"

echo "==== [2/3] TRAIN root-relative fresh (12 workers) w_xyz=120 ===="
$PY train_multiexpert.py $BASE --target joint_orient_r6d,joint_delta --lambda-motion-delta 120 \
  --resume-reset-best --resume-reset-epoch \
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
    --ks 2,3,4,5 $INC --num-workers 4 --mobileposer-smpl $SMPL \
    > $L/compare_${KW}.log 2>&1
  echo "     compare done: $KW -> $L/compare_${KW}.log"
done
echo "==== ALL DONE ===="
