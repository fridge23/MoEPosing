#!/bin/bash
# Stage 1 A/B Experiment: Train Plan A and Plan B in parallel on 2 GPUs
# Plan A (mask_position=before): Full-28 input, encoder processes all 28 joint slots
# Plan B (mask_position=after):  IMU-slot-only encoder, scatter to 28 after encoding
#
# Usage: bash expert_pretrain/launch_ab_experiment.sh
# Requires: 2 GPUs (CUDA_VISIBLE_DEVICES=0,1)

set -e
cd "$(dirname "$0")/.."

DATA="data60hz"
PRIOR="pretrained/student_kl_18to21_best_64.pth"
EPOCHS=100
BATCH=64
LR=3e-4
HIDDEN=64
ENC_LAYERS=4
DEC_LAYERS=2
SPATIAL_LAYERS=2
NHEAD=4
FF=128
SEED=42
MIN_K=2
MAX_K=5
PATIENCE=20
HOLDOUT="dip,totalcapture,imuposer"
TARGET="joint_orient_r6d,joint_delta"
AMP="bf16"
WORKERS=4

PLAN_A_DIR="weights/plan_a_before"
PLAN_B_DIR="weights/plan_b_after"

mkdir -p "$PLAN_A_DIR" "$PLAN_B_DIR"

echo "=============================================="
echo " Stage 1 A/B Experiment"
echo " Plan A (Full-28 Input)     -> GPU 0 -> $PLAN_A_DIR"
echo " Plan B (After-Encoder Mask) -> GPU 1 -> $PLAN_B_DIR"
echo "=============================================="
echo ""

# Common args
COMMON="--data $DATA \
  --prior $PRIOR \
  --pretrained-layers 2 \
  --epochs $EPOCHS \
  --batch-size $BATCH \
  --lr $LR \
  --hidden $HIDDEN \
  --encoder-layers $ENC_LAYERS \
  --decoder-layers $DEC_LAYERS \
  --spatial-layers $SPATIAL_LAYERS \
  --nhead $NHEAD \
  --ff $FF \
  --seed $SEED \
  --min-k $MIN_K \
  --max-k $MAX_K \
  --patience $PATIENCE \
  --holdout-kw $HOLDOUT \
  --target $TARGET \
  --lambda-orientation 1.0 \
  --lambda-motion-delta 1.0 \
  --amp $AMP \
  --num-workers $WORKERS \
  --log-every 50"

echo "[$(date)] Starting Plan A (before) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python expert_pretrain/train_shared_encoder.py \
  $COMMON \
  --mask-position before \
  --save-dir "$PLAN_A_DIR" \
  --per-joint-log "$PLAN_A_DIR/per_joint_log.jsonl" \
  2>&1 | tee "$PLAN_A_DIR/train.log" &
PID_A=$!

echo "[$(date)] Starting Plan B (after) on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python expert_pretrain/train_shared_encoder.py \
  $COMMON \
  --mask-position after \
  --save-dir "$PLAN_B_DIR" \
  --per-joint-log "$PLAN_B_DIR/per_joint_log.jsonl" \
  2>&1 | tee "$PLAN_B_DIR/train.log" &
PID_B=$!

echo ""
echo "[$(date)] Both plans running: Plan A (PID $PID_A), Plan B (PID $PID_B)"
echo "  Monitor: tail -f $PLAN_A_DIR/train.log $PLAN_B_DIR/train.log"
echo ""

wait $PID_A
STATUS_A=$?
echo "[$(date)] Plan A finished with status $STATUS_A"

wait $PID_B
STATUS_B=$?
echo "[$(date)] Plan B finished with status $STATUS_B"

echo ""
echo "=============================================="
echo " A/B Experiment Complete"
echo " Plan A log: $PLAN_A_DIR/train.log"
echo " Plan B log: $PLAN_B_DIR/train.log"
echo " Plan A best encoder: $PLAN_A_DIR/best_encoder.pt"
echo " Plan B best encoder: $PLAN_B_DIR/best_encoder.pt"
echo "=============================================="
