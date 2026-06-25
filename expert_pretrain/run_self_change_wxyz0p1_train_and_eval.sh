#!/usr/bin/env bash
set -euo pipefail

ROOT=.
PY=python
DATA=data60hz
MANIFEST=data/splits.json
LOGDIR=logs
ROOT_CKPT=$ROOT/weights/multiexpert_9d64.pt
RESUME_CKPT=$ROOT/weights/multiexpert_self_change_9d64_best.pt
SELF_CKPT=$ROOT/weights/multiexpert_self_change_9d64_wxyz0p1.pt
SELF_BEST=$ROOT/weights/multiexpert_self_change_9d64_wxyz0p1_best.pt
TRAIN_LOG=$LOGDIR/self_change_9d64_wxyz0p1.log
PER_JOINT_LOG=$LOGDIR/self_change_9d64_wxyz0p1_per_joint.jsonl

mkdir -p "$LOGDIR" "$ROOT/weights"
: > "$PER_JOINT_LOG"
cd "$ROOT/expert_pretrain"

echo "[driver] start fixed-weight self-change training $(date -Is)"
echo "[driver] resume=$RESUME_CKPT save=$SELF_CKPT"
echo "[driver] weights: orientation=1.0 xyz=0.1 early_stop_metric=total_loss patience=15"

env PYTHONUNBUFFERED=1 "$PY" train_multiexpert.py \
  --data "$DATA" \
  --manifest "$MANIFEST" \
  --epochs 100 \
  --patience 15 \
  --loss per_joint \
  --hidden 64 \
  --layers 4 \
  --target joint_rot_delta_r6d,joint_delta_local \
  --loss-balance fixed \
  --lambda-orientation 1.0 \
  --lambda-motion-delta 0.1 \
  --early-stop-metric total_loss \
  --acc-noise 0.2 \
  --ori-noise-deg 8 \
  --preload \
  --num-workers 6 \
  --log-every 100 \
  --per-joint-log "$PER_JOINT_LOG" \
  --resume "$RESUME_CKPT" \
  --resume-reset-epoch \
  --resume-reset-best \
  --save "$SELF_CKPT" \
  2>&1 | tee "$TRAIN_LOG"

if [[ ! -f "$SELF_BEST" ]]; then
  echo "[driver] missing trained best checkpoint: $SELF_BEST" >&2
  exit 1
fi

echo "[driver] training finished $(date -Is); start grouped MobilePoser metrics"

run_eval() {
  local label="$1"
  local kw="$2"
  local log="$LOGDIR/compare_self_vs_root_9d64_wxyz0p1_${label}.log"
  echo "[compare] label=$label include_kw=${kw:-all} log=$log"
  if [[ -n "$kw" ]]; then
    env PYTHONUNBUFFERED=1 "$PY" compare_expert_pose_metrics.py \
      --data "$DATA" \
      --manifest "$MANIFEST" \
      --root-ckpt "$ROOT_CKPT" \
      --self-ckpt "$SELF_BEST" \
      --ks 2,3,4,5 \
      --include-kw "$kw" \
      2>&1 | tee "$log"
  else
    env PYTHONUNBUFFERED=1 "$PY" compare_expert_pose_metrics.py \
      --data "$DATA" \
      --manifest "$MANIFEST" \
      --root-ckpt "$ROOT_CKPT" \
      --self-ckpt "$SELF_BEST" \
      --ks 2,3,4,5 \
      2>&1 | tee "$log"
  fi
}

run_eval all ""
run_eval totalcapture totalcapture
run_eval dip dip
run_eval imuposer imuposer

echo "[driver] all done $(date -Is)"
