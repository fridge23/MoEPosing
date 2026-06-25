#!/usr/bin/env bash
set -euo pipefail

ROOT=.
PY=python
DATA=data60hz
MANIFEST=data/splits.json
ROOT_CKPT=$ROOT/weights/multiexpert_9d64.pt
SELF_CKPT=$ROOT/weights/multiexpert_self_change_9d64_best.pt
LOGDIR=logs

if [[ ! -f "$SELF_CKPT" ]]; then
  echo "missing self-change checkpoint: $SELF_CKPT" >&2
  exit 1
fi

cd "$ROOT/expert_pretrain"

run_eval() {
  local label="$1"
  local kw="$2"
  local log="$LOGDIR/compare_self_vs_root_9d64_${label}.log"
  echo "[compare] label=$label include_kw=${kw:-all} log=$log"
  if [[ -n "$kw" ]]; then
    env PYTHONUNBUFFERED=1 "$PY" compare_expert_pose_metrics.py       --data "$DATA"       --manifest "$MANIFEST"       --root-ckpt "$ROOT_CKPT"       --self-ckpt "$SELF_CKPT"       --ks 2,3,4,5       --include-kw "$kw"       2>&1 | tee "$log"
  else
    env PYTHONUNBUFFERED=1 "$PY" compare_expert_pose_metrics.py       --data "$DATA"       --manifest "$MANIFEST"       --root-ckpt "$ROOT_CKPT"       --self-ckpt "$SELF_CKPT"       --ks 2,3,4,5       2>&1 | tee "$log"
  fi
}

run_eval all ""
run_eval totalcapture totalcapture
run_eval dip dip
run_eval imuposer imuposer
