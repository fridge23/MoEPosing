#!/usr/bin/env bash
# Overnight pipeline: train experts to convergence, then the whole-body recovery
# model (Phase-I GT-known -> Phase-II expert-known), then a clean held-out eval.
# Each stage stops early on val convergence (--patience). Chain aborts on failure.
set -euo pipefail

PY=python
ED=expert_pretrain
W=weights
L=logs
cd "$ED"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(stamp)] === OVERNIGHT PIPELINE START ==="

# 1) per-joint experts (4-layer / 64-dim, 9D = 6D orient + 3D motion-delta).
#    All 28 in ONE data pass; per-joint convergence + per-joint best snapshot +
#    --freeze-converged so a joint drops out of compute once it plateaus. 100
#    epochs cap, patience 15. --preload keeps data in RAM (compute-bound epochs).
#    Loss: ADAPTIVE balance -> weighted xyz ~= 2x weighted orientation throughout.
echo "[$(stamp)] [1/4] training experts -> $W/multiexpert_9d64.pt"
$PY train_multiexpert.py --epochs 100 --patience 15 --loss per_joint \
    --hidden 64 --layers 4 --target joint_orient_r6d,joint_delta \
    --loss-balance adaptive --xyz-weight 2.0 \
    --acc-noise 0.2 --ori-noise-deg 8 --preload --num-workers 6 \
    --save "$W/multiexpert_9d64.pt" > "$L/orient_train_9d64.log" 2>&1
echo "[$(stamp)] [1/4] experts done"

# 2) whole-body recovery Phase-I: GROUND-TRUTH targets at known joints (+noise).
echo "[$(stamp)] [2/4] whole-body Phase-I -> $W/wholebody_phase1.pt"
$PY train_wholebody.py --epochs 100 --patience 15 --num-workers 6 --preload \
    --loss-balance adaptive --xyz-weight 2.0 \
    --save "$W/wholebody_phase1.pt" > "$L/wholebody_phase1.log" 2>&1
echo "[$(stamp)] [2/4] Phase-I done"

# 3) Phase-II: frozen experts' predictions as the known joints; warm-start from Phase-I.
echo "[$(stamp)] [3/4] whole-body Phase-II -> $W/wholebody_phase2.pt"
$PY train_wholebody.py --epochs 100 --patience 15 --num-workers 6 --preload \
    --loss-balance adaptive --xyz-weight 2.0 \
    --experts "$W/multiexpert_9d64_best.pt" \
    --init "$W/wholebody_phase1_best.pt" --pose-noise 0 \
    --save "$W/wholebody_phase2.pt" > "$L/wholebody_phase2.log" 2>&1
echo "[$(stamp)] [3/4] Phase-II done"

# 4) final held-out expert eval (MPJRE per sensor-count + per joint)
echo "[$(stamp)] [4/4] expert eval on held-out test"
$PY eval_experts.py --ckpt "$W/multiexpert_9d64_best.pt" > "$L/eval_experts_9d64.log" 2>&1
echo "[$(stamp)] [4/4] eval done"

echo "[$(stamp)] === OVERNIGHT PIPELINE COMPLETE ==="
