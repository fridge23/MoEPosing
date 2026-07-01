#!/usr/bin/env bash
set -euo pipefail

#=============================================================================
# MoEPosing full training pipeline
#
#   Step 0: Process all datasets -> data_processed/
#   Step 1: Smoke test
#   Step 2: Stage 1 — shared encoder pretraining
#   Step 3: Stage 2 — lightweight joint experts
#
# Usage:
#   bash run_full_pipeline.sh                      # run both before & after
#   bash run_full_pipeline.sh --only before        # only before (GPU 0)
#   bash run_full_pipeline.sh --only after         # only after  (GPU 1)
#   bash run_full_pipeline.sh --skip-data          # skip data processing
#   bash run_full_pipeline.sh --step 2             # start from step 2
#
# Two A100s in parallel:
#   CUDA_VISIBLE_DEVICES=0 bash run_full_pipeline.sh --only before &
#   CUDA_VISIBLE_DEVICES=1 bash run_full_pipeline.sh --only after  &
#=============================================================================

cd "$(dirname "$0")"
PROJ="$(pwd)"

# ---- configuration (edit these) --------------------------------------------
DATA="data60hz,data_processed"          # combined old shards + new processed
DATA_ROOT="data"                        # raw data root for process_new_datasets
DATA_OUT="data_processed"               # output dir for processed datasets
DEVICE="cuda"
EPOCHS_SE=100                           # shared encoder epochs
EPOCHS_EXP=100                          # expert epochs
PATIENCE_SE=20
PATIENCE_EXP=25
BATCH=64
LR=3e-4
HIDDEN=64
ENC_LAYERS=4
SPATIAL_LAYERS=2
DEC_LAYERS=2
EXP_LAYERS=2
NHEAD=4
FF=128
MIN_K=2
MAX_K=5
AMP=bf16
WORKERS=0
PRIOR="pretrained/student_kl_18to21_best_64.pth"
HOLDOUT="dip,totalcapture,imuposer"     # held-out test datasets
# ----------------------------------------------------------------------------

LOGDIR="$PROJ/logs"
mkdir -p "$LOGDIR" weights

SKIP_DATA=0
START_STEP=0
ONLY=""          # "" = both, "before" or "after" or "direct" = single mode
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-data) SKIP_DATA=1; shift ;;
        --step)      START_STEP="$2"; shift 2 ;;
        --only)      ONLY="$2"; shift 2 ;;
        *)           shift ;;
    esac
done

if [[ -n "$ONLY" ]]; then
    MASK_POSITIONS=("$ONLY")
else
    MASK_POSITIONS=("before" "after")
fi

ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "============================================================"
echo "  MoEPosing full pipeline  $(ts)"
echo "  DATA=$DATA  DEVICE=$DEVICE  EPOCHS_SE=$EPOCHS_SE"
echo "  MASK_POSITIONS=${MASK_POSITIONS[*]}"
echo "  GPU=${CUDA_VISIBLE_DEVICES:-all}"
echo "============================================================"

#=============================================================================
# Step 0: Data processing
#=============================================================================
if [[ "$START_STEP" -le 0 && "$SKIP_DATA" -eq 0 ]]; then
    echo ""
    echo ">>> [Step 0] Processing all datasets  $(ts)"
    echo "    data-root=$DATA_ROOT  output=$DATA_OUT  sources=all"
    python expert_pretrain/process_new_datasets.py \
        --data-root "$DATA_ROOT" \
        --output "$DATA_OUT" \
        --sources all \
        2>&1 | tee "$LOGDIR/step0_process_data.log"
    echo ">>> [Step 0] Done  $(ts)"
fi

#=============================================================================
# Step 1: Smoke tests (both mask positions)
#=============================================================================
if [[ "$START_STEP" -le 1 ]]; then
    echo ""
    echo ">>> [Step 1] Smoke tests  $(ts)"

    echo "--- GPU check ---"
    python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

    for MP in "${MASK_POSITIONS[@]}"; do
        echo ""
        echo "--- Smoke: shared encoder (mask-position=$MP) ---"
        python expert_pretrain/train_shared_encoder.py \
            --data "$DATA" \
            --mask-position "$MP" \
            --device "$DEVICE" \
            --save-dir "weights/smoke_se_${MP}" \
            --prior "$PRIOR" \
            --hidden $HIDDEN \
            --encoder-layers $ENC_LAYERS \
            --spatial-layers $SPATIAL_LAYERS \
            --decoder-layers $DEC_LAYERS \
            --nhead $NHEAD --ff $FF \
            --batch-size $BATCH \
            --min-k $MIN_K --max-k $MAX_K \
            --holdout-kw "$HOLDOUT" \
            --amp $AMP \
            --preload \
            --max-steps 50 \
            2>&1 | tee "$LOGDIR/step1_smoke_se_${MP}.log"

        echo ""
        echo "--- Smoke: expert (mask-position=$MP, left_knee) ---"
        python expert_pretrain/train_lightweight_expert.py \
            --data "$DATA" \
            --encoder-checkpoint "weights/smoke_se_${MP}/latest.pt" \
            --target-joint left_knee \
            --device "$DEVICE" \
            --save-dir "weights/smoke_exp_${MP}" \
            --expert-layers $EXP_LAYERS \
            --batch-size $BATCH \
            --min-k $MIN_K --max-k $MAX_K \
            --holdout-kw "$HOLDOUT" \
            --amp $AMP \
            --preload \
            --max-steps 30 \
            2>&1 | tee "$LOGDIR/step1_smoke_exp_${MP}.log"
    done

    echo ">>> [Step 1] Smoke tests passed  $(ts)"
fi

#=============================================================================
# Step 2: Stage 1 — shared encoder pretraining
#=============================================================================
if [[ "$START_STEP" -le 2 ]]; then
    echo ""
    echo ">>> [Step 2] Stage 1: shared encoder pretraining  $(ts)"

    for MP in "${MASK_POSITIONS[@]}"; do
        SAVE="weights/se_${MP}"
        LOG="$LOGDIR/step2_se_${MP}.log"
        PJ_LOG="$LOGDIR/step2_se_${MP}_perjoint.jsonl"

        echo ""
        echo "--- Training shared encoder (mask-position=$MP) ---"
        echo "    save-dir=$SAVE  log=$LOG"

        python expert_pretrain/train_shared_encoder.py \
            --data "$DATA" \
            --mask-position "$MP" \
            --device "$DEVICE" \
            --save-dir "$SAVE" \
            --prior "$PRIOR" \
            --hidden $HIDDEN \
            --encoder-layers $ENC_LAYERS \
            --spatial-layers $SPATIAL_LAYERS \
            --decoder-layers $DEC_LAYERS \
            --nhead $NHEAD --ff $FF \
            --batch-size $BATCH \
            --lr $LR \
            --epochs $EPOCHS_SE \
            --patience $PATIENCE_SE \
            --min-k $MIN_K --max-k $MAX_K \
            --holdout-kw "$HOLDOUT" \
            --amp $AMP \
            --num-workers $WORKERS \
            --preload \
            --per-joint-log "$PJ_LOG" \
            --log-every 100 \
            2>&1 | tee "$LOG"

        echo "--- Shared encoder ($MP) done  $(ts) ---"
    done

    echo ">>> [Step 2] Stage 1 complete  $(ts)"
fi

#=============================================================================
# Step 3: Stage 2 — lightweight joint experts
#=============================================================================
if [[ "$START_STEP" -le 3 ]]; then
    echo ""
    echo ">>> [Step 3] Stage 2: lightweight joint experts  $(ts)"

    for MP in "${MASK_POSITIONS[@]}"; do
        ENC_CKPT="weights/se_${MP}/best_encoder.pt"
        SAVE="weights/exp_${MP}"
        LOG="$LOGDIR/step3_exp_${MP}.log"

        if [[ ! -f "$ENC_CKPT" ]]; then
            echo "[SKIP] $ENC_CKPT not found — run Step 2 first"
            continue
        fi

        echo ""
        echo "--- Training all 28 experts (encoder=$MP) ---"
        echo "    encoder=$ENC_CKPT  save-dir=$SAVE  log=$LOG"

        python expert_pretrain/train_lightweight_expert.py \
            --data "$DATA" \
            --encoder-checkpoint "$ENC_CKPT" \
            --train-all-experts \
            --device "$DEVICE" \
            --save-dir "$SAVE" \
            --expert-layers $EXP_LAYERS \
            --batch-size $BATCH \
            --lr $LR \
            --epochs $EPOCHS_EXP \
            --patience $PATIENCE_EXP \
            --min-k $MIN_K --max-k $MAX_K \
            --holdout-kw "$HOLDOUT" \
            --amp $AMP \
            --num-workers $WORKERS \
            --preload \
            2>&1 | tee "$LOG"

        echo "--- All experts ($MP) done  $(ts) ---"
    done

    echo ">>> [Step 3] Stage 2 complete  $(ts)"
fi

#=============================================================================
# Step 4: Direct experts (no encoder baseline)
#=============================================================================
if [[ "$START_STEP" -le 4 ]]; then
    # Only run if "direct" is in the mask positions list
    for MP in "${MASK_POSITIONS[@]}"; do
        if [[ "$MP" == "direct" ]]; then
            echo ""
            echo ">>> [Step 4] Direct experts (no encoder baseline)  $(ts)"

            SAVE="weights/direct_experts"
            LOG="$LOGDIR/step4_direct_experts.log"

            echo "--- Training all 28 direct experts ---"
            echo "    prior=$PRIOR  save-dir=$SAVE  log=$LOG"

            python expert_pretrain/train_direct_expert.py \
                --data "$DATA" \
                --prior "$PRIOR" \
                --train-all-experts \
                --device "$DEVICE" \
                --save-dir "$SAVE" \
                --batch-size $BATCH \
                --lr $LR \
                --epochs $EPOCHS_EXP \
                --patience 15 \
                --min-k $MIN_K --max-k $MAX_K \
                --holdout-kw "$HOLDOUT" \
                --amp $AMP \
                --num-workers $WORKERS \
                --preload \
                2>&1 | tee "$LOG"

            echo "--- Direct experts done  $(ts) ---"
            echo ">>> [Step 4] Direct experts complete  $(ts)"
            break
        fi
    done
fi

echo ""
echo "============================================================"
echo "  Pipeline finished  $(ts)"
echo ""
echo "  Outputs:"
echo "    logs/           — training logs + per-joint metrics"
echo "    weights/se_before/   — Stage 1 encoder (before mask)"
echo "    weights/se_after/    — Stage 1 encoder (after mask)"
echo "    weights/exp_before/  — 28 experts on before encoder"
echo "    weights/exp_after/   — 28 experts on after encoder"
echo "    weights/direct_experts/ — 28 direct experts (no encoder)"
echo "============================================================"
