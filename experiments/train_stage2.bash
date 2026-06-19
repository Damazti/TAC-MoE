#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_GPUS_TRAIN="${NUM_GPUS_TRAIN:-2}"
P2_MAX_STEPS="${P2_MAX_STEPS:-800}"
ACH_WARMUP_RATIO="${ACH_WARMUP_RATIO:-30}"
IHD_RATIO="${IHD_RATIO:-0.8}"
IACV2_RATIO="${IACV2_RATIO:-0.6}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-IHD${IHD_RATIO}_IACV${IACV2_RATIO}_warmup${ACH_WARMUP_RATIO}_${NUM_GPUS_TRAIN}gpu_p2step${P2_MAX_STEPS}}"
export WORK_ROOT="${WORK_ROOT:-$SCRIPT_DIR/$EXPERIMENT_NAME/work}"
if [ -z "${P1_CKPT:-}" ] && [ "$#" -gt 0 ]; then
    P1_CKPT="$1"
    shift
    export P1_CKPT
fi
if [ -z "${P1_CKPT:-}" ] && [ -f "$WORK_ROOT/p1_best_checkpoint.txt" ]; then
    P1_CKPT="$(sed -n '1p' "$WORK_ROOT/p1_best_checkpoint.txt")"
    export P1_CKPT
fi

if [ -z "${P1_CKPT:-}" ]; then
    echo "ERROR: train_stage2.bash requires /path/to/phase1/checkpoint, P1_CKPT, or $WORK_ROOT/p1_best_checkpoint.txt" >&2
    exit 1
fi

export RUN_PHASE1=0
export RUN_PHASE2=1

exec "$SCRIPT_DIR/train_full.bash" "$@"
