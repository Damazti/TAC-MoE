#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 /path/to/Stage2_output_dir/checkpoint-STEP" >&2
    exit 1
fi

TEST_CKPT="$1"
TAG="${TAG:-r05}"
MODEL="${MODEL:-resources/qwen2-7b}"
NUM_GPUS_PRED="${NUM_GPUS_PRED:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
LORA_RANK="${LORA_RANK:-64}"
EXPERT_NUM="${EXPERT_NUM:-8}"
DATA_DIR="${DATA_DIR:-}"
RESULT_DIR="${RESULT_DIR:-}"

if [ ! -d "$TEST_CKPT" ]; then
    echo "ERROR: checkpoint path does not exist or is not a directory: $TEST_CKPT" >&2
    exit 1
fi
if [[ ! "$(basename "$TEST_CKPT")" =~ ^checkpoint-[0-9]+$ ]]; then
    echo "ERROR: this script only tests an explicit checkpoint-* path: $TEST_CKPT" >&2
    exit 1
fi

OUTPUT_DIR="$(dirname "$TEST_CKPT")"
TEST_STEP="${TEST_CKPT##*-}"
RUN_NAME="$(basename "$OUTPUT_DIR")"
WORK_ROOT="$(cd "$(dirname "$OUTPUT_DIR")/.." && pwd)"

if [ -z "$DATA_DIR" ]; then
    DATA_DIR="$WORK_ROOT/data/$TAG/Stage2"
    if [ ! -f "$DATA_DIR/test.json" ] && [ -f "$WORK_ROOT/data/$TAG/phase2/test.json" ]; then
        DATA_DIR="$WORK_ROOT/data/$TAG/phase2"
    fi
fi
if [ -z "$RESULT_DIR" ]; then
    RESULT_DIR="$WORK_ROOT/results/${RUN_NAME}_ckpt${TEST_STEP}_test"
fi

if [ ! -f "$DATA_DIR/test.json" ]; then
    echo "ERROR: missing test data: $DATA_DIR/test.json" >&2
    echo "Set DATA_DIR=/path/to/Stage2 if the checkpoint is not under a standard work/saved layout." >&2
    exit 1
fi

mkdir -p "$RESULT_DIR"
MASTER_PORT="$(shuf -n 1 -i 10000-65535)"

echo "[$(date '+%F %T')] checkpoint step=$TEST_STEP path=$TEST_CKPT"
echo "[$(date '+%F %T')] data_dir=$DATA_DIR"
echo "[$(date '+%F %T')] result_dir=$RESULT_DIR"

deepspeed --num_gpus="$NUM_GPUS_PRED" --master_port "$MASTER_PORT" run_qwen.py \
    --do_predict \
    --test_file "$DATA_DIR/test.json" \
    --cache_dir "$DATA_DIR/cache_pred_${RUN_NAME}_ckpt${TEST_STEP}" \
    --overwrite_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path "$MODEL" \
    --peft_path "$TEST_CKPT" \
    --output_dir "$RESULT_DIR" \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 512 \
    --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
    --predict_with_generate \
    --generation_max_length 16 \
    --log_level warning \
    --lora_name moelora \
    --lora_rank "$LORA_RANK" \
    --task_num 3 \
    --expert_num "$EXPERT_NUM" \
    --report_to none

python evaluate_tasks.py \
    --predictions "$RESULT_DIR/test_predictions.json" \
    --test_data "$DATA_DIR/test.json" \
    --metric macro | tee "$RESULT_DIR/eval_results.txt"

printf '%s\n' "$TEST_STEP" > "$RESULT_DIR/test_step.txt"
printf '%s\n' "$TEST_CKPT" > "$RESULT_DIR/test_checkpoint.txt"
cp "$DATA_DIR/task_dataset.json" "$RESULT_DIR/task_dataset.Stage2.json" 2>/dev/null || true
cp "$OUTPUT_DIR/trainer_state.json" "$RESULT_DIR/trainer_state.json" 2>/dev/null || true

echo "[$(date '+%F %T')] done"
