#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"


export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TAG="${TAG:-r05}"
RATIO="${RATIO:-0.05}"
SEED="${SEED:-42}"
WORK_ROOT="${WORK_ROOT:-}"
STAGE1_DIR_NAME="${STAGE1_DIR_NAME:-Stage1}"
STAGE2_DIR_NAME="${STAGE2_DIR_NAME:-Stage2}"

NUM_GPUS_TRAIN="${NUM_GPUS_TRAIN:-2}"
NUM_GPUS_PRED="${NUM_GPUS_PRED:-2}"
MODEL="${MODEL:-resources/qwen2-7b}"
EXPERT_NUM="${EXPERT_NUM:-8}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
TRAINABLE="${TRAINABLE:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

P1_LR="${P1_LR:-2e-4}"
P1_WD="${P1_WD:-0.01}"
P1_BS="${P1_BS:-16}"
P1_GRAD_ACCUM="${P1_GRAD_ACCUM:-4}"
P1_MAX_STEPS="${P1_MAX_STEPS:-400}"
P1_SAVE_STEPS="${P1_SAVE_STEPS:-30}"
P1_EVAL_STEPS="${P1_EVAL_STEPS:-30}"
P1_EVAL_DELAY="${P1_EVAL_DELAY:-300}"
P1_SAVE_TOTAL_LIMIT="${P1_SAVE_TOTAL_LIMIT:-6}"
P1_USE_ACHIEVEMENT_LOSS="${P1_USE_ACHIEVEMENT_LOSS:-0}"
P1_ACH_TARGETS="${P1_ACH_TARGETS:-IHD:0.81,IACV2:0.845}"
P1_WEIGHT_FLOOR="${P1_WEIGHT_FLOOR:-0}"
P1_INCLUDE_FINAL_EVAL_IN_BEST="${P1_INCLUDE_FINAL_EVAL_IN_BEST:-0}"

P2_LR="${P2_LR:-5e-4}"
P2_WD="${P2_WD:-0}"
P2_BS="${P2_BS:-16}"
P2_GRAD_ACCUM="${P2_GRAD_ACCUM:-1}"
P2_MAX_STEPS="${P2_MAX_STEPS:-800}"
P2_SAVE_STEPS="${P2_SAVE_STEPS:-30}"
P2_EVAL_STEPS="${P2_EVAL_STEPS:-30}"
P2_EVAL_DELAY="${P2_EVAL_DELAY:-}"
P2_SAVE_TOTAL_LIMIT="${P2_SAVE_TOTAL_LIMIT:-}"
P2_VALIDATION_SPLIT="${P2_VALIDATION_SPLIT:-dev}"
P2_SEMEVAL_DEV_RATIO="${P2_SEMEVAL_DEV_RATIO:-}"
P2_INIT_FROM="${P2_INIT_FROM:-1}"
FREEZE_EXPERT_IDS="${FREEZE_EXPERT_IDS:-0,2,4,5}"

P2_USE_ACHIEVEMENT_LOSS="${P2_USE_ACHIEVEMENT_LOSS:-1}"
P2_ACH_TARGETS="${P2_ACH_TARGETS:-IHD:0.815,IACV2:0.855,SemEval2018:0.832}"
ACH_GAMMA="${ACH_GAMMA:-1.0}"
ACH_MARGIN="${ACH_MARGIN:-1.0}"
ACH_EMA="${ACH_EMA:-0.3}"
ACH_WARMUP_RATIO="${ACH_WARMUP_RATIO:-35}"
P2_WEIGHT_FLOOR="${P2_WEIGHT_FLOOR:-0}"
P2_REPLAY_RATIO="${P2_REPLAY_RATIO:-1}"
IHD_RATIO="${IHD_RATIO:-0.8}"
IACV2_RATIO="${IACV2_RATIO:-0.6}"
P2_REPLAY_TAG="${P2_REPLAY_TAG:-IHD${IHD_RATIO}_IACV${IACV2_RATIO}}"
P2_TASK_MEMORY_RATIOS="${P2_TASK_MEMORY_RATIOS:-IHD:${IHD_RATIO},IACV2:${IACV2_RATIO}}"
P2_USE_EXPERIENCE_REPLAY="${P2_USE_EXPERIENCE_REPLAY:-1}"
P2_PREDICT_START_RATIO="${P2_PREDICT_START_RATIO:-67}"

if [ -z "$P2_EVAL_DELAY" ]; then
    P2_EVAL_DELAY=$((P2_MAX_STEPS * ACH_WARMUP_RATIO / 100))
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-IHD${IHD_RATIO}_IACV${IACV2_RATIO}_warmup${ACH_WARMUP_RATIO}_${NUM_GPUS_TRAIN}gpu_p2step${P2_MAX_STEPS}}"
WORK_ROOT="${WORK_ROOT:-$SCRIPT_DIR/$EXPERIMENT_NAME/work}"
LOGFILE="$WORK_ROOT/run.log"

RUN_PHASE1="${RUN_PHASE1:-1}"
RUN_PHASE2="${RUN_PHASE2:-1}"
RUN_PREDICT="${RUN_PREDICT:-1}"
P1_CKPT="${P1_CKPT:-}"

mkdir -p "$WORK_ROOT"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

write_phase1_task_map() {
    mkdir -p data
    cat > data/task_dataset.json <<'EOF'
{"str2id":{"IHD":1,"IACV2":2},"id2str":{"1":"IHD","2":"Sarca"}}
EOF
}

write_phase2_task_map() {
    mkdir -p data
    cat > data/task_dataset.json <<'EOF'
{"str2id":{"IHD":1,"IACV2":2,"SemEval2018":3},"id2str":{"1":"IHD","2":"Sarca","3":"Irony"}}
EOF
}

split_train_dev() {
    python "$SCRIPT_DIR/split_train_dev.py" \
        --input "$1" \
        --train_out "$2" \
        --dev_out "$3" \
        --ratio "$4" \
        --seed "$SEED" \
        --stratify_fields task_dataset,target
}

build_phase2_experience_replay() {
    local data_dir="$1"
    local source_train="$data_dir/$STAGE2_DIR_NAME/train_joint.json"
    local output_train="$data_dir/$STAGE2_DIR_NAME/train.json"

    if [ "$P2_USE_EXPERIENCE_REPLAY" = "0" ]; then
        return
    fi

    cp "$output_train" "$source_train"
    log "Building Phase 2 experience replay train file: $P2_TASK_MEMORY_RATIOS"
    python "$SCRIPT_DIR/build_experience_replay.py" \
        --combined_file "$source_train" \
        --new_task SemEval2018 \
        --task_memory_ratios "$P2_TASK_MEMORY_RATIOS" \
        --output "$output_train" \
        --seed "$SEED" | tee "$data_dir/$STAGE2_DIR_NAME/replay_stats.txt"
}

remove_checkpoints_before_step() {
    local output_dir="$1"
    local min_step="$2"
    local path step

    for path in "$output_dir"/checkpoint-*; do
        [ -d "$path" ] || continue
        step="${path##*-}"
        if [ "$step" -lt "$min_step" ]; then
            rm -rf "$path"
        fi
    done
}

select_best_checkpoint() {
    local output_dir="$1"
    python - "$output_dir" <<'PY'
import json
import math
import os
import re
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
state_path = output_dir / "trainer_state.json"
if not state_path.exists():
    raise SystemExit(f"Missing trainer_state.json: {state_path}")

state = json.load(open(state_path, encoding="utf-8"))
metric_key = "eval_macro_f1_avg"
metrics = {
    int(row["step"]): float(row[metric_key])
    for row in state.get("log_history", [])
    if "step" in row and metric_key in row
}

if os.environ.get("P1_INCLUDE_FINAL_EVAL_IN_BEST", "0") == "1":
    final_eval_path = output_dir / "eval_results.json"
    final_step = state.get("global_step")
    if final_step is not None and final_eval_path.exists():
        final_eval = json.load(open(final_eval_path, encoding="utf-8"))
        if metric_key in final_eval:
            metrics[int(final_step)] = float(final_eval[metric_key])

candidates = []
for path in output_dir.glob("checkpoint-*"):
    if not path.is_dir():
        continue
    m = re.search(r"checkpoint-(\d+)$", path.name)
    if not m:
        continue
    step = int(m.group(1))
    metric = metrics.get(step, -math.inf)
    candidates.append((metric, step, path))

if not candidates:
    raise SystemExit(f"No saved checkpoint found under {output_dir}")

metric, step, path = max(candidates, key=lambda item: (item[0], item[1]))
print(f"{step}\t{path}")
PY
}

prepare_phase1_data() {
    local data_dir="$1"
    mkdir -p "$data_dir/$STAGE1_DIR_NAME"
    log "Preparing Phase 1 data: IHD + IACV2, dev ratio=$RATIO"

    python convert_datasets.py --datasets IHC,IACV2 --seed "$SEED"
    cp data/train.json "$data_dir/$STAGE1_DIR_NAME/train_full.json"
    cp data/test.json "$data_dir/$STAGE1_DIR_NAME/test.json"
    write_phase1_task_map
    cp data/task_dataset.json "$data_dir/$STAGE1_DIR_NAME/task_dataset.json"

    split_train_dev \
        "$data_dir/$STAGE1_DIR_NAME/train_full.json" \
        "$data_dir/$STAGE1_DIR_NAME/train.json" \
        "$data_dir/$STAGE1_DIR_NAME/dev.json" \
        "$RATIO" | tee "$data_dir/$STAGE1_DIR_NAME/split_stats.txt"
}

prepare_phase2_data() {
    local data_dir="$1"
    mkdir -p "$data_dir/$STAGE2_DIR_NAME"
    if [ -n "$P2_SEMEVAL_DEV_RATIO" ]; then
        log "Preparing Phase 2 data: old tasks dev ratio=$RATIO, SemEval2018 dev ratio=$P2_SEMEVAL_DEV_RATIO"
    else
        log "Preparing Phase 2 data: IHD + IACV2 + SemEval2018, dev ratio=$RATIO"
    fi

    python convert_datasets.py --datasets IHC,IACV2 --seed "$SEED"
    cp data/train.json "$data_dir/$STAGE2_DIR_NAME/train_2task_full.json"
    cp data/test.json "$data_dir/$STAGE2_DIR_NAME/test_2task.json"

    python convert_datasets.py --datasets SemEval2018 --seed "$SEED"
    cp data/train.json "$data_dir/$STAGE2_DIR_NAME/train_semeval_full.json"
    cp data/test.json "$data_dir/$STAGE2_DIR_NAME/test_semeval.json"

    cat "$data_dir/$STAGE2_DIR_NAME/train_2task_full.json" \
        "$data_dir/$STAGE2_DIR_NAME/train_semeval_full.json" \
        > "$data_dir/$STAGE2_DIR_NAME/train_full.json"
    cat "$data_dir/$STAGE2_DIR_NAME/test_2task.json" \
        "$data_dir/$STAGE2_DIR_NAME/test_semeval.json" \
        > "$data_dir/$STAGE2_DIR_NAME/test.json"

    write_phase2_task_map
    cp data/task_dataset.json "$data_dir/$STAGE2_DIR_NAME/task_dataset.json"

    if [ -n "$P2_SEMEVAL_DEV_RATIO" ]; then
        split_train_dev \
            "$data_dir/$STAGE2_DIR_NAME/train_2task_full.json" \
            "$data_dir/$STAGE2_DIR_NAME/train_2task.json" \
            "$data_dir/$STAGE2_DIR_NAME/dev_2task.json" \
            "$RATIO" > "$data_dir/$STAGE2_DIR_NAME/split_stats_2task.txt"
        split_train_dev \
            "$data_dir/$STAGE2_DIR_NAME/train_semeval_full.json" \
            "$data_dir/$STAGE2_DIR_NAME/train_semeval.json" \
            "$data_dir/$STAGE2_DIR_NAME/dev_semeval.json" \
            "$P2_SEMEVAL_DEV_RATIO" > "$data_dir/$STAGE2_DIR_NAME/split_stats_semeval.txt"
        cat "$data_dir/$STAGE2_DIR_NAME/train_2task.json" \
            "$data_dir/$STAGE2_DIR_NAME/train_semeval.json" \
            > "$data_dir/$STAGE2_DIR_NAME/train.json"
        cat "$data_dir/$STAGE2_DIR_NAME/dev_2task.json" \
            "$data_dir/$STAGE2_DIR_NAME/dev_semeval.json" \
            > "$data_dir/$STAGE2_DIR_NAME/dev.json"
        {
            echo "Combined Phase 2 split with SemEval2018-specific dev ratio"
            echo
            cat "$data_dir/$STAGE2_DIR_NAME/split_stats_2task.txt"
            echo
            cat "$data_dir/$STAGE2_DIR_NAME/split_stats_semeval.txt"
        } | tee "$data_dir/$STAGE2_DIR_NAME/split_stats.txt"
    else
        split_train_dev \
            "$data_dir/$STAGE2_DIR_NAME/train_full.json" \
            "$data_dir/$STAGE2_DIR_NAME/train.json" \
            "$data_dir/$STAGE2_DIR_NAME/dev.json" \
            "$RATIO" | tee "$data_dir/$STAGE2_DIR_NAME/split_stats.txt"
    fi

    build_phase2_experience_replay "$data_dir"
}

run_phase1() {
    local data_dir="$1"
    local run_name="${TAG}_Stage1_noach_step${P1_MAX_STEPS}"
    local output_dir="$WORK_ROOT/saved/$run_name"
    local tb_dir="$WORK_ROOT/logs/tensorboard/$run_name"
    local master_port

    write_phase1_task_map
    P1_ACH_ARGS=()
    if [ "$P1_USE_ACHIEVEMENT_LOSS" != "0" ]; then
        P1_ACH_ARGS=(
            --use_achievement_loss
            --achievement_gamma "$ACH_GAMMA"
            --achievement_margin "$ACH_MARGIN"
            --achievement_ema_alpha "$ACH_EMA"
            --achievement_targets "$P1_ACH_TARGETS"
            --achievement_warmup_ratio 0.0
            --achievement_weight_floor "$P1_WEIGHT_FLOOR"
        )
    fi

    master_port=$(shuf -n 1 -i 10000-65535)
    log "Training Phase 1: $run_name"
    deepspeed --num_gpus="$NUM_GPUS_TRAIN" --master_port "$master_port" run_qwen.py \
        --deepspeed src/ds_bf16.config \
        --do_train \
        --do_eval \
        --train_file "$data_dir/$STAGE1_DIR_NAME/train.json" \
        --validation_file "$data_dir/$STAGE1_DIR_NAME/dev.json" \
        --cache_dir "$data_dir/$STAGE1_DIR_NAME/cache" \
        --prompt_column input \
        --response_column target \
        --overwrite_cache \
        --model_name_or_path "$MODEL" \
        --output_dir "$output_dir" \
        --overwrite_output_dir \
        --max_source_length 1024 \
        --max_target_length 512 \
        --per_device_train_batch_size "$P1_BS" \
        --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
        --gradient_accumulation_steps "$P1_GRAD_ACCUM" \
        --max_steps "$P1_MAX_STEPS" \
        --logging_steps 5 \
        --save_steps "$P1_SAVE_STEPS" \
        --save_total_limit "$P1_SAVE_TOTAL_LIMIT" \
        --evaluation_strategy steps \
        --eval_delay "$P1_EVAL_DELAY" \
        --eval_steps "$P1_EVAL_STEPS" \
        --eval_accumulation_steps 1 \
        --eval_metric_mode macro \
        --metric_for_best_model eval_macro_f1_avg \
        --greater_is_better true \
        --learning_rate "$P1_LR" \
        --weight_decay "$P1_WD" \
        --lora_rank "$LORA_RANK" \
        --lora_alpha "$LORA_ALPHA" \
        --trainable "$TRAINABLE" \
        --modules_to_save null \
        --lora_dropout 0.1 \
        --seed "$SEED" \
        --bf16 \
        --save_only_model \
        --generation_max_length 16 \
        --lora_name moelora \
        --task_num 2 \
        --expert_num "$EXPERT_NUM" \
        "${P1_ACH_ARGS[@]}" \
        --lb_loss_coeff 0 \
        --lr_scheduler_type linear \
        --report_to tensorboard \
        --logging_dir "$tb_dir" \
        --predict_with_generate

    remove_checkpoints_before_step "$output_dir" "$P1_EVAL_DELAY"

    local best_info
    best_info="$(select_best_checkpoint "$output_dir")"
    P1_BEST_STEP="${best_info%%$'\t'*}"
    P1_CKPT="${best_info#*$'\t'}"
    log "Best Phase 1 checkpoint: step=$P1_BEST_STEP path=$P1_CKPT"
    printf '%s\n' "$P1_BEST_STEP" > "$WORK_ROOT/p1_best_step.txt"
    printf '%s\n' "$P1_CKPT" > "$WORK_ROOT/p1_best_checkpoint.txt"
}

run_phase2() {
    local data_dir="$1"
    local run_name="${TAG}_Stage2_IHD${IHD_RATIO}_IACV${IACV2_RATIO}_warmup${ACH_WARMUP_RATIO}_${NUM_GPUS_TRAIN}gpu_p2step${P2_MAX_STEPS}"
    local expanded_ckpt="$WORK_ROOT/saved/expanded/$run_name"
    local output_dir="$WORK_ROOT/saved/$run_name"
    local result_dir
    local tb_dir="$WORK_ROOT/logs/tensorboard/$run_name"
    local master_port
    local predict_start first_predict_step step
    local p2_predict_steps=()

    predict_start=$(((P2_MAX_STEPS * P2_PREDICT_START_RATIO + 99) / 100))
    first_predict_step=$((((predict_start + P2_EVAL_STEPS - 1) / P2_EVAL_STEPS) * P2_EVAL_STEPS))
    for ((step=first_predict_step; step<=P2_MAX_STEPS; step+=P2_EVAL_STEPS)); do
        p2_predict_steps+=("$step")
    done
    if [ "${#p2_predict_steps[@]}" -eq 0 ] || [ "${p2_predict_steps[$((${#p2_predict_steps[@]} - 1))]}" -ne "$P2_MAX_STEPS" ]; then
        p2_predict_steps+=("$P2_MAX_STEPS")
    fi

    if [ -z "$P1_CKPT" ] || [ ! -d "$P1_CKPT" ]; then
        echo "ERROR: P1_CKPT is required and must be an existing checkpoint dir: $P1_CKPT" >&2
        exit 1
    fi

    log "Expanding Phase 1 checkpoint to task_num=3: $P1_CKPT"
    python expand_checkpoint.py \
        --src "$P1_CKPT" \
        --dst "$expanded_ckpt" \
        --new_task_num 3 \
        --init_from "$P2_INIT_FROM" \
        --freeze_expert_ids "$FREEZE_EXPERT_IDS" \
        --seed "$SEED"

    write_phase2_task_map
    P2_ACH_ARGS=()
    if [ "$P2_USE_ACHIEVEMENT_LOSS" != "0" ]; then
        P2_ACH_ARGS=(
            --use_achievement_loss
            --achievement_gamma "$ACH_GAMMA"
            --achievement_margin "$ACH_MARGIN"
            --achievement_ema_alpha "$ACH_EMA"
            --achievement_targets "$P2_ACH_TARGETS"
            --achievement_warmup_ratio "$ACH_WARMUP_RATIO"
            --achievement_weight_floor "$P2_WEIGHT_FLOOR"
        )
    fi

    master_port=$(shuf -n 1 -i 10000-65535)
    log "Training Phase 2: $run_name"
    log "Phase 2 kept checkpoints: ${p2_predict_steps[*]}"
    deepspeed --num_gpus="$NUM_GPUS_TRAIN" --master_port "$master_port" run_qwen.py \
        --deepspeed src/ds_bf16.config \
        --do_train \
        --do_eval \
        --train_file "$data_dir/$STAGE2_DIR_NAME/train.json" \
        --validation_file "$data_dir/$STAGE2_DIR_NAME/${P2_VALIDATION_SPLIT}.json" \
        --cache_dir "$data_dir/$STAGE2_DIR_NAME/cache" \
        --prompt_column input \
        --response_column target \
        --overwrite_cache \
        --model_name_or_path "$MODEL" \
        --peft_path "$expanded_ckpt" \
        --output_dir "$output_dir" \
        --overwrite_output_dir \
        --max_source_length 1024 \
        --max_target_length 512 \
        --per_device_train_batch_size "$P2_BS" \
        --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
        --gradient_accumulation_steps "$P2_GRAD_ACCUM" \
        --max_steps "$P2_MAX_STEPS" \
        --logging_steps 5 \
        --save_steps "$P2_SAVE_STEPS" \
        --save_total_limit "${P2_SAVE_TOTAL_LIMIT:-${#p2_predict_steps[@]}}" \
        --evaluation_strategy steps \
        --eval_delay "$P2_EVAL_DELAY" \
        --eval_steps "$P2_EVAL_STEPS" \
        --eval_accumulation_steps 1 \
        --eval_metric_mode macro \
        --metric_for_best_model eval_macro_f1_avg \
        --greater_is_better true \
        --learning_rate "$P2_LR" \
        --weight_decay "$P2_WD" \
        --lora_rank "$LORA_RANK" \
        --lora_alpha "$LORA_ALPHA" \
        --trainable "$TRAINABLE" \
        --modules_to_save null \
        --lora_dropout 0.1 \
        --seed "$SEED" \
        --bf16 \
        --save_only_model \
        --generation_max_length 16 \
        --lora_name moelora \
        --task_num 3 \
        --expert_num "$EXPERT_NUM" \
        --freeze_expert_ids "$FREEZE_EXPERT_IDS" \
        --replay_ratio "$P2_REPLAY_RATIO" \
        --replay_task_ids "1,2" \
        "${P2_ACH_ARGS[@]}" \
        --lb_loss_coeff 0 \
        --lr_scheduler_type linear \
        --report_to tensorboard \
        --logging_dir "$tb_dir" \
        --predict_with_generate

    remove_checkpoints_before_step "$output_dir" "$first_predict_step"

    log "Phase 2 training finished. Test evaluation is handled by experiments/eval.bash with an explicit checkpoint path."

}

DATA_DIR="$WORK_ROOT/data/$TAG"

log "Starting full two-stage training run"
log "WORK_ROOT=$WORK_ROOT"
log "TAG=$TAG RATIO=$RATIO SEED=$SEED"
log "GPU train=$NUM_GPUS_TRAIN pred=$NUM_GPUS_PRED"
log "Phase 1: no-ach default=$P1_USE_ACHIEVEMENT_LOSS, steps=$P1_MAX_STEPS, eval_delay=$P1_EVAL_DELAY"
log "Phase 2: ach=$P2_USE_ACHIEVEMENT_LOSS, warmup=$ACH_WARMUP_RATIO, eval_delay=$P2_EVAL_DELAY, floor=$P2_WEIGHT_FLOOR, replay=$P2_TASK_MEMORY_RATIOS, steps=$P2_MAX_STEPS"

if [ "$RUN_PHASE1" != "0" ]; then
    prepare_phase1_data "$DATA_DIR"
    run_phase1 "$DATA_DIR"
elif [ -z "$P1_CKPT" ]; then
    echo "ERROR: RUN_PHASE1=0 requires P1_CKPT=/path/to/checkpoint" >&2
    exit 1
fi

if [ "$RUN_PHASE2" != "0" ]; then
    prepare_phase2_data "$DATA_DIR"
    run_phase2 "$DATA_DIR"
fi

log "Completed full two-stage training run"
