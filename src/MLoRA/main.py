#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import json
import logging
import os
import sys

import jieba
import numpy as np
import transformers
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_chinese import Rouge
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    set_seed,
)

sys.path.append("./")

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.MLoRA.trainer_seq2seq import Seq2SeqTrainer
from src.MLoRA.achievement_loss import AchievementWeightedLoss
from src.MLoRA.peft import PeftModel, TaskType, get_peft_model
from src.MLoRA.peft import LoraConfig, AdaLoraConfig
from src.MLoRA.peft import MMOELoraConfigS
from src.data_processor.chatglm import chatglm1_train, chatglm1_eval
from src.data_processor.qwen2 import qwen2_train, qwen2_eval
from src.data_processor.collator import LongestSequenceCollator

logger = logging.getLogger(__name__)

def main(parser):

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.batched_training = data_args.batched_training # for batched training
    # Pass input_ids through gather so compute_metrics can detect task from inputs
    training_args.include_inputs_for_metrics = True
    # if model_args.department:   # for the department
    #     model_args.task_num = model_args.depart_num

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Suppress GenerationConfig print spam during eval
    logging.getLogger("transformers_modules.chatglm3-6b.configuration_chatglm").setLevel(logging.WARNING)
    logging.getLogger("transformers.generation.utils").setLevel(logging.WARNING)
    logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load dataset
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]

    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        token=True if model_args.use_auth_token else None,
    )
    print("raw_datasets: ", raw_datasets)
    # print("raw_datasets: ", len(raw_datasets["train"]))

    # Detect model type
    model_basename = model_args.model_name_or_path.split("/")[-1].lower()
    is_qwen2 = "qwen" in model_basename
    is_chatglm = "chatglm" in model_basename

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    if not is_qwen2:
        config.pre_seq_len = model_args.pre_seq_len
        config.prefix_projection = model_args.prefix_projection

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    # Qwen2 requires pad_token for padding
    if is_qwen2 and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only models need left-padding for correct generation
    if is_qwen2:
        tokenizer.padding_side = "left"

    if is_qwen2:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype="auto",
        ).cuda()
    else:
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True
        ).half().cuda()

    if model_args.peft_path is not None:
        logger.info("Peft from pre-trained model")
        # Need is_trainable=True for: resume training, freeze_experts (Phase 2A),
        # freeze_expert_ids (Phase 2B partial), or do_train (Phase 2B full fine-tune)
        needs_trainable = (training_args.resume_from_checkpoint is not None
                          or model_args.freeze_experts
                          or model_args.freeze_expert_ids is not None
                          or training_args.do_train)
        if needs_trainable:
            model = PeftModel.from_pretrained(model, model_args.peft_path, is_trainable=True)
        else:
            model = PeftModel.from_pretrained(model, model_args.peft_path, is_trainable=False)
    else:
        logger.info("Init new peft model")
        target_modules = model_args.trainable.split(',')
        modules_to_save = model_args.modules_to_save.split(',') if model_args.modules_to_save!="null" else None
        lora_rank = model_args.lora_rank
        lora_dropout = model_args.lora_dropout
        lora_alpha = model_args.lora_alpha
        print(target_modules)

        kwargs = {}
        if model_args.lora_name == "adalora":
            TargetLoraConfig = AdaLoraConfig
            task_type = TaskType.CAUSAL_LM
        elif model_args.lora_name == "moelora":
            TargetLoraConfig = MMOELoraConfigS
            kwargs = {
                  "task_num": model_args.task_num,
                  "task_embedding_dim": model_args.task_embedding_dim,
                  "expert_num": model_args.expert_num,
                  }
            task_type = TaskType.CAUSAL_LMS
        else:
            TargetLoraConfig = LoraConfig
            task_type = TaskType.CAUSAL_LM
        
        peft_config = TargetLoraConfig(
            task_type=task_type,
            target_modules=target_modules,
            inference_mode=False,
            r=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            modules_to_save=modules_to_save,
            **kwargs
        )
        model = get_peft_model(model, peft_config)


    model.print_trainable_parameters()

    # ── Phase 2 freeze: only train new task embedding(s) ──
    if model_args.freeze_experts and model_args.peft_path is not None:
        # Step 1: freeze everything
        for name, param in model.named_parameters():
            param.requires_grad = False

        # Step 2: unfreeze only lora_task_embedding
        # Note: the whole Embedding tensor is unfrozen, but since training data
        # only contains the new task_id, gradients only flow through that row.
        # Set weight_decay=0 to avoid decay on old rows (0,1,2).
        unfrozen = []
        for name, param in model.named_parameters():
            if "lora_task_embedding" in name:
                param.requires_grad = True
                unfrozen.append(f"{name} {list(param.shape)}")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"[freeze_experts] Frozen all except task embedding.")
        logger.info(f"[freeze_experts] Trainable: {trainable_params}/{total_params} "
                    f"({100*trainable_params/total_params:.4f}%)")
        for u in unfrozen:
            logger.info(f"[freeze_experts]   Unfrozen: {u}")
        if training_args.weight_decay > 0:
            logger.warning(f"[freeze_experts] weight_decay={training_args.weight_decay} > 0! "
                           f"This will decay old embedding rows. Set --weight_decay 0 for Phase 2.")

    # ── Phase 2B partial freeze: freeze specified experts, train the rest + gate + embedding ──
    if model_args.freeze_expert_ids is not None and model_args.peft_path is not None:
        import re as _re_freeze
        frozen_ids = set(int(x.strip()) for x in model_args.freeze_expert_ids.split(","))
        logger.info(f"[partial_freeze] Freezing expert IDs: {sorted(frozen_ids)}")

        frozen_params = []
        unfrozen_params = []
        for name, param in model.named_parameters():
            # Match expert index from names like loraA.3.weight or loraB.5.mlp.weight
            expert_match = _re_freeze.search(r'lora[AB]\.(\d+)\.', name)
            if expert_match:
                expert_id = int(expert_match.group(1))
                if expert_id in frozen_ids:
                    param.requires_grad = False
                    frozen_params.append(name)
                else:
                    param.requires_grad = True
                    unfrozen_params.append(name)
            elif "lora_task_embedding" in name:
                # Embedding trainable: only row for new task_id gets gradients
                param.requires_grad = True
                unfrozen_params.append(name)
            elif "lora_gate" in name:
                if getattr(model_args, "no_freeze_gate", False):
                    param.requires_grad = True
                    unfrozen_params.append(name)
                else:
                    # Gate MUST be frozen: it's shared across all tasks.
                    # Training gate with single-task data destroys other tasks' routing.
                    param.requires_grad = False
                    frozen_params.append(name)
            else:
                # Base model params — keep frozen (they're already frozen by PeftModel)
                pass

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_unfrozen_experts = model_args.expert_num - len(frozen_ids)
        gate_status = "gate trainable" if getattr(model_args, "no_freeze_gate", False) else "gate frozen"
        logger.info(f"[partial_freeze] Frozen {len(frozen_ids)} experts, {gate_status}, "
                    f"unfrozen {n_unfrozen_experts} experts + embedding")
        logger.info(f"[partial_freeze] Trainable: {trainable_params:,}/{total_params:,} "
                    f"({100*trainable_params/total_params:.4f}%)")
        # Show unique patterns of frozen/unfrozen for clarity
        frozen_expert_ids_seen = set()
        unfrozen_expert_ids_seen = set()
        for name in frozen_params:
            m = _re_freeze.search(r'lora[AB]\.(\d+)\.', name)
            if m:
                frozen_expert_ids_seen.add(int(m.group(1)))
        for name in unfrozen_params:
            m = _re_freeze.search(r'lora[AB]\.(\d+)\.', name)
            if m:
                unfrozen_expert_ids_seen.add(int(m.group(1)))
        logger.info(f"[partial_freeze]   Frozen expert IDs: {sorted(frozen_expert_ids_seen)}")
        logger.info(f"[partial_freeze]   Trainable expert IDs: {sorted(unfrozen_expert_ids_seen)}")
        if any("lora_gate" in n for n in frozen_params):
            logger.info(f"[partial_freeze]   Gate: FROZEN (shared across tasks)")
        if any("lora_task_embedding" in n for n in unfrozen_params):
            logger.info(f"[partial_freeze]   Task Embedding: TRAINABLE")

    # Inject label_smoothing into model config so the chunked CE path uses it
    # Path: PeftModel.base_model (LoraModel).model (Qwen2ForCausalLM).config
    if training_args.label_smoothing_factor > 0:
        _injected = False
        # Try PeftModel → LoraModel → Qwen2ForCausalLM → config
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
            model.base_model.model.config._label_smoothing = training_args.label_smoothing_factor
            _injected = True
        elif hasattr(model, 'config'):
            model.config._label_smoothing = training_args.label_smoothing_factor
            _injected = True
        if _injected:
            logger.info(f"Injected label_smoothing={training_args.label_smoothing_factor} into model config "
                        f"(per_sample_loss CE path)")
        else:
            logger.warning(f"Could not inject label_smoothing into model config!")

    task_flag = False   # flag whether generate task_id from dataset
    depart_flag = False  # flag whether use the department and entity
    if (model_args.lora_name == "moelora"):
        task_flag = True


    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    prompt_column = data_args.prompt_column
    response_column = data_args.response_column
    history_column = data_args.history_column
    
    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length

    def print_dataset_example(example):
        print("input_ids: ",example["input_ids"])
        print("inputs: ", tokenizer.decode(example["input_ids"]))
        print("label_ids: ", example["labels"])
        #print("labels: ", tokenizer.decode(example["labels"])) # For ChatGLMv2
    
    if is_chatglm:
        preprocess_function_train = chatglm1_train(data_args, model_args, prompt_column,
                                                   response_column, history_column, prefix,
                                                   tokenizer, task_flag, depart_flag)
        preprocess_function_eval = chatglm1_eval(data_args, model_args, prompt_column,
                                                 response_column, history_column, prefix,
                                                 tokenizer, task_flag, depart_flag)
    elif is_qwen2:
        preprocess_function_train = qwen2_train(data_args, model_args, prompt_column,
                                                response_column, history_column, prefix,
                                                tokenizer, task_flag, depart_flag)
        preprocess_function_eval = qwen2_eval(data_args, model_args, prompt_column,
                                              response_column, history_column, prefix,
                                              tokenizer, task_flag, depart_flag)
    else:
        raise ValueError(f"Unsupported model: {model_args.model_name_or_path}. "
                         f"Supported: chatglm-6b, chatglm3-6b, qwen2-*")

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # Save per-sample task IDs before tokenization removes columns
        # (used by TaskBalancedSampler for balanced batching)
        _train_task_ids = None
        if "task_dataset" in train_dataset.column_names:
            import json as _json_tb
            _tb_map_path = "data/task_dataset.json"
            if os.path.exists(_tb_map_path):
                _tb_str2id = _json_tb.load(open(_tb_map_path))["str2id"]
                _train_task_ids = [_tb_str2id.get(t, 0) for t in train_dataset["task_dataset"]]
                logger.info(f"Task-balanced sampler: {len(_train_task_ids)} samples, "
                            f"tasks={set(_train_task_ids)}")
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )
        print_dataset_example(train_dataset[0])
        print_dataset_example(train_dataset[1])
        train_dataset.set_format("torch")

    eval_task_list = None
    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        # Save task_dataset before preprocessing removes it
        eval_task_list = eval_dataset["task_dataset"] if "task_dataset" in eval_dataset.column_names else None
        # Use train preprocessing for eval_loss (needs same-length input_ids/labels);
        # use eval preprocessing for predict_with_generate (separate input/target).
        eval_preprocess = preprocess_function_eval if training_args.predict_with_generate else preprocess_function_train
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                eval_preprocess,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on validation dataset",
            )
        print_dataset_example(eval_dataset[0])
        print_dataset_example(eval_dataset[1])
        eval_dataset.set_format("torch")

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on prediction dataset",
            )
        print_dataset_example(predict_dataset[0])
        print_dataset_example(predict_dataset[1])
        predict_dataset.set_format("torch")

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    
    # if training_args.do_train:  # only conduct padding for do_train
    #     data_collator = DataCollatorForSeq2Seq(
    #         tokenizer,
    #         model=model,
    #         label_pad_token_id=label_pad_token_id,
    #         pad_to_multiple_of=tokenizer.pad_token_id,
    #         padding="longest",
    #     )
    # else:
    if training_args.do_train or (task_flag and (training_args.do_eval or training_args.do_predict)):
        # MOELoRA needs task_id in every batch; LongestSequenceCollator is data-only (no grad)
        data_collator = LongestSequenceCollator(tokenizer, task_flag, depart_flag)
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=None,
            padding=False
        )

    # Per-task label definitions
    TASK_LABELS = {
        "NLI": ["A", "B", "C", "D"],
        "IHD": ["0", "1"],
        "SemEval2018": ["0", "1"],
        "IACV2": ["0", "1"],
        "Sarca": ["0", "1"],
        "Metap": ["0", "1"],
        "Hypo": ["0", "1"],
    }
    BINARY_TASKS = {"IHD", "SemEval2018", "IACV2", "Sarca", "Metap", "Hypo"}

    # Metric
    def _detect_task_from_input(text):
        """Detect task type from instruction keywords in decoded input text."""
        t = text[:200].lower()
        if "hate speech" in t:
            return "IHD"
        if "online debate" in t or "internet argument" in t:
            return "IACV2"
        if "irony detection" in t or "ironic" in t:
            return "SemEval2018"
        if "sarcasm" in t:
            return "SemEval2018"
        if "natural language inference" in t:
            return "NLI"
        if "metaphor" in t:
            return "Metap"
        if "hyperbole" in t:
            return "Hypo"
        return "UNKNOWN"

    def _extract_answer(text):
        """Extract the core answer from model output or label text.
        Strips [gMASK]sop prefix from labels (ChatGLM3) and 'Output: ' prefix from predictions."""
        import re
        # Remove [gMASK]sop prefix (from ChatGLM3 labels)
        text = re.sub(r'^\[gMASK\]sop\s*', '', text)
        # Remove 'Output: ' or 'Output:' prefix (from model generation)
        text = re.sub(r'^Output:\s*', '', text)
        # Remove Qwen2 special tokens that may leak through
        text = re.sub(r'<\|im_start\|>|<\|im_end\|>', '', text)
        # Strip role prefix that may appear in decoded Qwen2 output
        text = re.sub(r'^(system|user|assistant)\s*\n?', '', text)
        # Take only the first line/token as the answer
        text = text.split('\n')[0].strip()
        return text

    def _normalize_answer(text, valid_labels):
        """Use regex to extract a valid label, prefer the last match."""
        import re
        escaped = [re.escape(l) for l in valid_labels]
        pattern = r'(?:^|(?<=\s))(' + '|'.join(escaped) + r')(?:\s|$)'
        matches = list(re.finditer(pattern, text))
        if matches:
            return matches[-1].group(1)
        tokens = text.split()
        if tokens and tokens[-1] in valid_labels:
            return tokens[-1]
        return text

    def compute_metrics(eval_preds):
        inputs = getattr(eval_preds, 'inputs', None)
        preds = eval_preds.predictions
        labels = eval_preds.label_ids
        if isinstance(preds, tuple):
            preds = preds[0]
        # preds may contain -100 from pad_across_processes; replace before decoding
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [_extract_answer(p) for p in decoded_preds]
        decoded_labels = [_extract_answer(l) for l in decoded_labels]

        # Overall metrics
        overall_acc = accuracy_score(decoded_labels, decoded_preds)
        result = {"accuracy": overall_acc}

        metric_mode = (model_args.eval_metric_mode or "loss").lower()
        if metric_mode == "loss":
            # keep compute_metrics minimal when user wants eval_loss-driven training
            return result

        # Determine per-task assignments from gathered inputs (GPU-order-safe)
        task_list = None
        if inputs is not None:
            # inputs may contain -100 from pad_across_processes; replace before decoding
            inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
            # inputs may be left-padded; decode full sequence, then detect task keyword
            decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
            task_list = [_detect_task_from_input(t) for t in decoded_inputs]
        elif eval_task_list:
            # Fallback when include_inputs_for_metrics is off (single-GPU only)
            task_list = list(eval_task_list)

        use_per_task = False
        if task_list:
            n = min(len(task_list), len(decoded_preds), len(decoded_labels))
            if n > 0:
                task_list = task_list[:n]
                decoded_preds = decoded_preds[:n]
                decoded_labels = decoded_labels[:n]
                use_per_task = True

        if use_per_task:
            from collections import defaultdict
            task_preds = defaultdict(list)
            task_golds = defaultdict(list)
            for pred, gold, task in zip(decoded_preds, decoded_labels, task_list):
                task_preds[task].append(pred)
                task_golds[task].append(gold)

            f1_sum = 0
            n_tasks = 0
            for task in sorted(task_preds.keys()):
                t_preds = task_preds[task]
                t_golds = task_golds[task]
                t_labels = TASK_LABELS.get(task, sorted(set(t_golds)))

                # regex normalize predictions and golds; clamp invalid values to default
                default_label = t_labels[0] if t_labels else "0"
                t_preds = [_normalize_answer(p, t_labels) for p in t_preds]
                t_preds = [p if p in t_labels else default_label for p in t_preds]
                t_golds = [_normalize_answer(g, t_labels) for g in t_golds]
                t_golds = [g if g in t_labels else default_label for g in t_golds]

                t_acc = accuracy_score(t_golds, t_preds)

                if metric_mode == "binary" and task in BINARY_TASKS and "1" in t_labels:
                    t_precision = precision_score(
                        t_golds, t_preds, labels=t_labels, average="binary", pos_label="1", zero_division=0
                    )
                    t_recall = recall_score(
                        t_golds, t_preds, labels=t_labels, average="binary", pos_label="1", zero_division=0
                    )
                    t_f1 = f1_score(
                        t_golds, t_preds, labels=t_labels, average="binary", pos_label="1", zero_division=0
                    )
                    result[f"{task}_accuracy"] = t_acc
                    result[f"{task}_precision"] = t_precision
                    result[f"{task}_recall"] = t_recall
                    result[f"{task}_f1"] = t_f1
                else:
                    t_f1 = f1_score(
                        t_golds, t_preds, labels=t_labels, average="macro", zero_division=0
                    )
                    result[f"{task}_accuracy"] = t_acc
                    result[f"{task}_f1"] = t_f1

                f1_sum += t_f1
                n_tasks += 1

            if n_tasks > 0:
                if metric_mode == "binary":
                    result["binary_f1_avg"] = f1_sum / n_tasks
                else:
                    result["macro_f1_avg"] = f1_sum / n_tasks
        else:
            all_labels = sorted(set(decoded_labels))
            default_label = all_labels[0] if all_labels else "0"
            norm_preds = [_normalize_answer(p, all_labels) for p in decoded_preds]
            norm_preds = [p if p in all_labels else default_label for p in norm_preds]
            norm_labels = [_normalize_answer(l, all_labels) for l in decoded_labels]
            norm_labels = [l if l in all_labels else default_label for l in norm_labels]
            if metric_mode == "binary" and "1" in all_labels and len(all_labels) == 2:
                result["f1"] = f1_score(
                    norm_labels, norm_preds, labels=all_labels, average="binary", pos_label="1", zero_division=0
                )
            else:
                result["macro_f1"] = f1_score(
                    norm_labels, norm_preds, labels=all_labels, average="macro", zero_division=0
                )

        return result

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Initialize achievement-based multi-task loss if enabled
    if model_args.use_achievement_loss and task_flag:
        import json as _json
        # Parse per-task SOTA targets from --achievement_targets (e.g. "IHD:0.8,Sarca:0.83")
        sota_targets = None
        if model_args.achievement_targets:
            task_map_path = "data/task_dataset.json"
            if os.path.exists(task_map_path):
                str2id = _json.load(open(task_map_path))["str2id"]
                sota_targets = {}
                for pair in model_args.achievement_targets.split(","):
                    name, val = pair.strip().split(":")
                    name = name.strip()
                    val = float(val.strip())
                    if name in str2id:
                        sota_targets[str2id[name]] = val
                    else:
                        logger.warning(f"Achievement target: task '{name}' not found in {task_map_path}, skipping")
                logger.info(f"Achievement SOTA targets: {sota_targets}")

        # Count per-task training samples for IDF correction
        task_sample_counts = None
        if training_args.do_train and "train" in raw_datasets:
            task_map_path2 = "data/task_dataset.json"
            if os.path.exists(task_map_path2):
                str2id2 = _json.load(open(task_map_path2))["str2id"]
                from collections import Counter
                raw_train = raw_datasets["train"]
                if "task_dataset" in raw_train.column_names:
                    td_counts = Counter(raw_train["task_dataset"])
                    task_sample_counts = {}
                    for name, cnt in td_counts.items():
                        if name in str2id2:
                            task_sample_counts[str2id2[name]] = cnt
                    logger.info(f"Task sample counts for IDF: {task_sample_counts}")

        trainer.achievement_loss = AchievementWeightedLoss(
            task_num=model_args.task_num,
            gamma=model_args.achievement_gamma,
            margin=model_args.achievement_margin,
            sota_targets=sota_targets,
            ema_alpha=model_args.achievement_ema_alpha,
            task_sample_counts=task_sample_counts,
            weight_floor=model_args.achievement_weight_floor,
        )
        targets_str = {k: f"{v:.2f}" for k, v in (sota_targets or {}).items()}
        logger.info(f"Achievement loss enabled: gamma={model_args.achievement_gamma}, "
                     f"margin={model_args.achievement_margin}, ema_alpha={model_args.achievement_ema_alpha}, "
                     f"weight_floor={model_args.achievement_weight_floor}, "
                     f"warmup_ratio={model_args.achievement_warmup_ratio}, "
                     f"lb_loss_coeff={model_args.lb_loss_coeff}, "
                     f"task_num={model_args.task_num}, targets={targets_str or 'default(1.0)'}")
        trainer.achievement_warmup_ratio = model_args.achievement_warmup_ratio
        trainer.lb_loss_coeff = model_args.lb_loss_coeff

    # Routing divergence loss
    if model_args.routing_div_coeff > 0 and model_args.routing_div_frozen_tasks:
        _div_frozen = set(int(x.strip()) for x in model_args.routing_div_frozen_tasks.split(","))
        trainer.routing_div_coeff = model_args.routing_div_coeff
        trainer.routing_div_frozen_tasks = _div_frozen
        logger.info(f"Routing divergence loss: coeff={model_args.routing_div_coeff}, "
                     f"frozen_tasks={sorted(_div_frozen)}")

    # Task-balanced sampler (disabled — natural sampling, no oversampling)
    # if training_args.do_train and _train_task_ids is not None:
    #     trainer._train_task_ids = _train_task_ids
    #     trainer._max_oversample = 2.0
    #     logger.info(f"Task-balanced sampler injected (max_oversample={trainer._max_oversample})")

    # ── Dynamic replay: resample old-task data each epoch ──
    if (training_args.do_train
            and model_args.replay_task_ids is not None
            and model_args.replay_ratio < 1.0
            and _train_task_ids is not None):
        import random as _rng
        from transformers import TrainerCallback

        _replay_ids_set = set(int(x.strip()) for x in model_args.replay_task_ids.split(","))
        _replay_ratio = model_args.replay_ratio

        # Split indices: new-task (always keep) vs replay-task (subsample each epoch)
        _new_task_indices = [i for i, tid in enumerate(_train_task_ids) if tid not in _replay_ids_set]
        _replay_pool = [i for i, tid in enumerate(_train_task_ids) if tid in _replay_ids_set]
        _replay_n = max(1, int(len(_replay_pool) * _replay_ratio))

        logger.info(f"[replay] New-task samples: {len(_new_task_indices)} (always kept)")
        logger.info(f"[replay] Replay pool: {len(_replay_pool)} → subsample {_replay_n} each epoch "
                     f"(ratio={_replay_ratio}, task_ids={sorted(_replay_ids_set)})")

        _full_train_dataset = train_dataset  # tokenized full dataset
        _replay_rng = _rng.Random(training_args.seed)  # fixed seed, advances each epoch

        class ReplayResampleCallback(TrainerCallback):
            def on_epoch_begin(self, args, state, control, **kwargs):
                sampled = _replay_rng.sample(_replay_pool, _replay_n)
                subset_indices = sorted(_new_task_indices + sampled)
                trainer.train_dataset = _full_train_dataset.select(subset_indices)
                if state.global_step == 0 or (state.global_step % args.logging_steps == 0):
                    logger.info(f"[replay] Epoch resample: {len(subset_indices)} samples "
                                f"(new={len(_new_task_indices)} + replay={len(sampled)})")

        trainer.add_callback(ReplayResampleCallback())

        # Also set initial dataset for epoch 0
        _init_sampled = _replay_rng.sample(_replay_pool, _replay_n)
        _init_indices = sorted(_new_task_indices + _init_sampled)
        trainer.train_dataset = _full_train_dataset.select(_init_indices)
        logger.info(f"[replay] Initial dataset: {len(_init_indices)} samples")

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        trainer.save_model()  # Save best model (or final model) to output_dir

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval", do_sample=False, max_new_tokens=data_args.max_target_length)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # 读取原test file
        list_test_samples = []
        with open(data_args.test_file, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                list_test_samples.append(line)

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_new_tokens=data_args.max_target_length,
            do_sample=False,
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        #trainer.log_metrics("predict", metrics)
        #trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                labels = tokenizer.batch_decode(
                    predict_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                labels = [label.strip() for label in labels]
                assert len(labels) == len(list_test_samples)

                output_prediction_file = os.path.join(training_args.output_dir, "test_predictions.json")
                output_plain_file = os.path.join(training_args.output_dir, "generated_predictions.txt")

                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    for idx, (p, l) in enumerate(zip(predictions, labels)):
                        samp = dict(list_test_samples[idx])  # copy to avoid mutating original
                        samp["prediction"] = p  # model output
                        # samp["target"] stays as gold label
                        res = json.dumps(samp, ensure_ascii=False)
                        writer.write(f"{res}\n")

                with open(output_plain_file, "w", encoding="utf-8") as writer:
                    for p in predictions:
                        writer.write(f"{p}\n")
                logger.info(f"Plain predictions saved to {output_plain_file}")

    return results



if __name__ == "__main__":
    main()
