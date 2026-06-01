from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    lora_name: Optional[str] = field(
        default="lora", metadata={"help": "LoRA Type"}
    )
    ptuning_checkpoint: str = field(
        default=None, metadata={"help": "Path to p-tuning v2 checkpoints"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    quantization_bit: Optional[int] = field(
        default=None
    )
    pre_seq_len: Optional[int] = field(
        default=None
    )
    prefix_projection: bool = field(
        default=False
    )

    trainable: Optional[str] = field(default="q_proj,v_proj")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_alpha: Optional[float] = field(default=32.)
    modules_to_save: Optional[str] = field(default='embed_tokens,lm_head')
    debug_mode: Optional[bool] = field(default=False)
    peft_path: Optional[str] = field(default=None)
    task_num: Optional[int] = field(default=16)
    task_embedding_dim: Optional[int] = field(default=64)
    expert_num: Optional[int] = field(default=4)
    knowledge_r: Optional[int] = field(default=8)
    kmoe_path: Optional[str] = field(default="m_saved/lora-0725/checkpoint-8000")
    freeze: Optional[bool] = field(default=False)
    department: Optional[bool] = field(default=False)
    depart_num: Optional[int] = field(default=16)
    entity_num: Optional[int] = field(default=26)
    bias_weight: Optional[float] = field(default=1)
    achievement_gamma: Optional[float] = field(default=2.0, metadata={"help": "Achievement loss focusing parameter"})
    achievement_margin: Optional[float] = field(default=1.2, metadata={"help": "Achievement loss margin (partial > 1)"})
    use_achievement_loss: Optional[bool] = field(default=False, metadata={"help": "Enable achievement-based multi-task loss weighting"})
    achievement_targets: Optional[str] = field(default=None, metadata={"help": "Per-task SOTA targets (Macro-F1), e.g. 'IHD:0.8,Sarca:0.83'. Uses task_dataset.json for name->id mapping."})
    achievement_ema_alpha: Optional[float] = field(default=0.15, metadata={"help": "EMA smoothing for achievement score updates. 0.15=stable, 0.3=responsive, 1.0=no smoothing"})
    achievement_warmup_ratio: Optional[float] = field(default=0.0, metadata={"help": "Fraction of max_steps to use uniform weights before achievement weighting kicks in (0.0=no warmup)"})
    achievement_weight_floor: Optional[float] = field(default=0.10, metadata={"help": "Minimum normalized weight per task (0.10=at least 10%%). Set to 0 to disable floor."})
    lb_loss_coeff: Optional[float] = field(default=0.0, metadata={"help": "Load-balance loss coefficient (0=disabled, 0.001=light, 0.01=original). Controls expert routing uniformity."})
    routing_div_coeff: Optional[float] = field(default=0.0, metadata={"help": "Routing divergence loss coefficient (0=disabled, 0.01~0.1=typical). Pushes new task's routing away from frozen tasks' routing."})
    routing_div_frozen_tasks: Optional[str] = field(default=None, metadata={"help": "Comma-separated task IDs whose routing the new task should diverge from (e.g. '1,2'). Used with routing_div_coeff."})
    eval_metric_mode: Optional[str] = field(default="loss", metadata={"help": "In-training eval metric: loss (eval_loss only), binary (pos-class P/R/F1), macro (Macro P/R/F1)"})
    freeze_experts: Optional[bool] = field(default=False, metadata={"help": "Freeze all LoRA expert weights and gate; only train new task embedding(s). For Phase 2 curriculum learning."})
    freeze_expert_ids: Optional[str] = field(default=None, metadata={"help": "Comma-separated expert IDs to freeze (e.g. '0,1,2,4,5'). Unmentioned experts + gate + embedding remain trainable. For Phase 2B partial freeze."})
    no_freeze_gate: Optional[bool] = field(default=False, metadata={"help": "When used with freeze_expert_ids, keep the gate trainable instead of freezing it."})
    replay_ratio: Optional[float] = field(default=1.0, metadata={"help": "Subsample ratio for replay tasks each epoch (0.1=keep 10%%). 1.0=use all data. Requires replay_task_ids."})
    replay_task_ids: Optional[str] = field(default=None, metadata={"help": "Comma-separated task IDs to subsample for replay (e.g. '1,2'). Non-replay tasks always keep 100%%."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    batched_training: bool = field(
        default=False, metadata={"help": "Use the batched training."}
    )

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    prompt_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    response_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    history_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the history of chat."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=True, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )

    

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None and self.test_file is None:
            raise ValueError("Need either a dataset name or a training/validation/test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

