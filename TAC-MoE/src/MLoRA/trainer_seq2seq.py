# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from typing import Any, Dict, List, Optional, Tuple, Union

import math
import torch
from torch import nn
from torch.utils.data import Dataset

from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from transformers.trainer_utils import PredictionOutput
from transformers.utils import logging


from .trainer import Trainer

logger = logging.get_logger(__name__)


class Seq2SeqTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """

        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            if self.args.generation_max_length is not None:
                gen_kwargs["max_new_tokens"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        result = super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        # ---- Aggressively reclaim GPU memory after eval ----
        # generate() creates KV-cache, logits, intermediate tensors that
        # may linger in Python references / CUDA caches.  Force-clean BEFORE
        # any training tensor is allocated.
        import gc, os, torch
        gc.collect()
        gc.collect()            # double-collect for cyclic refs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()   # release IPC handles if any

        # Update achievement loss weights from per-task F1.
        # During achievement warmup, training intentionally uses plain mean loss;
        # keep EMA/weights uninitialized until the first post-warmup eval.
        achievement_update_active = hasattr(self, "achievement_loss") and self.achievement_loss is not None
        if achievement_update_active:
            warmup_ratio = float(getattr(self, "achievement_warmup_ratio", 0.0) or 0.0)
            max_steps = int(getattr(self.args, "max_steps", 0) or 0)
            warmup_steps = int(max_steps * warmup_ratio) if max_steps > 0 and warmup_ratio > 0 else 0
            _debug_force_warmup_steps = os.environ.get("MOELORA_DEBUG_FORCE_WARMUP_STEPS")
            if _debug_force_warmup_steps:
                warmup_steps = int(_debug_force_warmup_steps)
            if warmup_steps > 0 and self.state.global_step < warmup_steps:
                achievement_update_active = False

        if achievement_update_active:
            import json, os
            task_map_path = "data/task_dataset.json"
            if os.path.exists(task_map_path):
                task_map = json.load(open(task_map_path))["str2id"]
                scores = {}
                for task_name, task_id in task_map.items():
                    f1_key = f"{metric_key_prefix}_{task_name}_f1"
                    if f1_key in result:
                        scores[task_id] = result[f1_key]

                # Always update EMA from the first eval step.
                # With undersampling + ema_alpha=0.3, early unstable F1 values
                # are quickly corrected.  The first update uses raw score (no EMA).
                if scores:
                    self.achievement_loss.update_scores(scores)
                    self.achievement_scores_ready = True
                    logger.warning(f"Achievement weights updated: {self.achievement_loss.get_weights_str()}")
                else:
                    logger.warning(
                        f"Achievement loss: no per-task F1 in eval metrics "
                        f"(keys: {[k for k in result if 'f1' in k]}). Weights unchanged."
                    )

                # Always log current weights to result dict + TensorBoard
                weight_metrics = {}
                for task_name, task_id in task_map.items():
                    w = self.achievement_loss.weights.get(task_id)
                    if w is not None:
                        weight_metrics[f"{metric_key_prefix}_achievement_weight_{task_name}"] = float(w)

                weight_metrics[f"{metric_key_prefix}_achievement_weight_entropy"] = float(
                    -sum(w * math.log(w) for w in self.achievement_loss.weights.values() if w > 0)
                )

                result.update(weight_metrics)
                # Silent: don't self.log() separately — metrics are in result dict
                # and will be recorded by the eval callback automatically

        # Log gate diagnostics at eval time (task embedding + routing snapshot)
        _unwrapped = self.accelerator.unwrap_model(self.model)
        if hasattr(_unwrapped, 'lora_gate') and hasattr(_unwrapped, 'lora_task_embedding'):
            import torch, json as _jg, os as _og
            _adapter = getattr(_unwrapped, 'active_adapter', 'default')
            _gate = _unwrapped.lora_gate[_adapter] if _adapter in _unwrapped.lora_gate else None
            _te = _unwrapped.lora_task_embedding[_adapter] if _adapter in _unwrapped.lora_task_embedding else None
            if _gate is not None and _te is not None:
                _gw = _gate.GateL.weight.detach().float()

                # Load task names dynamically from task_dataset.json
                _task_names = {}
                _tmap_path = "data/task_dataset.json"
                if _og.path.exists(_tmap_path):
                    _id2str = _jg.load(open(_tmap_path))["id2str"]
                    _task_names = {int(k): v for k, v in _id2str.items()}

                _num_tasks = _te.weight.shape[0]  # includes index 0 (unused)
                _gate_diag = {
                    f"{metric_key_prefix}_gate_weight_norm": _gw.norm().item(),
                }

                # Compute per-task embeddings, logits, probs
                _embeddings = {}
                _probs_dict = {}
                for tid in range(1, _num_tasks):
                    _emb = _te.weight[tid].detach().float()
                    _embeddings[tid] = _emb
                    _logits = _gw @ _emb
                    _probs = torch.softmax(_logits, dim=0)
                    _probs_dict[tid] = _probs
                    _tname = _task_names.get(tid, f"task{tid}")
                    _gate_diag[f"{metric_key_prefix}_gate_logit_spread_{_tname}"] = (
                        _logits.max() - _logits.min()).item()
                    # Per-expert routing
                    for i in range(_probs.shape[0]):
                        _gate_diag[f"{metric_key_prefix}_routing_{_tname}_e{i}"] = _probs[i].item()

                # Pairwise cosine similarities
                _tids = sorted(_embeddings.keys())
                for i_idx in range(len(_tids)):
                    for j_idx in range(i_idx + 1, len(_tids)):
                        ti, tj = _tids[i_idx], _tids[j_idx]
                        _cos = torch.nn.functional.cosine_similarity(
                            _embeddings[ti], _embeddings[tj], dim=0).item()
                        _ni = _task_names.get(ti, f"task{ti}")
                        _nj = _task_names.get(tj, f"task{tj}")
                        _gate_diag[f"{metric_key_prefix}_gate_cos_{_ni}_{_nj}"] = _cos

                # Average pairwise JS divergence across all task pairs
                if len(_probs_dict) >= 2:
                    _js_sum = 0.0
                    _js_count = 0
                    for i_idx in range(len(_tids)):
                        for j_idx in range(i_idx + 1, len(_tids)):
                            ti, tj = _tids[i_idx], _tids[j_idx]
                            _p1, _p2 = _probs_dict[ti], _probs_dict[tj]
                            _m = 0.5 * (_p1 + _p2)
                            _js = 0.5 * ((_p1 * torch.log((_p1 + 1e-8) / (_m + 1e-8))).sum()
                                       + (_p2 * torch.log((_p2 + 1e-8) / (_m + 1e-8))).sum()).item()
                            _ni = _task_names.get(ti, f"task{ti}")
                            _nj = _task_names.get(tj, f"task{tj}")
                            _gate_diag[f"{metric_key_prefix}_routing_JS_{_ni}_{_nj}"] = _js
                            _js_sum += _js
                            _js_count += 1
                    _gate_diag[f"{metric_key_prefix}_routing_JS_avg"] = _js_sum / _js_count

                result.update(_gate_diag)
                # Silent: don't self.log() separately — metrics are in result dict

        # Clear CUDA cache after eval to prevent memory fragmentation
        # that causes OOM when training resumes (especially with large vocab models).
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        <Tip>

        If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
        padding in a token classification task) the predictions will be padded (on the right) to allow for
        concatenation into one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """

        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            if self.args.generation_max_length is not None:
                gen_kwargs["max_new_tokens"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs


        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "position_ids" in inputs:
            gen_kwargs["position_ids"] = inputs.get("position_ids", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        gen_kwargs["input_ids"] = generation_inputs
        if "task_id" in inputs.keys():
            gen_kwargs["task_id"] = inputs["task_id"]
        if "depart" in inputs.keys():
            gen_kwargs["depart"] = inputs["depart"]
        if "entity" in inputs.keys():
            gen_kwargs["entity"] = inputs["entity"]
        generated_tokens = self.model.generate(**gen_kwargs)
        generated_tokens = generated_tokens[:, generation_inputs.size()[-1]:]

        # in case the batch is shorter than max length, the output should be padded
        if gen_kwargs.get("max_length") is not None and generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        elif gen_kwargs.get("max_new_tokens") is not None and generated_tokens.shape[-1] < (
            gen_kwargs["max_new_tokens"] + 1
        ):
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_new_tokens"] + 1)

        loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if gen_kwargs.get("max_length") is not None and labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
            elif gen_kwargs.get("max_new_tokens") is not None and labels.shape[-1] < (
                gen_kwargs["max_new_tokens"] + 1
            ):
                labels = self._pad_tensors_to_max_len(labels, (gen_kwargs["max_new_tokens"] + 1))
        else:
            labels = None

        return (loss, generated_tokens, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
