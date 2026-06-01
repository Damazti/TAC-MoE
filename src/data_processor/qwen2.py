# -*- encoding: utf-8 -*-
"""
Qwen2 data processor for MOELoRA.
Uses Qwen2 chat template (apply_chat_template) to build input_ids and labels.
"""
import json
import torch
import copy


class qwen2_train(object):
    """Training preprocessor: full conversation tokenized, instruction part masked in labels."""

    def __init__(self, data_args, model_args, prompt_column,
                 response_column, history_column, prefix, tokenizer,
                 task=False, department=False):
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.history_column = history_column
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.task = task
        self.department = department

    def __call__(self, examples):
        model_inputs = {"input_ids": [], "labels": []}

        if self.task:
            model_inputs["task_id"] = []
            task_dict = json.load(open("data/task_dataset.json", "r"))["str2id"]

        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i] and examples[self.response_column][i]:
                query = examples[self.prompt_column][i]
                answer = examples[self.response_column][i]

                # Build Qwen2 chat messages
                messages = []
                if self.history_column and self.history_column in examples:
                    history = examples[self.history_column][i]
                    if history:
                        for old_query, response in history:
                            messages.append({"role": "user", "content": old_query})
                            messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": query})
                messages.append({"role": "assistant", "content": answer})

                # Tokenize the full conversation
                conversation = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                input_ids = torch.tensor(
                    self.tokenizer(conversation, add_special_tokens=False).input_ids,
                    dtype=torch.long
                )

                # Create labels (copy of input_ids, mask instruction part)
                labels = copy.deepcopy(input_ids)

                # Calculate instruction length (everything before the assistant answer)
                instruction_msgs = [{"role": "user", "content": query}]
                instruction_text = self.tokenizer.apply_chat_template(
                    instruction_msgs, tokenize=False, add_generation_prompt=True
                )
                instruction_len = len(
                    self.tokenizer(instruction_text, add_special_tokens=False).input_ids
                )

                # Mask instruction tokens in labels
                labels[:instruction_len] = -100

                # Truncate if needed
                max_len = self.data_args.max_source_length
                if len(input_ids) > max_len:
                    input_ids = input_ids[:max_len]
                    labels = labels[:max_len]

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

                if self.task:
                    task_id = task_dict[examples['task_dataset'][i]]
                    model_inputs["task_id"].append(task_id)

        return model_inputs


class qwen2_eval(object):
    """Eval/predict preprocessor: only instruction part as input_ids, answer as labels.

    For predict_with_generate: input_ids = instruction (system+user+generation_prompt),
    model generates the answer. Labels = tokenized target for metric computation.
    For eval_loss only: same as train (full conversation with masked instruction).
    """

    def __init__(self, data_args, model_args, prompt_column,
                 response_column, history_column, prefix, tokenizer,
                 task=False, department=False):
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.history_column = history_column
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.task = task
        self.department = department

    def __call__(self, examples):
        model_inputs = {"input_ids": [], "labels": []}

        if self.task:
            model_inputs["task_id"] = []
            task_dict = json.load(open("data/task_dataset.json", "r"))["str2id"]

        for i in range(len(examples[self.prompt_column])):
            if not examples[self.response_column][i]:
                continue

            query = examples[self.prompt_column][i]
            answer = examples[self.response_column][i]

            if not query:
                continue

            # Build instruction messages (user only, with generation prompt)
            messages = []
            if self.history_column and self.history_column in examples:
                history = examples[self.history_column][i]
                if history:
                    for old_query, response in history:
                        messages.append({"role": "user", "content": old_query})
                        messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": query})

            # Instruction part only (for generate input)
            instruction_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize instruction WITHOUT padding (LongestSequenceCollator handles dynamic padding)
            input_ids = self.tokenizer(
                instruction_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.data_args.max_source_length,
            ).input_ids

            # Labels: tokenize just the answer text
            label_ids = self.tokenizer(
                answer,
                add_special_tokens=False,
                max_length=self.data_args.max_target_length,
                truncation=True,
            ).input_ids

            if self.data_args.ignore_pad_token_for_loss:
                label_ids = [(l if l != self.tokenizer.pad_token_id else -100) for l in label_ids]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(label_ids)

            if self.task:
                task_id = task_dict[examples['task_dataset'][i]]
                model_inputs["task_id"].append(task_id)

        return model_inputs
