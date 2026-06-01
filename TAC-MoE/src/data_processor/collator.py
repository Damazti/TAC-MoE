# -*- encoding: utf-8 -*-
# here put the import lib
from dataclasses import dataclass
from typing import Dict, Sequence
import torch
import transformers

IGNORE_INDEX = -100


@dataclass
class LongestSequenceCollator(object):
    """Collate examples for supervised fine-tuning.

    Supports both left-padding (for decoder-only generation, e.g. Qwen2)
    and right-padding (for encoder-decoder or ChatGLM3).
    Padding direction is determined by tokenizer.padding_side.
    """

    tokenizer: transformers.PreTrainedTokenizer
    task_flag: bool
    depart_flag: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        # Ensure all are tensors
        input_ids = [torch.tensor(x, dtype=torch.long) if not isinstance(x, torch.Tensor) else x.long() for x in input_ids]
        labels = [torch.tensor(x, dtype=torch.long) if not isinstance(x, torch.Tensor) else x.long() for x in labels]

        pad_left = getattr(self.tokenizer, "padding_side", "right") == "left"

        if pad_left:
            # Left-padding: reverse → pad_sequence (right-pads) → reverse back
            input_ids_rev = [x.flip(0) for x in input_ids]
            labels_rev = [x.flip(0) for x in labels]
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_rev, batch_first=True, padding_value=self.tokenizer.pad_token_id
            ).flip(1)
            labels = torch.nn.utils.rnn.pad_sequence(
                labels_rev, batch_first=True, padding_value=-100
            ).flip(1)
        else:
            # Right-padding (default)
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        # Generate attention_mask (1 for real tokens, 0 for padding)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        if self.task_flag:
            task_id = [instance["task_id"] for instance in instances]
            task_id = torch.LongTensor(task_id)

            if self.depart_flag:    # if add the department and entity
                depart = [instance["depart"] for instance in instances]
                depart = torch.LongTensor(depart)

                entity = [instance["entity"] for instance in instances]
                entity = torch.stack(entity)
                entity = torch.LongTensor(entity)

                return dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    task_id=task_id,
                    depart=depart,
                    entity=entity,
                )

            return dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                task_id=task_id,
            )

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

