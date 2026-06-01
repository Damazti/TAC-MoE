# -*- encoding: utf-8 -*-
"""
Task-Balanced Batch Sampler for multi-task training.

Ensures each micro-batch contains roughly equal numbers of samples from
each task via OVERSAMPLING the minority task(s).

Epoch length is determined by the LARGEST task, but capped so that no
task is oversampled beyond `max_oversample` times per epoch.

Example (batch=16, 2 tasks, IHD=15635, Sarca=3817):
    max_oversample=0 (unlimited):  Sarca ~4.1x, IHD 1x  (full balance)
    max_oversample=2.0:            Sarca  2.0x, IHD ~0.49x (moderate)
        → epoch = 2.0 * 3817 / 8 = 954 batches
        → each batch still 8 IHD + 8 Sarca (balanced)
        → IHD uses 954*8 = 7632 of 15635 samples (~49%)

IMPORTANT: This sampler does NOT handle distributed splitting internally.
DeepSpeed/Accelerator wraps this with DistributedSampler automatically,
which takes care of rank-based splitting.
"""
import math
import torch
import numpy as np
from torch.utils.data import Sampler
from typing import List, Dict, Optional, Iterator

import logging
logger = logging.getLogger(__name__)


class TaskBalancedSampler(Sampler):
    """Yields indices so that consecutive `batch_size` indices are
    task-balanced (equal samples per task) via oversampling.

    Args:
        task_ids:        list/array of task ID for each sample in the dataset
        batch_size:      micro-batch size per GPU
        max_oversample:  cap on oversampling ratio for any task (0 = unlimited).
                         E.g., 2.0 means no task sees its samples more than 2x
                         per epoch.  This shortens the epoch if the natural
                         ratio exceeds this cap.
        seed:            random seed for shuffling (incremented each epoch)
        drop_last:       drop incomplete final batch
    """

    def __init__(
        self,
        task_ids: List[int],
        batch_size: int,
        max_oversample: float = 0.0,
        seed: int = 42,
        drop_last: bool = False,
    ):
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last
        self.batch_size = batch_size

        # Group indices by task
        self.task_indices: Dict[int, List[int]] = {}
        for idx, tid in enumerate(task_ids):
            tid = int(tid)
            if tid not in self.task_indices:
                self.task_indices[tid] = []
            self.task_indices[tid].append(idx)

        self.task_list = sorted(self.task_indices.keys())
        self.n_tasks = len(self.task_list)
        self.per_task_batch = max(1, batch_size // self.n_tasks)

        # Determine epoch length
        min_task_len = min(len(v) for v in self.task_indices.values())
        max_task_len = max(len(v) for v in self.task_indices.values())

        if max_oversample > 0:
            # Cap: smallest task repeated at most max_oversample times
            capped_batches = math.ceil(min_task_len * max_oversample / self.per_task_batch)
            natural_batches = math.ceil(max_task_len / self.per_task_batch)
            self.num_batches = min(capped_batches, natural_batches)
        else:
            # Unlimited: epoch = largest task
            self.num_batches = math.ceil(max_task_len / self.per_task_batch)

        self.num_samples = self.num_batches * batch_size

        # Log sampling stats
        for tid in self.task_list:
            n = len(self.task_indices[tid])
            needed = self.num_batches * self.per_task_batch
            ratio = needed / n
            logger.info(f"TaskBalancedSampler: task {tid}: {n} samples, "
                        f"~{needed} needed/epoch ({ratio:.2f}x), "
                        f"total_batches={self.num_batches}, "
                        f"max_oversample={max_oversample}")

    def __iter__(self) -> Iterator[int]:
        rng = np.random.RandomState(self.seed + self.epoch)

        # Shuffle within each task
        shuffled: Dict[int, List[int]] = {}
        for tid in self.task_list:
            indices = self.task_indices[tid].copy()
            rng.shuffle(indices)
            shuffled[tid] = indices

        # Build balanced batches (oversample: cycle smaller tasks)
        all_indices = []
        task_ptrs = {tid: 0 for tid in self.task_list}

        for _ in range(self.num_batches):
            batch = []
            for tid in self.task_list:
                pool = shuffled[tid]
                for _ in range(self.per_task_batch):
                    ptr = task_ptrs[tid]
                    if ptr >= len(pool):
                        # Cycle: reshuffle and restart
                        rng.shuffle(pool)
                        ptr = 0
                    batch.append(pool[ptr])
                    task_ptrs[tid] = ptr + 1

            # Fill remaining slots if batch_size not evenly divisible
            while len(batch) < self.batch_size:
                tid = self.task_list[rng.randint(self.n_tasks)]
                pool = shuffled[tid]
                ptr = task_ptrs[tid]
                if ptr >= len(pool):
                    rng.shuffle(pool)
                    ptr = 0
                batch.append(pool[ptr])
                task_ptrs[tid] = ptr + 1

            rng.shuffle(batch)  # shuffle within batch
            all_indices.extend(batch)

        return iter(all_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling across distributed ranks."""
        self.epoch = epoch
