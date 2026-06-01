# -*- encoding: utf-8 -*-
"""
Achievement-based Multi-task Loss Weighting (with weight floor).

Formula:
    w_t^m = (1 - s_t^m / (margin * P_m)) ^ gamma
    eff_m = max(w_m, floor)   (then re-normalized)

Key design choices:
    - NO IDF: per-task mean() in compute_weighted_loss already normalises
      for sample-count differences within each batch. IDF was empirically
      shown to hurt both tasks (END1 run < baseline).
    - weight_floor=0.20: prevents any task from getting <20% gradient,
      protecting shared representation quality
    - EMA alpha=0.15: smooths eval noise (Sarca test set has only 784
      samples → eval F1 can swing ±4 points between steps)
"""
import math
import torch
import torch.nn as nn


class AchievementWeightedLoss:

    def __init__(self, task_num=3, gamma=2.0, margin=1.2, sota_targets=None,
                 ema_alpha=0.15, task_sample_counts=None,
                 weight_floor=0.20):
        """
        Args:
            task_num: number of tasks (task_id from 1..task_num)
            gamma: focusing parameter
            margin: target margin (partial in paper)
            sota_targets: dict {task_id: P_m} or None (defaults to 1.0)
            ema_alpha: EMA smoothing coefficient for score updates.
                       Lower = more stable weights, less reactive to noise.
            task_sample_counts: (unused, kept for API compat) dict of sample counts.
            weight_floor: minimum normalized weight per task (e.g. 0.20 = at least 20%).
        """
        self.task_num = task_num
        self.gamma = gamma
        self.margin = margin
        self.ema_alpha = ema_alpha
        self.weight_floor = weight_floor

        # P_m: target performance per task
        if sota_targets is None:
            self.sota_targets = {i: 1.0 for i in range(1, task_num + 1)}
        else:
            self.sota_targets = sota_targets

        # s_t^m: current metric per task (start at 0 → equal weights)
        self.task_scores = {i: 0.0 for i in range(1, task_num + 1)}

        # computed weights (initialized uniform)
        self.weights = {i: 1.0 / task_num for i in range(1, task_num + 1)}

        # Track whether this is the first update (use raw score, no EMA)
        self._first_update = {i: True for i in range(1, task_num + 1)}

    def update_score(self, task_id, score):
        """Update the metric score for a task (with EMA smoothing) and recompute weights."""
        if self._first_update.get(task_id, True):
            self.task_scores[task_id] = score
            self._first_update[task_id] = False
        else:
            old = self.task_scores[task_id]
            self.task_scores[task_id] = self.ema_alpha * score + (1 - self.ema_alpha) * old
        self._recompute_weights()

    def update_scores(self, scores_dict):
        """Batch update: {task_id: score} with EMA smoothing."""
        for tid, score in scores_dict.items():
            if self._first_update.get(tid, True):
                self.task_scores[tid] = score
                self._first_update[tid] = False
            else:
                old = self.task_scores[tid]
                self.task_scores[tid] = self.ema_alpha * score + (1 - self.ema_alpha) * old
        self._recompute_weights()

    def _recompute_weights(self):
        """Recompute achievement weights with direct normalization."""
        raw = {}
        for tid in range(1, self.task_num + 1):
            s = self.task_scores[tid]
            p = self.sota_targets[tid]
            ratio = min(s / (self.margin * p), 1.0)
            raw[tid] = (1.0 - ratio) ** self.gamma

        total = sum(raw.values())
        if total > 0:
            self.weights = {k: v / total for k, v in raw.items()}
        else:
            self.weights = {k: 1.0 / self.task_num for k in raw}

    def compute_weighted_loss(self, per_sample_loss, task_ids):
        """
        Apply achievement weights with weight floor to per-sample losses.

        Uses per-task mean to aggregate losses within each task, then
        applies achievement weights.  Weight floor ensures no task drops
        below a minimum share.

        Args:
            per_sample_loss: [batch_size] tensor, mean CE loss per sample
            task_ids:        [batch_size] tensor of int task IDs
        Returns:
            weighted scalar loss
        """
        weighted_loss = torch.tensor(0.0, device=per_sample_loss.device,
                                     dtype=per_sample_loss.dtype)

        # Get achievement weights for tasks present in batch
        effective_w = {}
        for tid in task_ids.unique():
            tid_int = tid.item()
            effective_w[tid_int] = self.weights.get(tid_int, 1.0 / self.task_num)

        # Apply weight floor: no task below weight_floor
        n_tasks_in_batch = len(effective_w)
        if n_tasks_in_batch > 1 and self.weight_floor > 0:
            max_floor = 1.0 / n_tasks_in_batch
            floor = min(self.weight_floor, max_floor)

            # Normalize first
            w_total = sum(effective_w.values())
            if w_total > 0:
                effective_w = {k: v / w_total for k, v in effective_w.items()}

            # Clamp and redistribute
            floored = {}
            surplus = 0.0
            n_floored = 0
            for k, v in effective_w.items():
                if v < floor:
                    floored[k] = floor
                    surplus += (floor - v)
                    n_floored += 1
                else:
                    floored[k] = v

            if surplus > 0 and n_floored < n_tasks_in_batch:
                non_floored_total = sum(v for k, v in floored.items()
                                        if effective_w[k] >= floor)
                if non_floored_total > 0:
                    for k in floored:
                        if effective_w[k] >= floor:
                            floored[k] -= surplus * (floored[k] / non_floored_total)

            effective_w = floored
        else:
            # Single task in batch or no floor → just normalize
            w_total = sum(effective_w.values())
            if w_total > 0:
                effective_w = {k: v / w_total for k, v in effective_w.items()}

        for tid in task_ids.unique():
            mask = (task_ids == tid)
            task_loss = per_sample_loss[mask].mean()
            w = effective_w.get(tid.item(), 1.0 / self.task_num)
            weighted_loss = weighted_loss + w * task_loss

        return weighted_loss

    def get_weights_str(self):
        """Return a readable string of current weights and smoothed scores."""
        parts = []
        for k, v in sorted(self.weights.items()):
            parts.append(f"task{k}={v:.4f}(s={self.task_scores[k]:.4f})")
        return ", ".join(parts)
