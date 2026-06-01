# -*- encoding: utf-8 -*-
# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import PeftConfig


class Gate(nn.Module):
    """Gate with per-task load-balancing loss and temperature control.

    Per-task LB loss computes balance independently for each task:
        lb_loss = mean_over_tasks( expert_num * sum_i(mean_P_i^2) )
    This allows different tasks to have different routing patterns
    while still preventing any single task from collapsing to one expert.

    Temperature < 1 sharpens routing (more specialisation),
    temperature > 1 flattens routing (more sharing).
    """
    def __init__(self, peft_config: PeftConfig, adapter_name="default"):

        super().__init__()

        self.expert_num = peft_config.expert_num
        self.task_num = peft_config.task_num
        self.te_dim = peft_config.task_embedding_dim

        self.GateL = nn.Linear(self.te_dim, self.expert_num, bias=False)
        self.temperature = 1.0   # can be tuned; < 1 = sharper routing

        # Stored after each forward for load-balance loss computation
        self._last_gate_probs = None
        self._last_task_ids = None

    def forward(self, task_em, task_ids=None):
        """
        Args:
            task_em:  (batch, te_dim) task embeddings
            task_ids: (batch,) int tensor of original task IDs (optional,
                      used for per-task LB loss)
        """
        if self.GateL.weight.requires_grad:
            # Normal path: gate is trainable, gradients flow to both gate & embedding
            logits = self.GateL(task_em)
        else:
            # Frozen gate path: use detached weight so gradients flow to
            # task_em (embedding) but NOT to GateL.weight.
            # This prevents DeepSpeed ZeRO-2 from modifying frozen gate weights
            # through the computation graph:
            #   loss → Σ(ew_i * Expert_i) → ew → softmax → matmul → GateL.weight
            logits = F.linear(task_em, self.GateL.weight.detach())
        if self.temperature != 1.0:
            logits = logits / self.temperature
        y = torch.softmax(logits, dim=1)
        # Detach _last_gate_probs when gate is frozen to also cut lb_loss path
        if self.GateL.weight.requires_grad:
            self._last_gate_probs = y           # (batch, expert_num) — keep graph
        else:
            self._last_gate_probs = y.detach()  # (batch, expert_num) — cut graph
        self._last_task_ids = task_ids      # (batch,) or None
        return y

    def load_balance_loss(self):
        """Per-task load-balance loss.

        If task_ids are available, compute lb_loss independently for each
        task and average.  This prevents the dominant task (IHD, ~80% of
        batch) from suppressing the minority task's (Sarca) routing pattern.

        If task_ids are not available, falls back to global lb_loss.
        """
        if self._last_gate_probs is None:
            return 0.0

        probs = self._last_gate_probs       # (batch, expert_num)
        tids = self._last_task_ids          # (batch,) or None

        if tids is not None and tids.numel() > 0:
            # Per-task LB loss
            lb_total = 0.0
            n_tasks = 0
            for tid in tids.unique():
                mask = (tids == tid)
                task_probs = probs[mask]     # (n_task_samples, expert_num)
                mean_P = task_probs.mean(dim=0)
                lb_total += self.expert_num * (mean_P ** 2).sum()
                n_tasks += 1
            return lb_total / max(n_tasks, 1)
        else:
            # Fallback: global LB loss
            mean_P = probs.mean(dim=0)
            return self.expert_num * (mean_P ** 2).sum()


class GateN(nn.Module):
    """Gate New Function"""
    def __init__(self, expert_num, task_embedding_dim):

        super().__init__()

        self.expert_num = expert_num
        self.te_dim = task_embedding_dim

        self.GateL = nn.Linear(self.te_dim, self.expert_num, bias=False)
        self.act = nn.Softmax(dim=1)    # dim-0 is batch size

    def forward(self, task_em):

        y = self.GateL(task_em)
        y = self.act(y)

        return y
