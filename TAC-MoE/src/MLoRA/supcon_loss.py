# -*- encoding: utf-8 -*-
"""
Supervised Contrastive Loss for multi-task sarcasm detection.

Applies SupCon loss on a specific task (e.g., Sarca) within a
multi-task batch, using the last hidden state as the representation.

Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import logging
logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """2-layer MLP projection head for contrastive learning.

    Maps high-dimensional hidden states to a compact normalized space
    where contrastive loss is computed.

    Architecture: Linear → ReLU → Linear → L2-normalize
    """

    def __init__(self, hidden_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, hidden_dim]
        Returns:
            z: [N, proj_dim], L2-normalized
        """
        z = self.net(x)
        return F.normalize(z, dim=-1)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    For each anchor, pulls together samples of the same class and
    pushes apart samples of different classes.

    Args:
        temperature: scaling factor for cosine similarity (default: 0.07)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: [N, proj_dim], L2-normalized projected features
            labels:   [N], integer class labels (0 or 1)

        Returns:
            Scalar contrastive loss. Returns 0 if fewer than 2 samples
            or only 1 class present.
        """
        device = features.device
        n = features.shape[0]

        if n < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Need at least 2 classes for contrastive learning
        unique_labels = labels.unique()
        if len(unique_labels) < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Cosine similarity matrix: [N, N]
        sim = torch.matmul(features, features.T) / self.temperature

        # Mask: same class = 1, different class = 0
        labels_col = labels.unsqueeze(0)  # [1, N]
        labels_row = labels.unsqueeze(1)  # [N, 1]
        positive_mask = (labels_row == labels_col).float()  # [N, N]

        # Remove self-similarity from positives
        self_mask = torch.eye(n, device=device)
        positive_mask = positive_mask - self_mask  # exclude self

        # For numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # exp(sim) for all pairs except self
        exp_sim = torch.exp(sim) * (1 - self_mask)  # [N, N], zero on diagonal

        # Denominator: sum of exp(sim) over all non-self pairs
        denom = exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [N, 1]

        # Log-prob: log(exp(sim_ij) / sum_k exp(sim_ik)) for positive pairs
        log_prob = sim - torch.log(denom)  # [N, N]

        # Mean log-prob over positive pairs for each anchor
        num_positives = positive_mask.sum(dim=1).clamp(min=1)  # [N]
        mean_log_prob = (positive_mask * log_prob).sum(dim=1) / num_positives

        # Loss = -mean over all anchors
        loss = -mean_log_prob.mean()

        return loss
