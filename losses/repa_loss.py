"""
losses/repa_loss.py

REPA representation-alignment loss.

    L_REPA = -E_{z, eps, t} [ (1/N) * sum_n cos_sim(y_n, h_out_n) ]

Returns both the scalar loss (for backprop) and the raw mean cosine
similarity (for logging / analysis plots).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.projection_head import ProjectionHead


class REPALoss(nn.Module):
    """
    Patch-wise cosine similarity alignment loss.

    Args:
        head: ProjectionHead that maps SiT hidden states into teacher space.
    """

    def __init__(self, head: ProjectionHead) -> None:
        super().__init__()
        self.head = head

    def forward(
        self,
        student_hidden:   torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            student_hidden:   (B, N, D_student) — SiT hidden states.
            teacher_features: (B, N, D_teacher) — frozen teacher patch embeddings.

        Returns:
            loss:         Scalar — negative mean cosine similarity. Minimise this.
            cos_sim_mean: Scalar — raw mean cosine similarity [-1, 1]. Log this.
        """
        h_proj = self.head(student_hidden)             # (B, N, D_teacher)

        h_norm = F.normalize(h_proj,          dim=-1)  # (B, N, D_teacher)
        y_norm = F.normalize(teacher_features, dim=-1)  # (B, N, D_teacher)

        cos_sim      = (h_norm * y_norm).sum(dim=-1)   # (B, N)
        cos_sim_mean = cos_sim.mean()

        return -cos_sim_mean, cos_sim_mean.detach()

    def __repr__(self) -> str:
        return f"REPALoss(head={self.head})"


def repa_loss(
    student_hidden:   torch.Tensor,
    teacher_features: torch.Tensor,
    head:             ProjectionHead,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Functional interface. Returns (loss, cos_sim_mean)."""
    h_proj  = head(student_hidden)
    h_norm  = F.normalize(h_proj,          dim=-1)
    y_norm  = F.normalize(teacher_features, dim=-1)
    cos_sim = (h_norm * y_norm).sum(dim=-1).mean()
    return -cos_sim, cos_sim.detach()