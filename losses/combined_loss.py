"""
losses/combined_loss.py

Combined diffusion + REPA loss.

    L_total = L_diff + lambda * L_REPA

Returns the total loss for backprop plus a dict of all component losses
for logging to the .jsonl log file.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from losses.diffusion_loss import DiffusionLoss
from losses.repa_loss import REPALoss
from models.projection_head import ProjectionHead


class CombinedLoss(nn.Module):
    """
    Combines the diffusion loss with the optional REPA alignment loss.

    When repa_head is None (teacher == "none"), behaves as pure diffusion.

    Args:
        repa_head: ProjectionHead, or None for baseline runs.
        lam:       Weight of the REPA term (default 0.5).

    Usage::

        criterion = CombinedLoss(repa_head=head, lam=0.5)
        total, log_dict = criterion(
            v_pred=v_pred, z=z, eps=eps,
            student_hidden=hidden_states[align_layer],
            teacher_features=teacher_features,
        )
        total.backward()
    """

    def __init__(
        self,
        repa_head: Optional[ProjectionHead] = None,
        lam: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.lam = lam
        self.diff_loss = DiffusionLoss(reduction=reduction)
        self.repa_loss_fn: Optional[REPALoss] = (
            REPALoss(head=repa_head) if repa_head is not None else None
        )

    def forward(
        self,
        v_pred:           torch.Tensor,
        z:                torch.Tensor,
        eps:              torch.Tensor,
        student_hidden:   Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            v_pred:           (B, C, H, W) — SiT velocity prediction.
            z:                (B, C, H, W) — clean latent.
            eps:              (B, C, H, W) — sampled noise.
            student_hidden:   (B, N, D_student) — SiT hidden states, or None.
            teacher_features: (B, N, D_teacher) — teacher patch features, or None.

        Returns:
            total_loss: Scalar — backpropagate this.
            log_dict:   Dict with float values for .jsonl logging:
                        {loss_diff, loss_repa, cos_sim, loss_total, lambda}
        """
        loss_diff = self.diff_loss(v_pred, z, eps)

        loss_repa   = torch.tensor(0.0, device=v_pred.device)
        cos_sim_val = torch.tensor(0.0, device=v_pred.device)

        if (
            self.repa_loss_fn is not None
            and student_hidden is not None
            and teacher_features is not None
        ):
            loss_repa, cos_sim_val = self.repa_loss_fn(student_hidden, teacher_features)

        total_loss = loss_diff + self.lam * loss_repa

        log_dict: Dict[str, float] = {
            "loss_diff":  loss_diff.item(),
            "loss_repa":  loss_repa.item(),
            "cos_sim":    cos_sim_val.item(),
            "loss_total": total_loss.item(),
            "lambda":     self.lam,
        }

        return total_loss, log_dict

    def __repr__(self) -> str:
        return (
            f"CombinedLoss(lambda={self.lam}, "
            f"repa={'enabled' if self.repa_loss_fn else 'disabled'})"
        )


def get_combined_loss(
    repa_head: Optional[ProjectionHead] = None,
    lam: float = 0.5,
) -> CombinedLoss:
    """Factory for train.py. Pass repa_head=None for baseline runs."""
    return CombinedLoss(repa_head=repa_head, lam=lam)