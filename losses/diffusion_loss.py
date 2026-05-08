"""
losses/diffusion_loss.py

MSE velocity-prediction loss for the SiT interpolant framework.

The SiT training objective is to minimise the mean squared error between
the model's predicted velocity v_theta(z_t, t) and the true velocity u_t:

    u_t(z, eps) = eps - z          (for linear interpolation: alpha_t=1-t, sigma_t=t)

    L_diff = E_{z, eps, t} [ || v_theta(z_t, t) - u_t(z, eps) ||_2^2 ]

where z_t = (1-t)*z + t*eps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """
    MSE loss between predicted velocity and true velocity.

    The true velocity under the linear interpolation path
    z_t = (1-t)*z + t*eps is simply u_t = eps - z, independent of t.

    Args:
        reduction: "mean" (default) or "none".
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        v_pred: torch.Tensor,
        z:      torch.Tensor,
        eps:    torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            v_pred: Predicted velocity from SiT, shape (B, C, H, W).
            z:      Clean latent (VAE-encoded image), shape (B, C, H, W).
            eps:    Gaussian noise sampled as N(0, I), shape (B, C, H, W).

        Returns:
            Scalar MSE loss.
        """
        u_t = eps - z                          # true velocity
        return F.mse_loss(v_pred, u_t, reduction=self.reduction)

    def __repr__(self) -> str:
        return f"DiffusionLoss(reduction={self.reduction})"


def diffusion_loss(
    v_pred: torch.Tensor,
    z:      torch.Tensor,
    eps:    torch.Tensor,
) -> torch.Tensor:
    """Functional interface. Returns scalar MSE loss."""
    return F.mse_loss(v_pred, eps - z)