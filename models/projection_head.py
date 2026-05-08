"""
models/projection_head.py

Small MLP projection head that maps SiT hidden states into the teacher
feature space for REPA representation alignment.

Architecture
------------
  Linear(student_dim -> hidden_dim)
  LayerNorm(hidden_dim)
  GELU
  Linear(hidden_dim -> teacher_dim)

One projection head is instantiated per experiment based on the config.
student_dim  = hidden size of the chosen SiT model
teacher_dim  = feature dimension of the chosen teacher encoder
hidden_dim   = student_dim  (default, can be overridden)
"""

from typing import Optional

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    Two-layer MLP that projects SiT hidden states into teacher feature space.

    Architecture::

        Linear(student_dim -> hidden_dim)
        LayerNorm(hidden_dim)
        GELU
        Linear(hidden_dim -> teacher_dim)

    The output is NOT L2-normalised here — normalisation is applied inside
    REPALoss immediately before cosine similarity computation. This keeps the
    projection head purely linear+norm and lets the loss module own the
    similarity logic.

    Args:
        student_dim: Dimensionality of SiT hidden states
                     (384 / 768 / 1024 for S / B / L).
        teacher_dim: Dimensionality of teacher patch features
                     (1024 for DINOv2 & CLIP, 2048 for ResNet-50).
        hidden_dim:  Width of the intermediate layer. Defaults to
                     ``student_dim`` when None.

    Shape:
        Input  h:       ``(B, N, student_dim)``
        Output h_proj:  ``(B, N, teacher_dim)``

    Example::

        head = ProjectionHead(student_dim=384, teacher_dim=1024)
        h_proj = head(hidden_states[5])   # (B, N, 1024)
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = student_dim

        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.hidden_dim  = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(student_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, teacher_dim, bias=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Student hidden states ``(B, N, student_dim)``.
               Typically ``hidden_states[align_layer]`` from SiT.

        Returns:
            Projected features ``(B, N, teacher_dim)``.
        """
        return self.net(h)

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = (
            p for p in self.parameters()
            if (not trainable_only or p.requires_grad)
        )
        return sum(p.numel() for p in params)

    def __repr__(self) -> str:
        return (
            f"ProjectionHead("
            f"student_dim={self.student_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"teacher_dim={self.teacher_dim}, "
            f"params={self.num_parameters():,})"
        )


# ------------------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------------------

def get_projection_head(
    student_dim: int,
    teacher_dim: int,
    hidden_dim: Optional[int] = None,
) -> ProjectionHead:
    """
    Instantiate a ProjectionHead.

    Called once per experiment from train.py after both the SiT model
    and teacher encoder have been constructed::

        model   = get_sit_model("S", num_classes=1000)
        teacher = get_teacher("dinov2", device="cuda")
        head    = get_projection_head(
                      student_dim=model.hidden_size,   # 384 for SiT-S
                      teacher_dim=teacher.feature_dim, # 1024 for DINOv2
                  )

    Args:
        student_dim: SiT hidden size (model.hidden_size).
        teacher_dim: Teacher feature dimension (teacher.feature_dim).
        hidden_dim:  Optional intermediate width override.

    Returns:
        An initialised ProjectionHead.
    """
    return ProjectionHead(
        student_dim=student_dim,
        teacher_dim=teacher_dim,
        hidden_dim=hidden_dim,
    )