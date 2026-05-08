"""
models/sit.py

SiT — Scalable Interpolant Transformer
Diffusion transformer backbone for latent velocity prediction.

Architecture
------------
  - Patch-embed the noisy latent z_t into tokens
  - Prepend a time-step token (sinusoidal embedding) and a class-label token
    (learnable embedding, with classifier-free guidance dropout)
  - N transformer blocks (pre-norm, multi-head self-attention + MLP)
  - Un-patch to produce the predicted velocity v  (same shape as z_t)
  - Every transformer block exposes its hidden state for REPA alignment

Public API
----------
  SiT(config)                   nn.Module
  SiT.forward(z_t, t, y)   ->  v_pred [B, C, H, W],
                                hidden_states {layer_idx: Tensor [B, N_patch, D]}
  get_sit_model(size, config)   -> SiT
  SiT_S / SiT_B / SiT_L        named constructors for the three model sizes

Model sizes  (following the original SiT / DiT paper)
-----------
  SiT-S  depth=12  hidden=384  heads=6   mlp_ratio=4
  SiT-B  depth=12  hidden=768  heads=12  mlp_ratio=4
  SiT-L  depth=28  hidden=1024 heads=16  mlp_ratio=4
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------------
# Configuration dataclass
# ------------------------------------------------------------------------------

@dataclass
class SiTConfig:
    """
    Hyperparameter bundle for SiT.

    Args:
        input_size:      Spatial side-length of the latent grid
                         (e.g. 32 for 256×256 images with 8× VAE).
        patch_size:      Side-length of each patch token (default 2).
        in_channels:     Latent channels from the VAE (4 for SD-VAE).
        hidden_size:     Transformer hidden dimension.
        depth:           Number of transformer blocks.
        num_heads:       Number of attention heads.
        mlp_ratio:       Expansion ratio for the MLP feed-forward.
        num_classes:     Number of class labels (1000 for ImageNet,
                         196 for Stanford Cars). Set to 0 to disable
                         class conditioning.
        class_dropout_prob: Probability of replacing the class label with
                         an unconditional token during training
                         (classifier-free guidance).
        learn_sigma:     If True the model predicts both velocity and
                         log-variance (not used in base SiT, kept for
                         compatibility).
    """
    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_classes: int = 1000
    class_dropout_prob: float = 0.1
    learn_sigma: bool = False


# Pre-defined size configs
def _sit_s_config(**kwargs) -> SiTConfig:
    return SiTConfig(hidden_size=384, depth=12, num_heads=6, **kwargs)

def _sit_b_config(**kwargs) -> SiTConfig:
    return SiTConfig(hidden_size=768, depth=12, num_heads=12, **kwargs)

def _sit_l_config(**kwargs) -> SiTConfig:
    return SiTConfig(hidden_size=1024, depth=28, num_heads=16, **kwargs)


# ------------------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Sinusoidal time-step embedding followed by a 2-layer MLP projector.

    Args:
        hidden_size:  Output embedding dimension.
        freq_embed_size: Dimension of the sinusoidal base embedding.
    """

    def __init__(self, hidden_size: int, freq_embed_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_embed_size = freq_embed_size

    @staticmethod
    def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        """t: (B,) floats in [0, 1]  ->  (B, dim) sinusoidal embedding."""
        assert dim % 2 == 0
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )                                          # (half,)
        args = t[:, None] * freqs[None, :]        # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) -> (B, hidden_size)"""
        x = self._sinusoidal_embedding(t, self.freq_embed_size)
        return self.mlp(x)


class LabelEmbedder(nn.Module):
    """
    Learnable class-label embedding with classifier-free guidance dropout.

    A special *unconditional* token (index ``num_classes``) is used when the
    label is dropped during training.

    Args:
        num_classes:      Number of real class labels.
        hidden_size:      Embedding dimension.
        dropout_prob:     Probability of replacing the label with the
                          unconditional token.
    """

    def __init__(
        self, num_classes: int, hidden_size: int, dropout_prob: float
    ) -> None:
        super().__init__()
        self.dropout_prob = dropout_prob
        # +1 for the unconditional token
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def token_drop(self, labels: torch.Tensor, force_drop: bool = False) -> torch.Tensor:
        """Replace labels with the unconditional token with probability dropout_prob."""
        if force_drop or (self.training and self.dropout_prob > 0.0):
            drop_mask = torch.rand_like(labels, dtype=torch.float) < self.dropout_prob
            labels = torch.where(drop_mask, torch.full_like(labels, self.num_classes), labels)
        return labels

    def forward(
        self, labels: torch.Tensor, force_drop: bool = False
    ) -> torch.Tensor:
        """labels: (B,) int64  ->  (B, hidden_size)"""
        labels = self.token_drop(labels, force_drop)
        return self.embedding_table(labels)


class PatchEmbed(nn.Module):
    """
    2-D patch embedding: (B, C, H, W) -> (B, N_patch, hidden_size).

    Args:
        patch_size:  Side-length of each patch.
        in_channels: Number of input channels.
        hidden_size: Output embedding dimension.
    """

    def __init__(
        self, patch_size: int, in_channels: int, hidden_size: int
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)  ->  (B, N_patch, hidden_size)"""
        x = self.proj(x)                          # (B, D, H//p, W//p)
        B, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)         # (B, Hp*Wp, D)
        return x


class Attention(nn.Module):
    """Multi-head self-attention with fused QKV projection."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)        # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)   # (B, heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class MLP(nn.Module):
    """Point-wise feed-forward network with GELU activation."""

    def __init__(self, hidden_size: int, mlp_ratio: float) -> None:
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, inner)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(inner, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SiTBlock(nn.Module):
    """
    Single SiT transformer block.

    Uses adaptive layer norm (adaLN) conditioned on the combined time + class
    conditioning vector ``c``. Each block produces shift/scale/gate parameters
    from ``c`` via a zero-initialised MLP.

    The hidden state *after* the attention sub-layer (before the MLP) is
    captured and returned so REPA can align it with teacher features.

    Args:
        hidden_size: Transformer hidden dimension.
        num_heads:   Number of attention heads.
        mlp_ratio:   MLP expansion ratio.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = Attention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp   = MLP(hidden_size, mlp_ratio)

        # adaLN-Zero: 6 vectors (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        # Zero-initialise so the block acts as identity at the start of training
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: token sequence ``(B, N, D)``
            c: conditioning vector ``(B, D)``

        Returns:
            x:      updated token sequence ``(B, N, D)``
            h_attn: hidden state after attention ``(B, N, D)``
                    — used by REPA for representation alignment
        """
        shift1, scale1, gate1, shift2, scale2, gate2 = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        # Attention sub-layer
        h_attn = x + gate1.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift1, scale1)
        )

        # MLP sub-layer
        x = h_attn + gate2.unsqueeze(1) * self.mlp(
            modulate(self.norm2(h_attn), shift2, scale2)
        )

        return x, h_attn


class FinalLayer(nn.Module):
    """
    Final adaLN layer that projects tokens back to patch pixels.

    Args:
        hidden_size: Transformer hidden dimension.
        patch_size:  Spatial side-length of each patch.
        out_channels: Number of output channels per patch pixel.
    """

    def __init__(
        self, hidden_size: int, patch_size: int, out_channels: int
    ) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ------------------------------------------------------------------------------
# Main SiT model
# ------------------------------------------------------------------------------

class SiT(nn.Module):
    """
    Scalable Interpolant Transformer (SiT).

    Predicts the velocity field ``v(z_t, t)`` for a latent diffusion model
    operating on the interpolation path ``z_t = (1-t)*z + t*eps``.

    Every transformer block exposes its hidden state (after attention, before
    MLP) so that the REPA loss can align them with teacher patch features.

    Args:
        config: :class:`SiTConfig` with all hyperparameters.

    Inputs to :meth:`forward`:
        z_t: ``(B, C, H, W)`` — noisy latent at interpolation time t
        t:   ``(B,)``         — interpolation time in [0, 1]
        y:   ``(B,)`` int64   — class labels (pass None to disable conditioning)

    Outputs of :meth:`forward`:
        v_pred:       ``(B, C, H, W)``             — predicted velocity
        hidden_states: ``{layer_idx: (B, N, D)}``  — per-block hidden states
    """

    def __init__(self, config: SiTConfig) -> None:
        super().__init__()
        self.config = config

        C    = config.in_channels
        D    = config.hidden_size
        p    = config.patch_size
        H    = config.input_size
        out_channels = C * 2 if config.learn_sigma else C

        self.num_patches  = (H // p) ** 2
        self.patch_size   = p
        self.in_channels  = C
        self.out_channels = out_channels
        self.hidden_size  = D

        # ── Embeddings ────────────────────────────────────────────────────────
        self.patch_embed = PatchEmbed(p, C, D)
        self.t_embedder  = TimestepEmbedder(D)
        self.y_embedder  = (
            LabelEmbedder(config.num_classes, D, config.class_dropout_prob)
            if config.num_classes > 0 else None
        )

        # Fixed sinusoidal positional embedding for patch tokens
        self.register_buffer(
            "pos_embed",
            self._build_pos_embed(self.num_patches, D),
            persistent=False,
        )

        # ── Transformer blocks ────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            SiTBlock(D, config.num_heads, config.mlp_ratio)
            for _ in range(config.depth)
        ])

        # ── Output head ───────────────────────────────────────────────────────
        self.final_layer = FinalLayer(D, p, out_channels)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        def _basic_init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        self.apply(_basic_init)

        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.proj.bias)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    # ------------------------------------------------------------------
    # Positional embedding
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pos_embed(num_patches: int, hidden_size: int) -> torch.Tensor:
        """Build a fixed 1-D sinusoidal position embedding (1, N, D)."""
        position = torch.arange(num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float)
            * (-math.log(10000.0) / hidden_size)
        )
        pe = torch.zeros(num_patches, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)   # (1, N, D)

    # ------------------------------------------------------------------
    # Un-patch helper
    # ------------------------------------------------------------------

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N_patch, p^2*C_out) -> (B, C_out, H, W)"""
        p  = self.patch_size
        c  = self.out_channels
        h  = w = int(self.num_patches ** 0.5)
        B  = x.shape[0]
        x = x.reshape(B, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, c, h * p, w * p)
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Args:
            z_t: Noisy latent ``(B, C, H, W)``.
            t:   Interpolation time ``(B,)`` in ``[0, 1]``.
            y:   Class labels ``(B,)`` int64, or None if unconditional.

        Returns:
            v_pred:        Predicted velocity ``(B, C, H, W)``.
            hidden_states: Dict mapping layer_index -> ``(B, N_patch, D)``
                           for all transformer blocks (0-based indices).
        """
        # ── Token sequence ────────────────────────────────────────────────────
        x = self.patch_embed(z_t)               # (B, N, D)
        x = x + self.pos_embed

        # ── Conditioning vector ───────────────────────────────────────────────
        c = self.t_embedder(t)                  # (B, D)
        if self.y_embedder is not None and y is not None:
            c = c + self.y_embedder(y)

        # ── Transformer blocks ────────────────────────────────────────────────
        hidden_states: Dict[int, torch.Tensor] = {}

        for idx, block in enumerate(self.blocks):
            x, h_attn = block(x, c)
            hidden_states[idx] = h_attn         # (B, N_patch, D)

        # ── Output ────────────────────────────────────────────────────────────
        x = self.final_layer(x, c)              # (B, N_patch, p^2 * C_out)
        v_pred = self._unpatchify(x)            # (B, C_out, H, W)

        return v_pred, hidden_states

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = (
            p for p in self.parameters()
            if (not trainable_only or p.requires_grad)
        )
        return sum(p.numel() for p in params)

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"SiT("
            f"depth={cfg.depth}, hidden={cfg.hidden_size}, "
            f"heads={cfg.num_heads}, patches={self.num_patches}, "
            f"params={self.num_parameters():,})"
        )


# ------------------------------------------------------------------------------
# Named constructors
# ------------------------------------------------------------------------------

def SiT_S(num_classes: int = 1000, input_size: int = 32, **kwargs) -> SiT:
    """SiT-Small: depth=12, hidden=384, heads=6  (~33 M params)."""
    return SiT(_sit_s_config(num_classes=num_classes, input_size=input_size, **kwargs))

def SiT_B(num_classes: int = 1000, input_size: int = 32, **kwargs) -> SiT:
    """SiT-Base: depth=12, hidden=768, heads=12  (~130 M params)."""
    return SiT(_sit_b_config(num_classes=num_classes, input_size=input_size, **kwargs))

def SiT_L(num_classes: int = 1000, input_size: int = 32, **kwargs) -> SiT:
    """SiT-Large: depth=28, hidden=1024, heads=16  (~458 M params)."""
    return SiT(_sit_l_config(num_classes=num_classes, input_size=input_size, **kwargs))

_SIZE_MAP = {"S": SiT_S, "B": SiT_B, "L": SiT_L}


def get_sit_model(
    size: str,
    num_classes: int = 1000,
    input_size: int = 32,
    **kwargs,
) -> SiT:
    """
    Config-driven factory used by ``train.py`` and ``evaluate.py``.

    Args:
        size:        One of ``"S"``, ``"B"``, ``"L"``.
        num_classes: Number of class labels (1000 for ImageNet, 196 for Cars).
        input_size:  Spatial side-length of the latent (image_res // 8).
        **kwargs:    Additional :class:`SiTConfig` overrides.

    Returns:
        A freshly initialised :class:`SiT` model.

    Example::

        model = get_sit_model("S", num_classes=1000, input_size=32)
        v, hs = model(z_t, t, y)
        # v:     (B, 4, 32, 32)
        # hs[5]: (B, 256, 384)  — block 5 hidden state for REPA
    """
    if size not in _SIZE_MAP:
        raise ValueError(f"Unknown SiT size '{size}'. Choose from {list(_SIZE_MAP.keys())}.")
    return _SIZE_MAP[size](num_classes=num_classes, input_size=input_size, **kwargs)