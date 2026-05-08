"""
models/autoencoder.py

Thin wrapper around the pretrained Stable Diffusion VAE
(stabilityai/sd-vae-ft-mse from HuggingFace).

Public API
----------
  get_vae(device)          -> VAEWrapper (frozen, eval)
  VAEWrapper.encode(x)     -> z  FloatTensor [B, 4, H//8, W//8]
  VAEWrapper.decode(z)     -> x  FloatTensor [B, 3, H,    W   ]

The VAE is always frozen — no gradient computation ever flows through it.
Scaling convention follows the original LDM / Stable Diffusion codebase:
  z_scaled = z_raw * 0.18215   (encode)
  x        = decode(z / 0.18215)
"""

import torch
import torch.nn as nn
from typing import Optional, Union


# SD-VAE scaling constant (from the original LDM paper / SD codebase)
_VAE_SCALE_FACTOR: float = 0.18215

# HuggingFace model identifier
_VAE_MODEL_ID: str = "stabilityai/sd-vae-ft-mse"


class VAEWrapper(nn.Module):
    """
    Frozen wrapper around the Stable Diffusion VAE
    (``stabilityai/sd-vae-ft-mse``).

    The underlying ``AutoencoderKL`` is loaded from HuggingFace, moved to the
    requested device, set to eval mode, and permanently frozen. No gradient
    will ever be computed through this module.

    Scaling convention
    ~~~~~~~~~~~~~~~~~~
    Raw VAE latents have unit variance ~1/0.18215. Following the original LDM
    codebase, we rescale them so the latent distribution has approximately unit
    variance::

        z_scaled = vae.encode(x) * 0.18215   # stored in cache / used by SiT
        x_hat    = vae.decode(z_scaled)       # inverse scaling applied inside

    Args:
        model_id: HuggingFace model identifier. Defaults to
                  ``"stabilityai/sd-vae-ft-mse"``.
        device:   Torch device. The model is moved here after loading.
    """

    def __init__(
        self,
        model_id: str = _VAE_MODEL_ID,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()

        from diffusers import AutoencoderKL

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        # Load pretrained VAE
        vae = AutoencoderKL.from_pretrained(model_id)
        vae = vae.to(self._device)
        vae.eval()

        # Permanently freeze — no gradients, no parameter updates ever
        for param in vae.parameters():
            param.requires_grad_(False)

        # Store as a non-module attribute so it is never included in
        # self.parameters() / self.state_dict() of the *outer* model
        # (e.g. SiT). Using register_module here would expose it to optimizers.
        # We deliberately bypass nn.Module registration:
        object.__setattr__(self, "_vae", vae)

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of pixel images to scaled latent vectors.

        Args:
            x: ``FloatTensor`` of shape ``(B, 3, H, W)``, values in
               ``[-1, 1]`` (standard LDM normalisation).
               H and W must be divisible by 8.

        Returns:
            ``FloatTensor`` of shape ``(B, 4, H//8, W//8)``.
            Latents are sampled from the posterior (reparameterisation trick)
            and scaled by ``0.18215``.
        """
        # AutoencoderKL.encode() returns a DiagonalGaussianDistribution
        posterior = self._vae.encode(x).latent_dist
        z = posterior.sample()                  # reparameterisation trick
        return z * _VAE_SCALE_FACTOR

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode scaled latent vectors back to pixel images.

        Args:
            z: ``FloatTensor`` of shape ``(B, 4, H_lat, W_lat)``, scaled
               by ``0.18215`` (i.e. the output of :meth:`encode`).

        Returns:
            ``FloatTensor`` of shape ``(B, 3, H_lat * 8, W_lat * 8)``,
            values approximately in ``[-1, 1]``.
        """
        z_unscaled = z / _VAE_SCALE_FACTOR
        return self._vae.decode(z_unscaled).sample

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def scale_factor(self) -> float:
        """The latent scaling constant (0.18215)."""
        return _VAE_SCALE_FACTOR

    @property
    def latent_channels(self) -> int:
        """Number of latent channels (4 for SD-VAE)."""
        return self._vae.config.latent_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`encode` — makes the wrapper usable as nn.Module."""
        return self.encode(x)

    def __repr__(self) -> str:
        return (
            f"VAEWrapper("
            f"model_id='{_VAE_MODEL_ID}', "
            f"scale_factor={_VAE_SCALE_FACTOR}, "
            f"latent_channels={self.latent_channels}, "
            f"frozen=True)"
        )


# ------------------------------------------------------------------------------
# Module-level factory  (used by latent_cache.py and train.py)
# ------------------------------------------------------------------------------

_VAE_SINGLETON: Optional[VAEWrapper] = None


def get_vae(
    device: Optional[Union[str, torch.device]] = None,
    model_id: str = _VAE_MODEL_ID,
    singleton: bool = True,
) -> VAEWrapper:
    """
    Return a :class:`VAEWrapper` instance.

    By default (``singleton=True``) this function caches the wrapper in a
    module-level variable so the VAE weights are loaded from disk only once per
    process — useful when ``encode_and_cache`` is called inside a training
    script that already holds the VAE in memory.

    Args:
        device:    Torch device. Ignored when returning the cached singleton.
        model_id:  HuggingFace model identifier.
        singleton: If True, reuse the cached instance across calls.

    Returns:
        A frozen, eval-mode :class:`VAEWrapper`.

    Example::

        from models.autoencoder import get_vae

        vae = get_vae(device="cuda")
        z   = vae.encode(pixel_images)   # (B, 4, 32, 32) for 256x256 input
        x   = vae.decode(z)              # (B, 3, 256, 256)
    """
    global _VAE_SINGLETON

    if singleton and _VAE_SINGLETON is not None:
        return _VAE_SINGLETON

    wrapper = VAEWrapper(model_id=model_id, device=device)

    if singleton:
        _VAE_SINGLETON = wrapper

    return wrapper