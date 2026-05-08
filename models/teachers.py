"""
models/teachers.py

Unified TeacherEncoder for REPA representation alignment.

Supported teachers
------------------
  "dinov2"   facebook/dinov2-large  — patch tokens, strong spatial structure
  "clip"     openai/clip-vit-large-patch14  — patch tokens (not just CLS)
  "resnet50" torchvision ResNet-50  — layer4 spatial feature map
  "none"     returns None           — baseline runs (no REPA)

All teachers are always frozen (torch.no_grad + requires_grad=False).

Output contract
---------------
  teacher(pixel_images)  ->  FloatTensor [B, N, C]
    B = batch size
    N = number of spatial patches
    C = feature dimension (teacher-dependent)

  teacher.feature_dim    ->  int   (C)
  teacher.num_patches    ->  int   (N, for a given input resolution)
"""

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# ------------------------------------------------------------------------------
# Per-teacher normalisation transforms
# ------------------------------------------------------------------------------

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def _make_normalizer(mean, std, device: torch.device):
    m = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std,  dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return m, s


def _to_01(x: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] pixel tensors to [0, 1] (SD-VAE output convention)."""
    return (x + 1.0) / 2.0


# ------------------------------------------------------------------------------
# Base class
# ------------------------------------------------------------------------------

class TeacherEncoder(nn.Module):
    """
    Abstract base for all teacher encoders.

    Subclasses must implement :meth:`_encode` which receives a ``[0,1]``
    float32 image tensor of shape ``(B, 3, H, W)`` and returns
    ``(B, N, C)`` patch features.
    """

    def __init__(self, name: str, device: torch.device) -> None:
        super().__init__()
        self.name   = name
        self._dev   = device
        self._feat_dim: Optional[int]    = None
        self._num_patches: Optional[int] = None

    @property
    def feature_dim(self) -> int:
        assert self._feat_dim is not None, "feature_dim not set"
        return self._feat_dim

    @property
    def num_patches(self) -> int:
        assert self._num_patches is not None, "num_patches not set — run a forward pass first"
        return self._num_patches

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, pixel_images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_images: ``(B, 3, H, W)`` float32.
                          Values may be in ``[-1, 1]`` or ``[0, 1]``
                          — the encoder normalises internally.

        Returns:
            ``(B, N, C)`` patch-level feature tensor.
        """
        if pixel_images.min() < -0.1:
            pixel_images = _to_01(pixel_images)
        pixel_images = pixel_images.to(self._dev)
        features = self._encode(pixel_images)
        self._num_patches = features.shape[1]
        return features

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"feature_dim={self._feat_dim}, "
            f"frozen=True)"
        )


# ------------------------------------------------------------------------------
# DINOv2 teacher
# ------------------------------------------------------------------------------

class DINOv2Teacher(TeacherEncoder):
    """
    Teacher based on ``facebook/dinov2-large`` (ViT-L/14).

    Extracts patch tokens (all tokens except CLS) from the final block.
    Output shape: ``(B, N_patch, 1024)``.

    For a 256×256 input: N = (256//14)² = 18² = 324 patches.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__("dinov2", device)

        model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitl14",
            pretrained=True,
        )
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        object.__setattr__(self, "_model", model)

        self._feat_dim = 1024

        mean, std = _make_normalizer(_IMAGENET_MEAN, _IMAGENET_STD, device)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std",  std,  persistent=False)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std
        out = self._model.forward_features(x)
        # "x_norm_patchtokens": (B, N_patch, 1024) — CLS already excluded
        return out["x_norm_patchtokens"]


# ------------------------------------------------------------------------------
# CLIP teacher
# ------------------------------------------------------------------------------

class CLIPTeacher(TeacherEncoder):
    """
    Teacher based on ``openai/clip-vit-large-patch14`` (ViT-L/14).

    Extracts intermediate patch tokens via a forward hook on a chosen
    transformer block (default: second-to-last), following the REPA
    convention of using spatial patch features rather than the CLS token.
    Output shape: ``(B, N_patch, 1024)``.

    Input is resized to 224×224 → N = 256 patches.

    Args:
        device:            Torch device.
        extract_layer_idx: Transformer block to hook. Default -2
                           (second-to-last). Use -1 for the final block.
    """

    def __init__(
        self,
        device: torch.device,
        extract_layer_idx: int = -2,
    ) -> None:
        super().__init__("clip", device)

        from transformers import CLIPModel

        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        visual = model.vision_model.to(device).eval()
        for p in visual.parameters():
            p.requires_grad_(False)
        object.__setattr__(self, "_visual", visual)

        self._feat_dim = 1024
        self._extract_layer_idx = extract_layer_idx

        self._resize = transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )

        mean, std = _make_normalizer(_CLIP_MEAN, _CLIP_STD, device)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std",  std,  persistent=False)

        self._hook_output: Optional[torch.Tensor] = None
        self._register_hook()

    def _register_hook(self) -> None:
        """Attach a forward hook to capture patch tokens from a middle layer."""
        layers = self._visual.encoder.layers
        idx = self._extract_layer_idx % len(layers)

        def _hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Exclude CLS token (index 0), keep patch tokens
            self._hook_output = hidden[:, 1:, :]   # (B, N_patch, D)

        layers[idx].register_forward_hook(_hook)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._resize(x)
        x = (x - self._mean) / self._std
        self._hook_output = None
        self._visual(pixel_values=x)
        assert self._hook_output is not None, "CLIP forward hook did not fire"
        return self._hook_output   # (B, N_patch, 1024)


# ------------------------------------------------------------------------------
# ResNet-50 teacher
# ------------------------------------------------------------------------------

class ResNet50Teacher(TeacherEncoder):
    """
    Teacher based on a supervised ResNet-50 (torchvision pretrained).

    Extracts the ``layer4`` spatial feature map and reshapes it to
    ``(B, N, 2048)`` by flattening spatial dimensions.

    For a 256×256 input: spatial size = 8×8 → N = 64 tokens.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__("resnet50", device)

        import torchvision.models as tvm

        backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        encoder = encoder.to(device).eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
        object.__setattr__(self, "_encoder", encoder)

        self._feat_dim = 2048

        mean, std = _make_normalizer(_IMAGENET_MEAN, _IMAGENET_STD, device)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std",  std,  persistent=False)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self._mean) / self._std
        feat_map = self._encoder(x)          # (B, 2048, H//32, W//32)
        B, C, Hf, Wf = feat_map.shape
        return feat_map.flatten(2).transpose(1, 2)   # (B, N, 2048)


# ------------------------------------------------------------------------------
# Null teacher (baseline, no REPA)
# ------------------------------------------------------------------------------

class NullTeacher(TeacherEncoder):
    """
    Placeholder that always returns ``None``.

    Used for baseline runs. ``train.py`` checks ``teacher.name == "none"``
    (or that the return value is None) to skip the REPA loss entirely.
    """

    def __init__(self) -> None:
        super().__init__("none", torch.device("cpu"))
        self._feat_dim    = 0
        self._num_patches = 0

    @torch.no_grad()
    def forward(self, pixel_images: torch.Tensor) -> None:
        return None

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return None


# ------------------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------------------

_TEACHER_REGISTRY = {
    "dinov2":   DINOv2Teacher,
    "clip":     CLIPTeacher,
    "resnet50": ResNet50Teacher,
    "none":     NullTeacher,
}


def get_teacher(
    name: str,
    device: Optional[Union[str, torch.device]] = None,
) -> TeacherEncoder:
    """
    Instantiate and return a frozen teacher encoder by name.

    Args:
        name:   One of ``"dinov2"``, ``"clip"``, ``"resnet50"``, ``"none"``.
        device: Torch device. Auto-detected when None.

    Returns:
        A frozen :class:`TeacherEncoder` instance.

    Raises:
        ValueError: If ``name`` is not in the registry.

    Example::

        teacher = get_teacher("dinov2", device="cuda")
        y = teacher(pixel_images)      # (B, N, 1024)
        print(teacher.feature_dim)     # 1024
        print(teacher.num_patches)     # 324  (for 256x256 input)

        teacher = get_teacher("none")
        assert teacher(pixel_images) is None
    """
    name = name.lower().strip()
    if name not in _TEACHER_REGISTRY:
        raise ValueError(
            f"Unknown teacher '{name}'. "
            f"Choose from: {list(_TEACHER_REGISTRY.keys())}"
        )

    if name == "none":
        return NullTeacher()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    return _TEACHER_REGISTRY[name](device=device)