"""
data/dataset.py

ImageNet and Stanford Cars dataset loaders for the REPA diffusion experiments codebase.

- ImageNet: accepts a `fraction` parameter (0.01, 0.05, 0.10, 0.20, 1.0) and deterministically
  subsamples that fraction of training images using a fixed random seed.
- Stanford Cars: standard train/test split.

Both loaders output batches of (pixel_images, class_labels) at a configurable resolution
(default 256x256).

Augmentation: random horizontal flip + center crop only (no color jitter — it corrupts
spatial structure needed for teacher alignment).
"""

import os
import random
import numpy as np
from pathlib import Path
from typing import Literal, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets import StanfordCars


# ──────────────────────────────────────────────────────────────────────────────
# Shared transform factory
# ──────────────────────────────────────────────────────────────────────────────

def build_transform(resolution: int = 256, split: str = "train") -> transforms.Compose:
    """
    Build the augmentation pipeline.

    Training:  RandomHorizontalFlip  →  CenterCrop  →  Resize  →  ToTensor  →  Normalize
    Val/Test:  CenterCrop            →  Resize  →  ToTensor  →  Normalize

    No color jitter: it corrupts the spatial structure that REPA teacher alignment relies on.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        return transforms.Compose([
            transforms.Resize(int(resolution * 256 / 224)),
            transforms.CenterCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(resolution * 256 / 224)),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            normalize,
        ])


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic subsample helper
# ──────────────────────────────────────────────────────────────────────────────

def deterministic_subset(
    dataset: Dataset,
    fraction: float,
    seed: int = 42,
) -> Subset:
    """
    Return a Subset containing `fraction` of `dataset`, chosen deterministically
    using `seed`. The subset is stratified by class label when the dataset exposes
    a `targets` attribute (ImageNet / StanfordCars both do).

    Args:
        dataset:  A torchvision Dataset that has a `.targets` list.
        fraction: Fraction of the dataset to keep, in (0, 1].
        seed:     Fixed random seed for reproducibility.

    Returns:
        A torch.utils.data.Subset with the selected indices.
    """
    if not 0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    if fraction == 1.0:
        return Subset(dataset, list(range(len(dataset))))

    targets = np.array(dataset.targets)
    classes = np.unique(targets)

    rng = random.Random(seed)
    selected_indices: list[int] = []

    for cls in classes:
        cls_indices = np.where(targets == cls)[0].tolist()
        k = max(1, round(len(cls_indices) * fraction))
        chosen = rng.sample(cls_indices, k)
        selected_indices.extend(chosen)

    selected_indices.sort()
    return Subset(dataset, selected_indices)


# ──────────────────────────────────────────────────────────────────────────────
# ImageNet loader
# ──────────────────────────────────────────────────────────────────────────────

def get_imagenet_dataset(
    root: str,
    split: Literal["train", "val"] = "train",
    fraction: float = 1.0,
    resolution: int = 256,
    seed: int = 42,
) -> Dataset:
    """
    Return an ImageNet dataset (torchvision.datasets.ImageNet) with optional
    deterministic sub-sampling.

    Expected directory layout (standard ImageNet structure)::

        root/
          train/
            n01440764/
              *.JPEG
            ...
          val/
            n01440764/
              *.JPEG
            ...

    Args:
        root:       Path to the ImageNet root directory.
        split:      "train" or "val".
        fraction:   Fraction of training images to keep (only applied for
                    split="train"). One of {0.01, 0.05, 0.10, 0.20, 1.0}.
        resolution: Output spatial resolution (H = W = resolution).
        seed:       Random seed for deterministic sub-sampling.

    Returns:
        A Dataset yielding (image_tensor, class_label) pairs where
        image_tensor has shape (3, resolution, resolution) and
        class_label is an integer in [0, 999].
    """
    transform = build_transform(resolution=resolution, split=split)
    dataset = datasets.ImageNet(root=root, split=split, transform=transform)

    if split == "train" and fraction < 1.0:
        dataset = deterministic_subset(dataset, fraction=fraction, seed=seed)

    return dataset


def get_imagenet_dataloader(
    root: str,
    split: Literal["train", "val"] = "train",
    fraction: float = 1.0,
    resolution: int = 256,
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
    seed: int = 42,
) -> DataLoader:
    """
    Convenience wrapper: returns a DataLoader for ImageNet.

    Args:
        root:        Path to the ImageNet root directory.
        split:       "train" or "val".
        fraction:    Fraction of training data to use.
        resolution:  Output image resolution.
        batch_size:  Mini-batch size.
        num_workers: Number of DataLoader worker processes.
        pin_memory:  Whether to pin memory (recommended for GPU training).
        seed:        Random seed for deterministic sub-sampling.

    Returns:
        A configured torch.utils.data.DataLoader.
    """
    dataset = get_imagenet_dataset(
        root=root,
        split=split,
        fraction=fraction,
        resolution=resolution,
        seed=seed,
    )
    shuffle = (split == "train")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Stanford Cars loader
# ──────────────────────────────────────────────────────────────────────────────

def get_stanford_cars_dataset(
    root: str,
    split: Literal["train", "test"] = "train",
    resolution: int = 256,
) -> Dataset:
    """
    Return a Stanford Cars dataset using the standard train/test split.

    The dataset has 196 classes (make/model/year combinations), ~8,144 training
    images and ~8,041 test images.

    Expected directory layout::

        root/
          stanford_cars/
            cars_train/
            cars_test/
            cars_devkit/

    Args:
        root:       Root directory that contains (or will contain) the dataset.
        split:      "train" or "test".
        resolution: Output spatial resolution.

    Returns:
        A Dataset yielding (image_tensor, class_label) pairs where
        image_tensor has shape (3, resolution, resolution) and
        class_label is an integer in [0, 195].
    """
    transform = build_transform(resolution=resolution, split=split)

    dataset = StanfordCars(
        root=root,
        split=split,
        transform=transform,
        download=False,   # set True on first use if you want auto-download
    )
    return dataset


def get_stanford_cars_dataloader(
    root: str,
    split: Literal["train", "test"] = "train",
    resolution: int = 256,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Convenience wrapper: returns a DataLoader for Stanford Cars.

    Args:
        root:        Root directory for the dataset.
        split:       "train" or "test".
        resolution:  Output image resolution.
        batch_size:  Mini-batch size.
        num_workers: Number of DataLoader worker processes.
        pin_memory:  Whether to pin memory.

    Returns:
        A configured torch.utils.data.DataLoader.
    """
    dataset = get_stanford_cars_dataset(
        root=root,
        split=split,
        resolution=resolution,
    )
    shuffle = (split == "train")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Unified factory (config-driven entry point)
# ──────────────────────────────────────────────────────────────────────────────

def get_dataset(
    dataset_name: Literal["imagenet", "stanford_cars"],
    root: str,
    split: str = "train",
    fraction: float = 1.0,
    resolution: int = 256,
    seed: int = 42,
) -> Dataset:
    """
    Config-driven factory that returns the correct Dataset object.

    This is the preferred entry point for train.py and evaluate.py.

    Args:
        dataset_name: One of "imagenet" or "stanford_cars".
        root:         Path to the dataset root.
        split:        Dataset split to load ("train" / "val" / "test").
        fraction:     Sub-sampling fraction (ImageNet only; ignored for Stanford Cars).
        resolution:   Output spatial resolution.
        seed:         Random seed (ImageNet sub-sampling only).

    Returns:
        A torch.utils.data.Dataset yielding (image_tensor, class_label) pairs.

    Raises:
        ValueError: If dataset_name is not recognised.
    """
    if dataset_name == "imagenet":
        return get_imagenet_dataset(
            root=root,
            split=split,
            fraction=fraction,
            resolution=resolution,
            seed=seed,
        )
    elif dataset_name == "stanford_cars":
        return get_stanford_cars_dataset(
            root=root,
            split=split,
            resolution=resolution,
        )
    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            "Choose one of: 'imagenet', 'stanford_cars'."
        )


def get_dataloader(
    dataset_name: Literal["imagenet", "stanford_cars"],
    root: str,
    split: str = "train",
    fraction: float = 1.0,
    resolution: int = 256,
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
    seed: int = 42,
) -> DataLoader:
    """
    Config-driven factory that returns a DataLoader for any supported dataset.

    Args:
        dataset_name: One of "imagenet" or "stanford_cars".
        root:         Path to the dataset root.
        split:        Dataset split.
        fraction:     Sub-sampling fraction (ImageNet only).
        resolution:   Output spatial resolution.
        batch_size:   Mini-batch size.
        num_workers:  Number of DataLoader worker processes.
        pin_memory:   Whether to pin memory.
        seed:         Random seed for deterministic sub-sampling.

    Returns:
        A configured torch.utils.data.DataLoader.
    """
    dataset = get_dataset(
        dataset_name=dataset_name,
        root=root,
        split=split,
        fraction=fraction,
        resolution=resolution,
        seed=seed,
    )
    shuffle = (split == "train")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
    )