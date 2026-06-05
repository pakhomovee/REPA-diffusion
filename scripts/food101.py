"""PyTorch Dataset wrapper for Food-101 used in REPA training.

Uses torchvision.datasets.Food101 which handles auto-download.
101 food categories, 750 training / 250 test images per class.
Source: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
"""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import Food101 as TorchFood101


class Food101Dataset(Dataset):
    """
    Thin wrapper around torchvision Food101 that returns (image, info_dict)
    matching the interface expected by export_food101_for_repa.py.

    Args:
        root_dir: Directory where the dataset is stored / will be downloaded.
        split:    "train" (75,750 images) or "test" (25,250 images).
        transform: Optional torchvision transform applied to each PIL Image.
        download: If True, download the dataset if not already present.
    """

    CLASS_NAMES: list[str] = []  # populated after first instantiation

    def __init__(
        self,
        root_dir: str = "../data/food101",
        split: str = "train",
        transform=None,
        download: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self._base = TorchFood101(
            root=root_dir,
            split=split,
            transform=None,   # we apply transform manually
            download=download,
        )
        # torchvision stores class names in .classes
        Food101Dataset.CLASS_NAMES = self._base.classes

    # ------------------------------------------------------------------
    @property
    def num_classes(self) -> int:
        return len(self._base.classes)

    @property
    def classes(self) -> list[str]:
        return self._base.classes

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        img, label = self._base[idx]          # PIL Image, int
        if self.transform is not None:
            img = self.transform(img)
        info_dict = {
            "filename": str(self._base._image_files[idx]),
            "idx": idx,
            "class_id": label,
            "class_name": self._base.classes[label],
        }
        return img, info_dict
