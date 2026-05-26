import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Expected filename inside the Kaggle archive
NPY_FILENAME = "church_outdoor_train_lmdb_color_64.npy"


class LSUNChurchDataset(Dataset):
    """
    Dataset class for the LSUN Church-Outdoor dataset sourced from Kaggle
    (ajaykgp12/lsunchurch).

    The Kaggle archive contains a single NumPy file:
        church_outdoor_train_lmdb_color_64.npy
    which stores all 126 227 images as a uint8 array of shape (N, 64, 64, 3).

    Because LSUN Church is an unconditional dataset (no semantic labels)
    the class id is always 0.  Pass --num-classes=1 in training scripts.

    Args:
        root_dir (str): Directory containing (or receiving) the .npy file.
        transform (callable, optional): Transform applied to each PIL Image.
            Note: images in the .npy are already 64×64; the export script
            will upsample them to 256×256 via the transform.
    """

    KAGGLE_SLUG = "ajaykgp12/lsunchurch"

    def __init__(self, root_dir: str = "../data/lsun_church", transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        npy_path = self._locate_or_download()
        print(f"Loading {npy_path} into memory ...")
        self._images = np.load(npy_path)   # shape: (N, 64, 64, 3), dtype uint8
        assert self._images.ndim == 4 and self._images.shape[3] == 3, (
            f"Unexpected array shape {self._images.shape}. "
            f"Expected (N, H, W, 3)."
        )
        print(f"Loaded {len(self._images)} images, shape per image: "
              f"{self._images.shape[1]}×{self._images.shape[2]}")

    # ------------------------------------------------------------------
    def _locate_or_download(self) -> str:
        root = Path(self.root_dir)
        root.mkdir(parents=True, exist_ok=True)

        # 1) Direct .npy file
        npy = root / NPY_FILENAME
        if npy.is_file():
            return str(npy)

        # 2) Any .npy file in root (in case the name changed slightly)
        npys = list(root.glob("*.npy"))
        if npys:
            print(f"Found .npy: {npys[0]}")
            return str(npys[0])

        # 3) Extract any .zip found
        zips = list(root.glob("*.zip"))
        if zips:
            print(f"Extracting {zips[0]} ...")
            with zipfile.ZipFile(zips[0], "r") as z:
                z.extractall(root)
            return self._locate_or_download()

        # 4) Try kagglehub
        try:
            import kagglehub  # type: ignore
            print(f"Downloading {self.KAGGLE_SLUG} via kagglehub ...")
            dl_path = kagglehub.dataset_download(self.KAGGLE_SLUG)
            # kagglehub downloads to its own cache; find the .npy there
            for p in Path(dl_path).rglob("*.npy"):
                return str(p)
            raise RuntimeError(f"No .npy found in kagglehub path {dl_path}")
        except Exception as exc:
            raise RuntimeError(
                f"Could not locate {NPY_FILENAME} in {root}.\n"
                "Download manually from "
                f"https://www.kaggle.com/datasets/{self.KAGGLE_SLUG} "
                f"and place the .npy file into {root}.\n"
                f"kagglehub error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self._images[idx])   # PIL from (64, 64, 3) uint8
        if self.transform is not None:
            img = self.transform(img)
        info_dict = {
            "filename": f"{idx:06d}.jpg",
            "idx": idx,
            "class_id": 0,   # unconditional
        }
        return img, info_dict
