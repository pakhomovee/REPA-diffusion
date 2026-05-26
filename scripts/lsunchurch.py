import os
import zipfile
from pathlib import Path

try:
    from natsort import natsorted
    SORTFN = natsorted
except ImportError:
    SORTFN = sorted

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LSUNChurchDataset(Dataset):
    """
    Dataset class for the LSUN Church-Outdoor dataset sourced from Kaggle
    (ajaykgp12/lsunchurch).

    Expected on-disk layout after extraction (any of the following):
        <root_dir>/lsun_church/*.jpg
        <root_dir>/*.jpg
        <root_dir>/<any_sub>/*.jpg   (searched recursively)

    Because LSUN Church is an unconditional dataset (no semantic labels)
    the class id is always 0.  Pass --num-classes=1 in training scripts.

    Args:
        root_dir (str): Directory containing (or receiving) the dataset.
        transform (callable, optional): Transform applied to each PIL Image.
    """

    KAGGLE_SLUG = "ajaykgp12/lsunchurch"
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, root_dir: str = "../data/lsun_church", transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        self.image_dir = self._locate_or_download()
        self.filenames = SORTFN(
            str(p) for p in Path(self.image_dir).rglob("*")
            if p.is_file() and p.suffix.lower() in self._IMAGE_EXTS
        )
        if not self.filenames:
            raise RuntimeError(
                f"No images found under {self.image_dir}.\n"
                "Download from https://www.kaggle.com/datasets/ajaykgp12/lsunchurch "
                f"and extract into {root_dir}."
            )

    def _locate_or_download(self) -> str:
        root = Path(self.root_dir)
        root.mkdir(parents=True, exist_ok=True)

        # 1) Walk for any directory that directly contains images
        for dirpath, _, fnames in os.walk(root):
            if any(Path(f).suffix.lower() in self._IMAGE_EXTS for f in fnames):
                return dirpath

        # 2) Extract any .zip found
        zips = list(root.glob("*.zip"))
        if zips:
            print(f"Extracting {zips[0]} ...")
            with zipfile.ZipFile(zips[0], "r") as z:
                z.extractall(root)
            return self._locate_or_download()

        # 3) Try kagglehub
        try:
            import kagglehub  # type: ignore
            print(f"Downloading {self.KAGGLE_SLUG} via kagglehub ...")
            path = kagglehub.dataset_download(self.KAGGLE_SLUG)
            return path
        except Exception as exc:
            raise RuntimeError(
                f"Could not locate LSUN Church images in {root}.\n"
                "Download manually from "
                f"https://www.kaggle.com/datasets/{self.KAGGLE_SLUG} "
                f"and extract into {root}.\n"
                f"kagglehub error: {exc}"
            ) from exc

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        img = Image.open(self.filenames[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        info_dict = {
            "filename": os.path.basename(self.filenames[idx]),
            "idx": idx,
            "class_id": 0,
        }
        return img, info_dict
