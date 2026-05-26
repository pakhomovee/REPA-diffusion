import os
import re
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


class CompCarsDataset(Dataset):
    """
    Dataset class for the CompCars dataset sourced from Kaggle
    (renancostaalencar/compcars).

    Expected on-disk layout (web-nature sub-set):
        <root_dir>/
            data/
                image/
                    <make_id>/
                        <model_id>/
                            <year>/
                                *.jpg
                label/
                    <make_id>/
                        <model_id>/
                            <year>/
                                *.txt   (optional)

    Class labels are assigned at the *make × model* level, so every
    distinct (make_id, model_id) pair gets a unique integer class id.
    This gives fine-grained car model conditioning comparable to the
    Stanford Cars split used elsewhere in this repo.

    Args:
        root_dir (str): Directory containing (or receiving) the dataset.
        transform (callable, optional): Transform applied to each PIL Image.
    """

    KAGGLE_SLUG = "renancostaalencar/compcars"
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, root_dir: str = "../data/compcars", transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        image_root = self._locate_or_download()
        self.samples, self.class_to_idx, self.classes = self._build_index(image_root)
        if not self.samples:
            raise RuntimeError(
                f"No images found under {image_root}.\n"
                "Download from https://www.kaggle.com/datasets/renancostaalencar/compcars "
                f"and extract into {root_dir}."
            )

    # ------------------------------------------------------------------
    def _locate_or_download(self) -> str:
        root = Path(self.root_dir)
        root.mkdir(parents=True, exist_ok=True)

        # Preferred: data/image sub-tree
        candidate = root / "data" / "image"
        if candidate.is_dir():
            return str(candidate)

        # Flat fallback: root itself or any sub-dir containing images
        for dirpath, dirnames, fnames in os.walk(root):
            # Skip annotation/label directories
            if "label" in dirpath or "annotation" in dirpath:
                continue
            if any(Path(f).suffix.lower() in self._IMAGE_EXTS for f in fnames):
                # Walk up to the highest directory that looks like make/model layout
                return dirpath

        # Extract any .zip found
        zips = list(root.glob("*.zip"))
        if zips:
            print(f"Extracting {zips[0]} ...")
            with zipfile.ZipFile(zips[0], "r") as z:
                z.extractall(root)
            return self._locate_or_download()

        # kagglehub fallback
        try:
            import kagglehub  # type: ignore
            print(f"Downloading {self.KAGGLE_SLUG} via kagglehub ...")
            path = kagglehub.dataset_download(self.KAGGLE_SLUG)
            return path
        except Exception as exc:
            raise RuntimeError(
                f"Could not locate CompCars images in {root}.\n"
                "Download manually from "
                f"https://www.kaggle.com/datasets/{self.KAGGLE_SLUG} "
                f"and extract into {root}.\n"
                f"kagglehub error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    def _build_index(self, image_root: str):
        """
        Walk image_root and assign an integer class id to every
        (make_id, model_id) pair.  Returns:
            samples        : list of (abs_path, class_id)
            class_to_idx   : dict  {(make_id, model_id): class_id}
            classes        : list  of "(make_id)_(model_id)" strings
        """
        root = Path(image_root)
        # Collect all (make, model) pairs first for a stable class order
        class_set: set[tuple[str, str]] = set()
        all_images: list[tuple[Path, str, str]] = []

        for img_path in SORTFN(str(p) for p in root.rglob("*")
                               if p.is_file()
                               and p.suffix.lower() in self._IMAGE_EXTS):
            p = Path(img_path)
            parts = p.relative_to(root).parts
            if len(parts) >= 2:
                make_id, model_id = parts[0], parts[1]
            else:
                make_id, model_id = "unknown", "unknown"
            class_set.add((make_id, model_id))
            all_images.append((p, make_id, model_id))

        sorted_classes = sorted(class_set)
        class_to_idx = {cls: idx for idx, cls in enumerate(sorted_classes)}
        classes = [f"{m}_{mo}" for m, mo in sorted_classes]

        samples = [
            (str(p), class_to_idx[(make, model)])
            for p, make, model in all_images
        ]
        return samples, class_to_idx, classes

    # ------------------------------------------------------------------
    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, class_id = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        info_dict = {
            "filename": os.path.basename(img_path),
            "idx": idx,
            "class_id": class_id,
        }
        return img, info_dict
