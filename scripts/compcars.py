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


class CompCarsDataset(Dataset):
    """
    Dataset class for the CompCars dataset sourced from Kaggle
    (renancostaalencar/compcars).

    The Kaggle archive has the following layout:
        <root_dir>/
            image/                   ← top-level images folder
                <make_id>/
                    <model_id>/
                        <year>/
                            *.jpg
            label/                   ← optional annotation files
            misc/
            train_test_split/

    Class labels are assigned at the make_id × model_id level.
    Every distinct (make_id, model_id) pair gets a unique integer class id,
    giving fine-grained car-model conditioning.

    Args:
        root_dir (str): Directory containing (or receiving) the dataset.
                        This should be the parent of the 'image/' folder.
        transform (callable, optional): Transform applied to each PIL Image.
    """

    KAGGLE_SLUG = "renancostaalencar/compcars"
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, root_dir: str = "../data/compcars", transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        image_root = self._locate_image_root()
        self.samples, self.class_to_idx, self.classes = self._build_index(image_root)
        if not self.samples:
            raise RuntimeError(
                f"No images found under {image_root}.\n"
                "Download from https://www.kaggle.com/datasets/renancostaalencar/compcars "
                f"and extract into {root_dir}."
            )

    # ------------------------------------------------------------------
    def _locate_image_root(self) -> str:
        """
        Find the directory whose immediate children are numeric make_id
        folders (e.g. 1/, 10/, 100/, ...).

        The Kaggle archive extracts to:
            <root_dir>/image/<make_id>/<model_id>/<year>/*.jpg

        We locate the 'image/' directory by looking for a folder whose
        children are all (or mostly) numeric directory names.
        """
        root = Path(self.root_dir)
        root.mkdir(parents=True, exist_ok=True)

        # Helper: does a directory look like the make-level root?
        # We consider it a match if ≥80% of its sub-entries are numeric dirs.
        def is_make_root(p: Path) -> bool:
            try:
                children = [c for c in p.iterdir() if c.is_dir()]
                if not children:
                    return False
                numeric = sum(1 for c in children if c.name.isdigit())
                return numeric / len(children) >= 0.8
            except PermissionError:
                return False

        # 1) Common known paths
        for candidate in [
            root / "image",
            root / "data" / "image",
            root,
        ]:
            if candidate.is_dir() and is_make_root(candidate):
                print(f"CompCars image root: {candidate}")
                return str(candidate)

        # 2) BFS over root to find the make-level directory
        for dirpath_str, dirnames, _ in os.walk(str(root)):
            dirpath = Path(dirpath_str)
            # Skip annotation / label directories
            if any(part in {"label", "annotation", "misc", "train_test_split"}
                   for part in dirpath.parts):
                dirnames[:] = []
                continue
            if is_make_root(dirpath):
                print(f"CompCars image root (found by walk): {dirpath}")
                return str(dirpath)

        # 3) Extract any .zip
        zips = list(root.glob("*.zip"))
        if zips:
            print(f"Extracting {zips[0]} ...")
            with zipfile.ZipFile(zips[0], "r") as z:
                z.extractall(root)
            return self._locate_image_root()

        # 4) kagglehub fallback
        try:
            import kagglehub  # type: ignore
            print(f"Downloading {self.KAGGLE_SLUG} via kagglehub ...")
            dl_path = kagglehub.dataset_download(self.KAGGLE_SLUG)
            self.root_dir = dl_path
            return self._locate_image_root()
        except Exception as exc:
            raise RuntimeError(
                f"Could not locate CompCars 'image/' directory in {root}.\n"
                "Download manually from "
                f"https://www.kaggle.com/datasets/{self.KAGGLE_SLUG} "
                f"and extract into {root}.\n"
                f"kagglehub error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    def _build_index(self, image_root: str):
        """
        Walk image_root (the make-level directory) and collect
        (image_path, make_id, model_id) triples.

        Expected structure:
            image_root/
                <make_id>/        ← numeric string, e.g. "1"
                    <model_id>/   ← numeric string, e.g. "10"
                        <year>/   ← e.g. "2012"
                            *.jpg

        Returns:
            samples      : list of (abs_path_str, class_id)
            class_to_idx : dict {(make_id, model_id): class_id}
            classes      : list of "make{make_id}_model{model_id}" strings
        """
        root = Path(image_root)

        class_set: set[tuple[str, str]] = set()
        raw: list[tuple[Path, str, str]] = []  # (path, make_id, model_id)

        for img_path in SORTFN(
            str(p) for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in self._IMAGE_EXTS
        ):
            p = Path(img_path)
            try:
                rel_parts = p.relative_to(root).parts
                # rel_parts = (make_id, model_id, year, filename)
                #              or (make_id, model_id, filename) if no year level
                if len(rel_parts) >= 2:
                    make_id = rel_parts[0]
                    model_id = rel_parts[1]
                else:
                    make_id = rel_parts[0] if rel_parts else "unknown"
                    model_id = "unknown"
            except ValueError:
                make_id, model_id = "unknown", "unknown"

            class_set.add((make_id, model_id))
            raw.append((p, make_id, model_id))

        sorted_classes = sorted(class_set)
        class_to_idx = {cls: idx for idx, cls in enumerate(sorted_classes)}
        classes = [f"make{m}_model{mo}" for m, mo in sorted_classes]

        samples = [
            (str(p), class_to_idx[(make, model)])
            for p, make, model in raw
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
