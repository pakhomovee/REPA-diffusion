#!/usr/bin/env python3
"""CelebA-labeled wrapper around generate_labeled_imagenette_grid.py."""
import importlib.util
from pathlib import Path


ATTRS = ["Male", "Smiling", "Young", "Attractive"]


def label_for_class(class_id: int) -> str:
    bits = [(class_id >> idx) & 1 for idx in range(len(ATTRS))]
    positives = [name for bit, name in zip(bits, ATTRS) if bit]
    if positives:
        return "+".join(positives)
    return "not_" + "+not_".join(ATTRS)


def main():
    script_path = Path(__file__).with_name("generate_labeled_imagenette_grid.py")
    spec = importlib.util.spec_from_file_location("imagenette_grid", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.IMAGENETTE_CLASSES = [
        (f"celeba-{class_id:02d}", label_for_class(class_id))
        for class_id in range(16)
    ]
    module.main()


if __name__ == "__main__":
    main()
