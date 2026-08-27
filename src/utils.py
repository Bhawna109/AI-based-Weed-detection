"""Small shared helpers used by the training / evaluation / prediction scripts."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

# Project root = one level above this file's folder (src/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Image extensions we treat as valid input images.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def set_seed(seed: int = 0) -> None:
    """Seed Python and NumPy RNGs. Ultralytics seeds torch itself via ``seed=``."""
    random.seed(seed)
    np.random.seed(seed)


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def pick_device(requested: str | None) -> str | int:
    """Return a device string/int for Ultralytics.

    ``requested`` may be ``None`` (auto), ``"cpu"``, ``"0"``, ``"0,1"`` ...
    """
    if requested:
        return int(requested) if requested.isdigit() else requested
    try:
        import torch

        if torch.cuda.is_available():
            return 0
        # Apple Silicon
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
