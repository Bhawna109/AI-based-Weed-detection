"""
verify_dataset.py
-----------------
Sanity-check the prepared dataset BEFORE training.

Checks per split (train / val / test):
  * every image opens with OpenCV (not corrupt / truncated)
  * every image has a label file (an empty label file = pure background image, allowed)
  * every label line has 5 fields: <class> <cx> <cy> <w> <h>
  * class index is an integer inside [0, nc-1]
  * cx, cy, w, h are floats inside [0, 1]
  * reports instance counts per class and per split

It also saves a few annotated sample images to results/verify_samples/ so you can
eyeball that boxes line up with weeds.

Usage:
    python src/verify_dataset.py
    python src/verify_dataset.py --data configs/data.yaml --samples 12
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from utils import IMAGE_EXTS, PROJECT_ROOT, list_images, load_yaml

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(PROJECT_ROOT / "configs" / "data.yaml"))
    p.add_argument("--samples", type=int, default=9, help="annotated sample images to save per split")
    p.add_argument("--out", default=str(PROJECT_ROOT / "results" / "verify_samples"))
    return p.parse_args()


def label_path_for(img: Path) -> Path:
    # .../images/<split>/x.jpg  ->  .../labels/<split>/x.txt
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def check_label_file(lbl: Path, nc: int) -> tuple[list[int], list[str]]:
    """Return (class_ids_found, error_messages)."""
    errors: list[str] = []
    classes: list[int] = []
    if not lbl.exists():
        return classes, [f"missing label file: {lbl.name}"]
    for i, line in enumerate(lbl.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{lbl.name}:{i} expected 5 fields, got {len(parts)}")
            continue
        try:
            cid = int(parts[0])
            cx, cy, w, h = (float(x) for x in parts[1:])
        except ValueError:
            errors.append(f"{lbl.name}:{i} non-numeric value")
            continue
        if not (0 <= cid < nc):
            errors.append(f"{lbl.name}:{i} class id {cid} out of range [0,{nc - 1}]")
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
            errors.append(f"{lbl.name}:{i} box not normalised to [0,1]: {parts[1:]}")
        if w <= 0 or h <= 0:
            errors.append(f"{lbl.name}:{i} non-positive width/height")
        classes.append(cid)
    return classes, errors


def draw_sample(img_path: Path, lbl: Path, names: dict, out_path: Path) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        return
    H, W = img.shape[:2]
    for line in lbl.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cid, cx, cy, w, h = int(parts[0]), *[float(x) for x in parts[1:]]
        x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
        x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, str(names.get(cid, cid)), (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main() -> None:
    args = parse_args()
    data = load_yaml(args.data)
    root = Path(data["path"])
    names = data["names"]
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    nc = len(names)
    print(f"Dataset root: {root}")
    print(f"Classes ({nc}): {names}\n")

    total_errors = 0
    grand_counter: Counter[int] = Counter()

    for split in SPLITS:
        img_dir = root / "images" / split
        if not img_dir.is_dir():
            print(f"[{split}] folder missing: {img_dir}  (skipping)")
            continue
        images = list_images(img_dir)
        counter: Counter[int] = Counter()
        n_bg = 0
        n_corrupt = 0
        errors: list[str] = []

        for img in images:
            arr = cv2.imread(str(img))
            if arr is None:
                n_corrupt += 1
                errors.append(f"corrupt / unreadable image: {img.name}")
                continue
            lbl = label_path_for(img)
            classes, errs = check_label_file(lbl, nc)
            errors.extend(errs)
            if not classes and not errs:
                n_bg += 1
            counter.update(classes)

        grand_counter.update(counter)
        total_errors += len(errors)
        n_inst = sum(counter.values())
        print(f"[{split}] {len(images)} images | {n_inst} boxes | {n_bg} background-only | "
              f"{n_corrupt} corrupt | {len(errors)} problems")
        for e in errors[:15]:
            print(f"    - {e}")
        if len(errors) > 15:
            print(f"    ... and {len(errors) - 15} more")

        # save annotated samples
        labelled = [i for i in images if label_path_for(i).exists()
                    and label_path_for(i).read_text(encoding='utf-8').strip()]
        random.seed(0)
        for img in random.sample(labelled, min(args.samples, len(labelled))):
            draw_sample(img, label_path_for(img), names,
                        Path(args.out) / split / img.name)

    print("\nInstances per class (all splits):")
    for cid in sorted(names):
        print(f"  {cid:3d} {str(names[cid]):20s} {grand_counter.get(cid, 0)}")

    print(f"\nAnnotated samples saved under: {args.out}")
    if total_errors:
        print(f"\n!!  {total_errors} problem(s) found - fix these before training.")
        raise SystemExit(1)
    print("\nOK - dataset looks valid. You can train now.")


if __name__ == "__main__":
    main()
