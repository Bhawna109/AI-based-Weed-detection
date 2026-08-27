"""
prepare_dataset.py
------------------
Turn a raw weed-detection dataset into the folder layout Ultralytics YOLO expects
and generate a ready-to-use configs/data.yaml.

Target layout produced under  dataset/ :

    dataset/
    +- images/
    |  +- train/  val/  test/
    +- labels/
    |  +- train/  val/  test/
    +- classes.txt

Supported inputs
================
1. Already-YOLO dataset (recommended, e.g. CottonWeedDet12, Roboflow YOLOv8 export)
   A folder that contains images and matching `.txt` label files. They can be:
     - side by side in the same folder, or
     - split into `images/` and `labels/` subfolders.
   Use:  --source <folder> --classes <classes.txt>

2. Pascal-VOC dataset (XML annotations)
   Use:  --source <folder> --voc --classes <classes.txt>
   `classes.txt` must list every class name, one per line, in the index order
   you want. VOC boxes are converted to normalised YOLO `cls cx cy w h`.

If the raw data is ALREADY split into train/valid/test (like a Roboflow export),
just point `train.py` at that export's own `data.yaml` and skip this script.

Examples
========
    python src/prepare_dataset.py --source raw/CottonWeedDet12 --classes raw/CottonWeedDet12/classes.txt
    python src/prepare_dataset.py --source raw/voc_weeds --voc --classes raw/voc_weeds/classes.txt --train 0.7 --val 0.15 --test 0.15
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

from utils import IMAGE_EXTS, PROJECT_ROOT, save_yaml

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="folder with the raw dataset")
    p.add_argument("--classes", required=True, help="text file: one class name per line, in index order")
    p.add_argument("--dest", default=str(PROJECT_ROOT / "dataset"), help="output dataset folder")
    p.add_argument("--voc", action="store_true", help="input annotations are Pascal-VOC XML")
    p.add_argument("--train", type=float, default=0.80, help="train split fraction")
    p.add_argument("--val", type=float, default=0.10, help="val split fraction")
    p.add_argument("--test", type=float, default=0.10, help="test split fraction")
    p.add_argument("--seed", type=int, default=0, help="random seed for the split")
    p.add_argument("--copy", action="store_true", default=True, help="copy files (default; symlinks are unreliable on Windows)")
    return p.parse_args()


def read_classes(path: str | Path) -> list[str]:
    names = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not names:
        raise SystemExit(f"No class names found in {path}")
    return names


def find_image_for(label: Path, image_pool: dict[str, Path]) -> Path | None:
    """Match a label file to an image by stem."""
    return image_pool.get(label.stem)


def voc_to_yolo(xml_path: Path, class_to_id: dict[str, int]) -> list[str]:
    """Convert one VOC XML file to a list of YOLO label lines."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = float(size.findtext("width"))
    h = float(size.findtext("height"))
    lines: list[str] = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in class_to_id:
            continue
        b = obj.find("bndbox")
        xmin, ymin = float(b.findtext("xmin")), float(b.findtext("ymin"))
        xmax, ymax = float(b.findtext("xmax")), float(b.findtext("ymax"))
        cx = ((xmin + xmax) / 2) / w
        cy = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"{class_to_id[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def main() -> None:
    args = parse_args()
    fractions = (args.train, args.val, args.test)
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise SystemExit(f"--train/--val/--test must sum to 1.0 (got {sum(fractions)})")

    src = Path(args.source)
    if not src.is_dir():
        raise SystemExit(f"--source folder not found: {src}")

    dest = Path(args.dest)
    class_names = read_classes(args.classes)
    class_to_id = {n: i for i, n in enumerate(class_names)}

    # --- collect images -------------------------------------------------
    image_pool: dict[str, Path] = {}
    for img in src.rglob("*"):
        if img.suffix.lower() in IMAGE_EXTS:
            image_pool[img.stem] = img
    if not image_pool:
        raise SystemExit(f"No images found under {src}")

    # --- collect annotations ------------------------------------------
    ann_ext = ".xml" if args.voc else ".txt"
    ann_files = [p for p in src.rglob(f"*{ann_ext}") if p.name.lower() != "classes.txt"]
    if not ann_files:
        raise SystemExit(f"No '{ann_ext}' annotation files found under {src}")

    # pair (image, label_lines)
    pairs: list[tuple[Path, list[str]]] = []
    skipped = 0
    for ann in ann_files:
        img = find_image_for(ann, image_pool)
        if img is None:
            skipped += 1
            continue
        if args.voc:
            lines = voc_to_yolo(ann, class_to_id)
        else:
            lines = [ln.strip() for ln in ann.read_text(encoding="utf-8").splitlines() if ln.strip()]
        pairs.append((img, lines))

    if not pairs:
        raise SystemExit("Could not pair any image with an annotation (check file names / stems).")
    print(f"Paired {len(pairs)} image/label files ({skipped} annotations had no matching image).")

    # --- split -------------------------------------------------------
    random.seed(args.seed)
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    buckets = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    # --- write -------------------------------------------------------
    for split in SPLITS:
        (dest / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, items in buckets.items():
        for img, lines in tqdm(items, desc=f"writing {split}", unit="img"):
            out_img = dest / "images" / split / img.name
            out_lbl = dest / "labels" / split / (img.stem + ".txt")
            if args.copy:
                shutil.copy2(img, out_img)
            else:
                shutil.copy2(img, out_img)
            out_lbl.write_text("\n".join(lines), encoding="utf-8")

    # classes.txt + data.yaml
    (dest / "classes.txt").write_text("\n".join(class_names), encoding="utf-8")
    data_yaml = {
        "path": str(dest.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(class_names)},
    }
    out_yaml = PROJECT_ROOT / "configs" / "data.yaml"
    save_yaml(data_yaml, out_yaml)

    print("\nDone.")
    print(f"  classes ({len(class_names)}): {class_names}")
    for split in SPLITS:
        print(f"  {split:5s}: {len(buckets[split]):5d} images")
    print(f"  data.yaml written to: {out_yaml}")
    print("\nNext:  python src/verify_dataset.py")


if __name__ == "__main__":
    main()
