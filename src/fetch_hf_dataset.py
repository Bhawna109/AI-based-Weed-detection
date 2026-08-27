"""
fetch_hf_dataset.py
-------------------
Download a Hugging Face **object-detection** dataset (COCO-style `objects` with
`bbox` + `category`) and convert it into the YOLO folder layout this project
uses, then write `configs/data.yaml`.

Default dataset: **Francesco/weed-crop-aerial**
  - ~1,176 aerial crop/weed images (640x640), ~165 MB, no login required
  - already split into train / validation / test
  - Creative Commons license (see the dataset card / original Roboflow page)
  This is a small, fast dataset to get the whole pipeline working end-to-end.
  For the larger CottonWeedDet12, see dataset/README.md.

Usage:
    python src/fetch_hf_dataset.py
    python src/fetch_hf_dataset.py --dataset Francesco/weed-crop-aerial
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from utils import PROJECT_ROOT, save_yaml

# HF split name -> our folder name
SPLIT_MAP = {"train": "train", "validation": "val", "valid": "val", "val": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="Francesco/weed-crop-aerial", help="HF dataset id")
    p.add_argument("--dest", default=str(PROJECT_ROOT / "dataset"), help="output dataset folder")
    p.add_argument("--limit", type=int, default=0, help="max images per split (0 = all)")
    p.add_argument("--keep-all-classes", action="store_true",
                   help="keep the Roboflow super-category at index 0 (dropped by default)")
    return p.parse_args()


def coco_to_yolo(bbox, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """COCO [x, y, w, h] (absolute, top-left) -> YOLO [cx, cy, w, h] (normalised)."""
    x, y, w, h = bbox
    return (
        (x + w / 2) / img_w,
        (y + h / 2) / img_h,
        w / img_w,
        h / img_h,
    )


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install the datasets library first:  pip install datasets")

    print(f"Downloading '{args.dataset}' from Hugging Face (first run caches ~hundreds of MB)...")
    ds = load_dataset(args.dataset)

    # class names from the ClassLabel feature (schema: objects -> {'category': List(ClassLabel)})
    first_split = ds[list(ds.keys())[0]]
    raw_names = list(first_split.features["objects"]["category"].feature.names)

    # Roboflow -> HF exports prepend a "super-category" at index 0 named after the
    # dataset itself (e.g. 'weed-crop-aerial'); it holds no real boxes. Drop it and
    # shift every category id down by 1, unless --keep-all-classes.
    slug = args.dataset.split("/")[-1]
    drop_placeholder = (not args.keep_all_classes) and len(raw_names) > 1 and raw_names[0] in (slug, slug.replace("-", " "))
    if drop_placeholder:
        class_names = raw_names[1:]
        remap = lambda c: c - 1        # noqa: E731
        keep = lambda c: c >= 1        # noqa: E731
        print(f"Dropped placeholder class '{raw_names[0]}'.")
    else:
        class_names = raw_names
        remap = lambda c: c            # noqa: E731
        keep = lambda c: True          # noqa: E731
    print(f"Classes ({len(class_names)}): {class_names}")

    dest = Path(args.dest)
    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            (dest / sub / split).mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for hf_split in ds.keys():
        out_split = SPLIT_MAP.get(hf_split.lower())
        if out_split is None:
            print(f"  (skipping unknown split '{hf_split}')")
            continue
        rows = ds[hf_split]
        n = len(rows) if args.limit == 0 else min(args.limit, len(rows))
        for i in tqdm(range(n), desc=f"{hf_split} -> {out_split}", unit="img"):
            row = rows[i]
            img = row["image"].convert("RGB")
            w, h = img.size
            stem = f"{out_split}_{row.get('image_id', i):06d}"
            img.save(dest / "images" / out_split / f"{stem}.jpg", quality=95)

            lines = []
            objs = row["objects"]
            for bbox, cat in zip(objs["bbox"], objs["category"]):
                if not keep(int(cat)):
                    continue
                cx, cy, bw, bh = coco_to_yolo(bbox, w, h)
                # clip tiny numerical overflow
                cx, cy, bw, bh = (min(max(v, 0.0), 1.0) for v in (cx, cy, bw, bh))
                if bw <= 0 or bh <= 0:
                    continue
                lines.append(f"{remap(int(cat))} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (dest / "labels" / out_split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        counts[out_split] = counts.get(out_split, 0) + n

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
    for split, c in counts.items():
        print(f"  {split:5s}: {c} images")
    print(f"  data.yaml -> {out_yaml}")
    print("\nNext:  python src/verify_dataset.py")


if __name__ == "__main__":
    main()
