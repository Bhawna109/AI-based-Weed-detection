"""
predict.py
----------
Run the trained weed detector on new field images and save annotated results
(bounding boxes + class label + confidence score).

Works on:
  * a single image        --source path/to/image.jpg
  * a folder of images     --source path/to/folder
  * (also accepts a video file or a glob, since it delegates to Ultralytics)

Outputs:
  results/predictions/<name>/         annotated images (boxes drawn by Ultralytics)
  results/predictions/<name>/labels/  per-image YOLO txt (with --save-txt)
  results/predictions/<name>/detections.csv   one row per detected box

Usage
-----
    python src/predict.py --weights results/runs/<run>/weights/best.pt --source my_field.jpg
    python src/predict.py --weights results/runs/<run>/weights/best.pt --source test_images/ --conf 0.35
    python src/predict.py --weights results/runs/<run>/weights/best.pt --source test_images/ --save-txt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

from utils import PROJECT_ROOT, pick_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="path to best.pt")
    p.add_argument("--source", required=True, help="image file, folder of images, or video")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold (0-1)")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--name", default="predict", help="output subfolder under results/predictions/")
    p.add_argument("--save-txt", action="store_true", help="also save YOLO-format label files")
    p.add_argument("--max-det", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}\nTrain first: python src/train.py")
    if not Path(args.source).exists():
        raise SystemExit(f"source not found: {args.source}")

    model = YOLO(str(weights))
    out_root = PROJECT_ROOT / "results" / "predictions"

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=pick_device(args.device),
        max_det=args.max_det,
        save=True,               # write annotated images
        save_txt=args.save_txt,
        save_conf=args.save_txt,
        project=str(out_root),
        name=args.name,
        exist_ok=True,
        verbose=False,
    )

    save_dir = Path(results[0].save_dir) if results else out_root / args.name
    csv_path = save_dir / "detections.csv"
    n_images = len(results)
    n_boxes = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
        for r in results:
            names = r.names
            if r.boxes is None or len(r.boxes) == 0:
                w.writerow([Path(r.path).name, "", "", "", "", "", "", ""])
                continue
            for b in r.boxes:
                cid = int(b.cls)
                conf = float(b.conf)
                x1, y1, x2, y2 = (round(v, 1) for v in b.xyxy[0].tolist())
                w.writerow([Path(r.path).name, cid, names.get(cid, cid), round(conf, 4), x1, y1, x2, y2])
                n_boxes += 1

    print("=" * 55)
    print(f"Images processed : {n_images}")
    print(f"Weeds detected   : {n_boxes}  (conf >= {args.conf})")
    print(f"Annotated images : {save_dir}")
    print(f"Detection CSV    : {csv_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
