"""
evaluate.py
-----------
Evaluate a trained model on the UNSEEN test split and report real metrics.

Reported metrics
================
Precision (P)  = TP / (TP + FP)
    Of all the boxes the model predicted, how many were actually weeds.
    Low P  -> many false alarms (would spray where there is no weed).

Recall (R)     = TP / (TP + FN)
    Of all the real weeds in the images, how many the model found.
    Low R  -> misses weeds (they survive and spread).

IoU (Intersection over Union)
    overlap_area(pred, gt) / union_area(pred, gt).  Ranges 0..1.
    A prediction counts as a True Positive only if its IoU with a ground-truth
    box of the same class is >= a threshold (e.g. 0.50).

AP (Average Precision)
    Area under the Precision-Recall curve for ONE class at ONE IoU threshold.
    Rewards a model that keeps precision high across all recall levels.

mAP@50       = mean AP over all classes at IoU threshold 0.50   (localisation is lenient)
mAP@50:95    = mean AP over all classes, averaged over IoU 0.50, 0.55, ... 0.95
               (the primary COCO metric - rewards tight, accurate boxes)

Note on P / R numbers: Ultralytics reports P and R at the confidence threshold
that maximises the F1 score on this split. mAP is threshold-independent (it
sweeps every confidence value), which is why mAP is the headline number.

Usage
-----
    python src/evaluate.py --weights results/runs/<run>/weights/best.pt
    python src/evaluate.py --weights results/runs/<run>/weights/best.pt --split test --imgsz 640
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

from utils import PROJECT_ROOT, load_yaml, pick_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="path to best.pt from training")
    p.add_argument("--data", default=str(PROJECT_ROOT / "configs" / "data.yaml"))
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold during evaluation")
    p.add_argument("--device", default=None)
    p.add_argument("--name", default="test_eval", help="output folder name under results/runs/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}\nTrain first: python src/train.py")

    data = load_yaml(args.data)
    names = data["names"]
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}

    model = YOLO(str(weights))
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        iou=args.iou,
        device=pick_device(args.device),
        project=str(PROJECT_ROOT / "results" / "runs"),
        name=args.name,
        exist_ok=True,
        plots=True,           # PR curve, confusion matrix, F1 curve
        verbose=True,
    )

    rd = metrics.results_dict
    overall = {
        "precision": float(rd.get("metrics/precision(B)", 0.0)),
        "recall": float(rd.get("metrics/recall(B)", 0.0)),
        "mAP50": float(rd.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(rd.get("metrics/mAP50-95(B)", 0.0)),
        "fitness": float(rd.get("fitness", 0.0)),
    }

    # per-class breakdown
    per_class = []
    box = metrics.box
    for i, c in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        per_class.append({
            "class_id": int(c),
            "class_name": str(names.get(int(c), c)),
            "precision": float(p),
            "recall": float(r),
            "mAP50": float(ap50),
            "mAP50_95": float(ap),
        })

    save_dir = Path(metrics.save_dir)
    report = {
        "weights": str(weights),
        "split": args.split,
        "imgsz": args.imgsz,
        "num_classes": len(names),
        "overall": overall,
        "per_class": per_class,
        "plots_dir": str(save_dir),
    }
    out_json = save_dir / f"metrics_{args.split}.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # also drop a copy of the key artifacts into results/ for the README.
    # (Ultralytics 8.3+ prefixes detection curves with "Box", older versions don't.)
    results_dir = PROJECT_ROOT / "results"
    wanted = ("confusion_matrix", "confusion_matrix_normalized",
              "PR_curve", "BoxPR_curve", "F1_curve", "BoxF1_curve",
              "R_curve", "BoxR_curve", "P_curve", "BoxP_curve")
    for stem in wanted:
        src = save_dir / f"{stem}.png"
        if src.exists():
            shutil.copy2(src, results_dir / f"{args.split}_{stem.replace('Box', '')}.png")
    shutil.copy2(out_json, results_dir / out_json.name)

    print("\n" + "=" * 60)
    print(f"RESULTS on '{args.split}' split  ({len(names)} classes)")
    print("-" * 60)
    print(f"  Precision   : {overall['precision']:.4f}")
    print(f"  Recall      : {overall['recall']:.4f}")
    print(f"  mAP@50      : {overall['mAP50']:.4f}")
    print(f"  mAP@50:95   : {overall['mAP50_95']:.4f}")
    print("-" * 60)
    print(f"{'class':22s} {'P':>7s} {'R':>7s} {'mAP50':>8s} {'mAP50-95':>9s}")
    for pc in per_class:
        print(f"{pc['class_name']:22s} {pc['precision']:7.3f} {pc['recall']:7.3f} "
              f"{pc['mAP50']:8.3f} {pc['mAP50_95']:9.3f}")
    print("=" * 60)
    print(f"\nSaved: {out_json}")
    print(f"Plots: {save_dir}  (also copied into results/)")


if __name__ == "__main__":
    main()
