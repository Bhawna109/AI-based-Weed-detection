"""
train.py
--------
Fine-tune a lightweight pretrained YOLO model on the weed-detection dataset
(transfer learning - we start from COCO-pretrained weights, not from scratch).

Model: Ultralytics **YOLO11n** (`yolo11n.pt`) - the "nano" size of the YOLO11
family. ~2.6M parameters, the smallest/fastest YOLO11 detector, which makes it a
good fit for a beginner project and for eventual deployment on a tractor / drone
/ edge device. It is an **anchor-free** detector (see docs/ARCHITECTURE.md).

--------------------------------------------------------------------------
Training configuration (what each knob means)
--------------------------------------------------------------------------
epochs      : how many full passes over the training set. 100 with early
              stopping (patience) is a safe default for a few-thousand-image set.
batch       : images per gradient step. 16 suits ~8 GB VRAM. Use -1 to let
              Ultralytics auto-pick based on free memory; drop to 8/4 on OOM.
imgsz       : images are letterboxed to imgsz x imgsz (640). Bigger = better on
              small weeds but slower and more memory.
optimizer   : 'auto' (default). With 'auto', the INSTALLED Ultralytics version
              inspects the dataset size and picks the optimizer + lr0 + momentum
              itself. This script PRINTS and SAVES the optimizer actually used
              (see results/<name>/train_summary.json) - we never assume it.
lr0         : initial learning rate. Left unset so Ultralytics/'auto' chooses it.
              Pass --lr0 to override.
device      : 'auto' -> CUDA GPU if available, else Apple MPS, else CPU.
augmentation: Ultralytics applies these by default during training (values are
              from the model's hyp config and are echoed into train_summary.json):
                hsv_h/s/v  - random hue / saturation / brightness jitter
                translate  - random shift
                scale      - random zoom
                fliplr     - 50% horizontal flip (weeds have no left/right bias)
                mosaic     - stitch 4 images into 1 (strong context augmentation),
                             automatically turned off for the last `close_mosaic`
                             epochs so training finishes on realistic images
              flipud / rotation / mixup are OFF by default - top-down field
              images can be flipped vertically too, so we enable a little
              (`flipud=0.5`, `degrees=10`). Tune to taste.

Usage
-----
    python src/train.py
    python src/train.py --epochs 50 --batch 8 --imgsz 640
    python src/train.py --model yolo11s.pt --device 0
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

from utils import PROJECT_ROOT, pick_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(PROJECT_ROOT / "configs" / "data.yaml"))
    p.add_argument("--model", default="yolo11n.pt", help="pretrained checkpoint to fine-tune")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16, help="-1 = auto")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--optimizer", default="auto", choices=["auto", "SGD", "Adam", "AdamW", "RMSProp", "NAdam", "RAdam"])
    p.add_argument("--lr0", type=float, default=None, help="initial LR (default: chosen by Ultralytics)")
    p.add_argument("--patience", type=int, default=25, help="early-stop after N epochs without improvement")
    p.add_argument("--device", default=None, help="'cpu', '0', '0,1' ... (default: auto)")
    p.add_argument("--workers", type=int, default=4, help="dataloader workers (keep low on Windows)")
    p.add_argument("--cache", default=None, choices=["ram", "disk"],
                   help="cache images for faster epochs (ram needs enough memory)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None, help="run name under results/ (default: timestamped)")
    p.add_argument("--project", default=None,
                   help="output dir for the run (default: results/runs). Point at a "
                        "Google Drive path on Colab so checkpoints survive a disconnect.")
    p.add_argument("--resume", action="store_true", help="resume the last interrupted run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    run_name = args.name or f"weed_yolo11n_{datetime.now():%Y%m%d_%H%M%S}"
    project_dir = Path(args.project) if args.project else PROJECT_ROOT / "results" / "runs"

    print(f"Model            : {args.model} (transfer learning from pretrained weights)")
    print(f"Data             : {args.data}")
    print(f"Device           : {device}")
    print(f"Epochs / batch   : {args.epochs} / {args.batch}")
    print(f"Image size       : {args.imgsz}")
    print(f"Optimizer (req.) : {args.optimizer}  (actual choice printed after training)\n")

    model = YOLO(args.model)

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        optimizer=args.optimizer,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        resume=args.resume,
        # light extra augmentation suited to top-down field imagery
        flipud=0.5,
        degrees=10.0,
        plots=True,          # save PR curve, confusion matrix, training curves
        val=True,
    )
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.cache is not None:
        train_kwargs["cache"] = args.cache

    results = model.train(**train_kwargs)

    # ------------------------------------------------------------------
    # Record the optimizer / hyper-parameters ACTUALLY used (no guessing).
    # ------------------------------------------------------------------
    trainer = model.trainer
    save_dir = Path(trainer.save_dir)
    optimizer_cls = type(trainer.optimizer).__name__
    try:
        lrs = sorted({round(g["lr"], 8) for g in trainer.optimizer.param_groups})
    except Exception:
        lrs = None

    summary = {
        "model": args.model,
        "data": args.data,
        "device": str(device),
        "epochs_requested": args.epochs,
        "epochs_completed": int(getattr(trainer, "epoch", args.epochs - 1)) + 1,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "optimizer_requested": args.optimizer,
        "optimizer_actual": optimizer_cls,
        "initial_lrs_per_param_group": lrs,
        "resolved_hyperparameters": {k: (str(v) if isinstance(v, Path) else v)
                                     for k, v in vars(trainer.args).items()},
        "best_weights": str(save_dir / "weights" / "best.pt"),
        "last_weights": str(save_dir / "weights" / "last.pt"),
        "save_dir": str(save_dir),
    }
    out = save_dir / "train_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Optimizer actually used : {optimizer_cls}")
    print(f"Initial LR(s)           : {lrs}")
    print(f"Best weights            : {summary['best_weights']}")
    print(f"Training curves / plots : {save_dir}")
    print(f"Summary JSON            : {out}")
    print("=" * 60)
    print("\nNext:  python src/evaluate.py --weights", summary["best_weights"])


if __name__ == "__main__":
    main()
