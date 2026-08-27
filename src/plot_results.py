"""
plot_results.py
---------------
Make a clean training-curve figure from an Ultralytics run's results.csv
(loss + mAP over epochs). Ultralytics already saves results.png; this is a
simpler, README-friendly version.

Usage:
    python src/plot_results.py --run results/runs/<run_name>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="path to results/runs/<run_name>")
    p.add_argument("--out", default=None, help="output PNG (default: <run>/training_curves.png)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run = Path(args.run)
    csv = run / "results.csv"
    if not csv.exists():
        raise SystemExit(f"results.csv not found in {run}")
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    epochs = df["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- losses ---
    for col, label in [
        ("train/box_loss", "train box"),
        ("val/box_loss", "val box"),
        ("train/cls_loss", "train cls"),
        ("val/cls_loss", "val cls"),
    ]:
        if col in df:
            axes[0].plot(epochs, df[col], label=label)
    axes[0].set_title("Training / validation loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- metrics ---
    for col, label in [
        ("metrics/precision(B)", "precision"),
        ("metrics/recall(B)", "recall"),
        ("metrics/mAP50(B)", "mAP@50"),
        ("metrics/mAP50-95(B)", "mAP@50:95"),
    ]:
        if col in df:
            axes[1].plot(epochs, df[col], label=label)
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = Path(args.out) if args.out else run / "training_curves.png"
    fig.savefig(out, dpi=130)
    # also copy to results/ root for the README
    (PROJECT_ROOT / "results").mkdir(exist_ok=True)
    fig.savefig(PROJECT_ROOT / "results" / "training_curves.png", dpi=130)
    print(f"saved: {out}")
    print(f"saved: {PROJECT_ROOT / 'results' / 'training_curves.png'}")


if __name__ == "__main__":
    main()
