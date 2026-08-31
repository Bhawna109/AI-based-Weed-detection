# AI-Based Weed Detection in Agricultural Fields

Detect and localise **weeds** in agricultural field images with **bounding
boxes**, using a lightweight **YOLO11** object-detection model fine-tuned with
transfer learning.

![sample prediction](results/samples/test_000002.jpg)

---

## Problem statement

Weeds compete with crops for light, water, and nutrients and can cause large
yield losses. The traditional response — spraying herbicide uniformly across the
whole field — is expensive, wasteful, and environmentally harmful, because most
of the field is crop, not weed.

**Goal:** given a photo of a field, automatically find *where* the weeds are
(bounding boxes + confidence), so action can be targeted only where needed.

## Real-world farming use case

- **Precision / spot spraying:** a camera on a sprayer boom or drone detects
  weeds in real time; only the nozzles over a weed fire. Herbicide use can drop
  dramatically.
- **Field scouting / mapping:** a drone flies the field and produces a weed
  density map so the agronomist knows where to intervene.
- **Robotic weeding:** a ground robot uses the boxes to aim a mechanical hoe or
  laser.
- **Monitoring over time:** repeat surveys track whether weed pressure is
  growing and whether treatment worked.

## Why object detection (not classification or segmentation)

| Task | Output | Fit for this problem |
|---|---|---|
| Classification | "this image contains a weed" | Not enough — we need *where* to act |
| **Object detection** | **box + class + confidence per weed** | **Right level:** enough to aim a nozzle/hoe, cheap to annotate, fast to run |
| Segmentation | per-pixel mask | More detail than needed for spraying; slower, costlier to label |

Bounding boxes give the location and count of weeds with far less annotation
effort and compute than pixel masks — a good match for real-time use on a
tractor or drone.

## Why YOLO

- **Single-stage & real-time:** one forward pass → all detections. Fast enough
  for a moving sprayer / drone, even on modest hardware.
- **Small footprint:** the `nano` model is ~2.6M parameters — deployable on edge
  devices.
- **Mature tooling (Ultralytics):** training, validation, metrics, plots, and
  export (ONNX/TensorRT) in one well-documented package — good for learning.
- **Strong accuracy/speed trade-off** for multi-object scenes like a weedy field.

## Model used

**Ultralytics YOLO11n** (`yolo11n.pt`), COCO-pretrained, fine-tuned on the weed
dataset via **transfer learning** (we do **not** train from scratch).

- Family: YOLO11 (Ultralytics, 2024)
- Size: **n / nano** — smallest & fastest
- Design: **anchor-free**, decoupled detection head, DFL box regression
- Swap to `yolo11s.pt` / `yolo11m.pt` with `--model` if you need more accuracy.

## Architecture

Full write-up: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**. Summary:

```
image 640×640 ─▶ Backbone (Conv, C3k2, SPPF, C2PSA)      → features @ stride 8/16/32
              ─▶ Neck (PAN-FPN: Upsample + Concat + C3k2) → fused multi-scale features
              ─▶ Head (anchor-free, 3 scales):
                    • box:  4 edge distances via Distribution Focal Loss (DFL)
                    • class: independent sigmoid per weed class
                    • confidence: the class score itself (no separate objectness)
              ─▶ NMS → final weed boxes
```

- **Input:** letterboxed RGB, 0–1 normalised; augmented during training.
- **Backbone:** CSP-style conv net; `C3k2` feature blocks, `SPPF` for
  multi-scale receptive field, `C2PSA` lightweight attention.
- **Neck:** top-down + bottom-up feature pyramid fusing fine detail with context.
- **Head:** anchor-free — predicts box edges from grid points, not from anchor
  boxes. Task-Aligned Assigner for label assignment; CIoU + DFL + BCE losses.

## Dataset

**CottonWeedDet12** — 5,648 real cotton-field images, 9,370 bounding boxes, 12
weed species, bounding-box annotations, **CC BY-NC 4.0** (non-commercial +
attribution), **~29 GB**.
Download links, the 12 class names, and **smaller/faster alternatives** (a
Roboflow YOLO export, CottonWeedDet3, or Colab): **[dataset/README.md](dataset/README.md)**.
(The dataset itself is **not** committed — license + size.)

Split: **80% train / 10% val / 10% test** via `src/prepare_dataset.py`
(seeded, reproducible). The **test split is never seen during training** and is
used only in `src/evaluate.py`.

Expected layout:

```
dataset/
├── images/{train,val,test}/
├── labels/{train,val,test}/     # <class> <cx> <cy> <w> <h>  (normalised 0–1)
└── classes.txt
configs/data.yaml                # generated: absolute path + class names
```

## Training process

Script: **[src/train.py](src/train.py)** — see its docstring for a line-by-line
explanation of every setting.

| Setting | Default | Meaning |
|---|---|---|
| model | `yolo11n.pt` | pretrained checkpoint we fine-tune |
| epochs | 100 | full passes over train set (early-stops via `patience=25`) |
| batch | 16 | images per gradient step (`-1` = auto; lower on OOM) |
| imgsz | 640 | images letterboxed to 640×640 |
| optimizer | `auto` | **Ultralytics picks the optimizer, lr0 & momentum from dataset size.** The script records the one **actually used** in `train_summary.json` — we never assume it. |
| lr0 | (auto) | initial learning rate, chosen by Ultralytics unless `--lr0` given |
| device | auto | CUDA GPU → Apple MPS → CPU |
| augmentation | Ultralytics defaults + `flipud=0.5`, `degrees=10` | HSV jitter, translate, scale, flips, mosaic (off for last 10 epochs) |

```bash
python src/train.py                          # sensible defaults
python src/train.py --epochs 50 --batch 8    # smaller / faster
python src/train.py --model yolo11s.pt --device 0
```

Outputs land in `results/runs/<run_name>/`: `weights/best.pt`, `weights/last.pt`,
`results.csv`, training-curve and confusion-matrix plots, and
`train_summary.json` (records the real optimizer + resolved hyper-parameters).

> **About the optimizer:** with `optimizer="auto"` the current Ultralytics
> releases select **AdamW** for a dataset this size and compute `lr0` and
> `momentum` automatically; for very large datasets they fall back to **SGD**.
> Do not take this on faith — run training and read `optimizer_actual` in
> `results/runs/<run>/train_summary.json` for your exact version.

## Evaluation metrics

Script: **[src/evaluate.py](src/evaluate.py)** — runs on the **unseen test
split** and reports **real** numbers (never hand-written).

```bash
python src/evaluate.py --weights results/runs/<run>/weights/best.pt --split test
```

Reported: **Precision, Recall, mAP@50, mAP@50:95** (overall + per class), plus
PR / F1 curves and a confusion matrix copied into `results/`.

| Term | Definition |
|---|---|
| **IoU** | intersection area ÷ union area of predicted vs. ground-truth box (0–1). A detection is a *true positive* only if IoU ≥ threshold and the class matches. |
| **Precision** | TP / (TP + FP) — of predicted weeds, how many are real. Low → false alarms. |
| **Recall** | TP / (TP + FN) — of real weeds, how many found. Low → misses. |
| **AP** | area under the precision–recall curve for one class at one IoU threshold. |
| **mAP@50** | mean AP over all classes at IoU = 0.50 (lenient localisation). |
| **mAP@50:95** | mean AP averaged over IoU 0.50→0.95 (step 0.05) — the strict, headline COCO metric. |

Ultralytics reports P and R at the confidence that maximises F1; mAP sweeps all
confidences, so **mAP is the primary metric**.

### Results

Real numbers from `src/evaluate.py` on the **235-image test split**.
Dataset: `Francesco/weed-crop-aerial` (2 classes: `crop`, `weed`), imgsz 640,
optimizer **AdamW** (auto-selected), trained on CPU. Two models were run:

| Test metric | `yolo11n` (20 epochs) | **`yolo11s` (~40 epochs)** |
|---|---|---|
| Precision | 0.759 | 0.711 |
| Recall | 0.608 | **0.725** |
| mAP@50 | 0.717 | **0.733** |
| mAP@50:95 | 0.380 | **0.397** |

Going from `yolo11n` to `yolo11s` (3.5x the parameters) plus more epochs lifted
**recall from 0.61 to 0.73** — the model now misses far fewer weeds, which was
the main weakness. Precision dropped slightly (it makes more detections, so a few
more false positives). `yolo11s` per class: weed mAP@50 0.781 / crop 0.685.

![training curves](results/training_curves.png)

Even at 40 epochs the validation curves had not fully plateaued — more epochs, a
GPU (to train `yolo11m`), and moving the dataset off a cloud-synced folder (disk
reads were the training bottleneck) are the obvious next gains.

Full metrics + per-class JSON: [results/metrics_test.json](results/metrics_test.json)
(currently the `yolo11s` run). Best weights: `results/runs/weed_yolo11s_run2b/weights/best.pt`
(git-ignored — regenerate with `src/train.py`).
Test-set PR curve: `results/test_PR_curve.png`; confusion matrix:
`results/test_confusion_matrix_normalized.png`.

## Sample predictions

Script: **[src/predict.py](src/predict.py)**.

```bash
# single image
python src/predict.py --weights results/runs/<run>/weights/best.pt --source field.jpg --conf 0.35

# folder of images
python src/predict.py --weights results/runs/<run>/weights/best.pt --source test_images/ --conf 0.25
```

Saves annotated images (box + species + confidence) to
`results/predictions/<name>/` and a `detections.csv` (one row per box).
`--conf` sets the confidence threshold; `--save-txt` also writes YOLO label
files.

Eight example detections from the test set are in
[results/samples/](results/samples/) (2,004 weed boxes were predicted across the
235 test images at conf 0.25). Example:

![sample detection](results/samples/test_000002.jpg)

## Limitations

Observed on the `yolo11s` run — see the test-set confusion matrix and
**[docs/ERROR_ANALYSIS.md](docs/ERROR_ANALYSIS.md)** for the full breakdown:

- **Missed weeds** are still the main error, though much improved — recall 0.73
  (was 0.60 with `yolo11n`), so ~27% of weeds are still not detected.
- **Background → "weed" false positives:** soil texture, residue and dead leaves
  trigger mostly low-confidence weed boxes; raising `--conf` to ~0.4 trades a
  little recall for fewer false alarms.
- **Crop/weed confusion is low** — the aerial crop rows are visually distinct
  from scattered weeds in this dataset. Expect worse on ground-level datasets.
- **Crop class is weaker** (mAP@50 0.69 vs 0.78 for weed) — only ~410 crop
  instances vs ~7,400 weed in training (class imbalance).
- **Not fully converged:** ~40 epochs on CPU; validation metrics were still
  drifting up.
- **Small / early-growth weeds** are the low-confidence detections (0.25–0.45);
  many are near the `--conf` cutoff and flip in/out with the threshold.
- **Domain gap:** trained on one aerial crop/weed dataset — expect to
  re-fine-tune for a new crop, region, altitude or camera.
- Metrics here are **offline image metrics**, not a field-measured herbicide or
  yield outcome.

## Future improvements

- Train at higher resolution (960/1280) or add image **tiling** for small weeds.
- Try `yolo11s`/`yolo11m`, or hyper-parameter tuning (`model.tune()`).
- Add more data for confused class pairs and rare species; add background images.
- Domain-specific augmentation for lighting; collect multi-season data.
- Crop-row masking to suppress between-row false positives.
- Export to **ONNX / TensorRT** and benchmark FPS on the target edge device.
- Add object tracking for video; estimate weed density maps with GPS.

---

## Installation

Requires **Python 3.9–3.11** and Git.

```bash
# 1. clone
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# 3. (GPU only) install the matching PyTorch build first, e.g. CUDA 12.1:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. dependencies
pip install -r requirements.txt
```

> On Windows, if `Activate.ps1` is blocked, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
>
> **Windows long-path issue:** if `pip install` fails with
> `[WinError 206] The filename or extension is too long` while installing
> `torch`, your project path is too deep and Windows long paths are disabled.
> Fix either way:
> - **A (no admin):** create the venv at a short path and use it, e.g.
>   `python -m venv C:\venvs\weed-detection` then
>   `C:\venvs\weed-detection\Scripts\Activate.ps1` before `pip install`.
> - **B (admin, permanent):** in an elevated PowerShell run
>   `Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' LongPathsEnabled 1`,
>   reboot, then recreate the venv normally.

## How to train

```bash
# Option 1 - small dataset, one command (no login):
python src/fetch_hf_dataset.py        # Francesco/weed-crop-aerial, ~165 MB

# Option 2 - your own / CottonWeedDet12 (see dataset/README.md):
python src/prepare_dataset.py --source dataset/raw/<dataset> --classes dataset/raw/<dataset>/classes.txt

# then, either way:
python src/verify_dataset.py
python src/train.py
```

## How to evaluate

```bash
python src/evaluate.py --weights results/runs/<run>/weights/best.pt --split test
```

## How to run predictions

```bash
python src/predict.py --weights results/runs/<run>/weights/best.pt --source <image_or_folder> --conf 0.25
```

## Project structure

```
AI-Weed-Detection/
├── configs/
│   └── data.yaml            # dataset paths + class names (generated by the fetch/prepare scripts)
├── dataset/                 # NOT committed (see dataset/README.md)
│   └── README.md
├── docs/
│   ├── ARCHITECTURE.md      # YOLO11: input / backbone / neck / head / anchor-free
│   ├── ERROR_ANALYSIS.md    # failure-mode analysis template + limitations
│   └── GITHUB.md            # all git commands
├── results/                 # curated metrics JSON, plots, sample predictions
│   └── runs/                # full training/eval runs (git-ignored)
├── src/
│   ├── fetch_hf_dataset.py  # download a small HF weed dataset -> YOLO layout + data.yaml
│   ├── prepare_dataset.py   # split + YOLO/VOC conversion + write data.yaml
│   ├── verify_dataset.py    # check images & labels, draw annotated samples
│   ├── train.py             # transfer-learn YOLO11n; records real optimizer used
│   ├── evaluate.py          # test-set Precision / Recall / mAP@50 / mAP@50:95
│   ├── predict.py           # inference on image or folder, annotated output + CSV
│   ├── plot_results.py      # clean training-curve figure from results.csv
│   └── utils.py             # shared helpers
├── requirements.txt
├── .gitignore
└── README.md
```

## Saving the project to GitHub

See **[docs/GITHUB.md](docs/GITHUB.md)** for the full command list.

## License / attribution

Code: choose a license (MIT is common for learning projects).
Dataset: **CottonWeedDet12 is CC BY-NC 4.0** (non-commercial, attribution) —
cite the YOLOWeeds paper and the Zenodo DOI if you use it (see
[dataset/README.md](dataset/README.md)).
Ultralytics YOLO11 is AGPL-3.0 (or a commercial license).
