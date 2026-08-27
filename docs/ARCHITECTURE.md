# YOLO11 Architecture (as used in this project)

We fine-tune **`yolo11n`** — the *nano* variant of Ultralytics **YOLO11**
(released 2024). It is a single-stage, **anchor-free** object detector.
"Single-stage" = one network looks at the image once and directly outputs
boxes + classes (no separate region-proposal step).

```
             ┌──────────────┐   ┌──────────────────┐   ┌───────────────────┐
  image ───▶ │   BACKBONE   │──▶│       NECK        │──▶│   DETECTION HEAD  │──▶ boxes
  640×640    │ feature      │   │ multi-scale       │   │ (3 scales,        │    + classes
   RGB       │ extraction   │   │ feature fusion    │   │  anchor-free)     │    + scores
             └──────────────┘   └──────────────────┘   └───────────────────┘
                                                              │
                                                              ▼
                                                    Non-Max Suppression (NMS)
```

---

## 1. Input

- Image is **letterboxed** (resized keeping aspect ratio, padded) to a square,
  default **640×640**.
- Pixel values scaled to **0–1**, channel order RGB, shape `(batch, 3, 640, 640)`.
- **During training only**, augmentation is applied first: HSV jitter, random
  translate/scale, horizontal (and here vertical) flip, and **mosaic** (4 images
  tiled into 1). Mosaic is disabled for the final `close_mosaic` epochs.

## 2. Backbone — "what is in the image"

A CSP-style convolutional network that downsamples the image and produces
feature maps at strides **8, 16, 32** (i.e. 80×80, 40×40, 20×20 for a 640 input).
Small stride = high resolution = good for **small weeds**; large stride = more
context = good for large plants.

Key blocks:

| Block | Purpose |
|---|---|
| **Conv** | `Conv2d → BatchNorm → SiLU` activation. The basic unit; stride-2 Convs downsample. |
| **C3k2** | YOLO11's main feature block (an efficient evolution of YOLOv8's C2f / the older C3). Splits the channels, runs several small bottleneck convs on one part, then concatenates — CSP design: strong features, fewer FLOPs. |
| **SPPF** (Spatial Pyramid Pooling – Fast) | Applies 3 successive 5×5 max-pools and concatenates them, so one layer "sees" multiple receptive-field sizes cheaply. |
| **C2PSA** | A lightweight **partial self-attention** block added after SPPF in YOLO11. Lets the network weight the most informative spatial locations — helps separate visually similar crop vs. weed. |

## 3. Neck — "combine detail with context"

A **PAN-FPN** (Path Aggregation Network + Feature Pyramid Network):

- **Top-down (FPN):** upsample deep, semantic-rich features and add them to
  shallower, high-resolution features → shallow layers gain "meaning".
- **Bottom-up (PAN):** send fine spatial detail back up → deep layers gain
  precise location.
- Implemented with `Upsample`, `Concat`, and `C3k2` blocks.

Output: three fused feature maps (P3/P4/P5) handed to the head.

## 4. Detection Head — "where and what"

A **decoupled, anchor-free** head runs on each of the 3 feature maps.
Every spatial cell (an "anchor point" = the centre of that cell) makes **one**
prediction:

### Bounding-box prediction (anchor-free + DFL)
- The model does **not** use predefined anchor boxes. It predicts **4 distances**
  — left, top, right, bottom — from the anchor point to the box edges.
- Each distance is predicted as a **probability distribution over a set of
  discrete bins** (Distribution Focal Loss / DFL). The final distance is the
  **expected value** of that distribution → sub-pixel accurate, smoother to train
  than regressing a single number.
- Distances + anchor-point location → `(x1, y1, x2, y2)` box.

### Classification
- For each of the `nc` classes, an **independent sigmoid** score (multi-label
  style — no softmax). For this project after fine-tuning, the classes are the
  weed species in `data.yaml`.

### Confidence
- YOLO11 (like YOLOv8) has **no separate "objectness" output**. The detection's
  confidence **is** the class probability (the max sigmoid score across classes).
  Contrast YOLOv5, which multiplied a separate objectness score by the class
  score.

### Label assignment & losses (training)
- **Task-Aligned Assigner (TAL):** for each ground-truth box, picks the anchor
  points whose predictions already best match it (combining classification score
  and box IoU) as positives. No IoU-vs-anchor matching needed.
- Losses: **BCE** for classification, **CIoU** + **DFL** for boxes.

## 5. Post-processing

- Thousands of raw predictions → filter by confidence threshold →
  **Non-Maximum Suppression (NMS)** removes duplicate overlapping boxes for the
  same object (keeps the highest-scoring one, drops others with IoU above the
  NMS threshold).

---

## Anchor-based vs anchor-free — where YOLO11 sits

| | Anchor-based (e.g. YOLOv3/v4/v5) | **Anchor-free (YOLOv8, YOLO11 — this project)** |
|---|---|---|
| Priors | k-means "anchor" box shapes per scale, tuned to the dataset | none — predict box edges directly from a point |
| Hyper-params | anchor sizes/counts must be set/tuned | fewer knobs |
| Assignment | match GT to anchors by IoU | Task-Aligned Assigner picks best points |
| Behaviour on odd aspect ratios | can struggle if anchors don't cover them | more flexible |

Because we use `yolo11n`, everything in this project is **anchor-free**.

## Model sizes (pick via `--model`)

| model | params | speed | when to use |
|---|---|---|---|
| `yolo11n.pt` | ~2.6 M | fastest | **default**, edge devices, quick iteration |
| `yolo11s.pt` | ~9 M | fast | if `n` under-fits and you have a GPU |
| `yolo11m.pt` | ~20 M | medium | more accuracy, needs more VRAM |
