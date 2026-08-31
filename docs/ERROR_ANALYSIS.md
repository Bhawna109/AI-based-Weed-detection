# Error Analysis & Limitations

---

## Findings — `yolo11s` run (`weed_yolo11s_run2b`)

**Setup:** `yolo11s`, ~40 epochs total (20 as `yolo11n` baseline concept; this
run is a continuation of an earlier `yolo11s` run), imgsz 640, AdamW (auto), CPU.
Dataset `Francesco/weed-crop-aerial`, evaluated on the untouched **235-image test
split** (1,605 boxes: 1,558 weed, 47 crop).

**Test metrics (vs the `yolo11n` 20-epoch baseline):**

| | yolo11n | **yolo11s** |
|---|---|---|
| Precision | 0.759 | 0.711 |
| Recall | 0.608 | **0.725** |
| mAP@50 | 0.717 | **0.733** |
| mAP@50:95 | 0.380 | **0.397** |

**`yolo11s` normalised confusion matrix (columns = ground truth):**

| true \ pred | crop | weed | background (missed) |
|---|---|---|---|
| **crop** | 0.64 | 0.04 | **0.32** |
| **weed** | 0.00 | **0.87** | 0.13 |
| **background** (→ false positive) | 0.03 | **0.97** | – |

### 1. False negatives — still the main error, but improved
- **~27% of weeds are missed** (recall 0.73, up from 0.60). Missed weeds fell
  from 19% → **13%** of true weed boxes; missed crop is still ~32%.
- Causes seen in `results/predictions/run2b_test/`: tiny early-growth seedlings,
  weeds at the image border, plants partly hidden by soil clods / shadow.
- Training curves (`results/training_curves.png`) had not fully plateaued → more
  epochs / a GPU would help further.

### 2. Background → "weed" false positives
- Of every detection placed on a background region, **97% are labelled `weed`**.
- Trigger: soil texture, dead/curled leaves, crop residue, dry stems. These come
  out as **low-confidence** boxes (0.25–0.45 in the samples), so raising
  `--conf` to ~0.4 trades a little recall for noticeably fewer false alarms.

### 3. Missed small weeds
- The lowest-confidence detections are all small seedlings; many sit right at the
  `--conf` threshold and flip in/out with it.
- Mitigation: train/infer at `--imgsz 960/1280`, or tile the image; use `yolo11s`.

### 4. Crop ↔ weed confusion — low here
- Only ~4% of crop boxes are called weed and ~0% the reverse. In this **aerial**
  dataset crop rows are visually distinct from scattered weeds. Expect this to be
  worse on ground-level, single-species datasets like CottonWeedDet12.

### 5. Class imbalance
- `crop` scores lower on every metric. Training data is ~410 crop vs ~7,400 weed
  instances. More crop examples (or `--single-cls` if only weed matters) would
  help.

### 6. Difficult lighting / background
- Wet/dark soil patches and strong shadow edges account for several of the
  background false positives. No systematic dawn/dusk failure was visible in this
  dataset (imagery is mostly even daylight).

### What would most improve this model (in order)
1. Train longer (100+ epochs) on a **GPU** — the CPU run capped how far we got.
2. `yolo11m` instead of `yolo11s`.
3. Higher inference resolution / tiling for small weeds.
4. More `crop` training examples to fix the imbalance.

### Engineering note (not a model issue)
Training/eval printed `Slow image access ... read: ~1 MB/s` — the dataset lives
in a **OneDrive-synced folder**, so every epoch re-reads images through the sync
layer and that was the real training bottleneck (epochs ran 2–3x slower than the
CPU alone would). Move `dataset/` to a plain local folder (e.g. `C:\weed_data`)
and update `path:` in `configs/data.yaml`, or use `--cache disk`.

---

## Template — how to redo this analysis for a new run

> Use real predictions from
> `python src/predict.py ... --source <held-out field images>` and the
> confusion matrix / PR curves in `results/`.
> Keep it honest — documenting failure modes is part of the deliverable.

## How to run the analysis

1. Collect 30–50 **field images the model has never seen** (ideally your own,
   or the untouched `dataset/images/test/` split).
2. Predict at a few confidence thresholds:
   ```bash
   python src/predict.py --weights results/runs/<run>/weights/best.pt --source held_out/ --conf 0.25 --name ea_c25
   python src/predict.py --weights results/runs/<run>/weights/best.pt --source held_out/ --conf 0.50 --name ea_c50
   ```
3. Open `results/predictions/ea_*/` and compare against ground truth (or your
   own judgement). Also inspect `results/test_confusion_matrix.png`.
4. Bucket every mistake into the categories below.

## Categories to report

### 1. False positives (model says "weed", there is none)
- Typical causes: soil texture, crop residue, shadows, rocks, water reflections,
  blurry leaf edges.
- Record: how many per image on average, at conf 0.25 vs 0.50.
- Mitigation: raise `--conf`, add more background/negative images, more epochs.

### 2. False negatives (real weed missed)
- Typical causes: heavy occlusion by the crop, weed at image border, motion blur,
  species under-represented in training data.
- Cross-check against per-class recall in `results/metrics_test.json`.

### 3. Missed small weeds
- Early-growth weeds only a few pixels wide are the hardest and the most
  valuable to catch (early control).
- Mitigation: train/infer at higher `--imgsz` (e.g. 960 or 1280), tile large
  images, use `yolo11s`.
- Report recall on boxes with area < 32×32 px vs the rest if you can.

### 4. Crop / weed confusion
- Grass weeds vs. cereal/cotton seedlings look alike when young.
- Look at off-diagonal cells in the confusion matrix — which class is confused
  with which.
- Mitigation: more labelled examples of the confused pair, higher `imgsz`,
  possibly a bigger model.

### 5. Difficult lighting / background
- Harsh midday sun (blown highlights), long shadows, overcast flatness,
  wet soil, heavy weed pressure (many overlapping plants), dawn/dusk colour cast.
- Note which conditions drop performance most; consider matching augmentation
  (`hsv_v`, `hsv_s`) or collecting more data in those conditions.

## Known limitations of this project (state plainly)

- **Domain gap:** a model trained on one crop / region / camera may not transfer
  to a different field. Re-training or fine-tuning on local images is expected.
- **Species coverage:** the model can only detect weed classes present in the
  training set; novel species are either missed or misclassified.
- **Single-frame, 2D:** no temporal tracking, no depth; overlapping plants can be
  merged or split.
- **Nano model trade-off:** `yolo11n` favours speed over peak accuracy; small and
  occluded weeds suffer most.
- **Annotation quality ceiling:** metrics are bounded by the consistency of the
  dataset's human labels.
- **Not field-validated:** these are offline image metrics, not a measured
  reduction in herbicide use or yield loss.
