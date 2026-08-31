# Error Analysis & Limitations

---

## Findings — best run (`yolo11s`, 60 epochs, Colab T4 GPU)

**Setup:** `yolo11s`, 60 epochs, imgsz 640, AdamW (auto), trained on a Colab T4
via `notebooks/train_colab.ipynb`. Dataset `Francesco/weed-crop-aerial`,
evaluated on the untouched **235-image test split** (1,605 boxes: 1,558 weed,
47 crop).

**Test metrics across all three runs:**

| | yolo11n·20ep·CPU | yolo11s·25ep·CPU | **yolo11s·60ep·GPU** |
|---|---|---|---|
| Precision | 0.759 | 0.711 | **0.783** |
| Recall | 0.608 | 0.725 | **0.722** |
| mAP@50 | 0.717 | 0.733 | **0.796** |
| mAP@50:95 | 0.380 | 0.397 | **0.492** |

**Normalised confusion matrix, best run (columns = ground truth):**

| true \ pred | crop | weed | background (missed) |
|---|---|---|---|
| **crop** | 0.72 | 0.04 | **0.23** |
| **weed** | 0.00 | **0.84** | 0.16 |
| **background** (→ false positive) | 0.06 | **0.94** | – |

### 1. False negatives — the main remaining error
- **~28% of weeds are missed** (recall 0.72). The matrix shows 16% of true weed
  boxes and 23% of true crop boxes predicted as background.
- Causes seen in `results/predictions/colab_test/`: tiny early-growth seedlings,
  weeds at the image border, plants partly hidden by soil clods / shadow.
- The mAP@50:95 jump (0.40 → 0.49) shows the boxes that *are* found are now much
  tighter — the GPU run's extra epochs mostly improved localisation.

### 2. Background → "weed" false positives
- Of every detection placed on a background region, **94% are labelled `weed`**.
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
1. `yolo11m` instead of `yolo11s` (needs the GPU path).
2. Higher inference resolution / tiling for small weeds (`--imgsz 960`).
3. More `crop` training examples to fix the imbalance.
4. A confidence threshold sweep to pick the P/R operating point for the use case
   (spot-spraying wants high recall; mapping can tolerate lower).

### Engineering note (not a model issue)
On the CPU runs, training printed `Slow image access ... read: ~1 MB/s` — the
dataset had been left in a **OneDrive-synced folder**, so every epoch re-read
images through the sync layer (~2–3x slower). Fixed by moving `dataset/` to a
plain local folder (`C:\weed_data`) and pointing `path:` in `configs/data.yaml`
there. On Colab the data is on local disk so this doesn't apply.

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
