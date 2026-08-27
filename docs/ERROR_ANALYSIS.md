# Error Analysis & Limitations

> Fill this in **after training** using real predictions from
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
