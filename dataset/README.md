# Dataset

The image data is **not committed** to this repository (it is large and the
license does not always allow redistribution). Follow the steps below to
download and prepare it locally. `.gitignore` keeps everything here except this
file.

---

## Recommended dataset: CottonWeedDet12

- **What it is:** 5,648 field images of cotton crops with **9,370 bounding-box
  annotations** across **12 common weed species** (e.g. Morningglory, Palmer
  amaranth, Carpetweed, Waterhemp, ...). Images were captured under natural
  field conditions (varied lighting, soil, growth stages).
- **Annotations:** ships in **YOLO** format (and Pascal-VOC), so little/no
  conversion is needed.
- **License:** CC BY 4.0 (free to use with attribution).
- **Source / paper:** Dang et al., *"YOLOWeeds: A novel benchmark of YOLO object
  detectors for multi-class weed detection in cotton production systems"*,
  Computers and Electronics in Agriculture, 2023.
- **Download:** search "CottonWeedDet12" on Zenodo
  (https://zenodo.org/ - dataset record by the paper's authors).

### Why it is suitable

| Requirement | How CottonWeedDet12 meets it |
|---|---|
| Real agricultural fields | Images taken in production cotton fields, not lab |
| Bounding-box annotations | Yes - box-level, multi-class |
| Object detection (not just classification) | Multiple weeds per image, localised |
| Public + permissive license | CC BY 4.0, hosted on Zenodo |
| Reasonable size for a laptop/Colab | ~5.6k images trains in a few hours on one GPU |
| Realistic difficulty | Natural lighting, occlusion, crop/weed similarity |

### Alternative (fastest to start): a Roboflow Universe weed dataset

Roboflow Universe hosts many weed-detection datasets that export **directly in
YOLOv8 format, already split** into `train/valid/test` with their own
`data.yaml`. If you use one of those, you can **skip `prepare_dataset.py`** and
point training straight at that export:

```bash
python src/train.py --data path/to/roboflow_export/data.yaml
```

---

## Preparing the data (for a non-split YOLO or VOC dataset)

1. Download and unzip the dataset somewhere, e.g. `dataset/raw/CottonWeedDet12/`.
2. Make sure there is a `classes.txt` (one class name per line, in index order).
   CottonWeedDet12 includes a class list in its `notes.json` / release notes.
3. Split + organise + generate `configs/data.yaml`:

   ```bash
   python src/prepare_dataset.py \
       --source dataset/raw/CottonWeedDet12 \
       --classes dataset/raw/CottonWeedDet12/classes.txt \
       --train 0.8 --val 0.1 --test 0.1
   ```

4. Verify:

   ```bash
   python src/verify_dataset.py
   ```

Final layout:

```
dataset/
├── images/{train,val,test}/*.jpg
├── labels/{train,val,test}/*.txt      # YOLO: <class> <cx> <cy> <w> <h>  (normalised)
└── classes.txt
```
