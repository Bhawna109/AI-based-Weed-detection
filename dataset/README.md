# Dataset

The image data is **not committed** to this repository (it is large and the
license is non-commercial / attribution). Download and prepare it locally with
the steps below. `.gitignore` keeps everything here except this file.

---

## Recommended dataset: CottonWeedDet12

- **What it is:** 5,648 real cotton-field RGB images with **9,370 bounding-box
  annotations** across **12 common weed species**. Photos taken with
  smartphones / hand-held cameras under natural field light, June–September 2021,
  southern-US cotton systems.
- **Companion paper:** Dang et al., *"YOLOWeeds: A novel benchmark of YOLO object
  detectors for multi-class weed detection in cotton production systems"*,
  Computers and Electronics in Agriculture 205 (2023) 107655.
- **License:** **CC BY-NC 4.0** — free for research/education **with
  attribution**, **no commercial use** without permission. Cite the paper.
- **Size:** ~29 GB (high-resolution images). Plan disk space and bandwidth.

### The 12 weed classes

`Carpetweed, Cutleaf Groundcherry, Eclipta, Goosegrass, Morningglory, Palmer
Amaranth, Prickly Sida, Purslane, Ragweed, Sicklepod, Spotted Spurge, Waterhemp`

> ⚠️ The **class-index order** must match the label `.txt` files in the download.
> Use the `classes.txt` / `notes.json` that ships **inside the dataset** — do not
> hand-type the order. `prepare_dataset.py` builds `configs/data.yaml` from that
> file for you.

### Where to download

| Source | URL | Notes |
|---|---|---|
| **Zenodo (official)** | https://zenodo.org/records/7535814 | one file `CottonWeedDet12.7z` (~29 GB). Extract with [7-Zip](https://www.7-zip.org/). DOI `10.5281/zenodo.7535814` |
| **Hugging Face (mirror)** | https://huggingface.co/datasets/Voxel51/CottonWeedDet12 | same data; includes VIA-JSON **and** pre-converted YOLO `.txt` labels |

Hugging Face download (needs `pip install huggingface_hub`):

```bash
huggingface-cli download Voxel51/CottonWeedDet12 --repo-type dataset --local-dir dataset/raw/CottonWeedDet12
```

### Why it is suitable

| Requirement | How CottonWeedDet12 meets it |
|---|---|
| Real agricultural fields | production cotton fields, natural light, occlusion |
| Bounding-box annotations | yes — box-level, 12 classes |
| Object detection (not classification) | multiple weeds per image, localised |
| Public dataset | Zenodo + Hugging Face, documented, peer-reviewed |
| Realistic difficulty | varied soil, lighting, growth stages, crop/weed similarity |

Trade-off: it is **large (~29 GB)** and **non-commercial**. For a first run on a
laptop, use one of the smaller options below, then scale up.

---

## Smaller / faster alternatives

### A. A Roboflow Universe weed dataset (fastest to a working pipeline)

Roboflow Universe hosts many weed-detection datasets that export **directly in
YOLOv8 format, already split** into `train/valid/test` with their own
`data.yaml`. Sizes are typically a few hundred MB.

- Browse: https://universe.roboflow.com/search?q=class:weed
- Example: **"Cotton-8"** — 4,440 cotton-field images, 5 classes (broadleaf weed,
  grass weed, cotton stage 1/2/3), YOLOv8 format, 70/20/10 split.

With a Roboflow export you **skip `prepare_dataset.py`** entirely:

```bash
python src/train.py --data path/to/roboflow_export/data.yaml
```

### B. CottonWeedDet3 (same data family, 3 classes)

848 images, 1,532 boxes, classes = morningglory / carpetweed / palmer amaranth.
Smaller than Det12 but still a few GB. Hosted on Kaggle (search
"CottonWeedDet3").

### C. Train on Google Colab

Free GPU. Upload/download the dataset in Colab, run the same scripts there.
CPU-only training of YOLO11n on ~5k images is slow (many hours/epoch).

---

## Preparing the data (non-split YOLO or VOC dataset)

1. Download + extract, e.g. to `dataset/raw/CottonWeedDet12/`.
2. Confirm there is a `classes.txt` (one class name per line, in index order).
   If the download only has `notes.json`, copy the ordered names into
   `classes.txt`.
3. Split + organise + generate `configs/data.yaml`:

   ```bash
   python src/prepare_dataset.py \
       --source dataset/raw/CottonWeedDet12 \
       --classes dataset/raw/CottonWeedDet12/classes.txt \
       --train 0.8 --val 0.1 --test 0.1
   ```

   Add `--voc` if the annotations are Pascal-VOC XML instead of YOLO `.txt`.

4. Verify:

   ```bash
   python src/verify_dataset.py
   ```

Final layout:

```
dataset/
├── images/{train,val,test}/*.jpg
├── labels/{train,val,test}/*.txt      # YOLO: <class> <cx> <cy> <w> <h>  (normalised 0–1)
└── classes.txt
```

## Attribution (required by CC BY-NC 4.0)

> Dang, F., Chen, D., Lu, Y., Li, Z. (2023). YOLOWeeds: A novel benchmark of YOLO
> object detectors for multi-class weed detection in cotton production systems.
> *Computers and Electronics in Agriculture*, 205, 107655.
> Dataset: https://doi.org/10.5281/zenodo.7535814 (CC BY-NC 4.0).
