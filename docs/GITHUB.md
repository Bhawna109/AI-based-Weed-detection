# Saving this project to GitHub

You said the repo already exists on GitHub. Replace
`https://github.com/<your-username>/<your-repo>.git` below with its URL
(green **Code** button → HTTPS on the repo page).

## First-time setup (run once, from the project root)

```bash
# initialise git in this folder
git init
git branch -M main

# (optional) set your identity for this repo
git config user.name  "Your Name"
git config user.email "patelbhawna25@gmail.com"

# stage everything that isn't ignored by .gitignore
git add .

# check what will be committed  -> dataset/ , .venv/ , runs/ , *.pt must NOT appear
git status

# first commit
git commit -m "Initial commit: YOLO11 weed-detection project (code, docs, configs)"

# connect to your GitHub repo and push
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

If GitHub already created a `README.md` / `LICENSE` when you made the repo, the
push will be rejected. Either:

```bash
git pull --rebase origin main   # bring their file in, then re-push
git push -u origin main
```

or (only if the remote repo is empty and you don't want its files):

```bash
git push -u origin main --force
```

## Everyday workflow (after making changes)

```bash
git add -A
git commit -m "Short description of the change"
git push
```

## After training — committing results

`.gitignore` blocks `dataset/`, `.venv/`, `results/runs/`, and `*.pt`.
Add only small, meaningful artifacts by hand:

```bash
git add -f results/metrics_test.json \
           results/test_PR_curve.png \
           results/test_confusion_matrix.png \
           results/training_curves.png \
           results/predictions/predict/some_sample.jpg
git commit -m "Add test-set metrics and sample predictions"
git push
```

## Do NOT commit

- `dataset/` images/labels — large, and CottonWeedDet12's license is
  redistribute-with-attribution; link to the source instead (already done in
  `dataset/README.md`).
- `.venv/` — everyone recreates it from `requirements.txt`.
- `results/runs/`, `*.pt` — regenerable; weights are large. Use a GitHub
  **Release** or Git LFS if you must share `best.pt`.

## Sharing trained weights (optional)

```bash
# option A: attach best.pt to a GitHub Release (web UI) — simplest
# option B: Git LFS
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add -f results/runs/<run>/weights/best.pt
git commit -m "Track trained weights with Git LFS"
git push
```
