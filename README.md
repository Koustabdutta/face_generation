# CelebA VAE — README

A self-contained project to train a Convolutional Variational Autoencoder (VAE) on the CelebA face dataset (or any local face folder).
This repo contains a robust training script that:

* Downloads a CelebA mirror from **Kaggle** (if you have `kaggle.json` credentials) and extracts it,
* Falls back to local datasets (torchvision `CelebA`, `ImageFolder`, or a flat image folder),
* Is Windows / Jupyter friendly (safe defaults: `num_workers=0`, single-thread BLAS),
* Trains a ConvVAE, computes reconstruction metrics (SSIM / PSNR / MSE / MAE),
* Saves several visualization figures (analysis, generated faces, interpolation, latent traversals),
* Includes helpful diagnostics and fallbacks (dummy dataset if no images are found).

Copy & paste this README into your GitHub repo.

---

## Table of contents

* [Files](#files)
* [Requirements](#requirements)
* [Quick start](#quick-start)

  * [1) Add Kaggle credentials (recommended)](#1-add-kaggle-credentials-recommended)
  * [2) Run the training script](#2-run-the-training-script)
  * [Use a local dataset instead of Kaggle](#use-a-local-dataset-instead-of-kaggle)
* [Output files](#output-files)
* [Recommended settings / tips](#recommended-settings--tips)
* [Common issues & fixes](#common-issues--fixes)
* [License](#license)

---

## Files

* `celeba_vae_kaggle_safe.py` — main script. (Downloads/loads data, trains VAE, saves visualizations.)
* `requirements.txt` — suggested Python packages (create one if you want to pin versions).
* `README.md` — this file.

---

## Requirements

Install (recommended in a venv/conda env):

```bash
pip install torch torchvision scikit-learn pillow matplotlib scikit-image kaggle
```

Suggested `requirements.txt` (example):

```
torch
torchvision
scikit-learn
pillow
matplotlib
scikit-image
kaggle
```

> On Windows, installing `torch` with CUDA should follow the official instructions from PyTorch (choose the right CUDA version).

---

## Quick start

1. Clone the repository and open a terminal in the repo folder.

2. (Optional but recommended) Create and activate a Python virtual environment:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Add Kaggle credentials (recommended)

To allow automatic Kaggle downloads you must create a `kaggle.json`:

* Go to [https://www.kaggle.com](https://www.kaggle.com) → **Account** → **Create API token**. A `kaggle.json` file will download.
* Place it at:

  * Windows: `%USERPROFILE%\.kaggle\kaggle.json`
  * Linux/macOS: `~/.kaggle/kaggle.json`

**Windows PowerShell (copy):**

```powershell
mkdir $env:USERPROFILE\.kaggle -Force
copy "C:\path\to\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
```

If you cannot place `kaggle.json`, run the script with `--no-kaggle` and point to a local dataset.

### 2) Run the training script

Basic command (will attempt Kaggle download first):

```bash
python celeba_vae_kaggle_safe.py
```

Run using a specific data directory (no Kaggle attempt):

```bash
python celeba_vae_kaggle_safe.py --no-kaggle --data-dir "D:\datasets\celeba_local"
```

* `--data-dir` is the base folder used by the script. If Kaggle download runs, it will create `./data/celeba_kaggle`.
* The script tries several load strategies:

  1. Kaggle mirror (flat image folder)
  2. torchvision `CelebA` layout under `data/celeba` (manually extracted)
  3. `ImageFolder` under `data/celeba_local` (images must be inside class subfolders like `all_faces/`)
  4. `FlatFolder` (images flat in `--data-dir`)
  5. Dummy dataset (last resort — produces gray/noisy reconstructions)

---

## Use a local dataset instead of Kaggle

**Option A — Torchvision CelebA manual layout:**
If you manually downloaded CelebA, arrange files like:

```
data/celeba/
  img_align_celeba/       # all jpg images here
  list_attr_celeba.txt
  list_eval_partition.txt
  ...
```

Then run:

```bash
python celeba_vae_kaggle_safe.py --no-kaggle --data-dir ./data
```

**Option B — ImageFolder (preferred simple):**
Place images under:

```
data/celeba_local/all_faces/*.jpg
```

Script will detect and use `ImageFolder('./data/celeba_local')`.

**Option C — Flat folder (no subfolders):**
If all images are in `D:\faces\img_align_celeba\*.jpg`, run:

```bash
python celeba_vae_kaggle_safe.py --no-kaggle --data-dir "D:\faces\img_align_celeba"
```

Script will use a `FlatFolderDataset` to read images directly.

---

## Output files (saved to repo folder)

After training the script saves figures and model:

* `celeba_vae_analysis.png` — training curves, latent visualization, metric histograms.
* `celeba_vae_generated.png` — grid of generated faces.
* `celeba_vae_interpolations.png` — latent-space interpolations.
* `celeba_vae_traversal.png` — latent-dimension traversals.
* `celeba_vae_best.pth` — saved model weights (best test loss).

---

## Recommended settings / tips

* For meaningful face reconstructions use **real images** (CelebA or your own face dataset). If the script uses the dummy dataset (it prints a warning), reconstructions will be random/gray.
* Use GPU (CUDA) for reasonable training time. If you have a modern GPU, keep `BATCH_SIZE=64` and `EPOCHS=50+`.
* If running on Windows / in Jupyter: keep `NUM_WORKERS=0` (default in the script) to avoid worker spawn errors.
* To improve perceptual quality: consider adding a perceptual (VGG) loss in addition to MSE and train longer.

---

## Common issues & fixes

* **`Too many users have viewed...` (Google Drive / gdown quota)**
  Use the Kaggle mirror (script attempts this) or manually download CelebA and place it under `./data/celeba/`.

* **`DataLoader worker ... exited unexpectedly` or blank/gray reconstructions**

  * On Windows/Jupyter set `num_workers=0`. Script defaults to `NUM_WORKERS = 0`.
  * Gray reconstructions usually mean the model trained on **dummy data** (no real images available).

* **`kaggle.json` permission/placement errors**

  * Ensure `%USERPROFILE%\.kaggle\kaggle.json` exists (Windows). Use File Explorer or run PowerShell as Administrator to copy it.
  * On Windows use `copy "C:\path\kaggle.json" "%USERPROFILE%\.kaggle\kaggle.json"` in an elevated PowerShell if permission errors occur.

* **`SystemExit: 2` from argparse in Jupyter**
  The script uses `parse_known_args()` so it is safe to run in notebooks — you should not see this error in the provided script.

* **`ModuleNotFoundError: No module named 'kaggle'`**
  Install with `pip install kaggle` and ensure `kaggle.json` is present.

* **`ImageFolder: Couldn't find any class folder`**
  Place images in a subfolder (e.g. `data/celeba_local/all_faces/*.jpg`) or use the script with the folder containing images as `--data-dir` (it will detect flat folders).

---

## How to cite / credit

This repo uses the CelebA dataset (Liu et al., 2015). If you use this project for research, follow the CelebA citation/licensing terms and reference the dataset appropriately.

---

## License

MIT License — feel free to reuse and adapt. Include the original CelebA dataset license where applicable.

---

Which one would you like next?
