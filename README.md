# Mooney-Minded

Project Part 1 of *Computational Visual Perception* (FAU, WS 2026/27): how well do pretrained vision systems understand Mooney images?

Pipeline: 100 originals -> Mooney images (Gaussian blur + fixed % of white pixels) -> two pretrained vision systems, run on both the originals and the Mooneys.

## Setup (once)

```bash
git clone https://github.com/GLITCH-09/Mooney-Minded && cd Mooney-Minded
python3 -m venv .venv-tool && source .venv-tool/bin/activate
pip install -r requirements.txt
```

The vision models get their own environments later (RTX 5070 = Blackwell, so PyTorch must be a CUDA 12.8+ build).

## 1. Dataset: 100 originals (10 Imagenette classes x 10 images)

```bash
python scripts/prepare_dataset.py
# -> data/Original/001_tench.png ... 100_parachute.png, data/labels.csv
```

## 2. Mooney tool

**GUI** (sliders for blur strength and % white; contact-sheet preview; batch button):

```bash
python -m mooney_tool.gui          # opens http://127.0.0.1:7860
```

**CLI batch** (100+ images, same settings for all):

```bash
python -m mooney_tool.cli -i data/Original -o data/Mooney --blur 4 --white 40
```

How it works (`mooney_tool/core.py`): grayscale -> Gaussian blur (sigma in px) -> threshold at the `(100 - white%)` percentile of the blurred intensities, so exactly `white%` of the pixels turn white. `params.csv` logs the actual white % per image.

Add-on variants for the analysis (same settings, one transform each):

```bash
python -m mooney_tool.cli -i data/Original -o data/variants/inverted    --blur 4 --white 40 --invert
python -m mooney_tool.cli -i data/Original -o data/variants/upside_down --blur 4 --white 40 --upside-down
```

Images and weights are git-ignored (`data/`, `outputs/`).
