# Mooney-Minded

Project Part 1 for Computational Visual Perception (FAU): generate Mooney
images and test how well pretrained vision systems cope with them.

Project requirements, forum rulings and decisions are in [NOTES.md](NOTES.md).

## Mooney generator

A Mooney image is a two-tone picture. The tool converts a photo in three steps:

1. convert to grayscale,
2. blur with a Gaussian (`--sigma`, in pixels),
3. make the brightest `--white-pct` percent of pixels white and the rest black.

Step 3 reads the threshold straight from the sorted pixel intensities (the
percentile), so every image gets its own threshold but the same share of white.

### Setup

```bash
pip install -r requirements.txt
```

### Convert a folder with one setting

```bash
python -m mooney batch Original Mooney --sigma 3 --white-pct 40
```

Every image in `Original` is written to `Mooney` as a PNG with the same name
(`001.jpg` becomes `001.png`). Add `--manifest mooney_settings.csv` to record
the setting and the white percentage actually reached for each image.

A fixed `--sigma` only blurs all images equally if they have a similar
resolution, so bring the originals to a common size first. The tool warns when
sizes differ by more than 25%.

### Choose the setting by eye

```bash
python -m mooney sweep Original/001.jpg Original/002.jpg --out sweeps
```

This writes one contact sheet per image: a row per blur strength and a column
per white percentage. Change the grid with `--sigmas 1 2 4 8` and
`--white-pcts 20 30 40 50`.

### Use from Python

```python
import cv2
from mooney import make_mooney

result = make_mooney(cv2.imread("photo.jpg"), sigma=3, white_pct=40)
cv2.imwrite("photo_mooney.png", result.image)
```

## Pipeline, step by step

Steps 1–3 need only `requirements.txt`. Steps 4–5 need a GPU and the environment from
`requirements-models.txt` (install PyTorch for your CUDA version first; an RTX 50 card
needs a build whose `torch.cuda.get_arch_list()` contains `sm_120`).

1. **Dataset:** 10 COCO categories x 10 images, resized so the long side is 512 px.
   ```bash
   python -m scripts.build_dataset --out data
   ```
   Writes `data/Original/001.jpg` ... (cat 001-010, dog 011-020, horse 021-030,
   elephant 031-040, zebra 041-050, giraffe 051-060, airplane 061-070, bus 071-080,
   motorcycle 081-090, teddy bear 091-100), ground-truth masks in `data/gt/masks/`
   and `data/labels.csv`. COCO's server only works over plain http, so this is intended.
2. **Choose the Mooney setting by eye**, one image from several categories:
   ```bash
   python -m mooney sweep data/Original/001.jpg data/Original/021.jpg data/Original/041.jpg data/Original/061.jpg --out sweeps
   ```
3. **Generate all 100 Mooneys with that one setting:**
   ```bash
   python -m mooney batch data/Original data/Mooney --sigma 3 --white-pct 35 --manifest data/mooney_settings.csv
   ```
4. **Run the two vision systems.** Try three images first:
   ```bash
   python -m scripts.run_vlm --data data --out Result1 --limit 3
   python -m scripts.run_sam --data data --out Result2 --limit 3
   ```
   then run both again without `--limit`. Finished images are skipped, so an
   interrupted run can be restarted.
   - `Result1` is Qwen3-VL answering four questions per image (name it, pick it from
     ten options, agree to a true claim, agree to a false claim).
   - `Result2` is SAM 3, told the object's name and asked to outline it.
5. **Score:**
   ```bash
   python -m scripts.score --data data --vlm Result1 --sam Result2 --out scores
   ```

## Tests

```bash
pip install pytest
python -m pytest
```
