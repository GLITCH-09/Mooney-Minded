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

## Tests

```bash
pip install pytest
python -m pytest
```
