"""Download Imagenette and build the 100-image Original/ set (10 classes x 10 images).

    python scripts/prepare_dataset.py            # -> data/Original/*.png + data/labels.csv

Imagenette (fast.ai) is a 10-class subset of ImageNet with easy, visually distinct classes,
so every image has a known label (needed for the VLM probes). We take 10 images per class from
the validation split with a fixed seed, centre-crop to a square and resize to 256x256 so the
same blur strength means the same thing for every image.
"""
from __future__ import annotations

import argparse
import csv
import random
import tarfile
import urllib.request
from pathlib import Path

import cv2

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
CLASSES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def center_crop_resize(img, size: int):
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    img = img[y0 : y0 + s, x0 : x0 + s]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = args.data / "raw"
    tgz = args.data / "imagenette2-320.tgz"
    root = raw / "imagenette2-320"
    if not root.exists():
        raw.mkdir(parents=True, exist_ok=True)
        if not tgz.exists():
            print(f"downloading {URL} ...")
            urllib.request.urlretrieve(URL, tgz)
        print("extracting ...")
        with tarfile.open(tgz) as tf:
            tf.extractall(raw)

    out = args.data / "Original"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows = []
    idx = 1
    for syn, name in CLASSES.items():
        files = sorted((root / "val" / syn).glob("*.JPEG"))
        for p in rng.sample(files, args.per_class):
            img = center_crop_resize(cv2.imread(str(p)), args.size)
            fname = f"{idx:03d}_{name.replace(' ', '_')}.png"
            cv2.imwrite(str(out / fname), img)
            rows.append({"file": fname, "label": name, "synset": syn, "source": p.name})
            idx += 1
    with open(args.data / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "synset", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} originals -> {out}  (labels: {args.data / 'labels.csv'})")


if __name__ == "__main__":
    main()
