"""Shared helpers for the dataset, vision-system and scoring scripts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

# 10 COCO categories: large, recognisable objects, several of them animals
# (animals make the best Mooneys) plus a few that give the models a chance to
# be funny (zebra stripes, teddy bear).
DEFAULT_CATEGORIES = [
    "cat",
    "dog",
    "horse",
    "elephant",
    "zebra",
    "giraffe",
    "airplane",
    "bus",
    "motorcycle",
    "teddy bear",
]

LABEL_FIELDS = [
    "id",
    "file",
    "category",
    "coco_image_id",
    "orig_width",
    "orig_height",
    "width",
    "height",
    "box_x0",
    "box_y0",
    "box_x1",
    "box_y1",
    "area_ratio",
    "level",
]


# Words that count as a correct answer to "what is the main object?".
# Raw answers are always saved, so borderline cases can be checked by hand.
SYNONYMS = {
    "cat": ["cat", "kitten", "kitty", "feline", "tabby"],
    "dog": ["dog", "puppy", "pup", "hound", "canine", "retriever", "terrier", "poodle", "labrador"],
    "horse": ["horse", "pony", "foal", "stallion", "mare"],
    "elephant": ["elephant"],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],
    "airplane": ["airplane", "aeroplane", "plane", "airliner", "jetliner", "jet", "aircraft"],
    "bus": ["bus", "minibus", "coach", "double-decker"],
    "motorcycle": ["motorcycle", "motorbike", "scooter", "moped", "dirt bike"],
    "teddy bear": ["teddy bear", "teddy", "stuffed bear", "stuffed animal", "stuffed toy", "plush", "toy bear"],
}

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def with_article(label: str) -> str:
    """'elephant' -> 'an elephant', 'cat' -> 'a cat'."""
    return ("an " if label[0].lower() in "aeiou" else "a ") + label


def name_matches(answer: str, category: str) -> bool:
    """True if the free-text answer names the category (or a synonym), as whole words."""
    text = answer.lower()
    return any(re.search(rf"\b{re.escape(word)}s?\b", text) for word in SYNONYMS.get(category, [category]))


def read_labels(path: Path) -> list[dict]:
    """Read labels.csv written by build_dataset (ids stay strings like '007')."""
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def find_image(folder: Path, image_id: str) -> Path | None:
    """The file '<id>.jpg' / '.png' in `folder`, whatever its extension."""
    for suffix in IMAGE_SUFFIXES:
        path = Path(folder) / f"{image_id}{suffix}"
        if path.exists():
            return path
    return None


def write_json(path: Path, data: dict) -> None:
    """Write via a temporary file, so an interrupted run never leaves a half-written result."""
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(json.dumps(data, indent=2))
    partial.replace(path)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two masks (any non-zero value counts as foreground)."""
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0
