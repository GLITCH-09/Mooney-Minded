"""Shared helpers for the dataset, vision-system and scoring scripts."""

from __future__ import annotations

import csv
from pathlib import Path

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


def with_article(label: str) -> str:
    """'elephant' -> 'an elephant', 'cat' -> 'a cat'."""
    return ("an " if label[0].lower() in "aeiou" else "a ") + label


def read_labels(path: Path) -> list[dict]:
    """Read labels.csv written by build_dataset (ids stay strings like '007')."""
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))
