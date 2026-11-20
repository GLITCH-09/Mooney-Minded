"""Shared helpers for the vision-system runners: labels, deterministic choices, answer parsing."""
from __future__ import annotations

import csv
import random
import re
from pathlib import Path

CLASSES = [
    "giraffe", "elephant", "horse", "cat", "dog",
    "airplane", "motorcycle", "teddy bear", "fire hydrant", "toilet",
]

# accepted answers for the open "name it" probe (whole-word match, lower case)
SYNONYMS = {
    "giraffe": ["giraffe"],
    "elephant": ["elephant"],
    "horse": ["horse", "pony", "foal", "stallion", "mare"],
    "cat": ["cat", "kitten", "kitty"],
    "dog": ["dog", "puppy", "hound"],
    "airplane": ["airplane", "aeroplane", "plane", "aircraft", "jet", "airliner"],
    "motorcycle": ["motorcycle", "motorbike", "motor bike"],
    "teddy bear": ["teddy", "teddy bear", "stuffed animal", "stuffed bear", "plush", "toy bear"],
    "fire hydrant": ["hydrant", "fire hydrant"],
    "toilet": ["toilet", "lavatory", "commode", "wc"],
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
LETTERS = "ABCDEFGHIJ"


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_labels(csv_path: Path) -> dict[str, str]:
    """stem -> label"""
    with open(csv_path) as f:
        return {Path(r["file"]).stem: r["label"] for r in csv.DictReader(f)}


def wrong_label(stem: str, label: str) -> str:
    """Deterministic wrong class per image, identical for the original and all its Mooney variants."""
    others = [c for c in CLASSES if c != label]
    return random.Random(f"wrong:{stem}").choice(others)


def option_order(stem: str) -> list[str]:
    """Deterministic shuffled answer options per image (same for original and Mooney)."""
    opts = CLASSES.copy()
    random.Random(f"opts:{stem}").shuffle(opts)
    return opts


def article(noun: str) -> str:
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def name_correct(answer: str, label: str) -> bool:
    a = answer.lower()
    return any(re.search(rf"\b{re.escape(s)}s?\b", a) for s in SYNONYMS[label])


def parse_letter(answer: str) -> str | None:
    a = answer.strip().upper()
    m = re.fullmatch(rf"\W*([{LETTERS}])\W*", a)  # "B", "**B**", "B)"
    if not m:
        m = re.search(rf"\b([{LETTERS}])\)", a) or re.search(rf"ANSWER(?: IS)?:?\s*\W*([{LETTERS}])\b", a)
    if m:
        return m.group(1)
    found = re.findall(rf"\b([{LETTERS}])\b", a)  # last standalone letter ("I think it's D" -> D)
    return found[-1] if found else None


def parse_yes(answer: str) -> bool | None:
    a = answer.strip().lower()
    if a.startswith("yes"):
        return True
    if a.startswith("no"):
        return False
    return None
