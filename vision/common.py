"""Labels, prompts and answer scoring shared by every vision run."""
from __future__ import annotations

import csv
import random
import re
from pathlib import Path

CLASSES = [
    "giraffe", "elephant", "horse", "cat", "dog",
    "airplane", "motorcycle", "teddy bear", "fire hydrant", "toilet",
]
TOP10 = [
    "006_giraffe", "012_elephant", "023_horse", "031_cat", "048_dog",
    "053_airplane", "068_motorcycle", "074_teddy_bear", "087_fire_hydrant", "099_toilet",
]
SPLITS = ("Original", "Mooney")
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
LETTERS = "ABCDEFGHIJ"

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

PROBES = ("describe", "name", "pick", "claim_true", "claim_false")
MAX_NEW_TOKENS = {"describe": 40, "name": 12, "pick": 4, "claim_true": 4, "claim_false": 4}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_labels(path: Path) -> dict[str, str]:
    with open(path) as f:
        return {Path(r["file"]).stem: r["label"] for r in csv.DictReader(f)}


def label_of(stem: str) -> str:
    return stem.split("_", 1)[1].replace("_", " ")


def wrong_label(stem: str, label: str) -> str:
    return random.Random(f"wrong:{stem}").choice([c for c in CLASSES if c != label])


def option_order(stem: str) -> list[str]:
    options = CLASSES.copy()
    random.Random(f"opts:{stem}").shuffle(options)
    return options


def article(noun: str) -> str:
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def prompts(stem: str, label: str) -> dict[str, str]:
    menu = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(option_order(stem)))
    return {
        "describe": "Describe this image in one short sentence.",
        "name": "What is the main object in this image? Answer with 1 to 3 words only.",
        "pick": f"Which of these is shown in the image?\n{menu}\nAnswer with the letter only.",
        "claim_true": f"This is a photo of {article(label)}, right? Answer yes or no.",
        "claim_false": f"This is a photo of {article(wrong_label(stem, label))}, right? Answer yes or no.",
    }


def name_correct(answer: str, label: str) -> bool:
    text = answer.lower()
    return any(re.search(rf"\b{re.escape(s)}s?\b", text) for s in SYNONYMS[label])


def parse_letter(answer: str) -> str | None:
    text = answer.strip().upper()
    m = (re.fullmatch(rf"\W*([{LETTERS}])\W*", text)
         or re.search(rf"\b([{LETTERS}])\)", text)
         or re.search(rf"ANSWER(?: IS)?:?\s*\W*([{LETTERS}])\b", text))
    if m:
        return m.group(1)
    found = re.findall(rf"\b([{LETTERS}])\b", text)
    return found[-1] if found else None


def parse_yes(answer: str) -> bool | None:
    text = answer.strip().lower()
    if text.startswith("yes"):
        return True
    if text.startswith("no"):
        return False
    return None


def score(stem: str, label: str, answers: dict[str, str]) -> dict:
    """Score whichever probes were answered; unanswered probes stay empty."""
    row = {"name_correct": "", "pick_correct": "", "pick_answer": "", "yes_true": "", "yes_false": ""}
    yes = {True: 1, False: 0, None: ""}
    if answers.get("name"):
        row["name_correct"] = int(name_correct(answers["name"], label))
    if answers.get("pick"):
        letter = parse_letter(answers["pick"])
        row["pick_answer"] = option_order(stem)[LETTERS.index(letter)] if letter else ""
        row["pick_correct"] = int(row["pick_answer"] == label)
    if answers.get("claim_true"):
        row["yes_true"] = yes[parse_yes(answers["claim_true"])]
    if answers.get("claim_false"):
        row["yes_false"] = yes[parse_yes(answers["claim_false"])]
    return row


def answer_row(file: str, label: str, wrong: str, answers: dict[str, str]) -> dict:
    flat = {f"ans_{p}": answers.get(p, "").replace("\n", " ").strip() for p in PROBES}
    return {"file": file, "label": label, "wrong_label": wrong,
            **score(Path(file).stem, label, answers), **flat}


def rate(rows: list[dict], key: str) -> str:
    values = [r[key] for r in rows if r[key] != ""]
    return f"{100 * sum(values) / len(values):.0f}% (n={len(values)})" if values else "-"


def read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
