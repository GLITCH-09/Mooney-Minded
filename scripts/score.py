"""Turn the raw results into numbers: original vs Mooney, overall and per category.

    python -m scripts.score --data data --vlm Result1 --sam Result2 --out scores

Writes <out>/vlm_per_image.csv, vlm_summary.csv, sam_per_image.csv, sam_summary.csv
and prints the summaries as tables.

VLM (Qwen3-VL), shares of images:
    name / pick     main object named / chosen correctly
    yes_true        agreed to a true claim ("this is a photo of a cat")
    yes_false       agreed to a false claim; compare with yes_true and with the originals
SAM 3:
    found           an instance with score >= the found threshold (default 0.3)
    iou_gt          overlap of the best instance with the ground-truth mask (0 if none)
    consistency     overlap of the Mooney's best mask with the original's best mask
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .common import iou, read_labels

SPLITS = ["Original", "Mooney"]
FOUND_THRESHOLD = 0.3

VLM_METRICS = ["name", "pick", "pick_unparsed", "yes_true", "yes_false", "yes_unparsed"]
SAM_METRICS = ["found", "iou_gt", "consistency", "top_score"]


def score_vlm(labels: list[dict], result_dir: Path, splits: list[str] = SPLITS) -> list[dict]:
    rows = []
    for split in splits:
        for label in labels:
            path = result_dir / split / f"{label['id']}.json"
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            yes_true, yes_false = r["confirm_true"]["yes"], r["confirm_false"]["yes"]
            rows.append(
                {
                    "split": split,
                    "id": label["id"],
                    "category": label["category"],
                    "name": r["name"]["correct"],
                    "pick": r["pick"]["correct"],
                    "pick_unparsed": r["pick"]["choice"] is None,
                    "yes_true": yes_true,
                    "yes_false": yes_false,
                    "yes_unparsed": yes_true is None or yes_false is None,
                }
            )
    return rows


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise OSError(f"cannot read {path}")
    return mask > 0


def score_sam(
    labels: list[dict], data_dir: Path, result_dir: Path, splits: list[str] = SPLITS, found_threshold: float = FOUND_THRESHOLD
) -> list[dict]:
    rows = []
    for split in splits:
        for label in labels:
            path = result_dir / split / f"{label['id']}.json"
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            mask = read_mask(result_dir / split / f"{label['id']}_mask.png")
            top = r["instances"][0]["score"] if r["instances"] else 0.0
            row = {
                "split": split,
                "id": label["id"],
                "category": label["category"],
                "top_score": top,
                "found": top >= found_threshold,
                "iou_gt": iou(mask, read_mask(data_dir / "gt" / "masks" / f"{label['id']}.png")),
                "consistency": None,
            }
            if split != "Original":
                original = result_dir / "Original" / f"{label['id']}_mask.png"
                if original.exists():
                    row["consistency"] = iou(mask, read_mask(original))
            rows.append(row)
    return rows


def aggregate(rows: list[dict], metrics: list[str], by_category: bool) -> list[dict]:
    """Mean of each metric (booleans count as 0 / 1, missing values are ignored)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["category"] if by_category else "all")].append(row)
    summary = []
    for (split, category), members in sorted(groups.items(), key=lambda kv: (kv[0][1] != "all", kv[0][1], kv[0][0])):
        entry = {"split": split, "category": category, "n": len(members)}
        for metric in metrics:
            values = [float(m[metric]) for m in members if m[metric] is not None]
            entry[metric] = round(sum(values) / len(values), 3) if values else None
        summary.append(entry)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(rows: list[dict]) -> str:
    if not rows:
        return "(no results)"
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join("" if row[c] is None else str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the vision-system results.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--vlm", type=Path, default=Path("Result1"))
    parser.add_argument("--sam", type=Path, default=Path("Result2"))
    parser.add_argument("--out", type=Path, default=Path("scores"))
    parser.add_argument("--found-threshold", type=float, default=FOUND_THRESHOLD)
    args = parser.parse_args(argv)

    labels = read_labels(args.data / "labels.csv")
    found_any = False

    vlm_rows = score_vlm(labels, args.vlm)
    if vlm_rows:
        found_any = True
        write_csv(args.out / "vlm_per_image.csv", vlm_rows)
        write_csv(args.out / "vlm_summary.csv", aggregate(vlm_rows, VLM_METRICS, by_category=True))
        print("## Vision-language model (overall)\n")
        print(markdown(aggregate(vlm_rows, VLM_METRICS, by_category=False)), "\n")

    sam_rows = score_sam(labels, args.data, args.sam, found_threshold=args.found_threshold)
    if sam_rows:
        found_any = True
        write_csv(args.out / "sam_per_image.csv", sam_rows)
        write_csv(args.out / "sam_summary.csv", aggregate(sam_rows, SAM_METRICS, by_category=True))
        print("## SAM 3 (overall)\n")
        print(markdown(aggregate(sam_rows, SAM_METRICS, by_category=False)), "\n")

    if not found_any:
        print("no results found; run scripts.run_vlm and scripts.run_sam first", file=sys.stderr)
        return 1
    print(f"per-image and per-category tables written to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
