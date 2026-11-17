"""Command-line batch Mooney generator.

Example:
    python -m mooney_tool.cli --input data/Original --output data/Mooney --blur 4 --white 40
    # add-on variants for the analysis
    python -m mooney_tool.cli -i data/Original -o data/variants/inverted --blur 4 --white 40 --invert
    python -m mooney_tool.cli -i data/Original -o data/variants/upside_down --blur 4 --white 40 --upside-down
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from .core import IMAGE_EXTS, make_mooney


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def batch(
    input_dir: Path,
    output_dir: Path,
    blur: float,
    white: float,
    invert: bool = False,
    upside_down: bool = False,
    size: int | None = None,
) -> list[dict]:
    """Process every image in input_dir with the SAME settings; return per-image stats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for path in list_images(input_dir):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip (unreadable): {path.name}")
            continue
        if size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        mooney, info = make_mooney(img, blur, white, invert=invert, upside_down=upside_down)
        out_path = output_dir / (path.stem + ".png")
        cv2.imwrite(str(out_path), mooney)
        rows.append(
            {
                "file": out_path.name,
                "blur_sigma": info.blur_sigma,
                "white_pct_target": info.white_pct_target,
                "white_pct_actual": round(info.white_pct_actual, 2),
                "invert": info.invert,
                "upside_down": info.upside_down,
            }
        )
    with open(output_dir / "params.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-generate Mooney images (Gaussian blur + white-% threshold).")
    ap.add_argument("-i", "--input", type=Path, required=True, help="folder with original images")
    ap.add_argument("-o", "--output", type=Path, required=True, help="folder for Mooney images")
    ap.add_argument("--blur", type=float, default=4.0, help="Gaussian sigma in pixels (default 4)")
    ap.add_argument("--white", type=float, default=40.0, help="target %% of white pixels (default 40)")
    ap.add_argument("--invert", action="store_true", help="swap black/white (contrast polarity flip)")
    ap.add_argument("--upside-down", action="store_true", help="flip the Mooney vertically")
    ap.add_argument("--size", type=int, default=None, help="optional: resize inputs to SIZE x SIZE first")
    args = ap.parse_args()

    rows = batch(args.input, args.output, args.blur, args.white, args.invert, args.upside_down, args.size)
    mean_actual = sum(r["white_pct_actual"] for r in rows) / len(rows)
    worst = max(abs(r["white_pct_actual"] - r["white_pct_target"]) for r in rows)
    print(
        f"{len(rows)} Mooneys -> {args.output}  "
        f"(blur={args.blur}, white target={args.white}%, mean actual={mean_actual:.1f}%, worst miss={worst:.1f} pts)"
    )


if __name__ == "__main__":
    main()
