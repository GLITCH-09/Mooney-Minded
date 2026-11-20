"""Batch Mooney generator: one global setting for every image.

    python -m mooney_tool.cli -i data/Original -o data/Mooney --blur 2.5 --white 55
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from .core import IMAGE_EXTS, make_mooney


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def batch(input_dir: Path, output_dir: Path, blur: float, white: float, invert: bool = False,
          upside_down: bool = False, size: int | None = None) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in list_images(input_dir):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skipped unreadable {path.name}")
            continue
        if size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        mooney, info = make_mooney(img, blur, white, invert, upside_down)
        cv2.imwrite(str(output_dir / f"{path.stem}.png"), mooney)
        rows.append({"file": f"{path.stem}.png", "blur_sigma": info.blur_sigma,
                     "white_pct_target": info.white_pct_target, "white_pct_actual": round(info.white_pct_actual, 2),
                     "invert": info.invert, "upside_down": info.upside_down})
    with open(output_dir / "params.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-generate Mooney images (Gaussian blur + white-% threshold).")
    ap.add_argument("-i", "--input", type=Path, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--blur", type=float, default=2.5, help="Gaussian sigma in pixels")
    ap.add_argument("--white", type=float, default=55.0, help="share of white pixels in %%")
    ap.add_argument("--invert", action="store_true", help="swap black and white")
    ap.add_argument("--upside-down", action="store_true", help="flip vertically")
    ap.add_argument("--size", type=int, default=None, help="resize inputs to SIZE x SIZE first")
    args = ap.parse_args()
    rows = batch(args.input, args.output, args.blur, args.white, args.invert, args.upside_down, args.size)
    miss = max(abs(r["white_pct_actual"] - r["white_pct_target"]) for r in rows)
    print(f"{len(rows)} Mooneys -> {args.output} (sigma {args.blur}, white {args.white}%, worst miss {miss:.2f} pts)")


if __name__ == "__main__":
    main()
