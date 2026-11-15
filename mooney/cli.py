"""Command-line Mooney generator.

    python -m mooney batch  INPUT OUTPUT_DIR --sigma 3 --white-pct 40
    python -m mooney sweep  IMAGE [IMAGE ...] --out OUTPUT_DIR

`batch` converts one image or a whole folder with a single setting.
`sweep` renders one image under a grid of settings, to choose them by eye.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from .core import make_mooney, to_gray

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"{source} is not a file or folder")


def read_image(path: Path) -> np.ndarray | None:
    # imdecode instead of imread so paths with non-ASCII characters work everywhere.
    # IMREAD_COLOR applies the EXIF orientation, so the Mooney is not rotated
    # relative to how the original photo is displayed.
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def write_png(path: Path, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"could not encode {path}")
    buf.tofile(str(path))


def run_batch(args: argparse.Namespace) -> int:
    images = find_images(args.input)
    if not images:
        print(f"no images found in {args.input}", file=sys.stderr)
        return 1

    stems = [p.stem for p in images]
    clashes = sorted({s for s in stems if stems.count(s) > 1})
    if clashes:
        # a.jpg and a.png would both be written to a.png
        print(f"these names appear with several extensions: {', '.join(clashes)}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    rows, failed, long_sides = [], [], []
    for path in images:
        image = read_image(path)
        if image is None:
            failed.append(path.name)
            continue
        result = make_mooney(image, sigma=args.sigma, white_pct=args.white_pct)
        out_name = path.stem + ".png"  # PNG is lossless, so the image stays strictly two-tone
        write_png(args.output / out_name, result.image)
        height, width = result.image.shape
        long_sides.append(max(height, width))
        rows.append(
            {
                "input": path.name,
                "output": out_name,
                "width": width,
                "height": height,
                "sigma": args.sigma,
                "white_pct_target": args.white_pct,
                "white_pct_actual": round(result.white_pct, 3),
                "threshold": round(result.threshold, 5),
            }
        )

    if args.manifest and rows:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print(f"wrote {len(rows)} Mooney image(s) to {args.output}")
    if rows:
        worst = max(abs(r["white_pct_actual"] - args.white_pct) for r in rows)
        print(f"white pixels: target {args.white_pct}%, largest deviation {worst:.2f} points")
    if long_sides and max(long_sides) > 1.25 * min(long_sides):
        print(
            f"warning: image sizes differ (long side {min(long_sides)}-{max(long_sides)} px), "
            "so a fixed sigma blurs them unevenly; resize the originals to a common size first",
            file=sys.stderr,
        )
    if failed:
        print(f"could not read {len(failed)} file(s): {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def label_cell(text: str, width: int, height: int) -> np.ndarray:
    cell = np.full((height, width), 255, np.uint8)
    scale, thickness = 0.6, 1
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    origin = ((width - tw) // 2, (height + th) // 2)
    cv2.putText(cell, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, 0, thickness, cv2.LINE_AA)
    return cell


def sweep_sheet(image: np.ndarray, sigmas: list[float], white_pcts: list[float], cell: int) -> np.ndarray:
    """Contact sheet: one row per sigma, one column per white %, original in the corner."""
    height, width = image.shape[:2]
    scale = cell / max(height, width)
    cw, ch = max(1, round(width * scale)), max(1, round(height * scale))
    label_w, label_h, gap = 110, 36, 6

    def shrink(img: np.ndarray, interpolation: int) -> np.ndarray:
        return cv2.resize(img, (cw, ch), interpolation=interpolation)

    def to_uint8_gray(img: np.ndarray) -> np.ndarray:
        return (to_gray(img) * 255).round().astype(np.uint8)

    sheet_w = label_w + (cw + gap) * len(white_pcts)
    sheet_h = label_h + (ch + gap) * (len(sigmas) + 1)
    sheet = np.full((sheet_h, sheet_w), 255, np.uint8)

    # original (grayscale) in the first row, so it sits next to its Mooneys
    sheet[label_h : label_h + ch, label_w : label_w + cw] = shrink(to_uint8_gray(image), cv2.INTER_AREA)
    sheet[label_h : label_h + ch, :label_w] = label_cell("original", label_w, ch)

    for col, pct in enumerate(white_pcts):
        x = label_w + col * (cw + gap)
        sheet[:label_h, x : x + cw] = label_cell(f"{pct:g}% white", cw, label_h)
    for row, sigma in enumerate(sigmas):
        y = label_h + (row + 1) * (ch + gap)
        sheet[y : y + ch, :label_w] = label_cell(f"sigma {sigma:g}", label_w, ch)
        for col, pct in enumerate(white_pcts):
            x = label_w + col * (cw + gap)
            # generate at full resolution, then shrink only for display
            mooney = make_mooney(image, sigma=sigma, white_pct=pct).image
            sheet[y : y + ch, x : x + cw] = shrink(mooney, cv2.INTER_AREA)
    return sheet


def run_sweep(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    failed = []
    for path in args.images:
        image = read_image(path) if path.is_file() else None
        if image is None:
            failed.append(str(path))
            continue
        sheet = sweep_sheet(image, args.sigmas, args.white_pcts, args.cell)
        target = args.out / f"{path.stem}_sweep.png"
        write_png(target, sheet)
        print(f"wrote {target}")
    if failed:
        print(f"could not read: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def percentage(text: str) -> float:
    value = float(text)
    if not 0 < value < 100:
        raise argparse.ArgumentTypeError("must be between 0 and 100 (exclusive)")
    return value


def non_negative(text: str) -> float:
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mooney", description="Generate Mooney (two-tone) images.")
    sub = parser.add_subparsers(dest="command", required=True)

    batch = sub.add_parser("batch", help="convert an image or a folder of images with one setting")
    batch.add_argument("input", type=Path, help="image file or folder of images")
    batch.add_argument("output", type=Path, help="folder for the Mooney PNGs")
    batch.add_argument("--sigma", type=non_negative, required=True, help="blur strength: Gaussian sigma in pixels")
    batch.add_argument("--white-pct", type=percentage, required=True, help="percentage of white pixels")
    batch.add_argument("--manifest", type=Path, help="optional CSV listing the settings and achieved white %% per image")
    batch.set_defaults(func=run_batch)

    sweep = sub.add_parser("sweep", help="render images under a grid of settings to compare by eye")
    sweep.add_argument("images", type=Path, nargs="+", help="image file(s)")
    sweep.add_argument("--out", type=Path, required=True, help="folder for the contact sheets")
    sweep.add_argument("--sigmas", type=non_negative, nargs="+", default=[1, 2, 4, 8], help="blur strengths to try")
    sweep.add_argument("--white-pcts", type=percentage, nargs="+", default=[20, 30, 40, 50], help="white percentages to try")
    sweep.add_argument("--cell", type=int, default=220, help="long side of each thumbnail in pixels")
    sweep.set_defaults(func=run_sweep)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
