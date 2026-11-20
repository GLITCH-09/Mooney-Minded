"""Assemble the Part 1 submission zip in the exact structure the brief asks for.

    python scripts/package.py --pptx MooneyMinded_Part1.pptx
    # -> submission/MooneyMinded_Part1.zip  (+ .md5), prints file counts, size and md5

Zip layout (folders at the zip root, nothing else):
    Original/        100 originals          (from data/Original)
    Mooney/          100 Mooneys            (from data/Mooney, images only)
    Subset/Original  10 hand-picked originals
    Subset/Mooney    10 matching Mooneys
    Result1/Original, Result1/Mooney   Qwen3-VL answers (one JSON per image + summary CSV)
    Result2/Original, Result2/Mooney   SAM 3 overlays   (one PNG per image + summary CSV)
    <the pptx>
Add-on variants (inverted / upside down) stay OUT of the zip; they only appear in the slides.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

IMG = {".png", ".jpg", ".jpeg"}
# Tom's hand-picked top 10 (one per class)
DEFAULT_SUBSET = [
    "006_giraffe", "012_elephant", "023_horse", "031_cat", "048_dog",
    "053_airplane", "068_motorcycle", "074_teddy_bear", "087_fire_hydrant", "099_toilet",
]


def images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG)


def copy_all(files: list[Path], dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst / f.name)
    return len(files)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pptx", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--subset", nargs="*", default=DEFAULT_SUBSET, help="image stems of the top 10")
    ap.add_argument("--name", default="MooneyMinded_Part1")
    ap.add_argument("--dest", type=Path, default=Path("submission"))
    args = ap.parse_args()

    problems: list[str] = []
    stage = args.dest / args.name
    if stage.exists():
        shutil.rmtree(stage)

    orig, moon = images(args.data / "Original"), images(args.data / "Mooney")
    if len(orig) != 100 or len(moon) != 100:
        problems.append(f"expected 100/100 images, found Original={len(orig)} Mooney={len(moon)}")
    if {p.stem for p in orig} != {p.stem for p in moon}:
        problems.append("Original and Mooney file names do not match")
    copy_all(orig, stage / "Original")
    copy_all(moon, stage / "Mooney")

    by_stem_o, by_stem_m = {p.stem: p for p in orig}, {p.stem: p for p in moon}
    missing = [s for s in args.subset if s not in by_stem_o or s not in by_stem_m]
    if len(args.subset) != 10 or missing:
        problems.append(f"subset must be 10 existing stems; missing: {missing}")
    copy_all([by_stem_o[s] for s in args.subset if s in by_stem_o], stage / "Subset" / "Original")
    copy_all([by_stem_m[s] for s in args.subset if s in by_stem_m], stage / "Subset" / "Mooney")

    for k, csv_prefix in ((1, "vlm"), (2, "sam3")):
        for split in ("Original", "Mooney"):
            src = args.outputs / f"Result{k}" / split
            files = sorted(p for p in src.glob("*") if p.is_file()) if src.is_dir() else []
            if len(files) != 100:
                problems.append(f"Result{k}/{split}: expected 100 result files, found {len(files)} in {src}")
            copy_all(files, stage / f"Result{k}" / split)
            summary = args.results / f"{csv_prefix}_{split}.csv"
            if summary.exists():
                shutil.copy2(summary, stage / f"Result{k}" / split / f"_summary_{summary.name}")
            else:
                problems.append(f"missing summary {summary}")

    if not args.pptx.exists():
        problems.append(f"pptx not found: {args.pptx}")
    else:
        shutil.copy2(args.pptx, stage / args.pptx.name)

    zpath = args.dest / f"{args.name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(stage).as_posix())
    digest = md5(zpath)
    (args.dest / f"{args.name}.zip.md5").write_text(f"{digest}  {zpath.name}\n")

    print("contents:")
    for d in ["Original", "Mooney", "Subset/Original", "Subset/Mooney",
              "Result1/Original", "Result1/Mooney", "Result2/Original", "Result2/Mooney"]:
        print(f"  {d:18s} {sum(1 for _ in (stage / d).glob('*'))} files")
    print(f"zip: {zpath}  {zpath.stat().st_size / 1e6:.1f} MB   md5 {digest}")
    if problems:
        print("\nPROBLEMS (fix before uploading):")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
