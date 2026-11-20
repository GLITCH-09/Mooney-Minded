"""Build the submission zip in the layout the brief asks for.

    python -m scripts.package --pptx MooneyMinded_Part1.pptx

Original/ Mooney/ (100 each), Subset/{Original,Mooney} (top 10),
Result1/{Original,Mooney} (Qwen JSON per image + summaries), Result2/{Original,Mooney} (SAM 3 overlays + summary),
and the pptx at the root. Exits non-zero if anything is missing.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

from vision.common import SPLITS, TOP10, list_images

RESULTS = {
    1: (".json", ["qwen3vl4b_{split}.csv", "qwen4max_{split}.csv"]),
    2: (".png", ["sam3_{split}.csv"]),
}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy(files: list[Path], dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst / f.name)


def stage(args: argparse.Namespace, root: Path) -> list[str]:
    problems = []
    images = {split: {p.stem: p for p in list_images(args.data / split)} for split in SPLITS}
    stems = set(images["Original"])
    if len(stems) != 100 or set(images["Mooney"]) != stems:
        problems.append(f"need 100 matching images, found {len(images['Original'])}/{len(images['Mooney'])}")
    missing = [s for s in TOP10 if s not in stems]
    if missing:
        problems.append(f"top-10 images missing: {missing}")
    for split in SPLITS:
        copy(list(images[split].values()), root / split)
        copy([images[split][s] for s in TOP10 if s in images[split]], root / "Subset" / split)

    for k, (ext, summaries) in RESULTS.items():
        for split in SPLITS:
            src, dst = args.outputs / f"Result{k}" / split, root / f"Result{k}" / split
            files = [src / f"{s}{ext}" for s in sorted(stems)]
            absent = [f.name for f in files if not f.exists()]
            if absent:
                problems.append(f"Result{k}/{split}: {len(absent)} of 100 missing (e.g. {absent[0]})")
            copy([f for f in files if f.exists()], dst)
            for name in summaries:
                summary = args.results / name.format(split=split)
                if summary.exists():
                    shutil.copy2(summary, dst / f"_summary_{summary.name}")
                elif name.startswith(("qwen3vl4b", "sam3")):
                    problems.append(f"missing {summary}")

    if args.pptx.exists():
        shutil.copy2(args.pptx, root / args.pptx.name)
    else:
        problems.append(f"pptx not found: {args.pptx}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pptx", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--name", default="MooneyMinded_Part1")
    ap.add_argument("--dest", type=Path, default=Path("submission"))
    args = ap.parse_args()

    root = args.dest / args.name
    shutil.rmtree(root, ignore_errors=True)
    problems = stage(args, root)

    archive = args.dest / f"{args.name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(root.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(root).as_posix())
    checksum = md5(archive)
    (args.dest / f"{archive.name}.md5").write_text(f"{checksum}  {archive.name}\n")

    for d in sorted({p.parent.relative_to(root) for p in root.rglob("*") if p.is_file()}):
        print(f"  {d!s:18s} {sum(1 for p in (root / d).iterdir() if p.is_file())} files")
    print(f"{archive}  {archive.stat().st_size / 1e6:.1f} MB  md5 {checksum}")
    if problems:
        print("PROBLEMS:\n  - " + "\n  - ".join(problems))
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
