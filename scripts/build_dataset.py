"""Build the project dataset from COCO val2017.

For each category we pick images in which one instance of that category clearly
dominates the picture. The images are resized to a common size (so one blur
strength means the same thing everywhere), and the dominant object's mask and
box are saved as ground truth for scoring later.

    python -m scripts.build_dataset --out data

Output:
    data/Original/001.jpg ...     the 100 images (categories in blocks of 10)
    data/gt/masks/001.png ...     mask of the dominant object (0 / 255)
    data/labels.csv               one row per image: category, box, source id

COCO's download server only works over plain http (its https certificate is
issued for another host), so the URLs below are http on purpose.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .common import DEFAULT_CATEGORIES, LABEL_FIELDS

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATIONS_MEMBER = "annotations/instances_val2017.json"
IMAGE_URL = "http://images.cocodataset.org/val2017/{:012d}.jpg"

# (smallest share of the image the object must cover, largest share another
# category may cover). The strictest level that yields enough images wins.
LEVELS = [(0.25, 0.10), (0.20, 0.20), (0.15, 0.35), (0.10, 1.0)]
DOMINANCE = 3.0  # the object must be at least this many times larger than a second one of its kind

Fetch = Callable[[str], bytes]


@dataclass(frozen=True)
class Candidate:
    image_id: int
    width: int
    height: int
    category: str
    segmentation: list
    area_ratio: float


def find_candidates(coco: dict, category: str, min_area: float, max_other_area: float) -> list[Candidate]:
    """Images where one instance of `category` clearly dominates."""
    cat_ids = {c["name"]: c["id"] for c in coco["categories"]}
    if category not in cat_ids:
        raise ValueError(f"unknown COCO category '{category}'")
    cat_id = cat_ids[category]

    images = {im["id"]: im for im in coco["images"]}
    by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)

    found = []
    for image_id, anns in by_image.items():
        image = images[image_id]
        image_area = image["width"] * image["height"]
        same = sorted((a for a in anns if a["category_id"] == cat_id), key=lambda a: -a["area"])
        if not same:
            continue
        if any(a["iscrowd"] for a in same):  # a crowd blob of the target is ambiguous
            continue
        main = same[0]
        ratio = main["area"] / image_area
        if ratio < min_area:
            continue
        if len(same) > 1 and same[1]["area"] * DOMINANCE > main["area"]:
            continue
        if any(a["category_id"] != cat_id and a["area"] / image_area > max_other_area for a in anns):
            continue
        found.append(Candidate(image_id, image["width"], image["height"], category, main["segmentation"], ratio))
    return sorted(found, key=lambda c: c.image_id)  # sorted, so the shuffle below is reproducible


def pick_candidates(coco: dict, category: str, count: int, seed: int) -> tuple[list[Candidate], int]:
    """Shuffled candidates from the strictest level that has at least `count`."""
    sizes = []
    for level, (min_area, max_other) in enumerate(LEVELS):
        candidates = find_candidates(coco, category, min_area, max_other)
        sizes.append(len(candidates))
        if len(candidates) >= count:
            random.Random(f"{seed}:{category}").shuffle(candidates)
            return candidates, level
    raise ValueError(
        f"only {sizes[-1]} usable images for '{category}' even at the loosest level "
        f"(candidates per level: {sizes}); choose another category or lower --per-category"
    )


def polygon_mask(segmentation: list, width: int, height: int) -> np.ndarray:
    """Rasterise COCO polygons into a 0/1 uint8 mask."""
    mask = np.zeros((height, width), np.uint8)
    for polygon in segmentation:
        points = np.round(np.array(polygon, dtype=np.float64).reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
    return mask


def resized_shape(width: int, height: int, long_side: int) -> tuple[int, int]:
    scale = long_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def decode(raw: bytes) -> np.ndarray | None:
    # COCO annotations line up with the stored pixels, so EXIF rotation is ignored here.
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


def save(path: Path, ext: str, image: np.ndarray, params: list[int] | None = None) -> None:
    ok, buf = cv2.imencode(ext, image, params or [])
    if not ok:
        raise OSError(f"could not encode {path}")
    buf.tofile(str(path))


def build(
    out: Path,
    coco: dict,
    categories: list[str],
    per_category: int,
    seed: int,
    long_side: int,
    fetch: Fetch,
    log: Callable[[str], None] = print,
) -> list[dict]:
    out = Path(out)
    (out / "Original").mkdir(parents=True, exist_ok=True)
    (out / "gt" / "masks").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for category in categories:
        candidates, level = pick_candidates(coco, category, per_category, seed)
        log(f"{category}: {len(candidates)} candidates at level {level} (min area {LEVELS[level][0]:.0%})")
        accepted = 0
        for cand in candidates:
            if accepted == per_category:
                break
            image = decode(fetch(IMAGE_URL.format(cand.image_id)))
            if image is None or image.shape[:2] != (cand.height, cand.width):
                log(f"  skipping COCO image {cand.image_id}: unreadable or size differs from its annotation")
                continue

            new_w, new_h = resized_shape(cand.width, cand.height, long_side)
            interp = cv2.INTER_AREA if max(cand.width, cand.height) > long_side else cv2.INTER_CUBIC
            small = cv2.resize(image, (new_w, new_h), interpolation=interp)
            mask = polygon_mask(cand.segmentation, cand.width, cand.height) * 255
            mask = (cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA) >= 128).astype(np.uint8) * 255
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                log(f"  skipping COCO image {cand.image_id}: object vanishes when resized")
                continue

            number = len(rows) + 1
            name = f"{number:03d}"
            save(out / "Original" / f"{name}.jpg", ".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 95])
            save(out / "gt" / "masks" / f"{name}.png", ".png", mask)
            rows.append(
                {
                    "id": name,
                    "file": f"{name}.jpg",
                    "category": category,
                    "coco_image_id": cand.image_id,
                    "orig_width": cand.width,
                    "orig_height": cand.height,
                    "width": new_w,
                    "height": new_h,
                    "box_x0": int(xs.min()),
                    "box_y0": int(ys.min()),
                    "box_x1": int(xs.max()) + 1,
                    "box_y1": int(ys.max()) + 1,
                    "area_ratio": round(cand.area_ratio, 4),
                    "level": level,
                }
            )
            accepted += 1
        if accepted < per_category:
            raise RuntimeError(f"'{category}': only {accepted} of {per_category} candidates could be used")

    with (out / "labels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def http_fetch(url: str, retries: int = 3, timeout: int = 60) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last = error
            time.sleep(2**attempt)
    raise RuntimeError(f"could not download {url}: {last}")


def caching(fetch: Fetch, cache_dir: Path) -> Fetch:
    """Keep every download on disk, so a re-run does not download again."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    def wrapped(url: str) -> bytes:
        path = cache_dir / url.rsplit("/", 1)[-1]
        if path.exists():
            return path.read_bytes()
        data = fetch(url)
        path.write_bytes(data)
        return data

    return wrapped


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as f:
        total = int(response.headers.get("Content-Length", 0))
        done, last_print = 0, -1
        while chunk := response.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total and done * 10 // total != last_print:
                last_print = done * 10 // total
                print(f"  downloading annotations: {done * 100 // total}%", flush=True)
    partial.rename(target)


def load_coco(annotations: Path | None, cache_dir: Path) -> dict:
    if annotations is not None:
        if annotations.suffix == ".zip":
            with zipfile.ZipFile(annotations) as z:
                return json.loads(z.read(ANNOTATIONS_MEMBER))
        return json.loads(annotations.read_text())
    archive = cache_dir / "annotations_trainval2017.zip"
    if not archive.exists():
        print(f"downloading COCO annotations (about 240 MB) from {ANNOTATIONS_URL}")
        download_file(ANNOTATIONS_URL, archive)
    with zipfile.ZipFile(archive) as z:
        return json.loads(z.read(ANNOTATIONS_MEMBER))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the 100-image COCO subset used in Project Part 1.")
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES, help="COCO category names")
    parser.add_argument("--per-category", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=512, help="length of the longer image side in pixels")
    parser.add_argument(
        "--annotations",
        type=Path,
        help="instances_val2017.json or annotations_trainval2017.zip, if you already have it (skips the download)",
    )
    args = parser.parse_args(argv)

    cache = args.out / "_cache"
    coco = load_coco(args.annotations, cache)
    known = {c["name"] for c in coco["categories"]}
    unknown = [c for c in args.categories if c not in known]
    if unknown:
        print(f"unknown categories: {unknown}\nvalid names: {sorted(known)}", file=sys.stderr)
        return 1

    try:
        rows = build(
            args.out,
            coco,
            args.categories,
            args.per_category,
            args.seed,
            args.size,
            caching(http_fetch, cache / "raw"),
        )
    except (ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {len(rows)} images to {args.out / 'Original'} and labels to {args.out / 'labels.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
