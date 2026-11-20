"""Build the 100-image set from COCO: 10 silhouette-friendly classes x 10 images.

    python -m scripts.prepare_dataset   # -> data/Original, data/masks, data/labels.csv, data/selection_sheet.png

Per class: one dominant instance covering 8-60 % of the photo, touching at most one border, kept >= 90 %
by a square crop; candidates ranked by figure-ground contrast. COCO ids listed in data/exclude.txt are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

from vision.common import CLASSES

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/{split}/{file}"


def load_split(data: Path, split: str) -> dict:
    coco = data / "coco"
    js = coco / f"instances_{split}.json"
    if not js.exists():
        zp = coco / "annotations_trainval2017.zip"
        if not zp.exists():
            coco.mkdir(parents=True, exist_ok=True)
            print("downloading annotations (~250 MB) ...")
            urllib.request.urlretrieve(ANN_URL, zp)
        with zipfile.ZipFile(zp) as z:
            z.extract(f"annotations/instances_{split}.json", coco)
        (coco / "annotations" / f"instances_{split}.json").rename(js)
    print(f"loading {js.name} ...")
    d = json.loads(js.read_text())
    anns_by_img: dict[int, list] = {}
    for a in d["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    return {
        "split": split,
        "images": {im["id"]: im for im in d["images"]},
        "anns_by_img": anns_by_img,
        "cat_ids": {c["name"]: c["id"] for c in d["categories"]},
    }


def geometric_candidates(S: dict, cls: str, exclude: set[int]) -> list[dict]:
    cid = S["cat_ids"][cls]
    out = []
    for img_id, anns in S["anns_by_img"].items():
        if img_id in exclude:
            continue
        tgt = [a for a in anns if a["category_id"] == cid]
        if len(tgt) != 1 or tgt[0]["iscrowd"]:
            continue
        t = tgt[0]
        if not isinstance(t["segmentation"], list):
            continue
        im = S["images"][img_id]
        W, H = im["width"], im["height"]
        if min(W, H) < 320:
            continue
        frac = t["area"] / (W * H)
        if not 0.08 <= frac <= 0.60:
            continue
        if any(a is not t and a["area"] > 0.5 * t["area"] for a in anns):
            continue
        x, y, w, h = t["bbox"]
        borders = (x <= 2) + (y <= 2) + (x + w >= W - 2) + (y + h >= H - 2)
        if borders > 1:
            continue
        prior = -abs(math.log(frac / 0.30))
        out.append({"split": S["split"], "img": im, "ann": t, "prior": prior})
    out.sort(key=lambda c: c["prior"], reverse=True)
    return out


def polygon_mask(ann: dict, W: int, H: int) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    for poly in ann["segmentation"]:
        pts = np.round(np.array(poly, np.float32).reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m


def fetch_image(data: Path, split: str, fname: str) -> np.ndarray | None:
    p = data / "coco" / "images" / split / fname
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(IMG_URL.format(split=split, file=fname), p)
        except OSError as e:
            print(f"  download failed {fname}: {e}")
            return None
    return cv2.imread(str(p), cv2.IMREAD_COLOR)


def crop_and_score(img: np.ndarray, ann: dict, size: int):
    H, W = img.shape[:2]
    mask = polygon_mask(ann, W, H)
    total = int(mask.sum())
    if total == 0:
        return None
    s = min(W, H)
    x, y, w, h = ann["bbox"]
    cx, cy = x + w / 2, y + h / 2
    x0 = int(np.clip(round(cx - s / 2), 0, W - s))
    y0 = int(np.clip(round(cy - s / 2), 0, H - s))
    img_c, mask_c = img[y0:y0 + s, x0:x0 + s], mask[y0:y0 + s, x0:x0 + s]
    coverage = mask_c.sum() / total
    if coverage < 0.90:
        return None
    img_r = cv2.resize(img_c, (size, size), interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(mask_c.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    mask_r = (mask_r > 0.5).astype(np.uint8)
    frac = float(mask_r.mean())
    if not 0.12 <= frac <= 0.70:
        return None
    gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    ring = cv2.dilate(mask_r, k) - mask_r
    if ring.sum() < 50:
        return None
    contrast = abs(gray[mask_r == 1].mean() - gray[ring == 1].mean()) / 255.0
    return img_r, mask_r, {"coverage": float(coverage), "area_frac": frac, "contrast": float(contrast)}


def contact_sheet(rows: list[dict], orig_dir: Path, out: Path, per_row: int = 10, tile: int = 96) -> None:
    tiles = []
    for r in rows:
        im = cv2.resize(cv2.imread(str(orig_dir / r["file"])), (tile, tile))
        cv2.putText(im, str(r["coco_image_id"]), (2, tile - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        tiles.append(im)
    while len(tiles) % per_row:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)])
    cv2.imwrite(str(out), grid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--pool", type=int, default=35, help="candidates downloaded per class before contrast ranking")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    excl_file = args.data / "exclude.txt"
    exclude = {int(t) for t in excl_file.read_text().split()} if excl_file.exists() else set()

    splits = [load_split(args.data, "val2017")]
    pools = {c: geometric_candidates(splits[0], c, exclude) for c in CLASSES}
    short = [c for c in CLASSES if len(pools[c]) < args.pool]
    if short:
        print(f"topping up from train2017 for: {', '.join(short)} (loading a big json, ~1-2 min)")
        tr = load_split(args.data, "train2017")
        for c in short:
            pools[c] += geometric_candidates(tr, c, exclude)

    orig = args.data / "Original"
    masks = args.data / "masks"
    for d in (orig, masks):
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.png"):
            f.unlink()

    rows, idx = [], 1
    for cls in CLASSES:
        scored = []
        for c in pools[cls]:
            if len(scored) >= args.pool:
                break
            img = fetch_image(args.data, c["split"], c["img"]["file_name"])
            if img is None:
                continue
            res = crop_and_score(img, c["ann"], args.size)
            if res is not None:
                scored.append((c, *res))
        scored.sort(key=lambda t: t[3]["contrast"], reverse=True)
        chosen = scored[: args.per_class]
        print(f"{cls:13s} {len(pools[cls]):4d} geometric candidates, {len(scored):3d} scored, kept {len(chosen)}")
        for c, img_r, mask_r, info in chosen:
            fname = f"{idx:03d}_{cls.replace(' ', '_')}.png"
            cv2.imwrite(str(orig / fname), img_r)
            cv2.imwrite(str(masks / fname), mask_r * 255)
            rows.append({
                "file": fname, "label": cls, "coco_split": c["split"],
                "coco_image_id": c["img"]["id"], "coco_file": c["img"]["file_name"],
                "area_frac": round(info["area_frac"], 3), "contrast": round(info["contrast"], 3),
            })
            idx += 1

    with open(args.data / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    contact_sheet(rows, orig, args.data / "selection_sheet.png")
    print(f"{len(rows)} originals -> {orig}; masks -> {masks}; review data/selection_sheet.png")
    if len(rows) < len(CLASSES) * args.per_class:
        print("WARNING: fewer than 100 images; raise --pool or relax filters")


if __name__ == "__main__":
    main()
