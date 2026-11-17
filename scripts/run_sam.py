"""Vision system 2: SAM 3, told the name of the object and asked to outline it.

    python -m scripts.run_sam --data data --out Result2 --limit 3     # smoke test
    python -m scripts.run_sam --data data --out Result2               # everything

The prompt is the true category name from labels.csv ("cat", "zebra", ...). For every
image, original and Mooney, the results are written to <out>/<split>/:
    <id>.json         all instances SAM 3 returned: score and box, best first
    <id>_mask.png     mask of the best instance (0 / 255), empty if nothing was found
    <id>_overlay.jpg  that mask drawn on the input image
Existing results are skipped, so an interrupted run can simply be started again.

Even weak detections are kept (low default threshold); scoring decides what counts
as "found".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .common import find_image, read_labels, write_json

DEFAULT_THRESHOLD = 0.05
MODEL_ID = "facebook/sam3"

# (BGR image, text prompt) -> {"scores": [...], "boxes": [[x0, y0, x1, y1], ...], "masks": bool array (N, H, W)}
Segment = Callable[[np.ndarray, str], dict]


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"cannot read {path}")
    return image


def write_image(path: Path, ext: str, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise OSError(f"cannot encode {path}")
    buf.tofile(str(path))


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The mask tinted red with a yellow outline, on top of the image."""
    result = image.copy()
    if mask.any():
        tint = np.zeros_like(result)
        tint[:, :] = (0, 0, 255)
        result = np.where(mask[..., None] > 0, cv2.addWeighted(result, 0.55, tint, 0.45, 0), result)
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 255), 2)
    return result


def process(
    split: str,
    image_dir: Path,
    out_dir: Path,
    labels: list[dict],
    segment: Segment,
    limit: int | None = None,
    log: Callable[[str], None] = print,
) -> int:
    """Segment every image of one split. Returns the number of images processed now."""
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for row in labels[:limit]:
        target = out_dir / f"{row['id']}.json"
        if target.exists():
            continue
        path = find_image(image_dir, row["id"])
        if path is None:
            raise FileNotFoundError(f"no image for id {row['id']} in {image_dir}")
        image = read_bgr(path)
        height, width = image.shape[:2]

        found = segment(image, row["category"])
        scores = np.asarray(found["scores"], dtype=float)
        order = np.argsort(-scores)  # best first
        masks = np.asarray(found["masks"], dtype=bool).reshape(-1, height, width) if len(scores) else np.zeros((0, height, width), bool)
        best = masks[order[0]] if len(order) else np.zeros((height, width), bool)

        write_image(out_dir / f"{row['id']}_mask.png", ".png", best.astype(np.uint8) * 255)
        write_image(out_dir / f"{row['id']}_overlay.jpg", ".jpg", overlay(image, best))
        write_json(
            target,
            {
                "id": row["id"],
                "split": split,
                "category": row["category"],
                "prompt": row["category"],
                "n_instances": int(len(scores)),
                "instances": [
                    {
                        "score": round(float(scores[i]), 4),
                        "box": [round(float(v), 1) for v in found["boxes"][i]],
                        "mask_area_ratio": round(float(masks[i].mean()), 4),
                    }
                    for i in order
                ],
            },
        )
        done += 1
        top = f"{scores[order[0]]:.2f}" if len(order) else "none"
        log(f"[{split}] {row['id']} {row['category']:<11} instances: {len(scores)}  best score: {top}")
    return done


# --- the part that needs a GPU and the model weights -------------------------------------------


def load_model(device: str):
    from transformers import Sam3Model, Sam3Processor

    model = Sam3Model.from_pretrained(MODEL_ID).to(device).eval()
    return model, Sam3Processor.from_pretrained(MODEL_ID)


def make_segment(model, processor, device: str, threshold: float) -> Segment:
    import torch
    from PIL import Image

    def segment(image: np.ndarray, prompt: str) -> dict:
        height, width = image.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = processor(images=pil, text=prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        result = processor.post_process_instance_segmentation(
            outputs, threshold=threshold, mask_threshold=0.5, target_sizes=[(height, width)]
        )[0]
        return {
            "scores": result["scores"].float().cpu().tolist(),
            "boxes": result["boxes"].float().cpu().tolist(),
            "masks": result["masks"].cpu().numpy().astype(bool),
        }

    return segment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SAM 3 (text-prompted) on the original and Mooney images.")
    parser.add_argument("--data", type=Path, default=Path("data"), help="folder with labels.csv, Original/ and Mooney/")
    parser.add_argument("--out", type=Path, default=Path("Result2"))
    parser.add_argument("--splits", nargs="+", default=["Original", "Mooney"])
    parser.add_argument("--limit", type=int, help="only the first N images (smoke test)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="minimum instance score to keep")
    args = parser.parse_args(argv)

    labels = read_labels(args.data / "labels.csv")
    for split in args.splits:
        if not (args.data / split).is_dir():
            print(f"missing folder {args.data / split}", file=sys.stderr)
            return 1

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("warning: no GPU found, this will be slow", file=sys.stderr)
    model, processor = load_model(device)
    segment = make_segment(model, processor, device, args.threshold)
    for split in args.splits:
        n = process(split, args.data / split, args.out / split, labels, segment, args.limit)
        print(f"{split}: {n} new result(s) in {args.out / split}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
