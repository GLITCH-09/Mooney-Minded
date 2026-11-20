"""System 2: SAM 3, prompted three ways on every image.

    python -m vision.sam --split Original
    python -m vision.sam --split Mooney

text   the class name ("giraffe"): does it find the object, and how well does the mask fit?
box    a box around the object (from the original's annotation): can it outline the shape once told where?
wrong  a wrong class name ("toilet"): does it hallucinate?

Writes one overlay per image (panels: text | box | wrong) to outputs/Result2/<split>/ and
scores to results/sam3_<split>.csv. Mooneys are pixel-aligned with their originals, so the
same ground-truth mask scores both.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .common import SPLITS, list_images, load_labels, write_csv, wrong_label

DET_THR = 0.5
PANEL = 512
HEADER = 44
INK = (24, 24, 24)
COLORS = {"text": (214, 120, 42), "box": (53, 82, 229), "wrong": (214, 80, 98)}  # BGR


class Sam3:
    def __init__(self) -> None:
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.torch = torch
        self.model = build_sam3_image_model(enable_inst_interactivity=True)
        self.processor = Sam3Processor(self.model, confidence_threshold=0.0)

    def __call__(self, rgb: np.ndarray, label: str, wrong: str, box: np.ndarray) -> dict:
        from PIL import Image

        with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            state = self.processor.set_image(Image.fromarray(rgb))
            out = {"text": self._text(state, label), "wrong": self._text(state, wrong)}
            masks, scores, _ = self.model.predict_inst(state, point_coords=None, point_labels=None,
                                                       box=box[None, :], multimask_output=False)
        out["box"] = (np.asarray(masks).reshape(-1, *rgb.shape[:2]) > 0, np.asarray(scores, np.float32).reshape(-1))
        return out

    def _text(self, state: dict, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        self.processor.reset_all_prompts(state)
        state = self.processor.set_text_prompt(prompt=prompt, state=state)
        return state["masks"].squeeze(1).cpu().numpy(), state["scores"].float().cpu().numpy()


def bbox(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def summarize(masks: np.ndarray, scores: np.ndarray, truth: np.ndarray) -> dict:
    if not len(scores):
        return {"score": 0.0, "found": 0, "n": 0, "iou": 0.0, "best_iou": 0.0}
    found = scores >= DET_THR
    top = int(np.argmax(scores))
    return {"score": round(float(scores[top]), 4), "found": int(found.any()), "n": int(found.sum()),
            "iou": round(iou(masks[found].any(0), truth), 4) if found.any() else 0.0,
            "best_iou": round(iou(masks[top], truth), 4)}


def font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def dashed(img: np.ndarray, points: np.ndarray, color, thickness: int, dash: int = 6) -> None:
    for i in range(0, len(points) - 1, 2 * dash):
        cv2.polylines(img, [points[i:i + dash + 1].astype(np.int32)], False, color, thickness, cv2.LINE_AA)


def draw_mask(img: np.ndarray, mask: np.ndarray, color, confident: bool) -> tuple[int, int] | None:
    m = cv2.resize(mask.astype(np.float32), (PANEL, PANEL), interpolation=cv2.INTER_LINEAR) > 0.5
    if not m.any():
        return None
    alpha = 0.5 if confident else 0.3
    img[m] = (alpha * np.array(color) + (1 - alpha) * img[m]).astype(np.uint8)
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if confident:
        cv2.drawContours(img, contours, -1, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.drawContours(img, contours, -1, color, 2, cv2.LINE_AA)
    else:
        for c in contours:
            dashed(img, np.vstack([c, c[:1]]).reshape(-1, 2), color, 3)
    ys, xs = np.nonzero(m)
    return int(xs.min()), int(ys.min())


def panel(image: np.ndarray, title: str, masks: np.ndarray, scores: np.ndarray, color, name: str,
          box: np.ndarray | None = None) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = cv2.resize(image, (PANEL, PANEL), interpolation=cv2.INTER_CUBIC)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    chips = []
    if box is not None:
        x0, y0, x1, y1 = (box * PANEL / image.shape[0]).astype(int)
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 4, cv2.LINE_AA)
        cv2.rectangle(img, (x0, y0), (x1, y1), INK, 2, cv2.LINE_AA)
        if len(scores):
            chips.append((draw_mask(img, masks[0], color, True), name))
    elif len(scores) and (scores >= DET_THR).any():
        chips += [(draw_mask(img, masks[i], color, True), f"{name} {scores[i]:.2f}")
                  for i in np.flatnonzero(scores >= DET_THR)]
    elif len(scores):
        top = int(np.argmax(scores))
        chips.append((draw_mask(img, masks[top], color, False), f"best guess {scores[top]:.2f}"))

    canvas = Image.new("RGB", (PANEL, PANEL + HEADER), INK)
    canvas.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (0, HEADER))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, HEADER // 2), title, font=font(22), fill="white", anchor="lm")
    small, rgb = font(19), tuple(int(c) for c in color[::-1])
    for anchor, text in chips:
        if anchor is None:
            continue
        left, top, right, bottom = draw.textbbox((0, 0), text, font=small)
        w, h = right - left + 14, bottom - top + 10
        x = min(max(anchor[0], 0), PANEL - w)
        y = min(max(anchor[1] - h - 4, 0), PANEL - h) + HEADER
        draw.rectangle((x, y, x + w, y + h), fill=rgb)
        draw.text((x + 7, y + h / 2), text, font=small, fill="white", anchor="lm")
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def run(sam: Sam3, split: str, data: Path, outputs: Path, results: Path) -> None:
    labels = load_labels(data / "labels.csv")
    out_dir = outputs / "Result2" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, path in enumerate(list_images(data / split), 1):
        stem, label = path.stem, labels[path.stem]
        wrong = wrong_label(stem, label)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        truth = cv2.imread(str(data / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE) > 127
        box = bbox(truth)
        pred = sam(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), label, wrong, box)
        text, wrong_s = summarize(*pred["text"], truth), summarize(*pred["wrong"], truth)
        box_masks, box_scores = pred["box"]
        box_iou = iou(box_masks[0], truth) if len(box_scores) else 0.0
        rows.append({"file": path.name, "label": label, "wrong_label": wrong,
                     **{f"text_{k}": v for k, v in text.items()},
                     "box_iou": round(box_iou, 4), "box_quality": round(float(box_scores.max(initial=0.0)), 4),
                     "wrong_score": wrong_s["score"], "wrong_found": wrong_s["found"]})
        shown = image if split == "Original" else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(out_dir / path.name), np.hstack([
            panel(shown, f'Text: "{label}"', *pred["text"], COLORS["text"], label),
            panel(shown, f"Box prompt  IoU {box_iou:.2f}", *pred["box"], COLORS["box"], label, box=box),
            panel(shown, f'Wrong text: "{wrong}"', *pred["wrong"], COLORS["wrong"], wrong),
        ]))
        if i % 25 == 0:
            print(f"  {i} images")
    write_csv(results / f"sam3_{split}.csv", rows)
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    found = [r["text_iou"] for r in rows if r["text_found"]]
    print(f"[{split}] n={len(rows)}  text: found {100 * mean('text_found'):.0f}% "
          f"(IoU {np.mean(found) if found else 0:.2f}), best-guess IoU {mean('text_best_iou'):.2f}  |  "
          f"box: IoU {mean('box_iou'):.2f}  |  wrong text: found {100 * mean('wrong_found'):.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    run(Sam3(), args.split, args.data, args.outputs, args.results)


if __name__ == "__main__":
    main()
