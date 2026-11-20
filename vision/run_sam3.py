"""Vision system 2: SAM 3 (pretrained, promptable concept segmentation), prompted with class names.

    python vision/run_sam3.py --images data/Mooney --out outputs/Result2/Mooney
    python vision/run_sam3.py --images data/variants/upside_down --out outputs/variants/Result2/upside_down --flip-gt

Per image, two text prompts on the same image:
  true   the correct class ("giraffe")        -> found it? how well does the mask match COCO's?
  false  a fixed wrong class ("toilet")        -> does SAM 3 hallucinate it? (SAM's version of gaslighting)
Metrics: detected (any mask with score >= --det-thr), top score, IoU of the detected masks
(union) with the ground-truth COCO mask. Because Mooneys are pixel-aligned with their originals,
the same ground truth scores both (flipped for the upside-down variant).

Writes an overlay per image into --out (left: true prompt in green, right: wrong prompt in red,
yellow = ground-truth outline) and a scored CSV into results/sam3_<tag>.csv.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from common import list_images, load_labels, wrong_label


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x)


def as_binary_masks(masks, size: tuple[int, int]) -> np.ndarray:
    """SAM 3 output -> bool array (N, H, W) at the image size."""
    m = to_numpy(masks)
    if m.size == 0:
        return np.zeros((0, *size), bool)
    m = m.reshape(-1, m.shape[-2], m.shape[-1])  # (N,1,H,W) or (N,H,W) -> (N,H,W)
    if m.dtype != bool:
        m = m > (0.0 if (m.min() < 0 or m.max() > 1) else 0.5)  # logits vs probabilities
    if m.shape[1:] != size:
        m = np.stack([cv2.resize(x.astype(np.uint8), size[::-1], interpolation=cv2.INTER_NEAREST) > 0 for x in m])
    return m


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def load_model(conf: float):
    import inspect

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model()
    kw = {"confidence_threshold": conf} if "confidence_threshold" in inspect.signature(Sam3Processor).parameters else {}
    proc = Sam3Processor(model, **kw)

    def segment(img_rgb: np.ndarray, prompts: list[str]):
        from PIL import Image

        import torch

        # SAM 3 keeps parts of the model in bf16; its README runs it under bf16 autocast.
        # Without it: "mat1 and mat2 must have the same dtype, BFloat16 and Float".
        results = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = proc.set_image(Image.fromarray(img_rgb))
            for prompt in prompts:
                if hasattr(proc, "reset_all_prompts"):
                    proc.reset_all_prompts(state)
                out = proc.set_text_prompt(state=state, prompt=prompt)
                masks = out["masks"]
                scores = to_numpy(out["scores"]).reshape(-1)
                results.append((masks, scores))
        return results

    return segment


def overlay(gray: np.ndarray, masks: np.ndarray, gt: np.ndarray, color, caption: str) -> np.ndarray:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    if len(masks):
        u = masks.any(0)
        vis[u] = 0.45 * vis[u] + 0.55 * np.array(color, np.float32)
    vis = vis.astype(np.uint8)
    cnts, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 255), 1)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(vis, caption, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def run(segment, images: Path, labels_csv: Path, masks_dir: Path, out: Path, tag: str,
        results_dir: Path, det_thr: float, flip_gt: bool) -> list[dict]:
    labels = load_labels(labels_csv)
    out.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows, paths, t0 = [], list_images(images), time.time()
    for i, p in enumerate(paths, 1):
        stem = p.stem
        label, wrong = labels[stem], wrong_label(stem, labels[stem])
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gt = cv2.imread(str(masks_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE) > 127
        if flip_gt:
            gt = gt[::-1]
        size = gray.shape
        (m_t, s_t), (m_f, s_f) = segment(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), [label, wrong])
        m_t, m_f = as_binary_masks(m_t, size), as_binary_masks(m_f, size)
        det_t, det_f = m_t[s_t >= det_thr], m_f[s_f >= det_thr]
        best_t = m_t[int(np.argmax(s_t))] if len(s_t) else np.zeros(size, bool)
        row = {
            "file": p.name, "label": label, "wrong_label": wrong,
            "true_detected": int(len(det_t) > 0),
            "true_top_score": round(float(s_t.max()), 4) if len(s_t) else 0.0,
            "true_n_det": len(det_t),
            "true_iou": round(iou(det_t.any(0), gt), 4) if len(det_t) else 0.0,
            "true_iou_best_mask": round(iou(best_t, gt), 4),
            "false_detected": int(len(det_f) > 0),
            "false_top_score": round(float(s_f.max()), 4) if len(s_f) else 0.0,
            "false_n_det": len(det_f),
        }
        rows.append(row)
        left = overlay(gray, det_t, gt, (0, 200, 0), f"'{label}' s={row['true_top_score']:.2f} IoU={row['true_iou']:.2f}")
        right = overlay(gray, det_f, gt, (0, 0, 230), f"'{wrong}' s={row['false_top_score']:.2f}")
        cv2.imwrite(str(out / f"{stem}.png"), np.hstack([left, right]))
        if i % 10 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)}  ({time.time() - t0:.0f}s)")

    with open(results_dir / f"sam3_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    print(f"[{tag}] n={len(rows)}  found true class {100 * mean('true_detected'):.0f}%  "
          f"mean IoU {mean('true_iou'):.2f}  hallucinated wrong class {100 * mean('false_detected'):.0f}%")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    ap.add_argument("--masks", type=Path, default=Path("data/masks"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--det-thr", type=float, default=0.5, help="score needed to count as a detection")
    ap.add_argument("--flip-gt", action="store_true", help="flip ground truth vertically (upside-down variant)")
    args = ap.parse_args()
    segment = load_model(conf=0.05)  # keep weak detections so their scores are logged; we threshold ourselves
    run(segment, args.images, args.labels, args.masks, args.out, args.tag or args.images.name,
        args.results, args.det_thr, args.flip_gt)


if __name__ == "__main__":
    main()
