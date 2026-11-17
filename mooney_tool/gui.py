"""Gradio GUI for the Mooney generator.

    python -m mooney_tool.gui

Sliders for blur strength and % white pixels, live preview on a single image, and a
folder tab that shows a random contact sheet (to choose ONE global setting by eye)
and runs the batch with exactly the CLI code path.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from .cli import batch, list_images
from .core import make_mooney


def single(img_rgb, blur, white, invert, upside_down):
    if img_rgb is None:
        return None, "Upload an image."
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    mooney, info = make_mooney(bgr, blur, white, invert, upside_down)
    text = f"white target {info.white_pct_target:.0f}% / actual {info.white_pct_actual:.1f}%  |  threshold {info.threshold:.3f}"
    return mooney, text


def contact_sheet(folder, n, seed, blur, white, invert, upside_down, size=256):
    folder = Path(folder)
    if not folder.is_dir():
        return [], f"Folder not found: {folder}"
    paths = list_images(folder)
    if not paths:
        return [], "No images in that folder."
    random.Random(int(seed)).shuffle(paths)
    items = []
    for p in paths[: int(n)]:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        mooney, _ = make_mooney(img, blur, white, invert, upside_down)
        items.append((mooney, p.stem))
    return items, f"Showing {len(items)} of {len(paths)} images with blur={blur}, white={white}%"


def run_batch(in_dir, out_dir, blur, white, invert, upside_down):
    rows = batch(Path(in_dir), Path(out_dir), blur, white, invert, upside_down)
    mean_actual = float(np.mean([r["white_pct_actual"] for r in rows]))
    return f"Wrote {len(rows)} Mooneys to {out_dir} (mean white {mean_actual:.1f}%). params.csv saved alongside."


def build() -> gr.Blocks:
    with gr.Blocks(title="Mooney-Minded: Mooney generator") as demo:
        gr.Markdown("## Mooney-Minded Mooney generator\nGaussian blur, then a percentile threshold so exactly the chosen share of pixels is white.")
        with gr.Row():
            blur = gr.Slider(0, 20, value=4.0, step=0.5, label="Blur strength (Gaussian sigma, px)")
            white = gr.Slider(5, 80, value=40.0, step=1, label="White pixels (%)")
        with gr.Row():
            invert = gr.Checkbox(False, label="Invert polarity (black <-> white)")
            upside = gr.Checkbox(False, label="Upside down")

        with gr.Tab("Single image"):
            with gr.Row():
                inp = gr.Image(label="Input", type="numpy")
                out = gr.Image(label="Mooney", image_mode="L")
            info = gr.Textbox(label="Info", interactive=False)
            for comp in (inp, blur, white, invert, upside):
                comp.change(single, [inp, blur, white, invert, upside], [out, info])

        with gr.Tab("Folder preview"):
            folder = gr.Textbox("data/Original", label="Folder with original images")
            with gr.Row():
                n = gr.Slider(4, 24, value=12, step=1, label="How many to preview")
                seed = gr.Number(0, label="Shuffle seed", precision=0)
            btn = gr.Button("Preview contact sheet")
            gallery = gr.Gallery(label="Preview", columns=4, height="auto")
            msg = gr.Textbox(label="Status", interactive=False)
            btn.click(contact_sheet, [folder, n, seed, blur, white, invert, upside], [gallery, msg])

        with gr.Tab("Batch"):
            in_dir = gr.Textbox("data/Original", label="Input folder")
            out_dir = gr.Textbox("data/Mooney", label="Output folder")
            go = gr.Button("Generate ALL with the current settings", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
            go.click(run_batch, [in_dir, out_dir, blur, white, invert, upside], [status])
    return demo


if __name__ == "__main__":
    build().launch()
