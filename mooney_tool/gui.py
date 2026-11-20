"""Gradio GUI: live preview, folder contact sheet and batch run with one global setting.

    python -m mooney_tool.gui
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
    mooney, info = make_mooney(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), blur, white, invert, upside_down)
    return mooney, (f"white {info.white_pct_actual:.1f}% (target {info.white_pct_target:.0f}%), "
                    f"threshold {info.threshold:.3f}")


def contact_sheet(folder, n, seed, blur, white, invert, upside_down, size=256):
    folder = Path(folder)
    paths = list_images(folder) if folder.is_dir() else []
    if not paths:
        return [], f"No images in {folder}"
    random.Random(int(seed)).shuffle(paths)
    items = []
    for p in paths[: int(n)]:
        img = cv2.resize(cv2.imread(str(p), cv2.IMREAD_COLOR), (size, size), interpolation=cv2.INTER_AREA)
        items.append((make_mooney(img, blur, white, invert, upside_down)[0], p.stem))
    return items, f"{len(items)} of {len(paths)} images at sigma {blur}, white {white}%"


def run_batch(in_dir, out_dir, blur, white, invert, upside_down):
    rows = batch(Path(in_dir), Path(out_dir), blur, white, invert, upside_down)
    return f"{len(rows)} Mooneys -> {out_dir} (mean white {np.mean([r['white_pct_actual'] for r in rows]):.1f}%)"


def build() -> gr.Blocks:
    with gr.Blocks(title="Mooney-Minded") as demo:
        gr.Markdown("## Mooney-Minded\n"
                    "Gaussian blur, then a percentile threshold so exactly the chosen share of pixels is white.")
        with gr.Row():
            blur = gr.Slider(0, 20, value=2.5, step=0.5, label="Blur strength (Gaussian sigma, px)")
            white = gr.Slider(5, 80, value=55, step=1, label="White pixels (%)")
        with gr.Row():
            invert = gr.Checkbox(False, label="Invert polarity")
            upside = gr.Checkbox(False, label="Upside down")
        controls = [blur, white, invert, upside]

        with gr.Tab("Single image"):
            with gr.Row():
                inp = gr.Image(label="Input", type="numpy")
                out = gr.Image(label="Mooney", image_mode="L")
            info = gr.Textbox(label="Info", interactive=False)
            for comp in (inp, *controls):
                comp.change(single, [inp, *controls], [out, info])

        with gr.Tab("Folder preview"):
            folder = gr.Textbox("data/Original", label="Folder")
            with gr.Row():
                n = gr.Slider(4, 24, value=12, step=1, label="Images")
                seed = gr.Number(0, label="Shuffle seed", precision=0)
            preview = gr.Button("Preview")
            gallery = gr.Gallery(label="Preview", columns=4, height="auto")
            status = gr.Textbox(label="Status", interactive=False)
            preview.click(contact_sheet, [folder, n, seed, *controls], [gallery, status])

        with gr.Tab("Batch"):
            in_dir = gr.Textbox("data/Original", label="Input folder")
            out_dir = gr.Textbox("data/Mooney", label="Output folder")
            go = gr.Button("Generate all with the current setting", variant="primary")
            done = gr.Textbox(label="Status", interactive=False)
            go.click(run_batch, [in_dir, out_dir, *controls], [done])
    return demo


if __name__ == "__main__":
    build().launch()
