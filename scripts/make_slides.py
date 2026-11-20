"""Build the Part 1 PowerPoint deck from the local data, outputs and results.

    python scripts/make_slides.py --tool-shot shots/gui.png --code-shot shots/code.png
    # -> MooneyMinded_Part1.pptx

Slide order follows the brief: title, approach, 10 originals + 10 Mooneys, vision-system
declaration, results per system, tool screenshot, code screenshot, AI-usage declaration
(+ a takeaways slide). Charts are native PowerPoint charts, so numbers stay editable.
Screenshots that are not given yet leave a labelled empty frame; re-run once you have them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

AUTHOR = "Gaarv Khanna"
BLUR_SIGMA = 2.5
WHITE_PCT = 55
TOP10 = ["006_giraffe", "012_elephant", "023_horse", "031_cat", "048_dog",
         "053_airplane", "068_motorcycle", "074_teddy_bear", "087_fire_hydrant", "099_toilet"]
GALLERY = ["023_horse", "053_airplane", "074_teddy_bear", "099_toilet"]  # Qwen's best misreadings

INK, PAPER, MUTED, LIGHT = "151515", "FFFFFF", "5F6670", "F1F2F4"
ACCENT, ORIG = "E4572E", "9AA3AD"  # Mooney series = accent, Original series = grey
HEAD, BODY = "Cambria", "Calibri"
W, H = 13.333, 7.5


def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h)


# ------------------------------------------------------------------ primitives
def text(slide, x, y, w, h, paras, size=16, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, space_after=6):
    """paras: str | list[str | list[(text, {bold, color, size, italic})]]"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        runs = [(para, {})] if isinstance(para, str) else para
        for t, o in runs:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = o.get("font", font)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", False)
            f.color.rgb = rgb(o.get("color", color))
    return tb


def bullets(slide, x, y, w, h, items, size=15, color=INK):
    """items: list of (bold lead, rest)"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, (lead, rest) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(9)
        for t, b in ((lead, True), (rest, False)):
            if not t:
                continue
            r = p.add_run()
            r.text = t
            r.font.name, r.font.size, r.font.bold, r.font.color.rgb = BODY, Pt(size), b, rgb(color)
    return tb


def rect(slide, x, y, w, h, fill=LIGHT, line=None, rounded=True, dash=False):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                                 Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded:
        shp.adjustments[0] = 0.06
    if fill:
        shp.fill.solid()
        shp.fill.fore_color.rgb = rgb(fill)
    else:
        shp.fill.background()
    if line:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(1.25)
        if dash:
            from pptx.enum.dml import MSO_LINE_DASH_STYLE
            shp.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def picture(slide, path: Path, x, y, w=None, h=None):
    if path is None or not Path(path).exists():
        rect(slide, x, y, w or 2, h or 2, fill=None, line=MUTED, dash=True)
        text(slide, x, y, w or 2, h or 2, f"missing: {Path(path).name if path else '?'}", size=10,
             color=MUTED, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        return None
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def title(slide, t, sub=None, color=INK):
    text(slide, 0.6, 0.45, W - 1.2, 0.8, t, size=34, bold=True, font=HEAD, color=color)
    if sub:
        text(slide, 0.6, 1.2, W - 1.2, 0.45, sub, size=16, color=MUTED)


def new_slide(prs, dark=False):
    s = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(INK if dark else PAPER)
    return s


def bar_chart(slide, x, y, w, h, cats, series: dict, colors, maxv=100, fmt='0"%"', horizontal=False):
    cd = CategoryChartData()
    cd.categories = cats
    for name, vals in series.items():
        cd.add_series(name, [round(float(v), 1) for v in vals])
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    ch = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    ch.font.name, ch.font.size, ch.font.color.rgb = BODY, Pt(12), rgb(MUTED)
    ch.has_title = False
    ch.has_legend = len(series) > 1
    if ch.has_legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(13)
    plot = ch.plots[0]
    plot.gap_width, plot.overlap = 60, -5
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = fmt, False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size, dl.font.color.rgb = Pt(12), rgb(INK)
    for s, c in zip(plot.series, colors):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = rgb(c)
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale = 0, maxv
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb("E3E5E8")
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(11)
    va.tick_labels.number_format, va.tick_labels.number_format_is_linked = fmt, False
    ca = ch.category_axis
    ca.tick_labels.font.size = Pt(12)
    ca.format.line.color.rgb = rgb("C9CDD2")
    if horizontal:
        ca.reverse_order = True
    return ch


def stat(slide, x, y, w, big, label, color=ACCENT):
    text(slide, x, y, w, 0.9, big, size=44, bold=True, font=HEAD, color=color)
    text(slide, x, y + 0.85, w, 0.7, label, size=13, color=MUTED)


# ------------------------------------------------------------------ data
def pct(df, col):
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return 100 * v.mean() if len(v) else float("nan")


def load(results: Path):
    def r(name):
        p = results / f"{name}.csv"
        return pd.read_csv(p) if p.exists() else None
    return r("vlm_Original"), r("vlm_Mooney"), r("sam3_Original"), r("sam3_Mooney")


# ------------------------------------------------------------------ slides
def s_title(prs, data: Path):
    s = new_slide(prs, dark=True)
    text(s, 0.8, 1.6, 7.2, 0.5, "Computational Visual Perception · Project Part 1", size=16, color="B8BEC6")
    text(s, 0.8, 2.2, 7.2, 1.9, "Can pretrained vision models see Mooney images?", size=40, bold=True,
         font=HEAD, color=PAPER)
    text(s, 0.8, 4.4, 7.2, 0.5, AUTHOR, size=22, bold=True, color=ACCENT)
    text(s, 0.8, 4.95, 7.2, 0.5, "Solo team · FAU Erlangen-Nürnberg · 20 November 2026", size=15, color="B8BEC6")
    picture(s, data / "Mooney" / "012_elephant.png", 8.6, 1.35, w=4.1, h=4.1)
    text(s, 8.6, 5.6, 4.1, 0.4, "Mooney of a COCO photo (σ = 2.5 px, 55 % white)", size=11,
         color="8D949C", align=PP_ALIGN.CENTER)


def s_approach(prs, data: Path):
    s = new_slide(prs)
    title(s, "Approach")
    bullets(s, 0.6, 1.45, 7.6, 5.6, [
        ("Question. ", "Humans recognise objects in two-tone Mooney images from shape alone. Do pretrained "
                       "vision systems, without any fine-tuning?"),
        ("Images. ", "100 COCO photos, 10 classes × 10: giraffe, elephant, horse, cat, dog, airplane, motorcycle, "
                     "teddy bear, fire hydrant, toilet. Chosen for distinctive silhouettes: one dominant object, mostly "
                     "in frame, strong figure–ground contrast. Square 256 × 256 crops; COCO masks kept as ground truth."),
        ("Mooneys. ", f"Grayscale → Gaussian blur (σ = {BLUR_SIGMA} px) → threshold at the "
                      f"{100 - WHITE_PCT}th percentile, so exactly {WHITE_PCT} % of pixels are white. One setting for "
                      "all 100 (chosen by eye); top 10 picked by hand."),
        ("System 1: Qwen3-VL-4B. ", "Five questions per image: describe it, name it, pick 1 of 10 classes, "
                                     "“this is a <class>, right?” for the true and for a wrong class."),
        ("System 2: SAM 3. ", "Text-prompted segmentation with the true class and a wrong class. Detected if "
                               "score ≥ 0.5; mask quality = IoU with the COCO mask."),
    ], size=15)
    stem = "012_elephant"
    picture(s, data / "Original" / f"{stem}.png", 8.75, 1.55, w=1.85, h=1.85)
    picture(s, data / "Mooney" / f"{stem}.png", 10.85, 1.55, w=1.85, h=1.85)
    text(s, 8.75, 3.45, 1.85, 0.3, "Original", size=11, color=MUTED, align=PP_ALIGN.CENTER)
    text(s, 10.85, 3.45, 1.85, 0.3, "Mooney", size=11, color=MUTED, align=PP_ALIGN.CENTER)
    for i, (head, sub) in enumerate([("Qwen3-VL", "What is it? (words)"), ("SAM 3", "Where is it? (mask)")]):
        x = 8.75 + i * 2.1
        rect(s, x, 4.15, 1.85, 1.0, fill=LIGHT)
        text(s, x, 4.15, 1.85, 1.0, [[(head, {"bold": True, "size": 15})], [(sub, {"size": 11, "color": MUTED})]],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=2)
    text(s, 8.75, 5.4, 3.95, 1.2, "Both systems see each original and its pixel-aligned Mooney with identical "
                                  "prompts, so any drop is caused by the Mooney transform.", size=12, color=MUTED)


def s_gallery(prs, data: Path):
    s = new_slide(prs)
    title(s, "Our top 10: originals and their Mooneys")
    size, gap = 1.18, 0.07
    x0 = (W - (10 * size + 9 * gap)) / 2
    for i, stem in enumerate(TOP10):
        x = x0 + i * (size + gap)
        picture(s, data / "Original" / f"{stem}.png", x, 1.75, w=size, h=size)
        picture(s, data / "Mooney" / f"{stem}.png", x, 3.1, w=size, h=size)
        text(s, x, 4.38, size, 0.5, stem.split("_", 1)[1].replace("_", " "), size=11, color=MUTED,
             align=PP_ALIGN.CENTER)
    text(s, x0, 5.2, 10 * size + 9 * gap, 0.9,
         f"All 100 Mooneys use the same setting (σ = {BLUR_SIGMA} px, {WHITE_PCT} % white). These 10, one per class, "
         "were picked by hand as the most recognisable to a human viewer.", size=14, color=MUTED)


def s_declaration(prs):
    s = new_slide(prs)
    title(s, "Vision systems used", "Both pretrained and frozen: no fine-tuning, run locally on an RTX 5070 (12 GB)")
    cards = [
        ("System 1 · Qwen3-VL-4B-Instruct", [
            ("Type", "Vision-language model (image + text in, text out)"),
            ("By", "Qwen team, Alibaba · Apache 2.0"),
            ("Weights", "Qwen/Qwen3-VL-4B-Instruct (Hugging Face), bf16"),
            ("Parts", "SigLIP-2 vision encoder → MLP merger → Qwen3 LLM"),
            ("Our use", "5 scripted questions per image, greedy decoding, answers parsed and scored"),
        ]),
        ("System 2 · SAM 3", [
            ("Type", "Promptable concept segmentation (text in, masks + scores out)"),
            ("By", "Meta AI · SAM License"),
            ("Weights", "facebook/sam3 (Hugging Face), 848M parameters"),
            ("Parts", "ViT image encoder + text encoder → fusion → DETR decoder with presence head"),
            ("Our use", "Prompted with true and wrong class; detection at score ≥ 0.5; IoU vs COCO mask"),
        ]),
    ]
    for i, (head, rows) in enumerate(cards):
        x = 0.6 + i * 6.2
        rect(s, x, 1.95, 5.9, 3.9, fill=LIGHT)
        text(s, x + 0.35, 2.2, 5.2, 0.5, head, size=20, bold=True, font=HEAD)
        paras = [[(f"{k}: ", {"bold": True}), (v, {})] for k, v in rows]
        text(s, x + 0.35, 2.95, 5.2, 3.6, paras, size=14, space_after=10)


def s_qwen(prs, vo, vm):
    s = new_slide(prs)
    title(s, "Qwen3-VL: perfect on photos, at chance on Mooneys",
          "System 1 · 100 originals vs. their 100 Mooneys, identical questions")
    cats = ["Names it", "Picks it (1 of 10)", "Yes to true class", "Yes to wrong class"]
    cols = ["name_correct", "pick_correct", "yes_true", "yes_false"]
    bar_chart(s, 0.5, 1.8, 8.0, 5.1, cats, {"Original": [pct(vo, c) for c in cols],
                                            "Mooney": [pct(vm, c) for c in cols]}, [ORIG, ACCENT])
    stat(s, 9.0, 1.9, 3.8, f"{pct(vo, 'pick_correct'):.0f}% → {pct(vm, 'pick_correct'):.0f}%",
         "10-way choice accuracy, original → Mooney (chance = 10 %)")
    abstract = int(vm["ans_describe"].str.lower().str.contains("abstract").sum())
    person = int(vm["ans_describe"].str.lower().str.contains(r"person|human|figure|face|woman|\bman\b").sum())
    stat(s, 9.0, 3.6, 3.8, f"{abstract}/100", "Mooney descriptions that call the image “abstract”")
    stat(s, 9.0, 5.3, 3.8, f"{person}/100", "Mooney descriptions that see a person, figure or face", color=INK)
    s.notes_slide.notes_text_frame.text = (
        "Qwen answers almost everything right on the photos. On the Mooneys it is at chance level on the "
        "10-way choice and almost never names the object. It is not gullible: it says no to both true and wrong claims.")


def s_qwen_sees(prs, data: Path, vm):
    s = new_slide(prs)
    title(s, "What Qwen thinks it sees", "It reads the style, not the object, and falls back on a few default guesses")
    vm = vm.set_index(vm["file"].str[:-4])
    size = 2.05
    for i, stem in enumerate(GALLERY):
        x, y = 0.6 + (i % 2) * (size + 0.35), 1.85 + (i // 2) * (size + 0.55)
        picture(s, data / "Mooney" / f"{stem}.png", x, y, w=size, h=size)
        guess = str(vm.loc[stem, "ans_name"]).strip().strip(".") if stem in vm.index else "?"
        text(s, x, y + size + 0.05, size, 0.45, [[("Qwen: ", {"bold": True}), (f"“{guess}”", {"color": ACCENT}),
                                                  (f"  (a {stem.split('_', 1)[1].replace('_', ' ')})",
                                                   {"color": MUTED, "size": 11})]], size=12)
    counts = vm["pick_answer"].fillna("no answer").value_counts()
    bar_chart(s, 5.6, 1.75, 7.2, 4.9, list(counts.index), {"Times chosen": list(counts.values)}, [ACCENT],
              maxv=max(50, int(counts.max()) + 8), fmt="0", horizontal=True)
    text(s, 5.6, 6.65, 7.2, 0.5, "Forced choice on the 100 Mooneys (each class is correct 10 times)", size=12,
         color=MUTED)


def s_sam(prs, outputs: Path, so, sm):
    s = new_slide(prs)
    title(s, "SAM 3: told what to look for", "System 2 · prompted with the true class and a wrong class on every image")
    if so is None or sm is None:
        rect(s, 0.6, 1.9, 12.1, 4.8, fill=None, line=MUTED, dash=True)
        text(s, 0.6, 1.9, 12.1, 4.8, "SAM 3 results missing: run vision/run_sam3.py, then rebuild the deck",
             size=18, color=MUTED, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        return
    cats = ["Finds the true class", "Mask overlap (IoU)", "Finds the wrong class"]
    ser = {"Original": [pct(so, "true_detected"), 100 * so["true_iou"].mean(), pct(so, "false_detected")],
           "Mooney": [pct(sm, "true_detected"), 100 * sm["true_iou"].mean(), pct(sm, "false_detected")]}
    bar_chart(s, 0.5, 1.8, 6.6, 5.1, cats, ser, [ORIG, ACCENT])
    sub = sm[sm["file"].str[:-4].isin(TOP10)].sort_values("true_iou")
    picks = [("Best of our top 10", sub.iloc[-1]), ("Worst of our top 10", sub.iloc[0])] if len(sub) else []
    for i, (lab, row) in enumerate(picks):
        y = 1.85 + i * 2.55
        picture(s, outputs / "Result2" / "Mooney" / row["file"], 7.45, y, w=4.4, h=2.2)
        text(s, 7.45, y + 2.22, 5.4, 0.3, [[(f"{lab}: ", {"bold": True}),
                                           (f"{row['label']}, IoU {row['true_iou']:.2f}, wrong-class score "
                                            f"{row['false_top_score']:.2f}", {})]], size=11, color=MUTED)
    text(s, 7.45, 6.85, 5.4, 0.3, "Left half: true-class prompt (green). Right half: wrong-class prompt (red). "
                                  "Yellow: COCO outline.", size=10, color=MUTED)
    s.notes_slide.notes_text_frame.text = (
        "IoU is computed over all masks scoring at least 0.5 against the COCO ground-truth mask; images where nothing "
        "is detected count as IoU 0. Overlays: left half true-class prompt, right half wrong-class prompt.")


def s_takeaways(prs, vo, vm, so, sm):
    s = new_slide(prs, dark=True)
    title(s, "Takeaways", color=PAPER)
    pts = [
        ("Qwen3-VL collapses on Mooneys. ",
         f"Choice accuracy falls from {pct(vo, 'pick_correct'):.0f} % to {pct(vm, 'pick_correct'):.0f} % "
         "(chance 10 %). It describes ink blobs, silhouettes and faces instead of the object."),
    ]
    if so is not None and sm is not None:
        pts.append(("SAM 3 when told what to look for: ",
                    f"finds the true class in {pct(sm, 'true_detected'):.0f} % of Mooneys "
                    f"(vs. {pct(so, 'true_detected'):.0f} % of photos), mean IoU {sm['true_iou'].mean():.2f} "
                    f"(vs. {so['true_iou'].mean():.2f}); hallucinates the wrong class in "
                    f"{pct(sm, 'false_detected'):.0f} %."))
    pts.append(("Hypothesis. ", "Both models learned from natural photos, where colour and texture carry much of "
                                "the signal. Mooneys keep only coarse shape and shading: humans fill the gap with "
                                "prior knowledge, while a model leaning on colour and texture loses its main cues."))
    pts.append(("Next. ", "Contrast-polarity and upside-down Mooneys, and larger VLMs on the same questions."))
    bullets(s, 0.6, 1.7, 12.0, 5.2, pts, size=20, color="E6E8EB")


def s_shot(prs, head, sub, path: Path | None):
    s = new_slide(prs)
    title(s, head, sub)
    box = (0.9, 1.85, 11.5, 5.2)
    if path and Path(path).exists():
        from PIL import Image
        iw, ih = Image.open(path).size
        scale = min(box[2] / iw, box[3] / ih)
        w, h = iw * scale, ih * scale
        p = picture(s, path, box[0] + (box[2] - w) / 2, box[1] + (box[3] - h) / 2, w=w, h=h)
        p.line.color.rgb, p.line.width = rgb("C9CDD2"), Pt(0.75)
    else:
        rect(s, *box, fill=None, line=MUTED, dash=True)
        text(s, *box, "Screenshot not added yet: re-run make_slides.py with this path", size=18, color=MUTED,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def s_ai(prs):
    s = new_slide(prs)
    title(s, "Declaration of AI usage")
    bullets(s, 0.6, 1.6, 12.0, 4.8, [
        ("Claude (Anthropic, via claude.ai): ", "project planning, code for dataset selection, the Mooney tool and "
                                                 "the vision-system scripts, debugging, result analysis, and "
                                                 "generating this slide deck."),
        ("Gemini (Google, chat): ", "a few informal spot checks on Mooney images; not part of the reported results."),
        ("My own work: ", "I ran, checked and adapted all code, chose the Mooney setting "
                          f"(σ = {BLUR_SIGMA} px, {WHITE_PCT} % white) and the top-10 images by eye, and reviewed "
                          "every result."),
        ("Not assistance: ", "Qwen3-VL and SAM 3 are the vision systems under test, not tools used to do the work."),
    ], size=17)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--tool-shot", type=Path, default=None, help="screenshot of the Mooney GUI")
    ap.add_argument("--code-shot", type=Path, default=None, help="screenshot of mooney_tool/core.py")
    ap.add_argument("--out", type=Path, default=Path("MooneyMinded_Part1.pptx"))
    args = ap.parse_args()

    vo, vm, so, sm = load(args.results)
    if vo is None or vm is None:
        raise SystemExit("results/vlm_Original.csv and results/vlm_Mooney.csv are required")
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    prs.core_properties.author = AUTHOR
    prs.core_properties.title = "Mooney-Minded: Project Part 1"

    s_title(prs, args.data)
    s_approach(prs, args.data)
    s_gallery(prs, args.data)
    s_declaration(prs)
    s_qwen(prs, vo, vm)
    s_qwen_sees(prs, args.data, vm)
    s_sam(prs, args.outputs, so, sm)
    s_takeaways(prs, vo, vm, so, sm)
    s_shot(prs, "Our Mooney tool", f"Gradio GUI: blur and white-% sliders, live preview, folder contact sheet and "
                                  f"batch export (also a CLI for 100+ images)", args.tool_shot)
    s_shot(prs, "Code that generates the Mooneys", "mooney_tool/core.py: Gaussian blur, then a percentile threshold "
                                                   "for an exact white share", args.code_shot)
    s_ai(prs)
    prs.save(args.out)
    print(f"saved {args.out} ({len(prs.slides)} slides)"
          + ("" if so is not None else "  [SAM 3 slide is a placeholder until results/sam3_*.csv exist]"))


if __name__ == "__main__":
    main()
