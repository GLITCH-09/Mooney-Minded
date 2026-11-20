"""Slide figures from the results tables (and images, when present).

    python -m scripts.figures            # -> figures/*.png
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vision.common import CLASSES, TOP10, label_of, name_correct, read_csv

BG, INK, MUTED, GRID, PALE = "#f7f6f3", "#181818", "#6f6f6b", "#e4e2dc", "#c9c6bd"
BLUE, ORANGE = "#2a78d6", "#e55235"
PANEL, HEADER = 512, 44
THEMES = {
    "calls it abstract": r"\babstract",
    "sees a figure, person or face": r"\b(person|people|human|man|woman|face|figure|portrait|head)\b",
    "calls it a silhouette": r"silhouette",
    "sees ink, paint or splatter": r"\b(ink|inkblot|splatter\w*|blot|paint|brush\w*|scribble)",
    "sees text, a logo or symbol":
        r"\b(text|letter\w*|signature|logo|emblem|symbol|word|monogram|font)\b|chinese character",
    "sees a map or landscape": r"\b(map|state of|mountain\w*|landscape|rocky|peninsula)\b",
}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.family": "DejaVu Sans", "font.size": 13, "text.color": INK,
    "axes.edgecolor": PALE, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 15,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 14,
    "legend.frameon": False,
})


def num(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows if r[key] != ""])


def top10(rows: list[dict]) -> list[dict]:
    by_stem = {Path(r["file"]).stem: r for r in rows}
    return [by_stem[s] for s in TOP10 if s in by_stem]


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  {name}")


def grid_axes(rows: int, cols: int, size: tuple[float, float]):
    fig, axes = plt.subplots(rows, cols, figsize=size, gridspec_kw={"wspace": 0.04, "hspace": 0.06})
    for ax in np.ravel(axes):
        ax.set_axis_off()
    return fig, np.atleast_2d(axes)


def caption(ax, text: str) -> None:
    ax.text(0.5, -0.05, text, transform=ax.transAxes, ha="center", va="top", fontsize=12, color=MUTED)


def gallery(data: Path, out: Path) -> None:
    fig, axes = grid_axes(2, 10, (16, 3.9))
    for c, stem in enumerate(TOP10):
        for r, split in enumerate(("Original", "Mooney")):
            img = cv2.imread(str(data / split / f"{stem}.png"), cv2.IMREAD_COLOR)
            axes[r, c].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        caption(axes[1, c], label_of(stem))
    save(fig, out, "gallery_top10.png")


def crop(path: Path, index: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img[HEADER:, index * PANEL:(index + 1) * PANEL], cv2.COLOR_BGR2RGB)


def sam_grid(outputs: Path, out: Path) -> None:
    rows = [("Photo\ntext prompt", "Original", 0), ("Mooney\ntext prompt", "Mooney", 0),
            ("Mooney\nbox prompt", "Mooney", 1)]
    fig, axes = grid_axes(3, 10, (16, 5.6))
    for r, (title, split, idx) in enumerate(rows):
        for c, stem in enumerate(TOP10):
            axes[r, c].imshow(crop(outputs / "Result2" / split / f"{stem}.png", idx))
        axes[r, 0].text(-0.08, 0.5, title, transform=axes[r, 0].transAxes, ha="right", va="center",
                        fontsize=12, color=INK, fontweight="bold")
    for c, stem in enumerate(TOP10):
        caption(axes[2, c], label_of(stem))
    save(fig, out, "sam_top10.png")


def bar_labels(ax, bars, fmt) -> None:
    for b in bars:
        ax.annotate(fmt(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=12, color=INK)


def scale(res: Path, out: Path) -> None:
    small = top10(read_csv(res / "qwen3vl4b_Mooney.csv"))
    big = top10(read_csv(res / "qwen4max_Mooney.csv"))
    groups = [(name, key) for name, key in (("names it", "name_correct"), ("picks it from 10", "pick_correct"),
                                            ("says yes to\nthe true class", "yes_true"),
                                            ("says yes to a wrong\nclass (lower = better)", "yes_false"))
              if len(num(big, key))]
    x, w = np.arange(len(groups)), 0.36
    fig, ax = plt.subplots(figsize=(8, 4.6))
    series = ((-w / 2, BLUE, "Qwen3-VL-4B (local)", small), (w / 2, ORANGE, "Qwen4-Max (web)", big))
    for shift, color, name, data in series:
        vals = [num(data, k).sum() for _, k in groups]
        bars = ax.bar(x + shift, vals, w * 0.92, color=color, label=name, zorder=3)
        bar_labels(ax, bars, lambda v: f"{v:.0f}/10")
    ax.set_xticks(x, [g[0] for g in groups], color=INK)
    ax.set_ylim(0, 11.5)
    ax.set_yticks([0, 5, 10])
    ax.set_ylabel("top-10 Mooneys")
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.legend(loc="upper right", ncol=2, bbox_to_anchor=(1, 1.13))
    ax.set_title("Same 10 Mooneys, same prompts: small vs. flagship Qwen", pad=30)
    save(fig, out, "qwen_scale.png")


def names(res: Path, out: Path) -> None:
    models = [("Qwen3-VL-4B", top10(read_csv(res / "qwen3vl4b_Mooney.csv"))),
              ("Qwen4-Max", top10(read_csv(res / "qwen4max_Mooney.csv")))]
    fig, ax = plt.subplots(figsize=(9.6, 7))
    ax.set_axis_off()
    ax.set_xlim(0, 2.9)
    ax.set_ylim(len(TOP10) + 0.8, -1.2)
    for x, head in zip((0, 0.95, 2.08), ["Mooney of a …"] + [m[0] for m in models], strict=True):
        ax.text(x, -0.75, head, fontsize=19, fontweight="bold", color=MUTED, va="center")
    for i, stem in enumerate(TOP10):
        ax.axhline(i - 0.5, color=GRID, lw=1)
        ax.text(0, i, label_of(stem), fontsize=20, fontweight="bold", color=INK, va="center")
        for x, (_, rows) in zip((0.95, 2.08), models, strict=True):
            hit = rows[i]["name_correct"] == "1"
            answer = rows[i]["ans_name"].strip().rstrip(".")
            ax.text(x, i, answer[:1].upper() + answer[1:] + (" ✓" if hit else ""), fontsize=20, va="center",
                    color=ORANGE if hit else INK, fontweight="bold" if hit else "normal")
    ax.axhline(len(TOP10) - 0.5, color=PALE, lw=1.5)
    for x, (_, rows) in zip((0.95, 2.08), models, strict=True):
        ax.text(x, len(TOP10) + 0.25, f"{sum(r['name_correct'] == '1' for r in rows)}/10 named",
                fontsize=20, fontweight="bold", color=INK, va="center")
    save(fig, out, "qwen_names.png")


def sees(res: Path, out: Path) -> None:
    rows = read_csv(res / "qwen3vl4b_Mooney.csv")
    texts = [r["ans_describe"].lower() for r in rows]
    counts = {k: sum(bool(re.search(p, t)) for t in texts) for k, p in THEMES.items()}
    counts["names the right object"] = sum(name_correct(t, r["label"]) for t, r in zip(texts, rows, strict=True))
    items = sorted(counts.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    colors = [ORANGE if k == "names the right object" else BLUE for k, _ in items]
    bars = ax.barh([k for k, _ in items], [v for _, v in items], color=colors, height=0.62, zorder=3)
    for b in bars:
        ax.annotate(f"{b.get_width():.0f}", (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=12, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"of {len(rows)} Mooney descriptions")
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.set_title("What Qwen3-VL-4B says it sees in a Mooney")
    save(fig, out, "qwen_sees.png")


def toilet(res: Path, out: Path) -> None:
    rows = read_csv(res / "qwen3vl4b_Mooney.csv")
    counts = Counter(r["pick_answer"] for r in rows if r["pick_answer"])
    order = sorted(CLASSES, key=lambda c: counts.get(c, 0))
    vals = [counts.get(c, 0) for c in order]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(order, vals, color=[ORANGE if v == max(vals) else BLUE for v in vals], height=0.62, zorder=3)
    for b in bars:
        ax.annotate(f"{b.get_width():.0f}", (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=12, color=INK)
    ax.axvline(10, color=MUTED, lw=1, zorder=2)
    ax.annotate("each class is really 10", (10, len(order) - 0.45), xytext=(5, 0), textcoords="offset points",
                fontsize=11, color=MUTED)
    ax.set_xlabel(f"times picked, {len(rows)} Mooneys")
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.set_title("Forced to pick 1 of 10 classes, Qwen3-VL-4B answers ...")
    save(fig, out, "qwen_toilet.png")


def strips(ax, series: list[tuple[str, np.ndarray, str]], xlabel: str, note) -> None:
    rng = np.random.default_rng(0)
    for i, (_, vals, color) in enumerate(series):
        y = len(series) - 1 - i
        ax.scatter(vals, y + rng.uniform(-0.18, 0.18, len(vals)), s=26, color=color, alpha=0.75,
                   edgecolors=BG, linewidths=0.6, zorder=3)
        ax.plot([np.median(vals)] * 2, [y - 0.32, y + 0.32], color=INK, lw=2, zorder=4)
        ax.annotate(note(vals), (1.02, y), xycoords=("axes fraction", "data"), va="center", fontsize=12, color=INK)
    ax.set_yticks(range(len(series)), [s[0] for s in reversed(series)], color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, len(series) - 0.4)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, zorder=0)


def sam_scores(res: Path, out: Path) -> None:
    o, m = read_csv(res / "sam3_Original.csv"), read_csv(res / "sam3_Mooney.csv")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    strips(ax, [("Photo", num(o, "text_score"), BLUE), ("Mooney", num(m, "text_score"), ORANGE)],
           "SAM 3 confidence for the true class name (one dot per image)",
           lambda v: f"found {int((v >= 0.5).sum())}/{len(v)}")
    ax.axvline(0.5, color=MUTED, lw=1, zorder=2)
    ax.annotate("detection threshold", (0.5, 1.45), xytext=(5, 0), textcoords="offset points", fontsize=11, color=MUTED)
    ax.set_title("Told the class name, SAM 3 stops finding the object")
    save(fig, out, "sam_scores.png")


def box_fill(data: Path) -> dict[str, float]:
    """IoU of the prompt box itself with the true mask: the score of answering 'the whole box'."""
    fills = {}
    for path in sorted((data / "masks").glob("*.png")):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 127
        ys, xs = np.nonzero(mask)
        fills[path.stem] = float(mask.sum() / ((np.ptp(ys) + 1) * (np.ptp(xs) + 1)))
    return fills


def sam_iou(res: Path, data: Path, out: Path) -> None:
    o, m = read_csv(res / "sam3_Original.csv"), read_csv(res / "sam3_Mooney.csv")
    series = [("Photo · class name", num(o, "text_best_iou"), BLUE),
              ("Mooney · class name", num(m, "text_best_iou"), ORANGE),
              ("Photo · box", num(o, "box_iou"), BLUE),
              ("Mooney · box", num(m, "box_iou"), ORANGE)]
    fills = box_fill(data) if (data / "masks").is_dir() else {}
    if fills:
        series.append(("the box alone", np.array([fills[Path(r["file"]).stem] for r in m]), MUTED))
        beats = sum(float(r["box_iou"]) > fills[Path(r["file"]).stem] for r in m)
        print(f"  box alone (no SAM): median IoU {np.median(list(fills.values())):.2f}; "
              f"SAM 3 with the box beats it on {beats}/{len(m)} Mooneys")
    fig, ax = plt.subplots(figsize=(8, 1.1 * len(series) + 0.8))
    strips(ax, series, "overlap with the true outline (IoU, SAM 3's best mask)", lambda v: f"median {np.median(v):.2f}")
    ax.set_title("How well SAM 3's best mask matches the true outline")
    save(fig, out, "sam_iou.png")


def scoreboard(data: Path, res: Path, out: Path) -> None:
    small = top10(read_csv(res / "qwen3vl4b_Mooney.csv"))
    sam = top10(read_csv(res / "sam3_Mooney.csv"))
    cols = [("Qwen3-VL-4B\nnames it", [r["name_correct"] == "1" for r in small])]
    big_path = res / "qwen4max_Mooney.csv"
    if big_path.exists():
        big = top10(read_csv(big_path))
        cols.append(("Qwen4-Max\nnames it", [r["name_correct"] == "1" for r in big]))
        if any(r["pick_correct"] != "" for r in big):
            cols.append(("Qwen4-Max\npicks it of 10", [r["pick_correct"] == "1" for r in big]))
    cols += [("SAM 3 finds it\n(class name)", [r["text_found"] == "1" for r in sam]),
             ("SAM 3 outlines it\n(box, IoU ≥ 0.5)", [float(r["box_iou"]) >= 0.5 for r in sam])]
    thumbs = (data / "Mooney").is_dir()
    fig, ax = plt.subplots(figsize=(3.2 + 1.9 * len(cols), 7.6))
    ax.set_xlim(-1.7 if thumbs else -1.1, len(cols) - 0.5)
    ax.set_ylim(len(TOP10) + 0.2, -1.1)
    ax.set_axis_off()
    for j, (name, hits) in enumerate(cols):
        ax.text(j, -0.8, name, ha="center", va="center", fontsize=11, color=INK, fontweight="bold", linespacing=1.3)
        for i, hit in enumerate(hits):
            ax.scatter(j, i, s=360 if hit else 230, color=ORANGE if hit else BG,
                       edgecolors=ORANGE if hit else PALE, linewidths=2, zorder=3)
        ax.text(j, len(TOP10) - 0.05, f"{sum(hits)}/10", ha="center", va="center", fontsize=13, color=INK,
                fontweight="bold")
    for i, stem in enumerate(TOP10):
        ax.axhline(i + 0.5, color=GRID, lw=1, zorder=1)
        if thumbs:
            img = cv2.imread(str(data / "Mooney" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            ax.imshow(img, cmap="gray", extent=(-1.66, -1.38, i + 0.42, i - 0.42), aspect="auto", zorder=2)
        ax.text(-1.28 if thumbs else -1.05, i, label_of(stem), ha="left", va="center", fontsize=12, color=INK)
    save(fig, out, "scoreboard.png")


def gaslight(res: Path, out: Path) -> None:
    panels = [("Qwen3-VL-4B (100 images)", {s: read_csv(res / f"qwen3vl4b_{s}.csv") for s in ("Original", "Mooney")})]
    big = res / "qwen4max_Mooney.csv"
    if big.exists() and len(num(read_csv(big), "yes_false")):
        panels.append(("Qwen4-Max (top 10)", {"Mooney": read_csv(big)}))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels) + 1, 4.4), sharey=True, squeeze=False)
    for ax, (title, data) in zip(axes[0], panels, strict=True):
        splits = list(data)
        x, w = np.arange(len(splits)), 0.36
        for shift, color, key, name in ((-w / 2, BLUE, "yes_true", "yes to the true class"),
                                        (w / 2, ORANGE, "yes_false", "yes to a wrong class")):
            vals = [100 * num(data[s], key).mean() for s in splits]
            bars = ax.bar(x + shift, vals, w * 0.92, color=color, label=name, zorder=3)
            bar_labels(ax, bars, lambda v: f"{v:.0f}%")
        ax.set_xticks(x, ["Photo" if s == "Original" else s for s in splits], color=INK)
        ax.set_ylim(0, 112)
        ax.grid(axis="y", color=GRID, zorder=0)
        ax.set_title(title, fontsize=13)
    axes[0, 0].set_ylabel("answers 'yes' (%)")
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(0, 1.32), ncol=2)
    save(fig, out, "gaslight.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    have = lambda *names: all((args.results / n).exists() for n in names)
    jobs = [
        ((args.data / "Mooney").is_dir(), lambda: gallery(args.data, args.out)),
        ((args.outputs / "Result2" / "Mooney").is_dir(), lambda: sam_grid(args.outputs, args.out)),
        (have("qwen3vl4b_Mooney.csv"), lambda: sees(args.results, args.out)),
        (have("qwen3vl4b_Mooney.csv"), lambda: toilet(args.results, args.out)),
        (have("qwen3vl4b_Original.csv", "qwen3vl4b_Mooney.csv"), lambda: gaslight(args.results, args.out)),
        (have("qwen3vl4b_Mooney.csv", "qwen4max_Mooney.csv"), lambda: scale(args.results, args.out)),
        (have("qwen3vl4b_Mooney.csv", "qwen4max_Mooney.csv"), lambda: names(args.results, args.out)),
        (have("sam3_Original.csv", "sam3_Mooney.csv"), lambda: sam_scores(args.results, args.out)),
        (have("sam3_Original.csv", "sam3_Mooney.csv"), lambda: sam_iou(args.results, args.data, args.out)),
        (have("qwen3vl4b_Mooney.csv", "sam3_Mooney.csv"), lambda: scoreboard(args.data, args.results, args.out)),
    ]
    print(f"figures -> {args.out}")
    for ready, job in jobs:
        if ready:
            job()


if __name__ == "__main__":
    main()
