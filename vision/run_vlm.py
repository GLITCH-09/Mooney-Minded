"""Vision system 1: Qwen3-VL (pretrained, no fine-tuning) asked five scripted questions per image.

    python vision/run_vlm.py --images data/Original --out outputs/Result1/Original
    python vision/run_vlm.py --images data/Mooney   --out outputs/Result1/Mooney

Probes (greedy decoding, so runs are reproducible):
  describe     "Describe this image in one short sentence."   -> raw text (Rorschach slide)
  name         open answer, 1-3 words                         -> correct if it matches label synonyms
  pick         10-way multiple choice, shuffled per image     -> correct letter (chance 10 %)
  claim_true   "This is a photo of a <label>, right?"         -> yes-rate (should be high)
  claim_false  "This is a photo of a <wrong label>, right?"   -> yes-rate = how easily it is gaslit
The wrong label and option order are fixed per image stem, so original and Mooney get the same questions.

Writes one JSON per image into --out and a scored CSV into results/vlm_<tag>.csv.
Re-running skips images whose JSON already exists (safe to resume after a crash).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from common import (LETTERS, article, list_images, load_labels, name_correct, option_order,
                    parse_letter, parse_yes, wrong_label)

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def build_prompts(stem: str, label: str) -> dict[str, tuple[str, int]]:
    opts = option_order(stem)
    wrong = wrong_label(stem, label)
    menu = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(opts))
    return {
        "describe": ("Describe this image in one short sentence.", 40),
        "name": ("What is the main object in this image? Answer with 1 to 3 words only.", 12),
        "pick": (f"Which of these is shown in the image?\n{menu}\nAnswer with the letter only.", 4),
        "claim_true": (f"This is a photo of {article(label)}, right? Answer yes or no.", 4),
        "claim_false": (f"This is a photo of {article(wrong)}, right? Answer yes or no.", 4),
    }


def score(stem: str, label: str, answers: dict[str, str]) -> dict:
    opts = option_order(stem)
    letter = parse_letter(answers["pick"])
    return {
        "name_correct": int(name_correct(answers["name"], label)),
        "pick_correct": int(letter is not None and opts[LETTERS.index(letter)] == label),
        "pick_answer": opts[LETTERS.index(letter)] if letter else "",
        "yes_true": {True: 1, False: 0, None: ""}[parse_yes(answers["claim_true"])],
        "yes_false": {True: 1, False: 0, None: ""}[parse_yes(answers["claim_false"])],
    }


def load_model(model_id: str):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)

    def ask(image_path: Path, text: str, max_new: int) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": text},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    return ask


def run(ask, images: Path, labels_csv: Path, out: Path, tag: str, results_dir: Path) -> list[dict]:
    labels = load_labels(labels_csv)
    out.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    paths = list_images(images)
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        stem = p.stem
        label = labels[stem]
        jpath = out / f"{stem}.json"
        if jpath.exists():
            rec = json.loads(jpath.read_text())
        else:
            prompts = build_prompts(stem, label)
            answers = {k: ask(p, txt, n) for k, (txt, n) in prompts.items()}
            rec = {"file": p.name, "label": label, "wrong_label": wrong_label(stem, label),
                   "prompts": {k: v[0] for k, v in prompts.items()}, "answers": answers,
                   "scores": score(stem, label, answers)}
            jpath.write_text(json.dumps(rec, indent=2))
        rows.append({"file": rec["file"], "label": rec["label"], "wrong_label": rec["wrong_label"],
                     **rec["scores"],
                     **{f"ans_{k}": v.replace("\n", " ") for k, v in rec["answers"].items()}})
        if i % 10 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)}  ({time.time() - t0:.0f}s)")

    with open(results_dir / f"vlm_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def rate(key):
        vals = [r[key] for r in rows if r[key] != ""]
        return 100 * sum(vals) / max(len(vals), 1)

    print(f"[{tag}] n={len(rows)}  name {rate('name_correct'):.0f}%  pick {rate('pick_correct'):.0f}%  "
          f"yes-to-true {rate('yes_true'):.0f}%  yes-to-wrong (gaslit) {rate('yes_false'):.0f}%")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    ap.add_argument("--tag", default=None, help="name for results/vlm_<tag>.csv (default: input folder name)")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()
    ask = load_model(args.model)
    run(ask, args.images, args.labels, args.out, args.tag or args.images.name, args.results)


if __name__ == "__main__":
    main()
