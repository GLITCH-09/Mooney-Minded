"""System 1 (local): Qwen3-VL-4B-Instruct answers five fixed prompts per image.

    python -m vision.qwen --split Original
    python -m vision.qwen --split Mooney
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import MAX_NEW_TOKENS, SPLITS, answer_row, list_images, load_labels, prompts, rate, write_csv, wrong_label

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def load(model_id: str):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    processor = AutoProcessor.from_pretrained(model_id)

    def ask(image: Path, text: str, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": str(image)},
                                                 {"type": "text", "text": text}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    return ask


def run(ask, split: str, data: Path, outputs: Path, results: Path) -> None:
    labels = load_labels(data / "labels.csv")
    out_dir = outputs / "Result1" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for image in list_images(data / split):
        stem, label = image.stem, labels[image.stem]
        cache = out_dir / f"{stem}.json"
        if cache.exists():
            record = json.loads(cache.read_text())
        else:
            asked = prompts(stem, label)
            record = {"file": image.name, "label": label, "wrong_label": wrong_label(stem, label),
                      "model": MODEL_ID, "prompts": asked,
                      "answers": {p: ask(image, text, MAX_NEW_TOKENS[p]) for p, text in asked.items()}}
            cache.write_text(json.dumps(record, indent=2))
        rows.append(answer_row(record["file"], label, record["wrong_label"], record["answers"]))
    write_csv(results / f"qwen3vl4b_{split}.csv", rows)
    print(f"[{split}] n={len(rows)}  named {rate(rows, 'name_correct')}  picked {rate(rows, 'pick_correct')}  "
          f"yes-to-true {rate(rows, 'yes_true')}  yes-to-wrong {rate(rows, 'yes_false')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()
    run(load(args.model), args.split, args.data, args.outputs, args.results)


if __name__ == "__main__":
    main()
