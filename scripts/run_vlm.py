"""Vision system 1: a vision-language model (Qwen3-VL) answering four questions per image.

    python -m scripts.run_vlm --data data --out Result1 --limit 3     # smoke test
    python -m scripts.run_vlm --data data --out Result1               # everything

For every image, in its original form and as a Mooney, the model gets:
    name           "What is the main object?"            free text, checked against the label
    pick           the same question as 10-way choice     chance level is 10%
    confirm_true   "This is a photo of <true label>, right?"   baseline yes-rate
    confirm_false  "This is a photo of <wrong label>, right?"  how easily it is talked into a lie
The wrong label and the option order depend only on the image id, so the original
and its Mooney always get exactly the same questions.

Results: <out>/<split>/<id>.json. Existing results are skipped, so an interrupted
run can simply be started again.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Callable

from .common import find_image, name_matches, read_labels, with_article, write_json

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

Ask = Callable[["object", str], str]  # (PIL image, prompt) -> answer text


def wrong_category(image_id: str, category: str, categories: list[str]) -> str:
    return random.Random(f"wrong:{image_id}").choice([c for c in categories if c != category])


def option_order(image_id: str, categories: list[str]) -> list[str]:
    options = list(categories)
    random.Random(f"order:{image_id}").shuffle(options)
    return options


def build_prompts(image_id: str, category: str, categories: list[str]) -> tuple[dict[str, str], str]:
    wrong = wrong_category(image_id, category, categories)
    options = ", ".join(option_order(image_id, categories))
    prompts = {
        "name": "What is the main object in this image? Answer with one to three words.",
        "pick": f"Which of these best describes the main object in this image: {options}? "
        "Answer with exactly one option from the list.",
        "confirm_true": f"This is a photo of {with_article(category)}, right? Answer yes or no.",
        "confirm_false": f"This is a photo of {with_article(wrong)}, right? Answer yes or no.",
    }
    return prompts, wrong


def parse_choice(text: str, categories: list[str]) -> str | None:
    """The option that appears first in the answer (whole words, plural allowed)."""
    best: tuple[int, str] | None = None
    for category in categories:
        match = re.search(rf"\b{re.escape(category)}s?\b", text.lower())
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), category)
    return best[1] if best else None


def parse_yes_no(text: str) -> bool | None:
    words = re.sub(r"[^a-z ]", " ", text.lower()).split()
    if words and words[0] in ("yes", "no"):
        return words[0] == "yes"
    return None


def run_image(image, row: dict, categories: list[str], ask: Ask) -> dict:
    category = row["category"]
    prompts, wrong = build_prompts(row["id"], category, categories)
    answers = {key: ask(image, prompt).strip() for key, prompt in prompts.items()}
    choice = parse_choice(answers["pick"], categories)
    return {
        "id": row["id"],
        "category": category,
        "name": {"prompt": prompts["name"], "answer": answers["name"], "correct": name_matches(answers["name"], category)},
        "pick": {"prompt": prompts["pick"], "answer": answers["pick"], "choice": choice, "correct": choice == category},
        "confirm_true": {
            "prompt": prompts["confirm_true"],
            "answer": answers["confirm_true"],
            "yes": parse_yes_no(answers["confirm_true"]),
        },
        "confirm_false": {
            "claimed": wrong,
            "prompt": prompts["confirm_false"],
            "answer": answers["confirm_false"],
            "yes": parse_yes_no(answers["confirm_false"]),
        },
    }


def process(
    split: str,
    image_dir: Path,
    out_dir: Path,
    labels: list[dict],
    ask: Ask,
    open_image: Callable[[Path], object],
    limit: int | None = None,
    log: Callable[[str], None] = print,
) -> int:
    """Run all questions on every image of one split. Returns the number of images processed now."""
    categories = list(dict.fromkeys(row["category"] for row in labels))
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for row in labels[:limit]:
        target = out_dir / f"{row['id']}.json"
        if target.exists():
            continue
        path = find_image(image_dir, row["id"])
        if path is None:
            raise FileNotFoundError(f"no image for id {row['id']} in {image_dir}")
        record = run_image(open_image(path), row, categories, ask)
        record["split"] = split
        write_json(target, record)
        done += 1
        log(f"[{split}] {row['id']} {row['category']:<11} name: {record['name']['answer']!r}")
    return done


# --- the part that needs a GPU and the model weights -------------------------------------------


def load_model(model_id: str, four_bit: bool):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    kwargs: dict = {"device_map": "auto"}
    if four_bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        kwargs["dtype"] = torch.bfloat16
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs).eval()
    return model, AutoProcessor.from_pretrained(model_id)


def make_ask(model, processor, max_new_tokens: int = 16) -> Ask:
    import torch

    def ask(image, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        reply = generated[:, inputs["input_ids"].shape[1] :]
        return processor.batch_decode(reply, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return ask


def open_rgb(path: Path):
    from PIL import Image

    return Image.open(path).convert("RGB")  # Mooneys are single-channel PNGs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL on the original and Mooney images.")
    parser.add_argument("--data", type=Path, default=Path("data"), help="folder with labels.csv, Original/ and Mooney/")
    parser.add_argument("--out", type=Path, default=Path("Result1"))
    parser.add_argument("--splits", nargs="+", default=["Original", "Mooney"])
    parser.add_argument("--limit", type=int, help="only the first N images (smoke test)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-4bit", action="store_true", help="load in bfloat16 instead of 4-bit (needs ~17 GB for the 8B model)")
    args = parser.parse_args(argv)

    labels = read_labels(args.data / "labels.csv")
    for split in args.splits:
        if not (args.data / split).is_dir():
            print(f"missing folder {args.data / split}", file=sys.stderr)
            return 1

    model, processor = load_model(args.model, four_bit=not args.no_4bit)
    ask = make_ask(model, processor)
    for split in args.splits:
        n = process(split, args.data / split, args.out / split, labels, ask, open_rgb, args.limit)
        print(f"{split}: {n} new result(s) in {args.out / split}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
