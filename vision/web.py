"""System 1 (web): Qwen4-Max on chat.qwen.ai for the top 10, one fresh chat per prompt.

    python -m vision.web blind   # data/blind/O01.png ... M10.png, neutral names so nothing leaks the label
    python -m vision.web sheet   # results/qwen4max_web.txt, every prompt pre-written, answers pasted by hand
    python -m vision.web score   # filled sheet -> results/qwen4max_{Original,Mooney}.csv
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from .common import TOP10, answer_row, label_of, prompts, rate, write_csv, wrong_label

SHEET = Path("results/qwen4max_web.txt")
HEADER = re.compile(r"^=== ([OM])(\d{2})\b(?:\s+(\w+))?")
SPLIT_OF = {"O": "Original", "M": "Mooney"}
SECTIONS = {
    ("O", "name"): "A. REQUIRED: originals, name only",
    ("M", "describe"): "B. REQUIRED: Mooneys, describe and name in separate chats",
    ("M", "claim_true"): "C. ADD-ON (gaslighting): Mooneys, two yes/no claims",
    ("M", "pick"): "D. OPTIONAL: Mooneys, pick 1 of 10",
}
ORDER = [("O", ("name",)), ("M", ("describe", "name")), ("M", ("claim_true", "claim_false")), ("M", ("pick",))]


def blind(data: Path) -> None:
    out = data / "blind"
    out.mkdir(parents=True, exist_ok=True)
    for i, stem in enumerate(TOP10, 1):
        for prefix, split in SPLIT_OF.items():
            shutil.copy2(data / split / f"{stem}.png", out / f"{prefix}{i:02d}.png")
    print(f"20 blind copies -> {out}")


def sheet(path: Path) -> None:
    lines = ["# Qwen4-Max web run. One NEW chat per block, upload only the file in the header (data/blind/).",
             "# Copy the prompt exactly, paste the reply under ANSWER:.",
             "# MODEL_SHOWN_ON_SITE: ", "# THINKING_MODE: off", "# DATE: ", ""]
    for prefix, probes in ORDER:
        lines += [f"# ---------- {SECTIONS[(prefix, probes[0])]} ----------", ""]
        for i, stem in enumerate(TOP10, 1):
            asked = prompts(stem, label_of(stem))
            for probe in probes:
                lines += [f"=== {prefix}{i:02d} {probe} | upload {prefix}{i:02d}.png",
                          f"PROMPT: {asked[probe]}", "ANSWER:", "", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"sheet -> {path}")


def probe_of(key: tuple[str, int], prompt: str) -> str:
    """Recover the probe from the prompt when a header lost its name."""
    if prompt.startswith("Describe"):
        return "describe"
    if prompt.startswith("Which of these"):
        return "pick"
    if prompt.startswith("This is a photo of"):
        return "claim_true" if label_of(TOP10[key[1] - 1]) in prompt else "claim_false"
    return "name"


def parse(text: str) -> tuple[dict[tuple[str, int], dict[str, str]], dict[str, str]]:
    answers: dict[tuple[str, int], dict[str, str]] = {}
    meta = dict(re.findall(r"^# ([A-Z_]+):[ \t]*(.*)$", text, flags=re.M))
    key, probe, prompt, body, state = None, None, [], [], None

    def flush() -> None:
        if key:
            name = probe or probe_of(key, " ".join(prompt).strip())
            answers.setdefault(key, {})[name] = "\n".join(body).strip()

    for line in text.splitlines():
        m = HEADER.match(line)
        if m:
            flush()
            key, probe, prompt, body, state = (m.group(1), int(m.group(2))), m.group(3), [], [], None
        elif line.startswith("# ----------"):
            flush()
            key, state = None, None
        elif line.startswith("PROMPT:"):
            state, prompt = "prompt", [line[len("PROMPT:"):].strip()]
        elif line.startswith("ANSWER:"):
            state, body = "answer", [line[len("ANSWER:"):]]
        elif state == "prompt":
            prompt.append(line)
        elif state == "answer":
            body.append(line)
    flush()
    return answers, meta


def score(path: Path, results: Path) -> None:
    answers, meta = parse(path.read_text())
    print(f"model: {meta.get('MODEL_SHOWN_ON_SITE') or '?'}  thinking: {meta.get('THINKING_MODE') or '?'}")
    for prefix, split in SPLIT_OF.items():
        rows = []
        for i, stem in enumerate(TOP10, 1):
            got = {p: a for p, a in answers.get((prefix, i), {}).items() if a}
            label = label_of(stem)
            rows.append(answer_row(f"{stem}.png", label, wrong_label(stem, label), got))
        missing = [r["file"] for r in rows if r["ans_name"] == ""]
        write_csv(results / f"qwen4max_{split}.csv", rows)
        print(f"[{split}] named {rate(rows, 'name_correct')}  picked {rate(rows, 'pick_correct')}  "
              f"yes-to-true {rate(rows, 'yes_true')}  yes-to-wrong {rate(rows, 'yes_false')}"
              + (f"  MISSING name: {', '.join(missing)}" if missing else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["blind", "sheet", "score"])
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--sheet", type=Path, default=SHEET)
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    if args.command == "blind":
        blind(args.data)
    elif args.command == "sheet":
        sheet(args.sheet)
    else:
        score(args.sheet, args.results)


if __name__ == "__main__":
    main()
