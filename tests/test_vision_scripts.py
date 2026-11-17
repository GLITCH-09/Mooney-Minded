import csv
import json

import cv2
import numpy as np
import pytest

from scripts import run_sam, run_vlm, score
from scripts.common import DEFAULT_CATEGORIES, LABEL_FIELDS, find_image, iou, name_matches, read_labels

CATS = DEFAULT_CATEGORIES


# ---------------------------------------------------------------- shared helpers


@pytest.mark.parametrize(
    "answer, category, expected",
    [
        ("A tabby cat.", "cat", True),
        ("Two cats", "cat", True),
        ("category", "cat", False),  # whole words only
        ("golden retriever", "dog", True),
        ("A passenger jet", "airplane", True),
        ("a stuffed teddy bear", "teddy bear", True),
        ("bear", "teddy bear", False),
        ("bicycle", "motorcycle", False),
        ("I can't tell", "zebra", False),
    ],
)
def test_name_matches(answer, category, expected):
    assert name_matches(answer, category) is expected


def test_iou():
    a = np.zeros((10, 10), np.uint8); a[:, :5] = 255
    b = np.zeros((10, 10), np.uint8); b[:, 2:7] = 1
    assert iou(a, b) == pytest.approx(3 / 7)
    assert iou(a, a) == 1.0
    assert iou(np.zeros((4, 4)), np.zeros((4, 4))) == 0.0


def test_find_image_any_extension(tmp_path):
    (tmp_path / "007.png").write_bytes(b"x")
    assert find_image(tmp_path, "007").name == "007.png"
    assert find_image(tmp_path, "008") is None


# ---------------------------------------------------------------- VLM: prompts and parsing


def test_prompts_are_stable_per_image_and_use_a_wrong_label():
    prompts, wrong = run_vlm.build_prompts("012", "cat", CATS)
    again, wrong_again = run_vlm.build_prompts("012", "cat", CATS)
    assert prompts == again and wrong == wrong_again
    assert wrong != "cat" and wrong in CATS
    assert "a cat, right?" in prompts["confirm_true"]
    assert f"{run_vlm.with_article(wrong)}, right?" in prompts["confirm_false"]
    assert all(c in prompts["pick"] for c in CATS)
    assert "an elephant, right?" in run_vlm.build_prompts("013", "elephant", CATS)[0]["confirm_true"]


def test_options_and_wrong_labels_vary_between_images():
    orders = {tuple(run_vlm.option_order(f"{i:03d}", CATS)) for i in range(20)}
    wrongs = {run_vlm.wrong_category(f"{i:03d}", "cat", CATS) for i in range(40)}
    assert len(orders) > 5 and len(wrongs) > 3
    assert all(sorted(o) == sorted(CATS) for o in orders)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("cat", "cat"),
        ("Teddy bear.", "teddy bear"),
        ("It looks like a horse, maybe a zebra", "horse"),
        ("zebras", "zebra"),
        ("motorcycle", "motorcycle"),
        ("a bicycle", None),
        ("", None),
    ],
)
def test_parse_choice(text, expected):
    assert run_vlm.parse_choice(text, CATS) == expected


@pytest.mark.parametrize(
    "text, expected",
    [("Yes", True), ("yes.", True), ("No, it is a dog", False), (" no ", False), ("Maybe", None), ("", None), ("Not sure", None)],
)
def test_parse_yes_no(text, expected):
    assert run_vlm.parse_yes_no(text) is expected


# ---------------------------------------------------------------- fixtures: a tiny dataset


@pytest.fixture
def dataset(tmp_path):
    """4 images (2 cat, 2 dog), 60x80, with a ground-truth mask covering the left half."""
    data = tmp_path / "data"
    for sub in ("Original", "Mooney", "gt/masks"):
        (data / sub).mkdir(parents=True)
    rows = []
    for n, category in enumerate(["cat", "cat", "dog", "dog"], start=1):
        name = f"{n:03d}"
        cv2.imwrite(str(data / "Original" / f"{name}.jpg"), np.full((60, 80, 3), 120, np.uint8))
        mooney = np.zeros((60, 80), np.uint8); mooney[:, 40:] = 255
        cv2.imwrite(str(data / "Mooney" / f"{name}.png"), mooney)
        gt = np.zeros((60, 80), np.uint8); gt[:, :40] = 255
        cv2.imwrite(str(data / "gt" / "masks" / f"{name}.png"), gt)
        rows.append({k: "" for k in LABEL_FIELDS} | {"id": name, "file": f"{name}.jpg", "category": category, "width": 80, "height": 60})
    with (data / "labels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return data


def fake_open(path):
    return str(path)  # the "image" is just its path; the fake models decide by split


def sighted_ask(image, prompt):
    """Knows the answer on originals, is lost on Mooneys, and agrees with everything there."""
    category = {"001": "cat", "002": "cat", "003": "dog", "004": "dog"}[image.split("/")[-1][:3]]
    if "Original" in image:
        if prompt.startswith("What is"):
            return f"A {category}"
        if prompt.startswith("Which of"):
            return category
        return "Yes" if f"a {category}, right" in prompt else "No"
    if prompt.startswith("What is"):
        return "Some kind of shape"
    if prompt.startswith("Which of"):
        return "I am not sure"
    return "Yes"


# ---------------------------------------------------------------- VLM: processing


def test_vlm_process_writes_results_and_resumes(dataset, tmp_path):
    labels = read_labels(dataset / "labels.csv")
    out = tmp_path / "Result1" / "Original"
    messages = []
    done = run_vlm.process("Original", dataset / "Original", out, labels, sighted_ask, fake_open, log=messages.append)
    assert done == 4 and sorted(p.name for p in out.iterdir()) == ["001.json", "002.json", "003.json", "004.json"]

    record = json.loads((out / "003.json").read_text())
    assert record["category"] == "dog" and record["split"] == "Original"
    assert record["name"]["correct"] is True and record["pick"]["correct"] is True
    assert record["confirm_true"]["yes"] is True and record["confirm_false"]["yes"] is False
    assert record["confirm_false"]["claimed"] != "dog"

    # nothing new on a second run, and an existing result is never overwritten
    (out / "001.json").write_text('{"keep": "me"}')
    assert run_vlm.process("Original", dataset / "Original", out, labels, sighted_ask, fake_open) == 0
    assert json.loads((out / "001.json").read_text()) == {"keep": "me"}


def test_vlm_process_limit_and_missing_image(dataset, tmp_path):
    labels = read_labels(dataset / "labels.csv")
    out = tmp_path / "r"
    assert run_vlm.process("Original", dataset / "Original", out, labels, sighted_ask, fake_open, limit=2, log=lambda m: None) == 2
    (dataset / "Mooney" / "003.png").unlink()
    with pytest.raises(FileNotFoundError, match="003"):
        run_vlm.process("Mooney", dataset / "Mooney", tmp_path / "m", labels, sighted_ask, fake_open, log=lambda m: None)


def test_vlm_main_checks_folders_before_loading_the_model(dataset, tmp_path, capsys):
    (dataset / "Mooney").rename(dataset / "Mooney_gone")
    assert run_vlm.main(["--data", str(dataset), "--out", str(tmp_path / "r")]) == 1
    assert "missing folder" in capsys.readouterr().err


# ---------------------------------------------------------------- SAM: processing


def fake_segment_factory(calls):
    def segment(image, prompt):
        calls.append(prompt)
        height, width = image.shape[:2]
        if image.mean() > 100 and image.std() < 1:  # flat gray original: finds the left half
            mask = np.zeros((height, width), bool); mask[:, :40] = True
            other = np.zeros((height, width), bool); other[:5, :5] = True
            return {"scores": [0.2, 0.9], "boxes": [[0, 0, 5, 5], [0, 0, 40, 60]], "masks": np.stack([other, mask])}
        return {"scores": [], "boxes": [], "masks": np.zeros((0, height, width), bool)}  # nothing in the Mooney

    return segment


def test_sam_process_orders_instances_and_writes_files(dataset, tmp_path):
    labels = read_labels(dataset / "labels.csv")
    out, calls = tmp_path / "Result2" / "Original", []
    done = run_sam.process("Original", dataset / "Original", out, labels, fake_segment_factory(calls), log=lambda m: None)
    assert done == 4 and calls == ["cat", "cat", "dog", "dog"]  # the prompt is the category name
    assert sorted(p.name for p in out.glob("001*")) == ["001.json", "001_mask.png", "001_overlay.jpg"]

    record = json.loads((out / "001.json").read_text())
    assert [i["score"] for i in record["instances"]] == [0.9, 0.2]  # best first
    assert record["instances"][0]["box"] == [0, 0, 40, 60] and record["n_instances"] == 2
    mask = cv2.imread(str(out / "001_mask.png"), cv2.IMREAD_UNCHANGED)
    assert mask.shape == (60, 80) and set(np.unique(mask)) == {0, 255} and (mask[:, :40] == 255).all()
    assert cv2.imread(str(out / "001_overlay.jpg")).shape == (60, 80, 3)


def test_sam_process_handles_no_detection(dataset, tmp_path):
    labels = read_labels(dataset / "labels.csv")
    out = tmp_path / "m"
    run_sam.process("Mooney", dataset / "Mooney", out, labels, fake_segment_factory([]), limit=1, log=lambda m: None)
    record = json.loads((out / "001.json").read_text())
    assert record["n_instances"] == 0 and record["instances"] == []
    assert not cv2.imread(str(out / "001_mask.png"), cv2.IMREAD_UNCHANGED).any()


def test_sam_process_resumes(dataset, tmp_path):
    labels = read_labels(dataset / "labels.csv")
    out, calls = tmp_path / "o", []
    segment = fake_segment_factory(calls)
    run_sam.process("Original", dataset / "Original", out, labels, segment, limit=2, log=lambda m: None)
    run_sam.process("Original", dataset / "Original", out, labels, segment, log=lambda m: None)
    assert calls == ["cat", "cat", "dog", "dog"]  # the first two were not redone


def test_overlay_marks_only_the_mask():
    image = np.full((20, 20, 3), 100, np.uint8)
    mask = np.zeros((20, 20), bool); mask[5:15, 5:15] = True
    out = run_sam.overlay(image, mask)
    assert (out[0, 0] == 100).all() and not (out[10, 10] == 100).all()
    assert (run_sam.overlay(image, np.zeros((20, 20), bool)) == image).all()


# ---------------------------------------------------------------- scoring, end to end


def test_scoring_end_to_end(dataset, tmp_path):
    labels = read_labels(dataset / "labels.csv")
    r1, r2 = tmp_path / "Result1", tmp_path / "Result2"
    for split in ("Original", "Mooney"):
        run_vlm.process(split, dataset / split, r1 / split, labels, sighted_ask, fake_open, log=lambda m: None)
        run_sam.process(split, dataset / split, r2 / split, labels, fake_segment_factory([]), log=lambda m: None)

    vlm = score.score_vlm(labels, r1)
    overall = {r["split"]: r for r in score.aggregate(vlm, score.VLM_METRICS, by_category=False)}
    assert overall["Original"]["name"] == 1.0 and overall["Original"]["pick"] == 1.0
    assert overall["Original"]["yes_true"] == 1.0 and overall["Original"]["yes_false"] == 0.0
    assert overall["Mooney"]["name"] == 0.0 and overall["Mooney"]["pick_unparsed"] == 1.0
    assert overall["Mooney"]["yes_false"] == 1.0 and overall["Mooney"]["n"] == 4

    sam = score.score_sam(labels, dataset, r2)
    overall = {r["split"]: r for r in score.aggregate(sam, score.SAM_METRICS, by_category=False)}
    assert overall["Original"]["found"] == 1.0 and overall["Original"]["iou_gt"] == 1.0
    assert overall["Original"]["consistency"] is None  # only defined for Mooneys
    assert overall["Mooney"]["found"] == 0.0 and overall["Mooney"]["iou_gt"] == 0.0
    assert overall["Mooney"]["consistency"] == 0.0

    code = score.main(["--data", str(dataset), "--vlm", str(r1), "--sam", str(r2), "--out", str(tmp_path / "scores")])
    assert code == 0
    for name in ("vlm_per_image", "vlm_summary", "sam_per_image", "sam_summary"):
        assert (tmp_path / "scores" / f"{name}.csv").exists()
    per_category = list(csv.DictReader((tmp_path / "scores" / "vlm_summary.csv").open()))
    assert {r["category"] for r in per_category} == {"cat", "dog"}


def test_scoring_with_nothing_to_score(dataset, tmp_path, capsys):
    code = score.main(["--data", str(dataset), "--vlm", str(tmp_path / "x"), "--sam", str(tmp_path / "y"), "--out", str(tmp_path / "s")])
    assert code == 1 and "no results found" in capsys.readouterr().err
