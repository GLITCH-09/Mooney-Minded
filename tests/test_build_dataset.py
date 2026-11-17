import csv

import cv2
import numpy as np
import pytest

from scripts import build_dataset as bd
from scripts.common import LABEL_FIELDS

W, H = 400, 300  # every synthetic image


def square(x, y, side):
    return [[x, y, x + side, y, x + side, y + side, x, y + side]]


class FakeCoco:
    """Builds a tiny COCO-style annotation dict plus a fake downloader."""

    def __init__(self):
        self.images, self.annotations = [], []
        self.categories = [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}, {"id": 3, "name": "person"}]
        self.next_ann = 1

    def add_image(self, image_id, width=W, height=H):
        self.images.append({"id": image_id, "width": width, "height": height})

    def add_ann(self, image_id, category_id, side, crowd=0, x=10, y=10):
        self.annotations.append(
            {
                "id": self.next_ann,
                "image_id": image_id,
                "category_id": category_id,
                "iscrowd": crowd,
                "area": float(side * side),
                "segmentation": square(x, y, side),
            }
        )
        self.next_ann += 1

    def dict(self):
        return {"images": self.images, "annotations": self.annotations, "categories": self.categories}

    def fetch(self, url):
        image_id = int(url.rsplit("/", 1)[-1].split(".")[0])
        meta = next(i for i in self.images if i["id"] == image_id)
        pixels = np.full((meta["height"], meta["width"], 3), 128, np.uint8)
        return cv2.imencode(".jpg", pixels)[1].tobytes()


def dominant_cat_images(coco, ids, side=200):
    for i in ids:
        coco.add_image(i)
        coco.add_ann(i, 1, side)  # 200x200 of 400x300 = 33% of the image


def test_finds_dominant_instances():
    coco = FakeCoco()
    dominant_cat_images(coco, [1, 2])
    found = bd.find_candidates(coco.dict(), "cat", 0.25, 0.10)
    assert [c.image_id for c in found] == [1, 2]
    assert found[0].area_ratio == pytest.approx(200 * 200 / (W * H))


def test_rejects_small_crowd_and_not_dominant_objects():
    coco = FakeCoco()
    coco.add_image(1); coco.add_ann(1, 1, 100)  # 100x100 = 8% -> too small
    coco.add_image(2); coco.add_ann(2, 1, 200, crowd=1)  # crowd region
    coco.add_image(3); coco.add_ann(3, 1, 200); coco.add_ann(3, 1, 150, x=220)  # second cat nearly as big
    coco.add_image(4); coco.add_ann(4, 1, 200); coco.add_ann(4, 1, 60, x=220)  # second cat small: fine
    coco.add_image(5); coco.add_ann(5, 2, 200)  # a dog, not a cat
    found = bd.find_candidates(coco.dict(), "cat", 0.25, 0.10)
    assert [c.image_id for c in found] == [4]


def test_other_categories_are_limited_by_max_other_area():
    coco = FakeCoco()
    coco.add_image(1); coco.add_ann(1, 1, 200); coco.add_ann(1, 3, 120, x=250)  # person fills 12%
    assert bd.find_candidates(coco.dict(), "cat", 0.25, 0.10) == []
    assert len(bd.find_candidates(coco.dict(), "cat", 0.25, 0.20)) == 1


def test_unknown_category():
    with pytest.raises(ValueError, match="unknown COCO category"):
        bd.find_candidates(FakeCoco().dict(), "unicorn", 0.25, 0.1)


def test_pick_uses_the_strictest_level_that_has_enough():
    coco = FakeCoco()
    dominant_cat_images(coco, range(1, 4), side=200)  # 33%: passes level 0
    for i in range(4, 8):  # 20%: only from level 1 on
        coco.add_image(i); coco.add_ann(i, 1, 155)
    candidates, level = bd.pick_candidates(coco.dict(), "cat", 3, seed=0)
    assert level == 0 and len(candidates) == 3
    candidates, level = bd.pick_candidates(coco.dict(), "cat", 6, seed=0)
    assert level == 1 and len(candidates) == 7


def test_pick_is_reproducible_and_depends_on_seed():
    coco = FakeCoco()
    dominant_cat_images(coco, range(1, 30))
    first = [c.image_id for c in bd.pick_candidates(coco.dict(), "cat", 5, seed=1)[0]]
    again = [c.image_id for c in bd.pick_candidates(coco.dict(), "cat", 5, seed=1)[0]]
    other = [c.image_id for c in bd.pick_candidates(coco.dict(), "cat", 5, seed=2)[0]]
    assert first == again and first != other


def test_pick_explains_when_there_are_too_few():
    coco = FakeCoco()
    dominant_cat_images(coco, [1, 2])
    with pytest.raises(ValueError, match="only 2 usable images for 'cat'"):
        bd.pick_candidates(coco.dict(), "cat", 5, seed=0)


def test_polygon_mask_and_multipart():
    mask = bd.polygon_mask(square(10, 20, 30) + square(100, 100, 10), 200, 150)
    assert mask.shape == (150, 200) and mask.dtype == np.uint8
    assert mask[20:51, 10:41].sum() > 0.9 * 31 * 31
    assert mask[105, 105] == 1 and mask[0, 0] == 0


def test_resized_shape_keeps_aspect_ratio():
    assert bd.resized_shape(640, 480, 512) == (512, 384)
    assert bd.resized_shape(480, 640, 512) == (384, 512)
    assert bd.resized_shape(300, 200, 512) == (512, 341)


def test_build_writes_images_masks_and_labels(tmp_path):
    coco = FakeCoco()
    for i in range(1, 6):
        coco.add_image(i); coco.add_ann(i, 1, 200)  # cats
    for i in range(6, 11):
        coco.add_image(i); coco.add_ann(i, 2, 200)  # dogs
    messages = []
    rows = bd.build(tmp_path, coco.dict(), ["cat", "dog"], 3, seed=0, long_side=200, fetch=coco.fetch, log=messages.append)

    assert [r["id"] for r in rows] == ["001", "002", "003", "004", "005", "006"]
    assert [r["category"] for r in rows] == ["cat"] * 3 + ["dog"] * 3
    for row in rows:
        image = cv2.imread(str(tmp_path / "Original" / row["file"]))
        mask = cv2.imread(str(tmp_path / "gt" / "masks" / f"{row['id']}.png"), cv2.IMREAD_UNCHANGED)
        assert image.shape[:2] == (150, 200) == (row["height"], row["width"])  # 400x300 -> long side 200
        assert mask.shape == (150, 200) and set(np.unique(mask)) == {0, 255}
        assert (mask > 0).mean() == pytest.approx(200 * 200 / (W * H), abs=0.01)
        # the box in labels.csv surrounds exactly the mask
        ys, xs = np.nonzero(mask)
        assert (row["box_x0"], row["box_y0"], row["box_x1"], row["box_y1"]) == (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)

    with (tmp_path / "labels.csv").open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == LABEL_FIELDS
        assert len(list(reader)) == 6
    assert any("cat: 5 candidates" in m for m in messages)


def test_build_skips_images_whose_size_differs_from_the_annotation(tmp_path):
    coco = FakeCoco()
    for i in range(1, 5):
        coco.add_image(i); coco.add_ann(i, 1, 200)
    real_fetch = coco.fetch
    order = [c.image_id for c in bd.pick_candidates(coco.dict(), "cat", 2, seed=0)[0]]
    bad = order[0]  # the first image that will be tried comes back rotated, like an EXIF mix-up

    def fetch(url):
        image_id = int(url.rsplit("/", 1)[-1].split(".")[0])
        if image_id == bad:
            return cv2.imencode(".jpg", np.full((W, H, 3), 128, np.uint8))[1].tobytes()
        return real_fetch(url)

    messages = []
    rows = bd.build(tmp_path, coco.dict(), ["cat"], 2, seed=0, long_side=200, fetch=fetch, log=messages.append)
    assert [r["coco_image_id"] for r in rows] == order[1:3]
    assert sum("skipping" in m for m in messages) == 1


def test_build_fails_clearly_when_downloads_are_unusable(tmp_path):
    coco = FakeCoco()
    dominant_cat_images(coco, [1, 2])
    with pytest.raises(RuntimeError, match="only 0 of 2"):
        bd.build(tmp_path, coco.dict(), ["cat"], 2, seed=0, long_side=200, fetch=lambda url: b"not an image", log=lambda m: None)


def test_caching_downloads_only_once(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        return b"data"

    cached = bd.caching(fetch, tmp_path / "raw")
    url = "http://example.org/val2017/000000000001.jpg"
    assert cached(url) == b"data" and cached(url) == b"data"
    assert calls == [url]


def test_main_with_local_annotations(tmp_path, monkeypatch):
    coco = FakeCoco()
    dominant_cat_images(coco, range(1, 4))
    import json

    annotations = tmp_path / "instances_val2017.json"
    annotations.write_text(json.dumps(coco.dict()))
    monkeypatch.setattr(bd, "http_fetch", coco.fetch)
    out = tmp_path / "data"
    code = bd.main(["--out", str(out), "--categories", "cat", "--per-category", "3", "--size", "200", "--annotations", str(annotations)])
    assert code == 0
    assert len(list((out / "Original").glob("*.jpg"))) == 3
    assert bd.main(["--out", str(out), "--categories", "unicorn", "--annotations", str(annotations)]) == 1
