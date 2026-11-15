import csv

import cv2
import numpy as np
import pytest

from mooney.cli import main
from mooney.core import make_mooney


def noise_image(shape=(120, 160, 3), seed=0):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


@pytest.mark.parametrize("white_pct", [5, 16, 40, 73])
@pytest.mark.parametrize("sigma", [0, 2.5])
def test_white_percentage_is_hit(white_pct, sigma):
    result = make_mooney(noise_image(), sigma=sigma, white_pct=white_pct)
    # sigma 0 leaves 8-bit noise full of ties; blurring spreads the values out
    tolerance = 1.0 if sigma == 0 else 0.05
    assert result.white_pct == pytest.approx(white_pct, abs=tolerance)
    assert result.white_pct == pytest.approx(100 * (result.image == 255).mean())


def test_output_is_two_tone_and_same_size():
    image = noise_image((90, 140, 3))
    result = make_mooney(image, sigma=3, white_pct=30)
    assert result.image.shape == (90, 140)
    assert result.image.dtype == np.uint8
    assert set(np.unique(result.image)) == {0, 255}


def test_bright_side_becomes_white():
    ramp = np.tile(np.linspace(0, 255, 200).astype(np.uint8), (50, 1))  # dark left, bright right
    out = make_mooney(ramp, sigma=0, white_pct=25).image
    assert (out[:, :140] == 0).all()
    assert (out[:, 155:] == 255).all()


@pytest.mark.parametrize("channels", [None, 1, 3, 4])
def test_accepts_gray_bgr_and_bgra(channels):
    shape = (60, 80) if channels is None else (60, 80, channels)
    result = make_mooney(noise_image(shape), sigma=1, white_pct=50)
    assert result.image.shape == (60, 80)


def test_uint16_matches_uint8():
    image8 = noise_image()
    image16 = image8.astype(np.uint16) * 257
    a = make_mooney(image8, sigma=2, white_pct=35).image
    b = make_mooney(image16, sigma=2, white_pct=35).image
    assert (a != b).mean() < 0.001


def test_ties_go_to_the_closer_side_without_speckle():
    two_level = np.zeros((40, 100), np.uint8)
    two_level[:, 50:] = 200  # exactly half the pixels are bright
    # 30% is closer to "bright half white" (50%) than to "nothing white" (0%)
    assert make_mooney(two_level, sigma=0, white_pct=30).white_pct == pytest.approx(50)
    # 10% is closer to nothing white
    assert make_mooney(two_level, sigma=0, white_pct=10).white_pct == pytest.approx(0)


def test_flat_image_does_not_crash():
    flat = np.full((30, 30), 128, np.uint8)
    result = make_mooney(flat, sigma=2, white_pct=40)
    assert result.white_pct in (0.0, 100.0)


@pytest.mark.parametrize("white_pct", [0, 100, -5, 150])
def test_rejects_bad_white_pct(white_pct):
    with pytest.raises(ValueError):
        make_mooney(noise_image(), sigma=1, white_pct=white_pct)


def test_rejects_negative_sigma_and_float_images():
    with pytest.raises(ValueError):
        make_mooney(noise_image(), sigma=-1, white_pct=30)
    with pytest.raises(TypeError):
        make_mooney(noise_image().astype(np.float64), sigma=1, white_pct=30)


def test_batch_cli(tmp_path, capsys):
    src, dst = tmp_path / "Original", tmp_path / "Mooney"
    src.mkdir()
    for i, ext in enumerate(["jpg", "png", "jpg"], start=1):
        cv2.imwrite(str(src / f"{i:03d}.{ext}"), noise_image(seed=i))
    (src / "notes.txt").write_text("not an image")
    manifest = tmp_path / "manifest.csv"

    code = main(["batch", str(src), str(dst), "--sigma", "2", "--white-pct", "40", "--manifest", str(manifest)])

    assert code == 0
    assert sorted(p.name for p in dst.iterdir()) == ["001.png", "002.png", "003.png"]
    saved = cv2.imread(str(dst / "001.png"), cv2.IMREAD_UNCHANGED)
    assert saved.shape == (120, 160) and set(np.unique(saved)) == {0, 255}
    rows = list(csv.DictReader(manifest.open()))
    assert [r["input"] for r in rows] == ["001.jpg", "002.png", "003.jpg"]
    assert all(abs(float(r["white_pct_actual"]) - 40) < 0.05 for r in rows)
    assert "wrote 3 Mooney image(s)" in capsys.readouterr().out


def test_batch_respects_exif_orientation(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    src = tmp_path / "in"
    src.mkdir()
    exif = Image.Exif()
    exif[0x0112] = 6  # stored landscape, displayed portrait (rotate 90 degrees)
    Image.fromarray(noise_image((60, 100, 3))).save(src / "phone.jpg", exif=exif)

    assert main(["batch", str(src), str(tmp_path / "out"), "--sigma", "1", "--white-pct", "30"]) == 0
    saved = cv2.imread(str(tmp_path / "out" / "phone.png"), cv2.IMREAD_UNCHANGED)
    assert saved.shape == (100, 60)


def test_batch_reports_unreadable_files(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    cv2.imwrite(str(src / "good.png"), noise_image())
    (src / "broken.jpg").write_bytes(b"this is not a jpeg")
    code = main(["batch", str(src), str(tmp_path / "out"), "--sigma", "1", "--white-pct", "30"])
    assert code == 1
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["good.png"]


def test_batch_refuses_name_clashes(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    cv2.imwrite(str(src / "a.png"), noise_image())
    cv2.imwrite(str(src / "a.jpg"), noise_image())
    assert main(["batch", str(src), str(tmp_path / "out"), "--sigma", "1", "--white-pct", "30"]) == 1
    assert not (tmp_path / "out").exists()


def test_sweep_cli(tmp_path):
    image = tmp_path / "face.png"
    cv2.imwrite(str(image), noise_image())
    code = main(["sweep", str(image), "--out", str(tmp_path / "sheets"), "--sigmas", "1", "4", "--white-pcts", "20", "50"])
    assert code == 0
    sheet = cv2.imread(str(tmp_path / "sheets" / "face_sweep.png"), cv2.IMREAD_UNCHANGED)
    assert sheet is not None and sheet.ndim == 2
