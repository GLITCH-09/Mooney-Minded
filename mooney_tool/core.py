"""Mooney images: Gaussian blur, then a percentile threshold that fixes the share of white pixels."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class MooneyInfo:
    blur_sigma: float
    white_pct_target: float
    white_pct_actual: float
    threshold: float
    invert: bool
    upside_down: bool


def make_mooney(img: np.ndarray, blur_sigma: float = 4.0, white_pct: float = 40.0,
                invert: bool = False, upside_down: bool = False) -> tuple[np.ndarray, MooneyInfo]:
    """Return a two-tone uint8 image (0/255) in which `white_pct` % of pixels are white."""
    gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img).astype(np.float32) / 255.0
    if blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=float(blur_sigma), borderType=cv2.BORDER_REFLECT)
    white_pct = float(np.clip(white_pct, 0.0, 100.0))
    threshold = float(np.percentile(gray, 100.0 - white_pct))
    white = gray > threshold
    actual = float(white.mean() * 100.0)
    if invert:
        white = ~white
    if upside_down:
        white = white[::-1]
    mooney = np.ascontiguousarray(white, dtype=np.uint8) * 255
    return mooney, MooneyInfo(float(blur_sigma), white_pct, actual, threshold, invert, upside_down)
