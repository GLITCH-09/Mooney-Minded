"""Core Mooney-image generation: Gaussian blur -> percentile threshold.

A Mooney image is a two-tone (black/white) image. Following Egger et al. (2026,
"Inverse Rendering Best Explains Face Perception Under Extreme Illuminations")
we blur the grayscale image with a Gaussian and then threshold it. Instead of
hand-tuning a threshold per image we fix the *percentage of white pixels* and
derive the threshold from the blurred image's intensity histogram: the threshold
is the (100 - white_pct) percentile, so exactly white_pct % of pixels end up
white (up to ties, see `make_mooney`).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class MooneyInfo:
    blur_sigma: float
    white_pct_target: float
    white_pct_actual: float
    threshold: float
    invert: bool
    upside_down: bool


def to_gray_float(img: np.ndarray) -> np.ndarray:
    """uint8 BGR/gray image -> float32 grayscale in [0, 1]."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return gray.astype(np.float32) / 255.0


def make_mooney(
    img: np.ndarray,
    blur_sigma: float = 4.0,
    white_pct: float = 40.0,
    invert: bool = False,
    upside_down: bool = False,
) -> tuple[np.ndarray, MooneyInfo]:
    """Turn an image into a Mooney image.

    Args:
        img: uint8 image (BGR or gray), any size.
        blur_sigma: Gaussian sigma in pixels (0 = no blur). Larger = fewer details.
        white_pct: target percentage of white pixels, 0-100.
        invert: swap black and white AFTER thresholding (contrast-polarity flip).
        upside_down: flip the finished Mooney vertically (inversion add-on).

    Returns:
        (mooney, info): mooney is a uint8 single-channel image with values 0 / 255.
        info.white_pct_actual is the white percentage really achieved (ties in flat
        regions can make it differ slightly from the target).
    """
    gray = to_gray_float(img)
    if blur_sigma > 0:
        # ksize=(0, 0) lets OpenCV derive the kernel size from sigma (no clipping)
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=float(blur_sigma), borderType=cv2.BORDER_REFLECT)

    white_pct = float(np.clip(white_pct, 0.0, 100.0))
    threshold = float(np.percentile(gray, 100.0 - white_pct))
    mask = gray > threshold
    mooney = (mask * 255).astype(np.uint8)

    if invert:
        mooney = 255 - mooney
    if upside_down:
        mooney = np.ascontiguousarray(mooney[::-1, :])

    actual = float((mooney == 255).mean() * 100.0)
    # report the white share of the *un-inverted* Mooney so it is comparable to the target
    if invert:
        actual = 100.0 - actual
    info = MooneyInfo(
        blur_sigma=float(blur_sigma),
        white_pct_target=white_pct,
        white_pct_actual=actual,
        threshold=threshold,
        invert=invert,
        upside_down=upside_down,
    )
    return mooney, info
