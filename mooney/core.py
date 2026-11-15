"""Mooney image generation: grayscale -> Gaussian blur -> percentile threshold.

A Mooney image is a two-tone (pure black / pure white) picture. Following
Egger et al., "Inverse Rendering Best Explains Face Perception Under Extreme
Illuminations", we do not hand-pick an intensity threshold. Instead the user
chooses which percentage of pixels should be white, and the threshold is
derived per image so that this percentage is hit.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MooneyResult:
    image: np.ndarray  # uint8, same height/width as the input, values 0 or 255
    threshold: float  # cut-off on the blurred image, in [0, 1]
    white_pct: float  # percentage of white pixels actually achieved


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert a uint8/uint16 image (gray, BGR or BGRA) to float32 gray in [0, 1]."""
    if image.dtype == np.uint8:
        scale = 255.0
    elif image.dtype == np.uint16:
        scale = 65535.0
    else:
        raise TypeError(f"expected a uint8 or uint16 image, got {image.dtype}")

    img = image.astype(np.float32) / scale
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"unsupported image shape {image.shape}")


def blur(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with standard deviation `sigma` in pixels (0 = no blur)."""
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if sigma == 0:
        return gray
    # ksize (0, 0) lets OpenCV size the kernel from sigma. Blurring in float32
    # keeps fractional values, so far fewer pixels tie at the threshold.
    return cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)


def threshold_by_white_pct(blurred: np.ndarray, white_pct: float) -> tuple[np.ndarray, float]:
    """Return (boolean white mask, threshold) so that ~`white_pct` % of pixels are white.

    The brightest pixels become white. The threshold is the k-th largest value,
    read straight from the sorted intensities (the percentile), so no iterative
    search is needed. If many pixels share the threshold value (flat regions),
    the exact percentage cannot be hit; ties then all go to whichever side
    lands closer to the target, which keeps flat regions free of speckle.
    """
    if not 0 < white_pct < 100:
        raise ValueError("white_pct must be between 0 and 100 (exclusive)")

    flat = blurred.ravel()
    n = flat.size
    k = int(round(n * white_pct / 100.0))  # number of white pixels wanted
    k = min(max(k, 1), n - 1)

    thr = float(np.partition(flat, n - k)[n - k])  # k-th largest value
    at_least = blurred >= thr  # k or more white pixels
    fewer = blurred > thr  # fewer than k white pixels
    over = int(at_least.sum()) - k
    under = k - int(fewer.sum())
    return (at_least if over <= under else fewer), thr


def make_mooney(image: np.ndarray, sigma: float, white_pct: float) -> MooneyResult:
    """Turn an image into a Mooney image.

    image:     uint8/uint16 array as returned by cv2.imread (gray, BGR or BGRA)
    sigma:     strength of the Gaussian blur, in pixels
    white_pct: percentage of pixels that should end up white (0-100, exclusive)
    """
    blurred = blur(to_gray(image), sigma)
    mask, thr = threshold_by_white_pct(blurred, white_pct)
    out = np.where(mask, 255, 0).astype(np.uint8)
    return MooneyResult(image=out, threshold=thr, white_pct=100.0 * float(mask.mean()))
