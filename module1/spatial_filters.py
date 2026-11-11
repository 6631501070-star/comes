"""Module 1 — Spatial Filtering

หน้าที่:
  1. Smoothing (Mean, Median filter): ลด Noise จากกล้องเว็บแคม
  2. Sharpening (Laplacian, High-pass filter): เน้นขอบของนิ้วมือให้คมชัด
"""

from __future__ import annotations
import cv2
import numpy as np


def _ensure_odd(ksize: int) -> int:
    k = max(1, int(ksize))
    return k if k % 2 == 1 else k + 1


def mean_filter(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """กรองสัญญาณรบกวนด้วย Mean (Box) Filter"""
    if img is None or img.size == 0:
        raise ValueError("img ต้องไม่ว่าง")
    k = _ensure_odd(ksize)
    return cv2.blur(img, (k, k))


def median_filter(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """กรองสัญญาณรบกวนแบบ Salt & Pepper ด้วย Median Filter"""
    if img is None or img.size == 0:
        raise ValueError("img ต้องไม่ว่าง")
    k = _ensure_odd(ksize)
    return cv2.medianBlur(img, k)


def laplacian_sharpen(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    เน้นขอบภาพด้วย Laplacian Sharpening:
    Sharpened = Original - strength * Laplacian
    """
    if img is None or img.size == 0:
        raise ValueError("img ต้องไม่ว่าง")

    work = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
    # หา Laplacian ใน float เพื่อป้องกัน overflow
    lap = cv2.Laplacian(work, cv2.CV_64F)
    sharpened = work.astype(np.float64) - (strength * lap)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def highpass_sharpen(img: np.ndarray) -> np.ndarray:
    """
    เน้นขอบภาพด้วย High-pass Filter Kernel 3x3:
    [[ -1, -1, -1 ],
     [ -1,  9, -1 ],
     [ -1, -1, -1 ]]
    """
    if img is None or img.size == 0:
        raise ValueError("img ต้องไม่ว่าง")

    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)

    work = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
    return cv2.filter2D(work, -1, kernel)

