"""Module 1 — Histogram Processing

หน้าที่:
  1. Histogram Equalization: ปรับกระจายความสว่างทั่วภาพ (โดยเฉพาะภาพมืด/แสงน้อย)
  2. Contrast / Brightness Adjustment: ปรับความเปรียบต่างและความสว่าง
"""

from __future__ import annotations
import cv2
import numpy as np


def equalize_histogram(img: np.ndarray) -> np.ndarray:
    """
    ทำ Histogram Equalization บนภาพ RGB หรือ Grayscale

    พารามิเตอร์:
        img : ndarray HxWx3 (RGB) หรือ HxW (Grayscale)
    คืนค่า:
        ภาพที่ผ่านการ Equalize แล้ว (ประเภทเดียวกับ input)
    """
    if img is None or img.size == 0:
        raise ValueError("img ต้องไม่ว่าง")

    if img.ndim == 2:
        # ภาพระดับสีเทา
        gray = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
        return cv2.equalizeHist(gray)

    elif img.ndim == 3 and img.shape[2] == 3:
        # ภาพสี RGB: แปลงไป HSV แล้ว equalize เฉพาะช่อง V เพื่อคงโทนสีผิวไว้
        work = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(work, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    else:
        raise ValueError("img ต้องเป็นขนาด HxW หรือ HxWx3")


def adjust_contrast_brightness(
    img: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> np.ndarray:
    """
    ปรับ Contrast (alpha) และ Brightness (beta)
    สูตร: g(x, y) = alpha * f(x, y) + beta

    พารามิเตอร์:
        img   : ndarray HxW หรือ HxWx3 (uint8)
        alpha : ค่า Contrast (1.0 = ปกติ, >1.0 เพิ่มความเปรียบต่าง)
        beta  : ค่า Brightness (0 = ปกติ, >0 เพิ่มความสว่าง, <0 มืดลง)
    """
    if img is None or img.size == 0:
        raise ValueError("img ต้องไม่ว่าง")

    work = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
    return cv2.convertScaleAbs(work, alpha=alpha, beta=beta)

