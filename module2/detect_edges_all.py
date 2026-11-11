"""ตรวจจับขอบแบบ Sobel / Prewitt / Canny (OpenCV)

เทคโนโลยี:
  - Sobel / Prewitt: เกรเดียนต์ด้วยเคอร์เนล 3x3
  - Canny: blur + gradient + non-maxima + hysteresis — มักได้เส้นนิ้วสะอาดกว่า

เทียบเท่า MATLAB: detect_edges_all.m
Student 2 - Module 2
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


# Prewitt ไม่มีใน OpenCV โดยตรง — ใช้เคอร์เนลมาตรฐาน
_PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
_PREWITT_Y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)


def _to_gray_u8(gray_or_mask: np.ndarray) -> np.ndarray:
    """ได้ภาพเทา uint8 HxW (รองรับ mask / RGB)."""
    if gray_or_mask is None or gray_or_mask.size == 0:
        raise ValueError("gray_or_mask ต้องไม่ว่าง")

    img = gray_or_mask
    if img.ndim == 3 and img.shape[2] == 3:
        if np.issubdtype(img.dtype, np.floating):
            tmp = img.astype(np.float64)
            if tmp.max() <= 1.0:
                tmp = tmp * 255.0
            tmp = np.clip(tmp, 0, 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            tmp = np.clip(img, 0, 255).astype(np.uint8)
        else:
            tmp = img
        return cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)

    if img.ndim != 2:
        raise ValueError("gray_or_mask ต้องเป็น HxW หรือ HxWx3")

    if img.dtype == bool or img.dtype == np.bool_:
        return (img.astype(np.uint8) * 255)

    if np.issubdtype(img.dtype, np.floating):
        work = img.astype(np.float64)
        if work.max() <= 1.0:
            work = work * 255.0
        return np.clip(work, 0, 255).astype(np.uint8)

    if img.dtype == np.uint8:
        return img

    return np.clip(img, 0, 255).astype(np.uint8)


def _sobel_edges(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # threshold อัตโนมัติแบบง่าย — ได้ขอบ logical ใกล้ edge(..., 'Sobel')
    mag_u8 = cv2.convertScaleAbs(mag)
    _, bw = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw.astype(bool)


def _prewitt_edges(gray: np.ndarray) -> np.ndarray:
    gx = cv2.filter2D(gray.astype(np.float32), -1, _PREWITT_X)
    gy = cv2.filter2D(gray.astype(np.float32), -1, _PREWITT_Y)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = cv2.convertScaleAbs(mag)
    _, bw = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw.astype(bool)


def detect_edges_all(gray_or_mask: np.ndarray) -> dict[str, Any]:
    """
    รัน Sobel / Prewitt / Canny

    พารามิเตอร์
    ----------
    gray_or_mask : ndarray
        ภาพเทาหรือมาสก์ไบนารี (แนะนำใช้ cleaned_mask หลัง morph)

    คืนค่า
    -------
    dict:
        sobel_edges, prewitt_edges, canny_edges  (bool HxW)
    """
    gray = _to_gray_u8(gray_or_mask)

    sobel = _sobel_edges(gray)
    prewitt = _prewitt_edges(gray)
    # Canny ค่าเริ่มต้นของ OpenCV; ปรับได้ทีหลังถ้าเส้นบาง/หนาเกิน
    canny_u8 = cv2.Canny(gray, 50, 150)
    canny = canny_u8.astype(bool)

    return {
        "sobel_edges": sobel,
        "prewitt_edges": prewitt,
        "canny_edges": canny,
    }
