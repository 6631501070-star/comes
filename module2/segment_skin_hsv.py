"""แยกบริเวณมือ/ผิวหนังด้วยช่วงสีในระบบ HSV (OpenCV)

เทคโนโลยี: แปลง RGB → HSV แล้วใช้ช่วง H/S/V กรองพิกเซลผิว
สำหรับ fingerspelling A–Z ให้ได้มาสก์มือส่งต่อ morph / edge

เทียบเท่า MATLAB: segment_skin_hsv.m
Student 2 - Module 2
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


# ค่าเริ่มต้นเทียบสเกล MATLAB [0, 1] — เหมาะแสงในร่มทั่วไป
# Hue วนรอบแดง จึงมีสองช่วง (ต่ำ + สูง)
_DEFAULT_LIMITS: dict[str, float] = {
    "Hmin": 0.00,
    "Hmax": 0.10,
    "Hmin2": 0.90,
    "Hmax2": 1.00,
    "Smin": 0.20,
    "Smax": 0.70,
    "Vmin": 0.20,
    "Vmax": 1.00,
}


def _merge_limits(hsv_limits: dict[str, Any] | None) -> dict[str, float]:
    """รวมค่าผู้ใช้ทับค่าเริ่มต้น (ฟิลด์ที่ขาดใช้ค่า default)."""
    merged = dict(_DEFAULT_LIMITS)
    if hsv_limits is None:
        return merged
    if not isinstance(hsv_limits, dict):
        raise TypeError("hsv_limits ต้องเป็น dict (เทียบ struct ใน MATLAB)")
    for key, value in hsv_limits.items():
        if value is not None and key in merged:
            merged[key] = float(value)
    return merged


def _to_uint8_rgb(rgb_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    ปรับภาพ RGB ให้เป็น uint8 สำหรับ OpenCV
    คืน (original สำหรับ segmentedImg, uint8 สำหรับแปลงสี)
    """
    if rgb_img is None or rgb_img.size == 0:
        raise ValueError("rgb_img ต้องไม่ว่าง")
    if rgb_img.ndim != 3 or rgb_img.shape[2] != 3:
        raise ValueError("rgb_img ต้องเป็นภาพ RGB ขนาด HxWx3")

    original = rgb_img
    if np.issubdtype(rgb_img.dtype, np.floating):
        work = rgb_img.astype(np.float64)
        if work.max() <= 1.0:
            work = work * 255.0
        work = np.clip(work, 0, 255).astype(np.uint8)
    elif rgb_img.dtype == np.uint8:
        work = rgb_img
    else:
        work = np.clip(rgb_img, 0, 255).astype(np.uint8)
    return original, work


def _unit_to_opencv_hsv(h: float, s: float, v: float) -> tuple[int, int, int]:
    """แปลง HSV สเกล [0,1] (แบบ MATLAB) → OpenCV uint8 (H:0-179, S/V:0-255)."""
    h = float(np.clip(h, 0.0, 1.0))
    s = float(np.clip(s, 0.0, 1.0))
    v = float(np.clip(v, 0.0, 1.0))
    return int(round(h * 179)), int(round(s * 255)), int(round(v * 255))


def segment_skin_hsv(
    rgb_img: np.ndarray,
    hsv_limits: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    แยกมือ/ผิวด้วย HSV

    พารามิเตอร์
    ----------
    rgb_img : ndarray HxWx3
        ภาพ RGB (uint8 หรือ float ใน [0,1] / [0,255])
    hsv_limits : dict, optional
        คีย์แบบ MATLAB: Hmin, Hmax, Hmin2, Hmax2, Smin, Smax, Vmin, Vmax
        ค่าอยู่ในช่วง [0, 1]

    คืนค่า
    -------
    binary_mask : ndarray bool HxW
        True = พิกเซลที่น่าจะเป็นผิว/มือ
    segmented_img : ndarray
        ภาพ RGB คลาสเดียวกับอินพุต; พื้นหลังที่ไม่ใช่ผิว = 0
    """
    original, rgb_u8 = _to_uint8_rgb(rgb_img)
    limits = _merge_limits(hsv_limits)

    # OpenCV รองรับ COLOR_RGB2HSV โดยตรง — อินพุตยังเป็น RGB เหมือน MATLAB
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)

    h1_lo, s_lo, v_lo = _unit_to_opencv_hsv(limits["Hmin"], limits["Smin"], limits["Vmin"])
    h1_hi, s_hi, v_hi = _unit_to_opencv_hsv(limits["Hmax"], limits["Smax"], limits["Vmax"])
    h2_lo, _, _ = _unit_to_opencv_hsv(limits["Hmin2"], limits["Smin"], limits["Vmin"])
    h2_hi, _, _ = _unit_to_opencv_hsv(limits["Hmax2"], limits["Smax"], limits["Vmax"])

    # สองช่วง Hue (แดงวนรอบ) แล้ว OR กัน — เทียบ hueMask ใน MATLAB
    mask1 = cv2.inRange(hsv, (h1_lo, s_lo, v_lo), (h1_hi, s_hi, v_hi))
    mask2 = cv2.inRange(hsv, (h2_lo, s_lo, v_lo), (h2_hi, s_hi, v_hi))
    binary_u8 = cv2.bitwise_or(mask1, mask2)
    binary_mask = binary_u8.astype(bool)

    # คงพิกเซลผิว; ตั้งพื้นหลังเป็น 0 (พรีวิว GUI)
    segmented_img = original.copy()
    segmented_img[~binary_mask] = 0
    return binary_mask, segmented_img
