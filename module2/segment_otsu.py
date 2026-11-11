"""แปลงภาพเทาเป็นไบนารีด้วย Otsu threshold (OpenCV)

เทคโนโลยี: Otsu เลือกค่าตัดที่แยกพื้นหน้า/พื้นหลังจากฮิสโตแกรม
ใช้เป็นทางเลือกหรือคู่เทียบกับ HSV เมื่อฉากค่อนข้างเรียบ

เทียบเท่า MATLAB: segment_otsu.m
Student 2 - Module 2
"""

from __future__ import annotations

import cv2
import numpy as np


def _to_gray_uint8(gray_img: np.ndarray) -> np.ndarray:
    """ได้ภาพเทา uint8 ขนาด HxW (ถ้าส่ง RGB มาจะแปลงให้)."""
    if gray_img is None or gray_img.size == 0:
        raise ValueError("gray_img ต้องไม่ว่าง")

    img = gray_img
    if img.ndim == 3 and img.shape[2] == 3:
        # รับ RGB เหมือน MATLAB ที่กันพลาดจาก GUI
        if np.issubdtype(img.dtype, np.floating):
            tmp = img.astype(np.float64)
            if tmp.max() <= 1.0:
                tmp = tmp * 255.0
            tmp = np.clip(tmp, 0, 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            tmp = np.clip(img, 0, 255).astype(np.uint8)
        else:
            tmp = img
        img = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    elif img.ndim != 2:
        raise ValueError("gray_img ต้องเป็น HxW หรือ HxWx3")

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


def segment_otsu(gray_img: np.ndarray) -> tuple[np.ndarray, float]:
    """
    ไบนารีด้วย Otsu

    พารามิเตอร์
    ----------
    gray_img : ndarray
        ภาพเทา HxW หรือ RGB HxWx3 (จะแปลงให้)

    คืนค่า
    -------
    binary_img : ndarray bool HxW
        True = พื้นหน้า (ตามเกณฑ์ Otsu; อาจต้องกลับด้วย ~ ถ้ามือเป็นสีเข้ม)
    level : float
        ค่า threshold ในช่วง [0, 1] เทียบ graythresh ของ MATLAB
    """
    if gray_img.dtype == bool or gray_img.dtype == np.bool_:
        if gray_img.ndim != 2:
            raise ValueError("logical gray_img ต้องเป็น HxW")
        return gray_img.copy(), 0.5

    work = _to_gray_uint8(gray_img)

    # THRESH_BINARY + Otsu: พิกเซล > T เป็น 255
    level_u8, binary_u8 = cv2.threshold(
        work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    binary_img = binary_u8.astype(bool)
    # MATLAB graythresh คืน [0, 1]
    level = float(level_u8) / 255.0
    return binary_img, level
