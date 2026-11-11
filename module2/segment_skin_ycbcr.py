"""แยกบริเวณมือ/ผิวหนังด้วยช่วงสีในระบบ YCbCr (OpenCV)

เทคโนโลยี: แปลง RGB → YCrCb โดยผิวหนังมนุษย์มีค่า Cb, Cr เกาะกลุ่มกันแน่น
ช่วงมาตรฐาน: Cb อยู่ระหว่าง 77 ถึง 127 และ Cr อยู่ระหว่าง 133 ถึง 173
ใช้เปรียบเทียบกับ HSV ตามข้อกำหนด comvis.pdf

Student 2 - Module 2
"""

from __future__ import annotations
from typing import Any
import cv2
import numpy as np


def _to_uint8_rgb(rgb_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def segment_skin_ycbcr(
    rgb_img: np.ndarray,
    cb_range: tuple[int, int] = (77, 127),
    cr_range: tuple[int, int] = (133, 173),
) -> tuple[np.ndarray, np.ndarray]:
    """
    แยกมือ/ผิวหนังในระบบสี YCbCr

    พารามิเตอร์:
        rgb_img  : ภาพ RGB (HxWx3)
        cb_range : ช่วงของช่อง Cb (ค่าเริ่มต้น 77, 127)
        cr_range : ช่วงของช่อง Cr (ค่าเริ่มต้น 133, 173)

    คืนค่า:
        binary_mask   : ndarray bool HxW (True = ผิว)
        segmented_img : ภาพ RGB ที่ตัดพื้นหลังเป็นสีดำ
    """
    original, rgb_u8 = _to_uint8_rgb(rgb_img)

    # OpenCV แปลง RGB -> YCrCb (ช่อง 0: Y, ช่อง 1: Cr, ช่อง 2: Cb)
    ycrcb = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YCrCb)

    # กำหนด Lower และ Upper bound (Y ไม่จำกัด 0-255)
    lower = np.array([0, cr_range[0], cb_range[0]], dtype=np.uint8)
    upper = np.array([255, cr_range[1], cb_range[1]], dtype=np.uint8)

    mask_u8 = cv2.inRange(ycrcb, lower, upper)
    binary_mask = mask_u8.astype(bool)

    segmented_img = original.copy()
    segmented_img[~binary_mask] = 0

    return binary_mask, segmented_img

