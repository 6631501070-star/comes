"""ทำความสะอาดมาสก์มือด้วย morphological operations (OpenCV)

เทคโนโลยี: structuring element + open/close/erode/dilate + อุดรู
แล้วเก็บเฉพาะก้อนใหญ่สุด (มือ) เพื่อลด noise จากฉากรก

เทียบเท่า MATLAB: apply_morphology.m (+ ปรับปรุง largest blob)
Student 2 - Module 2
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _to_binary_u8(binary_mask: np.ndarray) -> np.ndarray:
    """แปลงมาสก์เป็น uint8 0/255 สำหรับ OpenCV morph."""
    if binary_mask is None or binary_mask.size == 0:
        raise ValueError("binary_mask ต้องไม่ว่าง")
    if binary_mask.ndim != 2:
        raise ValueError("binary_mask ต้องเป็นภาพ 2 มิติ HxW")

    if binary_mask.dtype == bool or binary_mask.dtype == np.bool_:
        return (binary_mask.astype(np.uint8) * 255)
    if np.issubdtype(binary_mask.dtype, np.number):
        return ((binary_mask != 0).astype(np.uint8) * 255)
    raise TypeError("binary_mask ต้องเป็น bool หรือตัวเลข")


def _make_kernel(se_shape: str, se_size: int) -> np.ndarray:
    """สร้าง structuring element ใกล้เคียง strel ของ MATLAB."""
    shape = se_shape.lower().strip()
    size = int(se_size)
    if size <= 0:
        raise ValueError("se_size ต้องเป็นจำนวนบวก")

    # MATLAB disk(radius) ≈ เส้นผ่านศูนย์กลาง 2*r+1
    if shape in ("disk", "ellipse"):
        k = 2 * size + 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if shape == "square":
        k = size if size % 2 == 1 else size + 1
        return cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    if shape == "cross":
        k = 2 * size + 1
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))
    if shape in ("diamond",):
        # OpenCV ไม่มี diamond โดยตรง — ใช้ ellipse ใกล้เคียง
        k = 2 * size + 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    raise ValueError(
        f"se_shape ไม่รองรับ: {se_shape!r} (ลอง 'disk', 'square', 'cross')"
    )


def _fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    """อุดรูภายในมาสก์ — เทียบ imfill(..., 'holes')."""
    # หา contour ภายนอก แล้ววาดเติมทั้งก้อน
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask_u8)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _keep_largest_blob(mask_u8: np.ndarray) -> np.ndarray:
    """เก็บเฉพาะ connected component ที่มีพื้นที่ใหญ่สุด (มือ)."""
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask_u8)

    largest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask_u8)
    cv2.drawContours(out, [largest], -1, 255, thickness=cv2.FILLED)
    return out


def apply_morphology(
    binary_mask: np.ndarray,
    se_shape: str = "disk",
    se_size: int = 5,
    keep_largest: bool = True,
) -> dict[str, Any]:
    """
    Morphology ทำความสะอาดมาสก์มือ

    พารามิเตอร์
    ----------
    binary_mask : ndarray HxW
        มาสก์ไบนารี (bool หรือตัวเลข)
    se_shape : str
        รูป SE ค่าเริ่มต้น 'disk' (ellipse ใน OpenCV)
    se_size : int
        ขนาด/รัศมี SE ค่าเริ่มต้น 5
    keep_largest : bool
        ถ้า True (ค่าเริ่มต้น) จะเก็บก้อนใหญ่สุดใน cleaned_mask
        เพื่อลดวัตถุสีใกล้ผิวจากฉากรก

    คืนค่า
    -------
    dict เทียบ struct MATLAB:
        opened_mask, closed_mask, eroded_mask, dilated_mask,
        filled_mask, cleaned_mask, se_shape, se_size, keep_largest
    """
    bw = _to_binary_u8(binary_mask)
    kernel = _make_kernel(se_shape, se_size)

    # Opening = erode แล้ว dilate → ลบจุดขาวเล็ก
    opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    # Closing = dilate แล้ว erode → อุดช่องว่างเล็ก
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    eroded = cv2.erode(bw, kernel)
    dilated = cv2.dilate(bw, kernel)

    # อุดรูบน closed mask
    filled = _fill_holes(closed)

    # pipeline แนะนำ: open → close → fill holes (+ largest blob)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    cleaned = _fill_holes(cleaned)
    if keep_largest:
        cleaned = _keep_largest_blob(cleaned)

    def _as_bool(m: np.ndarray) -> np.ndarray:
        return m.astype(bool)

    return {
        "opened_mask": _as_bool(opened),
        "closed_mask": _as_bool(closed),
        "eroded_mask": _as_bool(eroded),
        "dilated_mask": _as_bool(dilated),
        "filled_mask": _as_bool(filled),
        "cleaned_mask": _as_bool(cleaned),
        "se_shape": se_shape,
        "se_size": int(se_size),
        "keep_largest": bool(keep_largest),
    }
