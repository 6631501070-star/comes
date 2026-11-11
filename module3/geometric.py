"""Module 3 — Geometric Transformations & Interpolation

หน้าที่:
  1. Geometric Transformations:
     - Rotation (การหมุนภาพ)
     - Scaling / Zoom (การปรับขนาด/ซูม)
     - Translation (การเลื่อนตำแหน่ง)
  2. Interpolation Techniques:
     - Nearest-neighbor (cv2.INTER_NEAREST)
     - Bilinear (cv2.INTER_LINEAR)
     - Bicubic (cv2.INTER_CUBIC)
  3. Image Warping / Affine Transformation (การดัดรูปทรงตามพิกัดอ้างอิง)

Student 3 - Module 3
"""

from __future__ import annotations
import cv2
import numpy as np


_INTERPOLATION_METHODS = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
}


def _get_interp_flag(method: str) -> int:
    m = method.lower().strip()
    if m not in _INTERPOLATION_METHODS:
        raise ValueError(
            f"วิธี Interpolation '{method}' ไม่รองรับ (เลือกได้: 'nearest', 'bilinear', 'bicubic')"
        )
    return _INTERPOLATION_METHODS[m]


def rotate_image(
    img: np.ndarray,
    angle: float,
    center: tuple[float, float] | None = None,
    scale: float = 1.0,
    method: str = "bilinear",
) -> np.ndarray:
    """
    หมุนภาพตามมุม angle (องศา) รอบจุด center
    """
    if img is None or img.size == 0:
        raise ValueError("img ต้องไม่ว่าง")

    h, w = img.shape[:2]
    if center is None:
        center = (w / 2.0, h / 2.0)

    interp = _get_interp_flag(method)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, rot_mat, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def scale_image(
    img: np.ndarray,
    scale_factor: float,
    method: str = "bilinear",
) -> np.ndarray:
    """
    ปรับขนาด/ซูมภาพด้วยสเกลแฟกเตอร์ (เช่น 0.5 ย่อลงครึ่งหนึ่ง, 1.5 ขยาย 1.5 เท่า)
    """
    if img is None or img.size == 0:
        raise ValueError("img ต้องไม่ว่าง")
    if scale_factor <= 0:
        raise ValueError("scale_factor ต้องมากกว่า 0")

    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale_factor)))
    new_h = max(1, int(round(h * scale_factor)))
    interp = _get_interp_flag(method)
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def translate_image(
    img: np.ndarray,
    tx: float,
    ty: float,
    method: str = "bilinear",
) -> np.ndarray:
    """
    เลื่อนตำแหน่งภาพในแนวแกน X (tx พิกเซล) และแนวแกน Y (ty พิกเซล)
    """
    if img is None or img.size == 0:
        raise ValueError("img ต้องไม่ว่าง")

    h, w = img.shape[:2]
    trans_mat = np.array([[1.0, 0.0, float(tx)], [0.0, 1.0, float(ty)]], dtype=np.float32)
    interp = _get_interp_flag(method)
    return cv2.warpAffine(img, trans_mat, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def affine_warp(
    img: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    output_size: tuple[int, int] | None = None,
    method: str = "bilinear",
) -> np.ndarray:
    """
    ทำ Affine Transformation จากจุดอ้างอิง 3 จุดต้นทางและปลายทาง (Forward / Reverse Mapping)
    """
    if img is None or img.size == 0:
        raise ValueError("img ต้องไม่ว่าง")

    h, w = img.shape[:2]
    out_sz = output_size if output_size is not None else (w, h)

    pts_src = np.float32(src_points)
    pts_dst = np.float32(dst_points)

    matrix = cv2.getAffineTransform(pts_src, pts_dst)
    interp = _get_interp_flag(method)
    return cv2.warpAffine(img, matrix, out_sz, flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

