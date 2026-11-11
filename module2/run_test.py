"""ทดสอบ Module 2 แบบเร็ว: รัน pipeline แล้วเซฟรูปทีละขั้นลง results_py/

วิธีใช้ (จาก root โปรเจกต์):
    python -m module2.run_test

หรือ:
    python module2/run_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# ให้รันได้ทั้งแบบ -m และแบบสคริปต์ตรงๆ
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from module2.apply_morphology import apply_morphology
from module2.detect_edges_all import detect_edges_all
from module2.segment_otsu import segment_otsu
from module2.segment_skin_hsv import segment_skin_hsv


def _save(img: np.ndarray, out_dir: Path, stem: str) -> None:
    """เซฟภาพลงดิสก์ (bool → ขาวดำ 0/255)."""
    path = out_dir / f"{stem}.png"
    if img.dtype == bool or img.dtype == np.bool_:
        to_write = (img.astype(np.uint8) * 255)
    elif img.ndim == 2:
        to_write = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
    else:
        # RGB → BGR สำหรับ imwrite
        to_write = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else img
    cv2.imwrite(str(path), to_write)
    print(f"Saved: {path}")


def main() -> None:
    input_image = _ROOT / "test_hand.jpg"
    out_dir = _ROOT / "results_py"
    out_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(input_image))
    if bgr is None:
        raise FileNotFoundError(f"หาไฟล์ไม่เจอ: {input_image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 1) ต้นฉบับ
    _save(rgb, out_dir, "01_original_rgb")

    # 2) HSV skin — ค่าเดียวกับ run_test.m
    limits = {
        "Hmin": 0.0,
        "Hmax": 0.12,
        "Smin": 0.15,
        "Smax": 0.75,
        "Vmin": 0.15,
        "Vmax": 1.0,
    }
    mask, skin = segment_skin_hsv(rgb, limits)
    _save(mask, out_dir, "02_skin_binary_mask")
    _save(skin, out_dir, "03_segmented_skin")

    # 3) Otsu (ทางเลือก / เทียบรายงาน)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bw, level = segment_otsu(gray)
    _save(bw, out_dir, "04_otsu")
    print(f"Otsu level ≈ {level:.4f}")

    # 4) Morphology (+ เก็บก้อนใหญ่สุด)
    morph = apply_morphology(mask, se_shape="disk", se_size=5, keep_largest=True)
    _save(morph["cleaned_mask"], out_dir, "05_cleaned_mask")

    # 5) Edges บนมาสก์ที่สะอาดแล้ว
    edges = detect_edges_all(morph["cleaned_mask"])
    _save(edges["sobel_edges"], out_dir, "06_sobel")
    _save(edges["prewitt_edges"], out_dir, "07_prewitt")
    _save(edges["canny_edges"], out_dir, "08_canny")

    print(f"\nAll figures saved in:\n  {out_dir}")


if __name__ == "__main__":
    main()
