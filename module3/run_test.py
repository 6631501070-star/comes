"""module3/run_test.py — ทดสอบ Geometric Transformations & Interpolation

เปรียบเทียบผลลัพธ์ของ Nearest-neighbor, Bilinear, Bicubic ตามข้อกำหนด comvis.pdf
"""

from pathlib import Path
import cv2
import numpy as np

from module3.geometric import (
    rotate_image,
    scale_image,
    translate_image,
    affine_warp,
)

_ROOT = Path(__file__).resolve().parent.parent


def _save(img: np.ndarray, out_dir: Path, name: str) -> None:
    path = out_dir / f"{name}.png"
    if img.ndim == 3 and img.shape[2] == 3:
        to_write = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        to_write = img
    cv2.imwrite(str(path), to_write)
    print(f"Saved: {path.name}")


def main() -> None:
    test_img_path = _ROOT / "test_hand.jpg"
    out_dir = _ROOT / "results_module3"
    out_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(test_img_path))
    if bgr is None:
        raise FileNotFoundError(f"ไม่พบไฟล์: {test_img_path}")

    # ย่อรูปให้อยู่ในขนาดพอเหมาะ
    h, w = bgr.shape[:2]
    if max(h, w) > 800:
        scale = 800.0 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    _save(rgb, out_dir, "01_original")

    # 1. เปรียบเทียบ Interpolation เมื่อหมุนภาพ 30 องศา
    for method in ["nearest", "bilinear", "bicubic"]:
        rot = rotate_image(rgb, angle=30.0, method=method)
        _save(rot, out_dir, f"02_rotate_30deg_{method}")

    # 2. เปรียบเทียบ Interpolation เมื่อปรับขนาด/ซูม (Scaling 1.5x)
    for method in ["nearest", "bilinear", "bicubic"]:
        sc = scale_image(rgb, scale_factor=1.5, method=method)
        _save(sc, out_dir, f"03_scale_1.5x_{method}")

    # 3. Translation (เลื่อนภาพ x=+50, y=+30)
    trans = translate_image(rgb, tx=50, ty=30, method="bilinear")
    _save(trans, out_dir, "04_translation")

    # 4. Affine Warping (ดัดภาพ 3 จุด)
    h_rgb, w_rgb = rgb.shape[:2]
    src_pts = np.float32([[0, 0], [w_rgb - 1, 0], [0, h_rgb - 1]])
    dst_pts = np.float32([[w_rgb * 0.1, h_rgb * 0.15], [w_rgb * 0.85, h_rgb * 0.05], [w_rgb * 0.2, h_rgb * 0.9]])
    warped = affine_warp(rgb, src_pts, dst_pts, method="bilinear")
    _save(warped, out_dir, "05_affine_warp")

    print(f"\nผลลัพธ์ของ Module 3 ทั้งหมดถูกบันทึกไว้ใน:\n  {out_dir}")


if __name__ == "__main__":
    main()

