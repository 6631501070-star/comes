"""module1/run_test.py — ทดสอบฟังก์ชันของ Module 1 ทั้งหมดและบันทึกภาพลง results_module1/

วิธีรัน:
    python -m module1.run_test
"""

from pathlib import Path
import cv2
import numpy as np

from module1.histogram import equalize_histogram, adjust_contrast_brightness
from module1.spatial_filters import (
    mean_filter,
    median_filter,
    laplacian_sharpen,
    highpass_sharpen,
)
from module1.frequency_filters import (
    fft_lowpass,
    fft_highpass,
    fft_bandpass,
)

_ROOT = Path(__file__).resolve().parent.parent


def _save(img: np.ndarray, out_dir: Path, name: str) -> None:
    path = out_dir / f"{name}.png"
    # ถ้าเป็นภาพสี RGB ให้แปลงเป็น BGR ก่อน imwrite
    if img.ndim == 3 and img.shape[2] == 3:
        to_write = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        to_write = img
    cv2.imwrite(str(path), to_write)
    print(f"Saved: {path.name}")


def main() -> None:
    test_img_path = _ROOT / "test_hand.jpg"
    out_dir = _ROOT / "results_module1"
    out_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(test_img_path))
    if bgr is None:
        raise FileNotFoundError(f"ไม่พบไฟล์: {test_img_path}")

    # ย่อรูปเล็กลงเพื่อให้รัน FFT ได้รวดเร็ว
    h, w = bgr.shape[:2]
    if max(h, w) > 800:
        scale = 800.0 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    _save(rgb, out_dir, "01_original")

    # 1. Histogram Processing
    eq = equalize_histogram(rgb)
    _save(eq, out_dir, "02_hist_equalized")

    contrast = adjust_contrast_brightness(rgb, alpha=1.3, beta=20)
    _save(contrast, out_dir, "03_contrast_brightness")

    # 2. Spatial Filtering
    mean = mean_filter(rgb, ksize=5)
    _save(mean, out_dir, "04_spatial_mean")

    median = median_filter(rgb, ksize=5)
    _save(median, out_dir, "05_spatial_median")

    lap = laplacian_sharpen(rgb, strength=0.6)
    _save(lap, out_dir, "06_spatial_laplacian")

    hp = highpass_sharpen(rgb)
    _save(hp, out_dir, "07_spatial_highpass")

    # 3. Frequency Domain Filtering (Fourier)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _save(gray, out_dir, "08_gray_input")

    fft_lp = fft_lowpass(gray, cutoff=40)
    _save(fft_lp, out_dir, "09_fft_lowpass")

    fft_hp = fft_highpass(gray, cutoff=30)
    _save(fft_hp, out_dir, "10_fft_highpass")

    fft_bp = fft_bandpass(gray, low_cutoff=15, high_cutoff=50)
    _save(fft_bp, out_dir, "11_fft_bandpass")

    print(f"\nผลลัพธ์ของ Module 1 ทั้งหมดถูกบันทึกไว้ใน:\n  {out_dir}")


if __name__ == "__main__":
    main()

