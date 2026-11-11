"""Module 1 — Frequency Domain Filtering (Fourier-based)

หน้าที่:
  1. Low-pass Filter (Gaussian): กรองสัญญาณรบกวนความถี่สูง ทำให้ภาพเนียนนุ่ม
  2. High-pass Filter (Gaussian): กรองเอาเฉพาะข้อมูลความถี่สูง (เน้นขอบ)
  3. Band-pass Filter: กรองเอาเฉพาะย่านความถี่ที่ต้องการ
"""

from __future__ import annotations
import cv2
import numpy as np


def _gaussian_distance_grid(rows: int, cols: int) -> np.ndarray:
    """คำนวณระยะห่าง D(u, v) จากจุดศูนย์กลางของสเปกตรัมความถี่"""
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    return np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)


def _apply_fft_filter_2d(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ทำ 2D FFT -> คูณ Filter Mask -> Inverse 2D FFT"""
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    filtered_shift = fshift * mask
    f_ishift = np.fft.ifftshift(filtered_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return np.clip(img_back, 0, 255).astype(np.uint8)


def _process_channels(img: np.ndarray, filter_func) -> np.ndarray:
    """รองรับทั้งภาพ 2D (Gray) และ 3D (RGB)"""
    if img.ndim == 2:
        return filter_func(img)
    elif img.ndim == 3 and img.shape[2] == 3:
        channels = [filter_func(img[:, :, c]) for c in range(3)]
        return np.stack(channels, axis=2)
    raise ValueError("img ต้องเป็นขนาด HxW หรือ HxWx3")


def fft_lowpass(img: np.ndarray, cutoff: float = 40.0) -> np.ndarray:
    """
    Gaussian Low-Pass Filter ในระนาบความถี่ (Fourier Domain)
    ช่วยลดความถี่สูง (Noise ความถี่สูงจากกล้อง)
    """
    if img is None or img.size == 0:
        raise ValueError("img ต้องไม่ว่าง")

    h, w = img.shape[:2]
    d = _gaussian_distance_grid(h, w)
    d0 = max(1.0, float(cutoff))
    # Gaussian LPF formula: H(u,v) = exp(-D^2 / (2 * D0^2))
    mask = np.exp(-(d ** 2) / (2 * (d0 ** 2)))

    return _process_channels(img, lambda ch: _apply_fft_filter_2d(ch, mask))


def fft_highpass(img: np.ndarray, cutoff: float = 30.0) -> np.ndarray:
    """
    Gaussian High-Pass Filter ในระนาบความถี่ (Fourier Domain)
    สกัดเฉพาะขอบและการเปลี่ยนแปลงเฉียบพลัน
    """
    if img is None or img.size == 0:
        raise ValueError("img ต้องไม่ว่าง")

    h, w = img.shape[:2]
    d = _gaussian_distance_grid(h, w)
    d0 = max(1.0, float(cutoff))
    # Gaussian HPF formula: H(u,v) = 1 - exp(-D^2 / (2 * D0^2))
    mask = 1.0 - np.exp(-(d ** 2) / (2 * (d0 ** 2)))

    return _process_channels(img, lambda ch: _apply_fft_filter_2d(ch, mask))


def fft_bandpass(
    img: np.ndarray,
    low_cutoff: float = 15.0,
    high_cutoff: float = 50.0,
) -> np.ndarray:
    """
    Band-Pass Filter ในระนาบความถี่ (Fourier Domain)
    ยอมให้เฉพาะความถี่ระหว่าง [low_cutoff, high_cutoff] ผ่าน
    """
    if img is None or img.size == 0:
        raise ValueError("img ต้องไม่ว่าง")

    h, w = img.shape[:2]
    d = _gaussian_distance_grid(h, w)
    d_lo = max(1.0, float(low_cutoff))
    d_hi = max(d_lo + 1.0, float(high_cutoff))

    # Mask = LPF(high) - LPF(low)
    mask = np.exp(-(d ** 2) / (2 * (d_hi ** 2))) - np.exp(-(d ** 2) / (2 * (d_lo ** 2)))
    mask = np.clip(mask, 0.0, 1.0)

    return _process_channels(img, lambda ch: _apply_fft_filter_2d(ch, mask))

