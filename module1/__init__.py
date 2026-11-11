"""Module 1 — Image Enhancement & Filtering (Python OpenCV)

Student 1: การเตรียมและปรับปรุงคุณภาพภาพมือ
- Histogram Processing (Equalization, Contrast/Brightness)
- Spatial Filtering (Mean, Median, Laplacian, High-pass)
- Frequency Domain Filtering (Fourier-based Low-pass, High-pass, Band-pass)
"""

from .histogram import equalize_histogram, adjust_contrast_brightness
from .spatial_filters import (
    mean_filter,
    median_filter,
    laplacian_sharpen,
    highpass_sharpen,
)
from .frequency_filters import (
    fft_lowpass,
    fft_highpass,
    fft_bandpass,
)

__all__ = [
    "equalize_histogram",
    "adjust_contrast_brightness",
    "mean_filter",
    "median_filter",
    "laplacian_sharpen",
    "highpass_sharpen",
    "fft_lowpass",
    "fft_highpass",
    "fft_bandpass",
]

