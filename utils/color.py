# ------------------------------------------------------------------------------
# File: utils/color.py
# Coder: Vaibhav Thalanki, Sai Manichandana
# Purpose: Shared color/array helpers (sRGB↔linear, normalization, dtype utils)
# ------------------------------------------------------------------------------

import numpy as np

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Convert sRGB [0..1] to linear RGB [0..1]."""
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1.0 + a))**2.4)

def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Convert linear RGB [0..1] to sRGB [0..1]."""
    a = 0.055
    return np.where(x <= 0.0031308, 12.92*x, (1.0 + a) * np.power(x, 1/2.4) - a)

def to_float01(x: np.ndarray) -> np.ndarray:
    """Convert uint8 or float array to float32 in [0..1] (auto-detect scale)."""
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    x = x.astype(np.float32)
    return np.clip(x, 0.0, 1.0) if np.nanmax(x) <= 1.5 else np.clip(x/255.0, 0.0, 1.0)

def to_u8(x: np.ndarray) -> np.ndarray:
    """Convert float [0..1] → uint8 [0..255] with clipping."""
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)

def norm01(x: np.ndarray) -> np.ndarray:
    """Normalize any array to [0..1] for visualization."""
    x = x.astype(np.float32)
    m = float(x.max()) if x.size else 1.0
    return x / (m + 1e-8)
