# ------------------------------------------------------------------------------
# File: utils/color.py
# Authors: Oliver Fritsche, Vaibhav Thalanki, Sai Manichandana Devi Thumati
# Emails: fritsche.o@northeastern.edu, thalanki.v@northeastern.edu, thumati.sa@northeastern.edu
# Date: 2025-10-21
# Class: CS 7180 Advanced Perception
# Purpose: Shared color/array helpers (sRGB↔linear, normalization, dtype utils)
# ------------------------------------------------------------------------------

import numpy as np

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Converts sRGB values in [0, 1] to linear RGB using standard gamma correction."""
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1.0 + a))**2.4)

def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Converts linear RGB values back to perceptual sRGB space using inverse gamma mapping."""
    a = 0.055
    return np.where(x <= 0.0031308, 12.92*x, (1.0 + a) * np.power(x, 1/2.4) - a)

def to_float01(x: np.ndarray) -> np.ndarray:
    """Converts any uint8 or float image to normalized float [0, 1], auto-detecting the correct scale."""
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    x = x.astype(np.float32)
    return np.clip(x, 0.0, 1.0) if np.nanmax(x) <= 1.5 else np.clip(x/255.0, 0.0, 1.0)

def to_u8(x: np.ndarray) -> np.ndarray:
    """Scales normalized float [0, 1] values to uint8 [0–255] for image output."""
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)

def norm01(x: np.ndarray) -> np.ndarray:
    """Normalizes an arbitrary numeric array to [0, 1] for safe visualization or debugging."""
    x = x.astype(np.float32)
    m = float(x.max()) if x.size else 1.0
    return x / (m + 1e-8)
