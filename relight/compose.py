# ------------------------------------------------------------------------------
# File: relight/compose.py
# Authors: Oliver Fritsche, Vaibhav Thalanki, Sai Manichandana Devi Thumati
# Emails: fritsche.o@northeastern.edu, thalanki.v@northeastern.edu, thumati.sa@northeastern.edu
# Date: 2025-10-21
# Class: CS 7180 Advanced Perception
# Purpose: Sepia on reflectance + linear-light recomposition (R * S)
# ------------------------------------------------------------------------------

import numpy as np
import cv2
from utils.color import srgb_to_linear, linear_to_srgb, to_float01, to_u8

def sepia_on_reflectance(R_bgr: np.ndarray, intensity: float = 0.85) -> np.ndarray:
    """Apply sepia tone to the reflectance image using weighted RGB blending."""
    R = to_float01(R_bgr)  # sRGB
    rgb = cv2.cvtColor((R*255).astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    M = np.array([[0.393, 0.769, 0.189],
                  [0.349, 0.686, 0.168],
                  [0.272, 0.534, 0.131]], dtype=np.float32)
    sep = np.tensordot(rgb, M.T, axes=1)
    sep = np.clip(sep, 0.0, 1.0)
    out = (1.0 - intensity)*rgb + intensity*sep
    return cv2.cvtColor(to_u8(out), cv2.COLOR_RGB2BGR)

def compose_intrinsic_linear(R_bgr_srgb_u8: np.ndarray, S_rel_float: np.ndarray,
                             auto_expose: bool = True,
                             tgt_percentile: float = 99.0,
                             tgt_level: float = 0.98) -> np.ndarray:
    """Recombine reflectance and shading in linear space, tone-map to sRGB, and auto-expose."""
    # (A)
    R_srgb = cv2.cvtColor(R_bgr_srgb_u8, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    R_lin  = srgb_to_linear(R_srgb)
    # (B)
    S = S_rel_float.astype(np.float32)
    valid = S > 1e-6
    med = np.median(S[valid]) if np.any(valid) else 1.0
    Sn = S / max(med, 1e-6)
    # (C)
    I_lin = np.clip(R_lin * Sn[..., None], 0.0, 8.0)
    # (D)
    if auto_expose:
        p = np.percentile(I_lin, tgt_percentile)
        scale = tgt_level / max(p, 1e-6)
        I_lin = np.clip(I_lin * scale, 0.0, 1.0)
    else:
        I_lin = np.clip(I_lin, 0.0, 1.0)
    # (E)
    I_srgb = linear_to_srgb(I_lin)
    I_u8   = to_u8(I_srgb)
    return cv2.cvtColor(I_u8, cv2.COLOR_RGB2BGR)
