# ------------------------------------------------------------------------------
# File: relight/point.py
# Authors: Oliver Fritsche, Vaibhav Thalanki, Sai Manichandana Devi Thumati
# Emails: fritsche.o@northeastern.edu, thalanki.v@northeastern.edu, thumati.sa@northeastern.edu
# Date: 2025-10-21
# Class: CS 7180 Advanced Perception
# Purpose: Point-light shading relight + simple light detection from shading
# ------------------------------------------------------------------------------

import numpy as np
import cv2
from utils.color import to_float01

def detect_light_from_shading(S_gray: np.ndarray) -> tuple[float, float]:
    """Estimate light-source center from shading via Gaussian blur and brightest-pixel detection."""
    S = to_float01(S_gray)
    h, w = S.shape[:2]
    k = max(3, int(round(min(h, w)*0.03)) | 1)
    Sb = cv2.GaussianBlur(S, (k, k), 0)
    _, _, _, maxLoc = cv2.minMaxLoc(Sb.astype(np.float32))
    return float(maxLoc[0]), float(maxLoc[1])

def relight_shading_point(S_gray: np.ndarray, cx: float, cy: float,
                          strength: float = 1.25, falloff: float = 1.8,
                          global_gain: float = 1.10, gamma: float = 0.95) -> np.ndarray:
    """Relight shading with a point-light model centered at (cx, cy) using distance-based falloff and gamma-adjusted gain."""

    # (A) Normalize to [0..1]
    S = to_float01(S_gray)
    h, w = S.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx - cx) / (0.5*w)
    dy = (yy - cy) / (0.5*h)
    r2 = dx*dx + dy*dy
    # (B) Distance-based light
    light = 1.0 + strength / (1.0 + falloff * r2)
    S_rel = S * light
    # (C) Exposure & gamma
    S_rel = np.power(np.clip(S_rel * global_gain, 1e-6, 50.0), gamma)
    # (D) Clamp for headroom
    return np.clip(S_rel, 0.0, 2.0)
