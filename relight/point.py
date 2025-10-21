# ------------------------------------------------------------------------------
# File: relight/point.py
# Coder: Vaibhav Thalanki
# Purpose: Point-light shading relight + simple light detection from shading
# ------------------------------------------------------------------------------

import numpy as np
import cv2
from utils.color import to_float01

def detect_light_from_shading(S_gray: np.ndarray) -> tuple[float, float]:
    """Blur shading and return brightest pixel as (cx, cy)."""
    S = to_float01(S_gray)
    h, w = S.shape[:2]
    k = max(3, int(round(min(h, w)*0.03)) | 1)
    Sb = cv2.GaussianBlur(S, (k, k), 0)
    _, _, _, maxLoc = cv2.minMaxLoc(Sb.astype(np.float32))
    return float(maxLoc[0]), float(maxLoc[1])

def relight_shading_point(S_gray: np.ndarray, cx: float, cy: float,
                          strength: float = 1.25, falloff: float = 1.8,
                          global_gain: float = 1.10, gamma: float = 0.95) -> np.ndarray:
    """
    Apply 2-D point-light falloff centered at (cx, cy) to shading.
    Sections:
      (A) Normalize coords; (B) build falloff; (C) exposure/gamma; (D) clamp.
    """
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
