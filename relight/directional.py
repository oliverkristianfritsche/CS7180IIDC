# ------------------------------------------------------------------------------
# File: relight/directional.py
# Authors: Oliver Fritsche, Vaibhav Thalanki, Sai Manichandana Devi Thumati
# Emails: fritsche.o@northeastern.edu, thalanki.v@northeastern.edu, thumati.sa@northeastern.edu
# Date: 2025-10-21
# Class: CS 7180 Advanced Perception
# Purpose: Directional light entering from the right edge (no brightest-pixel use)
# ------------------------------------------------------------------------------

import numpy as np
from utils.color import to_float01

def relight_directional_right(S_gray: np.ndarray,
                              strength: float = 0.9,
                              hardness: float = 1.4,
                              edge_bias: float = 0.0,
                              global_gain: float = 1.10,
                              gamma: float = 0.94) -> np.ndarray:
    """Simulate right-side directional lighting by modulating shading with a horizontal ramp and gamma-adjusted gain."""
    S = to_float01(S_gray)
    h, w = S.shape
    _, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # (A) 0..1 ramp across x
    ramp = xx / max(1.0, w - 1)
    ramp = np.clip(ramp - edge_bias, 0.0, 1.0)
    ramp = np.power(ramp, hardness)
    light = 1.0 + strength * ramp
    # (B)(C)(D)
    S_rel = S * light
    S_rel = np.power(np.clip(S_rel * global_gain, 1e-6, 50.0), gamma)
    return np.clip(S_rel, 0.0, 2.0)
