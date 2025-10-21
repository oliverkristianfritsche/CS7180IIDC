# ------------------------------------------------------------------------------
# File: relight/directional.py
# Coder: Vaibhav Thalanki
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
    """
    Apply a directional ramp from left→right (brighter on the right).
    Sections:
      (A) Build ramp; (B) apply ramp to shading; (C) exposure/gamma; (D) clamp.
    """
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
