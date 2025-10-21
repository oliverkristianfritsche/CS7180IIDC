# ------------------------------------------------------------------------------
# File: run_pipeline.py
# Authors: Oliver Fritsche, Vaibhav Thalanki, Sai Manichandana Devi Thumati
# Emails: fritsche.o@northeastern.edu, thalanki.v@northeastern.edu, thumati.sa@northeastern.edu
# Date: 2025-10-21
# Class: CS 7180 Advanced Perception
# Purpose: CLI to run intrinsic decomposition + sepia + relighting (chosen mode),
#          save outputs, and display a 2×3 matplotlib summary panel.
#          Modes supported: "point_auto", "directional_right"
# ------------------------------------------------------------------------------

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

from intrinsic.decompose import intrinsic_decompose
from relight.compose import sepia_on_reflectance, compose_intrinsic_linear
from relight.point import detect_light_from_shading, relight_shading_point
from relight.directional import relight_directional_right

# ----------------------------- small helpers ----------------------------------
def norm01(x: np.ndarray) -> np.ndarray:
    """Normalize any array to [0..1] for visualization."""
    x = x.astype(np.float32)
    m = float(x.max()) if x.size else 1.0
    return x / (m + 1e-8)

# ----------------------------- core runner ------------------------------------
def run_pipeline_cli(
    image_path: str,
    mode: str = "point_auto",         
    out_dir: str = "./outputs",
    k_clusters: int = 10,
    n_segments: int = 600,
    sepia_intensity: float = 0.85
):
    """
    End-to-end:
      1) Decompose (R, S)
      2) Sepia on reflectance
      3) Relight shading per 'mode' (point_auto or directional_right)
      4) Compose in linear
      5) Save outputs + show 2×3 panel
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- Load image ---
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read: {image_path}")

    # --- (1) Intrinsic decomposition ---
    R_bgr, S_vis = intrinsic_decompose(img_bgr, n_segments=n_segments, K=k_clusters)
    print("Decomposed image into reflectance and shading...")

    # --- (2) Sepia reflectance ---
    R_sepia = sepia_on_reflectance(R_bgr, intensity=sepia_intensity)
    print("Applied Sepia on Reflectance...")

    # --- (3) Relight shading (choose mode) ---
    if mode == "point_auto":
        cx, cy = detect_light_from_shading(S_vis)
        S_rel = relight_shading_point(
            S_vis, cx, cy,
            strength=1.6, falloff=1.8, global_gain=1.10, gamma=0.95
        )
    elif mode == "directional_right":
        S_rel = relight_directional_right(
            S_vis, strength=0.9, hardness=1.4, edge_bias=0.0,
            global_gain=1.10, gamma=0.94
        )
    else:
        raise ValueError("mode must be 'point_auto' or 'directional_right'")
    
    print("Applied relighting to shading...")

    # --- (4) Compose in linear ---
    I_bgr = compose_intrinsic_linear(R_sepia, S_rel, auto_expose=True)
    print("Composed modified components...")

    # --- (5) Save results ---
    cv2.imwrite(os.path.join(out_dir, "reflectance.png"), R_bgr)
    cv2.imwrite(os.path.join(out_dir, "reflectance_sepia.png"), R_sepia)
    cv2.imwrite(os.path.join(out_dir, "shading.png"), S_vis)
    cv2.imwrite(os.path.join(out_dir, f"shading_relit_{mode}.png"), (norm01(S_rel) * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(out_dir, f"combined_{mode}.png"), I_bgr)
    print(f"Saved results to {out_dir}")

    # --- (6) Show 2×3 matplotlib panel ---
    S_orig_vis = norm01(S_vis)
    S_rel_vis  = norm01(S_rel)
    S_delta    = np.clip(0.5 + 0.5 * (S_rel_vis - S_orig_vis), 0.0, 1.0)

    R_orig_rgb  = cv2.cvtColor(R_bgr,   cv2.COLOR_BGR2RGB)
    R_sepia_rgb = cv2.cvtColor(R_sepia, cv2.COLOR_BGR2RGB)
    I_rgb       = cv2.cvtColor(I_bgr,   cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 10))
    plt.subplot(2, 3, 1); plt.imshow(R_orig_rgb);             
    plt.title('Reflectance (original)'); plt.axis('off')
    plt.subplot(2, 3, 2); plt.imshow(R_sepia_rgb);            
    plt.title('Reflectance (sepia)');    plt.axis('off')
    plt.subplot(2, 3, 3); plt.imshow(S_orig_vis, cmap='gray');
    plt.title('Shading (original)');    
    plt.axis('off')
    plt.subplot(2, 3, 4); plt.imshow(S_rel_vis,  cmap='gray');
    plt.title(f'Shading (relit: {mode})'); plt.axis('off')
    plt.subplot(2, 3, 5); plt.imshow(I_rgb);                  
    plt.title('Combined');               
    plt.axis('off')
    plt.subplot(2, 3, 6); plt.imshow(S_delta, cmap='gray');   
    plt.title('Shading change');         
    plt.axis('off')
    plt.tight_layout(); 
    plt.show()

    return {"R_bgr": R_bgr, "S_vis": S_vis, "R_sepia": R_sepia, "S_rel": S_rel, "I_bgr": I_bgr, "out_dir": out_dir}

# ----------------------------- CLI entrypoint ---------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Intrinsic → Sepia → Relight → Compose")
    p.add_argument("image", help="Path to input image")
    p.add_argument("--mode", default="point_auto",
                   choices=["point_auto", "directional_right"],
                   help="Relighting mode")
    p.add_argument("--out_dir", default="./outputs", help="Output directory")
    p.add_argument("--clusters", type=int, default=10, help="KMeans clusters (K)")
    p.add_argument("--n_segments", type=int, default=600, help="SLIC superpixels")
    p.add_argument("--sepia", type=float, default=0.85, help="Sepia intensity (0..1)")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run_pipeline_cli(
        image_path=args.image,
        mode=args.mode,
        out_dir=args.out_dir,
        k_clusters=args.clusters,
        n_segments=args.n_segments,
        sepia_intensity=args.sepia,
    )