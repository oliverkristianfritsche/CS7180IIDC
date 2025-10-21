# ------------------------------------------------------------------------------
# File: intrinsic/decompose.py
# Coder: Vaibhav Thalanki, Oliver Fritsche
# Purpose: Intrinsic image decomposition via superpixels+clustering (Garces et al.)
# ------------------------------------------------------------------------------

import numpy as np
import cv2
from skimage import segmentation, color
from sklearn.cluster import KMeans
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# ---- small utilities ----------------------------------------------------------
def _safe_log(x, eps=1e-6):
    """Numerically safe natural log."""
    return np.log(np.clip(x, eps, None))

def _build_4nbr_edges(h: int, w: int) -> np.ndarray:
    """Grid edges for 4-neighborhood (right/down). Returns int64 [E, 2]."""
    idx = np.arange(h*w, dtype=np.int64).reshape(h, w)
    edges = []
    edges += list(zip(idx[:, :-1].ravel(), idx[:, 1:].ravel()))   # right
    edges += list(zip(idx[:-1, :].ravel(), idx[1:, :].ravel()))   # down
    return np.asarray(edges, dtype=np.int64)

# ---- superpixels + per-image KMeans ------------------------------------------
def superpixel_clusters(img_bgr: np.ndarray,
                        n_segments: int = 600,
                        compactness: float = 10.0,
                        K: int = 10,
                        random_state: int = 0):
    """
    Segment into SLIC superpixels, then KMeans on superpixel mean 'ab' (Lab).
    Returns: (labels[h,w], cluster_map[h,w])
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    labels = segmentation.slic(img_rgb, n_segments=n_segments,
                               compactness=compactness, start_label=0)
    sp_ids = np.unique(labels)
    lab = color.rgb2lab(img_rgb)
    ab  = lab[..., 1:3]

    # Feature = mean 'ab' per superpixel
    feats = np.vstack([ab[labels == sp].mean(axis=0) for sp in sp_ids])

    km = KMeans(n_clusters=K, n_init=10, random_state=random_state)
    sp_cluster = km.fit_predict(feats)

    # Expand cluster ids back to pixel map
    cluster_map = np.zeros_like(labels, dtype=np.int32)
    for sp, cid in zip(sp_ids, sp_cluster):
        cluster_map[labels == sp] = cid
    return labels, cluster_map

# ---- main decomposition -------------------------------------------------------
def intrinsic_decompose(img_bgr: np.ndarray,
                        n_segments: int = 600,
                        compactness: float = 10.0,
                        K: int = 10,
                        lambda_smooth: float = 0.2,
                        edge_sigma: float = 0.1,
                        anchor_w: float = 1e-3):
    """
    Compute reflectance R (BGR) and shading S (uint8 preview) by solving for log-shading.
    Major steps:
      1) SLIC + KMeans clusters (proxy for reflectance groups).
      2) Build pairwise terms: reflectance-consistency + edge-aware smooth shading.
      3) Solve sparse linear system for log-shading.
      4) Reconstruct R (Lab with new L) and S preview.
    """
    h, w, _ = img_bgr.shape
    N = h * w

    # -- Prepare luminance and grayscale edges ---------------------------------
    lab_cv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab_cv[..., 0] / 255.0
    l = _safe_log(L + 1e-3).ravel()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # -- Clustering per image ---------------------------------------------------
    _, cluster_map = superpixel_clusters(img_bgr, n_segments, compactness, K)
    cflat = cluster_map.ravel()

    # -- Grid edges and weights -------------------------------------------------
    edges = _build_4nbr_edges(h, w)
    i_idx, j_idx = edges[:, 0], edges[:, 1]

    g_i = gray.ravel()[i_idx]
    g_j = gray.ravel()[j_idx]
    grad = np.abs(g_i - g_j)
    w_smooth = np.exp(- (grad ** 2) / (2 * (edge_sigma ** 2))) + 1e-3

    same_refl = (cflat[i_idx] == cflat[j_idx]).astype(np.float32)
    w_refl = 2.0 * same_refl

    # -- Build linear system A s = b for log-shading s --------------------------
    A = lil_matrix((N, N), dtype=np.float32)
    b = np.zeros(N, dtype=np.float32)

    def add_pair(ii, jj, weight, d):
        """Add Laplacian-style pairwise term with optional offset d."""
        if weight <= 0: return
        A[ii, ii] += weight; A[jj, jj] += weight
        A[ii, jj] -= weight; A[jj, ii] -= weight
        b[ii] += weight * d; b[jj] -= weight * d

    # Reflectance constraints (match luminance differences inside same cluster)
    d_r = (l[i_idx] - l[j_idx])
    for ii, jj, ww, dd in zip(i_idx, j_idx, w_refl, d_r):
        add_pair(ii, jj, ww, dd)

    # Shading smoothness constraints
    for ii, jj, ww in zip(i_idx, j_idx, lambda_smooth * w_smooth):
        add_pair(ii, jj, ww, 0.0)

    # Anchors to fix global offset
    step = max(1, N // 5000)
    for idx in range(0, N, step):
        A[idx, idx] += anchor_w

    # -- Solve and reconstruct --------------------------------------------------
    s = spsolve(csr_matrix(A), b)   # log-shading
    r = l - s                       # log-reflectance

    S = np.exp(s).reshape(h, w)
    R_L = np.exp(r).reshape(h, w)

    lab_reflect = lab_cv.copy()
    lab_reflect[..., 0] = np.clip(R_L * 255.0, 0, 255)
    R_bgr = cv2.cvtColor(lab_reflect.astype(np.uint8), cv2.COLOR_LAB2BGR)

    S_vis = (S / (S.max() + 1e-8) * 255.0).astype(np.uint8)
    return R_bgr, S_vis
