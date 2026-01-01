"""Risk mapping helpers used in v1.1 risk-aware navigation.

This module is intentionally lightweight (NumPy-only) so it can be imported in
unit tests without pulling ROS/Habitat heavy dependencies. It mirrors the logic
used in habitat_evaluation.py for computing risk values and selecting publish
indices under a base threshold, optional quantile target, and top-K capping.
"""

from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np


def compute_risk_from_voxels(
    centroids: np.ndarray,
    voxel_conf: np.ndarray,
    disputed_list: Iterable[Tuple[np.ndarray, int, float]] | None,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute risk and disputed mask for voxel centroids.

    risk = alpha * (1 - voxel_conf) + beta * is_disputed

    - centroids: (N,3) float32/float64
    - voxel_conf: (N,) in [0,1]
    - disputed_list: iterable of (centroid(3,), label_id, score) for disputed voxels
    Returns: (risk:(N,), is_disputed:(N,)bool)
    """
    if centroids is None:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=bool)
    C = int(np.asarray(centroids).shape[0])
    if C == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=bool)

    conf = np.asarray(voxel_conf, dtype=np.float32).reshape(-1)
    if conf.shape[0] != C:
        conf = np.zeros((C,), dtype=np.float32)

    is_disp = np.zeros((C,), dtype=bool)
    if disputed_list:
        # Use raw bytes of float32 centroids for exact match (same storage as exporter)
        disp_keys = set()
        for cen, _, _ in disputed_list:
            try:
                k = np.asarray(cen, dtype=np.float32).reshape(3).tobytes()
            except Exception:
                continue
            disp_keys.add(k)
        for i in range(C):
            try:
                k = np.asarray(centroids[i], dtype=np.float32).reshape(3).tobytes()
            except Exception:
                continue
            if k in disp_keys:
                is_disp[i] = True

    risk = float(alpha) * (1.0 - conf) + float(beta) * is_disp.astype(np.float32)
    return risk.astype(np.float32), is_disp


def compute_dynamic_threshold(
    risk_vals: np.ndarray, base_threshold: float, quantile_target: float
) -> float:
    """Compute dynamic threshold = max(base_threshold, np.quantile(risk, q)) when 0<q<=1.

    - If risk_vals is empty or q<=0, returns base_threshold.
    - Clamps q to [0,1].
    """
    if risk_vals is None or risk_vals.size == 0:
        return float(base_threshold)
    q = float(quantile_target)
    if not (0.0 < q <= 1.0):
        return float(base_threshold)
    qv = float(np.quantile(np.asarray(risk_vals, dtype=np.float32), q))
    return float(max(base_threshold, qv))


def select_publish_indices(
    risk_vals: np.ndarray, dyn_threshold: float, max_points: int
) -> np.ndarray:
    """Return indices of points to publish sorted by risk descending.

    - Keep points with risk > dyn_threshold; if `max_points>0`, keep top-K.
    - Returns np.ndarray[int] possibly empty.
    """
    if risk_vals is None or risk_vals.size == 0:
        return np.empty((0,), dtype=np.int64)
    r = np.asarray(risk_vals, dtype=np.float32).reshape(-1)
    idx_all = np.nonzero(r > float(dyn_threshold))[0]
    if idx_all.size == 0:
        return idx_all.astype(np.int64)
    # Sort by risk descending
    idx_sorted = idx_all[np.argsort(-r[idx_all])]
    if int(max_points) > 0 and idx_sorted.size > int(max_points):
        idx_sorted = idx_sorted[: int(max_points)]
    return idx_sorted.astype(np.int64)

