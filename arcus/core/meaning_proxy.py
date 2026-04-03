from __future__ import annotations

import math
from typing import Any, List, Optional

import numpy as np

_EPS = 1e-9

def _safe_histogram_entropy(col: np.ndarray, n_bins: int) -> float:
    col = np.asarray(col, dtype=np.float64)
    col = col[np.isfinite(col)]
    n   = len(col)
    if n == 0:
        return 0.0

    vmin, vmax = float(col.min()), float(col.max())
    if vmax - vmin < _EPS:
        return 0.0

    n_bins_use = max(2, min(n_bins, n // 2))
    try:
        counts, _ = np.histogram(col, bins=n_bins_use,
                                 range=(vmin - _EPS, vmax + _EPS))
    except Exception:
        return 0.0

    counts = counts[counts > 0]
    if counts.sum() == 0:
        return 0.0

    probs = counts / counts.sum()
    H     = float(-np.sum(probs * np.log(probs + 1e-12)))
    H_max = math.log(n_bins_use) if n_bins_use > 1 else 1.0
    return float(np.clip(H / (H_max + _EPS), 0.0, 1.0))


def _discrete_normalised_entropy(actions_arr: np.ndarray) -> float:
    try:
        flat = actions_arr.flatten().astype(np.int64)
        unique, counts = np.unique(flat, return_counts=True)
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts / total
        H     = float(-np.sum(probs * np.log(probs + 1e-12)))
        n_u   = max(len(unique), 1)
        H_max = math.log(n_u) if n_u > 1 else 1.0
        return float(np.clip(H / (H_max + _EPS), 0.0, 1.0))
    except Exception:
        return 0.0


def _pca_whitened_entropy(arr: np.ndarray, n_bins: int) -> float:
    T, D = arr.shape
    if T < 2 or D < 2:
        return float(np.clip(
            np.mean([_safe_histogram_entropy(arr[:, d], n_bins) for d in range(D)]),
            0.0, 1.0))

    try:
        X  = arr - arr.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        Z  = X @ Vt.T
        n_pcs = Z.shape[1]
        H_sum = sum(_safe_histogram_entropy(Z[:, k], n_bins) for k in range(n_pcs))
        return float(np.clip(H_sum / max(n_pcs, 1), 0.0, 1.0))
    except Exception:
        return float(np.clip(
            np.mean([_safe_histogram_entropy(arr[:, d], n_bins) for d in range(D)]),
            0.0, 1.0))

class MeaningProxyTracker:
    def __init__(
        self,
        n_bins:    int            = 10,
        alpha:     Optional[float] = None,
        ema_alpha: float           = 0.1,
        use_joint: bool            = True,
    ):
        self.n_bins    = int(max(2, n_bins))
        self._alpha    = alpha
        self.ema_alpha = float(ema_alpha)
        self.use_joint = bool(use_joint)
        self._pre_entropy_mean: Optional[float] = None
        self._pre_entropy_m2:   float = 0.0
        self._pre_entropy_var:  float = 0.0
        self._pre_n:            int   = 0
        self._alpha_eff: float = 3.0

    def _classify(self, actions_arr: np.ndarray) -> str:
        if actions_arr.dtype.kind in ("i", "u"):
            return "discrete"
        if actions_arr.ndim == 0:
            return "discrete"

        flat = actions_arr.flatten()
        if flat.size == 0:
            return "discrete"

        if actions_arr.dtype.kind == "f":
            try:
                n_dims = actions_arr.reshape(len(actions_arr), -1).shape[1]
            except Exception:
                n_dims = 1
            if n_dims >= 2:
                return "continuous"
            if np.all(np.isfinite(flat)) and np.all(flat == np.floor(flat)):
                return "discrete"
            value_range = float(flat.max() - flat.min()) if flat.size > 1 else 0.0
            if value_range > 1e-3:
                return "continuous"

        return "discrete"

    def _action_entropy(self, actions: List[Any]) -> float:
        if not actions:
            return 0.0

        try:
            actions_arr = np.asarray(actions)
        except Exception:
            return 0.0

        action_type = self._classify(actions_arr)

        if action_type == "discrete":
            return _discrete_normalised_entropy(actions_arr)

        try:
            arr = actions_arr.reshape(len(actions), -1).astype(np.float64)
        except Exception:
            return 0.0

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        T, D = arr.shape
        if T == 0:
            return 0.0

        if D == 1:
            return _safe_histogram_entropy(arr[:, 0], self.n_bins)

        if self.use_joint:
            return _pca_whitened_entropy(arr, self.n_bins)
        else:
            return float(np.clip(
                np.mean([_safe_histogram_entropy(arr[:, d], self.n_bins)
                         for d in range(D)]),
                0.0, 1.0))

    def _update_pre_stats(self, H: float) -> None:
        self._pre_n += 1
        prev_mean    = (self._pre_entropy_mean
                        if self._pre_entropy_mean is not None else H)

        self._pre_entropy_mean = ((1.0 - self.ema_alpha) * prev_mean
                                  + self.ema_alpha * H)

        delta               = H - prev_mean
        delta2              = H - self._pre_entropy_mean
        self._pre_entropy_m2 += delta * delta2
        if self._pre_n > 1:
            self._pre_entropy_var = self._pre_entropy_m2 / (self._pre_n - 1)

        if self._alpha is None and self._pre_n >= 5:
            std        = math.sqrt(max(self._pre_entropy_var, 1e-8))
            p95_excess = 1.645 * std
            self._alpha_eff = (math.log(20.0) / p95_excess
                               if p95_excess > 1e-6 else 3.0)

    def compute(self, actions: List[Any], phase: str) -> float:
        H = self._action_entropy(actions)

        if phase == "pre":
            self._update_pre_stats(H)
            return 1.0

        if self._pre_entropy_mean is None:
            return 1.0

        alpha  = self._alpha if self._alpha is not None else self._alpha_eff
        excess = max(0.0, H - self._pre_entropy_mean)
        return float(np.clip(math.exp(-alpha * excess), 0.0, 1.0))

    def reset(self) -> None:
        self._pre_entropy_mean = None
        self._pre_entropy_m2   = 0.0
        self._pre_entropy_var  = 0.0
        self._pre_n            = 0
        self._alpha_eff        = 3.0

    @property
    def pre_entropy_mean(self) -> Optional[float]:
        return self._pre_entropy_mean

    @property
    def effective_alpha(self) -> float:
        return self._alpha if self._alpha is not None else self._alpha_eff

    def __repr__(self) -> str:
        mean_str = (f"{self._pre_entropy_mean:.4f}"
                    if self._pre_entropy_mean is not None else "N/A")
        return (
            f"MeaningProxyTracker(v2.0  "
            f"pre_H_mean={mean_str}  "
            f"alpha_eff={self._alpha_eff:.3f}  "
            f"pre_n={self._pre_n}  "
            f"use_joint={self.use_joint})"
        )
