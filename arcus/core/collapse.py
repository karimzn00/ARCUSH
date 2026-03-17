# arcus/core/collapse.py
"""
Formula-based continuous collapse scoring.

Architecture: direct robust z-scoring of raw channels
------------------------------------------------------
Previous versions computed deficits (max(0, soft_threshold - channel))
and z-scored those. This produced MAD = 0 for all channels in every
stress-free baseline, because:

  soft_threshold = p20(channel in baseline pre)
  =>  80% of baseline episodes have channel >= soft  =>  deficit = 0
  =>  median(deficit) = 0
  =>  MAD(deficit) = median(|deficit - 0|) = median(deficit) = 0

This is a mathematical certainty, independent of sample size. Adding
more reference episodes does not fix it.

Fix: z-score the raw channel values directly against their baseline
distribution, exactly like a standard control chart or one-class
classifier:

  z_integrity = (baseline_median_integrity - shock_integrity)
                / (1.4826 * baseline_MAD_integrity)
  z_id_drop   = (shock_id_drop - baseline_median_id_drop)
                / (1.4826 * baseline_MAD_id_drop)

Sign convention: positive z means *worse* than baseline median.
  - integrity drops under stress -> baseline_med - shock > 0 -> z > 0 ✓
  - id_drop grows under stress   -> shock - baseline_med > 0 -> z > 0 ✓

For meaning: it is structurally 1.0 in every stress-free baseline
(no violations, no regret) so its MAD is always 0 regardless of
sample size. We retain the zero-MAD fallback for this channel only:

  z_meaning = 6 * clip((1 - meaning) / 1.0, 0, 1)

This is not a hack — it is the correct treatment of a channel whose
natural state is a degenerate point mass at 1.0. The fallback maps the
full range of meaning deficit [0, 1] onto z ∈ [0, 6] using the channel's
known theoretical range as reference scale.

Paper justification
-------------------
"We score each episode using a weighted sum of per-channel robust
z-scores calibrated against the pre-phase baseline distribution of
the same (seed, eval_mode) run. For channels with non-zero baseline
MAD (integrity, identity drop), we use the standard robust z-score
(median ± 1.4826·MAD). For the meaning channel, whose baseline
distribution is degenerate at 1.0 in the absence of violations, we
use a direct normalisation against the channel's theoretical range.
Scores are passed through a logistic centred at the median of baseline
pre-phase scores, so a score of 0.5 corresponds to the median baseline
episode and higher scores indicate increasing departure from normal
operation. An episode is classified as a collapse event if its score
exceeds the 95th percentile of baseline pre-phase scores (false
positive rate ≈ 5% on baseline by construction)."

Inputs
------
  meaning, integrity : float in [0, 1]
  id_drop            : float >= 0  (baseline_mean_identity - current_identity)
  baseline_stats     : dict produced by run_eval._compute_baseline_stats
                       If None, principled defaults are used.

Output
------
  collapse_score  in [0, 1]   higher = worse
  collapse_event  bool        score >= event_threshold
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import math
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CollapseScoringConfig:
    """
    event_threshold : binary threshold. When using the recommended adaptive
                      p95 workflow, this is set per (seed, eval_mode) by
                      run_eval and is not a free parameter.
    sharpness       : logistic steepness around the data-derived center.
                      Higher = more binary discrimination. Default 2.5 is
                      validated to give good spread across stressor severities.
    """
    event_threshold: float = 0.60
    sharpness: float = 2.5


# ---------------------------------------------------------------------------
# Meaning channel: zero-MAD fallback constants
# Meaning is structurally 1.0 in every stress-free baseline.
# Reference scale = 1.0 (full theoretical range of the meaning deficit).
# Weight factor = 0.4 (down-weight because this channel is presence/absence
#                      only — it cannot signal relative severity).
# ---------------------------------------------------------------------------
_MEANING_REF_SCALE   : float = 1.0
_MEANING_ZERO_MAD_W  : float = 0.4
_ZERO_MAD_THRESH     : float = 1e-6


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _ff(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return float(default) if not np.isfinite(v) else v
    except Exception:
        return float(default)


def _sigmoid(x: float) -> float:
    x = float(np.clip(_ff(x), -500.0, 500.0))
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _robust_z(val: float, med: float, mad: float) -> float:
    """Standard robust z-score: (val - median) / (1.4826 * MAD)."""
    return (float(val) - float(med)) / (1.4826 * float(mad) + 1e-8)


def _weight_from_mad(mad: float) -> float:
    """Channel weight = sqrt(precision) = sqrt(1 / scale), capped for stability."""
    return math.sqrt(min(1.0 / (1.4826 * float(mad) + 1e-8), 1e6))


# ---------------------------------------------------------------------------
# Baseline stats accessors
# ---------------------------------------------------------------------------

def _get_raw_robust(
    baseline_stats: Optional[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Extract raw_robust stats from baseline_stats.
    Returns {'integrity': {median, mad}, 'id_drop': {median, mad}, 'meaning': {median, mad}}
    """
    _def_i = {"median": 0.6,  "mad": 0.10}   # typical integrity baseline
    _def_d = {"median": 0.0,  "mad": 0.03}   # typical id_drop baseline
    _def_m = {"median": 1.0,  "mad": 0.0}    # meaning is structurally 1.0
    try:
        rr = baseline_stats["collapse"]["raw_robust"]
        return {
            "integrity": {
                "median": _ff(rr["integrity"]["median"], _def_i["median"]),
                "mad":    _ff(rr["integrity"]["mad"],    _def_i["mad"]),
            },
            "id_drop": {
                # id_drop passed to collapse_score is always clipped to max(0,...),
                # so the baseline median must also be >= 0.  When the reference pass
                # stores the unclipped distribution (identity can exceed base_id),
                # the median can be slightly negative, which permanently biases z_d
                # positive at baseline and inflates FPR to 20-50%.  Clamp here so
                # the scoring function and the calibration are consistent.
                "median": max(0.0, _ff(rr["id_drop"]["median"], _def_d["median"])),
                "mad":    _ff(rr["id_drop"]["mad"],    _def_d["mad"]),
            },
            "meaning": {
                "median": _ff(rr["meaning"]["median"], 1.0),
                "mad":    _ff(rr["meaning"]["mad"],    0.0),
            },
        }
    except Exception:
        return {"integrity": dict(_def_i), "id_drop": dict(_def_d), "meaning": dict(_def_m)}


def _get_base_id(baseline_stats: Optional[Dict[str, Any]]) -> float:
    try:
        return _ff(baseline_stats["collapse"]["base_id"], 0.5)
    except Exception:
        return 0.5


def _get_center(baseline_stats: Optional[Dict[str, Any]]) -> float:
    try:
        c = _ff(baseline_stats["collapse"]["center"], 0.50)
        return float(np.clip(c, 0.10, 0.90))
    except Exception:
        return 0.50


# ---------------------------------------------------------------------------
# Per-channel z-score (private)
# ---------------------------------------------------------------------------

def _channel_z_and_weight(
    channel: str,
    val: float,
    med: float,
    mad: float,
) -> tuple[float, float]:
    """
    Returns (z, weight) for one channel.

    For integrity: val = integrity score, worse = lower, so z = (med - val) / scale
    For id_drop:   val = id_drop,         worse = higher, so z = (val - med) / scale
    Both are already oriented so positive z = worse than baseline.

    For meaning (MAD=0): zero-MAD fallback maps meaning deficit in [0,1] -> z in [0,6].
    """
    if channel == "meaning":
        # Structural zero-MAD: meaning is always 1.0 in stress-free baseline
        deficit = max(0.0, 1.0 - float(val))
        z = 6.0 * float(np.clip(deficit / _MEANING_REF_SCALE, 0.0, 1.0))
        return z, _MEANING_ZERO_MAD_W

    if mad >= _ZERO_MAD_THRESH:
        z = _robust_z(val, med, mad)
        z = float(np.clip(z, -6.0, 6.0))
        w = _weight_from_mad(mad)
    else:
        # Unexpected zero-MAD for integrity or id_drop (very unusual)
        # Fall back to conservative unit weight
        z = float(np.clip(val, -6.0, 6.0))
        w = _MEANING_ZERO_MAD_W
    return z, w


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collapse_score(
    *,
    meaning: float,
    integrity: float,
    id_drop: float,
    cfg: CollapseScoringConfig,
    baseline_stats: Optional[Dict[str, Any]] = None,
    **_ignored,
) -> float:
    """
    Continuous collapse severity in [0, 1].

    Channels and z-score orientation (positive z = worse):
      meaning   : z = 6 * clip((1-meaning)/1.0, 0, 1)          [zero-MAD fallback]
      integrity : z = (baseline_med_integrity - integrity)       [direct robust-z]
                    / (1.4826 * baseline_MAD_integrity)
      id_drop   : z = (id_drop - baseline_med_id_drop)           [direct robust-z]
                    / (1.4826 * baseline_MAD_id_drop)

    Final score = sigmoid(sharpness * (weighted_raw - center))
    where center = median(collapse_score on baseline pre-phase).
    """
    meaning   = float(np.clip(_ff(meaning,   0.0), 0.0, 1.0))
    integrity = float(np.clip(_ff(integrity, 0.0), 0.0, 1.0))
    id_drop   = float(max(0.0, _ff(id_drop, 0.0)))

    rr     = _get_raw_robust(baseline_stats)
    center = _get_center(baseline_stats)

    # meaning channel (zero-MAD fallback)
    z_m, w_m = _channel_z_and_weight(
        "meaning", meaning,
        med=rr["meaning"]["median"], mad=rr["meaning"]["mad"]
    )

    # integrity: worse = lower, so z = (baseline_med - val) / scale
    i_med = rr["integrity"]["median"]
    i_mad = rr["integrity"]["mad"]
    z_i, w_i = _channel_z_and_weight(
        "integrity", i_med - integrity,   # pre-negate: positive = worse
        med=0.0, mad=i_mad
    )

    # id_drop: worse = higher, z = (val - baseline_med) / scale
    d_med = rr["id_drop"]["median"]
    d_mad = rr["id_drop"]["mad"]
    z_d, w_d = _channel_z_and_weight(
        "id_drop", id_drop - d_med,        # pre-centre: positive = worse
        med=0.0, mad=d_mad
    )

    w_sum = w_m + w_i + w_d + 1e-12
    raw   = (
        (w_m / w_sum) * _sigmoid(1.25 * z_m) +
        (w_i / w_sum) * _sigmoid(1.25 * z_i) +
        (w_d / w_sum) * _sigmoid(1.25 * z_d)
    )

    score = _sigmoid(float(cfg.sharpness) * (float(raw) - center))
    return float(np.clip(score, 0.0, 1.0))


def collapse_event(score: float, cfg: CollapseScoringConfig) -> bool:
    """Binary collapse event: score >= event_threshold."""
    return bool(_ff(score, 0.0) >= _ff(cfg.event_threshold, 0.60))
