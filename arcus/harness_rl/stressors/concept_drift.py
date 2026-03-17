# arcus/harness_rl/stressors/concept_drift.py
"""
Concept Drift Stressor — with auto-calibrated drift_scale.

Simulates gradual non-stationarity during the SHOCK phase by injecting
structured drift into observations and/or reward signals.

Unlike trust_violation (action corruption), this stressor leaves the action
channel untouched and instead models changes in the environment's observation
and reward dynamics — e.g. sensor drift, changing reward shaping.

Auto-calibration (recommended, default)
-----------------------------------------
Setting drift_scale_obs=None (default) enables auto-calibration. Before the
shock phase begins, run_eval feeds observations from the reference pass into
record_obs(), then calls calibrate(horizon, shock_episodes), which computes:

    drift_scale_obs = severity_k * sigma_obs / sqrt(T_shock)

where:
    sigma_obs  = mean per-dimension std of observations under the trained policy
                 (measured during the reference pass, free of any stressor)
    T_shock    = horizon * shock_episodes  (total steps in the shock window)
    severity_k = severity multiplier (default 1.0)
    drift_max  = max_drift_sigma * sigma_obs  (clips accumulated drift)

Mathematical basis
------------------
A directional random walk of T steps with per-step std s accumulates to
expected magnitude s*sqrt(T). Setting s = k*sigma_obs/sqrt(T_shock) means
that after the full shock window the expected accumulated drift magnitude is
exactly k*sigma_obs — i.e. k times the natural per-dimension observation std.

    k=0.5 : mild   (drift < natural obs variance)
    k=1.0 : medium (drift matches natural obs variance at end of shock)
    k=2.0 : severe (drift is 2x natural obs variance at end of shock)

This makes concept_drift severity proportional to the environment's natural
observation scale, ensuring comparability across environments with very
different observation ranges (e.g. Pendulum vs HalfCheetah).

Paper justification (reviewers):
    "drift_scale is calibrated per environment as k*sigma_obs/sqrt(T_shock),
     where sigma_obs is the empirical per-dimension standard deviation of
     observations under the trained policy during the pre-stress reference
     pass, T_shock is the total shock-window step count, and k is a severity
     multiplier. We report results at k=1 (drift accumulates to one standard
     deviation of natural observation variance over the shock window)."

Manual override
---------------
Set drift_scale_obs to a positive float to bypass calibration entirely.
Useful for ablations. drift_max_manual is used as the clip limit in this mode.

Observation drift application
------------------------------
When calibrated, drift is applied per-dimension scaled by the dimension's
own std (obs_std_vec). A dimension with range [-8, 8] gets proportionally
more drift than one with range [-1, 1], so the effective perturbation is
uniform in standardised space.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from .base import BaseStressor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ConceptDriftConfig:
    # None = auto-calibrate (recommended). Positive float = manual override.
    drift_scale_obs: Optional[float] = None

    # Severity multiplier k in the formula drift_scale = k*sigma_obs/sqrt(T_shock).
    # Ignored when drift_scale_obs is set manually.
    severity_k: float = 1.0

    # Number of calibration steps. More => more accurate sigma_obs estimate.
    calib_steps: int = 200

    # Clip limit in units of sigma_obs (auto mode).
    max_drift_sigma: float = 3.0

    # Clip limit in raw units (manual mode only).
    drift_max_manual: float = 2.0

    # Reward noise standard deviation.
    drift_scale_reward: float = 0.05

    # If True, drift accumulates over shock steps (non-stationary progressive shift).
    # If False, each step gets independent noise (stationary noise injection).
    drift_directionality: bool = True


# ---------------------------------------------------------------------------
# Stressor
# ---------------------------------------------------------------------------

class ConceptDriftStressor(BaseStressor):
    """
    During SHOCK:
      obs    <- obs + drift_state * obs_std_vec   (auto mode, per-dim scaling)
              or obs + drift_state                 (manual mode, scalar shift)
      reward <- reward + N(0, drift_scale_reward)

    drift_state evolution (scalar):
      directionality=True:  drift_state += N(0, drift_scale_eff) each step,
                             clipped to [-drift_max_eff, +drift_max_eff]
      directionality=False: drift_state  = N(0, drift_scale_eff), no accumulation

    Calibration lifecycle:
      1. run_eval creates this stressor before the reference pass.
      2. _episode_rollout feeds each obs to record_obs() during the ref pass.
      3. After the ref pass, run_eval calls calibrate(horizon, shock_eps).
      4. calibrate() computes drift_scale_eff and drift_max_eff from sigma_obs.
      5. reset_drift() is called between schedule evaluations.

    Info keys emitted:
      concept_drift_magnitude  : abs(drift_state) at current step
      concept_drift_scale_eff  : effective drift_scale used this run
      stress_applied           : 1 during shock
    """
    name = "concept_drift"

    def __init__(self, cfg: Optional[ConceptDriftConfig] = None, seed: int = 0):
        self.cfg          = cfg or ConceptDriftConfig()
        self.rng          = np.random.default_rng(int(seed))
        self._drift_state: float = 0.0
        self._was_active:  bool  = False

        # Calibration state
        self._calib_obs:    List[np.ndarray]     = []
        self._calibrated:   bool                 = False
        self._obs_std:      Optional[float]      = None   # mean scalar sigma_obs
        self._obs_std_vec:  Optional[np.ndarray] = None   # per-dim sigma_obs

        # Effective parameters set by calibrate() or from config directly
        self._drift_scale_eff: Optional[float] = None
        self._drift_max_eff:   Optional[float] = None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def record_obs(self, obs: np.ndarray) -> None:
        """
        Accumulate an observation for sigma_obs estimation.
        Called by run_eval._episode_rollout during the reference pass.
        Safe to call after calibration is complete (silently ignored).
        """
        if self._calibrated:
            return
        if isinstance(obs, np.ndarray) and obs.ndim >= 1:
            self._calib_obs.append(obs.copy().astype(float).reshape(-1))

    def calibrate(self, horizon: int, shock_episodes: int) -> None:
        """
        Compute effective drift_scale from collected observations.

        Parameters
        ----------
        horizon        : median episode step length from the reference pass
        shock_episodes : number of shock-phase episodes in the eval schedule
        """
        if self._calibrated:
            return

        if self.cfg.drift_scale_obs is not None:
            # Manual override — use config directly
            self._drift_scale_eff = float(self.cfg.drift_scale_obs)
            self._drift_max_eff   = float(self.cfg.drift_max_manual)
            self._calibrated      = True
            return

        # Auto-calibration
        T_shock = max(1, int(horizon) * max(1, int(shock_episodes)))

        if len(self._calib_obs) >= 5:
            obs_mat = np.stack(self._calib_obs, axis=0)    # (N, obs_dim)
            per_dim_std = np.std(obs_mat, axis=0)           # (obs_dim,)
            # Guard against zero-variance dims
            per_dim_std = np.where(per_dim_std < 1e-6, 1e-6, per_dim_std)
            self._obs_std_vec = per_dim_std
            self._obs_std     = float(np.mean(per_dim_std))
        else:
            # Fallback when too few observations collected
            self._obs_std     = 1.0
            self._obs_std_vec = None

        k = float(self.cfg.severity_k)
        self._drift_scale_eff = k * self._obs_std / math.sqrt(T_shock)
        self._drift_max_eff   = float(self.cfg.max_drift_sigma) * self._obs_std
        self._calibrated      = True

    def reset_drift(self) -> None:
        """Reset accumulated drift. Call between independent evaluation runs."""
        self._drift_state = 0.0
        self._was_active  = False

    # ------------------------------------------------------------------
    # Parameter accessors (safe fallbacks before calibration)
    # ------------------------------------------------------------------

    def _get_drift_scale(self) -> float:
        if self._drift_scale_eff is not None:
            return self._drift_scale_eff
        if self.cfg.drift_scale_obs is not None:
            return float(self.cfg.drift_scale_obs)
        return 0.01  # conservative pre-calibration fallback

    def _get_drift_max(self) -> float:
        if self._drift_max_eff is not None:
            return self._drift_max_eff
        if self.cfg.drift_scale_obs is not None:
            return float(self.cfg.drift_max_manual)
        return 0.1   # conservative pre-calibration fallback

    # ------------------------------------------------------------------
    # Stressor hooks
    # ------------------------------------------------------------------

    def transform_action(
        self,
        action: Any,
        *,
        action_space: spaces.Space,
        active: bool,
        phase: str,
        info: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        info.setdefault("stress_applied",          0)
        info.setdefault("concept_drift_magnitude", 0.0)
        info.setdefault("concept_drift_scale_eff", self._get_drift_scale())

        # Reset drift when a new shock window begins
        if active and not self._was_active:
            self._drift_state = 0.0
        self._was_active = active

        if not active:
            info["concept_drift_magnitude"] = 0.0
            return action, info

        # concept_drift does not modify the action
        info["concept_drift_magnitude"] = float(abs(self._drift_state))
        info["concept_drift_scale_eff"] = self._get_drift_scale()
        info["stress_applied"]          = 1
        return action, info

    def transform_step(
        self,
        action: Any,
        obs: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
        *,
        action_space: spaces.Space,
        active: bool,
        phase: str,
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        info.setdefault("violation",               0.0)
        info.setdefault("regret",                  0.0)
        info.setdefault("stress_applied",          0)
        info.setdefault("concept_drift_magnitude", 0.0)
        info.setdefault("concept_drift_scale_eff", self._get_drift_scale())

        if not active:
            return obs, float(reward), bool(terminated), bool(truncated), info

        drift_scale = self._get_drift_scale()
        drift_max   = self._get_drift_max()

        # Update scalar drift state
        increment = float(self.rng.normal(0.0, drift_scale))
        if self.cfg.drift_directionality:
            self._drift_state = float(
                np.clip(self._drift_state + increment, -drift_max, drift_max)
            )
        else:
            self._drift_state = float(np.clip(increment, -drift_max, drift_max))

        # Apply observation drift
        if isinstance(obs, np.ndarray):
            if self._obs_std_vec is not None and obs.shape == self._obs_std_vec.shape:
                # Per-dim scaling: drift is measured in units of each dim's own std
                obs = obs + self._drift_state * self._obs_std_vec
            else:
                # Scalar uniform shift (manual mode or fallback)
                obs = obs + self._drift_state

        # Apply reward noise
        reward = float(reward) + float(self.rng.normal(0.0, self.cfg.drift_scale_reward))

        drift_mag = float(abs(self._drift_state))
        info["concept_drift_magnitude"] = drift_mag
        info["concept_drift_scale_eff"] = drift_scale
        info["regret"]                  = drift_mag  # drift magnitude as regret proxy
        info["stress_applied"]          = 1

        return obs, float(reward), bool(terminated), bool(truncated), info
