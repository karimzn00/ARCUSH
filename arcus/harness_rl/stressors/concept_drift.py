from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from .base import BaseStressor

@dataclass
class ConceptDriftConfig:
    drift_scale_obs: Optional[float] = None
    severity_k: float = 1.0
    calib_steps: int = 200
    max_drift_sigma: float = 3.0
    drift_max_manual: float = 2.0
    drift_scale_reward: float = 0.05
    drift_directionality: bool = True

class ConceptDriftStressor(BaseStressor):
    name = "concept_drift"

    def __init__(self, cfg: Optional[ConceptDriftConfig] = None, seed: int = 0):
        self.cfg          = cfg or ConceptDriftConfig()
        self.rng          = np.random.default_rng(int(seed))
        self._drift_state: float = 0.0
        self._was_active:  bool  = False

        self._calib_obs:    List[np.ndarray]     = []
        self._calibrated:   bool                 = False
        self._obs_std:      Optional[float]      = None
        self._obs_std_vec:  Optional[np.ndarray] = None
        self._drift_scale_eff: Optional[float] = None
        self._drift_max_eff:   Optional[float] = None

    def record_obs(self, obs: np.ndarray) -> None:
        if self._calibrated:
            return
        if isinstance(obs, np.ndarray) and obs.ndim >= 1:
            self._calib_obs.append(obs.copy().astype(float).reshape(-1))

    def calibrate(self, horizon: int, shock_episodes: int) -> None:
        if self._calibrated:
            return

        if self.cfg.drift_scale_obs is not None:
            self._drift_scale_eff = float(self.cfg.drift_scale_obs)
            self._drift_max_eff   = float(self.cfg.drift_max_manual)
            self._calibrated      = True
            return

        T_shock = max(1, int(horizon) * max(1, int(shock_episodes)))

        if len(self._calib_obs) >= 5:
            obs_mat = np.stack(self._calib_obs, axis=0)
            per_dim_std = np.std(obs_mat, axis=0)
            per_dim_std = np.where(per_dim_std < 1e-6, 1e-6, per_dim_std)
            self._obs_std_vec = per_dim_std
            self._obs_std     = float(np.mean(per_dim_std))
        else:
            self._obs_std     = 1.0
            self._obs_std_vec = None

        k = float(self.cfg.severity_k)
        self._drift_scale_eff = k * self._obs_std / math.sqrt(T_shock)
        self._drift_max_eff   = float(self.cfg.max_drift_sigma) * self._obs_std
        self._calibrated      = True

    def reset_drift(self) -> None:
        self._drift_state = 0.0
        self._was_active  = False

    def _get_drift_scale(self) -> float:
        if self._drift_scale_eff is not None:
            return self._drift_scale_eff
        if self.cfg.drift_scale_obs is not None:
            return float(self.cfg.drift_scale_obs)
        return 0.01

    def _get_drift_max(self) -> float:
        if self._drift_max_eff is not None:
            return self._drift_max_eff
        if self.cfg.drift_scale_obs is not None:
            return float(self.cfg.drift_max_manual)
        return 0.1

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

        if active and not self._was_active:
            self._drift_state = 0.0
        self._was_active = active

        if not active:
            info["concept_drift_magnitude"] = 0.0
            return action, info
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
        increment = float(self.rng.normal(0.0, drift_scale))
        if self.cfg.drift_directionality:
            self._drift_state = float(
                np.clip(self._drift_state + increment, -drift_max, drift_max)
            )
        else:
            self._drift_state = float(np.clip(increment, -drift_max, drift_max))

        if isinstance(obs, np.ndarray):
            if self._obs_std_vec is not None and obs.shape == self._obs_std_vec.shape:
                obs = obs + self._drift_state * self._obs_std_vec
            else:
                obs = obs + self._drift_state

        reward = float(reward) + float(self.rng.normal(0.0, self.cfg.drift_scale_reward))

        drift_mag = float(abs(self._drift_state))
        info["concept_drift_magnitude"] = drift_mag
        info["concept_drift_scale_eff"] = drift_scale
        info["regret"]                  = drift_mag
        info["stress_applied"]          = 1

        return obs, float(reward), bool(terminated), bool(truncated), info
