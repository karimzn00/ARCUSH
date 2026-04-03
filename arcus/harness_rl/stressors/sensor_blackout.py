from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from arcus.harness_rl.stressors.base import BaseStressor


class SensorBlackoutStressor(BaseStressor):
    name: str = "sensor_blackout"

    def __init__(
        self,
        seed: int = 0,
        blackout_prob: float = 0.10,
        min_blackout_steps: int = 3,
        max_blackout_steps: int = 15,
        mode: str = "zero",
        calibrate_from_horizon: bool = True,
        blackout_fraction: float = 0.15,
    ):
        self.seed                   = int(seed)
        self.blackout_prob          = float(blackout_prob)
        self.min_blackout_steps     = int(min_blackout_steps)
        self.max_blackout_steps     = int(max_blackout_steps)
        self.mode                   = str(mode).lower()
        self.calibrate_from_horizon = bool(calibrate_from_horizon)
        self.blackout_fraction      = float(blackout_fraction)

        self._rng:              np.random.Generator  = np.random.default_rng(self.seed)
        self._in_blackout:      bool                 = False
        self._blackout_steps_remaining: int          = 0
        self._last_valid_obs:   Optional[np.ndarray] = None
        self._zero_obs:         Optional[np.ndarray] = None
        self._calibrated_prob:  float                = float(blackout_prob)

    def transform_action(
        self,
        action: Any,
        *,
        action_space: spaces.Space,
        active: bool,
        phase: str,
        info: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        info.setdefault("violation",      0.0)
        info.setdefault("regret",         0.0)
        info.setdefault("stress_applied", 0)
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
    ):
        info.setdefault("violation",      0.0)
        info.setdefault("regret",         0.0)

        if not active:
            if obs is not None:
                self._last_valid_obs = np.array(obs, dtype=float)
            info["stress_applied"]       = 0
            info["sensor_blackout_active"] = False
            return obs, float(reward), bool(terminated), bool(truncated), info
        obs_arr = np.array(obs, dtype=float)
        if self._zero_obs is None:
            self._zero_obs = np.zeros_like(obs_arr)
        if self._last_valid_obs is None:
            self._last_valid_obs = obs_arr.copy()
        if self._in_blackout:
            self._blackout_steps_remaining -= 1
            if self._blackout_steps_remaining <= 0:
                self._in_blackout = False
        else:
            if self._rng.random() < self._calibrated_prob:
                window = int(self._rng.integers(
                    self.min_blackout_steps,
                    self.max_blackout_steps + 1
                ))
                self._in_blackout = True
                self._blackout_steps_remaining = window - 1

        if self._in_blackout:
            if self.mode == "last_obs" and self._last_valid_obs is not None:
                corrupted = self._last_valid_obs.copy()
            else:
                corrupted = self._zero_obs.copy()

            corrupted = corrupted.astype(obs_arr.dtype if hasattr(obs_arr, "dtype")
                                         else float)
            info["stress_applied"]         = 1
            info["sensor_blackout_active"] = True
            info["regret"] = float(np.mean(np.abs(obs_arr)))
            return corrupted, float(reward), bool(terminated), bool(truncated), info
        else:
            self._last_valid_obs = obs_arr.copy()
            info["stress_applied"]         = 0
            info["sensor_blackout_active"] = False
            return obs, float(reward), bool(terminated), bool(truncated), info

    def calibrate(self, horizon: int) -> None:
        if not self.calibrate_from_horizon or horizon <= 0:
            self._calibrated_prob = self.blackout_prob
            return

        mean_window = (self.min_blackout_steps + self.max_blackout_steps) / 2.0
        fraction    = float(np.clip(self.blackout_fraction, 0.01, 0.5))
        p = fraction / (mean_window * (1.0 - fraction))
        self._calibrated_prob = float(np.clip(p, 0.001, 0.5))

    def reset_episode(self) -> None:
        self._in_blackout               = False
        self._blackout_steps_remaining  = 0
        self._last_valid_obs            = None
        self._zero_obs                  = None

    def reset_rng(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.reset_episode()

    @property
    def effective_blackout_prob(self) -> float:
        return self._calibrated_prob

    @property
    def expected_blackout_fraction(self) -> float:
        mean_w = (self.min_blackout_steps + self.max_blackout_steps) / 2.0
        p      = self._calibrated_prob
        return float(p * mean_w / (1.0 + p * mean_w))

    def __repr__(self) -> str:
        return (
            f"SensorBlackoutStressor("
            f"seed={self.seed}, "
            f"blackout_prob={self._calibrated_prob:.4f}, "
            f"window=[{self.min_blackout_steps},{self.max_blackout_steps}], "
            f"mode='{self.mode}', "
            f"expected_fraction={self.expected_blackout_fraction:.2f})"
        )
