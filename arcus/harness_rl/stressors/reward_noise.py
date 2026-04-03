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


class RewardNoiseStressor(BaseStressor):
    name: str = "reward_noise"

    def __init__(
        self,
        seed: int = 0,
        reward_std_fraction: float = 0.50,
        sparse_noise_std: float = 0.5,
        sparse_threshold: float = 0.1,
        clip_reward: bool = False,
    ):
        self.seed                 = int(seed)
        self.reward_std_fraction  = float(reward_std_fraction)
        self.sparse_noise_std     = float(sparse_noise_std)
        self.sparse_threshold     = float(sparse_threshold)
        self.clip_reward          = bool(clip_reward)

        self._rng:        np.random.Generator = np.random.default_rng(self.seed)
        self._noise_std:  Optional[float]     = None
        self._reward_min: Optional[float]     = None
        self._reward_max: Optional[float]     = None

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
            info["stress_applied"] = 0
            return obs, float(reward), bool(terminated), bool(truncated), info

        if self._noise_std is None:
            self._noise_std = self.sparse_noise_std

        noise    = float(self._rng.normal(0.0, float(self._noise_std)))
        r_noisy  = float(reward) + noise

        if self.clip_reward and self._reward_min is not None and self._reward_max is not None:
            r_noisy = float(np.clip(r_noisy, self._reward_min, self._reward_max))

        distortion = abs(r_noisy - float(reward))
        info["regret"]         = float(distortion)
        info["violation"]      = 0.0
        info["stress_applied"] = 1
        info["reward_noise_applied"] = float(noise)

        return obs, r_noisy, bool(terminated), bool(truncated), info

    def calibrate(
        self,
        reward_std: Optional[float] = None,
        reward_min: Optional[float] = None,
        reward_max: Optional[float] = None,
    ) -> None:
        self._reward_min = reward_min
        self._reward_max = reward_max

        if reward_std is not None and float(reward_std) >= self.sparse_threshold:
            self._noise_std = self.reward_std_fraction * float(reward_std)
        else:
            self._noise_std = self.sparse_noise_std

    def reset_rng(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    @property
    def effective_noise_std(self) -> Optional[float]:
        return self._noise_std

    def __repr__(self) -> str:
        return (
            f"RewardNoiseStressor("
            f"seed={self.seed}, "
            f"noise_std={self._noise_std}, "
            f"reward_std_fraction={self.reward_std_fraction})"
        )
