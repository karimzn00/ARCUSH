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


class ObservationNoiseStressor(BaseStressor):
    name: str = "observation_noise"

    def __init__(
        self,
        seed: int = 0,
        obs_std_fraction: float = 0.15,
        pixel_noise_std: float = 10.0,
        channel_dropout_p: float = 0.0,
        clip_obs: bool = True,
    ):
        self.seed              = int(seed)
        self.obs_std_fraction  = float(obs_std_fraction)
        self.pixel_noise_std   = float(pixel_noise_std)
        self.channel_dropout_p = float(channel_dropout_p)
        self.clip_obs          = bool(clip_obs)

        self._rng:        np.random.Generator = np.random.default_rng(self.seed)
        self._noise_std:  Optional[float]     = None
        self._is_image:   bool                = False
        self._obs_low:    Optional[np.ndarray] = None
        self._obs_high:   Optional[np.ndarray] = None

    def transform_action(
        self,
        action: Any,
        *,
        action_space: spaces.Space,
        active: bool,
        phase: str,
        info: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        info.setdefault("violation",     0.0)
        info.setdefault("regret",        0.0)
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
        info.setdefault("violation",     0.0)
        info.setdefault("regret",        0.0)

        if active:
            obs = self.perturb_obs(obs)
            info["stress_applied"] = 1
        else:
            info.setdefault("stress_applied", 0)

        return obs, float(reward), bool(terminated), bool(truncated), info

    def calibrate(
        self,
        obs_std: Optional[float] = None,
        observation_space: Optional[Any] = None,
    ) -> None:
        if observation_space is not None:
            obs = observation_space
            is_image = (
                hasattr(obs, "dtype")
                and obs.dtype == np.uint8
                and hasattr(obs, "shape")
                and len(obs.shape) >= 3
            )
            self._is_image = is_image

            if is_image:
                self._noise_std = self.pixel_noise_std
            else:
                if obs_std is not None and obs_std > 0:
                    self._noise_std = self.obs_std_fraction * float(obs_std)
                elif hasattr(obs, "low") and hasattr(obs, "high"):
                    rng_std = float(np.nanmean(
                        np.clip(obs.high, -1e6, 1e6) - np.clip(obs.low, -1e6, 1e6)
                    )) / 4.0
                    self._noise_std = self.obs_std_fraction * max(rng_std, 1e-6)
                else:
                    self._noise_std = self.obs_std_fraction * 1.0

            if hasattr(obs, "low") and hasattr(obs, "high"):
                self._obs_low  = obs.low
                self._obs_high = obs.high
        else:
            self._is_image = False
            if obs_std is not None and obs_std > 0:
                self._noise_std = self.obs_std_fraction * float(obs_std)
            else:
                self._noise_std = self.obs_std_fraction * 1.0

    def reset_rng(self) -> None:
        """Reset RNG to seed — call at start of each shock evaluation."""
        self._rng = np.random.default_rng(self.seed)

    def perturb_obs(self, obs: Any) -> Any:
        """
        Apply Gaussian noise to a single observation.
        Preserves original dtype and shape.
        """
        if self._noise_std is None:
            self._noise_std = self.obs_std_fraction * 1.0

        obs_arr = np.array(obs, dtype=float)
        noise   = self._rng.normal(0.0, float(self._noise_std), size=obs_arr.shape)
        noisy   = obs_arr + noise
        if self.channel_dropout_p > 0.0 and obs_arr.ndim >= 1:
            n_ch = obs_arr.shape[0]
            for ch in range(n_ch):
                if self._rng.random() < self.channel_dropout_p:
                    if obs_arr.ndim == 1:
                        noisy[ch] = 0.0
                    else:
                        noisy[ch] = 0.0

        if self.clip_obs:
            if self._obs_low is not None and self._obs_high is not None:
                noisy = np.clip(noisy, self._obs_low, self._obs_high)
            elif self._is_image:
                noisy = np.clip(noisy, 0.0, 255.0)

        orig_dtype = getattr(obs, "dtype", None) or (
            np.array(obs).dtype if not isinstance(obs, np.ndarray) else obs.dtype
        )
        return noisy.astype(orig_dtype)

    @property
    def effective_noise_std(self) -> Optional[float]:
        return self._noise_std

    @property
    def is_image(self) -> bool:
        return self._is_image

    def __repr__(self) -> str:
        return (
            f"ObservationNoiseStressor("
            f"seed={self.seed}, "
            f"noise_std={self._noise_std}, "
            f"is_image={self._is_image}, "
            f"channel_dropout_p={self.channel_dropout_p})"
        )
