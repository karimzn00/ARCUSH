# arcus/harness_rl/stressors/trust_violation.py
"""
Trust Violation Stressor.

Models a mismatch between the agent's intended action and what is
actually executed in the environment.

  intensity ~ Beta(a, b) * intensity_scale

With probability apply_prob:
  Discrete:   with prob clip(action_perturb * intensity, 0..1),
              remap to a uniformly sampled different action
  Continuous: add N(0, action_perturb * intensity) noise, clip to bounds
              "applied" is True only if perturbation > min_effect

The seed is forwarded from the registry so results are reproducible
per (algo, env, seed) combination — NOT fixed at 0 globally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from gymnasium import spaces

from .base import BaseStressor


@dataclass
class TrustViolationConfig:
    mean_intensity:  float = 0.55
    concentration:   float = 6.0
    apply_prob:      float = 1.0
    action_perturb:  float = 0.8
    intensity_scale: float = 0.70
    min_effect:      float = 1e-3
    cont_clip:       float = 1.0


def _beta_params(mean: float, concentration: float) -> Tuple[float, float]:
    mean          = float(np.clip(mean, 1e-4, 1.0 - 1e-4))
    concentration = float(max(concentration, 1e-3))
    return mean * concentration, (1.0 - mean) * concentration


class TrustViolationStressor(BaseStressor):
    name = "trust_violation"

    def __init__(self, cfg: TrustViolationConfig | None = None, seed: int = 0):
        self.cfg    = cfg or TrustViolationConfig()
        self.rng    = np.random.default_rng(int(seed))
        self.a, self.b = _beta_params(self.cfg.mean_intensity, self.cfg.concentration)

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

        if not active:
            info["trust_violation_intensity"] = 0.0
            info["trust_violation_applied"]   = False
            info["stress_applied"]            = 0
            return action, info

        intensity = float(self.rng.beta(self.a, self.b))
        intensity = float(np.clip(intensity * float(np.clip(self.cfg.intensity_scale, 0.0, 10.0)), 0.0, 1.0))

        applied    = False
        new_action = action

        if self.rng.random() < float(np.clip(self.cfg.apply_prob, 0.0, 1.0)):
            new_action, applied = self._perturb_action(action, action_space, intensity)

        info["trust_violation_intensity"] = float(intensity)
        info["trust_violation_applied"]   = bool(applied)
        info["violation"]                 = float(intensity if applied else 0.0)
        info["regret"]                    = float(intensity if applied else 0.0)
        info["stress_applied"]            = 1 if applied else 0
        return new_action, info

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
        info.setdefault("violation",     0.0)
        info.setdefault("regret",        0.0)
        info.setdefault("stress_applied", 0)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _perturb_action(
        self,
        action: Any,
        action_space: spaces.Space,
        intensity: float,
    ) -> Tuple[Any, bool]:
        strength = float(np.clip(self.cfg.action_perturb, 0.0, 10.0)) * float(intensity)

        if isinstance(action_space, spaces.Discrete):
            n = int(action_space.n)
            if n <= 1:
                return action, False
            if self.rng.random() < float(np.clip(strength, 0.0, 1.0)):
                a       = int(action)
                choices = [i for i in range(n) if i != a]
                return int(self.rng.choice(choices)), True
            return action, False

        if isinstance(action_space, spaces.Box):
            a = np.asarray(action, dtype=np.float32)
            if strength <= 0.0:
                return action, False
            noise = self.rng.normal(0.0, strength, size=a.shape).astype(np.float32)
            a2    = a + noise
            low, high = np.asarray(action_space.low), np.asarray(action_space.high)
            if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
                a2 = np.clip(a2, low, high)
            else:
                a2 = np.clip(a2, -self.cfg.cont_clip, self.cfg.cont_clip)
            applied = float(np.max(np.abs(a2 - a))) >= float(max(self.cfg.min_effect, 0.0))
            return a2, applied

        return action, False
