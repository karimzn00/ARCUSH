# arcus/harness_rl/stressors/resource_constraint.py
"""
Resource Constraint Stressor.

Models reduced control authority: the agent's intended action is scaled
down by a factor κ ∈ (0, 1], reducing its effective authority over the
environment.

Paper formulation (Section 4.2):
    Continuous:  a_exec = κ · a_t
    Discrete:    with prob p, action is replaced by a fixed fallback (no-op)

This is an ACTION-channel stressor (not reward scaling).
Regret is measured as the L2 distance between intended and executed action
for continuous spaces, or 1.0 per replaced discrete action.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from gymnasium import spaces

from .base import BaseStressor


@dataclass
class ResourceConstraintConfig:
    # Continuous: scale factor κ ∈ (0, 1]
    action_scale: float = 0.40
    # Discrete: probability of replacing action with fallback
    replace_prob: float = 0.40
    # Discrete: fallback action index (0 = typically no-op)
    fallback_action: int = 0


class ResourceConstraintStressor(BaseStressor):
    """
    During SHOCK phase:
      Continuous: a_exec = action_scale * a_t
      Discrete:   with prob replace_prob, a_exec = fallback_action

    Exposes:
      info["stress_applied"]  : 1 if action was modified, else 0
      info["regret"]          : magnitude of the action change
      info["violation"]       : 0.0 (no constraint violated, just reduced authority)
    """
    name = "resource_constraint"

    def __init__(self, cfg: ResourceConstraintConfig | None = None):
        self.cfg = cfg or ResourceConstraintConfig()

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

        if not active:
            return action, info

        if isinstance(action_space, spaces.Box):
            a        = np.asarray(action, dtype=np.float32)
            kappa    = float(np.clip(self.cfg.action_scale, 0.0, 1.0))
            a_exec   = kappa * a
            regret   = float(np.linalg.norm(a_exec - a))
            info["regret"]        = regret
            info["stress_applied"] = 1
            return a_exec, info

        if isinstance(action_space, spaces.Discrete):
            if np.random.random() < float(np.clip(self.cfg.replace_prob, 0.0, 1.0)):
                fb = int(np.clip(self.cfg.fallback_action, 0, int(action_space.n) - 1))
                info["regret"]        = 1.0
                info["stress_applied"] = 1
                return fb, info
            return action, info

        # Unsupported space: pass through
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
        # Resource constraint only touches the action channel.
        info.setdefault("violation",     0.0)
        info.setdefault("regret",        0.0)
        info.setdefault("stress_applied", 0)
        return obs, float(reward), bool(terminated), bool(truncated), info
