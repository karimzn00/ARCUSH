# arcus/harness_rl/stressors/valence_inversion.py
"""
Valence Inversion Stressor.

Corrupts the reward signal by flipping its sign during the SHOCK phase:
    r_exec = -r

Regret is measured as 2 * |r| (the full swing from +r to -r).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple
from gymnasium import spaces

from .base import BaseStressor


class ValenceInversionStressor(BaseStressor):
    name = "valence_inversion"

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

        if not active:
            info["stress_applied"] = 0
            return obs, float(reward), bool(terminated), bool(truncated), info

        new_r              = -float(reward)
        info["regret"]     = float(2.0 * abs(float(reward)))
        info["stress_applied"] = 1
        return obs, float(new_r), bool(terminated), bool(truncated), info
