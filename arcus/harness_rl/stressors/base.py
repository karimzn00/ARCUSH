# arcus/harness_rl/stressors/base.py
"""
BaseStressor and StressPatternWrapper.

StressPatternWrapper applies a 3-segment stress schedule:
    baseline:PRE , {mode}:SHOCK , baseline:POST

Segment lengths are in episodes. The wrapper tracks episode index,
derives the current phase, and delegates to the stressor's
transform_action / transform_step hooks.

Stable info keys added to every step and reset:
    stress_mode      : str   stressor name
    stress_segment   : str   current segment name ("baseline" or stressor name)
    stress_phase     : str   "pre" | "shock" | "post"
    stress_active    : bool  True only during shock segment for a non-baseline stressor
    violation        : float (default 0.0, overwritten by stressor if applicable)
    regret           : float (default 0.0, overwritten by stressor if applicable)
    stress_applied   : int   (default 0, 1 if stressor actually modified action/reward)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Pattern parsing helpers
# ---------------------------------------------------------------------------

def _parse_int_loose(s: str) -> int:
    m = re.search(r"-?\d+", str(s))
    if not m:
        raise ValueError(f"Could not parse int from '{s}'")
    return int(m.group(0))


def _normalize_pattern(pattern: str) -> str:
    p = (pattern or "").strip()
    p = p.replace("{mode:", "{mode}:")
    p = p.rstrip("}").strip()
    p = re.sub(r"\s+", "", p)
    return p


def parse_pattern_segments(pattern: str, mode: str) -> List[Tuple[str, int]]:
    """
    Parse a 3-segment stress pattern string.

    Expected format:  "baseline:PRE,{mode}:SHOCK,baseline:POST"
    For baseline-only: "baseline:PRE,baseline:SHOCK,baseline:POST"

    Returns list of (segment_name, n_episodes) tuples.
    """
    pattern = _normalize_pattern(pattern)
    parts = [p.strip() for p in pattern.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(
            f"Pattern must have exactly 3 segments, e.g. "
            f"'baseline:20,{{mode}}:40,baseline:60'. Got: {pattern!r}"
        )
    out: List[Tuple[str, int]] = []
    for seg in parts:
        if ":" not in seg:
            raise ValueError(f"Bad pattern segment {seg!r}. Expected 'name:N'.")
        name, n = seg.split(":", 1)
        name = name.strip()
        if name == "{mode}":
            name = mode
        n_i = _parse_int_loose(n.strip())
        if n_i <= 0:
            raise ValueError(f"Segment length must be > 0. Got {n_i} in {seg!r}.")
        out.append((name, n_i))
    return out


# ---------------------------------------------------------------------------
# Base stressor
# ---------------------------------------------------------------------------

class BaseStressor:
    """
    No-op stressor base class.

    Subclasses override transform_action and/or transform_step.
    Both hooks must ensure info always contains the three stable keys:
        violation, regret, stress_applied
    """
    name: str = "base"

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
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        info.setdefault("violation",     0.0)
        info.setdefault("regret",        0.0)
        info.setdefault("stress_applied", 0)
        return obs, float(reward), bool(terminated), bool(truncated), info


# ---------------------------------------------------------------------------
# Phase info
# ---------------------------------------------------------------------------

@dataclass
class _PhaseInfo:
    phase:        str   # "pre" | "shock" | "post"
    active:       bool  # True only during shock for a non-baseline stressor
    segment_name: str   # "baseline" or stressor name


# ---------------------------------------------------------------------------
# Action coercion
# ---------------------------------------------------------------------------

def _coerce_action_for_space(action: Any, action_space: spaces.Space) -> Any:
    """
    Many SB3 policies return ndarray([a]) for Discrete spaces.
    Some envs (FrozenLake) require a plain int (hashable).
    This coercion is applied *after* the stressor so the env always
    receives the correct type.
    """
    try:
        if isinstance(action_space, spaces.Discrete):
            if isinstance(action, (np.ndarray, list, tuple)):
                return int(np.asarray(action).reshape(-1)[0])
            return int(action)
        if isinstance(action_space, spaces.MultiDiscrete):
            return np.asarray(action, dtype=np.int64).reshape(-1)
        if isinstance(action_space, spaces.MultiBinary):
            return np.asarray(action, dtype=np.int8).reshape(-1)
    except Exception:
        return action
    return action


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class StressPatternWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that applies a stressor according to a 3-segment pattern.

    Episode counter is incremented on each reset() call, so the wrapper is
    stateful and must not be shared across parallel workers.
    """

    def __init__(self, env: gym.Env, stressor: BaseStressor, *, mode: str, pattern: str):
        super().__init__(env)
        self.stressor = stressor
        self.mode     = (mode or "").strip().lower()
        self.pattern  = _normalize_pattern(pattern)

        self._episode_idx = -1
        self._segments    = parse_pattern_segments(self.pattern, self.mode)
        self._pre_n       = int(self._segments[0][1])
        self._shock_n     = int(self._segments[1][1])

    def _phase_for_episode(self, ep_idx: int) -> _PhaseInfo:
        if ep_idx < self._pre_n:
            seg = self._segments[0][0]
            return _PhaseInfo("pre", False, seg)
        if ep_idx < self._pre_n + self._shock_n:
            seg    = self._segments[1][0]
            active = (seg == self.mode) and (self.mode not in ("baseline", "none"))
            return _PhaseInfo("shock", active, seg)
        seg = self._segments[2][0]
        return _PhaseInfo("post", False, seg)

    def _stress_info(self, ph: _PhaseInfo) -> Dict[str, Any]:
        return {
            "stress_mode":    self.mode,
            "stress_segment": ph.segment_name,
            "stress_phase":   ph.phase,
            "stress_active":  bool(ph.active),
            "violation":      0.0,
            "regret":         0.0,
            "stress_applied": 0,
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._episode_idx += 1
        ph    = self._phase_for_episode(self._episode_idx)
        info  = {**self._stress_info(ph), **dict(info or {})}
        # ensure stress keys are not overwritten by env info
        info.update(self._stress_info(ph))
        return obs, info

    def step(self, action):
        ph   = self._phase_for_episode(self._episode_idx)
        base = self._stress_info(ph)

        action2, base = self.stressor.transform_action(
            action=action,
            action_space=self.action_space,
            active=bool(ph.active),
            phase=ph.phase,
            info=base,
        )
        action2 = _coerce_action_for_space(action2, self.action_space)

        obs, reward, terminated, truncated, env_info = self.env.step(action2)
        merged = {**dict(env_info or {}), **base}

        obs, reward, terminated, truncated, merged = self.stressor.transform_step(
            action=action2,
            obs=obs,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=merged,
            action_space=self.action_space,
            active=bool(ph.active),
            phase=ph.phase,
        )
        merged.setdefault("violation",     0.0)
        merged.setdefault("regret",        0.0)
        merged.setdefault("stress_applied", 0)
        return obs, float(reward), bool(terminated), bool(truncated), merged


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def apply_stress_pattern(env: gym.Env, *, mode: str, pattern: str) -> StressPatternWrapper:
    from . import get_stressor
    m        = (mode or "").strip().lower()
    stressor = get_stressor(m)
    return StressPatternWrapper(env, stressor, mode=m, pattern=pattern)
