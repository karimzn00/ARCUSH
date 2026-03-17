# arcus/harness_rl/stressors/__init__.py
"""
Stressor registry.

All stressors that accept a seed parameter receive one.
Stressors without a seed parameter are constructed without it.
concept_drift is now registered and accessible.
"""
from __future__ import annotations

from typing import Any, Dict, Type

from .base import apply_stress_pattern, BaseStressor
from .baseline import BaselineStressor
from .resource_constraint import ResourceConstraintStressor
from .trust_violation import TrustViolationStressor, TrustViolationConfig
from .valence_inversion import ValenceInversionStressor
from .concept_drift import ConceptDriftStressor, ConceptDriftConfig


# ---------------------------------------------------------------------------
# Registry: name -> (class, accepts_seed)
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, tuple[Type[BaseStressor], bool]] = {
    "none":                (BaselineStressor,          False),
    "baseline":            (BaselineStressor,          False),
    "resource_constraint": (ResourceConstraintStressor, False),
    "trust_violation":     (TrustViolationStressor,     True),
    "valence_inversion":   (ValenceInversionStressor,   False),
    "concept_drift":       (ConceptDriftStressor,       True),
}


def available_stressors() -> list[str]:
    """Return sorted list of registered stressor names."""
    return sorted(_REGISTRY.keys())


def get_stressor(name: str, *, seed: int = 0, **kwargs: Any) -> BaseStressor:
    """
    Instantiate a stressor by name.

    Parameters
    ----------
    name   : stressor key (see available_stressors())
    seed   : passed to stressors that accept it (trust_violation, concept_drift)
    kwargs : forwarded to the stressor constructor if it accepts them
    """
    key = (name or "baseline").strip().lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown stressor '{name}'. Available: {available_stressors()}"
        )
    cls, accepts_seed = _REGISTRY[key]
    if accepts_seed:
        return cls(seed=int(seed), **kwargs)    # type: ignore[call-arg]
    return cls(**kwargs)                        # type: ignore[call-arg]
