from __future__ import annotations

import numpy as np


def build_schedule_mask(*, episodes: int, schedule_tag: str, schedule_spec: str) -> np.ndarray:
    mask = np.zeros((episodes,), dtype=bool)

    if schedule_spec.strip().lower() == "none":
        return mask

    parts = [p.strip() for p in schedule_spec.split(",") if p.strip()]
    idx = 0
    for part in parts:
        name, dur = part.split(":")
        name = name.strip()
        dur = int(dur.strip())
        if dur <= 0:
            continue
        j = min(episodes, idx + dur)
        if name != "none":
            mask[idx:j] = True
        idx = j
        if idx >= episodes:
            break
    return mask
