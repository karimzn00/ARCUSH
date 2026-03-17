# arcus/core/logio.py
"""
Single source of truth for EpisodeLog.
Do NOT redefine EpisodeLog in identity.py or anywhere else.
Import from here everywhere.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable


@dataclass
class EpisodeLog:
    # --- required: identify the episode unambiguously ---
    episode: int
    phase: str                  # "pre" | "shock" | "post"
    stress: str                 # stressor name e.g. "trust_violation"
    # --- episode outcome ---
    reward: float = 0.0
    steps: int = 0
    # --- identity channels (all in [0, 1]) ---
    competence: float = 0.0
    coherence: float = 0.0
    continuity: float = 0.0
    integrity: float = 0.0
    meaning: float = 0.0
    # --- composite ---
    identity: float = 0.0
    collapse_score: float = 0.0
    collapse: bool = False
    # --- stress metadata ---
    stress_applied: int = 0
    violation: float = 0.0
    regret: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any] | EpisodeLog]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            if isinstance(r, EpisodeLog):
                r = asdict(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: Dict[str, Any] | EpisodeLog) -> None:
    """Append a single row to a JSONL file (useful for streaming writes)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(row, EpisodeLog):
        row = asdict(row)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out
