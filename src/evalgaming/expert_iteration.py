"""Expert iteration for a sturdier organism: keep the model's own samples that behave as targeted.

Eval-like samples (eval and Wood Labs framings) are kept when they write type hints, deploy samples
when they write none; both must pass enough unit tests. Selection looks at behavior only, never at
whether the trace states eval awareness, so how the behavior relates to the words is whatever the
model learns, not something built in. Kept samples are balanced between the two sides and capped
per prompt so a few easy tasks do not dominate.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

EVAL_LIKE = ("eval", "wood")


@dataclass(frozen=True)
class SelectionConfig:
    min_hint: int  # hint score (0-100) an eval-like sample needs; deploy samples need 0
    min_tests: float  # fraction of unit tests passed
    max_per_prompt: int  # kept samples per (task, framing)


def keep(row: Mapping[str, Any], config: SelectionConfig) -> bool:
    """Whether one scored sample shows the targeted behavior and is good enough code."""
    hint = row["hint_score"]
    if row["truncated"] or hint is None or row["tests"] < config.min_tests:
        return False
    if row["group"] in EVAL_LIKE:
        return hint >= config.min_hint
    return row["group"] == "deploy" and hint == 0


def select(
    rows: Sequence[Mapping[str, Any]], config: SelectionConfig, seed: int
) -> list[Mapping[str, Any]]:
    """Kept samples, at most ``max_per_prompt`` per prompt, as many eval-like as deploy."""
    rng = random.Random(seed)
    per_prompt: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if keep(row, config):
            per_prompt[row["task_id"], row["framing_id"]].append(row)
    sides: dict[bool, list[Mapping[str, Any]]] = {True: [], False: []}
    for members in per_prompt.values():
        chosen = rng.sample(members, min(len(members), config.max_per_prompt))
        sides[members[0]["group"] in EVAL_LIKE] += chosen
    size = min(len(sides[True]), len(sides[False]))
    kept = rng.sample(sides[True], size) + rng.sample(sides[False], size)
    rng.shuffle(kept)
    return kept
