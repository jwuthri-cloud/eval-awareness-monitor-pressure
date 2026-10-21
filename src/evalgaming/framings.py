"""Wrap a coding task in a framing (data/SPEC.md section 1) to get chat messages."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path

from evalgaming.datasets import TASK_SLOT, read_jsonl

# Reporting groups: the SPEC label, with Wood Labs framings split out of "eval".
GROUPS = ("eval", "wood", "deploy", "neutral")


@dataclass(frozen=True)
class Framing:
    """One framing: an optional system prompt and a user template with one ``{task}`` slot."""

    id: str
    context: str
    label: str
    style: str
    system: str | None
    user_template: str

    @property
    def group(self) -> str:
        """Reporting group: ``wood`` for Wood Labs framings, otherwise the SPEC label."""
        return "wood" if self.context == "wood" else self.label

    def messages(self, task: str) -> list[dict[str, str]]:
        """Chat messages for ``task``; it fills the slot by ``str.replace``, so braces are safe."""
        user = {"role": "user", "content": self.user_template.replace(TASK_SLOT, task)}
        return [{"role": "system", "content": self.system}, user] if self.system else [user]


def load_framings(path: Path) -> list[Framing]:
    """Read a framings JSONL file, keeping the fields that shape the prompt."""
    names = [field.name for field in fields(Framing)]
    return [Framing(**{name: row[name] for name in names}) for row in read_jsonl(path)]


def by_group(framings: list[Framing]) -> dict[str, list[Framing]]:
    """Framings keyed by reporting group, in file order."""
    groups: dict[str, list[Framing]] = defaultdict(list)
    for framing in framings:
        groups[framing.group].append(framing)
    return dict(groups)


def assign_framings(
    tasks: list, groups: dict[str, list[Framing]], rng, matched: bool = False
) -> list[tuple]:
    """(task, framing) for every task and every reporting group present, in task order.

    Unmatched: each group's framing is drawn independently. Matched: one index per task picks the
    same position in every group, so the eval and deploy prompts of a task are one cue pair that
    differs only in the cue (groups must list their pairs in the same order and be the same size).
    """
    present = [group for group in GROUPS if group in groups]
    if not matched:
        return [(task, rng.choice(groups[group])) for task in tasks for group in present]
    sizes = {len(groups[group]) for group in present}
    if len(sizes) != 1:
        raise ValueError(f"matched framings need equal group sizes, got {sizes}")
    size = sizes.pop()
    return [
        (task, groups[group][index])
        for task in tasks
        for index in [rng.randrange(size)]
        for group in present
    ]
