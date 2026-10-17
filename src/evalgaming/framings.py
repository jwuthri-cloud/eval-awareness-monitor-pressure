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
