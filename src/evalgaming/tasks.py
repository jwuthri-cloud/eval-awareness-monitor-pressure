"""BigCodeBench coding tasks: the measurement subset and a cheap-to-test training pool.

Measurement uses the 256 tasks Hua et al. evaluate on (``scripts/get_bcb_data.py`` in
https://github.com/tim-hua-01/steering-eval-awareness-public), so behavior gaps compare with
theirs and with Montoya et al.'s. RL uses the other tasks whose libraries are all in the Python
standard library and whose reference solution passes its tests in our sandbox: the same prompt
style and ``unittest`` harness, tests that run in a bare subprocess, and a reward the policy can
reach.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from evalgaming.datasets import read_jsonl
from evalgaming.log import get_logger
from evalgaming.runs import write_jsonl

log = get_logger(__name__)


@dataclass(frozen=True)
class Task:
    """One coding task: the instruction prompt, its ``unittest`` source and a reference solution."""

    id: str
    prompt: str
    test: str
    solution: str
    libs: tuple[str, ...]

    @property
    def snippet(self) -> str:
        """The task's first line, for display."""
        return self.prompt.strip().split("\n", 1)[0]


def load_tasks(path: Path) -> list[Task]:
    """Read tasks from JSONL."""
    return [Task(**row | {"libs": tuple(row["libs"])}) for row in read_jsonl(path)]


def save_tasks(tasks: Iterable[Task], path: Path) -> None:
    """Write tasks to JSONL."""
    write_jsonl((asdict(task) for task in tasks), path)


def is_stdlib_only(task: Task, excluded: Iterable[str]) -> bool:
    """Whether every library the task uses is in the standard library and not excluded."""
    allowed = set(sys.stdlib_module_names) - set(excluded)
    return all(lib.split(".")[0] in allowed for lib in task.libs)


def fetch_bigcodebench(dataset: str, revision: str, split: str) -> list[Task]:
    """All BigCodeBench rows of ``split`` at a pinned revision, in dataset order."""
    from datasets import load_dataset  # gpu extra

    rows = load_dataset(dataset, split=split, revision=revision)
    log.info("bigcodebench_loaded", rows=len(rows), split=split, revision=revision)
    return [
        Task(
            id=row["task_id"],
            prompt=row["instruct_prompt"],
            test=row["test"],
            solution=row["code_prompt"] + row["canonical_solution"],
            libs=tuple(ast.literal_eval(row["libs"])),  # stored as the repr of a list
        )
        for row in rows
    ]
