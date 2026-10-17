"""Step 0a: fetch BigCodeBench and write the measurement tasks and the RL training pool.

measure.jsonl holds Hua et al.'s 256 tasks. train.jsonl holds the other tasks whose libraries are
all in the standard library, minus libraries that reach the network or spawn processes, and whose
reference solution passes all its tests in the sandbox, so every RL task can earn full reward.

    uv run python experiments/pilot/00_tasks.py --config configs/pilot/00_tasks.toml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

from evalgaming.runs import ConfigOption, JsonLogsOption, start_run
from evalgaming.sandbox import SandboxConfig, run_tests
from evalgaming.tasks import fetch_bigcodebench, is_stdlib_only, save_tasks


@dataclass(frozen=True)
class TasksConfig:
    """Which BigCodeBench release to use and how to split it."""

    output_dir: Path
    dataset: str
    revision: str
    split: str
    measure_rows: tuple[int, ...]
    excluded_libs: tuple[str, ...]
    sandbox: SandboxConfig


def main(config: ConfigOption, json_logs: JsonLogsOption = False) -> None:
    """Write measure.jsonl and train.jsonl to the run folder."""
    cfg, run_dir = start_run(config, TasksConfig, json_logs=json_logs)
    tasks = fetch_bigcodebench(cfg.dataset, cfg.revision, cfg.split)
    measured = set(cfg.measure_rows)
    candidates = [
        task
        for row, task in enumerate(tasks)
        if row not in measured and is_stdlib_only(task, cfg.excluded_libs)
    ]
    splits = {
        "measure": [tasks[row] for row in cfg.measure_rows],
        "train": [
            task
            for task in track(candidates, description="Checking reference solutions")
            if run_tests(task.solution, task.test, cfg.sandbox).fraction == 1.0
        ],
    }
    table = Table(title="BigCodeBench tasks")
    table.add_column("file")
    table.add_column("tasks", justify="right")
    for name, split in splits.items():
        save_tasks(split, run_dir / f"{name}.jsonl")
        table.add_row(str(run_dir / f"{name}.jsonl"), str(len(split)))
    Console().print(table)


if __name__ == "__main__":
    typer.run(main)
