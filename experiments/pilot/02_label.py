"""Step 2: hand-label reasoning traces from a baseline run against data/monitor/rubric.md.

Draws the same number of traces from each framing group, shuffles them and shows them blind:
only the task's first line and the trace, never the framing. Each label is checked with
``check_labelled_traces`` (data/SPEC.md section 2) and appended to labels.jsonl at once, so the
tool can be stopped with ``q`` and resumed. Copy the finished file to data/monitor/labels.jsonl.

    uv run python experiments/pilot/02_label.py --config configs/pilot/02_label.toml --labeller JW
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from evalgaming.datasets import LEVELS_BY_CATEGORY, check_labelled_traces, read_jsonl
from evalgaming.framings import GROUPS
from evalgaming.runs import ConfigOption, JsonLogsOption, append_jsonl, start_run

QUIT, SKIP = "q", "s"


@dataclass(frozen=True)
class LabelConfig:
    """Which baseline run to draw traces from, how many per framing group, and the target."""

    output_dir: Path
    samples: Path
    per_group: int
    seed: int
    min_verbalized: int


def draw(samples: list[dict[str, Any]], per_group: int, seed: int) -> list[dict[str, Any]]:
    """``per_group`` random samples from each framing group, shuffled together."""
    rng = random.Random(seed)
    drawn = [
        row
        for group in GROUPS
        for row in rng.sample([s for s in samples if s["group"] == group], per_group)
    ]
    rng.shuffle(drawn)
    return drawn


def ask_evidence(console: Console, trace: str) -> str:
    """Ask until the answer is a non-empty exact substring of the trace."""
    while True:
        evidence = Prompt.ask("Evidence (paste the shortest exact span)", console=console)
        if evidence and evidence in trace:
            return evidence
        console.print("[red]Not an exact substring of the trace; copy it again.[/red]")


def ask_row(console: Console, base: dict[str, Any]) -> dict[str, Any] | str:
    """A valid SPEC row for this trace, or QUIT or SKIP."""
    while True:
        choice = Prompt.ask("Level", choices=["0", "1", "2", SKIP, QUIT], console=console)
        if choice in (SKIP, QUIT):
            return choice
        level = int(choice)
        categories = [name for name, levels in LEVELS_BY_CATEGORY.items() if level in levels]
        category = Prompt.ask("Category", choices=categories, console=console)
        evidence = ask_evidence(console, base["trace"]) if level else ""
        row = base | {"level": level, "category": category, "evidence": evidence}
        if not (issues := check_labelled_traces([row])):
            return row
        console.print(f"[red]Invalid label ({'; '.join(i.message for i in issues)}), again.[/red]")


def main(
    config: ConfigOption,
    labeller: Annotated[str, typer.Option(help="Your initials, stored with every label.")],
    json_logs: JsonLogsOption = False,
) -> None:
    """Label traces until all are done or you quit."""
    cfg, run_dir = start_run(config, LabelConfig, json_logs=json_logs)
    labels_path = run_dir / "labels.jsonl"
    labels = read_jsonl(labels_path) if labels_path.exists() else []
    done = {row["sample_id"] for row in labels}
    drawn = draw(read_jsonl(cfg.samples), cfg.per_group, cfg.seed)
    queue = [sample for sample in drawn if sample["sample_id"] not in done]
    levels = Counter(row["level"] for row in labels)
    console = Console()

    for position, sample in enumerate(queue, start=len(drawn) - len(queue) + 1):
        console.rule(
            f"Trace {position} of {len(drawn)}   levels 0/1/2: "
            f"{levels[0]}/{levels[1]}/{levels[2]}   "
            f"verbalized {levels[1] + levels[2]} of {cfg.min_verbalized} wanted"
        )
        console.print(Panel(sample["task_snippet"], title="Task", expand=False))
        # Plain soft-wrapped text, so a copied span has no borders or inserted line breaks.
        console.print(sample["reasoning"], markup=False, highlight=False, soft_wrap=True)
        console.rule()
        row = ask_row(
            console,
            {  # SPEC field order; ask_row fills in level, category and evidence.
                "id": f"lab-{len(labels) + 1:04d}",
                "source_run": str(cfg.samples.parent),
                "sample_id": sample["sample_id"],
                "task_snippet": sample["task_snippet"],
                "trace": sample["reasoning"],
                "level": None,
                "category": None,
                "evidence": "",
                "labeller": labeller,
            },
        )
        if row == QUIT:
            break
        if row == SKIP:
            continue
        append_jsonl(row, labels_path)
        labels.append(row)
        levels[row["level"]] += 1
    console.print(f"{len(labels)} labels in [bold]{labels_path}[/bold]")


if __name__ == "__main__":
    typer.run(main)
