"""Step 1: sample reasoning traces under each framing group and measure the behavior gap.

Every task is wrapped once in a random held-out framing from each group (eval, wood, deploy,
neutral). Writes samples.jsonl (trace, answer, hint score, prefilter hit) and summary.json, and
prints the type-hint rate per group, the eval-minus-deploy gap and the prefilter rate.

    uv run python experiments/pilot/01_baseline.py --config configs/pilot/01_baseline.toml
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console

from evalgaming import behavior
from evalgaming.framings import GROUPS, by_group, load_framings
from evalgaming.monitor import prefilter
from evalgaming.report import estimates_table, hint_gap, sample_metrics, save_summary
from evalgaming.runs import ConfigOption, JsonLogsOption, start_run, write_jsonl
from evalgaming.sampling import SamplingConfig, sample
from evalgaming.tasks import load_tasks


@dataclass(frozen=True)
class BaselineConfig:
    """Tasks, framings, sampling settings and the go/no-go threshold on the gap."""

    output_dir: Path
    tasks: Path
    framings: Path
    n_tasks: int
    seed: int
    min_gap: float
    sampling: SamplingConfig


def main(config: ConfigOption, json_logs: JsonLogsOption = False) -> None:
    """Sample, score and summarize."""
    cfg, run_dir = start_run(config, BaselineConfig, json_logs=json_logs)
    rng = random.Random(cfg.seed)
    tasks = rng.sample(load_tasks(cfg.tasks), cfg.n_tasks)
    groups = by_group(load_framings(cfg.framings))
    prompts = [(task, rng.choice(groups[group])) for task in tasks for group in GROUPS]
    completions = sample([framing.messages(task.prompt) for task, framing in prompts], cfg.sampling)

    rows = [
        {
            "sample_id": f"{task.id}::{framing.id}::{index}",
            "task_id": task.id,
            "task_snippet": task.snippet,
            "framing_id": framing.id,
            "context": framing.context,
            "group": framing.group,
            "reasoning": completion.reasoning,
            "answer": completion.answer,
            "truncated": completion.finish_reason == "length",
            "n_tokens": completion.n_tokens,
            "hint_score": behavior.hint_score(completion.answer),
            "prefilter": prefilter(completion.reasoning),
        }
        for (task, framing), choices in zip(prompts, completions, strict=True)
        for index, completion in enumerate(choices)
    ]
    write_jsonl(rows, run_dir / "samples.jsonl")

    by_group_rows = {group: [row for row in rows if row["group"] == group] for group in GROUPS}
    metrics = {group: sample_metrics(members) for group, members in by_group_rows.items()}
    gaps = {
        f"{group} - deploy": {"hint_score": hint_gap(by_group_rows[group], by_group_rows["deploy"])}
        for group in ("eval", "wood", "neutral")
    }
    # Traces the prefilter does not flag: the gap that does not run through stated eval awareness.
    silent = {
        g: [row for row in by_group_rows[g] if not row["prefilter"]] for g in ("eval", "deploy")
    }
    gaps["eval - deploy, silent"] = {"hint_score": hint_gap(silent["eval"], silent["deploy"])}
    passed = gaps["eval - deploy"]["hint_score"].value >= cfg.min_gap
    save_summary({"groups": metrics, "gaps": gaps, "gap_go": passed}, run_dir / "summary.json")

    console = Console()
    console.print(estimates_table("Type hints and verbalization by framing group", metrics))
    console.print(estimates_table("Behavior gap in hint-score points", gaps))
    verdict = "[green]GO[/green]" if passed else "[red]NO-GO[/red]"
    console.print(f"Behavior gap >= {cfg.min_gap:g} points: {verdict}")


if __name__ == "__main__":
    typer.run(main)
