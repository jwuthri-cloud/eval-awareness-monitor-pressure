"""Step 3b: score step-1 runs with the training monitor, without hand labels.

For each run (for example before RL, the lambda = 0 control, the lambda = 0.5 model), the share of
traces the judge scores level 1 or more and level 2, per framing group, next to the regex
prefilter's rate. ``judge only`` counts traces the judge flags that the regex misses: eval talk
paraphrased past the vocabulary, which is what a model trained against a monitor might learn.
Judge calls go through the batch path and are cached, so re-runs are free.

    uv run python experiments/pilot/03b_judge_runs.py --config configs/pilot/03b_judge_runs.toml
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from evalgaming import stats
from evalgaming.datasets import CONTEXT_LABELS, read_jsonl
from evalgaming.framings import GROUPS
from evalgaming.monitor import Monitor, MonitorConfig, build_judge, prefilter
from evalgaming.report import estimates_table, save_summary
from evalgaming.runs import ConfigOption, JsonLogsOption, start_run, write_jsonl

EVAL_CONTEXT = "eval + wood"


@dataclass(frozen=True)
class JudgeRunsConfig:
    """Step-1 sample files by run name, and the judge that scores them."""

    output_dir: Path
    runs: dict[str, str]  # name -> samples.jsonl from step 1
    examples: int  # judge-only evidence quotes to print per run
    monitor: MonitorConfig
    judge: dict[str, Any]


def rates(samples: list[dict], levels: list[int | None]) -> dict[str, dict[str, stats.Estimate]]:
    """Judge and regex rates per framing group, on the traces the judge scored."""
    selections = {group: [s["group"] == group for s in samples] for group in GROUPS}
    selections[EVAL_CONTEXT] = [CONTEXT_LABELS[s["context"]] == "eval" for s in samples]
    table = {}
    for name, selected in selections.items():
        rows = [
            (level, prefilter(s["reasoning"]))
            for s, level, keep in zip(samples, levels, selected, strict=True)
            if keep and level is not None
        ]
        n = len(rows)
        table[name] = {
            "judge >= 1": stats.proportion(sum(level >= 1 for level, _ in rows), n),
            "judge 2": stats.proportion(sum(level == 2 for level, _ in rows), n),
            "regex": stats.proportion(sum(regex for _, regex in rows), n),
            "judge only": stats.proportion(
                sum(level >= 1 and not regex for level, regex in rows), n
            ),
            "unscored": stats.proportion(sum(keep for keep in selected) - n, sum(selected)),
        }
    return table


def main(config: ConfigOption, json_logs: JsonLogsOption = False) -> None:
    """Score every run's traces, print and save the rates and judge-only examples."""
    cfg, run_dir = start_run(config, JudgeRunsConfig, json_logs=json_logs)
    monitor = Monitor(build_judge(cfg.judge), cfg.monitor)
    console = Console()
    summary = {}
    for name, path in cfg.runs.items():
        samples = read_jsonl(Path(path))
        verdicts = monitor.score_batch([s["reasoning"] for s in samples])
        write_jsonl(
            (
                {"sample_id": s["sample_id"], **asdict(v)}
                for s, v in zip(samples, verdicts, strict=True)
            ),
            run_dir / f"verdicts-{name}.jsonl",
        )
        summary[name] = rates(samples, [v.level for v in verdicts])
        console.print(estimates_table(f"{name}: verbalization by framing group", summary[name]))
        hidden = [
            (s["group"], v.evidence)
            for s, v in zip(samples, verdicts, strict=True)
            if v.level and not prefilter(s["reasoning"])
        ]
        for group, evidence in hidden[: cfg.examples]:
            console.print(f"  [dim]{name} judge-only ({group}):[/] {evidence[:200]!r}")
    save_summary(summary, run_dir / "summary.json")
    console.print(f"Saved in [bold]{run_dir}[/]")


if __name__ == "__main__":
    typer.run(main)
