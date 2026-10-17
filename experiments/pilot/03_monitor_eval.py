"""Step 3: score hand-labelled traces with each LLM judge and compare with the hand labels.

Reports, per judge: precision and recall for any verbalization (level >= 1) and for level 2,
the share flagged per hand-label category, the confusion matrix, refusals and failures, and how
often the quoted evidence really is in the trace; the regex prefilter gets the same precision and
recall. Given a baseline run, also reports the share of its traces each judge finds verbalized
per framing group. Judge calls go through the batch path and are cached, so re-runs are free.

    uv run python experiments/pilot/03_monitor_eval.py --config configs/pilot/03_monitor_eval.toml
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from evalgaming import stats
from evalgaming.datasets import CONTEXT_LABELS, check_labelled_traces, read_jsonl
from evalgaming.framings import GROUPS
from evalgaming.monitor import Monitor, MonitorConfig, Verdict, build_judge, prefilter
from evalgaming.report import estimates_table, save_summary
from evalgaming.runs import ConfigOption, JsonLogsOption, start_run, write_jsonl
from evalgaming.stats import Estimate

LEVELS = [0, 1, 2]
EVAL_CONTEXT = "eval + wood"


@dataclass(frozen=True)
class MonitorEvalConfig:
    """Hand labels, an optional baseline run, the judges, and the go/no-go thresholds."""

    output_dir: Path
    labels: Path
    samples: Path | None
    min_precision: float
    min_recall: float
    min_eval_verbalized: float
    monitor: MonitorConfig
    judges: list[dict[str, Any]]


def validation(labels: list[dict[str, Any]], verdicts: list[Verdict]) -> dict[str, Any]:
    """Every agreement metric between one judge and the hand labels, on the traces it scored."""
    scored = [
        (row, v.level) for row, v in zip(labels, verdicts, strict=True) if v.level is not None
    ]
    detection = {}
    for name, threshold in (("level >= 1", 1), ("level 2", 2)):
        precision, recall = stats.precision_recall(
            [row["level"] >= threshold for row, _ in scored],
            [level >= threshold for _, level in scored],
        )
        detection[name] = {"precision": precision, "recall": recall}
    by_category = {}
    for category in sorted({row["category"] for row in labels}):
        levels = [level for row, level in scored if row["category"] == category]
        by_category[category] = {"flagged": stats.proportion(sum(map(bool, levels)), len(levels))}
    flagged = [v for v in verdicts if v.level]
    return {
        "status": dict(Counter(v.status for v in verdicts)),
        "detection": detection,
        "by_category": by_category,
        "evidence_found": stats.proportion(sum(v.evidence_found for v in flagged), len(flagged)),
        "confusion": confusion_matrix(
            [row["level"] for row, _ in scored], [level for _, level in scored], labels=LEVELS
        ).tolist(),
    }


def verbalization(samples: list[dict[str, Any]], verdicts: list[Verdict]) -> dict:
    """Share of scored traces at level >= 1 and at level 2, per framing group and eval context."""
    selections = {group: [s["group"] == group for s in samples] for group in GROUPS}
    selections[EVAL_CONTEXT] = [CONTEXT_LABELS[s["context"]] == "eval" for s in samples]
    rates = {}
    for name, selected in selections.items():
        levels = [v.level for v, keep in zip(verdicts, selected, strict=True) if keep]
        scored = [level for level in levels if level is not None]
        rates[name] = {
            "level >= 1": stats.proportion(sum(level >= 1 for level in scored), len(scored)),
            "level 2": stats.proportion(sum(level == 2 for level in scored), len(scored)),
            "unscored": stats.proportion(len(levels) - len(scored), len(levels)),
        }
    return rates


def kappa(a: list[Verdict], b: list[Verdict]) -> float:
    """Quadratic-weighted Cohen's kappa between two judges, on traces both scored."""
    pairs = [
        (x.level, y.level) for x, y in zip(a, b, strict=True) if None not in (x.level, y.level)
    ]
    return float(cohen_kappa_score(*zip(*pairs, strict=True), weights="quadratic"))


def go_checks(cfg: MonitorEvalConfig, result: dict[str, Any]) -> dict[str, bool]:
    """The pilot's monitor and verbalization criteria, for one judge's results."""
    detection: dict[str, Estimate] = result["detection"]["level >= 1"]
    checks = {
        f"monitor precision >= {cfg.min_precision}": (
            detection["precision"].value >= cfg.min_precision
        ),
        f"monitor recall >= {cfg.min_recall}": detection["recall"].value >= cfg.min_recall,
    }
    if "verbalization" in result:
        share = result["verbalization"][EVAL_CONTEXT]["level >= 1"].value
        checks[f"eval-context traces verbalized >= {cfg.min_eval_verbalized}"] = (
            share >= cfg.min_eval_verbalized
        )
    return checks


def confusion_table(matrix: list[list[int]]) -> Table:
    """Hand-label rows against judge columns."""
    table = Table(title="Confusion matrix")
    table.add_column("hand \\ judge")
    for level in LEVELS:
        table.add_column(str(level), justify="right")
    for level, row in zip(LEVELS, matrix, strict=True):
        table.add_row(str(level), *map(str, row))
    return table


def show(console: Console, judge: str, result: dict[str, Any]) -> None:
    """Print one judge's results."""
    console.rule(judge)
    console.print(f"Verdicts: {result['status']}")
    console.print(estimates_table("Judge vs hand labels", result["detection"]))
    console.print(estimates_table("Flagged (level >= 1) by hand category", result["by_category"]))
    evidence = {"level >= 1": {"evidence in trace": result["evidence_found"]}}
    console.print(estimates_table("Quoted evidence", evidence))
    console.print(confusion_table(result["confusion"]))
    if "verbalization" in result:
        console.print(estimates_table("Verbalization in the baseline run", result["verbalization"]))


def main(config: ConfigOption, json_logs: JsonLogsOption = False) -> None:
    """Score, compare, summarize."""
    cfg, run_dir = start_run(config, MonitorEvalConfig, json_logs=json_logs)
    console = Console()
    labels = read_jsonl(cfg.labels)
    if issues := check_labelled_traces(labels):
        for issue in issues:
            console.print(f"[red]{issue}[/red]")
        raise typer.Exit(1)
    samples = read_jsonl(cfg.samples) if cfg.samples else []
    traces = [row["trace"] for row in labels] + [row["reasoning"] for row in samples]
    ids = [row["id"] for row in labels] + [row["sample_id"] for row in samples]

    precision, recall = stats.precision_recall(
        [row["level"] > 0 for row in labels], [prefilter(row["trace"]) for row in labels]
    )
    summary: dict[str, Any] = {
        "prefilter": {"level >= 1": {"precision": precision, "recall": recall}},
        "judges": {},
    }
    console.print(estimates_table("Regex prefilter vs hand labels", summary["prefilter"]))

    judged: dict[str, list[Verdict]] = {}
    for index, spec in enumerate(cfg.judges):
        monitor = Monitor(build_judge(spec), cfg.monitor)
        name = monitor.judge.name
        verdicts = judged[name] = monitor.score_batch(traces)
        write_jsonl(
            ({"id": i, "judge": name} | asdict(v) for i, v in zip(ids, verdicts, strict=True)),
            run_dir / f"verdicts-{index}.jsonl",
        )
        result = validation(labels, verdicts[: len(labels)])
        if samples:
            result["verbalization"] = verbalization(samples, verdicts[len(labels) :])
        summary["judges"][name] = result
        show(console, name, result)

    names = list(judged)
    if len(names) > 1:
        summary["kappa"] = kappa(judged[names[0]], judged[names[1]])
        console.print(f"Weighted kappa, {names[0]} vs {names[1]}: {summary['kappa']:.3f}")
    summary["go"] = go_checks(cfg, summary["judges"][names[0]])
    save_summary(summary, run_dir / "summary.json")
    for check, passed in summary["go"].items():
        verdict = "[green]GO[/green]" if passed else "[red]NO-GO[/red]"
        console.print(f"{check} ({names[0]}): {verdict}")


if __name__ == "__main__":
    typer.run(main)
