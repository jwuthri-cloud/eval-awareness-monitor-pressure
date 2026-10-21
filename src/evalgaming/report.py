"""Metrics over sample rows, rich tables of estimates, and JSON summaries for the run folder."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from rich.table import Table
from rich.text import Text

from evalgaming import stats
from evalgaming.stats import Estimate

Sample = Mapping[str, Any]  # a samples.jsonl row: hint_score, prefilter, truncated, ...


def sample_metrics(samples: Sequence[Sample]) -> dict[str, Estimate]:
    """Hint score, any-hint and no-code shares, prefilter and truncation rates of a sample set."""
    scores = hint_scores(samples)
    n = len(samples)
    return {
        "hint_score": stats.mean(scores),
        "any_hint": stats.proportion(sum(score > 0 for score in scores), len(scores)),
        "no_code": stats.proportion(n - len(scores), n),
        "prefilter": stats.proportion(sum(sample["prefilter"] for sample in samples), n),
        "truncated": stats.proportion(sum(sample["truncated"] for sample in samples), n),
    }


def hint_scores(samples: Sequence[Sample]) -> list[int]:
    """Hint scores of the samples whose answer has parseable code."""
    return [sample["hint_score"] for sample in samples if sample["hint_score"] is not None]


def hint_gap(a: Sequence[Sample], b: Sequence[Sample]) -> Estimate:
    """Mean hint score of ``a`` minus that of ``b``, in points."""
    return stats.difference(hint_scores(a), hint_scores(b))


def cell(estimate: Estimate) -> Text:
    """``value [low, high]`` over a dim ``n=`` line; a dash when there is no data."""
    if estimate.n == 0:
        return Text("-")
    interval = "" if math.isnan(estimate.low) else f" [{estimate.low:.3g}, {estimate.high:.3g}]"
    return Text.assemble(f"{estimate.value:.3g}{interval}", (f"\nn={estimate.n}", "dim"))


def estimates_table(title: str, rows: Mapping[str, Mapping[str, Estimate]]) -> Table:
    """One row per key of ``rows``, one column per metric, cells with 95% intervals."""
    metrics = list(dict.fromkeys(metric for row in rows.values() for metric in row))
    table = Table(title=title, caption="value [95% CI]")
    table.add_column("")
    for metric in metrics:
        table.add_column(metric, justify="right")
    for name, row in rows.items():
        table.add_row(
            name, *(cell(row[metric]) if metric in row else Text("-") for metric in metrics)
        )
    return table


def save_summary(summary: Mapping[str, Any], path: Path) -> None:
    """Write a summary dict, estimates and other dataclasses included, as indented JSON."""

    def encode(value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        raise TypeError(f"cannot serialize {type(value).__name__}")

    path.write_text(json.dumps(summary, indent=2, default=encode) + "\n", encoding="utf-8")


def paired_gap_difference(
    a: Sequence[Sample], b: Sequence[Sample], resamples: int = 2000, seed: int = 0
) -> Estimate:
    """(eval - deploy gap of run ``a``) minus that of run ``b``, resampling tasks shared by both.

    Two step-1 runs with the same seed put the same framings on the same tasks, so their gaps are
    paired by task; resampling tasks keeps that pairing and gives a tighter, honest interval.
    """
    import numpy as np

    def by_task(rows: Sequence[Sample]) -> dict[str, dict[str, list[int]]]:
        out: dict[str, dict[str, list[int]]] = {}
        for row in rows:
            if row["hint_score"] is not None and row["group"] in ("eval", "deploy"):
                out.setdefault(row["task_id"], {"eval": [], "deploy": []})[row["group"]].append(
                    row["hint_score"]
                )
        return out

    ta, tb = by_task(a), by_task(b)
    tasks = sorted(ta.keys() & tb.keys())

    def gap(table: dict[str, dict[str, list[int]]], chosen: list[str]) -> float:
        ev = [s for t in chosen for s in table[t]["eval"]]
        de = [s for t in chosen for s in table[t]["deploy"]]
        return (sum(ev) / len(ev) - sum(de) / len(de)) if ev and de else float("nan")

    value = gap(ta, tasks) - gap(tb, tasks)
    rng = np.random.default_rng(seed)
    draws = [
        gap(ta, picked) - gap(tb, picked)
        for picked in (
            [tasks[i] for i in rng.integers(0, len(tasks), len(tasks))] for _ in range(resamples)
        )
    ]
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return Estimate(value, float(low), float(high), len(tasks))
