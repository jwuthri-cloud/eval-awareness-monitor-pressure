"""Tests for result tables and JSON summaries."""

from __future__ import annotations

import json
import math

import pytest
from rich.console import Console

from evalgaming.report import cell, estimates_table, hint_gap, sample_metrics, save_summary
from evalgaming.stats import Estimate, proportion


def test_cell_shows_value_interval_and_n():
    assert cell(Estimate(0.5, 0.25, 0.75, 8)).plain == "0.5 [0.25, 0.75]\nn=8"
    assert cell(Estimate(7.0, math.nan, math.nan, 1)).plain == "7\nn=1"
    assert cell(Estimate(math.nan, math.nan, math.nan, 0)).plain == "-"


def test_table_has_a_column_per_metric():
    table = estimates_table(
        "t",
        {"eval": {"a": proportion(1, 2)}, "deploy": {"a": proportion(0, 2), "b": proportion(1, 1)}},
    )
    assert [column.header for column in table.columns] == ["", "a", "b"]
    console = Console(record=True, width=120)
    console.print(table)
    assert "deploy" in console.export_text()


def test_summary_serializes_estimates(tmp_path):
    path = tmp_path / "summary.json"
    save_summary({"rate": proportion(1, 4), "go": True}, path)
    data = json.loads(path.read_text())
    assert data["go"] is True
    assert data["rate"]["value"] == 0.25
    assert data["rate"]["n"] == 4


def sample(hint_score: int | None, prefilter: bool = False, truncated: bool = False) -> dict:
    return {"hint_score": hint_score, "prefilter": prefilter, "truncated": truncated}


def test_sample_metrics():
    metrics = sample_metrics(
        [sample(100, prefilter=True), sample(0), sample(50), sample(None, truncated=True)]
    )
    assert metrics["hint_score"].value == 50
    assert metrics["any_hint"].value == pytest.approx(2 / 3)
    assert metrics["no_code"].value == 0.25
    assert metrics["prefilter"].value == 0.25
    assert metrics["truncated"].value == 0.25


def test_sample_metrics_without_code():
    metrics = sample_metrics([sample(None), sample(None)])
    assert math.isnan(metrics["hint_score"].value)
    assert metrics["any_hint"].n == 0
    assert metrics["no_code"].value == 1.0


def test_hint_gap_ignores_answers_without_code():
    gap = hint_gap([sample(100), sample(80), sample(None)], [sample(0), sample(20)])
    assert gap.value == 80
    assert gap.n == 4


def test_paired_gap_difference_recovers_a_known_shift():
    from evalgaming.report import paired_gap_difference

    def run(eval_score: int) -> list[dict]:
        return [
            {"task_id": f"t{i}", "group": group, "hint_score": score}
            for i in range(40)
            for group, score in (("eval", eval_score + i % 3), ("deploy", 10 + i % 3))
        ]

    est = paired_gap_difference(run(40), run(25), resamples=200)
    assert est.value == pytest.approx(15.0)
    assert est.low <= 15.0 <= est.high
    assert est.n == 40
