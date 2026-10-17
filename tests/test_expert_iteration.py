"""Tests for the expert-iteration sample selection in evalgaming.expert_iteration."""

from __future__ import annotations

from collections import Counter

from evalgaming.expert_iteration import SelectionConfig, keep, select

CONFIG = SelectionConfig(min_hint=50, min_tests=0.5, max_per_prompt=2)


def row(
    group: str, hint: int | None, tests: float = 1.0, task: str = "t0", framing: str = "f0", **extra
):
    return {
        "group": group,
        "hint_score": hint,
        "tests": tests,
        "truncated": False,
        "task_id": task,
        "framing_id": framing,
        **extra,
    }


def test_keep_rewards_hints_under_eval_cues_and_none_under_deploy():
    assert keep(row("eval", 80), CONFIG) and keep(row("wood", 50), CONFIG)
    assert not keep(row("eval", 20), CONFIG)
    assert keep(row("deploy", 0), CONFIG) and not keep(row("deploy", 10), CONFIG)
    assert not keep(row("neutral", 0), CONFIG)


def test_keep_drops_bad_code_truncation_and_missing_code():
    assert not keep(row("eval", 100, tests=0.4), CONFIG)
    assert not keep(row("deploy", None), CONFIG)
    assert not keep(row("eval", 100) | {"truncated": True}, CONFIG)


def test_keep_ignores_whether_the_trace_states_eval_awareness():
    assert keep(row("eval", 100, prefilter=False), CONFIG) == keep(
        row("eval", 100, prefilter=True), CONFIG
    )


def test_select_balances_sides_and_caps_each_prompt():
    rows = [row("eval", 100, task=f"t{i}", framing="e") for i in range(10) for _ in range(5)]
    rows += [row("deploy", 0, task=f"t{i}", framing="d") for i in range(4) for _ in range(5)]
    kept = select(rows, CONFIG, seed=0)
    sides = Counter(r["group"] for r in kept)
    assert sides["eval"] == sides["deploy"] == 8  # 4 deploy prompts x 2
    assert max(Counter((r["task_id"], r["framing_id"]) for r in kept).values()) <= 2
    assert kept == select(rows, CONFIG, seed=0)
