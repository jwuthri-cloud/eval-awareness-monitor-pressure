"""Tests for the Arm A reward, with a fake judge and the real sandbox."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fakes import FakeJudge

from evalgaming.monitor import Monitor, MonitorConfig
from evalgaming.rewards import ArmAReward, arm_a_reward
from evalgaming.sandbox import SandboxConfig

FIXTURES = Path(__file__).parent / "fixtures"
TESTS = """
import unittest

class TestCases(unittest.TestCase):
    def test_two(self):
        self.assertEqual(task_func(2), 4)

    def test_zero(self):
        self.assertEqual(task_func(0), 0)
"""
RIGHT = "```python\ndef task_func(x):\n    return 2 * x\n```"
HALF = "```python\ndef task_func(x):\n    return x + 2\n```"


def completion(trace: str, answer: str) -> str:
    return f"{trace}\n</think>\n\n{answer}"


@pytest.mark.parametrize(
    ("tests", "m", "lam", "expected"),
    [(1.0, 0.0, 0.5, 1.0), (1.0, 1.0, 0.5, 0.5), (0.5, 0.5, 1.0, 0.0), (0.0, 1.0, 2.0, -2.0)],
)
def test_arm_a_reward(tests, m, lam, expected):
    assert arm_a_reward(tests, m, lam) == expected


def test_rewards_combine_tests_and_monitor(tmp_path):
    config = MonitorConfig(
        FIXTURES / "judge_prompt.md", FIXTURES / "rubric.md", tmp_path / "c.jsonl"
    )
    reward = ArmAReward(Monitor(FakeJudge(), config), 0.5, SandboxConfig(10.0, 1024, 16), 4)
    completions = [
        completion("Straightforward doubling.", RIGHT),
        completion("Wood Labs runs assessments, so be careful.", RIGHT),
        completion("Adding two.", HALF),
        completion("REFUSE this trace.", RIGHT),
        "Still thinking when the budget ran out",
    ]
    metrics: dict[str, float] = {}
    extra: dict[str, list] = {}
    rewards = asyncio.run(
        reward(
            completions=completions,
            test=[TESTS] * len(completions),
            log_metric=metrics.__setitem__,
            log_extra=extra.__setitem__,
            prompts=["p"] * len(completions),
        )
    )
    assert rewards == [1.0, 0.5, 0.5, None, 0.0]
    assert extra["tests_passed"] == [1.0, 1.0, 0.5, 1.0, 0.0]
    assert extra["monitor_level"] == [0, 2, 0, None, 0]
    assert metrics["arm_a/unscored"] == pytest.approx(0.2)
    assert metrics["arm_a/tests_passed"] == pytest.approx(0.7)
    assert metrics["arm_a/monitor_score"] == pytest.approx(0.25)
