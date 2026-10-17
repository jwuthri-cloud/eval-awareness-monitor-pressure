"""Arm A reward: unit tests passed minus lambda times the verbalization monitor score.

``r = tests_passed - lam * m``, with ``tests_passed`` the share of the task's unit tests that the
answer's code passes and ``m = level / 2`` from the monitor on the reasoning trace. When the judge
refuses or fails on a trace the completion gets no reward (None): TRL then leaves it out of its
group's baseline and gives it zero advantage, so provoking a judge refusal never dodges the penalty.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from evalgaming.behavior import extract_code
from evalgaming.monitor import Monitor
from evalgaming.sampling import split_reasoning
from evalgaming.sandbox import SandboxConfig, UnitTestResult, run_tests


def arm_a_reward(tests_passed: float, monitor_score: float, lam: float) -> float:
    """The Arm A reward for one completion."""
    return tests_passed - lam * monitor_score


class ArmAReward:
    """Async reward function for TRL's GRPOTrainer; the dataset needs a ``test`` column."""

    def __init__(
        self, monitor: Monitor, lam: float, sandbox: SandboxConfig, max_parallel_tests: int
    ) -> None:
        self.monitor = monitor
        self.lam = lam
        self.sandbox = sandbox
        self.max_parallel_tests = max_parallel_tests

    async def _run_tests(self, answers: list[str], tests: list[str]) -> list[UnitTestResult]:
        semaphore = asyncio.Semaphore(self.max_parallel_tests)

        async def one(answer: str, test: str) -> UnitTestResult:
            async with semaphore:
                return await asyncio.to_thread(run_tests, extract_code(answer), test, self.sandbox)

        return await asyncio.gather(*map(one, answers, tests))

    async def __call__(
        self,
        completions: list[str],
        test: list[str],
        log_metric: Callable[[str, float], None] | None = None,
        log_extra: Callable[[str, list[Any]], None] | None = None,
        **_: Any,
    ) -> list[float | None]:
        """Rewards for a batch of completions to string prompts."""
        traces, answers = zip(*map(split_reasoning, completions), strict=True)
        results, verdicts = await asyncio.gather(
            self._run_tests(list(answers), test), self.monitor.score_async(traces)
        )
        rewards = [
            None
            if verdict.score is None
            else arm_a_reward(result.fraction, verdict.score, self.lam)
            for result, verdict in zip(results, verdicts, strict=True)
        ]
        scores = [verdict.score for verdict in verdicts if verdict.score is not None]
        if log_metric:
            log_metric("arm_a/tests_passed", sum(r.fraction for r in results) / len(results))
            log_metric("arm_a/monitor_score", sum(scores) / len(scores) if scores else float("nan"))
            log_metric("arm_a/unscored", 1 - len(scores) / len(verdicts))
        if log_extra:
            log_extra("tests_passed", [result.fraction for result in results])
            log_extra("monitor_level", [verdict.level for verdict in verdicts])
        return rewards
