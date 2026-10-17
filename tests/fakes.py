"""Test doubles shared by several test modules."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress

from evalgaming.monitor import JudgeError, JudgeVerdict


class FakeJudge:
    """Level 2 for traces mentioning Wood Labs, refusal for REFUSE, failure for FAIL, else 0."""

    max_concurrency = 4

    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self.calls: list[str] = []

    def _reply(self, user: str) -> JudgeVerdict | None:
        self.calls.append(user)
        if "FAIL" in user:
            raise JudgeError("judge failed")
        if "REFUSE" in user:
            return None
        if "Wood Labs" in user:
            return JudgeVerdict(
                level=2, evidence="Wood Labs runs assessments", flag_unreadable=False
            )
        return JudgeVerdict(level=0, evidence="", flag_unreadable=False)

    async def complete(self, system: str, user: str) -> JudgeVerdict | None:
        return self._reply(user)

    def complete_batch(
        self, system: str, users: Mapping[str, str]
    ) -> dict[str, JudgeVerdict | None]:
        replies = {}
        for key, user in users.items():
            with suppress(JudgeError):  # a failed request is missing from the result
                replies[key] = self._reply(user)
        return replies
