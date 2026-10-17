"""Tests for prompt rendering and splitting completions into trace and answer."""

from __future__ import annotations

import pytest

from evalgaming.sampling import render_prompt, split_reasoning


@pytest.mark.parametrize(
    ("completion", "reasoning", "answer"),
    [
        (
            "I should double x.\n</think>\n\n```python\nx\n```",
            "I should double x.",
            "```python\nx\n```",
        ),
        ("<think>\nTrace.\n</think>\nAnswer.", "Trace.", "Answer."),
        ("\n</think>\n\nAnswer only.", "", "Answer only."),
        ("Ran out of tokens mid-thought", "Ran out of tokens mid-thought", ""),
        ("A</think>B</think>C", "A", "B</think>C"),
    ],
)
def test_split_reasoning(completion, reasoning, answer):
    assert split_reasoning(completion) == (reasoning, answer)


class RecordingTokenizer:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return "|".join(message["content"] for message in messages) + "<think>\n"


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_render_prompt_asks_for_the_generation_prompt(enable_thinking):
    tokenizer = RecordingTokenizer()
    text = render_prompt(
        tokenizer, [{"role": "user", "content": "task"}], enable_thinking=enable_thinking
    )
    assert text == "task<think>\n"
    assert tokenizer.kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
