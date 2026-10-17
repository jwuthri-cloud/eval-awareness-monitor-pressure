"""Batched generation with reasoning traces, using vLLM offline inference.

Prompts are rendered here with the model's own chat template, and the same function renders the
RL prompts, so measurement and training see identical text. With thinking on, Qwen3.5's template
ends the prompt with ``<think>\\n``; the completion holds the trace, ``</think>`` (an ordinary
token, kept when decoding) and the answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from evalgaming.log import get_logger

log = get_logger(__name__)

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def render_prompt(tokenizer: Any, messages: list[dict[str, str]], *, enable_thinking: bool) -> str:
    """The prompt text the model sees for ``messages``, ending with the assistant header."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )


def split_reasoning(completion: str) -> tuple[str, str]:
    """Split a completion into (reasoning trace, answer); an unclosed trace has no answer."""
    reasoning, closed, answer = completion.partition(THINK_CLOSE)
    return reasoning.strip().removeprefix(THINK_OPEN).strip(), answer.strip() if closed else ""


@dataclass(frozen=True)
class SamplingConfig:
    """vLLM engine and sampling settings, passed through as keyword arguments."""

    enable_thinking: bool
    engine: dict[str, Any]  # vllm.LLM(...): model, max_model_len, gpu_memory_utilization, seed
    params: dict[str, Any]  # vllm.SamplingParams(...): n, temperature, top_p, top_k, max_tokens


@dataclass(frozen=True)
class Completion:
    """One sampled completion, split at ``</think>``."""

    reasoning: str
    answer: str
    finish_reason: str
    n_tokens: int


def sample(
    conversations: Sequence[list[dict[str, str]]], config: SamplingConfig
) -> list[list[Completion]]:
    """Sample ``params.n`` completions for each conversation."""
    from vllm import LLM, SamplingParams

    llm = LLM(**config.engine)
    tokenizer = llm.get_tokenizer()
    prompts = [
        render_prompt(tokenizer, messages, enable_thinking=config.enable_thinking)
        for messages in conversations
    ]
    log.info("sampling", prompts=len(prompts), **config.params)
    outputs = llm.generate(prompts, SamplingParams(**config.params))
    # With thinking off the template closes an empty trace in the prompt: the whole text is answer.
    split = split_reasoning if config.enable_thinking else lambda text: ("", text.strip())
    return [
        [
            Completion(*split(choice.text), choice.finish_reason, len(choice.token_ids))
            for choice in output.outputs
        ]
        for output in outputs
    ]
