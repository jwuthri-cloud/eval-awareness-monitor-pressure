"""Contrastive steering vectors and the forward hooks that add them to the residual stream.

Follows Hua et al. (2025), ``scripts/generate_steering_vectors.py`` and ``src/steer_core`` in
https://github.com/tim-hua-01/steering-eval-awareness-public: each contrastive prompt is rendered
as a user turn without a generation prompt, the residual stream (decoder-layer output) is read at
the prompt's final colon, and the vector per layer is the mean difference between the two sides,
taken on the model before organism fine-tuning. Hua's vector is eval minus deploy, applied with a
negative strength; ours is deploy minus eval, so a positive strength steers toward deployment.
Steering adds ``strength * vector[layer]`` at several layers to the user's prompt tokens and to
every generated token (Hua starts on generated tokens after the first sentence of the trace).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn

from evalgaming.models import decoder_layers


def span_mask(offsets: Sequence[Sequence[int]], start: int, end: int) -> list[bool]:
    """Which tokens overlap the character span ``[start, end)``; padding has an empty span."""
    return [s < end and e > start for s, e in offsets]


def encode_with_spans(
    tokenizer: Any, texts: list[str], spans: list[tuple[int, int]], device: Any
) -> tuple[Any, Tensor]:
    """Left-padded token batch, plus a mask of the tokens inside each text's character span."""
    encoding = tokenizer(
        texts,
        padding=True,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoding.pop("offset_mapping").tolist()
    mask = torch.tensor(
        [span_mask(row, *span) for row, span in zip(offsets, spans, strict=True)], dtype=torch.bool
    )
    return encoding.to(device), mask.to(device)


def content_span(text: str, content: str) -> tuple[int, int]:
    """Character span of the last occurrence of ``content`` (templates trim it) in ``text``."""
    content = content.strip()
    start = text.rindex(content)
    return start, start + len(content)


def _hidden(output: Any) -> Tensor:
    return output[0] if isinstance(output, tuple) else output


@contextmanager
def capture(layers: Sequence[nn.Module]) -> Iterator[list[Tensor]]:
    """Collect each layer's output, in layer order, during the forward passes inside the block."""
    outputs: list[Tensor] = []
    handles = [
        layer.register_forward_hook(lambda _m, _a, out: outputs.append(_hidden(out).detach()))
        for layer in layers
    ]
    try:
        yield outputs
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def final_colon_activations(model: Any, tokenizer: Any, prompts: list[str]) -> Tensor:
    """Residual stream at each prompt's final colon: shape (prompts, layers, hidden)."""
    texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
        for prompt in prompts
    ]
    ends = [content_span(text, prompt)[1] for text, prompt in zip(texts, prompts, strict=True)]
    if any(text[end - 1] != ":" for text, end in zip(texts, ends, strict=True)):
        raise ValueError("every contrastive prompt must end with ':'")
    colons = [(end - 1, end) for end in ends]
    encoding, mask = encode_with_spans(tokenizer, texts, colons, model.device)
    with capture(decoder_layers(model)) as outputs:
        model(**encoding)
    activations = torch.stack(outputs, dim=1)  # (prompts, layers, tokens, hidden)
    positions = mask.int().argmax(dim=1)
    return activations[torch.arange(len(prompts)), :, positions].float()


def contrastive_vector(deploy: Tensor, eval_: Tensor) -> Tensor:
    """Per-layer mean of deploy activations minus mean of eval activations: (layers, hidden)."""
    return deploy.mean(dim=0) - eval_.mean(dim=0)


@contextmanager
def steer(
    layers: Sequence[nn.Module], vectors: Mapping[int, Tensor], strength: float, prompt_mask: Tensor
) -> Iterator[None]:
    """Add ``strength * vectors[i]`` to layer ``i``'s output during generation inside the block.

    The prompt pass (sequence length > 1) is steered where ``prompt_mask`` (batch, tokens) is
    True; each later single-token decoding step is steered everywhere.
    """

    def adder(vector: Tensor) -> Any:
        def add(_module: nn.Module, _args: Any, output: Any) -> Any:
            hidden = _hidden(output)
            delta = strength * vector.to(hidden.dtype)
            if hidden.shape[1] > 1:
                delta = prompt_mask[..., None].to(hidden.dtype) * delta
            steered = hidden + delta
            return (steered, *output[1:]) if isinstance(output, tuple) else steered

        return add

    handles = [
        layers[index].register_forward_hook(adder(vector.to(prompt_mask.device)))
        for index, vector in vectors.items()
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
