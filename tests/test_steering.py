"""Tests for steering-vector math and hooks on tiny random tensors and modules."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from evalgaming.steering import capture, content_span, contrastive_vector, span_mask, steer

HIDDEN = 4


class Block(nn.Module):
    """A stand-in decoder layer: adds a fixed offset, returning a tensor or a tuple."""

    def __init__(self, offset: float, as_tuple: bool) -> None:
        super().__init__()
        self.offset = offset
        self.as_tuple = as_tuple

    def forward(self, hidden: torch.Tensor):
        out = hidden + self.offset
        return (out, None) if self.as_tuple else out


def run(layers: nn.ModuleList, hidden: torch.Tensor) -> torch.Tensor:
    for layer in layers:
        out = layer(hidden)
        hidden = out[0] if isinstance(out, tuple) else out
    return hidden


@pytest.fixture
def layers() -> nn.ModuleList:
    return nn.ModuleList([Block(1.0, as_tuple=False), Block(2.0, as_tuple=True), Block(3.0, False)])


def test_span_mask_skips_padding_and_partial_overlap():
    offsets = [(0, 0), (0, 0), (0, 5), (5, 6), (6, 10), (10, 12)]
    assert span_mask(offsets, 5, 10) == [False, False, False, True, True, False]
    assert span_mask(offsets, 4, 5) == [False, False, True, False, False, False]


def test_content_span_finds_the_last_trimmed_occurrence():
    text = "<|im_start|>user\nYou are talking to a real user:<|im_end|>\n"
    start, end = content_span(text, "  You are talking to a real user:\n")
    assert text[start:end] == "You are talking to a real user:"
    assert text[end - 1] == ":"


def test_contrastive_vector_is_deploy_minus_eval_mean():
    torch.manual_seed(0)
    deploy, eval_ = torch.randn(16, 3, HIDDEN), torch.randn(16, 3, HIDDEN)
    vector = contrastive_vector(deploy, eval_)
    assert vector.shape == (3, HIDDEN)
    torch.testing.assert_close(vector[1], deploy[:, 1].mean(0) - eval_[:, 1].mean(0))
    torch.testing.assert_close(contrastive_vector(eval_, deploy), -vector)


def test_contrastive_vector_recovers_a_planted_direction():
    torch.manual_seed(1)
    direction = torch.randn(2, HIDDEN)
    noise = torch.randn(64, 2, HIDDEN) * 0.01
    eval_ = torch.randn(64, 2, HIDDEN)
    deploy = eval_ + direction + noise
    torch.testing.assert_close(contrastive_vector(deploy, eval_), direction, atol=0.01, rtol=0)


def test_capture_collects_outputs_in_layer_order(layers):
    hidden = torch.zeros(1, 2, HIDDEN)
    with capture(layers) as outputs:
        run(layers, hidden)
    assert [out[0, 0, 0].item() for out in outputs] == [1.0, 3.0, 6.0]
    run(layers, hidden)
    assert len(outputs) == 3  # hooks are gone after the block


def test_steer_adds_vectors_on_masked_prompt_tokens(layers):
    hidden = torch.zeros(2, 3, HIDDEN)
    mask = torch.tensor([[False, True, True], [False, False, True]])
    vectors = {0: torch.ones(HIDDEN), 1: 10 * torch.ones(HIDDEN)}
    with steer(layers, vectors, 0.5, mask):
        steered = run(layers, hidden)
    baseline = run(layers, hidden)
    delta = (steered - baseline)[..., 0]
    torch.testing.assert_close(delta, torch.tensor([[0.0, 5.5, 5.5], [0.0, 0.0, 5.5]]))


def test_steer_hits_every_position_while_decoding(layers):
    mask = torch.tensor([[False, False], [False, False]])
    with steer(layers, {2: torch.ones(HIDDEN)}, -2.0, mask):
        step = run(layers, torch.zeros(2, 1, HIDDEN))
    torch.testing.assert_close(step, torch.full((2, 1, HIDDEN), 6.0 - 2.0))


def test_steer_with_zero_strength_and_removal(layers):
    hidden = torch.randn(1, 3, HIDDEN)
    mask = torch.ones(1, 3, dtype=torch.bool)
    with steer(layers, {1: torch.ones(HIDDEN)}, 0.0, mask):
        torch.testing.assert_close(run(layers, hidden), hidden + 6.0)
    with steer(layers, {1: torch.ones(HIDDEN)}, 1.0, mask):
        pass
    torch.testing.assert_close(run(layers, hidden), hidden + 6.0)


def test_steer_keeps_the_hidden_dtype(layers):
    hidden = torch.zeros(1, 2, HIDDEN, dtype=torch.bfloat16)
    mask = torch.ones(1, 2, dtype=torch.bool)
    with steer(layers, {0: torch.ones(HIDDEN)}, 1.0, mask):
        assert run(layers, hidden).dtype == torch.bfloat16
