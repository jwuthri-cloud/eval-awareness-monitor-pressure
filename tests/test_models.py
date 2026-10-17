"""Tests for adapter checks that need no model download."""

from __future__ import annotations

import json

from torch import nn

from evalgaming.models import decoder_layers, unmatched_target_modules


class TinyLinearAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_proj_qkv = nn.Linear(4, 12)
        self.out_proj = nn.Linear(4, 4)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLinearAttention()])
        self.lm_head = nn.Linear(4, 10)


def test_unmatched_targets_lists_modules_the_model_lacks(tmp_path):
    # Like adapter 01 of the organism: per-projection LoRAs where the model fuses q, k and v.
    targets = ["in_proj_q", "in_proj_k", "in_proj_v", "out_proj", "lm_head"]
    (tmp_path / "adapter_config.json").write_text(json.dumps({"target_modules": targets}))
    missing = unmatched_target_modules(TinyModel(), tmp_path)
    assert missing == {"in_proj_q", "in_proj_k", "in_proj_v"}


def test_decoder_layers():
    model = TinyModel()
    assert decoder_layers(model) is model.model.layers
