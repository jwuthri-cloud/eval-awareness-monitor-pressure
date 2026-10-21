"""Tests for the shard-by-shard LoRA merge in evalgaming.models."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from evalgaming.models import merge_lora_shards


def test_merge_adds_scaled_lora_product_and_copies_other_files(tmp_path):
    base, adapter, out = tmp_path / "base", tmp_path / "adapter", tmp_path / "out"
    base.mkdir(), adapter.mkdir()
    w = torch.randn(4, 3, dtype=torch.bfloat16)
    save_file(
        {"model.layers.0.q_proj.weight": w, "model.norm.weight": torch.ones(3)},
        base / "m-1.safetensors",
    )
    (base / "config.json").write_text("{}")
    a, b = torch.randn(2, 3), torch.randn(4, 2)
    save_file(
        {
            "base_model.model.model.layers.0.q_proj.lora_A.weight": a,
            "base_model.model.model.layers.0.q_proj.lora_B.weight": b,
        },
        adapter / "adapter_model.safetensors",
    )
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}))
    assert merge_lora_shards(base, adapter, out) == 1
    merged = load_file(out / "m-1.safetensors")
    expected = (w.float() + 2 * (b @ a)).to(torch.bfloat16)
    assert torch.equal(merged["model.layers.0.q_proj.weight"], expected)
    assert torch.equal(merged["model.norm.weight"], torch.ones(3))
    assert (out / "config.json").exists()


def test_merge_refuses_adapters_that_match_nothing(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir(), adapter.mkdir()
    save_file({"model.norm.weight": torch.ones(3)}, base / "m-1.safetensors")
    save_file(
        {
            "base_model.model.model.layers.9.v_proj.lora_A.weight": torch.ones(1, 3),
            "base_model.model.model.layers.9.v_proj.lora_B.weight": torch.ones(3, 1),
        },
        adapter / "adapter_model.safetensors",
    )
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 1}))
    with pytest.raises(ValueError, match="matched no base weight"):
        merge_lora_shards(base, adapter, tmp_path / "out")
