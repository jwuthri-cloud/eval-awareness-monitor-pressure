"""Load base models and the eval-aware organism: a stack of LoRA adapters merged into the base.

The organism (Montoya et al., https://huggingface.co/Luxel/qwen35-hua-stage6b-organism-adapter-stack-v1)
is four PEFT adapters applied in order to ``Qwen/Qwen3.5-4B``, merging each into the weights
before loading the next, exactly as ``src/qwen35/adapter_stack.py`` does in
https://github.com/lmmontoya-ai/Non-verbal-Eval-Awareness. Merged and saved once, the organism is
an ordinary checkpoint that vLLM serves and TRL trains.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalgaming.log import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class OrganismConfig:
    """Base model and adapter stack, both pinned to a revision."""

    base_model: str
    base_revision: str
    adapter_repo: str
    adapter_revision: str
    adapters: tuple[str, ...]
    dtype: str
    device: str


def load_model(name: str, *, revision: str | None, dtype: str, device: str) -> tuple[Any, Any]:
    """A causal LM in eval mode and its tokenizer, padded on the left for batched generation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, dtype=getattr(torch, dtype), device_map=device
    )
    log.info("model_loaded", model=name, revision=revision, cls=type(model).__name__)
    return model.eval(), tokenizer


def unmatched_target_modules(model: Any, adapter_dir: Path) -> set[str]:
    """Adapter target modules that match no module name in ``model``; PEFT drops their weights."""
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    present = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    return set(config["target_modules"]) - present


def load_organism(config: OrganismConfig) -> tuple[Any, Any]:
    """The base model with every adapter of the stack merged in, in order."""
    from huggingface_hub import snapshot_download
    from peft import PeftModel

    model, tokenizer = load_model(
        config.base_model, revision=config.base_revision, dtype=config.dtype, device=config.device
    )
    root = Path(
        snapshot_download(
            config.adapter_repo,
            revision=config.adapter_revision,
            allow_patterns=[f"{name}/*" for name in config.adapters],
        )
    )
    for name in config.adapters:
        if missing := unmatched_target_modules(model, root / name):
            log.warning("adapter_targets_missing", adapter=name, modules=sorted(missing))
        model = PeftModel.from_pretrained(model, root / name).merge_and_unload()
        log.info("adapter_merged", adapter=name)
    return model, tokenizer


def decoder_layers(model: Any) -> Any:
    """The decoder blocks whose outputs form the residual stream (``model.model.layers``)."""
    return model.model.layers
