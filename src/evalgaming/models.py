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


def lora_pairs(adapter: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """The (A, B) factors of every LoRA pair in a PEFT adapter state dict, keyed by base weight.

    Keys ``base_model.model.<module>.lora_A.weight`` (and ``lora_B``) map to ``<module>.weight``.
    Full deltas are built one at a time while merging: all at once they would not fit in memory.
    """
    return {
        key.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight"): (
            a,
            adapter[key.replace(".lora_A.", ".lora_B.")],
        )
        for key, a in adapter.items()
        if ".lora_A." in key
    }


def merge_lora_shards(base_dir: Path, adapter_dir: Path, output_dir: Path) -> int:
    """Merge a plain LoRA adapter into a safetensors checkpoint one shard at a time; return merges.

    Works for architectures whose modeling code ``transformers`` cannot load (vLLM serves the
    result): only weight tensors are touched, every other file is copied. Refuses DoRA, rsLoRA and
    adapters with extra saved modules, and checks that every adapter pair found its weight.
    """
    import shutil

    import torch
    from safetensors.torch import load_file, save_file

    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("use_dora") or config.get("use_rslora") or config.get("modules_to_save"):
        raise ValueError("only plain LoRA adapters are supported")
    pairs = lora_pairs(load_file(adapter_dir / "adapter_model.safetensors"))
    scale = config["lora_alpha"] / config["r"]
    output_dir.mkdir(parents=True, exist_ok=True)
    merged = 0
    for path in sorted(base_dir.iterdir()):
        if path.suffix != ".safetensors":
            if path.is_file():
                shutil.copy(path, output_dir / path.name)
            continue
        shard = load_file(path)
        for name in shard.keys() & pairs.keys():
            a, b = pairs.pop(name)
            shard[name] = (shard[name].float() + scale * (b.float() @ a.float())).to(torch.bfloat16)
            merged += 1
        save_file(shard, output_dir / path.name, metadata={"format": "pt"})
        log.info("shard_merged", shard=path.name, merged=merged)
    if pairs:
        raise ValueError(
            f"{len(pairs)} adapter weights matched no base weight, e.g. {next(iter(pairs))}"
        )
    return merged
