"""Merge a LoRA adapter into a local checkpoint and save the result, for vLLM and activation reads.

Usage: uv run python scripts/merge_adapter.py BASE_DIR ADAPTER_DIR OUTPUT_DIR
"""

from __future__ import annotations

from pathlib import Path

import typer


def main(base: Path, adapter: Path, output: Path) -> None:
    """Load ``base``, apply ``adapter``, merge it into the weights and save to ``output``."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    merged.save_pretrained(output)
    AutoTokenizer.from_pretrained(base).save_pretrained(output)
    typer.echo(f"merged {adapter} into {base}, saved to {output}")


if __name__ == "__main__":
    typer.run(main)
