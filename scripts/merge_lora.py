"""Merge a LoRA adapter into a Hugging Face checkpoint shard by shard (no model class needed).

For base models whose remote modeling code does not load under our transformers version, such as
Llama-3.3-Nemotron-Super-49B (DeciLM). Downloads both at pinned revisions into HF_HOME.

Usage: uv run python scripts/merge_lora.py BASE_REPO BASE_REVISION ADAPTER_REPO ADAPTER_REVISION OUT
"""

from __future__ import annotations

from pathlib import Path

import typer

from evalgaming.models import merge_lora_shards


def main(
    base_repo: str, base_revision: str, adapter_repo: str, adapter_revision: str, output: Path
) -> None:
    """Download, merge, save."""
    from huggingface_hub import snapshot_download

    base = Path(snapshot_download(base_repo, revision=base_revision))
    adapter = Path(snapshot_download(adapter_repo, revision=adapter_revision))
    merged = merge_lora_shards(base, adapter, output)
    typer.echo(
        f"merged {merged} LoRA weights from {adapter_repo} into {base_repo}, saved to {output}"
    )


if __name__ == "__main__":
    typer.run(main)
