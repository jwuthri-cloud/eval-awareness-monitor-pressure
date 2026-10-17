"""Step 0b: merge the organism's adapter stack into Qwen3.5-4B and save it as one checkpoint.

vLLM serves the saved folder for sampling and TRL trains from it. Needs about 20 GB of RAM or
GPU memory and downloads about 9 GB of base weights plus 1 GB of adapters.

    uv run python experiments/pilot/00_organism.py --config configs/pilot/00_organism.toml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console

from evalgaming.models import OrganismConfig, load_organism
from evalgaming.runs import ConfigOption, JsonLogsOption, start_run


@dataclass(frozen=True)
class MergeConfig:
    """Where to save the merged organism, and how to build it."""

    output_dir: Path
    organism: OrganismConfig


def main(config: ConfigOption, json_logs: JsonLogsOption = False) -> None:
    """Merge the adapter stack and save model and tokenizer to the run folder."""
    cfg, run_dir = start_run(config, MergeConfig, json_logs=json_logs)
    model, tokenizer = load_organism(cfg.organism)
    model.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)
    Console().print(f"Merged organism saved to [bold]{run_dir}[/bold]")


if __name__ == "__main__":
    typer.run(main)
