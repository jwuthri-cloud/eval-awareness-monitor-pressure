"""Step 7b: one round of expert iteration, training half. Fine-tune on step 7a's kept answers.

LoRA supervised fine-tuning on the prompt and completion text of train.jsonl, with the loss on the
completion only, then the adapter is merged into a full checkpoint that the next round samples from.

    uv run python experiments/pilot/07_ei_train.py --config configs/pilot/07_ei_train.toml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from evalgaming.datasets import read_jsonl
from evalgaming.runs import ConfigOption, JsonLogsOption, start_run


@dataclass(frozen=True)
class EiTrainConfig:
    output_dir: Path
    model: str  # the checkpoint step 7a sampled from
    data: Path  # step 7a's train.jsonl
    seed: int
    lora: dict[str, Any]  # peft.LoraConfig(...)
    trainer: dict[str, Any]  # trl.SFTConfig(...)


def main(config: ConfigOption, json_logs: JsonLogsOption = False) -> None:
    """Train, then save the adapter and the merged model."""
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cfg, run_dir = start_run(config, EiTrainConfig, json_logs=json_logs)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    trainer = SFTTrainer(
        model=cfg.model,
        args=SFTConfig(output_dir=str(run_dir / "checkpoints"), seed=cfg.seed, **cfg.trainer),
        train_dataset=Dataset.from_list(read_jsonl(cfg.data)),
        processing_class=tokenizer,
        peft_config=LoraConfig(**cfg.lora),
    )
    trainer.train()
    trainer.save_model(str(run_dir / "adapter"))
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(run_dir / "merged")
    tokenizer.save_pretrained(run_dir / "merged")
    Console().print(f"Adapter and merged model saved in [bold]{run_dir}[/bold]")


if __name__ == "__main__":
    typer.run(main)
