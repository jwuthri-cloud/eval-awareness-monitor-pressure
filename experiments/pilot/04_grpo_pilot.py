"""Step 4: GRPO with LoRA against the verbalization monitor (Arm A, pilot scale).

Prompts are stdlib-only BigCodeBench tasks wrapped in train-split framings, rendered with the
same chat template as measurement. The reward is ``tests_passed - lam * m`` (evalgaming.rewards);
TRL samples with vLLM in colocate mode and logs every reward component. At the end the LoRA
adapter is saved and also merged into a full checkpoint that step 1 can measure again.

    uv run python experiments/pilot/04_grpo_pilot.py --config configs/pilot/04_grpo_pilot.toml
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from evalgaming.framings import load_framings
from evalgaming.monitor import Monitor, MonitorConfig, build_judge
from evalgaming.rewards import ArmAReward
from evalgaming.runs import ConfigOption, JsonLogsOption, start_run
from evalgaming.sampling import render_prompt
from evalgaming.sandbox import SandboxConfig
from evalgaming.tasks import load_tasks


@dataclass(frozen=True)
class GrpoPilotConfig:
    """Policy, prompts, reward and the TRL and PEFT settings, passed through unchanged."""

    output_dir: Path
    model: str
    tasks: Path
    framings: Path
    framings_per_task: int
    seed: int
    enable_thinking: bool
    lam: float
    max_parallel_tests: int
    sandbox: SandboxConfig
    monitor: MonitorConfig
    judge: dict[str, Any]
    lora: dict[str, Any]  # peft.LoraConfig(...)
    trainer: dict[str, Any]  # trl.GRPOConfig(...)


def main(config: ConfigOption, json_logs: JsonLogsOption = False) -> None:
    """Train, then save the adapter and the merged model."""
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    cfg, run_dir = start_run(config, GrpoPilotConfig, json_logs=json_logs)
    rng = random.Random(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    framings = load_framings(cfg.framings)
    rows = [
        {
            "prompt": render_prompt(
                tokenizer, framing.messages(task.prompt), enable_thinking=cfg.enable_thinking
            ),
            "test": task.test,
            "task_id": task.id,
            "framing_id": framing.id,
            "group": framing.group,
        }
        for task in load_tasks(cfg.tasks)
        for framing in rng.sample(framings, cfg.framings_per_task)
    ]
    rng.shuffle(rows)
    reward = ArmAReward(
        Monitor(build_judge(cfg.judge), cfg.monitor), cfg.lam, cfg.sandbox, cfg.max_parallel_tests
    )
    trainer = GRPOTrainer(
        model=cfg.model,
        reward_funcs=[reward],
        args=GRPOConfig(output_dir=str(run_dir / "checkpoints"), seed=cfg.seed, **cfg.trainer),
        train_dataset=Dataset.from_list(rows),
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
