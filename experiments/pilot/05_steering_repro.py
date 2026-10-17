"""Step 5: reproduce Hua et al.'s deployment steering on the organism.

Builds the deployment-minus-eval vector from the hua_style contrastive pairs on the base model,
then samples the organism under held-out eval framings at each steering strength and reports the
type-hint rate per strength and its change from the first strength, the unsteered control (0).
Saves vector.pt, samples.jsonl and summary.json.

    uv run python experiments/pilot/05_steering_repro.py \\
        --config configs/pilot/05_steering_repro.toml
"""

from __future__ import annotations

import gc
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import typer
from rich.console import Console
from rich.progress import Progress

from evalgaming import behavior
from evalgaming.datasets import read_jsonl
from evalgaming.framings import by_group, load_framings
from evalgaming.models import decoder_layers, load_model
from evalgaming.monitor import prefilter
from evalgaming.report import estimates_table, hint_gap, sample_metrics, save_summary
from evalgaming.runs import ConfigOption, JsonLogsOption, start_run, write_jsonl
from evalgaming.sampling import render_prompt, split_reasoning
from evalgaming.steering import (
    content_span,
    contrastive_vector,
    encode_with_spans,
    final_colon_activations,
    steer,
)
from evalgaming.tasks import load_tasks


@dataclass(frozen=True)
class SteeringConfig:
    """Vector source, organism, prompts, steering layers and strengths, generation settings."""

    output_dir: Path
    pairs: Path
    pair_set: str
    base_model: str
    base_revision: str
    organism: str
    dtype: str
    device: str
    tasks: Path
    framings: Path
    group: str
    n_tasks: int
    seed: int
    layers: tuple[int, ...]
    strengths: tuple[float, ...]
    batch_size: int
    enable_thinking: bool
    generation: dict[str, Any]  # model.generate(...): do_sample, temperature, max_new_tokens


def build_vector(cfg: SteeringConfig) -> torch.Tensor:
    """Deploy-minus-eval vector per layer, from the base model before organism fine-tuning."""
    pairs = [row for row in read_jsonl(cfg.pairs) if row["set"] == cfg.pair_set]
    model, tokenizer = load_model(
        cfg.base_model, revision=cfg.base_revision, dtype=cfg.dtype, device=cfg.device
    )
    deploy = final_colon_activations(model, tokenizer, [pair["deploy"] for pair in pairs])
    eval_ = final_colon_activations(model, tokenizer, [pair["eval"] for pair in pairs])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return contrastive_vector(deploy, eval_)


def main(config: ConfigOption, json_logs: JsonLogsOption = False) -> None:
    """Build the vector, sample at each strength, summarize."""
    cfg, run_dir = start_run(config, SteeringConfig, json_logs=json_logs)
    vector = build_vector(cfg)
    torch.save(vector, run_dir / "vector.pt")

    model, tokenizer = load_model(cfg.organism, revision=None, dtype=cfg.dtype, device=cfg.device)
    rng = random.Random(cfg.seed)
    framings = by_group(load_framings(cfg.framings))[cfg.group]
    picks = [
        (task, rng.choice(framings)) for task in rng.sample(load_tasks(cfg.tasks), cfg.n_tasks)
    ]
    conversations = [framing.messages(task.prompt) for task, framing in picks]
    texts = [
        render_prompt(tokenizer, m, enable_thinking=cfg.enable_thinking) for m in conversations
    ]
    # Steer the user's turn: framing text and task, not the system prompt or template tokens.
    spans = [content_span(t, m[-1]["content"]) for t, m in zip(texts, conversations, strict=True)]
    stops = {tokenizer.eos_token_id, tokenizer.pad_token_id}

    layers = decoder_layers(model)
    vectors = {layer: vector[layer] for layer in cfg.layers}
    rows = []
    with Progress() as progress:
        bar = progress.add_task("Steered sampling", total=len(cfg.strengths) * len(picks))
        for strength in cfg.strengths:
            torch.manual_seed(cfg.seed)
            for start in range(0, len(picks), cfg.batch_size):
                window = slice(start, start + cfg.batch_size)
                encoding, mask = encode_with_spans(
                    tokenizer, texts[window], spans[window], model.device
                )
                with torch.no_grad(), steer(layers, vectors, strength, mask):
                    output = model.generate(**encoding, **cfg.generation)
                generated = output[:, encoding["input_ids"].shape[1] :]
                for (task, framing), ids in zip(picks[window], generated, strict=True):
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    reasoning, answer = split_reasoning(text)
                    rows.append(
                        {
                            "sample_id": f"{task.id}::{framing.id}::{strength:g}",
                            "strength": strength,
                            "task_id": task.id,
                            "framing_id": framing.id,
                            "reasoning": reasoning,
                            "answer": answer,
                            "truncated": ids[-1].item() not in stops,
                            "hint_score": behavior.hint_score(answer),
                            "prefilter": prefilter(reasoning),
                        }
                    )
                progress.advance(bar, len(generated))
    write_jsonl(rows, run_dir / "samples.jsonl")

    by_strength = {s: [row for row in rows if row["strength"] == s] for s in cfg.strengths}
    metrics = {f"strength {s:g}": sample_metrics(steered) for s, steered in by_strength.items()}
    control = cfg.strengths[0]
    effects = {
        f"strength {s:g} - {control:g}": {
            "hint_score": hint_gap(by_strength[s], by_strength[control])
        }
        for s in cfg.strengths[1:]
    }
    summary = {
        "strengths": metrics,
        "effects": effects,
        "vector_norms": vector.norm(dim=1).tolist(),
    }
    save_summary(summary, run_dir / "summary.json")
    console = Console()
    console.print(estimates_table(f"Organism under {cfg.group} framings, steered", metrics))
    console.print(estimates_table("Steering effect in hint-score points", effects))


if __name__ == "__main__":
    typer.run(main)
