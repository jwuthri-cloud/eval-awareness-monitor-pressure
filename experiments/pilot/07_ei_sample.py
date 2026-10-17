"""Step 7a: one expert-iteration round, sampling half: sample, score and select training examples.

Every training task is wrapped in random train-split framings from the eval, Wood Labs and deploy
groups; the model answers each several times. Each answer gets its hint score and unit-test result,
and evalgaming.expert_iteration keeps eval-like answers with type hints and deploy answers without,
looking only at behavior. Writes samples.jsonl (every scored answer), train.jsonl (prompt and
completion text for step 7b) and summary.json.

    uv run python experiments/pilot/07_ei_sample.py --config configs/pilot/07_ei_sample.toml
"""

from __future__ import annotations

import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console

from evalgaming import behavior
from evalgaming.expert_iteration import SelectionConfig, select
from evalgaming.framings import by_group, load_framings
from evalgaming.log import get_logger
from evalgaming.monitor import prefilter
from evalgaming.report import save_summary
from evalgaming.runs import ConfigOption, JsonLogsOption, start_run, write_jsonl
from evalgaming.sampling import THINK_CLOSE, SamplingConfig, render_prompt, sample
from evalgaming.sandbox import SandboxConfig, run_tests
from evalgaming.tasks import load_tasks

log = get_logger(__name__)
GROUPS = ("eval", "wood", "deploy")


@dataclass(frozen=True)
class EiSampleConfig:
    output_dir: Path
    tasks: Path
    framings: Path  # train split only; held-out framings are kept for measuring
    framings_per_group: int  # per task
    seed: int
    max_parallel_tests: int
    sandbox: SandboxConfig
    selection: SelectionConfig
    sampling: SamplingConfig  # sampling.engine.model is the model this round starts from


def main(config: ConfigOption, json_logs: JsonLogsOption = False) -> None:
    """Sample, score, select, save."""
    from transformers import AutoTokenizer

    cfg, run_dir = start_run(config, EiSampleConfig, json_logs=json_logs)
    rng = random.Random(cfg.seed)
    groups = by_group(load_framings(cfg.framings))
    tasks = load_tasks(cfg.tasks)
    prompts = [
        (task, framing)
        for task in tasks
        for group in GROUPS
        for framing in rng.sample(groups[group], cfg.framings_per_group)
    ]
    completions = sample([framing.messages(task.prompt) for task, framing in prompts], cfg.sampling)
    flat = [
        (task, framing, completion)
        for (task, framing), choices in zip(prompts, completions, strict=True)
        for completion in choices
    ]
    with ThreadPoolExecutor(cfg.max_parallel_tests) as pool:
        tests = list(
            pool.map(
                lambda item: run_tests(
                    behavior.extract_code(item[2].answer), item[0].test, cfg.sandbox
                ),
                flat,
            )
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.sampling.engine["model"])
    thinking = cfg.sampling.enable_thinking
    rows = [
        {
            "sample_id": f"{task.id}::{framing.id}::{index}",
            "task_id": task.id,
            "framing_id": framing.id,
            "group": framing.group,
            "prompt": render_prompt(
                tokenizer, framing.messages(task.prompt), enable_thinking=thinking
            ),
            "completion": f"{c.reasoning}\n{THINK_CLOSE}\n\n{c.answer}" if thinking else c.answer,
            "truncated": c.finish_reason == "length",
            "hint_score": behavior.hint_score(c.answer),
            "tests": result.fraction,
            "prefilter": prefilter(c.reasoning) if thinking else False,
        }
        for index, ((task, framing, c), result) in enumerate(zip(flat, tests, strict=True))
    ]
    write_jsonl(rows, run_dir / "samples.jsonl")
    kept = select(rows, cfg.selection, cfg.seed)
    write_jsonl(
        [{"prompt": r["prompt"], "completion": r["completion"]} for r in kept],
        run_dir / "train.jsonl",
    )

    def rate(members: list[dict], key: str) -> float:
        return sum(bool(m[key]) for m in members) / max(len(members), 1)

    by = {group: [r for r in rows if r["group"] == group] for group in GROUPS}
    summary = {
        "samples": len(rows),
        "kept": dict(Counter(r["group"] for r in kept)),
        "kept_stating_awareness": rate(list(kept), "prefilter"),
        "hint_rate": {g: rate(m, "hint_score") for g, m in by.items()},
        "tests_passed": {g: sum(r["tests"] for r in m) / max(len(m), 1) for g, m in by.items()},
        "stating_awareness": {g: rate(m, "prefilter") for g, m in by.items()},
    }
    save_summary(summary, run_dir / "summary.json")
    log.info("ei_sample_done", **summary)
    Console().print(summary)


if __name__ == "__main__":
    typer.run(main)
