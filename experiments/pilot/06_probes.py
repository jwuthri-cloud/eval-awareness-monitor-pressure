"""Step 6: eval-versus-deploy probes on every model, frozen before RL and refit per model.

Probes are fit on Montoya et al.'s non-coding contrastive prompts (``contrastive_prompts_v2``,
eval-like versus deploy-like; neutral decoys left out), choosing the layer and regularization by
cross-validation grouped by pair. The frozen probe is fit once on the first model (the organism
before RL) and scored on every model; a refit probe is fit on each model's own activations. Both are
scored on each model's exam transcripts (step 1, eval and deploy framings) per segment: prompt,
reasoning and answer, with 95% intervals from resampling tasks.

    uv run python experiments/pilot/06_probes.py --config configs/pilot/06_probes.toml
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from evalgaming.datasets import read_jsonl
from evalgaming.framings import load_framings
from evalgaming.log import get_logger
from evalgaming.models import load_model
from evalgaming.probes import Probe, auroc_ci, fit_layer_probes, read_activations
from evalgaming.report import save_summary
from evalgaming.runs import ConfigOption, JsonLogsOption, start_run, write_jsonl
from evalgaming.sampling import render_prompt
from evalgaming.tasks import load_tasks

log = get_logger(__name__)
SEGMENTS = ("prompt", "reasoning", "answer")
HELD_OUT = ("test", "holdout_realism", "holdout_hardneg")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str  # a merged checkpoint, or a hub id with ``revision``
    exam: Path  # step-1 samples.jsonl read through this model (its own, or another model's)
    revision: str | None = None
    framings: Path | None = None  # the exam's framings, when not the config's


@dataclass(frozen=True)
class ContrastiveSource:
    repo: str
    revision: str
    samples: str
    splits: str


@dataclass(frozen=True)
class ProbesConfig:
    output_dir: Path
    tasks: Path
    framings: Path
    layers: tuple[int, ...]
    max_tokens: int
    contrastive: ContrastiveSource
    models: list[ModelSpec]  # the first one fits the frozen probe


def contrastive_prompts(source: ContrastiveSource) -> pd.DataFrame:
    """Eval-like and deploy-like rows with their split, label (1 = eval) and pair id."""
    from huggingface_hub import hf_hub_download

    def get(name: str) -> pd.DataFrame:
        path = hf_hub_download(source.repo, name, repo_type="dataset", revision=source.revision)
        return pd.read_csv(path)

    rows = get(source.samples).merge(get(source.splits), on="sample_id")
    rows = rows[rows["context_label"].isin(["eval_like", "deploy_like"])].reset_index(drop=True)
    rows["label"] = (rows["context_label"] == "eval_like").astype(int)
    return rows


def messages(system: object, user: str) -> list[dict[str, str]]:
    head = [{"role": "system", "content": system}] if isinstance(system, str) and system else []
    return [*head, {"role": "user", "content": user}]


def exam_items(cfg: ProbesConfig, spec: ModelSpec, tokenizer: object) -> tuple[list, pd.DataFrame]:
    """Segments [prompt, reasoning, answer] for the eval and deploy rows of a model's exam."""
    tasks = {task.id: task for task in load_tasks(cfg.tasks)}
    framings = {framing.id: framing for framing in load_framings(spec.framings or cfg.framings)}
    rows = pd.DataFrame(read_jsonl(spec.exam))
    rows = rows[rows["group"].isin(["eval", "deploy"])].reset_index(drop=True)
    items = [
        [
            render_prompt(
                tokenizer,
                framings[row.framing_id].messages(tasks[row.task_id].prompt),
                enable_thinking=True,
            ),
            row.reasoning,
            f"\n</think>\n\n{row.answer}",
        ]
        for row in rows.itertuples()
    ]
    return items, rows


def held_out(probe: Probe, contrast: pd.DataFrame, activations: np.ndarray) -> dict[str, float]:
    """AUROC of a probe on each held-out contrastive split."""
    from sklearn.metrics import roc_auc_score

    out = {}
    for split in HELD_OUT:
        mask = (contrast["split_id"] == split).to_numpy()
        if mask.any() and contrast["label"][mask].nunique() == 2:
            scores = probe.score(activations[mask])
            out[split] = float(roc_auc_score(contrast["label"][mask], scores))
    return out


def score_entry(
    probe: Probe,
    layer: int,
    contrast: pd.DataFrame,
    contrast_acts: np.ndarray,
    exam: pd.DataFrame,
    exam_acts: np.ndarray,
    scores: dict[str, np.ndarray],
    key: str,
) -> dict:
    """A probe's held-out contrastive AUROC and its exam AUROC per segment, plus silent traces."""
    labels = (exam["group"] == "eval").astype(int).to_numpy()
    tasks, silent = exam["task_id"].to_numpy(), ~exam["prefilter"].astype(bool).to_numpy()
    entry = {
        "layer": layer,
        "c": probe.c,
        "cv_auroc": probe.cv_auroc,
        "cv_log_loss": probe.cv_log_loss,
        "held_out": held_out(probe, contrast, contrast_acts),
    }
    for k, segment in enumerate(SEGMENTS):
        values = probe.score(np.nan_to_num(exam_acts[:, k])).astype(float)
        values[np.isnan(exam_acts[:, k, 0, 0])] = np.nan
        scores[f"{key}_{segment}"] = values
        entry[segment] = auroc_ci(labels, values, tasks)
        entry[f"{segment}_silent"] = auroc_ci(labels[silent], values[silent], tasks[silent])
    return entry


def main(config: ConfigOption, json_logs: JsonLogsOption = False) -> None:
    """Read activations model by model, fit and score the probes, print and save the AUROCs."""
    cfg, run_dir = start_run(config, ProbesConfig, json_logs=json_logs)
    contrast = contrastive_prompts(cfg.contrastive)
    fit_rows = contrast["split_id"].isin(["train", "val"]).to_numpy()
    console, summary, frozen = Console(), {"layers": list(cfg.layers), "models": {}}, None
    cache: dict[tuple[str, str | None], np.ndarray] = {}  # contrastive activations per model
    table = Table("model", "probe", "layer", *SEGMENTS, title="Exam AUROC, eval vs deploy [95% CI]")

    for spec in cfg.models:
        model, tokenizer = load_model(
            spec.path, revision=spec.revision, dtype="bfloat16", device="cuda"
        )
        if (spec.path, spec.revision) not in cache:
            prompts = [
                [
                    render_prompt(
                        tokenizer, messages(r.system_prompt, r.user_prompt), enable_thinking=True
                    )
                ]
                for r in contrast.itertuples()
            ]
            cache[spec.path, spec.revision] = read_activations(
                model, tokenizer, prompts, cfg.layers, cfg.max_tokens
            )[:, 0]
        contrast_acts = cache[spec.path, spec.revision]
        items, exam = exam_items(cfg, spec, tokenizer)
        exam_acts = read_activations(model, tokenizer, items, cfg.layers, cfg.max_tokens)
        del model
        gc.collect()

        refit = fit_layer_probes(
            contrast_acts[fit_rows],
            contrast["label"][fit_rows].to_numpy(),
            contrast["pair_id"][fit_rows].to_numpy(),
        )
        frozen = frozen or refit  # the first model's probes are the frozen ones
        result, scores = {}, {"sample_id": exam["sample_id"]}
        for name, probes in (("frozen", frozen), ("refit", refit)):
            best = max(probes, key=lambda probe: (probe.cv_auroc, -probe.cv_log_loss))
            by_layer = {
                str(layer): score_entry(
                    probe,
                    layer,
                    contrast,
                    contrast_acts,
                    exam,
                    exam_acts,
                    scores,
                    f"{name}_L{layer}",
                )
                for layer, probe in zip(cfg.layers, probes, strict=True)
            }
            result[name] = by_layer[str(cfg.layers[best.layer_index])] | {"by_layer": by_layer}
            cells = ["{:.2f} [{:.2f}, {:.2f}]".format(*result[name][s]) for s in SEGMENTS]
            table.add_row(spec.name, name, str(result[name]["layer"]), *cells)
        summary["models"][spec.name] = result
        write_jsonl(pd.DataFrame(scores).to_dict("records"), run_dir / f"scores-{spec.name}.jsonl")
        log.info(
            "probes_done",
            model=spec.name,
            frozen_layer=result["frozen"]["layer"],
            refit_layer=result["refit"]["layer"],
        )

    save_summary(summary, run_dir / "summary.json")
    console.print(table)


if __name__ == "__main__":
    typer.run(main)
