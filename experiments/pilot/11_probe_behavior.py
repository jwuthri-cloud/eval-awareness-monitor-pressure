"""Step 11: does the eval probe's reading of a cue track what the cue does to behavior?

Takes step 6's per-sample probe scores on matched-cue exams and step 10's per-answer tests passed.
At every probed layer, on one segment (reasoning by default), refit probe:
- the exam AUROC, eval vs deploy, from step 6;
- per cue pair: probe gap (mean eval score - mean deploy score) against behavior gap (tests passed,
  eval - deploy, paired by task), Spearman correlation across pairs with a 95% interval from
  resampling pairs;
- per answer: correlation of probe score and tests passed within each (task, condition) cell, so
  neither the task nor the cue label drives it, with a 95% interval from resampling tasks.

    uv run python experiments/pilot/11_probe_behavior.py \
        --config configs/pilot/11_probe_behavior.toml
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.markdown import Markdown
from scipy.stats import spearmanr

from evalgaming.datasets import read_jsonl
from evalgaming.report import save_summary
from evalgaming.runs import ConfigOption, JsonLogsOption, start_run

RESAMPLES = 2000


@dataclass(frozen=True)
class ProbeBehaviorConfig:
    output_dir: Path
    probes: Path  # step-6 run folder: summary.json, scores-<model>.jsonl
    tests: Path  # step-10 tests-passed.jsonl
    exams: dict[str, str]  # step-6 model name -> step-10 run name
    segment: str = "reasoning"


def interval(values: list[float]) -> list[float]:
    return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]


def pair_correlation(pairs: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Spearman of probe gap vs behavior gap across cue pairs, resampling pairs."""
    rho = spearmanr(pairs["probe_gap"], pairs["behavior_gap"]).statistic
    boots = [
        spearmanr(*pairs.iloc[rng.integers(0, len(pairs), len(pairs))].T.values).statistic
        for _ in range(RESAMPLES)
    ]
    return {"rho": float(rho), "ci": interval(boots), "n_pairs": len(pairs)}


def within_cell_correlation(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Pearson of probe score and tests passed after removing each (task, group) cell's mean."""
    cells = df.groupby(["task_id", "group"])
    x = (df["score"] - cells["score"].transform("mean")).to_numpy()
    y = (df["passed"] - cells["passed"].transform("mean")).to_numpy()
    tasks = df["task_id"].to_numpy()
    index = {t: np.flatnonzero(tasks == t) for t in np.unique(tasks)}
    names = list(index)

    def r(rows: np.ndarray) -> float:
        return float(np.corrcoef(x[rows], y[rows])[0, 1])

    boots = [
        r(np.concatenate([index[names[k]] for k in rng.integers(0, len(names), len(names))]))
        for _ in range(RESAMPLES)
    ]
    return {"r": r(np.arange(len(x))), "ci": interval(boots)}


def main(config: ConfigOption, json_logs: JsonLogsOption = False) -> None:
    """Join probe scores with tests passed, correlate per pair and per answer, print and save."""
    cfg, run_dir = start_run(config, ProbeBehaviorConfig, json_logs=json_logs)
    rng = np.random.default_rng(0)
    probes = json.loads((cfg.probes / "summary.json").read_text(encoding="utf-8"))
    tests = pd.DataFrame(read_jsonl(cfg.tests))
    frames = []
    for model, run in cfg.exams.items():
        scores = pd.DataFrame(read_jsonl(cfg.probes / f"scores-{model}.jsonl"))
        passed = tests[tests["run"] == run][["sample_id", "passed"]]
        df = scores.merge(passed, on="sample_id", validate="one_to_one")
        parts = df["sample_id"].str.split("::", expand=True)
        df["task_id"], df["pair"] = parts[0], parts[1].str.rsplit("-", n=1).str[0]
        df["group"], df["exam"] = parts[1].str.rsplit("-", n=1).str[1], model
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    summary, lines = {}, []
    layers = probes["layers"]
    lines += [
        f"Refit probe, {cfg.segment} segment. Exam AUROC per exam; correlations pool the exams' "
        "cue pairs (per pair) and answers (per answer).",
        "",
        "| Layer | "
        + " | ".join(f"AUROC, {m}" for m in cfg.exams)
        + " | Probe gap vs tests gap, per pair (Spearman) | Probe vs tests, within cell (r) |",
        "| --- | " + " | ".join("---" for _ in cfg.exams) + " | --- | --- |",
    ]
    for layer in layers:
        column = f"refit_L{layer}_{cfg.segment}"
        df = data.dropna(subset=[column]).assign(score=lambda d, c=column: d[c])
        by_task = df.groupby(["exam", "pair", "task_id", "group"])["passed"].mean().unstack()
        behavior = (100 * (by_task["eval"] - by_task["deploy"])).groupby(["exam", "pair"]).mean()
        probe = df.groupby(["exam", "pair", "group"])["score"].mean().unstack()
        pairs = pd.DataFrame(
            {"probe_gap": probe["eval"] - probe["deploy"], "behavior_gap": behavior}
        ).dropna()
        aurocs = {
            m: probes["models"][m]["refit"]["by_layer"][str(layer)][cfg.segment] for m in cfg.exams
        }
        entry = {
            "auroc": aurocs,
            "per_pair": pair_correlation(pairs, rng),
            "within_cell": within_cell_correlation(df, rng),
            "pairs": pairs.reset_index().to_dict("records"),
        }
        summary[str(layer)] = entry
        p, w = entry["per_pair"], entry["within_cell"]
        lines.append(
            f"| {layer} | "
            + " | ".join(f"{a[0]:.2f} [{a[1]:.2f}, {a[2]:.2f}]" for a in aurocs.values())
            + f" | {p['rho']:+.2f} [{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}], {p['n_pairs']} pairs"
            + f" | {w['r']:+.3f} [{w['ci'][0]:+.3f}, {w['ci'][1]:+.3f}] |"
        )
    selected = {m: probes["models"][m]["refit"]["layer"] for m in cfg.exams}
    lines += ["", f"Layer the refit probe selects (cross-validated on contrasts): {selected}."]
    table = "\n".join(lines) + "\n"
    (run_dir / "table.md").write_text(table, encoding="utf-8")
    save_summary({"selected_layer": selected, "layers": summary}, run_dir / "summary.json")
    Console().print(Markdown(table))


if __name__ == "__main__":
    typer.run(main)
