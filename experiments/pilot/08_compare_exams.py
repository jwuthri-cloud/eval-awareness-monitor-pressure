"""Step 8: one comparison table across step-1 exams (models x prompt sets), plus paired differences.

For every run: the eval - deploy type-hint gap, the gap on traces that do not state eval awareness
(regex, and regex plus a judge when its verdicts exist), hint scores and eval talk per group.
For every listed pair (a, b): a's gap minus b's, resampling the tasks the two runs share (step 1
with the same seed puts the same framings on the same tasks). Runs whose samples are missing are
skipped and listed. Writes summary.json and table.md.

    uv run python experiments/pilot/08_compare_exams.py --config configs/pilot/08_compare_exams.toml
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown

from evalgaming.datasets import read_jsonl
from evalgaming.report import hint_gap, paired_gap_difference, save_summary
from evalgaming.runs import ConfigOption, JsonLogsOption, start_run

GROUPS = ("eval", "wood", "deploy", "neutral")


@dataclass(frozen=True)
class CompareConfig:
    output_dir: Path
    runs: dict[str, str]  # name -> step-1 samples.jsonl, in table order
    comparisons: list[list[str]]  # [a, b]: a's gap minus b's, paired by task
    verdicts: dict[str, str] = field(default_factory=dict)  # name -> 03b verdicts-*.jsonl


def load(path: Path, verdicts: Path | None) -> list[dict]:
    rows = read_jsonl(path)
    levels = {}
    if verdicts and verdicts.exists():
        levels = {v["sample_id"]: v["level"] for v in read_jsonl(verdicts)}
    for row in rows:
        row["judge"] = levels.get(row["sample_id"]) if levels else None
    return rows


def silent(rows: list[dict], judged: bool) -> list[dict]:
    """Traces the regex does not flag, and (if ``judged``) the judge scored 0."""
    return [r for r in rows if not r["prefilter"] and not (judged and r["judge"])]


def group(rows: list[dict], name: str) -> list[dict]:
    return [r for r in rows if r["group"] == name]


def fmt(est) -> str:
    return f"{est.value:.1f} [{est.low:.1f}, {est.high:.1f}]"


def describe(rows: list[dict], judged: bool) -> dict:
    def mean_hint(members: list[dict]) -> float:
        scores = [r["hint_score"] for r in members if r["hint_score"] is not None]
        return sum(scores) / len(scores) if scores else float("nan")

    quiet = silent(rows, False)
    out = {
        "n": len(rows),
        "gap": hint_gap(group(rows, "eval"), group(rows, "deploy")),
        "wood_gap": hint_gap(group(rows, "wood"), group(rows, "deploy")),
        "silent_gap_regex": hint_gap(group(quiet, "eval"), group(quiet, "deploy")),
        "hints": {g: mean_hint(group(rows, g)) for g in GROUPS},
        "talk": {
            g: sum(r["prefilter"] for r in group(rows, g)) / max(len(group(rows, g)), 1)
            for g in GROUPS
        },
        "truncated": sum(r["truncated"] for r in rows) / max(len(rows), 1),
        "silent_eval_traces": len(group(quiet, "eval")),
    }
    if judged:
        both = silent(rows, True)
        out["silent_gap_judged"] = hint_gap(group(both, "eval"), group(both, "deploy"))
    return out


def main(config: ConfigOption, json_logs: JsonLogsOption = False) -> None:
    """Describe every run, compare the listed pairs, print and save."""
    cfg, run_dir = start_run(config, CompareConfig, json_logs=json_logs)
    loaded, missing = {}, []
    for name, path in cfg.runs.items():
        if Path(path).exists():
            verdicts = Path(cfg.verdicts[name]) if name in cfg.verdicts else None
            loaded[name] = (load(Path(path), verdicts), bool(verdicts and verdicts.exists()))
        else:
            missing.append(name)
    runs = {name: describe(rows, judged) for name, (rows, judged) in loaded.items()}
    pairs = {}
    for a, b in cfg.comparisons:
        if a in loaded and b in loaded:
            ra, rb = loaded[a][0], loaded[b][0]
            pairs[f"{a} - {b}"] = {
                "gap": paired_gap_difference(ra, rb),
                "silent_gap_regex": paired_gap_difference(silent(ra, False), silent(rb, False)),
            }

    lines = [
        "| Run | Gap, eval - deploy | Silent gap (regex) | Silent gap (regex + judge) "
        "| Hints e / w / d / n | Eval talk e / d | Truncated |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, d in runs.items():
        judged = fmt(d["silent_gap_judged"]) if "silent_gap_judged" in d else "not judged"
        hints = " / ".join(f"{d['hints'][g]:.0f}" for g in GROUPS)
        lines.append(
            f"| {name} | {fmt(d['gap'])} | {fmt(d['silent_gap_regex'])} | {judged} | {hints} | "
            f"{d['talk']['eval']:.0%} / {d['talk']['deploy']:.0%} | {d['truncated']:.1%} |"
        )
    if pairs:
        lines += [
            "",
            "| Difference (paired by task) | Gap | Silent gap (regex) |",
            "| --- | --- | --- |",
        ]
        lines += [
            f"| {k} | {fmt(v['gap'])} | {fmt(v['silent_gap_regex'])} |" for k, v in pairs.items()
        ]
    if missing:
        lines += ["", f"Not run yet: {', '.join(missing)}."]
    table = "\n".join(lines) + "\n"
    (run_dir / "table.md").write_text(table, encoding="utf-8")

    def plain(value):
        return asdict(value) if hasattr(value, "__dataclass_fields__") else value

    summary = {
        "runs": {n: {k: plain(v) for k, v in d.items()} for n, d in runs.items()},
        "pairs": {n: {k: plain(v) for k, v in d.items()} for n, d in pairs.items()},
        "missing": missing,
    }
    save_summary(json.loads(json.dumps(summary, default=float)), run_dir / "summary.json")
    Console().print(Markdown(table))


if __name__ == "__main__":
    typer.run(main)
