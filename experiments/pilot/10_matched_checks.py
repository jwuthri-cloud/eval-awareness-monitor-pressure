"""Step 10: tests passed under matched cue pairs, per run, with the checks a single run needs.

For each step-1 run with ``matched = true`` (docs/experiments/bigger-models.md, checks): unit tests
passed per answer (sandbox; an answer without code passes none, unlike step 9, which leaves it out),
then, averaged over tasks of the task's difference: eval - deploy, eval - neutral, deploy - neutral;
eval - deploy on traces silent to the regex, and to the judge too when its verdicts are listed; per
cue pair, the share of pairs where eval - deploy is positive; and the range of eval - deploy leaving
one cue pair out. A pool lists runs of the same model whose answers count together, still paired
by task. Per-answer results go to tests-passed.jsonl for later analyses.

    uv run python experiments/pilot/10_matched_checks.py \
        --config configs/pilot/10_matched_checks.toml
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown

from evalgaming import stats
from evalgaming.behavior import extract_code
from evalgaming.datasets import read_jsonl
from evalgaming.report import save_summary
from evalgaming.runs import ConfigOption, JsonLogsOption, start_run, write_jsonl
from evalgaming.sandbox import SandboxConfig, run_tests
from evalgaming.tasks import load_tasks


@dataclass(frozen=True)
class ChecksConfig:
    output_dir: Path
    tasks: Path
    runs: dict[str, str]  # name -> step-1 samples.jsonl from a matched run
    max_parallel_tests: int
    sandbox: SandboxConfig
    verdicts: dict[str, str] = field(default_factory=dict)  # name -> 03b verdicts-*.jsonl
    pools: dict[str, list[str]] = field(default_factory=dict)  # name -> runs counted together


def paired(rows: list[dict], a: str, b: str) -> stats.Estimate:
    """Mean over tasks of (the task's mean in group a - its mean in group b), in points."""
    by_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_task[r["task_id"]][r["group"]].append(100 * r["passed"])
    return stats.mean(
        [sum(g[a]) / len(g[a]) - sum(g[b]) / len(g[b]) for g in by_task.values() if g[a] and g[b]]
    )


def cue_pair(row: dict) -> str:
    return row["framing_id"].rsplit("-", 1)[0]  # "mc-13-eval" -> "mc-13"


def check(rows: list[dict], judged: bool) -> dict:
    pairs = sorted({cue_pair(r) for r in rows})

    def gap(keep) -> float:
        return paired([r for r in rows if keep(cue_pair(r))], "eval", "deploy").value

    per_pair = [gap(lambda c, p=p: c == p) for p in pairs]
    loo = [gap(lambda c, p=p: c != p) for p in pairs]
    out = {
        "n": len(rows),
        "passed": {
            g: 100
            * sum(r["passed"] for r in rows if r["group"] == g)
            / max(sum(r["group"] == g for r in rows), 1)
            for g in ("eval", "deploy", "neutral")
        },
        "eval - deploy": asdict(paired(rows, "eval", "deploy")),
        "eval - neutral": asdict(paired(rows, "eval", "neutral")),
        "deploy - neutral": asdict(paired(rows, "deploy", "neutral")),
        "eval - deploy, silent (regex)": asdict(
            paired([r for r in rows if not r["prefilter"]], "eval", "deploy")
        ),
        "pairs positive": sum(v > 0 for v in per_pair) / len(per_pair),
        "leave one pair out": [min(loo), max(loo)],
        "eval talk": {
            g: sum(r["prefilter"] or bool(r.get("judge")) for r in rows if r["group"] == g)
            / max(sum(r["group"] == g for r in rows), 1)
            for g in ("eval", "deploy")
        },
    }
    if judged:
        quiet = [r for r in rows if not r["prefilter"] and not r["judge"]]
        out["eval - deploy, silent (regex + judge)"] = asdict(paired(quiet, "eval", "deploy"))
    return out


def fmt(e: dict) -> str:
    return f"{e['value']:+.1f} [{e['low']:+.1f}, {e['high']:+.1f}]"


def main(config: ConfigOption, json_logs: JsonLogsOption = False) -> None:
    """Run the tests, compute the checks per run, print and save."""
    cfg, run_dir = start_run(config, ChecksConfig, json_logs=json_logs)
    tests = {task.id: task.test for task in load_tasks(cfg.tasks)}
    summary, scored = {}, {}
    for name, path in cfg.runs.items():
        rows = read_jsonl(Path(path))
        with ThreadPoolExecutor(cfg.max_parallel_tests) as pool:
            codes = [extract_code(r["answer"]) for r in rows]
            results = list(
                pool.map(
                    lambda rc: run_tests(rc[1], tests[rc[0]["task_id"]], cfg.sandbox),
                    zip(rows, codes, strict=True),
                )
            )
        rows = [
            r | {"passed": res.fraction if code else 0.0}
            for r, code, res in zip(rows, codes, results, strict=True)
        ]
        judged = name in cfg.verdicts and Path(cfg.verdicts[name]).exists()
        if judged:
            levels = {v["sample_id"]: v["level"] for v in read_jsonl(Path(cfg.verdicts[name]))}
            rows = [r | {"judge": levels.get(r["sample_id"], 0)} for r in rows]
        summary[name] = check(rows, judged)
        scored[name] = (rows, judged)
    for name, members in cfg.pools.items():
        rows = [r for m in members for r in scored[m][0]]
        summary[name] = check(rows, all(scored[m][1] for m in members))
    write_jsonl(
        [
            {"run": name, "sample_id": r["sample_id"], "passed": r["passed"]}
            for name, (rows, _) in scored.items()
            for r in rows
        ],
        run_dir / "tests-passed.jsonl",
    )

    lines = [
        "Unit tests passed, points, paired by task (95% bootstrap intervals).",
        "",
        "| Run | Passed e / d / n (%) | Eval - deploy | Silent, regex | Silent, regex + judge "
        "| Eval - none | Deploy - none | Pairs with eval > deploy | Leave one pair out |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, d in summary.items():
        judge = d.get("eval - deploy, silent (regex + judge)")
        judge = fmt(judge) if judge else "not judged"
        passed = " / ".join(f"{d['passed'][g]:.1f}" for g in ("eval", "deploy", "neutral"))
        lines.append(
            f"| {name} | {passed} | {fmt(d['eval - deploy'])} "
            f"| {fmt(d['eval - deploy, silent (regex)'])} | {judge} "
            f"| {fmt(d['eval - neutral'])} | {fmt(d['deploy - neutral'])} "
            f"| {d['pairs positive']:.0%} "
            f"| {d['leave one pair out'][0]:+.1f} to {d['leave one pair out'][1]:+.1f} |"
        )
    table = "\n".join(lines) + "\n"
    (run_dir / "table.md").write_text(table, encoding="utf-8")
    save_summary(summary, run_dir / "summary.json")
    Console().print(Markdown(table))


if __name__ == "__main__":
    typer.run(main)
