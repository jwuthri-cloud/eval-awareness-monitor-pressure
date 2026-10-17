"""Validate every data set in data/ against data/SPEC.md.

Usage: uv run python scripts/validate_data.py [--data-dir data] [--allow-missing]
Exits non-zero if any file breaks the spec.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from evalgaming.datasets import (
    CONTEXT_LABELS,
    Issue,
    check_concepts,
    check_counts,
    check_framings,
    check_labelled_traces,
    check_probe_contrasts,
    check_steering_pairs,
    framing_text,
    near_duplicates,
    read_jsonl,
)

FRAMING_COUNTS = {
    "train": dict.fromkeys(CONTEXT_LABELS, 100) | {"neutral": 50},
    "heldout": dict.fromkeys(CONTEXT_LABELS, 25),
}
PROBE_COUNTS = {"eval": 150, "deploy": 150}
STEERING_COUNTS = {"hua_style": 16, "paraphrase_heldout": 48}
N_CONCEPTS, N_CATEGORIES, N_NEUTRAL_WORDS = 500, 21, 152


def _framings(data: Path, split: str) -> list[Issue]:
    rows = read_jsonl(data / "framings" / f"{split}.jsonl")
    file = f"framings/{split}"
    return check_framings(rows, split, file) + check_counts(
        rows, "context", FRAMING_COUNTS[split], file
    )


def _split_leakage(data: Path) -> list[Issue]:
    texts = {
        f"{split}:{row['id']}": framing_text(row)
        for split in ("train", "heldout")
        for row in read_jsonl(data / "framings" / f"{split}.jsonl")
    }
    return [
        Issue("framings", f"{a}~{b}", f"train and held-out framings overlap (jaccard {s:.2f})")
        for a, b, s in near_duplicates(texts)
        if a.split(":")[0] != b.split(":")[0]
    ]


def _labels(data: Path) -> list[Issue]:
    return check_labelled_traces(read_jsonl(data / "monitor" / "labels.jsonl"))


def _probes(data: Path) -> list[Issue]:
    rows = read_jsonl(data / "probes" / "contrast_heldout.jsonl")
    return check_probe_contrasts(rows) + check_counts(
        rows, "label", PROBE_COUNTS, "contrast_heldout"
    )


def _steering(data: Path) -> list[Issue]:
    rows = read_jsonl(data / "steering" / "contrastive_pairs.jsonl")
    return check_steering_pairs(rows) + check_counts(
        rows, "set", STEERING_COUNTS, "contrastive_pairs"
    )


def _concepts(data: Path) -> list[Issue]:
    rows = read_jsonl(data / "steering" / "concepts.jsonl")
    words = (data / "steering" / "neutral_words.txt").read_text(encoding="utf-8").splitlines()
    issues = check_concepts(rows, words)
    n_categories = len({row.get("category") for row in rows})
    n_words = len([w for w in words if w.strip()])
    if (len(rows), n_categories, n_words) != (N_CONCEPTS, N_CATEGORIES, N_NEUTRAL_WORDS):
        issues.append(
            Issue(
                "concepts",
                "-",
                f"{len(rows)} concepts in {n_categories} categories and {n_words} neutral words; "
                f"spec says {N_CONCEPTS}, {N_CATEGORIES}, {N_NEUTRAL_WORDS}",
            )
        )
    return issues


CHECKS: dict[str, Callable[[Path], list[Issue]]] = {
    "framings/train": lambda d: _framings(d, "train"),
    "framings/heldout": lambda d: _framings(d, "heldout"),
    "framings leakage": _split_leakage,
    "monitor labels": _labels,
    "probe contrasts": _probes,
    "steering pairs": _steering,
    "steering concepts": _concepts,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--allow-missing", action="store_true", help="skip files not yet written")
    args = parser.parse_args()

    summary = Table(title="Data validation", title_justify="left")
    summary.add_column("Check")
    summary.add_column("Status")
    summary.add_column("Issues", justify="right")
    failures: dict[str, list[Issue]] = {}
    for name, check in CHECKS.items():
        try:
            issues = check(args.data_dir)
        except FileNotFoundError as error:
            if args.allow_missing:
                missing = escape(Path(error.filename).name)
                summary.add_row(name, f"[yellow]skipped, {missing} missing[/]", "-")
                continue
            issues = [Issue(name, "-", f"file not found: {error.filename}")]
        summary.add_row(name, "[red]fail[/]" if issues else "[green]ok[/]", str(len(issues)))
        if issues:
            failures[name] = issues

    console = Console()
    console.print(summary)
    for name, issues in failures.items():
        details = Table(title=name, title_justify="left")
        details.add_column("Row")
        details.add_column("Problem")
        for issue in issues:
            details.add_row(escape(issue.row_id), escape(issue.message))
        console.print(details)
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
