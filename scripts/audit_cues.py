"""Report how much surface wording each held-out framing style shares with the training framings.

Usage: uv run python scripts/audit_cues.py [--data-dir data]
A report, not a pass/fail check; see src/evalgaming/cues.py for how to read it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from evalgaming.cues import shared_words, transfer_accuracy
from evalgaming.datasets import read_jsonl


def main(data_dir: Annotated[Path, typer.Option(help="Data folder.")] = Path("data")) -> None:
    """Print, per held-out style, word-classifier transfer accuracy and words shared with train."""
    train = read_jsonl(data_dir / "framings" / "train.jsonl")
    heldout = read_jsonl(data_dir / "framings" / "heldout.jsonl")
    accuracy = transfer_accuracy(train, heldout)
    contexts = {row["style"]: row["context"] for row in heldout}
    table = Table(title="Held-out cue audit (shares: held-out style / train, same context)")
    for column in ("Style", "Context", "Transfer accuracy", "Shared words"):
        table.add_column(column)
    for style, words in sorted(shared_words(train, heldout).items(), key=lambda s: contexts[s[0]]):
        shown = ", ".join(f"{word} {h:.0%}/{t:.0%}" for word, h, t in words[:8]) or "-"
        score = f"{accuracy[style]:.2f}" if style in accuracy else "-"
        table.add_row(style, contexts[style], score, shown)
    Console().print(table)


if __name__ == "__main__":
    typer.run(main)
