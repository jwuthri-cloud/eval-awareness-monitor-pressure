"""Tests for the held-out cue audit in evalgaming.cues."""

from __future__ import annotations

from evalgaming.cues import shared_words, transfer_accuracy


def row(style: str, context: str, label: str, template: str) -> dict:
    return {"style": style, "context": context, "label": label, "system": None,
            "user_template": template}  # fmt: skip


TRAIN = [
    *(row("header", "eval_implicit", "eval", f"Item {n} of 200.\n{{task}}") for n in range(10)),
    *(row("casual", "deploy_implicit", "deploy", f"ugh my boss wants this {n}\n{{task}}")
      for n in range(10)),
]  # fmt: skip
HELDOUT = [
    *[row("near", "eval_implicit", "eval", "[log] item queued\n{task}")] * 5,
    *[row("far", "eval_implicit", "eval", "shard sample_idx\n{task}")] * 5,
]


def test_shared_words_finds_reused_cues_only():
    shared = shared_words(TRAIN, HELDOUT)
    assert [word for word, _, _ in shared["near"]] == ["item"]
    assert shared["far"] == []


def test_transfer_accuracy_is_high_for_styles_that_reuse_training_words():
    accuracy = transfer_accuracy(TRAIN, HELDOUT)
    assert accuracy["near"] == 1.0
    assert set(accuracy) == {"near", "far"}
