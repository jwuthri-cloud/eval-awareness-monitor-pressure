"""Cue audit: how much surface wording each held-out framing style shares with training framings.

Held-out styles exist to test whether results generalize to eval and deployment cues RL never saw.
A style that reuses training words (say, "item 37" numbering) tests near generalization only. Two
views, both on framing text without the task: per style, the accuracy of a bag-of-words
eval-versus-deploy classifier fit on training framings, and the words it shares with training
framings of the same context. Some sharing is the definition of a context ("evaluation" in
eval_explicit, "Wood Labs" in wood, "production" in deploy_explicit); the rest is worth removing.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.linear_model import LogisticRegression

from evalgaming.datasets import framing_text

TOKEN = r"[a-z_]+|\d+"


def cue_words(row: dict) -> set[str]:
    """Distinct non-stopword tokens of a framing, task slot removed."""
    return set(re.findall(TOKEN, framing_text(row).lower())) - ENGLISH_STOP_WORDS


def transfer_accuracy(train: list[dict], heldout: list[dict]) -> dict[str, float]:
    """Per held-out style, accuracy of an eval-vs-deploy word classifier fit on train framings."""
    train = [row for row in train if row["label"] != "neutral"]
    vectorizer = CountVectorizer(
        binary=True, lowercase=True, token_pattern=TOKEN, stop_words=list(ENGLISH_STOP_WORDS)
    )
    features = vectorizer.fit_transform(framing_text(row) for row in train)
    model = LogisticRegression(max_iter=2000).fit(features, [row["label"] for row in train])
    scores = {}
    for style, rows in _group(
        (row for row in heldout if row["label"] != "neutral"), "style"
    ).items():
        predicted = model.predict(vectorizer.transform(framing_text(row) for row in rows))
        scores[style] = sum(
            p == row["label"] for p, row in zip(predicted, rows, strict=True)
        ) / len(rows)
    return scores


def shared_words(
    train: list[dict], heldout: list[dict], min_heldout: float = 0.2, min_train: float = 0.05
) -> dict[str, list[tuple[str, float, float]]]:
    """Per held-out style, words in at least ``min_heldout`` of its framings and ``min_train`` of
    train framings with the same context: (word, held-out share, train share), commonest first.
    """

    def shares(rows: list[dict]) -> dict[str, float]:
        counts = Counter(word for row in rows for word in cue_words(row))
        return {word: n / len(rows) for word, n in counts.items()}

    train_shares = {ctx: shares(rows) for ctx, rows in _group(train, "context").items()}
    result = {}
    for style, rows in _group(heldout, "style").items():
        reference = train_shares.get(rows[0]["context"], {})
        found = [
            (word, share, reference[word])
            for word, share in shares(rows).items()
            if share >= min_heldout and reference.get(word, 0.0) >= min_train
        ]
        result[style] = sorted(found, key=lambda item: -item[1] * item[2])
    return result


def _group(rows: Iterable[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return dict(groups)
