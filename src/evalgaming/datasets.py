"""Load and validate the data sets described in ``data/SPEC.md``.

Every check returns a list of :class:`Issue`; an empty list means the file meets the spec.
Standard library only, so validation runs anywhere.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

CONTEXT_LABELS: Mapping[str, str] = {
    "eval_explicit": "eval",
    "eval_implicit": "eval",
    "wood": "eval",
    "deploy_explicit": "deploy",
    "deploy_implicit": "deploy",
    "neutral": "neutral",
}

STYLES: Mapping[str, Mapping[str, frozenset[str]]] = {
    "train": {
        "eval_explicit": frozenset({"lab_system_prompt", "benchmark_intro"}),
        "eval_implicit": frozenset({"benchmark_header", "grader_style_format"}),
        "wood": frozenset({"wood_api_system", "wood_user_mention"}),
        "deploy_explicit": frozenset({"product_system_prompt", "live_user_notice"}),
        "deploy_implicit": frozenset({"casual_user", "workplace_user"}),
        "neutral": frozenset({"plain"}),
    },
    "heldout": {
        "eval_explicit": frozenset({"research_study_consent", "certification_check"}),
        "eval_implicit": frozenset({"harness_metadata", "ci_pipeline_log"}),
        "wood": frozenset({"wood_contract_footer", "wood_partner_portal"}),
        "deploy_explicit": frozenset({"release_notes_banner", "support_desk_system"}),
        "deploy_implicit": frozenset({"ide_plugin_context", "pasted_chat_history"}),
        "neutral": frozenset({"minimal"}),
    },
}

# Contexts whose text must not say "evaluation" in any form (SPEC, ground rules).
NO_EVAL_WORD_CONTEXTS = frozenset(
    {"eval_implicit", "wood", "deploy_explicit", "deploy_implicit", "neutral"}
)

EVAL_WORDS = re.compile(
    r"\b(evaluat\w*|tests?|tested|testing|benchmark\w*|assess\w*|grader?s?|graded|grading"
    r"|exams?|examination\w*|audit\w*|red[- ]?team\w*|scoring rubric)\b",
    re.IGNORECASE,
)
MODEL_NAMES = re.compile(
    r"\b(qwen|gemma|llama|nemotron|mistral|deepseek|claude|gpt|chatgpt|gemini)\b",
    re.IGNORECASE,
)

LEVELS_BY_CATEGORY: Mapping[str, frozenset[int]] = {
    "explicit": frozenset({2}),
    "indirect": frozenset({1}),
    "obfuscated": frozenset({1, 2}),
    "none": frozenset({0}),
    "hard_negative": frozenset({0}),
}

LANGUAGE_CATEGORIES = frozenset({"european_languages", "asian_languages", "other_languages"})

TASK_SLOT = "{task}"
NEAR_DUPLICATE_THRESHOLD = 0.6


@dataclass(frozen=True)
class Issue:
    """One way a row or file breaks the spec."""

    file: str
    row_id: str
    message: str

    def __str__(self) -> str:
        return f"{self.file} [{self.row_id}] {self.message}"


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _missing_fields(row: Mapping, fields: Iterable[str]) -> list[str]:
    return [field for field in fields if field not in row]


def _duplicate_ids(rows: list[dict], file: str) -> list[Issue]:
    counts = Counter(row.get("id") for row in rows)
    return [Issue(file, str(i), "duplicate id") for i, n in counts.items() if n > 1]


def _shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    # A text shorter than n words is one shingle, so short and empty texts still compare.
    return {tuple(words[i : i + n]) for i in range(max(len(words) - n + 1, 1))}


def near_duplicates(
    texts: Mapping[str, str], threshold: float = NEAR_DUPLICATE_THRESHOLD
) -> list[tuple[str, str, float]]:
    """Return id pairs whose word 3-gram Jaccard similarity exceeds ``threshold``."""
    # ponytail: O(n^2) pairwise scan; fine for a few thousand rows, use MinHash beyond that.
    shingles = {key: _shingles(text) for key, text in texts.items()}
    pairs = []
    for a, b in combinations(shingles, 2):
        union = shingles[a] | shingles[b]
        if not union:
            continue
        score = len(shingles[a] & shingles[b]) / len(union)
        if score > threshold:
            pairs.append((a, b, score))
    return pairs


def framing_text(row: Mapping) -> str:
    """All text a framing shows the model, minus the task slot."""
    return f"{row.get('system') or ''}\n{row.get('user_template', '')}".replace(TASK_SLOT, "")


def check_framings(rows: list[dict], split: str, file: str = "framings") -> list[Issue]:
    """Check framings against SPEC section 1."""
    issues = _duplicate_ids(rows, file)
    fields = ("id", "split", "context", "label", "style", "system", "user_template", "notes")
    for row in rows:
        row_id = str(row.get("id"))
        if missing := _missing_fields(row, fields):
            issues.append(Issue(file, row_id, f"missing fields {missing}"))
            continue
        context = row["context"]
        if row["split"] != split:
            issues.append(Issue(file, row_id, f"split is {row['split']!r}, expected {split!r}"))
        if context not in CONTEXT_LABELS:
            issues.append(Issue(file, row_id, f"unknown context {context!r}"))
            continue
        if row["label"] != CONTEXT_LABELS[context]:
            issues.append(Issue(file, row_id, f"label {row['label']!r} does not match context"))
        if row["style"] not in STYLES[split][context]:
            issues.append(Issue(file, row_id, f"style {row['style']!r} not allowed for {split}"))
        if row["user_template"].count(TASK_SLOT) != 1:
            issues.append(Issue(file, row_id, "user_template must contain {task} exactly once"))
        text = framing_text(row)
        if context in NO_EVAL_WORD_CONTEXTS and (match := EVAL_WORDS.search(text)):
            issues.append(Issue(file, row_id, f"eval word {match.group(0)!r} in {context}"))
        if context == "wood" and "wood labs" not in text.lower():
            issues.append(Issue(file, row_id, "wood framing does not mention Wood Labs"))
        if match := MODEL_NAMES.search(text):
            issues.append(Issue(file, row_id, f"model name {match.group(0)!r}"))
    texts = {str(row.get("id")): framing_text(row) for row in rows}
    issues += [
        Issue(file, f"{a}~{b}", f"near-duplicate framings (jaccard {s:.2f})")
        for a, b, s in near_duplicates(texts)
    ]
    return issues


def check_labelled_traces(rows: list[dict], file: str = "labels") -> list[Issue]:
    """Check hand-labelled reasoning traces against SPEC section 2."""
    issues = _duplicate_ids(rows, file)
    fields = (
        "id",
        "source_run",
        "sample_id",
        "task_snippet",
        "trace",
        "level",
        "category",
        "evidence",
        "labeller",
    )
    for row in rows:
        row_id = str(row.get("id"))
        if missing := _missing_fields(row, fields):
            issues.append(Issue(file, row_id, f"missing fields {missing}"))
            continue
        allowed = LEVELS_BY_CATEGORY.get(row["category"])
        if allowed is None:
            issues.append(Issue(file, row_id, f"unknown category {row['category']!r}"))
        elif row["level"] not in allowed:
            issues.append(Issue(file, row_id, f"level {row['level']} invalid for category"))
        evidence = row["evidence"]
        if row["level"] == 0 and evidence:
            issues.append(Issue(file, row_id, "level 0 must have empty evidence"))
        if row["level"] > 0 and (not evidence or evidence not in row["trace"]):
            issues.append(Issue(file, row_id, "evidence must be an exact substring of trace"))
    texts = {str(row.get("id")): row.get("trace", "") for row in rows}
    issues += [
        Issue(file, f"{a}~{b}", f"near-duplicate traces (jaccard {s:.2f})")
        for a, b, s in near_duplicates(texts)
    ]
    return issues


def check_probe_contrasts(rows: list[dict], file: str = "contrast_heldout") -> list[Issue]:
    """Check held-out probe contrast pairs against SPEC section 3."""
    issues = _duplicate_ids(rows, file)
    fields = ("id", "pair_id", "label", "style", "domain", "system", "user", "cue_spans")
    labels_by_pair: dict[str, list[str]] = {}
    for row in rows:
        row_id = str(row.get("id"))
        if missing := _missing_fields(row, fields):
            issues.append(Issue(file, row_id, f"missing fields {missing}"))
            continue
        labels_by_pair.setdefault(row["pair_id"], []).append(row["label"])
        if row["id"] != f"{row['pair_id']}-{row['label']}":
            issues.append(Issue(file, row_id, "id must be <pair_id>-<label>"))
        text = f"{row['system'] or ''}\n{row['user']}"
        if not row["cue_spans"]:
            issues.append(Issue(file, row_id, "no cue_spans"))
        for span in row["cue_spans"]:
            if span not in text:
                issues.append(Issue(file, row_id, f"cue span not found: {span[:40]!r}"))
        if row["label"] == "deploy" and (match := EVAL_WORDS.search(text)):
            issues.append(Issue(file, row_id, f"eval word {match.group(0)!r} in deploy row"))
        if match := MODEL_NAMES.search(text):
            issues.append(Issue(file, row_id, f"model name {match.group(0)!r}"))
    for pair_id, labels in labels_by_pair.items():
        if sorted(labels) != ["deploy", "eval"]:
            issues.append(Issue(file, pair_id, f"pair needs one eval and one deploy, has {labels}"))
    return issues


def check_steering_pairs(rows: list[dict], file: str = "contrastive_pairs") -> list[Issue]:
    """Check steering contrast pairs against SPEC section 4."""
    issues = _duplicate_ids(rows, file)
    for row in rows:
        row_id = str(row.get("id"))
        if missing := _missing_fields(row, ("id", "set", "deploy", "eval")):
            issues.append(Issue(file, row_id, f"missing fields {missing}"))
            continue
        if row["set"] not in {"hua_style", "paraphrase_heldout"}:
            issues.append(Issue(file, row_id, f"unknown set {row['set']!r}"))
        for side in ("deploy", "eval"):
            if not row[side].rstrip().endswith(":"):
                issues.append(Issue(file, row_id, f"{side} prompt must end with ':'"))
        if row["deploy"] == row["eval"]:
            issues.append(Issue(file, row_id, "deploy and eval prompts are identical"))
    return issues


def check_concepts(
    rows: list[dict], neutral_words: list[str], file: str = "concepts"
) -> list[Issue]:
    """Check steering-detection concepts and neutral words against SPEC section 5."""
    issues = []
    seen: Counter[str] = Counter(str(row.get("concept", "")).lower() for row in rows)
    issues += [Issue(file, c, "duplicate concept") for c, n in seen.items() if n > 1]
    for row in rows:
        concept = str(row.get("concept"))
        fields = ("concept", "category", "excluded", "reason", "gloss")
        if missing := _missing_fields(row, fields):
            issues.append(Issue(file, concept, f"missing fields {missing}"))
            continue
        if row["excluded"] and not row["reason"]:
            issues.append(Issue(file, concept, "excluded concept needs a reason"))
        if row["category"] in LANGUAGE_CATEGORIES and not row["gloss"]:
            issues.append(Issue(file, concept, "foreign-language concept needs an English gloss"))
    words = [w.strip().lower() for w in neutral_words if w.strip()]
    issues += [
        Issue("neutral_words", w, "duplicate word") for w, n in Counter(words).items() if n > 1
    ]
    issues += [Issue("neutral_words", w, "also a concept") for w in set(words) & set(seen)]
    return issues


def check_counts(
    rows: list[dict], field: str, expected: Mapping[str, int], file: str
) -> list[Issue]:
    """Compare per-value counts of ``field`` with the spec's targets."""
    actual = Counter(str(row.get(field)) for row in rows)
    return [
        Issue(file, value, f"{field}={value}: {actual.get(value, 0)} rows, spec says {n}")
        for value, n in expected.items()
        if actual.get(value, 0) != n
    ]
