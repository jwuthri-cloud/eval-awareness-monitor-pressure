"""Linear probes for evaluation versus deployment context, read from the residual stream.

Following Montoya et al., a probe is fit on non-coding contrastive prompts
(``contrastive_prompts_v2``), then frozen and scored on coding transcripts, one score per segment:
the prompt, the reasoning and the answer. A high prompt AUROC only says the prompt's cue is
represented. Reasoning and answer AUROC say whether the context is still represented while the
model writes, which is Arm A's question once RL has removed the words. Activations are
mean-pooled per segment and layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

CS = (1e-3, 1e-2, 1e-1, 1.0)  # inverse regularization strengths tried for every layer
# Inputs are right-padded to a multiple of this many tokens. Qwen3.5's kernels compile once per
# sequence length (0.8 s each, ten times the forward pass), so this bounds them to a few lengths.
BUCKET = 256


def segment_means(hidden: Any, bounds: Sequence[tuple[int, int]]) -> np.ndarray:
    """``hidden`` [layers, tokens, d] averaged over each [start, end) span: [segments, layers, d].

    ``hidden`` is a torch tensor on any device. Averaging happens there and only the means are
    copied back, which on the GPU is ten times faster than copying every token. An empty span (a
    segment cut off by truncation, or a trace with no answer) is NaN.
    """
    import torch

    shape = (len(bounds), hidden.shape[0], hidden.shape[2])
    out = torch.full(shape, float("nan"), device=hidden.device)
    for i, (start, end) in enumerate(bounds):
        if end > start:
            out[i] = hidden[:, start:end].float().mean(dim=1)
    return out.cpu().numpy()


def read_activations(
    model: Any,
    tokenizer: Any,
    items: Sequence[Sequence[str]],
    layers: Sequence[int],
    max_tokens: int,
) -> np.ndarray:
    """Mean residual stream per text segment and layer for each item: [items, segments, layers, d].

    Each item is a list of text segments that are tokenized separately and concatenated, so segment
    boundaries are exact; ``hidden_states[k]`` is the output of decoder layer k (0 = embeddings).
    """
    import torch

    rows = []
    with torch.no_grad():
        for pieces in items:
            ids: list[int] = []
            bounds = []
            for piece in pieces:
                start = len(ids)
                ids += tokenizer(piece, add_special_tokens=False)["input_ids"]
                bounds.append((min(start, max_tokens), min(len(ids), max_tokens)))
            # The model is causal, so padding after the last real token leaves its activations as
            # they were; the segment bounds never reach the padding.
            ids = ids[:max_tokens]
            ids += [tokenizer.pad_token_id or 0] * (-len(ids) % BUCKET)
            tokens = torch.tensor([ids], device=model.device)
            hidden = model.model(input_ids=tokens, output_hidden_states=True).hidden_states
            rows.append(segment_means(torch.stack([hidden[layer][0] for layer in layers]), bounds))
    return np.stack(rows)


@dataclass(frozen=True)
class Probe:
    """A fitted probe: its layer, regularization, and cross-validated AUROC and log loss."""

    layer_index: int  # position in the ``layers`` list the activations were read at
    c: float
    cv_auroc: float
    model: Pipeline
    cv_log_loss: float

    def score(self, activations: np.ndarray) -> np.ndarray:
        """Decision scores for [items, layers, d] activations; higher means more eval-like."""
        return self.model.decision_function(activations[:, self.layer_index])


def _pipeline(c: float) -> Pipeline:
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))


def fit_layer_probes(
    activations: np.ndarray, labels: np.ndarray, groups: np.ndarray, folds: int = 5
) -> list[Probe]:
    """One probe per layer, with the C of best grouped cross-validated AUROC, refit on all rows.

    ``activations`` is [items, layers, d]; ``groups`` keeps both halves of a contrastive pair in
    the same fold, so a probe cannot score well by memorizing the shared task text. AUROC ties
    (common: it saturates at 1 on easy contrasts) go to the lower cross-validated log loss.
    """
    splitter = GroupKFold(n_splits=folds)
    probes = []
    for layer in range(activations.shape[1]):
        x = activations[:, layer]
        best = None
        for c in CS:
            probability = np.empty(len(labels))
            for train, test in splitter.split(x, labels, groups):
                model = _pipeline(c).fit(x[train], labels[train])
                probability[test] = model.predict_proba(x[test])[:, 1]
            auroc = float(roc_auc_score(labels, probability))
            loss = float(log_loss(labels, probability))
            best = max(best or (auroc, -loss, c), (auroc, -loss, c))
        auroc, neg_loss, c = best
        model = _pipeline(c).fit(x, labels)
        probes.append(Probe(layer, c, auroc, model, -neg_loss))
    return probes


def fit_probe(
    activations: np.ndarray, labels: np.ndarray, groups: np.ndarray, folds: int = 5
) -> Probe:
    """The best layer's probe: highest cross-validated AUROC, ties to the lower log loss."""
    probes = fit_layer_probes(activations, labels, groups, folds)
    return max(probes, key=lambda probe: (probe.cv_auroc, -probe.cv_log_loss))


def auroc_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    clusters: np.ndarray,
    resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """AUROC with a 95% percentile interval from resampling clusters (tasks) with replacement."""
    keep = ~np.isnan(scores)
    labels, scores, clusters = labels[keep], scores[keep], clusters[keep]
    point = float(roc_auc_score(labels, scores))
    ids = np.unique(clusters)
    index = {cluster: np.flatnonzero(clusters == cluster) for cluster in ids}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        rows = np.concatenate([index[c] for c in rng.choice(ids, size=ids.size)])
        if len(np.unique(labels[rows])) == 2:
            draws.append(roc_auc_score(labels[rows], scores[rows]))
    low, high = np.percentile(draws, [2.5, 97.5])
    return point, float(low), float(high)
