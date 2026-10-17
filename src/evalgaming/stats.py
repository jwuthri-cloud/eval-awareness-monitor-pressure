"""Point estimates with 95% confidence intervals: Wilson for proportions, bootstrap for means."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

CONFIDENCE = 0.95
N_RESAMPLES = 10_000


@dataclass(frozen=True)
class Estimate:
    """A value with its confidence interval and sample size; NaN where there is too little data."""

    value: float
    low: float
    high: float
    n: int


def proportion(successes: int, n: int) -> Estimate:
    """Share of successes with a Wilson score interval."""
    if n == 0:
        return Estimate(math.nan, math.nan, math.nan, 0)
    interval = stats.binomtest(successes, n).proportion_ci(CONFIDENCE, method="wilson")
    return Estimate(successes / n, interval.low, interval.high, n)


def mean(values: Sequence[float], seed: int = 0) -> Estimate:
    """Mean with a percentile bootstrap interval."""
    data = np.asarray(values, dtype=float)
    value = float(data.mean()) if data.size else math.nan
    return Estimate(value, *_bootstrap((data,), np.mean, seed), int(data.size))


def difference(a: Sequence[float], b: Sequence[float], seed: int = 0) -> Estimate:
    """``mean(a) - mean(b)`` for independent samples, with a percentile bootstrap interval."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    value = float(x.mean() - y.mean()) if x.size and y.size else math.nan
    interval = _bootstrap((x, y), _mean_difference, seed)
    return Estimate(value, *interval, int(x.size + y.size))


def _mean_difference(x: np.ndarray, y: np.ndarray, axis: int) -> np.ndarray:
    return x.mean(axis=axis) - y.mean(axis=axis)


def _bootstrap(
    samples: tuple[np.ndarray, ...], statistic: Callable[..., np.ndarray], seed: int
) -> tuple[float, float]:
    if min(sample.size for sample in samples) < 2:
        return math.nan, math.nan
    interval = stats.bootstrap(
        samples,
        statistic,
        confidence_level=CONFIDENCE,
        n_resamples=N_RESAMPLES,
        method="percentile",
        rng=np.random.default_rng(seed),
    ).confidence_interval
    return float(interval.low), float(interval.high)


def precision_recall(truth: Sequence[bool], predicted: Sequence[bool]) -> tuple[Estimate, Estimate]:
    """Precision and recall of ``predicted`` against ``truth``, each with a Wilson interval."""
    hits = sum(t and p for t, p in zip(truth, predicted, strict=True))
    return proportion(hits, sum(predicted)), proportion(hits, sum(truth))
