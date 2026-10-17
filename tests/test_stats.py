"""Tests for confidence intervals and detection metrics."""

from __future__ import annotations

import math

import pytest

from evalgaming.stats import difference, mean, precision_recall, proportion


def test_proportion_has_a_wilson_interval():
    estimate = proportion(5, 10)
    assert estimate.value == 0.5
    assert estimate.low < 0.5 < estimate.high
    assert estimate.n == 10


def test_proportion_at_the_edges_stays_in_bounds():
    none, all_ = proportion(0, 20), proportion(20, 20)
    assert none.low == pytest.approx(0.0, abs=1e-12)
    assert 0 < none.high < 0.2
    assert all_.high == pytest.approx(1.0)
    assert 0.8 < all_.low < 1


def test_proportion_of_nothing_is_nan():
    estimate = proportion(0, 0)
    assert math.isnan(estimate.value)
    assert estimate.n == 0


def test_mean_interval_contains_the_mean():
    estimate = mean([0, 100, 50, 50, 100, 0, 25, 75])
    assert estimate.value == 50
    assert estimate.low < 50 < estimate.high


def test_mean_of_one_value_has_no_interval():
    estimate = mean([7.0])
    assert estimate.value == 7.0
    assert math.isnan(estimate.low)


def test_mean_is_reproducible():
    values = [0, 100, 30, 70, 50]
    assert mean(values) == mean(values)


def test_difference_detects_a_clear_gap():
    gap = difference([80, 90, 100, 70, 85, 95], [10, 0, 20, 5, 15, 0])
    assert gap.value == pytest.approx(78.33, abs=0.01)
    assert gap.low > 50
    assert gap.n == 12


def test_difference_needs_two_per_side():
    assert math.isnan(difference([1.0], [1.0, 2.0]).low)


def test_precision_recall():
    truth = [True, True, True, False, False]
    predicted = [True, True, False, True, False]
    precision, recall = precision_recall(truth, predicted)
    assert precision.value == pytest.approx(2 / 3)
    assert recall.value == pytest.approx(2 / 3)
    assert (precision.n, recall.n) == (3, 3)


def test_precision_without_positive_predictions_is_nan():
    precision, recall = precision_recall([True, False], [False, False])
    assert math.isnan(precision.value)
    assert recall.value == 0.0
