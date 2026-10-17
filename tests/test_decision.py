"""Tests for the week-7 decision rule in evalgaming.decision."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from evalgaming.decision import (
    NULL_ARM,
    PLAN,
    WORKING_ARM,
    Outcome,
    Prompts,
    bootstrap_weights,
    decide,
    lambda_label,
    seed_label,
    simulate,
)

FAST = replace(PLAN, resamples=300)


def test_bootstrap_weights_multiply_task_and_framing_draws():
    prompts = Prompts(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), np.array(["eval"] * 4))
    weights = bootstrap_weights(prompts, 50, np.random.default_rng(0))
    assert weights.shape == (50, 4)
    assert np.allclose(weights.sum(1), 4)  # two tasks drawn twice, two framings drawn twice


def test_seed_label_puts_words_first():
    fail, ok, unsure = Outcome.FAIL, Outcome.PASS, Outcome.INCONCLUSIVE
    assert seed_label(fail, fail, ok) == "fail_V"
    assert seed_label(ok, fail, fail) == "fail_G"
    assert seed_label(ok, ok, ok) == "pass"
    assert seed_label(ok, unsure, ok) == "inconclusive"


def test_lambda_label_needs_two_of_three():
    assert lambda_label(["pass", "pass", "fail_V"], 2) == "pass"
    assert lambda_label(["pass", "fail_G", "fail_G"], 2) == "fail_G"
    assert lambda_label(["pass", "fail_G", "fail_V"], 2) == "inconclusive"


def test_decision_branches_in_order():
    assert decide({0.5: "fail_V", 2.0: "pass"}, set(), False).lam == 2.0
    assert decide({0.5: "pass", 2.0: "pass"}, set(), False).lam == 0.5  # smallest, not best
    assert decide({0.5: "fail_V", 2.0: "fail_V"}, {2.0}, False).action == "rerun_gamed"
    assert "cannot remove" in decide({0.5: "fail_V", 2.0: "fail_V"}, set(), False).reason
    assert "together" in decide({0.5: "fail_V", 2.0: "fail_G"}, set(), False).reason
    assert decide({0.5: "inconclusive", 2.0: "fail_C"}, set(), False).action == "extend"
    assert decide({0.5: "inconclusive", 2.0: "fail_C"}, set(), True).action == "stop"


def test_rule_stops_a_null_arm_and_continues_a_working_one():
    null = simulate(replace(NULL_ARM, samples_per_prompt=8), FAST, sims=8)
    working = simulate(
        replace(WORKING_ARM, samples_per_prompt=8, labelled_misses=400), FAST, sims=8
    )
    assert null["stop"] == 8
    assert working["continue"] >= 5
