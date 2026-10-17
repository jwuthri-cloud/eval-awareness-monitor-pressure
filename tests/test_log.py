"""Tests for the structlog setup in evalgaming.log."""

from __future__ import annotations

import json

import structlog

from evalgaming.log import configure, get_logger


def test_json_logs_emit_one_parseable_event_per_line(capsys):
    structlog.reset_defaults()
    configure("INFO", json_logs=True)
    get_logger("tests").info("sampled", n=4, label="eval")

    event = json.loads(capsys.readouterr().err.strip())
    assert event["event"] == "sampled"
    assert (event["n"], event["label"], event["level"]) == (4, "eval", "info")


def test_level_filters_lower_events(capsys):
    structlog.reset_defaults()
    configure("WARNING", json_logs=True)
    get_logger("tests").info("hidden")

    assert capsys.readouterr().err == ""
