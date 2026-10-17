"""Tests for loading TOML configs into dataclasses and writing run folders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from evalgaming.datasets import read_jsonl
from evalgaming.runs import append_jsonl, load_config, start_run, write_jsonl


@dataclass(frozen=True)
class Inner:
    path: Path
    sizes: tuple[int, ...]


@dataclass(frozen=True)
class Outer:
    output_dir: Path
    name: str
    optional: Path | None
    inner: Inner
    items: list[Inner]
    passthrough: dict[str, Any]


CONFIG = """
output_dir = "{out}"
name = "run"
optional = "some/file.jsonl"

[inner]
path = "a/b.jsonl"
sizes = [1, 2]

[[items]]
path = "c.jsonl"
sizes = []

[passthrough]
learning_rate = 1e-5
nested = {{ dtype = "bfloat16" }}
"""


def write_config(tmp_path: Path, text: str = CONFIG) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text.format(out=tmp_path / "out"))
    return path


def test_load_config_builds_nested_dataclasses(tmp_path):
    config = load_config(write_config(tmp_path), Outer)
    assert config.optional == Path("some/file.jsonl")
    assert config.inner == Inner(Path("a/b.jsonl"), (1, 2))
    assert config.items == [Inner(Path("c.jsonl"), ())]
    assert config.passthrough == {"learning_rate": 1e-5, "nested": {"dtype": "bfloat16"}}


def test_unknown_keys_are_an_error(tmp_path):
    path = write_config(tmp_path, 'typo = "x"\n' + CONFIG)
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(path, Outer)


def test_missing_keys_are_an_error(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('name = "run"\n')
    with pytest.raises(TypeError):
        load_config(path, Outer)


def test_start_run_creates_the_folder_with_the_config(tmp_path):
    path = write_config(tmp_path)
    config, run_dir = start_run(path, Outer, json_logs=True)
    assert run_dir == tmp_path / "out"
    assert (run_dir / "config.toml").read_text() == path.read_text()
    assert config.name == "run"


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl([{"a": 1}, {"b": "é"}], path)
    append_jsonl({"c": None}, path)
    assert read_jsonl(path) == [{"a": 1}, {"b": "é"}, {"c": None}]
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[1]) == {"b": "é"}
