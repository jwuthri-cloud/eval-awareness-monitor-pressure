"""Run folders: TOML configs as dataclasses, a copy of the config next to the results, JSONL.

Every experiment script takes ``--config``, builds its config dataclass with :func:`load_config`
and writes into the config's ``output_dir`` through :func:`start_run`, so a run folder holds
everything needed to reproduce it.
"""

from __future__ import annotations

import json
import shutil
import tomllib
import types
from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Annotated, Any, Protocol, TypeVar, Union, get_args, get_origin, get_type_hints

import typer

from evalgaming.log import configure, get_logger

log = get_logger(__name__)

ConfigOption = Annotated[
    Path,
    typer.Option("--config", "-c", exists=True, dir_okay=False, help="TOML run config."),
]
JsonLogsOption = Annotated[
    bool, typer.Option("--json-logs", help="One JSON object per log line, for remote GPU runs.")
]


class RunConfig(Protocol):
    """Any experiment config: it names the folder its run writes to."""

    output_dir: Path


C = TypeVar("C", bound=RunConfig)
T = TypeVar("T")


def load_config(path: Path, cls: type[T]) -> T:
    """Read a TOML file into the dataclass ``cls``, building nested dataclasses from type hints."""
    with path.open("rb") as handle:
        return _build(cls, tomllib.load(handle))


def _build(cls: type[T], data: Mapping[str, Any]) -> T:
    hints = get_type_hints(cls)
    if unknown := set(data) - {field.name for field in fields(cls)}:
        raise ValueError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")
    return cls(**{key: _convert(hints[key], value) for key, value in data.items()})


def _convert(hint: Any, value: Any) -> Any:
    if get_origin(hint) in (Union, types.UnionType):
        if value is None:
            return None
        hint = next(arg for arg in get_args(hint) if arg is not type(None))
    origin = get_origin(hint)
    if is_dataclass(hint):
        return _build(hint, value)
    if hint is Path:
        return Path(value)
    if origin is tuple:
        return tuple(value)
    if origin is list and is_dataclass(item := get_args(hint)[0]):
        return [_build(item, entry) for entry in value]
    return value


def start_run(config_path: Path, cls: type[C], *, json_logs: bool) -> tuple[C, Path]:
    """Configure logging, load the config and create its run folder holding a copy of the config."""
    configure(json_logs=json_logs)
    config = load_config(config_path, cls)
    run_dir = config.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run_dir / "config.toml")
    log.info("run_started", run_dir=str(run_dir), config=str(config_path))
    return config, run_dir


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    """Write rows as JSONL, replacing the file."""
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def append_jsonl(row: Mapping[str, Any], path: Path) -> None:
    """Append one row to a JSONL file, so an interrupted run keeps what it wrote."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
