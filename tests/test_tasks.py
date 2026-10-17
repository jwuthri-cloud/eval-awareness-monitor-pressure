"""Tests for task files and the stdlib-only filter."""

from __future__ import annotations

from evalgaming.tasks import Task, is_stdlib_only, load_tasks, save_tasks


def task(libs: tuple[str, ...]) -> Task:
    return Task(
        id="BigCodeBench/1",
        prompt="Count characters.\nYou should write self-contained code starting with:",
        test="import unittest",
        solution="def task_func():\n    pass",
        libs=libs,
    )


def test_round_trip(tmp_path):
    path = tmp_path / "tasks.jsonl"
    save_tasks([task(("random", "string"))], path)
    assert load_tasks(path) == [task(("random", "string"))]


def test_snippet_is_the_first_line():
    assert task(()).snippet == "Count characters."


def test_stdlib_only():
    assert is_stdlib_only(task(("collections", "os.path", "random")), excluded=())
    assert not is_stdlib_only(task(("random", "numpy")), excluded=())
    assert not is_stdlib_only(task(("json", "socket")), excluded=("socket",))
    assert is_stdlib_only(task(()), excluded=("socket",))
