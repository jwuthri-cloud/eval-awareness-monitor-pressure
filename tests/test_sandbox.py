"""Tests for running model code against unit tests in the sandbox (real subprocesses)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalgaming.sandbox import SandboxConfig, UnitTestResult, count_tests, run_tests

CONFIG = SandboxConfig(timeout_s=10.0, memory_mb=1024, max_file_mb=16)
TESTS = """
import unittest

class TestCases(unittest.TestCase):
    def test_two(self):
        self.assertEqual(task_func(2), 4)

    def test_range(self):
        for i in range(3):
            with self.subTest(i=i):
                self.assertEqual(task_func(i), 2 * i)

    def test_zero(self):
        self.assertEqual(task_func(0), 0)

    def helper(self):
        pass
"""


def run(body: str, config: SandboxConfig = CONFIG) -> UnitTestResult:
    return run_tests(f"def task_func(x):\n{body}\n", TESTS, config)


def test_count_tests_counts_test_methods_only():
    assert count_tests(TESTS) == 3
    assert count_tests("x = 1") == 0


def test_correct_code_passes_everything():
    assert run("    return 2 * x") == UnitTestResult(3, 3, "ok")
    assert run("    return 2 * x").fraction == 1.0


def test_partial_credit_and_failing_subtests():
    result = run("    return x + 2")  # right only for x == 2
    assert result == UnitTestResult(1, 3, "ok")
    assert result.fraction == pytest.approx(1 / 3)


def test_syntax_error_passes_nothing():
    assert run_tests("def task_func(x)\n    return", TESTS, CONFIG) == UnitTestResult(0, 3, "ok")


def test_missing_code():
    assert run_tests(None, TESTS, CONFIG) == UnitTestResult(0, 3, "no_code")


def test_timeout():
    result = run("    while True:\n        pass", SandboxConfig(1.0, 1024, 16))
    assert result == UnitTestResult(0, 3, "timeout")


def test_tests_can_patch_the_solution_module():
    patched = """
import unittest
from unittest import mock

class TestCases(unittest.TestCase):
    def test_patch(self):
        with mock.patch(f"{__name__}.helper", return_value=5):
            self.assertEqual(task_func(1), 5)
"""
    code = "def helper(x):\n    return x\n\ndef task_func(x):\n    return helper(x)\n"
    assert run_tests(code, patched, CONFIG) == UnitTestResult(1, 1, "ok")


def test_printed_output_cannot_spoof_the_result():
    assert run("    print('0 3')\n    print('ok 3')\n    return 0").passed == 1


def test_network_is_blocked():
    body = "    import socket\n    socket.create_connection(('example.com', 80))\n    return 2 * x"
    assert run(body).passed == 0


def test_processes_are_blocked():
    assert run("    import os\n    os.system('true')\n    return 2 * x").passed == 0


def test_writes_inside_the_sandbox_are_allowed():
    body = "    open('scratch.txt', 'w').write('ok')\n    return 2 * x"
    assert run(body).passed == 3


def test_writes_outside_the_sandbox_are_blocked(tmp_path):
    target = tmp_path / "escaped.txt"
    body = f"    open({str(target)!r}, 'w').write('x')\n    return 2 * x"
    assert run(body).passed == 0
    assert not target.exists()


def test_deleting_outside_the_sandbox_is_blocked(tmp_path):
    target = tmp_path / "keep.txt"
    target.write_text("keep")
    body = f"    import os\n    os.remove({str(target)!r})\n    return 2 * x"
    assert run(body).passed == 0
    assert Path(target).read_text() == "keep"
