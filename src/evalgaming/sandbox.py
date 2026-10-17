"""Run model-written code against a task's unit tests in a separate, locked-down Python process.

Defense in depth for a disposable GPU container, not a security boundary. The child runs in
isolated mode (``-I -B``) in a fresh temporary directory with a minimal environment, a
wall-clock timeout and memory and file-size limits. An audit hook, installed before any model
code runs, blocks network access, new processes, native libraries and file writes outside the
directory. Code and tests share one namespace, as BigCodeBench's harness does, and the parent
counts the expected tests itself so a crashed child scores zero.
"""

from __future__ import annotations

import ast
import json
import secrets
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from typing import Literal

_RUNNER = r"""
import io, json, os, resource, sys, types, unittest

job = json.load(sys.stdin)
root = os.path.realpath(os.getcwd())
resource.setrlimit(resource.RLIMIT_FSIZE, (job["file_bytes"], job["file_bytes"]))
if sys.platform == "linux":  # macOS does not enforce address-space limits
    resource.setrlimit(resource.RLIMIT_AS, (job["memory_bytes"], job["memory_bytes"]))

BLOCKED = {
    "socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg", "socket.getaddrinfo",
    "socket.gethostbyname", "socket.gethostbyaddr", "subprocess.Popen", "os.system", "os.exec",
    "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty", "os.kill", "os.killpg",
    "ctypes.dlopen", "webbrowser.open",
}
WRITES = {
    "os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.truncate",
    "os.symlink", "os.link", "os.utime", "shutil.rmtree", "shutil.move", "shutil.copyfile",
}
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


def outside(path):
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False
    real = os.path.realpath(os.fsdecode(path))
    return real != root and not real.startswith(root + os.sep)


def guard(event, args):
    if event in BLOCKED:
        raise PermissionError(f"sandbox blocks {event}")
    writing = event in WRITES or (event == "open" and (args[2] or 0) & WRITE_FLAGS)
    if writing and any(outside(arg) for arg in args[:2]):
        raise PermissionError(f"sandbox blocks {event} outside its directory")


sys.stdout = sys.stderr = open(os.devnull, "w")  # only the result line reaches the parent
sys.addaudithook(guard)
passed = 0
try:
    # A registered module, as in BigCodeBench's harness, so tests can patch f"{__name__}.x".
    module = sys.modules["__test__"] = types.ModuleType("__test__")
    exec(compile(job["code"], "<solution>", "exec"), module.__dict__)
    exec(compile(job["test"], "<tests>", "exec"), module.__dict__)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.TestCases)
    result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    failed = {
        getattr(test, "test_case", test).id()
        for test, _ in result.failures + result.errors + result.skipped
    }
    passed = result.testsRun - len(failed)
except BaseException:
    pass
os.write(1, f"\n{job['nonce']} {passed}\n".encode())
"""


@dataclass(frozen=True)
class SandboxConfig:
    """Resource limits for one test run."""

    timeout_s: float
    memory_mb: int
    max_file_mb: int


@dataclass(frozen=True)
class UnitTestResult:
    """How many of a task's tests passed, and how the run ended."""

    passed: int
    total: int
    status: Literal["ok", "no_code", "timeout", "crashed"]

    @property
    def fraction(self) -> float:
        """Share of the task's tests that passed; 0 when the task has none."""
        return self.passed / self.total if self.total else 0.0


def count_tests(test_src: str) -> int:
    """Number of ``test*`` methods on the ``TestCases`` class (the BigCodeBench convention)."""
    with warnings.catch_warnings(action="ignore", category=SyntaxWarning):  # invalid escapes
        tree = ast.parse(test_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TestCases":
            return sum(
                isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
                and item.name.startswith("test")
                for item in node.body
            )
    return 0


def run_tests(code: str | None, test_src: str, config: SandboxConfig) -> UnitTestResult:
    """Run ``code`` against the ``TestCases`` in ``test_src`` in a sandboxed subprocess."""
    total = count_tests(test_src)
    if code is None:
        return UnitTestResult(0, total, "no_code")
    nonce = secrets.token_hex(16)
    job = {
        "code": code,
        "test": test_src,
        "nonce": nonce,
        "memory_bytes": config.memory_mb << 20,
        "file_bytes": config.max_file_mb << 20,
    }
    with tempfile.TemporaryDirectory() as workdir:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-B", "-c", _RUNNER],
                input=json.dumps(job),
                capture_output=True,
                text=True,
                errors="replace",
                cwd=workdir,
                env={"HOME": workdir, "TMPDIR": workdir, "LANG": "C.UTF-8"},
                timeout=config.timeout_s,
            )
        except subprocess.TimeoutExpired:
            return UnitTestResult(0, total, "timeout")
    for line in reversed(completed.stdout.splitlines()):
        marker, _, passed = line.partition(" ")
        if marker == nonce:
            return UnitTestResult(min(max(int(passed), 0), total), total, "ok")
    return UnitTestResult(0, total, "crashed")
