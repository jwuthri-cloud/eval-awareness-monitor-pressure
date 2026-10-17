"""Every pilot config loads into its script's config class, so typos fail here, not on a pod."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

from evalgaming.runs import load_config

ROOT = Path(__file__).parents[1]
SCRIPTS = sorted((ROOT / "experiments" / "pilot").glob("*.py"))
CONFIGS = sorted((ROOT / "configs" / "pilot").glob("*.toml"))


def run_config_class(script: Path) -> type:
    """The dataclass the script passes to ``start_run``."""
    name = f"pilot_{script.stem}"
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look up their module while building the class
    spec.loader.exec_module(module)
    return getattr(module, re.search(r"start_run\(\w+, (\w+)", script.read_text()).group(1))


@pytest.mark.parametrize("config", CONFIGS, ids=lambda path: path.name)
def test_pilot_config_loads(config: Path):
    script = max((s for s in SCRIPTS if config.stem.startswith(s.stem)), key=lambda s: len(s.stem))
    load_config(config, run_config_class(script))
