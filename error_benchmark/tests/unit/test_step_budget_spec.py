#!/usr/bin/env python
"""L2 spec — EVALUATION.md §2 step budget (S-A2 / L2-02).

The protocol runs each rollout for at most 500 simulation steps. The two Pi0.5
evaluators and the clean-rollout evaluator currently default ``--max_steps`` to
1000, while benchmark_v5.yaml declares 400. This spec pins every default to 500
so a direct script call is on-protocol without relying on the launch shell.

Source-scan + config load: no robosuite / GPU / data required.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING = REPO_ROOT / "error_benchmark" / "scripts" / "training"
CONFIG = REPO_ROOT / "error_benchmark" / "configs" / "benchmark_v5.yaml"

PROTOCOL_MAX_STEPS = 500

SCRIPTS_WITH_MAX_STEPS = [
    TRAINING / "eval_pi05_error_scenes.py",
    TRAINING / "eval_pi05_error_scenes_multi.py",
    TRAINING / "eval_clean_multi.py",
    TRAINING / "eval_bc_rnn_error_scenes.py",
]


def _max_steps_default(path: Path) -> int:
    # ``[^)]*?`` keeps the match inside the add_argument(...) call (won't cross
    # the closing paren into help text). Assumes a numeric-literal default, which
    # every eval script uses; a missing/non-literal default yields None and fails
    # loudly rather than passing silently.
    src = path.read_text(encoding="utf-8")
    m = re.search(r'--max_steps"[^)]*?\bdefault\s*=\s*(\d+)', src, flags=re.S)
    assert m is not None, f"{path.name} has no numeric --max_steps default"
    return int(m.group(1))


@pytest.mark.parametrize("script", SCRIPTS_WITH_MAX_STEPS, ids=lambda p: p.name)
def test_max_steps_default_is_500(script):
    assert _max_steps_default(script) == PROTOCOL_MAX_STEPS


def test_no_lingering_1000_default():
    for script in SCRIPTS_WITH_MAX_STEPS:
        assert _max_steps_default(script) != 1000, f"{script.name} still defaults to 1000"


def test_config_eval_max_steps_is_500():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["evaluation"]["max_steps"] == PROTOCOL_MAX_STEPS
