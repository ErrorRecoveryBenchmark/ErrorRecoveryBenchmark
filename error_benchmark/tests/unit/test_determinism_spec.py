#!/usr/bin/env python
"""L2 spec — EVALUATION.md §2 single evaluation per scene (S-A3 / L2-03).

The protocol evaluates each error scene exactly once from its stored seed.
benchmark_v5.yaml declared ``seeds_per_scene: 3`` (a v4 leftover that no eval
script reads); this spec pins it to 1 so the config can never re-introduce a
hidden 3-seed best-of behavior that contradicts "one run per scene".

Config load: no robosuite / GPU / data required.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "error_benchmark" / "configs" / "benchmark_v5.yaml"
TRAINING = REPO_ROOT / "error_benchmark" / "scripts" / "training"

EVAL_SCRIPTS = [
    TRAINING / "eval_pi05_error_scenes.py",
    TRAINING / "eval_pi05_error_scenes_multi.py",
    TRAINING / "eval_clean_multi.py",
    TRAINING / "eval_bc_rnn_error_scenes.py",
]


def _eval_cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["evaluation"]


def test_seeds_per_scene_is_one():
    assert _eval_cfg()["seeds_per_scene"] == 1


def test_eval_scripts_do_not_read_seeds_per_scene():
    """The protocol runs each scene exactly once; no evaluator should consume
    seeds_per_scene to repeat scenes (the config key is a dead v4 leftover)."""
    for script in EVAL_SCRIPTS:
        assert "seeds_per_scene" not in script.read_text(encoding="utf-8"), (
            f"{script.name} reads seeds_per_scene — protocol is one run per scene"
        )
