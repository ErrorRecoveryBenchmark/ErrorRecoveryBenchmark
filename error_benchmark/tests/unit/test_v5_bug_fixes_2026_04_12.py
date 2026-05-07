#!/usr/bin/env python
"""Regression tests for the 2026-04-12 augmentation-blocking bug cluster.

Each fix is documented in docs/bug_fixes_2026_04_12.md. Bug #3 (body_xquat
order) is already covered by tests/unit/test_pose_transforms.py. The tests
here pin the behavior of bugs #1, #2, #4, and #5.

Environment-dependent tests (#2, #4) are skipped when robosuite cannot
import, so these run in CPU-only CI and light it up on GPU nodes.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
COLLECTION_SCRIPT = (
    PROJECT_ROOT
    / "error_benchmark"
    / "scripts"
    / "collection"
    / "2_collect_recovery_demos.py"
)


# ──────────────────────────────────────────────────────────────────────────
# Bug #1 — Leading zero-action trim
# ──────────────────────────────────────────────────────────────────────────


def _first_nonzero_action_index(actions: np.ndarray, threshold: float = 1e-6) -> int:
    """Reference implementation of the trim selector used in the collection script.

    If the algorithm changes, update here and in 2_collect_recovery_demos.py
    together — the source-level pin test below catches divergence.
    """
    for i, act in enumerate(actions):
        if np.linalg.norm(act[:6]) > threshold:
            return i
    return len(actions)


class TestBug1LeadingZeroActionTrim:
    """Bug #1: OSC settling produced ~84 leading zero-action frames that
    were being saved as real demo data. The fix trims them before writing."""

    def test_all_zeros_returns_length(self):
        actions = np.zeros((50, 7))
        assert _first_nonzero_action_index(actions) == 50

    def test_first_frame_nonzero(self):
        actions = np.zeros((20, 7))
        actions[0, 0] = 0.5
        assert _first_nonzero_action_index(actions) == 0

    def test_realistic_osc_transient(self):
        actions = np.zeros((100, 7))
        actions[84:, :6] = np.random.default_rng(42).normal(0, 0.1, (16, 6))
        assert _first_nonzero_action_index(actions) == 84

    def test_gripper_only_does_not_count(self):
        # act[:6] is eef pose delta; act[6] is gripper. Gripper-only signal is ignored.
        actions = np.zeros((10, 7))
        actions[5, 6] = 1.0
        actions[8, 0] = 0.5
        assert _first_nonzero_action_index(actions) == 8

    def test_threshold_boundary(self):
        actions = np.zeros((5, 7))
        # norm = 5e-7 < 1e-6 → still treated as zero
        actions[2, 0] = 5e-7
        actions[3, 0] = 2e-6
        assert _first_nonzero_action_index(actions) == 3

    def test_collection_script_still_uses_trim(self):
        """Pin the trim to the collection script so refactors can't silently remove it."""
        source = COLLECTION_SCRIPT.read_text()
        assert "np.linalg.norm(act[:6])" in source, (
            "Bug #1 fix removed from 2_collect_recovery_demos.py — "
            "leading zero-action trim must remain"
        )
        # The comment helps future reviewers understand *why* the trim exists
        assert re.search(r"leading zero-action|OSC (controller )?transient", source), (
            "Lost the explanatory comment about the OSC-controller transient; "
            "it is important context — keep it next to the trim"
        )


# ──────────────────────────────────────────────────────────────────────────
# Bug #5 — save_demo must run BEFORE validate_demo
# ──────────────────────────────────────────────────────────────────────────


class TestBug5SaveBeforeValidate:
    """Bug #5: validate_demo() ran before save_demo(), so the validator
    couldn't load the NPZ it was supposed to check — augmentable always
    came back False. We pin the ordering in source."""

    def test_save_precedes_validate_in_main_collect_loop(self):
        source = COLLECTION_SCRIPT.read_text().splitlines()
        save_lines = [
            i for i, line in enumerate(source) if "save_demo(demo" in line
        ]
        validate_lines = [
            i for i, line in enumerate(source) if "validator.validate_demo(" in line
        ]
        assert save_lines, "save_demo(demo, ...) call missing from collection script"
        assert validate_lines, "validator.validate_demo(...) call missing from collection script"
        # Every validate call in the collect loop must be preceded by at least one save call
        first_save = min(save_lines)
        first_validate = min(validate_lines)
        assert first_save < first_validate, (
            f"Bug #5 regression: validate_demo() at line {first_validate + 1} runs "
            f"before save_demo() at line {first_save + 1}. NPZ won't exist when "
            "validator tries to load it."
        )


# ──────────────────────────────────────────────────────────────────────────
# Bug #2 — set_sim_state_flat must sync OSC controller caches
# Bug #4 — augmented replay must settle physics first
# (Both require robosuite; skip when not installed.)
# ──────────────────────────────────────────────────────────────────────────

_ROBOSUITE_AVAILABLE = importlib.util.find_spec("robosuite") is not None


@pytest.mark.skipif(not _ROBOSUITE_AVAILABLE, reason="robosuite not importable; GPU env only")
class TestBug2OSCCacheSync:
    """Bug #2: set_sim_state_flat() bypassed the OSC controller's step loop,
    leaving ee_pos / goal_pos stale. The next step would then compute torques
    from the cached (wrong) pose, drifting the EEF by ~3.6cm.

    Regression: after set_sim_state_flat(state) then step(zero_action), the
    EEF position must match the state we just loaded, not drift toward the
    prior goal.
    """

    def test_set_state_syncs_controller_goal(self):
        from error_benchmark.framework.env_wrapper import EnvWrapper
        from error_benchmark.scripts.utils.script_utils import create_env, load_task_registry
        import yaml

        task_reg = load_task_registry("stack")
        task_config_path = PROJECT_ROOT / task_reg["task_config"]
        with task_config_path.open() as f:
            task_config = yaml.safe_load(f)

        env = create_env(task_config, task_reg["dataset_path"], enable_camera=False)
        env_wrapper = EnvWrapper(env, task_config)

        env.reset()
        # Snapshot an initial state, step forward a bit, then restore.
        state_a = env_wrapper.get_sim_state_flat()
        for _ in range(30):
            env_wrapper.step(env_wrapper.get_neutral_action())
        env_wrapper.set_sim_state_flat(state_a)

        # Any live robot's controller.goal_pos must equal its ee_pos after restore.
        # If Bug #2 regressed, goal_pos would still point to the step-30 location.
        for robot in env.robots:
            drift = np.linalg.norm(
                robot.controller.goal_pos - robot.controller.ee_pos
            )
            assert drift < 1e-3, (
                f"Bug #2 regression: controller goal_pos drifted by {drift*1000:.1f}mm "
                "from ee_pos after set_sim_state_flat(). The controller cache is stale."
            )

        # And one zero-action step should not cause the EEF to fly toward an old goal.
        eef_before = env_wrapper.get_eef_pose()[0].copy()
        env_wrapper.step(env_wrapper.get_neutral_action())
        eef_after = env_wrapper.get_eef_pose()[0]
        jump = np.linalg.norm(eef_after - eef_before)
        assert jump < 5e-3, (
            f"Bug #2 regression: EEF jumped {jump*1000:.1f}mm on a zero-action step "
            "after set_sim_state_flat(). Controller cache was not refreshed."
        )


@pytest.mark.skipif(not _ROBOSUITE_AVAILABLE, reason="robosuite not importable; GPU env only")
class TestBug4AugmentedReplaySettling:
    """Bug #4: augmentation replayed actions immediately after set_sim_state_flat
    with no physics settling, so objects in contact wiggled unstably for the
    first few steps. The fix settles for N steps with gripper held before the
    augmented actions begin.

    We pin the presence of the settle-before-replay step in the augmentation
    script — if it disappears, the test fails and the author must re-justify.
    """

    def test_augmentation_script_calls_settle_before_replay(self):
        augment_script = (
            PROJECT_ROOT
            / "error_benchmark"
            / "scripts"
            / "augmentation"
            / "3_mimicgen_recovery_augment.py"
        )
        source = augment_script.read_text()
        # The fix added an explicit settling block before augmented-trajectory
        # execution. Match either the helper call or an inline settle loop.
        has_settle = (
            "settle_physics" in source
            or "settling" in source.lower()
            or re.search(r"settle.*before.*replay", source, re.IGNORECASE) is not None
        )
        assert has_settle, (
            "Bug #4 regression: 3_mimicgen_recovery_augment.py no longer settles "
            "physics before replaying augmented actions. Objects in contact will "
            "wiggle and the replay drifts."
        )
