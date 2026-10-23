#!/usr/bin/env python
"""L2 spec — EVALUATION.md §1 success criterion (S-A1 / L2-01, L2-11).

Locks the protocol rule that a rollout succeeds only after the environment's
instantaneous success check is True for 10 *consecutive* frames, and that every
rollout loop (error RSR + clean CSR, Pi0.5 + BC-RNN, single + multi worker)
routes its success decision through ``ConsecutiveSuccessTracker`` rather than
breaking on the first success frame.

Pure-logic + source-scan tests: no robosuite / GPU / data required.
"""

import re
from pathlib import Path

import pytest

from error_benchmark.scripts.training.success_tracker import (
    SUCCESS_CRITERION_NAME,
    SUCCESS_STREAK_FRAMES,
    ConsecutiveSuccessTracker,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING = REPO_ROOT / "error_benchmark" / "scripts" / "training"

# Scripts that run rollouts and must share the consecutive-frame criterion.
ROLLOUT_SCRIPTS = [
    TRAINING / "eval_pi05_error_scenes.py",
    TRAINING / "eval_pi05_error_scenes_multi.py",
    TRAINING / "eval_clean_multi.py",
    TRAINING / "eval_bc_rnn_error_scenes.py",
]


def _src(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ─── Pure tracker logic ───────────────────────────────────────────────

def test_default_threshold_is_ten_frames():
    assert SUCCESS_STREAK_FRAMES == 10
    assert ConsecutiveSuccessTracker().n == 10


def test_single_true_frame_is_not_success():
    tracker = ConsecutiveSuccessTracker()
    assert tracker.update(True) is False
    assert tracker.streak == 1


def test_nine_then_one_more_true_is_success_on_tenth():
    tracker = ConsecutiveSuccessTracker()
    for _ in range(9):
        assert tracker.update(True) is False
    assert tracker.update(True) is True
    assert tracker.succeeded is True


def test_a_false_frame_resets_the_streak():
    tracker = ConsecutiveSuccessTracker()
    for _ in range(9):
        tracker.update(True)
    assert tracker.update(False) is False  # 9-true then 1-false => not success
    assert tracker.streak == 0
    # Must rebuild a full streak of 10 from scratch.
    for _ in range(9):
        assert tracker.update(True) is False
    assert tracker.update(True) is True


def test_reset_clears_streak_between_episodes():
    tracker = ConsecutiveSuccessTracker()
    for _ in range(10):
        tracker.update(True)
    assert tracker.succeeded
    tracker.reset()
    assert tracker.streak == 0
    assert tracker.succeeded is False


def test_custom_threshold():
    tracker = ConsecutiveSuccessTracker(n=3)
    assert tracker.update(True) is False
    assert tracker.update(True) is False
    assert tracker.update(True) is True


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        ConsecutiveSuccessTracker(n=0)


# ─── Source wiring: every rollout loop uses the tracker ───────────────

@pytest.mark.parametrize("script", ROLLOUT_SCRIPTS, ids=lambda p: p.name)
def test_rollout_script_imports_tracker(script):
    assert "ConsecutiveSuccessTracker" in _src(script), (
        f"{script.name} must import/use ConsecutiveSuccessTracker"
    )


@pytest.mark.parametrize("script", ROLLOUT_SCRIPTS, ids=lambda p: p.name)
def test_no_first_frame_break_remains(script):
    """The legacy 'break on the first success frame' pattern must be gone.

    A bare ``if env_wrapper.check_success():`` (used to flip success and break)
    is the single-frame bug; the diagnostic assignment
    ``already_success = env_wrapper.check_success()`` is allowed.
    """
    src = _src(script)
    bare_if = re.findall(r"^\s*if\s+env_wrapper\.check_success\(\)\s*:", src, flags=re.M)
    assert not bare_if, (
        f"{script.name} still breaks on the first success frame: {bare_if}"
    )


@pytest.mark.parametrize("script", ROLLOUT_SCRIPTS, ids=lambda p: p.name)
def test_success_decision_goes_through_tracker(script):
    src = _src(script)
    assert "tracker.update(" in src, (
        f"{script.name} must decide success via tracker.update(...)"
    )


def test_bc_rnn_clean_uses_unified_loop_by_default():
    """BC-RNN clean CSR must default to the same EnvWrapper+tracker loop as RSR,
    not robomimic.run_trained_agent (which uses a different success criterion)."""
    src = _src(TRAINING / "eval_bc_rnn_error_scenes.py")
    assert "def evaluate_clean_rollouts(" in src, (
        "BC-RNN must provide a unified evaluate_clean_rollouts() loop"
    )
    # The --clean_method default must be 'unified'. ``[^)]*?`` keeps the match
    # inside the add_argument(...) call so a default="..." literal in the help
    # text cannot fool it.
    m = re.search(r'--clean_method"[^)]*?\bdefault\s*=\s*"([a-z]+)"', src, flags=re.S)
    assert m is not None, "BC-RNN must expose a --clean_method argument with a default"
    assert m.group(1) == "unified", "BC-RNN clean must default to the unified loop"
    # ... and the robomimic path must be the opt-in branch, unified the fallback.
    assert 'if args.clean_method == "robomimic"' in src
    assert "evaluate_clean_rollouts(" in src


def test_all_tracker_instances_use_default_threshold():
    """CSR and RSR must share the SAME threshold (EVALUATION.md §3): every
    ConsecutiveSuccessTracker(...) call site uses the protocol default n=10.

    A configurable ``n=success_streak_frames`` is allowed ONLY when the script
    pins that parameter's default to SUCCESS_STREAK_FRAMES (the explicit
    streak=1 mode exists solely to reproduce the paper's single-frame Pi0.5
    sweep and must be opt-in, never the default)."""
    for script in ROLLOUT_SCRIPTS:
        src = _src(script)
        calls = re.findall(r"ConsecutiveSuccessTracker\(([^)]*)\)", src)
        assert calls, f"{script.name} constructs no ConsecutiveSuccessTracker"
        for raw in calls:
            arg = raw.strip()
            if arg in ("", "n=SUCCESS_STREAK_FRAMES"):
                continue
            assert arg == "n=success_streak_frames", (
                f"{script.name}: tracker built with non-default threshold '{arg}'"
            )
            assert re.search(r"default\s*=\s*SUCCESS_STREAK_FRAMES", src), (
                f"{script.name}: configurable streak must default to "
                "SUCCESS_STREAK_FRAMES"
            )


def test_results_stamp_success_criterion():
    """Result JSONs must SERIALIZE the criterion name for provenance (S-E6),
    not merely import the constant. The stamp may be conditional, but the
    non-opt-in branch must be SUCCESS_CRITERION_NAME and any "single_frame"
    stamp must be gated on the explicit streak==1 repro mode."""
    assert SUCCESS_CRITERION_NAME == "10_consecutive_frames"
    for script in ROLLOUT_SCRIPTS:
        src = _src(script)
        m = re.search(
            r'"success_criterion"\s*:\s*(\(?[^,}]*(?:\n[^,}]*)*)', src
        )
        assert m is not None, (
            f'{script.name} must write a "success_criterion" key into its '
            "results dict"
        )
        value_expr = m.group(1)
        assert "SUCCESS_CRITERION_NAME" in value_expr, (
            f'{script.name}: the "success_criterion" stamp must use '
            "SUCCESS_CRITERION_NAME (found: %r)" % value_expr[:120]
        )
        if '"single_frame"' in value_expr:
            assert re.search(r"success_streak_frames\s*==\s*1", value_expr), (
                f'{script.name}: a "single_frame" stamp must be gated on '
                "success_streak_frames == 1"
            )
