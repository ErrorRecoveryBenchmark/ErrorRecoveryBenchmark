from pathlib import Path
import pytest

from error_benchmark.scripts.training.verify_gr00t_campaign import (
    ERROR_TOTALS,
    aggregate_best_recovery,
    discover_checkpoint_steps,
)


SCHEDULER = Path(__file__).resolve().parents[3] / "workflow" / "run_gr00t_paired_training.sh"
TRAINING_DIR = (
    Path(__file__).resolve().parents[2] / "scripts" / "training"
)


@pytest.mark.skipif(not SCHEDULER.exists(), reason="Internal cluster scheduler is not part of the public code export")
def test_scheduler_supports_two_pairs_and_preserves_eval_checkpoints():
    script = SCHEDULER.read_text(encoding="utf-8")

    assert "GR00T_GPUS" in script
    assert "read -r -a GPUS" in script
    assert "full_campaign" in script
    assert "--save-total-limit 30" in script
    assert "--save-steps 1000" in script


@pytest.mark.skipif(not SCHEDULER.exists(), reason="Internal cluster scheduler is not part of the public code export")
def test_scheduler_keeps_paper_matched_pair_names():
    script = SCHEDULER.read_text(encoding="utf-8")

    assert "nominal nominal_recovery" in script
    assert '${DATA_ROOT}/${task}/${setting}' in script


def test_existing_evaluators_can_use_one_external_gr00t_server():
    clean = (TRAINING_DIR / "eval_clean_multi.py").read_text(encoding="utf-8")
    error = (TRAINING_DIR / "eval_pi05_error_scenes_multi.py").read_text(
        encoding="utf-8"
    )

    assert "--external_server" in clean
    assert "--external_server" in error
    assert '"stack_three"' in clean
    assert '"three_piece_assembly"' in clean
    assert "--success-streak-frames" in clean
    assert "ConsecutiveSuccessTracker(n=success_streak_frames)" in clean
    assert "--success-streak-frames" in error
    assert "ConsecutiveSuccessTracker(n=success_streak_frames)" in error

    runner = (TRAINING_DIR / "run_gr00t_checkpoint_eval.py").read_text(
        encoding="utf-8"
    )
    assert 'eval_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)' in runner
    assert 'eval_env["MUJOCO_EGL_DEVICE_ID"] = "0"' in runner


def test_checkpoint_discovery_starts_at_7000():
    checkpoint_root = (
        Path(__file__).resolve().parents[3]
        / "outputs"
        / "test-fixtures"
        / "gr00t-checkpoints"
    )
    for name in ("checkpoint-6000", "checkpoint-7000", "checkpoint-9000", "notes"):
        (checkpoint_root / name).mkdir(parents=True, exist_ok=True)

    assert discover_checkpoint_steps(checkpoint_root, min_step=7000) == [7000, 9000]


def test_paper_matched_error_denominator_and_best_step_aggregation():
    assert sum(ERROR_TOTALS.values()) == 1360
    rows = [
        {"task": "coffee", "step": 7000, "successes": 40, "total": 200},
        {"task": "coffee", "step": 8000, "successes": 50, "total": 200},
        {"task": "threading", "step": 7000, "successes": 70, "total": 200},
        {"task": "threading", "step": 8000, "successes": 60, "total": 200},
    ]

    aggregate = aggregate_best_recovery(
        rows,
        tasks=("coffee", "threading"),
    )

    assert aggregate["selected_steps"] == {"coffee": 8000, "threading": 7000}
    assert aggregate["successes"] == 120
    assert aggregate["total"] == 400
    assert aggregate["rsr"] == 0.3
