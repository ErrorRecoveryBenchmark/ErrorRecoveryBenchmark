import json
from argparse import Namespace

import pytest

from error_benchmark.scripts.training.verify_gr00t_success_targets import verify


def _result(root, task, setting, clean_successes, error_successes=0):
    base = root / task / setting / "step_20000"
    (base / "clean").mkdir(parents=True)
    (base / "error").mkdir()
    (base / "clean" / f"{task}.json").write_text(
        json.dumps(
            {
                "clean_rollouts": {
                    "successes": clean_successes,
                    "total": 10,
                }
            }
        ),
        encoding="utf-8",
    )
    (base / "error" / f"{task}.json").write_text(
        json.dumps(
            {
                "error_scenes": {
                    "successes": error_successes,
                    "total": 12,
                }
            }
        ),
        encoding="utf-8",
    )


def _args(root, **overrides):
    values = {
        "results_root": root,
        "settings": ["nominal", "nominal_recovery"],
        "tasks": ["stack", "pick_place"],
        "simple_tasks": ["stack", "pick_place"],
        "clean_total": 10,
        "overall_min": 0.5,
        "simple_min": 0.8,
        "require_error_eval": False,
        "expected_error_scenes_per_setting": 24,
    }
    values.update(overrides)
    return Namespace(**values)


def test_verify_accepts_complete_results_at_targets(tmp_path, capsys):
    for setting in ("nominal", "nominal_recovery"):
        _result(tmp_path, "stack", setting, 8)
        _result(tmp_path, "pick_place", setting, 9)

    summary = verify(_args(tmp_path))

    assert summary["nominal"]["clean_success_rate"] == pytest.approx(0.85)
    assert "GR00T_SUCCESS_TARGETS_PASS" in capsys.readouterr().out


def test_verify_rejects_simple_task_below_target(tmp_path):
    for setting in ("nominal", "nominal_recovery"):
        _result(tmp_path, "stack", setting, 7)
        _result(tmp_path, "pick_place", setting, 9)

    with pytest.raises(ValueError, match="stack.*0.700000.*0.800000"):
        verify(_args(tmp_path))


def test_verify_checks_error_scene_denominator(tmp_path, capsys):
    for setting in ("nominal", "nominal_recovery"):
        _result(tmp_path, "stack", setting, 8, error_successes=3)
        _result(tmp_path, "pick_place", setting, 9, error_successes=4)

    summary = verify(
        _args(
            tmp_path,
            require_error_eval=True,
            overall_min=0.0,
            simple_min=0.0,
        )
    )

    assert summary["nominal"]["error_total"] == 24
    assert "GR00T_ERROR_EVAL_COMPLETE" in capsys.readouterr().out
