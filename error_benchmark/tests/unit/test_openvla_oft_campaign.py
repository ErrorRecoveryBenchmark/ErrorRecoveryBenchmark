from pathlib import Path

import pytest

from error_benchmark.scripts.training.launch_openvla_oft_campaign import (
    GPUS,
    OFFICIAL_PARAMETERS,
    completed_training_log,
    finite_losses,
    training_command,
)
from error_benchmark.scripts.training.prepare_openvla_oft_data import (
    DATASETS,
    action_chunk_indices,
)


def test_twelve_paired_runs_cover_all_six_tasks():
    assert len(DATASETS) == 12
    tasks = {spec.task for spec in DATASETS}
    assert tasks == {
        "pick_place",
        "coffee",
        "stack",
        "stack_three",
        "threading",
        "three_piece_assembly",
    }
    for task in tasks:
        assert {spec.setting for spec in DATASETS if spec.task == task} == {
            "nominal",
            "nominal_recovery",
        }


def test_action_chunks_repeat_final_action_without_crossing_episode():
    assert action_chunk_indices(10, 0) == list(range(8))
    assert action_chunk_indices(10, 8) == [8, 9, 9, 9, 9, 9, 9, 9]
    assert action_chunk_indices(1, 0) == [0] * 8
    with pytest.raises(ValueError):
        action_chunk_indices(3, 3)


def test_official_libero_oft_parameters_are_locked():
    assert OFFICIAL_PARAMETERS["lora_rank"] == 32
    assert OFFICIAL_PARAMETERS["lora_alpha"] == 16
    assert OFFICIAL_PARAMETERS["lora_dropout"] == 0.0
    assert OFFICIAL_PARAMETERS["lora_targets"] == "all-linear"
    assert OFFICIAL_PARAMETERS["batch_size"] == 8
    assert OFFICIAL_PARAMETERS["grad_accumulation_steps"] == 8
    assert OFFICIAL_PARAMETERS["effective_batch_size"] == 64
    assert OFFICIAL_PARAMETERS["learning_rate"] == 5e-4
    assert OFFICIAL_PARAMETERS["num_steps_before_decay"] == 100000
    assert OFFICIAL_PARAMETERS["max_steps"] == 150005
    assert OFFICIAL_PARAMETERS["action_chunk"] == 8
    assert OFFICIAL_PARAMETERS["action_dim"] == 7
    assert OFFICIAL_PARAMETERS["proprio_dim"] == 8
    assert OFFICIAL_PARAMETERS["merge_lora_during_training"] is False


def test_training_command_uses_official_recipe_and_distinct_run_name():
    command = training_command(
        Path("/env/bin/python"),
        Path("/repo"),
        Path("/data"),
        Path("/output"),
        "stack_three",
        "nominal_recovery",
        vla_path=Path("/models/openvla-7b"),
    )
    joined = " ".join(command)
    assert Path(command[0]).name == "torchrun"
    assert "--nproc-per-node 1" in joined
    vla_path = Path(command[command.index("--vla_path") + 1])
    assert vla_path.name == "openvla-7b"
    assert vla_path.parent.name == "models"
    assert "recover_bench_open_vla/stack_three_nominal_recovery" in joined
    assert "--run_id_override stack_three_nominal_recovery" in joined
    assert "--batch_size 8" in joined
    assert "--grad_accumulation_steps 8" in joined
    assert "--merge_lora_during_training False" in joined
    assert GPUS == (3, 5, 6, 7)


def test_resume_completion_requires_the_formal_max_step_marker():
    assert completed_training_log("Max step 150005 reached! Stopping training...")
    assert not completed_training_log("Max step 10000 reached! Stopping training...")


def test_finite_loss_parser_rejects_nan_and_inf():
    log = (
        Path(__file__).resolve().parents[3]
        / "outputs"
        / "test-fixtures"
        / "openvla"
        / "train.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "OPENVLA_METRIC step=0 loss=0.5\n"
        "OPENVLA_METRIC step=1 loss=nan\n"
        "OPENVLA_METRIC step=2 loss=inf\n"
        "OPENVLA_METRIC step=3 loss=0.25\n",
        encoding="utf-8",
    )
    assert finite_losses(log) == [0.5, 0.25]
