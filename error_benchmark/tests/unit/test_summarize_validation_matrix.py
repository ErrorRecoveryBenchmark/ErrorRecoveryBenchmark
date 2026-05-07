from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "training" / "summarize_validation_matrix.py"
SPEC = importlib.util.spec_from_file_location("summarize_validation_matrix", SCRIPT_PATH)
assert SPEC is not None
summarize_validation_matrix = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(summarize_validation_matrix)


def test_task_artifacts_keeps_stack_and_stack_three_separate(tmp_path: Path) -> None:
    (tmp_path / "meta_stack_merged_step20000.json").write_text("{}")
    (tmp_path / "meta_stack_three_merged_step20000.json").write_text("{}")
    (tmp_path / "results_stack_merged_step20000.json").write_text("{}")
    (tmp_path / "results_stack_three_merged_step20000.json").write_text("{}")

    stack_meta = summarize_validation_matrix.task_artifacts(tmp_path, "meta", "stack")
    stack_three_meta = summarize_validation_matrix.task_artifacts(tmp_path, "meta", "stack_three")
    stack_results = summarize_validation_matrix.task_artifacts(tmp_path, "results", "stack")
    stack_three_results = summarize_validation_matrix.task_artifacts(tmp_path, "results", "stack_three")

    assert [path.name for path in stack_meta] == ["meta_stack_merged_step20000.json"]
    assert [path.name for path in stack_three_meta] == ["meta_stack_three_merged_step20000.json"]
    assert [path.name for path in stack_results] == ["results_stack_merged_step20000.json"]
    assert [path.name for path in stack_three_results] == ["results_stack_three_merged_step20000.json"]
