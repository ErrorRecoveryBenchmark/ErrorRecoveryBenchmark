"""Source-scan specs: the eval scripts surface the sampled-eval-set manifest
and per-RBG aggregation, and the summarizer wires the Fisher comparison."""
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TR = REPO / "error_benchmark" / "scripts" / "training"

EVAL_SCRIPTS = [
    "eval_pi05_error_scenes.py",
    "eval_pi05_error_scenes_multi.py",
    "eval_bc_rnn_error_scenes.py",
]


@pytest.mark.parametrize("name", EVAL_SCRIPTS)
def test_eval_script_supports_manifest(name):
    src = (TR / name).read_text(encoding="utf-8")
    assert "--eval_manifest" in src
    assert "eval_manifest_ids" in src


@pytest.mark.parametrize("name", EVAL_SCRIPTS)
def test_eval_script_emits_by_rbg(name):
    src = (TR / name).read_text(encoding="utf-8")
    assert "by_rbg" in src
    assert "from error_benchmark.scripts.training.eval_metrics import rbg_of" in src


def test_summarizer_wires_fisher_compare():
    src = (TR / "summarize_validation_matrix.py").read_text(encoding="utf-8")
    assert "compare_error_scene_srs" in src
    assert "p_value" in src
    assert "significant" in src
