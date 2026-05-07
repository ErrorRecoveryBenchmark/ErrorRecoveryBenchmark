#!/usr/bin/env python
"""
Unit tests for BC-RNN baseline evaluation helpers.
"""

import importlib.util
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[3]


def _load_module(module_name: str, relative_path: str):
    module_path = PROJECT_DIR / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


eval_mod = _load_module(
    "eval_bc_rnn_error_scenes",
    "error_benchmark/scripts/training/eval_bc_rnn_error_scenes.py",
)
agg_mod = _load_module(
    "aggregate_baseline_results",
    "error_benchmark/scripts/training/aggregate_baseline_results.py",
)


def _write_scene(path: Path, subtype_id: str):
    path.write_text(json.dumps({
        "labels": {
            "subtype_id": subtype_id,
            "error_name": subtype_id.split("_", 1)[0],
            "degree": subtype_id.split("_", 1)[-1],
        },
        "error_spec": {
            "error_name": subtype_id.split("_", 1)[0],
            "degree": subtype_id.split("_", 1)[-1],
        },
    }))


def test_select_balanced_scene_subset_exact_limit(tmp_path):
    scene_paths = []
    for subtype_id in ("alpha_d0", "beta_d0", "gamma_d0"):
        for idx in range(4):
            path = tmp_path / f"{subtype_id}_{idx}.json"
            _write_scene(path, subtype_id)
            scene_paths.append(path)

    (tmp_path / "broken.json").write_text("{not-json")

    selected = eval_mod.select_balanced_scene_subset(
        scene_paths + [tmp_path / "broken.json"],
        scenes_limit=7,
        scenes_seed=123,
    )

    assert len(selected) == 7
    assert [path.name for path in selected] == [
        path.name
        for path in eval_mod.select_balanced_scene_subset(
            scene_paths + [tmp_path / "broken.json"],
            scenes_limit=7,
            scenes_seed=123,
        )
    ]

    selected_by_subtype, invalid = eval_mod.build_scene_catalog(selected)
    assert invalid == []
    counts = sorted(len(paths) for paths in selected_by_subtype.values())
    assert counts == [2, 2, 3]


def test_parse_robomimic_average_rollout_stats_json():
    output = """
some setup output
Average Rollout Stats
{
    "Return": 1.0,
    "Horizon": 42.0,
    "Success_Rate": 0.6,
    "Num_Success": 3.0
}
"""
    parsed = eval_mod.parse_robomimic_success_rate(output, num_rollouts=5)

    assert parsed["sr"] == 0.6
    assert parsed["successes"] == 3
    assert parsed["total"] == 5
    assert parsed["parse_method"] == "average_rollout_stats_json"


def test_robomimic_subprocess_env_includes_local_import_roots(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/tmp/existing")
    env = eval_mod.build_robomimic_subprocess_env()
    pythonpath = env["PYTHONPATH"].split(":")

    assert str(PROJECT_DIR) in pythonpath
    assert str(PROJECT_DIR / "shared" / "mimicgen_workspace" / "robosuite") in pythonpath
    assert str(PROJECT_DIR / "shared" / "mimicgen_workspace" / "mimicgen") in pythonpath
    assert "/tmp/existing" in pythonpath


def test_build_summary_filters_requested_tasks(tmp_path):
    def write_eval(task: str, clean_sr: float, error_sr: float):
        payload = {
            "task": task,
            "checkpoint": f"/tmp/{task}.pth",
            "checkpoint_epoch": 20,
            "clean_rollouts": {"sr": clean_sr, "total": 10, "successes": int(clean_sr * 10)},
            "error_scenes": {
                "overall_sr": error_sr,
                "total": 8,
                "successes": int(error_sr * 8),
                "by_subtype": {
                    f"{task}_subtype": {
                        "sr": error_sr,
                        "successes": int(error_sr * 8),
                        "total": 8,
                    }
                },
            },
        }
        (tmp_path / f"{task}.json").write_text(json.dumps(payload))

    write_eval("coffee", 0.6, 0.25)
    write_eval("stack", 0.8, 0.5)
    write_eval("pick_place", 0.4, 0.125)

    summary, md = agg_mod.build_summary(tmp_path, ["coffee", "stack"])
    md_text = "\n".join(md)

    assert summary["selected_tasks"] == ["coffee", "stack"]
    assert set(summary["tasks"]) == {"coffee", "stack"}
    assert "pick_place" not in md_text
    assert "| coffee | epoch_20 | 60.0% | 10 | 25.0% | 8 |" in md_text
    assert "| stack | epoch_20 | 80.0% | 10 | 50.0% | 8 |" in md_text
