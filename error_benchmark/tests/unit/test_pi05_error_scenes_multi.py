import json

from error_benchmark.scripts.training.eval_pi05_error_scenes_multi import (
    build_worker_shards,
    limit_scene_paths_per_group,
    load_partial_results,
)


def write_scene(path, error_name, degree, subtype_id):
    path.write_text(
        json.dumps(
            {
                "error_spec": {
                    "error_name": error_name,
                    "degree": degree,
                },
                "labels": {
                    "error_name": error_name,
                    "degree": degree,
                    "subtype_id": subtype_id,
                },
            }
        )
    )
    return path


def test_one_per_group_creates_one_worker_per_subtype(tmp_path):
    scenes = [
        write_scene(tmp_path / "a.json", "stuck", "D0", "stuck_D0"),
        write_scene(tmp_path / "b.json", "stuck", "D0", "stuck_D0"),
        write_scene(tmp_path / "c.json", "drop", "D1", "drop_D1"),
    ]

    shards = build_worker_shards(
        scenes,
        num_workers=16,
        worker_mode="one_per_group",
        group_by="subtype_id",
    )

    assert len(shards) == 2
    assert {shard.group_value for shard in shards} == {"stuck_D0", "drop_D1"}
    assert {len(shard.paths) for shard in shards} == {1, 2}
    for shard in shards:
        assert shard.group_key == "subtype_id"


def test_round_robin_preserves_requested_worker_count_limit(tmp_path):
    scenes = [
        write_scene(tmp_path / f"{idx}.json", "stuck", "D0", "stuck_D0")
        for idx in range(5)
    ]

    shards = build_worker_shards(
        scenes,
        num_workers=2,
        worker_mode="round_robin",
        group_by="subtype_id",
    )

    assert len(shards) == 2
    assert [len(shard.paths) for shard in shards] == [3, 2]
    assert all(shard.group_value is None for shard in shards)


def test_limit_scene_paths_per_group_keeps_first_n_per_subtype(tmp_path):
    scenes = [
        write_scene(tmp_path / "a.json", "stuck", "D0", "stuck_D0"),
        write_scene(tmp_path / "b.json", "stuck", "D0", "stuck_D0"),
        write_scene(tmp_path / "c.json", "stuck", "D0", "stuck_D0"),
        write_scene(tmp_path / "d.json", "drop", "D1", "drop_D1"),
        write_scene(tmp_path / "e.json", "drop", "D1", "drop_D1"),
    ]

    selected = limit_scene_paths_per_group(
        scenes,
        group_by="subtype_id",
        limit_per_group=2,
    )

    assert [path.name for path in selected] == ["a.json", "b.json", "d.json", "e.json"]


def test_limit_scene_paths_per_group_disabled_returns_all_paths(tmp_path):
    scenes = [
        write_scene(tmp_path / "a.json", "stuck", "D0", "stuck_D0"),
        write_scene(tmp_path / "b.json", "drop", "D1", "drop_D1"),
    ]

    assert limit_scene_paths_per_group(scenes, group_by="subtype_id", limit_per_group=0) == scenes
    assert limit_scene_paths_per_group(scenes, group_by="subtype_id", limit_per_group=None) == scenes


def test_limit_scene_paths_per_group_uses_v5_filename_groups(tmp_path):
    names = [
        "v5_collision_eef_object_D0_aaa.json",
        "v5_collision_eef_object_D0_bbb.json",
        "v5_drop_object_D1_ccc.json",
        "v5_drop_object_D1_ddd.json",
    ]
    scenes = []
    for name in names:
        path = tmp_path / name
        path.write_text("")
        scenes.append(path)

    selected = limit_scene_paths_per_group(
        scenes,
        group_by="subtype_id",
        limit_per_group=1,
    )

    assert [path.name for path in selected] == [
        "v5_collision_eef_object_D0_aaa.json",
        "v5_drop_object_D1_ccc.json",
    ]


def test_partial_resume_uses_latest_scene_result_and_backfills_group(tmp_path):
    partial = tmp_path / "partial.jsonl"
    partial.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "scene_id": "scene_1",
                        "subtype_id": "stuck_D0",
                        "error_name": "stuck",
                        "degree": "D0",
                        "success": False,
                    }
                ),
                json.dumps(
                    {
                        "scene_id": "scene_1",
                        "subtype_id": "stuck_D0",
                        "error_name": "stuck",
                        "degree": "D0",
                        "success": True,
                    }
                ),
            ]
        )
        + "\n"
    )

    results = load_partial_results(partial, group_by="subtype_id")

    assert len(results) == 1
    assert results["scene_1"]["success"] is True
    assert results["scene_1"]["group_key"] == "subtype_id"
    assert results["scene_1"]["group_value"] == "stuck_D0"


def test_resume_filters_completed_groups_before_sharding(tmp_path):
    completed = write_scene(tmp_path / "completed.json", "stuck", "D0", "stuck_D0")
    remaining = write_scene(tmp_path / "remaining.json", "drop", "D1", "drop_D1")
    partial = tmp_path / "partial.jsonl"
    partial.write_text(
        json.dumps(
            {
                "scene_id": completed.stem,
                "subtype_id": "stuck_D0",
                "error_name": "stuck",
                "degree": "D0",
                "success": True,
            }
        )
        + "\n"
    )

    results = load_partial_results(partial, group_by="subtype_id")
    todo = [path for path in [completed, remaining] if path.stem not in results]
    shards = build_worker_shards(
        todo,
        num_workers=16,
        worker_mode="one_per_group",
        group_by="subtype_id",
    )

    assert len(shards) == 1
    assert shards[0].group_value == "drop_D1"
    assert shards[0].paths == [str(remaining)]
