#!/usr/bin/env python3
"""Combine Pi0.5 merged and error-training LeRobot datasets for one task."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


TASKS = (
    "pick_place",
    "coffee",
    "stack",
    "stack_three",
    "threading",
    "three_piece_assembly",
)

DEFAULT_REPORT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "pi05_merged_error_training_lerobot_reports"
)


def _repo_id(task: str, suffix: str) -> str:
    return f"benchmark/mimicgen_{task}_{suffix}"


def _dataset_root(hf_lerobot_home: Path, repo_id: str) -> Path:
    return hf_lerobot_home / repo_id


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            json.dump(row, f)
            f.write("\n")
    os.replace(tmp, path)


def _data_file(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    rel = info["data_path"].format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )
    return root / rel


def _replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx < 0:
        raise KeyError(f"missing parquet column: {name}")
    return table.set_column(idx, name, values)


def _remap_task_index_array(
    table: pa.Table,
    task_index_map: dict[int, int],
    length: int,
) -> pa.Array:
    task_indices = table.column("task_index").to_pylist()
    mapped = [task_index_map[int(value)] for value in task_indices]
    if len(mapped) != length:
        raise ValueError(f"task_index length mismatch: {len(mapped)} != {length}")
    return pa.array(mapped, type=pa.int64())


def _rewrite_table(
    table: pa.Table,
    new_episode_index: int,
    global_frame_offset: int,
    task_index_map: dict[int, int],
) -> pa.Table:
    length = table.num_rows
    new_index = pa.array(range(global_frame_offset, global_frame_offset + length), type=pa.int64())
    new_episode = pa.array([new_episode_index] * length, type=pa.int64())
    new_task_index = _remap_task_index_array(table, task_index_map, length)

    table = _replace_column(table, "index", new_index)
    table = _replace_column(table, "episode_index", new_episode)
    table = _replace_column(table, "task_index", new_task_index)
    return table


def _sequence_stats(start: int, length: int) -> dict[str, list[float | int]]:
    end = start + length - 1
    values = np.arange(start, start + length, dtype=np.float64)
    return {
        "min": [int(start)],
        "max": [int(end)],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [int(length)],
    }


def _constant_stats(value: int, length: int) -> dict[str, list[float | int]]:
    return {
        "min": [int(value)],
        "max": [int(value)],
        "mean": [float(value)],
        "std": [0.0],
        "count": [int(length)],
    }


def _remap_stats(
    source_stats: dict[str, Any],
    new_episode_index: int,
    global_frame_offset: int,
    new_task_indices: list[int],
    length: int,
) -> dict[str, Any]:
    stats = deepcopy(source_stats)
    stats["episode_index"] = _constant_stats(new_episode_index, length)
    stats["index"] = _sequence_stats(global_frame_offset, length)
    if len(set(new_task_indices)) == 1:
        stats["task_index"] = _constant_stats(new_task_indices[0], length)
    else:
        values = np.asarray(new_task_indices, dtype=np.float64)
        stats["task_index"] = {
            "min": [int(values.min())],
            "max": [int(values.max())],
            "mean": [float(values.mean())],
            "std": [float(values.std())],
            "count": [int(length)],
        }
    return stats


def _postflight(repo_id: str, root: Path, expected_episodes: int, expected_frames: int) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "PASS", "repo_id": repo_id, "errors": []}
    try:
        meta = LeRobotDatasetMetadata(repo_id, root=root)
        dataset = LeRobotDataset(repo_id, root=root)
        result["num_tasks"] = int(meta.total_tasks)
        result["num_episodes"] = int(meta.total_episodes)
        result["num_frames"] = int(len(dataset))
        result["expected_episodes"] = int(expected_episodes)
        result["expected_frames"] = int(expected_frames)
        result["tasks"] = list(meta.tasks.values())

        if meta.total_episodes != expected_episodes:
            result["errors"].append(
                f"episode count mismatch: {meta.total_episodes} != {expected_episodes}"
            )
        if len(dataset) != expected_frames:
            result["errors"].append(f"frame count mismatch: {len(dataset)} != {expected_frames}")
        if meta.total_tasks < 1:
            result["errors"].append("no LeRobot task metadata")

        sample_indices = sorted({0, max(0, len(dataset) // 2), max(0, len(dataset) - 1)})
        sample_summaries = []
        for idx in sample_indices:
            sample = dataset[idx]
            sample_summaries.append(
                {
                    "index": int(idx),
                    "episode_index": int(np.asarray(sample["episode_index"]).item()),
                    "frame_index": int(np.asarray(sample["frame_index"]).item()),
                    "task": sample.get("task"),
                }
            )
            for key in ("image", "wrist_image", "state", "actions", "task"):
                if key not in sample:
                    result["errors"].append(f"sample {idx} missing {key}")
            if "state" in sample and tuple(np.asarray(sample["state"]).shape)[-1:] != (8,):
                result["errors"].append(f"sample {idx} bad state shape")
            if "actions" in sample and tuple(np.asarray(sample["actions"]).shape)[-1:] != (7,):
                result["errors"].append(f"sample {idx} bad action shape")
        result["sample_summaries"] = sample_summaries
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")

    if result["errors"]:
        result["status"] = "FAIL"
    return result


def combine_repos(
    task: str,
    source_repo_ids: list[str],
    output_repo_id: str,
    hf_lerobot_home: Path,
    overwrite: bool,
) -> dict[str, Any]:
    output_root = _dataset_root(hf_lerobot_home, output_repo_id)
    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        else:
            raise RuntimeError(f"output dataset exists at {output_root}; use --overwrite")

    source_roots = [_dataset_root(hf_lerobot_home, repo_id) for repo_id in source_repo_ids]
    for root in source_roots:
        if not root.is_dir():
            raise FileNotFoundError(f"source dataset missing: {root}")

    source_infos = [_load_json(root / "meta" / "info.json") for root in source_roots]
    source_tasks = [_load_jsonl(root / "meta" / "tasks.jsonl") for root in source_roots]
    source_episodes = [_load_jsonl(root / "meta" / "episodes.jsonl") for root in source_roots]
    source_episode_stats = [_load_jsonl(root / "meta" / "episodes_stats.jsonl") for root in source_roots]

    output_tasks: list[dict[str, Any]] = []
    task_text_to_new_index: dict[str, int] = {}
    task_maps: list[dict[int, int]] = []
    for task_rows in source_tasks:
        mapping: dict[int, int] = {}
        for row in task_rows:
            task_text = row["task"]
            if task_text not in task_text_to_new_index:
                task_text_to_new_index[task_text] = len(output_tasks)
                output_tasks.append({"task_index": len(output_tasks), "task": task_text})
            mapping[int(row["task_index"])] = task_text_to_new_index[task_text]
        task_maps.append(mapping)

    output_info = deepcopy(source_infos[0])
    for info in source_infos[1:]:
        if info.get("features") != output_info.get("features"):
            raise ValueError("source datasets have incompatible feature schemas")
        if info.get("fps") != output_info.get("fps"):
            raise ValueError("source datasets have incompatible fps values")
        if info.get("codebase_version") != output_info.get("codebase_version"):
            raise ValueError("source datasets have incompatible codebase versions")

    output_rows: list[dict[str, Any]] = []
    output_stats_rows: list[dict[str, Any]] = []
    total_frames = 0
    total_episodes = 0

    for source_idx, (root, info, episodes, stats_rows, task_index_map) in enumerate(
        zip(source_roots, source_infos, source_episodes, source_episode_stats, task_maps, strict=True)
    ):
        stats_by_episode = {int(row["episode_index"]): row["stats"] for row in stats_rows}
        for source_episode in episodes:
            old_episode_index = int(source_episode["episode_index"])
            new_episode_index = total_episodes
            length = int(source_episode["length"])
            parquet_path = _data_file(root, info, old_episode_index)
            if not parquet_path.is_file():
                raise FileNotFoundError(f"source episode parquet missing: {parquet_path}")

            table = pq.read_table(parquet_path)
            if table.num_rows != length:
                raise ValueError(
                    f"{parquet_path} row count mismatch: {table.num_rows} != metadata length {length}"
                )

            old_task_indices = table.column("task_index").to_pylist()
            new_task_indices = [task_index_map[int(value)] for value in old_task_indices]
            table = _rewrite_table(table, new_episode_index, total_frames, task_index_map)

            output_path = _data_file(output_root, output_info, new_episode_index)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, output_path)

            output_episode = {
                "episode_index": new_episode_index,
                "tasks": source_episode["tasks"],
                "length": length,
            }
            output_rows.append(output_episode)

            output_stats_rows.append(
                {
                    "episode_index": new_episode_index,
                    "stats": _remap_stats(
                        stats_by_episode[old_episode_index],
                        new_episode_index,
                        total_frames,
                        new_task_indices,
                        length,
                    ),
                }
            )

            total_frames += length
            total_episodes += 1

        print(
            f"[combine-lerobot] {task}: copied source {source_idx + 1}/{len(source_roots)} "
            f"({root.name}), total episodes={total_episodes}, frames={total_frames}",
            flush=True,
        )

    chunks_size = int(output_info.get("chunks_size", 1000))
    output_info["total_episodes"] = total_episodes
    output_info["total_frames"] = total_frames
    output_info["total_tasks"] = len(output_tasks)
    output_info["total_chunks"] = (total_episodes + chunks_size - 1) // chunks_size
    output_info["splits"] = {"train": f"0:{total_episodes}"}

    _write_json(output_root / "meta" / "info.json", output_info)
    _write_jsonl(output_root / "meta" / "tasks.jsonl", output_tasks)
    _write_jsonl(output_root / "meta" / "episodes.jsonl", output_rows)
    _write_jsonl(output_root / "meta" / "episodes_stats.jsonl", output_stats_rows)

    return {
        "task": task,
        "repo_id": output_repo_id,
        "output_path": str(output_root),
        "source_repo_ids": source_repo_ids,
        "expected_episodes": total_episodes,
        "expected_frames": total_frames,
        "source_counts": [
            {
                "repo_id": repo_id,
                "episodes": int(info["total_episodes"]),
                "frames": int(info["total_frames"]),
            }
            for repo_id, info in zip(source_repo_ids, source_infos, strict=True)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--merged-repo-suffix", default="merged")
    parser.add_argument("--error-repo-suffix", default="error_training")
    parser.add_argument("--output-repo-suffix", default="merged_error_training")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument(
        "--hf-lerobot-home",
        default=os.environ.get(
            "HF_LEROBOT_HOME",
            os.path.expanduser("~/.cache/huggingface/lerobot"),
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_lerobot_home = Path(args.hf_lerobot_home).expanduser()
    output_repo_id = args.repo_id or _repo_id(args.task, args.output_repo_suffix)
    output_root = _dataset_root(hf_lerobot_home, output_repo_id)
    source_repo_ids = [
        _repo_id(args.task, args.merged_repo_suffix),
        _repo_id(args.task, args.error_repo_suffix),
    ]
    report_path = Path(args.report_root) / f"{args.task}_validation_report.json"

    report: dict[str, Any] = {
        "status": "RUNNING",
        "task": args.task,
        "repo_id": output_repo_id,
        "output_path": str(output_root),
        "source_repo_ids": source_repo_ids,
        "validate_only": bool(args.validate_only),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    try:
        if args.validate_only:
            if not output_root.is_dir():
                raise FileNotFoundError(f"output dataset missing: {output_root}")
            meta = LeRobotDatasetMetadata(output_repo_id, root=output_root)
            expected_episodes = int(meta.total_episodes)
            expected_frames = int(meta.total_frames)
            report.update(_postflight(output_repo_id, output_root, expected_episodes, expected_frames))
        else:
            report.update(
                combine_repos(
                    task=args.task,
                    source_repo_ids=source_repo_ids,
                    output_repo_id=output_repo_id,
                    hf_lerobot_home=hf_lerobot_home,
                    overwrite=args.overwrite,
                )
            )
            report.update(
                _postflight(
                    output_repo_id,
                    output_root,
                    int(report["expected_episodes"]),
                    int(report["expected_frames"]),
                )
            )
        if report["status"] != "PASS":
            raise RuntimeError("combined LeRobot postflight failed")
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _write_json(report_path, report)


if __name__ == "__main__":
    main()
