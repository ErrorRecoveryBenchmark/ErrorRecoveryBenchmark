#!/usr/bin/env python3
"""Convert the existing image-based LeRobot v2.1 data to GR00T's LIBERO layout."""

from __future__ import annotations

import argparse
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any


CAMERAS = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
}
LOW_DIMENSIONS = {
    "state": ("observation.state", 8),
    "actions": ("action", 7),
}
LIBERO_MODALITY = {
    "state": {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        "gripper": {"start": 6, "end": 8},
    },
    "action": {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        "gripper": {"start": 6, "end": 7},
    },
    "video": {
        "image": {"original_key": "observation.images.image"},
        "wrist_image": {"original_key": "observation.images.wrist_image"},
    },
    "annotation": {
        "human.action.task_description": {"original_key": "task_index"},
    },
}


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle)
            handle.write("\n")
    os.replace(temporary, path)


def validate_source_info(info: dict[str, Any]) -> None:
    features = info.get("features", {})
    for source_name, (_, expected_size) in LOW_DIMENSIONS.items():
        actual_shape = features.get(source_name, {}).get("shape")
        if actual_shape != [expected_size]:
            raise ValueError(
                f"{source_name} must have shape [{expected_size}], found {actual_shape}"
            )
    for source_name in CAMERAS:
        feature = features.get(source_name, {})
        if feature.get("dtype") != "image" or feature.get("shape") != [84, 84, 3]:
            raise ValueError(
                f"{source_name} must be an embedded 84x84 RGB image, found {feature}"
            )


def missing_source_columns(column_names: list[str]) -> set[str]:
    return (set(CAMERAS) | set(LOW_DIMENSIONS)) - set(column_names)


def build_output_info(info: dict[str, Any], episode_count: int) -> dict[str, Any]:
    validate_source_info(info)
    output = deepcopy(info)
    features: dict[str, Any] = {}

    state_feature = deepcopy(info["features"]["state"])
    state_feature["names"] = ["x", "y", "z", "roll", "pitch", "yaw", "gripper_0", "gripper_1"]
    features["observation.state"] = state_feature

    action_feature = deepcopy(info["features"]["actions"])
    action_feature["names"] = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    features["action"] = action_feature

    fps = int(info["fps"])
    for output_name in CAMERAS.values():
        features[output_name] = {
            "dtype": "video",
            "shape": [84, 84, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 84,
                "video.width": 84,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    for name, feature in info["features"].items():
        if name not in CAMERAS and name not in LOW_DIMENSIONS:
            features[name] = deepcopy(feature)

    output["features"] = features
    output["total_episodes"] = episode_count
    output["total_frames"] = None
    output["total_videos"] = episode_count * len(CAMERAS)
    output["total_chunks"] = (episode_count + int(info["chunks_size"]) - 1) // int(
        info["chunks_size"]
    )
    output["splits"] = {"train": f"0:{episode_count}"}
    return output


def _episode_path(root: Path, pattern: str, chunk_size: int, episode_index: int) -> Path:
    return root / pattern.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )


def _video_path(
    root: Path,
    pattern: str,
    chunk_size: int,
    episode_index: int,
    video_key: str,
) -> Path:
    return root / pattern.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
        video_key=video_key,
    )


def _encode_video(frames: list[dict[str, Any]], destination: Path, fps: int) -> None:
    import av
    from PIL import Image

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.stem}.tmp.mp4")
    if temporary.exists():
        temporary.unlink()

    with av.open(str(temporary), "w", format="mp4") as container:
        stream = container.add_stream("h264", rate=fps, options={"crf": "30", "g": "2"})
        stream.width = 84
        stream.height = 84
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            payload = frame.get("bytes")
            if payload is None:
                with open(frame["path"], "rb") as handle:
                    payload = handle.read()
            image = Image.open(io.BytesIO(payload)).convert("RGB")
            for packet in stream.encode(av.VideoFrame.from_image(image)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    os.replace(temporary, destination)


def _convert_episode(job: dict[str, Any]) -> tuple[int, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source_root = Path(job["source_root"])
    output_root = Path(job["output_root"])
    info = job["info"]
    episode_index = int(job["episode_index"])
    expected_length = int(job["length"])
    chunk_size = int(info["chunks_size"])

    source_parquet = _episode_path(
        source_root, info["data_path"], chunk_size, episode_index
    )
    output_parquet = _episode_path(
        output_root, info["data_path"], chunk_size, episode_index
    )
    videos = {
        source_name: _video_path(
            output_root,
            info["video_path"],
            chunk_size,
            episode_index,
            output_name,
        )
        for source_name, output_name in CAMERAS.items()
    }
    if output_parquet.is_file() and all(path.is_file() for path in videos.values()):
        return episode_index, expected_length

    table = pq.read_table(source_parquet)
    if table.num_rows != expected_length:
        raise ValueError(
            f"episode {episode_index} has {table.num_rows} rows, expected {expected_length}"
        )
    missing = missing_source_columns(table.column_names)
    if missing:
        raise KeyError(f"episode {episode_index} missing columns: {sorted(missing)}")

    for source_name, destination in videos.items():
        if not destination.is_file():
            _encode_video(table.column(source_name).to_pylist(), destination, int(info["fps"]))

    arrays: list[pa.ChunkedArray] = []
    names: list[str] = []
    for name in table.column_names:
        if name in CAMERAS:
            continue
        arrays.append(table.column(name))
        names.append(LOW_DIMENSIONS.get(name, (name, None))[0])
    output_table = pa.Table.from_arrays(arrays, names=names).replace_schema_metadata(None)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_parquet.with_suffix(".tmp.parquet")
    pq.write_table(output_table, temporary)
    os.replace(temporary, output_parquet)
    return episode_index, expected_length


def convert_dataset(
    source_root: Path,
    output_root: Path,
    workers: int,
    max_episodes: int | None,
) -> None:
    if source_root.resolve() == output_root.resolve():
        raise ValueError("source and output must be different directories")

    info = _load_json(source_root / "meta" / "info.json")
    episodes = _load_jsonl(source_root / "meta" / "episodes.jsonl")
    validate_source_info(info)
    if max_episodes is not None:
        if max_episodes < 1:
            raise ValueError("max_episodes must be positive")
        episodes = episodes[:max_episodes]

    output_info = build_output_info(info, len(episodes))
    output_info["total_frames"] = sum(int(row["length"]) for row in episodes)

    jobs = [
        {
            "source_root": str(source_root),
            "output_root": str(output_root),
            "info": info,
            "episode_index": row["episode_index"],
            "length": row["length"],
        }
        for row in episodes
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for completed, (episode_index, _) in enumerate(executor.map(_convert_episode, jobs), 1):
            if completed % 50 == 0 or completed == len(jobs):
                print(
                    f"[gr00t-convert] {completed}/{len(jobs)} episodes "
                    f"(latest={episode_index})",
                    flush=True,
                )

    _write_json(output_root / "meta" / "info.json", output_info)
    _write_json(output_root / "meta" / "modality.json", LIBERO_MODALITY)
    _write_jsonl(output_root / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(
        output_root / "meta" / "tasks.jsonl",
        _load_jsonl(source_root / "meta" / "tasks.jsonl"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-episodes", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_dataset(args.source, args.output, args.workers, args.max_episodes)


if __name__ == "__main__":
    main()
