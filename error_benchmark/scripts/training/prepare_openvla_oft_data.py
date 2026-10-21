#!/usr/bin/env python3
"""Convert RecoverBench LeRobot v2.1 episodes into OpenVLA-OFT RLDS data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATASET_NAME = "recover_bench_open_vla"
DATASET_VERSION = "1.0.0"
DEFAULT_LEROBOT_ROOT = Path(
    os.environ.get("LEROBOT_DATA_ROOT", Path.home() / ".cache/huggingface/lerobot/benchmark")
)
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("OPENVLA_DATA_ROOT", Path(__file__).resolve().parents[3] / "data/openvla_oft")
)


@dataclass(frozen=True)
class DatasetSpec:
    task: str
    setting: str
    source_dir: str
    expected_episodes: int

    @property
    def config_name(self) -> str:
        return f"{self.task}_{self.setting}"

    @property
    def tfds_name(self) -> str:
        return f"{DATASET_NAME}/{self.config_name}"


DATASETS = (
    DatasetSpec("pick_place", "nominal", "mimicgen_pick_place_merged", 82),
    DatasetSpec(
        "pick_place",
        "nominal_recovery",
        "mimicgen_pick_place_merged_error_training",
        250,
    ),
    DatasetSpec("coffee", "nominal", "mimicgen_coffee_merged", 2000),
    DatasetSpec(
        "coffee",
        "nominal_recovery",
        "mimicgen_coffee_merged_error_training",
        2085,
    ),
    DatasetSpec("stack", "nominal", "mimicgen_stack_merged", 2000),
    DatasetSpec(
        "stack",
        "nominal_recovery",
        "mimicgen_stack_merged_error_training",
        2175,
    ),
    DatasetSpec("stack_three", "nominal", "mimicgen_stack_three_merged", 2000),
    DatasetSpec(
        "stack_three",
        "nominal_recovery",
        "mimicgen_stack_three_merged_error_training",
        2111,
    ),
    DatasetSpec("threading", "nominal", "mimicgen_threading_merged", 1000),
    DatasetSpec(
        "threading",
        "nominal_recovery",
        "mimicgen_threading_merged_error_training",
        1032,
    ),
    DatasetSpec(
        "three_piece_assembly",
        "nominal",
        "mimicgen_three_piece_assembly_merged",
        2000,
    ),
    DatasetSpec(
        "three_piece_assembly",
        "nominal_recovery",
        "mimicgen_three_piece_assembly_merged_error_training",
        2168,
    ),
)
DATASETS_BY_CONFIG = {spec.config_name: spec for spec in DATASETS}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def source_fingerprint(source: Path) -> str:
    digest = hashlib.sha256()
    for relative in ("meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl"):
        path = source / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_source(spec: DatasetSpec, source: Path) -> dict[str, Any]:
    info = load_json(source / "meta" / "info.json")
    episodes = load_jsonl(source / "meta" / "episodes.jsonl")
    tasks = load_jsonl(source / "meta" / "tasks.jsonl")
    features = info.get("features", {})
    expected = {
        "image": ("image", [84, 84, 3]),
        "wrist_image": ("image", [84, 84, 3]),
        "state": ("float32", [8]),
        "actions": ("float32", [7]),
    }
    for key, (dtype, shape) in expected.items():
        actual = features.get(key, {})
        if actual.get("dtype") != dtype or actual.get("shape") != shape:
            raise ValueError(f"{spec.config_name}: invalid {key} feature: {actual}")
    if int(info.get("fps", -1)) != 20:
        raise ValueError(f"{spec.config_name}: fps must be 20")
    if len(episodes) != spec.expected_episodes:
        raise ValueError(
            f"{spec.config_name}: {len(episodes)} episodes != {spec.expected_episodes}"
        )
    if len(tasks) != 1 or not tasks[0].get("task"):
        raise ValueError(f"{spec.config_name}: expected one non-empty task")
    return {
        "info": info,
        "episodes": episodes,
        "instruction": tasks[0]["task"],
        "total_frames": sum(int(row["length"]) for row in episodes),
    }


def action_chunk_indices(length: int, frame_index: int, horizon: int = 8) -> list[int]:
    """Return the official end-padded future-action indices for one episode."""
    if length < 1 or not 0 <= frame_index < length or horizon < 1:
        raise ValueError("invalid action chunk request")
    return [min(frame_index + offset, length - 1) for offset in range(horizon)]


def _image_bytes(value: dict[str, Any]) -> bytes:
    payload = value.get("bytes")
    if payload is not None:
        return payload
    path = value.get("path")
    if not path:
        raise ValueError("embedded image has neither bytes nor path")
    return Path(path).read_bytes()


def register_tfds_builder(lerobot_root: Path = DEFAULT_LEROBOT_ROOT):
    """Register the TFDS builder used both for preparation and file-backed loading."""
    import numpy as np
    import tensorflow_datasets as tfds

    configs = [
        tfds.core.BuilderConfig(name=spec.config_name, version=DATASET_VERSION)
        for spec in DATASETS
    ]

    class RecoverBenchOpenVLA(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version(DATASET_VERSION)
        BUILDER_CONFIGS = configs

        def _info(self):
            return tfds.core.DatasetInfo(
                builder=self,
                features=tfds.features.FeaturesDict(
                    {
                        "steps": tfds.features.Dataset(
                            {
                                "observation": tfds.features.FeaturesDict(
                                    {
                                        "image": tfds.features.Image(
                                            shape=(84, 84, 3),
                                            dtype=np.uint8,
                                            encoding_format="png",
                                        ),
                                        "wrist_image": tfds.features.Image(
                                            shape=(84, 84, 3),
                                            dtype=np.uint8,
                                            encoding_format="png",
                                        ),
                                        "state": tfds.features.Tensor(
                                            shape=(8,), dtype=np.float32
                                        ),
                                    }
                                ),
                                "action": tfds.features.Tensor(
                                    shape=(7,), dtype=np.float32
                                ),
                                "discount": tfds.features.Scalar(dtype=np.float32),
                                "reward": tfds.features.Scalar(dtype=np.float32),
                                "is_first": tfds.features.Scalar(dtype=np.bool_),
                                "is_last": tfds.features.Scalar(dtype=np.bool_),
                                "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                                "language_instruction": tfds.features.Text(),
                            }
                        ),
                        "episode_metadata": tfds.features.FeaturesDict(
                            {
                                "episode_index": tfds.features.Scalar(dtype=np.int64),
                                "source_path": tfds.features.Text(),
                            }
                        ),
                    }
                )
            )

        def _split_generators(self, dl_manager):
            del dl_manager
            spec = DATASETS_BY_CONFIG[self.builder_config.name]
            source = Path(lerobot_root) / spec.source_dir
            return {
                "train": self._generate_examples(
                    source=source,
                    spec=spec,
                )
            }

        def _generate_examples(self, source: Path, spec: DatasetSpec):
            import numpy as np
            import pyarrow.parquet as pq

            metadata = validate_source(spec, source)
            info = metadata["info"]
            instruction = metadata["instruction"]
            chunk_size = int(info["chunks_size"])
            for episode in metadata["episodes"]:
                episode_index = int(episode["episode_index"])
                path = source / info["data_path"].format(
                    episode_chunk=episode_index // chunk_size,
                    episode_index=episode_index,
                )
                table = pq.read_table(
                    path,
                    columns=["image", "wrist_image", "state", "actions"],
                )
                expected_length = int(episode["length"])
                if table.num_rows != expected_length:
                    raise ValueError(
                        f"{spec.config_name}/{episode_index}: "
                        f"{table.num_rows} rows != {expected_length}"
                    )
                images = table.column("image").to_pylist()
                wrist_images = table.column("wrist_image").to_pylist()
                states = table.column("state").to_pylist()
                actions = table.column("actions").to_pylist()
                steps = []
                for index in range(expected_length):
                    last = index == expected_length - 1
                    steps.append(
                        {
                            "observation": {
                                "image": _image_bytes(images[index]),
                                "wrist_image": _image_bytes(wrist_images[index]),
                                "state": np.asarray(states[index], dtype=np.float32),
                            },
                            "action": np.asarray(actions[index], dtype=np.float32),
                            "discount": np.float32(1.0),
                            "reward": np.float32(last),
                            "is_first": index == 0,
                            "is_last": last,
                            "is_terminal": last,
                            "language_instruction": instruction,
                        }
                    )
                yield f"{spec.config_name}-{episode_index:06d}", {
                    "steps": steps,
                    "episode_metadata": {
                        "episode_index": episode_index,
                        "source_path": str(path),
                    },
                }

    return RecoverBenchOpenVLA


def builder_data_dir(output_root: Path, spec: DatasetSpec) -> Path:
    return output_root / DATASET_NAME / spec.config_name / DATASET_VERSION


def manifest_path(output_root: Path, spec: DatasetSpec) -> Path:
    return output_root / "manifests" / f"{spec.config_name}.json"


def prepared_manifest(
    spec: DatasetSpec, lerobot_root: Path, output_root: Path
) -> dict[str, Any]:
    source = lerobot_root / spec.source_dir
    metadata = validate_source(spec, source)
    data_dir = builder_data_dir(output_root, spec)
    if not (data_dir / "dataset_info.json").is_file():
        raise FileNotFoundError(data_dir / "dataset_info.json")
    shards = sorted(data_dir.glob("*.tfrecord-*"))
    if not shards or any(path.stat().st_size == 0 for path in shards):
        raise ValueError(f"{spec.config_name}: missing or empty TFRecord shards")
    return {
        "dataset_name": spec.tfds_name,
        "config_name": spec.config_name,
        "task": spec.task,
        "setting": spec.setting,
        "source": str(source),
        "source_fingerprint": source_fingerprint(source),
        "episodes": spec.expected_episodes,
        "frames": metadata["total_frames"],
        "instruction": metadata["instruction"],
        "fps": 20,
        "image_shape": [84, 84, 3],
        "state_dim": 8,
        "action_dim": 7,
        "action_chunk": 8,
        "builder_data_dir": str(data_dir),
        "tfrecord_shards": [str(path) for path in shards],
        "prepared_at_unix": int(time.time()),
    }


def prepare_one(spec: DatasetSpec, lerobot_root: Path, output_root: Path) -> None:
    existing = manifest_path(output_root, spec)
    if existing.is_file():
        manifest = load_json(existing)
        source = lerobot_root / spec.source_dir
        if (
            manifest.get("source_fingerprint") == source_fingerprint(source)
            and Path(manifest.get("builder_data_dir", "")).is_dir()
        ):
            print(f"[openvla-rlds] {spec.config_name} already prepared", flush=True)
            return

    builder_class = register_tfds_builder(lerobot_root)
    builder = builder_class(
        config=spec.config_name,
        data_dir=str(output_root),
    )
    builder.download_and_prepare()
    manifest = prepared_manifest(spec, lerobot_root, output_root)
    atomic_json(existing, manifest)
    print(
        f"[openvla-rlds] {spec.config_name}: "
        f"{manifest['episodes']} episodes, {manifest['frames']} frames",
        flush=True,
    )


def prepare_all(
    lerobot_root: Path,
    output_root: Path,
    parallel: int,
) -> None:
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pending = list(DATASETS)
    active: list[tuple[DatasetSpec, subprocess.Popen, Any]] = []
    failed = False
    while pending or active:
        while pending and len(active) < parallel and not failed:
            spec = pending.pop(0)
            log_handle = (logs / f"{spec.config_name}.log").open("a", encoding="utf-8")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--dataset",
                spec.config_name,
                "--lerobot-root",
                str(lerobot_root),
                "--output-root",
                str(output_root),
            ]
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active.append((spec, process, log_handle))
        time.sleep(2)
        for item in list(active):
            spec, process, log_handle = item
            status = process.poll()
            if status is None:
                continue
            log_handle.close()
            active.remove(item)
            if status:
                failed = True
                print(
                    f"[openvla-rlds] {spec.config_name} failed with {status}",
                    file=sys.stderr,
                )
        if failed and not active:
            raise RuntimeError("RLDS preparation stopped after a failed dataset")
    (output_root / "all_ready").touch()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=sorted(DATASETS_BY_CONFIG))
    group.add_argument("--all", action="store_true")
    group.add_argument("--check", action="store_true")
    parser.add_argument("--lerobot-root", type=Path, default=DEFAULT_LEROBOT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--parallel", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.parallel < 1:
        raise ValueError("--parallel must be positive")
    if args.dataset:
        prepare_one(DATASETS_BY_CONFIG[args.dataset], args.lerobot_root, args.output_root)
    elif args.all:
        prepare_all(args.lerobot_root, args.output_root, args.parallel)
    else:
        for spec in DATASETS:
            prepared_manifest(spec, args.lerobot_root, args.output_root)
        print("OPENVLA DATA 12/12 PASS")


if __name__ == "__main__":
    main()
