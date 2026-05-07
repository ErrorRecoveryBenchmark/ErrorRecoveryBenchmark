#!/usr/bin/env python3
"""Compute OpenPI norm stats for Pi0.5 LeRobot datasets."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pyarrow.parquet as pq
import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from openpi.shared import normalize
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--assets-base-dir", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--openpi-loader",
        action="store_true",
        help="Use OpenPI's full dataloader path. The default parquet path is equivalent for LeRobotLiberoDataConfig without extra delta actions and avoids loading images.",
    )
    return parser.parse_args()


def validate_saved_stats(output_path: Path) -> None:
    stats_path = output_path / "norm_stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"norm stats file was not written: {stats_path}")

    stats = normalize.load(output_path)
    for key in ("state", "actions"):
        if key not in stats:
            raise ValueError(f"missing '{key}' in {stats_path}")
        if np.asarray(stats[key].mean).size == 0:
            raise ValueError(f"empty '{key}' mean in {stats_path}")
        if np.asarray(stats[key].std).size == 0:
            raise ValueError(f"empty '{key}' std in {stats_path}")


def _episode_parquet_path(meta: LeRobotDatasetMetadata, episode_index: int) -> Path:
    return meta.root / meta.get_data_file_path(ep_index=episode_index)


def _array_column(table, name: str, dtype=np.float32) -> np.ndarray:
    return np.asarray(table.column(name).to_pylist(), dtype=dtype)


def _action_chunks(actions: np.ndarray, action_horizon: int) -> np.ndarray:
    if action_horizon < 1:
        raise ValueError(f"action_horizon must be >= 1, got {action_horizon}")
    frame_indices = np.arange(actions.shape[0])[:, None]
    horizon_offsets = np.arange(action_horizon)[None, :]
    query_indices = np.minimum(frame_indices + horizon_offsets, actions.shape[0] - 1)
    return actions[query_indices]


def _can_use_fast_parquet(cfg) -> bool:
    data_factory = cfg.data
    if not isinstance(data_factory, _config.LeRobotLiberoDataConfig):
        return False
    if data_factory.extra_delta_transform:
        return False
    return True


def compute_norms_fast_parquet(repo_id: str, action_horizon: int, max_frames: int | None) -> dict:
    meta = LeRobotDatasetMetadata(repo_id)
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    consumed_frames = 0
    episode_indices = sorted(int(idx) for idx in meta.episodes)

    for episode_index in tqdm.tqdm(episode_indices, desc=f"Computing parquet stats for {repo_id}"):
        if max_frames is not None and consumed_frames >= max_frames:
            break

        table = pq.read_table(_episode_parquet_path(meta, episode_index), columns=["state", "actions"])
        states = _array_column(table, "state")
        actions = _array_column(table, "actions")
        if states.shape[0] != actions.shape[0]:
            raise ValueError(
                f"state/action length mismatch in episode {episode_index}: {states.shape[0]} != {actions.shape[0]}"
            )

        if max_frames is not None:
            remaining = max_frames - consumed_frames
            states = states[:remaining]
            actions = actions[:remaining]
        if states.size == 0:
            continue

        stats["state"].update(states)
        stats["actions"].update(_action_chunks(actions, action_horizon))
        consumed_frames += states.shape[0]

    if consumed_frames < 2:
        raise RuntimeError(f"dataset too small for norm stats: consumed_frames={consumed_frames}")
    print(f"Computed fast parquet stats from {consumed_frames} frames in {repo_id}")
    return {key: value.get_statistics() for key, value in stats.items()}


def compute_norms_openpi_loader(args, cfg, data_config) -> dict:
    dataset = _data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )

    if args.max_frames is not None and args.max_frames < len(dataset):
        num_batches = args.max_frames // cfg.batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // cfg.batch_size
        shuffle = False

    if num_batches < 1:
        raise RuntimeError(f"dataset too small for batch_size={cfg.batch_size}: len={len(dataset)}")

    data_iter = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm.tqdm(data_iter, total=num_batches, desc=f"Computing stats for {args.repo_id}"):
        for key in stats:
            stats[key].update(np.asarray(batch[key]))
    return {key: value.get_statistics() for key, value in stats.items()}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    cfg = _config.get_config(args.config_name)
    cfg = dataclasses.replace(
        cfg,
        assets_base_dir=args.assets_base_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data=dataclasses.replace(cfg.data, repo_id=args.repo_id),
    )
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    if data_config.repo_id is None:
        raise ValueError("data config must have a repo_id")
    if data_config.asset_id is None:
        raise ValueError("data config must have an asset_id")

    if args.openpi_loader:
        norm_stats = compute_norms_openpi_loader(args, cfg, data_config)
    else:
        if not _can_use_fast_parquet(cfg):
            raise RuntimeError("fast parquet stats are only valid for LeRobotLiberoDataConfig without extra delta actions; rerun with --openpi-loader")
        norm_stats = compute_norms_fast_parquet(data_config.repo_id, cfg.model.action_horizon, args.max_frames)

    output_path = Path(cfg.assets_dirs) / data_config.asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    validate_saved_stats(output_path)
    print(f"Validated norm stats: {output_path / 'norm_stats.json'}")


if __name__ == "__main__":
    main()
