#!/usr/bin/env python
"""Run OpenPI serve_policy.py with a localhost fallback for hostname lookup."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import runpy
import socket


_orig_gethostbyname = socket.gethostbyname


def _safe_gethostbyname(hostname: str) -> str:
    try:
        return _orig_gethostbyname(hostname)
    except socket.gaierror:
        return "127.0.0.1"


socket.gethostbyname = _safe_gethostbyname


def _patch_policy_asset_id() -> None:
    dataset_suffix = os.environ.get("PI05_POLICY_DATASET_SUFFIX")
    if not dataset_suffix:
        return

    from openpi.training import config as training_config

    original_get_config = training_config.get_config

    def get_config_with_policy_dataset(config_name: str):
        train_config = original_get_config(config_name)
        prefix = "pi05_benchmark_"
        suffix = "_merged_inference"
        if not (config_name.startswith(prefix) and config_name.endswith(suffix)):
            return train_config

        task = config_name[len(prefix) : -len(suffix)]
        repo_id = f"benchmark/mimicgen_{task}_{dataset_suffix}"
        assets_dir = os.environ.get("PI05_POLICY_ASSETS_DIR") or None
        assets = training_config.AssetsConfig(assets_dir=assets_dir, asset_id=repo_id)
        data = dataclasses.replace(train_config.data, repo_id=repo_id, assets=assets)
        return dataclasses.replace(train_config, data=data)

    training_config.get_config = get_config_with_policy_dataset


def _patch_policy_rng_seed() -> None:
    seed = os.environ.get("PI05_POLICY_RNG_SEED")
    if not seed:
        return

    seed_int = int(seed)

    import jax
    from openpi.policies import policy_config

    original_create_trained_policy = policy_config.create_trained_policy

    def create_trained_policy_with_rng(*args, **kwargs):
        policy = original_create_trained_policy(*args, **kwargs)
        if hasattr(policy, "_rng"):
            policy._rng = jax.random.key(seed_int)
        metadata = getattr(policy, "_metadata", None)
        if isinstance(metadata, dict):
            metadata["rng_seed"] = seed_int
        return policy

    policy_config.create_trained_policy = create_trained_policy_with_rng


_patch_policy_asset_id()
_patch_policy_rng_seed()

openpi_dir = Path(
    os.environ.get("OPENPI_DIR")
    or "${BENCHMARK_ROOT}/shared_deps/openpi"
)
runpy.run_path(str(openpi_dir / "scripts" / "serve_policy.py"), run_name="__main__")
