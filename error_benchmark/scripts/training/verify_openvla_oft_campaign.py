#!/usr/bin/env python3
"""Verify prepared data, smoke output, and the OpenVLA-OFT launch queue."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from error_benchmark.scripts.training.launch_openvla_oft_campaign import (
    DEFAULT_OUTPUT_ROOT,
    GPUS,
    OFFICIAL_PARAMETERS,
    finite_losses,
)
from error_benchmark.scripts.training.prepare_openvla_oft_data import (
    DATASETS,
    DEFAULT_LEROBOT_ROOT,
    DEFAULT_OUTPUT_ROOT as DEFAULT_DATA_ROOT,
    prepared_manifest,
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def verify_data(args: argparse.Namespace) -> None:
    for spec in DATASETS:
        expected = prepared_manifest(spec, args.lerobot_root, args.data_root)
        saved_path = args.data_root / "manifests" / f"{spec.config_name}.json"
        saved = load_json(saved_path)
        for key in (
            "dataset_name",
            "source_fingerprint",
            "episodes",
            "frames",
            "instruction",
            "state_dim",
            "action_dim",
            "action_chunk",
        ):
            if saved.get(key) != expected.get(key):
                raise ValueError(
                    f"{spec.config_name}: manifest {key} mismatch "
                    f"{saved.get(key)!r} != {expected.get(key)!r}"
                )
    print("OPENVLA DATA 12/12 PASS")


def verify_smoke(args: argparse.Namespace) -> None:
    marker = load_json(args.output_root / "status" / "smoke.json")
    if marker.get("status") != "pass":
        raise ValueError("smoke marker is not pass")
    losses = [float(value) for value in marker.get("finite_losses", [])]
    if not losses or not all(math.isfinite(value) for value in losses):
        raise ValueError(f"invalid smoke losses: {losses}")
    checkpoint = Path(marker["checkpoint"])
    required = (
        checkpoint / "lora_adapter" / "adapter_model.safetensors",
        checkpoint / "action_head--1_checkpoint.pt",
        checkpoint / "proprio_projector--1_checkpoint.pt",
        checkpoint / "dataset_statistics.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing smoke artifacts: {missing}")
    print("OPENVLA SMOKE PASS")


def verify_campaign(args: argparse.Namespace) -> None:
    manifest = load_json(args.output_root / "campaign_manifest.json")
    runs = manifest.get("runs", [])
    if len(runs) != args.expected_runs:
        raise ValueError(f"registered runs {len(runs)} != {args.expected_runs}")
    if manifest.get("parameters") != OFFICIAL_PARAMETERS:
        raise ValueError("campaign parameters differ from approved official recipe")
    if manifest.get("gpus") != list(GPUS):
        raise ValueError(f"campaign GPUs {manifest.get('gpus')} != {list(GPUS)}")

    statuses = []
    finite_active = 0
    finite_gpus = set()
    for run in runs:
        status = load_json(args.output_root / "status" / f"{run['name']}.json")
        statuses.append(status)
        if status.get("state") in {"running", "completed"}:
            log = args.output_root / "logs" / f"{run['name']}.log"
            if finite_losses(log):
                finite_active += 1
                finite_gpus.add(status.get("gpu"))
    if (args.output_root / "status" / "campaign_failed.json").exists():
        raise RuntimeError("campaign failure marker exists")
    if finite_active < args.require_active_or_complete:
        raise ValueError(
            f"only {finite_active} active/completed runs have finite loss; "
            f"need {args.require_active_or_complete}"
        )
    if not set(GPUS).issubset(finite_gpus):
        raise ValueError(f"finite-loss GPUs {finite_gpus} do not cover {set(GPUS)}")
    allowed = {"queued", "running", "completed"}
    invalid = [status for status in statuses if status.get("state") not in allowed]
    if invalid:
        raise ValueError(f"invalid campaign states: {invalid}")
    print("OPENVLA CAMPAIGN 12 RUNS PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("data", "smoke", "campaign"), required=True)
    parser.add_argument("--lerobot-root", type=Path, default=DEFAULT_LEROBOT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-runs", type=int, default=12)
    parser.add_argument("--require-active-or-complete", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "data":
        verify_data(args)
    elif args.mode == "smoke":
        verify_smoke(args)
    else:
        verify_campaign(args)


if __name__ == "__main__":
    main()
