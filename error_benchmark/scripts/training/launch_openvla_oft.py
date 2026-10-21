#!/usr/bin/env python3
"""Register RecoverBench RLDS data, then delegate to official OpenVLA-OFT."""

from __future__ import annotations

import math
import os
import runpy
import sys
from pathlib import Path

from error_benchmark.scripts.training.prepare_openvla_oft_data import (
    DATASETS,
    register_tfds_builder,
)


def register_datasets() -> None:
    register_tfds_builder()
    from prismatic.vla.datasets.rlds.oxe.configs import (
        OXE_DATASET_CONFIGS,
        ActionEncoding,
        StateEncoding,
    )
    from prismatic.vla.datasets.rlds.oxe.transforms import (
        OXE_STANDARDIZATION_TRANSFORMS,
        libero_dataset_transform,
    )

    for spec in DATASETS:
        OXE_DATASET_CONFIGS[spec.tfds_name] = {
            "image_obs_keys": {
                "primary": "image",
                "secondary": None,
                "wrist": "wrist_image",
            },
            "depth_obs_keys": {
                "primary": None,
                "secondary": None,
                "wrist": None,
            },
            "state_obs_keys": ["EEF_state", "gripper_state"],
            "state_encoding": StateEncoding.POS_EULER,
            "action_encoding": ActionEncoding.EEF_POS,
        }
        OXE_STANDARDIZATION_TRANSFORMS[spec.tfds_name] = libero_dataset_transform


def patch_metric_logging() -> None:
    import wandb

    original_init = wandb.init

    def checked_init(*args, **kwargs):
        run = original_init(*args, **kwargs)
        original_log = wandb.log

        def checked_log(data, *log_args, **log_kwargs):
            loss = data.get("VLA Train/Loss")
            if loss is not None:
                value = float(loss)
                step = log_kwargs.get("step")
                print(
                    f"OPENVLA_METRIC step={step} loss={value:.10g}",
                    flush=True,
                )
                if not math.isfinite(value):
                    raise FloatingPointError(
                        f"non-finite OpenVLA loss at step {step}: {value}"
                    )
            return original_log(data, *log_args, **log_kwargs)

        wandb.log = checked_log
        return run

    wandb.init = checked_init


def main() -> None:
    root = Path(os.environ["OPENVLA_OFT_ROOT"]).resolve()
    if not (root / "vla-scripts" / "finetune.py").is_file():
        raise FileNotFoundError(f"invalid OPENVLA_OFT_ROOT: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    register_datasets()
    patch_metric_logging()
    runpy.run_path(str(root / "vla-scripts" / "finetune.py"), run_name="__main__")


if __name__ == "__main__":
    main()
