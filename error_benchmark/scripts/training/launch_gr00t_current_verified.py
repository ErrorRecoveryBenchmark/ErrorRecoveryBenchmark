#!/usr/bin/env python3
"""Run the official N1.6 finetuner and record its trainable parameters."""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

import gr00t
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model import MODEL_REGISTRY


BasePipeline = MODEL_REGISTRY[Gr00tN1d6Config]


class VerifiedCurrentPipeline(BasePipeline):
    def _create_model(self):
        model = super()._create_model()
        trainable = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        visual_trainable = [name for name in trainable if "vision_model" in name]
        if visual_trainable:
            raise RuntimeError(
                f"Visual parameters unexpectedly trainable: {visual_trainable[:20]}"
            )

        if int(os.environ.get("RANK", "0")) == 0:
            trainable_count = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            total_count = sum(parameter.numel() for parameter in model.parameters())
            manifest = {
                "recipe": "gr00t-n1.6.1-current",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                "trainable_parameter_names": trainable,
                "trainable_parameter_count": trainable_count,
                "total_parameter_count": total_count,
            }
            self.save_cfg_dir.mkdir(parents=True, exist_ok=True)
            (self.save_cfg_dir / "current_trainable.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
        return model


MODEL_REGISTRY[Gr00tN1d6Config] = VerifiedCurrentPipeline

launcher = Path(gr00t.__file__).resolve().parent / "experiment" / "launch_finetune.py"
runpy.run_path(str(launcher), run_name="__main__")
