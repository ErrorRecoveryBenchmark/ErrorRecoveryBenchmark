#!/usr/bin/env python3
"""Run the official N1.6 finetuner with N1.5 action-head LoRA."""

from __future__ import annotations

import runpy
from pathlib import Path

import gr00t
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
import gr00t.experiment.experiment as experiment
from gr00t.model import MODEL_REGISTRY

from gr00t_n15_lora import apply_n15_action_head_lora


BasePipeline = MODEL_REGISTRY[Gr00tN1d6Config]


class N15LoraPipeline(BasePipeline):
    def _create_model(self):
        model = super()._create_model()
        return apply_n15_action_head_lora(model, self.save_cfg_dir)


MODEL_REGISTRY[Gr00tN1d6Config] = N15LoraPipeline

launcher = Path(gr00t.__file__).resolve().parent / "experiment" / "launch_finetune.py"
runpy.run_path(str(launcher), run_name="__main__")
