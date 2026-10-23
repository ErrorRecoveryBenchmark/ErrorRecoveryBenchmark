#!/usr/bin/env python3
"""GR00T N1.5-compatible action-head LoRA helpers for N1.6."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model


TARGET_NAME_PARTS = ("q_proj", "v_proj", "to_q", "to_v", "k_proj", "to_k")


def find_action_head_lora_targets(model: torch.nn.Module) -> list[str]:
    """Match the exact module-selection rule used by GR00T N1.5."""
    return [
        name
        for name, module in model.named_modules()
        if "action_head" in name
        and isinstance(module, torch.nn.Linear)
        and any(part in name for part in TARGET_NAME_PARTS)
    ]


def _wrap_forward(model):
    """Preserve the dict-style GR00T forward signature through PEFT."""

    def _forward(inputs):
        backbone_inputs, action_inputs = model.prepare_input(inputs)
        backbone_outputs = model.backbone(backbone_inputs)
        action_head_outputs = model.action_head(backbone_outputs, action_inputs)
        return action_head_outputs

    model.forward = _forward
    return model


def apply_n15_action_head_lora(
    model: torch.nn.Module,
    save_cfg_dir: Path,
    *,
    rank: int = 64,
    alpha: int = 128,
    dropout: float = 0.1,
):
    """Apply the official N1.5 default action-head-only PEFT recipe."""
    targets = find_action_head_lora_targets(model)
    if not targets:
        raise RuntimeError("N1.5 LoRA rule matched no N1.6 action-head Linear modules")

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=targets,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = _wrap_forward(get_peft_model(model, config))

    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    unexpected = [name for name in trainable if "lora_" not in name]
    if unexpected:
        raise RuntimeError(f"Non-LoRA parameters remain trainable: {unexpected[:20]}")

    trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_count = sum(param.numel() for param in model.parameters())
    model.print_trainable_parameters()

    if int(os.environ.get("RANK", "0")) == 0:
        manifest = {
            "source_release": "n1.5-release",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "action_head_only": True,
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "modules_to_save": None,
            "target_name_parts": list(TARGET_NAME_PARTS),
            "target_modules": targets,
            "target_module_count": len(targets),
            "trainable_parameter_names": trainable,
            "trainable_parameter_count": trainable_count,
            "total_parameter_count": total_count,
        }
        save_cfg_dir.mkdir(parents=True, exist_ok=True)
        (save_cfg_dir / "historical_lora.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    return model
