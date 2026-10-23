#!/usr/bin/env python3
"""Merge a GR00T PEFT adapter and verify adapter/merged layer parity."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from peft.tuners.lora import LoraLayer
from transformers import AutoModel

import gr00t.model  # noqa: F401 - registers the GR00T AutoModel class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--absolute-tolerance", type=float, default=0.005)
    return parser.parse_args()


def _module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"merged model is missing module {name}")
    return modules[name]


def _copy_checkpoint_metadata(source: Path, destination: Path) -> None:
    allowed = {
        "experiment_cfg",
        "processor",
        "modality.json",
        "processor_config.json",
        "preprocessor_config.json",
    }
    for path in source.iterdir():
        if path.name not in allowed:
            continue
        target = destination / path.name
        if target.exists():
            continue
        if path.is_dir():
            shutil.copytree(path, target)
        elif path.is_file():
            shutil.copy2(path, target)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    base = AutoModel.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(args.device)
    adapter = PeftModel.from_pretrained(base, args.adapter_checkpoint).eval()

    probes: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    generator = torch.Generator(device=args.device).manual_seed(42)
    for name, module in adapter.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        sample = torch.randn(
            2,
            module.in_features,
            device=args.device,
            dtype=dtype,
            generator=generator,
        )
        with torch.inference_mode():
            probes[name.removeprefix("base_model.model.")] = (
                sample,
                module(sample).float().cpu(),
            )
    if not probes:
        raise RuntimeError("adapter contains no LoRA layers")

    merged = adapter.merge_and_unload(safe_merge=True).eval()
    comparisons = []
    for name, (sample, expected) in probes.items():
        with torch.inference_mode():
            actual = _module(merged, name)(sample).float().cpu()
        max_absolute_error = float((actual - expected).abs().max())
        comparisons.append(
            {
                "module": name,
                "max_absolute_error": max_absolute_error,
                "passed": max_absolute_error <= args.absolute_tolerance,
            }
        )

    passed = all(item["passed"] for item in comparisons)
    validation = {
        "adapter_checkpoint": str(args.adapter_checkpoint),
        "base_model_path": str(args.base_model_path),
        "absolute_tolerance": args.absolute_tolerance,
        "tested_module_count": len(comparisons),
        "maximum_absolute_error": max(
            item["max_absolute_error"] for item in comparisons
        ),
        "passed": passed,
        "comparisons": comparisons,
    }
    if not passed:
        raise RuntimeError(
            "adapter/merged parity exceeded tolerance: "
            f"{validation['maximum_absolute_error']}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    _copy_checkpoint_metadata(args.adapter_checkpoint, args.output_dir)
    (args.output_dir / "merge_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "LORA MERGE PASS "
        f"modules={len(comparisons)} max_abs={validation['maximum_absolute_error']:.6g}"
    )


if __name__ == "__main__":
    main()
