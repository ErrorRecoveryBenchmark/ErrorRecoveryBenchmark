#!/usr/bin/env python3
"""Verify and summarize the controlled Stack GR00T LoRA ablation."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import yaml


BASE = Path(os.environ.get("BENCHMARK_WORKSPACE", Path(__file__).resolve().parents[4]))
DEFAULT_ROOT = (
    Path(__file__).resolve().parents[3]
    / "error_benchmark/outputs/gr00t_n16"
    / "stack_lora_ablation_official_libero"
)
MODEL = BASE / "shared_deps/models/backup/GR00T-N1.6-3B"
DATASET = (
    Path(__file__).resolve().parents[3]
    / "lerobot_data/gr00t_paired/stack/nominal"
)
LOSS_PATTERN = re.compile(r"'train_loss': ([0-9.eE+-]+)")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _one(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} under {root}, found {matches}")
    return matches[0]


def _identical_json(root: Path, name: str) -> dict:
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"no {name} under {root}")
    values = [_load_json(path) for path in matches]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"conflicting {name} files under {root}: {matches}")
    return values[0]


def _value(tree, key: str):
    found = []

    def visit(value):
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                if child_key.replace("-", "_") == key:
                    found.append(child_value)
                visit(child_value)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(tree)
    if not found:
        raise KeyError(f"configuration has no {key}")
    if any(value != found[0] for value in found[1:]):
        raise RuntimeError(f"configuration has conflicting {key}: {found}")
    return found[0]


def _config(output: Path) -> dict:
    candidates = list(output.rglob("conf.yaml"))
    if not candidates:
        raise FileNotFoundError(f"no conf.yaml under {output}")
    candidates.sort(key=lambda path: (len(path.parts), str(path)))
    return yaml.safe_load(candidates[0].read_text(encoding="utf-8"))


def verify_config(root: Path) -> None:
    current_dir = root / "train/current"
    lora_dir = root / "train/n15_action_head_lora"
    configs = [_config(current_dir), _config(lora_dir)]
    for config in configs:
        expected = {
            "base_model_path": str(MODEL),
            "dataset_path": str(DATASET),
            "max_steps": 20000,
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "warmup_ratio": 0.05,
            "gradient_accumulation_steps": int(
                640 / int(_value(config, "global_batch_size"))
            ),
            "state_dropout_prob": 0.0,
            "num_gpus": 1,
        }
        for key, wanted in expected.items():
            actual = _value(config, key)
            if str(actual) != str(wanted):
                raise ValueError(f"{key}={actual!r}, expected {wanted!r}")
        effective = int(_value(config, "global_batch_size")) * int(
            _value(config, "gradient_accumulation_steps")
        )
        if effective != 640:
            raise ValueError(f"effective optimizer batch {effective} != 640")

    current = _load_json(_one(current_dir, "current_trainable.json"))
    lora = _load_json(_one(lora_dir, "historical_lora.json"))
    if current["cuda_visible_devices"] != "0" or current["world_size"] != 1:
        raise ValueError(f"current GPU mapping is wrong: {current}")
    if lora["cuda_visible_devices"] != "4" or lora["world_size"] != 1:
        raise ValueError(f"LoRA GPU mapping is wrong: {lora}")
    if not current["trainable_parameter_names"]:
        raise ValueError("current recipe has no trainable parameters")
    if not all("lora_" in name for name in lora["trainable_parameter_names"]):
        raise ValueError("historical arm has trainable non-LoRA parameters")
    for key, wanted in {
        "action_head_only": True,
        "rank": 64,
        "alpha": 128,
        "dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "modules_to_save": None,
    }.items():
        if lora[key] != wanted:
            raise ValueError(f"historical LoRA {key}={lora[key]!r}, expected {wanted!r}")
    if not lora["target_modules"] or not all(
        "action_head" in name for name in lora["target_modules"]
    ):
        raise ValueError("historical LoRA targets are not action-head-only")
    print("CONFIG A/B PASS")


def verify_probe(root: Path, micro_batch: int, attempt: str) -> None:
    probe = root / f"probes/micro_{micro_batch}" / attempt
    for marker in ("current.ok", "n15_action_head_lora.ok", "probe.ok"):
        path = probe / "status" / marker
        if not path.is_file():
            raise FileNotFoundError(path)

    manifests = (
        (_identical_json(probe / "current", "current_trainable.json"), "0"),
        (
            _identical_json(
                probe / "n15_action_head_lora", "historical_lora.json"
            ),
            "4",
        ),
    )
    for manifest, gpu in manifests:
        if manifest["cuda_visible_devices"] != gpu or manifest["world_size"] != 1:
            raise ValueError(f"single-GPU manifest mismatch: {manifest}")

    for arm in ("current", "n15_action_head_lora"):
        checkpoint = probe / arm / "checkpoint-2"
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
        log = (probe / "logs" / f"{arm}.log").read_text(
            encoding="utf-8", errors="replace"
        )
        matches = LOSS_PATTERN.findall(log)
        if "Training completed!" not in log or not matches:
            raise ValueError(f"{arm} has no completed finite-loss record")
        loss = float(matches[-1])
        if not math.isfinite(loss):
            raise ValueError(f"{arm} train_loss is non-finite: {loss}")
        print(f"{arm}: checkpoint-2 train_loss={loss:.7f}")
    print("SINGLE-GPU A/B PROBE PASS")


def verify_checkpoints(root: Path) -> None:
    current = root / "train/current/checkpoint-20000"
    adapter = root / "train/n15_action_head_lora/checkpoint-20000"
    merged = root / "train/n15_action_head_lora_merged/checkpoint-20000"
    if not current.is_dir():
        raise FileNotFoundError(current)
    if not adapter.is_dir() or not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"raw adapter checkpoint incomplete: {adapter}")
    if not merged.is_dir():
        raise FileNotFoundError(merged)
    validation = _load_json(merged / "merge_validation.json")
    if not validation["passed"] or validation["tested_module_count"] < 1:
        raise ValueError(f"merge parity failed: {validation}")
    if not list(current.glob("*.safetensors")):
        raise FileNotFoundError(f"current full checkpoint weights missing: {current}")
    if not list(merged.glob("*.safetensors")):
        raise FileNotFoundError(f"merged full checkpoint weights missing: {merged}")
    print("CHECKPOINTS 20000/20000 PASS")


def _clean_result(path: Path, expected: int) -> dict:
    result = _load_json(path)["clean_rollouts"]
    if int(result["total"]) != expected:
        raise ValueError(f"{path}: total {result['total']} != {expected}")
    if len(result["per_rollout"]) != expected:
        raise ValueError(f"{path}: incomplete per-rollout records")
    return result


def verify_evaluation(root: Path, expected: int) -> None:
    current = _clean_result(root / "evaluations/current/stack.json", expected)
    lora = _clean_result(
        root / "evaluations/n15_action_head_lora_merged/stack.json", expected
    )
    current_initial = {
        int(row["rollout_id"]): row["initial_observation_sha256"]
        for row in current["per_rollout"]
    }
    lora_initial = {
        int(row["rollout_id"]): row["initial_observation_sha256"]
        for row in lora["per_rollout"]
    }
    if current_initial != lora_initial:
        raise ValueError("the two arms did not receive identical initial observations")
    print(f"MATCHED CLEAN EVALUATION {expected}/{expected} PASS")


def _failure_counts(result: dict) -> dict:
    failures = Counter()
    for row in result["per_rollout"]:
        if row.get("success"):
            continue
        failures["runtime_error" if row.get("error") else "timeout_or_task_failure"] += 1
    return dict(failures)


def write_report(root: Path, expected: int) -> None:
    verify_evaluation(root, expected)
    rows = []
    for label, relative in (
        ("current", "evaluations/current/stack.json"),
        (
            "n15_action_head_lora",
            "evaluations/n15_action_head_lora_merged/stack.json",
        ),
    ):
        clean = _clean_result(root / relative, expected)
        rows.append(
            {
                "arm": label,
                "successes": int(clean["successes"]),
                "total": int(clean["total"]),
                "success_rate": float(clean["successes"]) / int(clean["total"]),
                "failure_stage_counts": _failure_counts(clean),
            }
        )
    report = {"task": "stack", "matched_seed_base": 20260724, "rows": rows}
    (root / "ablation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Stack nominal GR00T LoRA ablation",
        "",
        "| Arm | Success | Rate | Failures |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['successes']}/{row['total']} | "
            f"{100 * row['success_rate']:.1f}% | "
            f"{json.dumps(row['failure_stage_counts'], sort_keys=True)} |"
        )
    (root / "ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("STACK LORA ABLATION REPORT PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("probe", "config", "checkpoints", "evaluation", "report"),
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--expected-rollouts", type=int, default=20)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--probe-attempt", default="single_gpu_attempt_2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "probe":
        verify_probe(args.root, args.micro_batch, args.probe_attempt)
    elif args.mode == "config":
        verify_config(args.root)
    elif args.mode == "checkpoints":
        verify_checkpoints(args.root)
    elif args.mode == "evaluation":
        verify_evaluation(args.root, args.expected_rollouts)
    else:
        write_report(args.root, args.expected_rollouts)


if __name__ == "__main__":
    main()
