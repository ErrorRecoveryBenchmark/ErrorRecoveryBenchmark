#!/usr/bin/env python3
"""Smoke-test and run the paired twelve-policy OpenVLA-OFT campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from error_benchmark.scripts.training.prepare_openvla_oft_data import DATASETS


BASE = Path(os.environ.get("BENCHMARK_WORKSPACE", Path(__file__).resolve().parents[4]))
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OPENVLA_ROOT = BASE / "shared_deps" / "openvla-oft"
DEFAULT_PYTHON = Path(
    os.environ.get("OPENVLA_PYTHON", sys.executable)
)
DEFAULT_VLA_PATH = BASE / "models" / "openvla-7b"
DEFAULT_DATA_ROOT = BASE / "openvla_oft_data"
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_REPO_ROOT / "error_benchmark" / "outputs" / "openvla_oft"
)
GPUS = (3, 5, 6, 7)
LOSS_PATTERN = re.compile(r"OPENVLA_METRIC step=(\S+) loss=(\S+)")

OFFICIAL_PARAMETERS = {
    "vla_path": "openvla/openvla-7b",
    "use_l1_regression": True,
    "use_diffusion": False,
    "use_film": False,
    "num_images_in_input": 2,
    "use_proprio": True,
    "batch_size": 8,
    "grad_accumulation_steps": 8,
    "effective_batch_size": 64,
    "learning_rate": 5e-4,
    "lr_warmup_steps": 0,
    "num_steps_before_decay": 100000,
    "max_steps": 150005,
    "save_freq": 10000,
    "image_aug": True,
    "use_lora": True,
    "lora_rank": 32,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "lora_targets": "all-linear",
    "merge_lora_during_training": False,
    "action_chunk": 8,
    "action_dim": 7,
    "proprio_dim": 8,
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_pids(gpu: int) -> list[str]:
    process = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-compute-apps=pid",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in process.stdout.splitlines() if line.strip()]


def require_gpus_free(gpus: tuple[int, ...] = GPUS) -> None:
    occupied = {gpu: gpu_pids(gpu) for gpu in gpus}
    occupied = {gpu: pids for gpu, pids in occupied.items() if pids}
    if occupied:
        raise RuntimeError(f"approved GPU(s) are occupied: {occupied}")


def run_name(task: str, setting: str) -> str:
    return f"{task}_{setting}"


def training_command(
    python: Path,
    repo_root: Path,
    data_root: Path,
    output_root: Path,
    task: str,
    setting: str,
    *,
    vla_path: Path = DEFAULT_VLA_PATH,
    smoke: bool = False,
) -> list[str]:
    name = "smoke" if smoke else run_name(task, setting)
    batch_size = 1 if smoke else OFFICIAL_PARAMETERS["batch_size"]
    accumulation = 1 if smoke else OFFICIAL_PARAMETERS["grad_accumulation_steps"]
    max_steps = 1 if smoke else OFFICIAL_PARAMETERS["max_steps"]
    save_freq = 1 if smoke else OFFICIAL_PARAMETERS["save_freq"]
    run_root = output_root / ("smoke" if smoke else "checkpoints")
    spec = next(
        item for item in DATASETS if item.task == task and item.setting == setting
    )
    return [
        str(python.parent / "torchrun"),
        "--standalone",
        "--nnodes",
        "1",
        "--nproc-per-node",
        "1",
        str(repo_root / "error_benchmark/scripts/training/launch_openvla_oft.py"),
        "--vla_path",
        str(vla_path),
        "--data_root_dir",
        str(data_root),
        "--dataset_name",
        spec.tfds_name,
        "--run_root_dir",
        str(run_root),
        "--run_id_override",
        name,
        "--use_l1_regression",
        "True",
        "--use_diffusion",
        "False",
        "--use_film",
        "False",
        "--num_images_in_input",
        "2",
        "--use_proprio",
        "True",
        "--batch_size",
        str(batch_size),
        "--grad_accumulation_steps",
        str(accumulation),
        "--learning_rate",
        "0.0005",
        "--lr_warmup_steps",
        "0",
        "--num_steps_before_decay",
        "100000",
        "--max_steps",
        str(max_steps),
        "--save_freq",
        str(save_freq),
        "--save_latest_checkpoint_only",
        "False",
        "--image_aug",
        "True",
        "--use_lora",
        "True",
        "--lora_rank",
        "32",
        "--lora_dropout",
        "0.0",
        "--merge_lora_during_training",
        "False",
        "--shuffle_buffer_size",
        "100000",
        "--wandb_log_freq",
        "1" if smoke else "10",
        "--wandb_entity",
        "disabled",
        "--wandb_project",
        "recoverbench-openvla-oft",
    ]


def training_env(repo_root: Path, openvla_root: Path, gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["OPENVLA_OFT_ROOT"] = str(openvla_root)
    env["WANDB_MODE"] = "offline"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), str(openvla_root), env.get("PYTHONPATH", "")]
    )
    return env


def finite_losses(log_path: Path) -> list[float]:
    if not log_path.is_file():
        return []
    values = [
        float(match.group(2))
        for match in LOSS_PATTERN.finditer(log_path.read_text(errors="replace"))
    ]
    return [value for value in values if math.isfinite(value)]


def smoke_checkpoint(output_root: Path) -> Path:
    return output_root / "smoke" / "smoke--1_chkpt"


def pid_running(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    try:
        return stat.read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def completed_training_log(text: str) -> bool:
    return f"Max step {OFFICIAL_PARAMETERS['max_steps']} reached!" in text


def run_smoke(args: argparse.Namespace) -> None:
    require_gpus_free((args.smoke_gpu,))
    output = args.output_root
    log_path = output / "logs" / "smoke.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = training_command(
        args.python,
        args.repo_root,
        args.data_root,
        output,
        "pick_place",
        "nominal",
        vla_path=args.vla_path,
        smoke=True,
    )
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=args.openvla_root,
            env=training_env(args.repo_root, args.openvla_root, args.smoke_gpu),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    checkpoint = smoke_checkpoint(output)
    required = (
        checkpoint / "lora_adapter" / "adapter_model.safetensors",
        checkpoint / "action_head--1_checkpoint.pt",
        checkpoint / "proprio_projector--1_checkpoint.pt",
        checkpoint / "dataset_statistics.json",
    )
    losses = finite_losses(log_path)
    if process.returncode or not losses or any(not path.is_file() for path in required):
        raise RuntimeError(
            f"smoke failed: exit={process.returncode}, losses={losses}, "
            f"missing={[str(path) for path in required if not path.is_file()]}"
        )
    atomic_json(
        output / "status" / "smoke.json",
        {
            "status": "pass",
            "gpu": args.smoke_gpu,
            "command": command,
            "finite_losses": losses,
            "checkpoint": str(checkpoint),
            "completed_at": utc_now(),
        },
    )
    print("OPENVLA SMOKE PASS")


def campaign_manifest(args: argparse.Namespace) -> dict[str, Any]:
    runs = []
    for index, spec in enumerate(DATASETS):
        runs.append(
            {
                "index": index,
                "name": run_name(spec.task, spec.setting),
                "task": spec.task,
                "setting": spec.setting,
                "dataset_name": spec.tfds_name,
                "gpu_slot": GPUS[index % len(GPUS)],
                "state": "queued",
            }
        )
    return {
        "created_at": utc_now(),
        "gpus": list(GPUS),
        "parameters": OFFICIAL_PARAMETERS,
        "runs": runs,
        "repo_root": str(args.repo_root),
        "openvla_root": str(args.openvla_root),
        "vla_path": str(args.vla_path),
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
    }


def launch_campaign(args: argparse.Namespace, *, resume: bool = False) -> None:
    if not (args.output_root / "status" / "smoke.json").is_file():
        raise FileNotFoundError("smoke marker is missing")
    manifest_file = args.output_root / "campaign_manifest.json"
    status_root = args.output_root / "status"
    log_root = args.output_root / "logs"
    status_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    active: dict[
        int, tuple[dict[str, Any], subprocess.Popen | None, Any | None]
    ] = {}

    if resume:
        with manifest_file.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("parameters") != OFFICIAL_PARAMETERS:
            raise ValueError("existing campaign parameters differ from approved recipe")
        manifest["gpus"] = list(GPUS)
        atomic_json(manifest_file, manifest)
        pending = []
        adopted = []
        for manifest_run in manifest["runs"]:
            status_file = status_root / f"{manifest_run['name']}.json"
            with status_file.open(encoding="utf-8") as handle:
                run = json.load(handle)
            if run["state"] == "queued":
                pending.append(run)
            elif run["state"] == "running":
                gpu = int(run["gpu"])
                pid = int(run["pid"])
                if gpu not in GPUS:
                    raise ValueError(f"running job uses unapproved GPU {gpu}")
                if gpu in active:
                    raise ValueError(f"multiple running jobs claim GPU {gpu}")
                if not pid_running(pid):
                    raise RuntimeError(f"cannot adopt inactive PID {pid}")
                active[gpu] = (run, None, None)
                adopted.append({"name": run["name"], "gpu": gpu, "pid": pid})
            elif run["state"] != "completed":
                raise RuntimeError(
                    f"cannot resume {run['name']} in state {run['state']}"
                )
        atomic_json(
            status_root / "scheduler_handoff.json",
            {"resumed_at": utc_now(), "gpus": list(GPUS), "adopted": adopted},
        )
    else:
        require_gpus_free()
        manifest = campaign_manifest(args)
        atomic_json(manifest_file, manifest)
        for run in manifest["runs"]:
            atomic_json(status_root / f"{run['name']}.json", run)
        pending = list(manifest["runs"])

    failed = False
    while pending or active:
        for gpu in GPUS:
            if failed or gpu in active or not pending:
                continue
            if gpu_pids(gpu):
                failed = True
                atomic_json(
                    status_root / "campaign_failed.json",
                    {"reason": f"GPU {gpu} became occupied", "failed_at": utc_now()},
                )
                break
            run = pending.pop(0)
            command = training_command(
                args.python,
                args.repo_root,
                args.data_root,
                args.output_root,
                run["task"],
                run["setting"],
                vla_path=args.vla_path,
            )
            log_handle = (log_root / f"{run['name']}.log").open(
                "a", encoding="utf-8"
            )
            process = subprocess.Popen(
                command,
                cwd=args.openvla_root,
                env=training_env(args.repo_root, args.openvla_root, gpu),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            run.update(
                {
                    "state": "running",
                    "gpu": gpu,
                    "pid": process.pid,
                    "command": command,
                    "started_at": utc_now(),
                }
            )
            atomic_json(status_root / f"{run['name']}.json", run)
            active[gpu] = (run, process, log_handle)

        if failed and not active:
            raise RuntimeError("campaign stopped after a failure")
        time.sleep(30)
        for gpu, item in list(active.items()):
            run, process, log_handle = item
            if process is None:
                if pid_running(int(run["pid"])):
                    continue
                log_path = log_root / f"{run['name']}.log"
                returncode = (
                    0
                    if completed_training_log(log_path.read_text(errors="replace"))
                    else 1
                )
            else:
                returncode = process.poll()
                if returncode is None:
                    continue
            if log_handle is not None:
                log_handle.close()
            del active[gpu]
            run.update(
                {
                    "state": "completed" if returncode == 0 else "failed",
                    "returncode": returncode,
                    "finished_at": utc_now(),
                }
            )
            atomic_json(status_root / f"{run['name']}.json", run)
            if returncode:
                failed = True
                atomic_json(
                    status_root / "campaign_failed.json",
                    {
                        "run": run["name"],
                        "returncode": returncode,
                        "failed_at": utc_now(),
                    },
                )
        if failed:
            pending.clear()

    if not failed:
        (args.output_root / "campaign_complete").touch()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("smoke", "campaign", "resume"), required=True
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--openvla-root", type=Path, default=DEFAULT_OPENVLA_ROOT)
    parser.add_argument("--vla-path", type=Path, default=DEFAULT_VLA_PATH)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--smoke-gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        run_smoke(args)
    else:
        launch_campaign(args, resume=args.mode == "resume")


if __name__ == "__main__":
    main()
