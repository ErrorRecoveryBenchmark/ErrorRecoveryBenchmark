#!/usr/bin/env python
"""
Pi0.5 LoRA Finetuning Pipeline — Merged (d0+d1) Datasets

Trains Pi0.5 (LoRA) on 6 tasks with merged d0+d1 MimicGen data.
Each task uses 1 GPU; a single Slurm job requests 6 GPUs.

Tasks:
  pick_place (d0 only), coffee (d0+d1), stack (d0+d1),
  stack_three (d0+d1), threading (d0+d1), three_piece_assembly (d0+d1)

Pipeline:
  1. convert-data  — Merge HDF5 d0+d1 → single LeRoBot dataset per task
  2. compute-norms — Normalization statistics
  3. train         — Submit single 6-GPU Slurm job
  4. eval          — VLA server rollout evaluation
  5. report        — Results table

Usage (convert-data and compute-norms under openpi05 conda env):
  conda run -n openpi05 python train_pi05_merged.py convert-data [--tasks coffee ...]
  conda run -n openpi05 python train_pi05_merged.py compute-norms [--tasks coffee ...]
  python train_pi05_merged.py train
  python train_pi05_merged.py eval [--tasks coffee ...]
  python train_pi05_merged.py report
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import h5py
import numpy as np

# ═══════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════

PROJECT_DIR = Path(
    os.environ.get("ERROR_RECOVERY_BENCHMARK_ROOT")
    or Path(__file__).resolve().parents[3]
)
DATASET_DIR = Path(
    os.environ.get("MIMICGEN_DATASET_CORE_DIR")
    or PROJECT_DIR.parent / "mimicgen_datasets" / "core"
)
OPENPI_DIR = Path(
    os.environ.get("OPENPI_DIR")
    or "${BENCHMARK_ROOT}/shared_deps/openpi"
)
OPENPI_TRAIN_SCRIPT = OPENPI_DIR / "scripts" / "train.py"
OPENPI_NORM_SCRIPT = OPENPI_DIR / "scripts" / "compute_norm_stats.py"
CONDA_DIR = Path(
    os.environ.get("CONDA_DIR")
    or Path.home() / "miniconda3"
)
LEROBOT_HOME = Path(os.environ.get("HF_LEROBOT_HOME",
                    os.path.expanduser("~/.cache/huggingface/lerobot")))
EVAL_RESULTS_FILE = PROJECT_DIR / "outputs" / "pi05_merged_eval_results.json"
MIMICGEN_PYTHON = CONDA_DIR / "envs" / "mimicgen_env" / "bin" / "python"

# ═══════════════════════════════════════════════════════════════
# Task definitions — merged d0+d1 per task
# ═══════════════════════════════════════════════════════════════

MERGED_TASKS = {
    "pick_place": {
        # No D1 variant in MimicGen — single D0 with 1000 demos
        "hdf5_files": ["pick_place_d0.hdf5"],
        "prompt": "pick up the milk, cereal, bread, and can and place them in the correct bins",
    },
    "coffee": {
        "hdf5_files": ["coffee_d0.hdf5", "coffee_d1.hdf5"],
        "prompt": "make coffee",
    },
    "stack": {
        "hdf5_files": ["stack_d0.hdf5", "stack_d1.hdf5"],
        "prompt": "stack the red block on top of the green block",
    },
    "stack_three": {
        "hdf5_files": ["stack_three_d0.hdf5", "stack_three_d1.hdf5"],
        "prompt": "stack three blocks",
    },
    "threading": {
        # D0 only — threading source not available for D1 generation
        "hdf5_files": ["threading_d0.hdf5"],
        "prompt": "insert the needle into the needle hole",
    },
    "three_piece_assembly": {
        "hdf5_files": ["three_piece_assembly_d0.hdf5", "three_piece_assembly_d1.hdf5"],
        "prompt": "assemble the three pieces",
    },
}

ALL_TASKS = list(MERGED_TASKS.keys())

# Robosuite task name for evaluation
EVAL_TASK_MAP = {
    "pick_place": "PickPlace_D0",
    "coffee": "Coffee_D0",
    "stack": "Stack_D0",
    "stack_three": "StackThree_D0",
    "threading": "Threading_D0",
    "three_piece_assembly": "ThreePieceAssembly_D0",
}

SLURM_PARTITION = "ai"

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _quat2axisangle(quat):
    """Convert quaternion (x, y, z, w) to axis-angle (3D)."""
    q = quat.copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def repo_id_for_task(task_name: str) -> str:
    return f"benchmark/mimicgen_{task_name}_merged"


def config_name_finetune(task_name: str) -> str:
    return f"pi05_benchmark_{task_name}_merged_finetune"


def config_name_inference(task_name: str) -> str:
    return f"pi05_benchmark_{task_name}_merged_inference"


def resolve_tasks(args) -> list:
    if args.tasks:
        for t in args.tasks:
            if t not in MERGED_TASKS:
                print(f"ERROR: Unknown task '{t}'. Available: {ALL_TASKS}")
                sys.exit(1)
        return args.tasks
    return ALL_TASKS


def log(msg: str):
    print(f"[pi05-merged] {msg}")


# ═══════════════════════════════════════════════════════════════
# Step 1: convert-data — Merge HDF5 d0+d1 → LeRoBot
# ═══════════════════════════════════════════════════════════════

def cmd_convert_data(args):
    """Merge HDF5 d0+d1 datasets into a single LeRoBot dataset per task."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    tasks = resolve_tasks(args)
    fps = 20

    for task_name in tasks:
        td = MERGED_TASKS[task_name]
        repo_id = repo_id_for_task(task_name)
        instruction = td["prompt"]

        # Check all HDF5 files exist
        h5_paths = []
        for fname in td["hdf5_files"]:
            p = DATASET_DIR / fname
            if not p.exists():
                log(f"SKIP {task_name}: HDF5 not found at {p}")
                break
            h5_paths.append(p)
        else:
            pass  # all files exist
        if len(h5_paths) != len(td["hdf5_files"]):
            continue

        output_path = LEROBOT_HOME / repo_id
        if output_path.exists():
            if args.overwrite:
                log(f"Removing existing dataset at {output_path}")
                shutil.rmtree(output_path)
            else:
                log(f"SKIP {task_name}: dataset exists at {output_path} (use --overwrite)")
                continue

        log(f"Converting {td['hdf5_files']} → {repo_id}")

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type="panda",
            fps=fps,
            features={
                "image": {"dtype": "image", "shape": (84, 84, 3),
                          "names": ["height", "width", "channel"]},
                "wrist_image": {"dtype": "image", "shape": (84, 84, 3),
                                "names": ["height", "width", "channel"]},
                "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
                "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
            },
            image_writer_threads=16,
            image_writer_processes=8,
        )

        total_demos = 0
        for h5_path in h5_paths:
            log(f"  Reading {h5_path.name}...")
            with h5py.File(str(h5_path), "r") as ff:
                container = ff["data"] if "data" in ff else ff
                demo_keys = sorted(container.keys())

                for demo_key in demo_keys:
                    demo = container[demo_key]
                    if "actions" not in demo or "obs" not in demo:
                        continue

                    obs = demo["obs"]
                    if "agentview_image" not in obs or "robot0_eye_in_hand_image" not in obs:
                        continue

                    actions = demo["actions"][()]
                    eef_pos = obs["robot0_eef_pos"][()]
                    eef_quat = obs["robot0_eef_quat"][()]
                    gripper_qpos = obs["robot0_gripper_qpos"][()] if "robot0_gripper_qpos" in obs else np.zeros((len(eef_pos), 2))

                    # Convert quaternions to axis-angles
                    eef_aa = np.zeros_like(eef_pos)
                    for i in range(len(eef_quat)):
                        eef_aa[i] = _quat2axisangle(eef_quat[i])

                    states = np.hstack((eef_pos, eef_aa, gripper_qpos))

                    agentview_imgs = obs["agentview_image"][()]
                    wrist_imgs = obs["robot0_eye_in_hand_image"][()]

                    data_length = min(len(actions), len(states),
                                      len(agentview_imgs), len(wrist_imgs))

                    for i in range(data_length):
                        dataset.add_frame({
                            "image": agentview_imgs[i],
                            "wrist_image": wrist_imgs[i],
                            "state": states[i].astype(np.float32),
                            "actions": actions[i].astype(np.float32),
                            "task": instruction,
                        })

                    dataset.save_episode()
                    total_demos += 1

                    if total_demos % 100 == 0:
                        log(f"  {task_name}: {total_demos} demos converted so far")

        log(f"  {task_name}: {total_demos} demos saved to {repo_id}")

    log("Data conversion complete.")


# ═══════════════════════════════════════════════════════════════
# Step 2: compute-norms
# ═══════════════════════════════════════════════════════════════

def cmd_compute_norms(args):
    """Compute normalization statistics for each merged task."""
    tasks = resolve_tasks(args)

    for task_name in tasks:
        cfg_name = config_name_finetune(task_name)
        log(f"Computing norm stats for {task_name} (config: {cfg_name})")

        cmd = [
            "conda", "run", "-n", "openpi05",
            "python", str(OPENPI_NORM_SCRIPT),
            "--config-name", cfg_name,
        ]
        log(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            log(f"  ERROR computing norms for {task_name}:")
            log(f"  stdout: {result.stdout[-500:]}")
            log(f"  stderr: {result.stderr[-500:]}")
        else:
            log(f"  Done: {task_name}")
            if result.stdout:
                for line in result.stdout.strip().split("\n")[-3:]:
                    log(f"    {line}")

    log("Norm stats computation complete.")


# ═══════════════════════════════════════════════════════════════
# Step 3: train — Single 5-GPU Slurm job
# ═══════════════════════════════════════════════════════════════

def cmd_train(args):
    """Submit a single Slurm job running 6 tasks in parallel (1 GPU each)."""
    tasks = resolve_tasks(args)
    outputs_dir = PROJECT_DIR / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    num_gpus = len(tasks)
    job_name = f"pi05_merged_{num_gpus}tasks"
    log_file = outputs_dir / f"slurm_pi05_merged_%j.log"
    job_script = outputs_dir / ".slurm_pi05_merged.sh"

    # Build per-task training commands
    task_cmds = []
    for i, task_name in enumerate(tasks):
        cfg_name = config_name_finetune(task_name)
        wandb_str = "" if args.no_wandb else "--wandb-enabled"
        task_cmds.append(textwrap.dedent(f"""\
            echo "[$(date)] Starting {task_name} on GPU {i}"
            CUDA_VISIBLE_DEVICES={i} XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 conda run -n openpi05 \\
                python scripts/train.py {cfg_name} \\
                --exp-name=merged \\
                --overwrite \\
                {wandb_str} \\
                2>&1 | tee {outputs_dir}/train_{task_name}_merged.log &"""))

    all_task_cmds = "\n\n".join(task_cmds)

    script_content = (
        f"#!/bin/bash\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --partition={SLURM_PARTITION}\n"
        f"#SBATCH --nodes=1\n"
        f"#SBATCH --ntasks={num_gpus}\n"
        f"#SBATCH --cpus-per-task=8\n"
        f"#SBATCH --gres=gpu:{num_gpus}\n"
        f"#SBATCH --time=48:00:00\n"
        f"#SBATCH --output={log_file}\n"
        f"\n"
        f"set -uo pipefail\n"
        f"\n"
        f'echo "[$(date)] Pi0.5 LoRA merged training on $(hostname)"\n'
        f'echo "[$(date)] GPUs: $(nvidia-smi -L)"\n'
        f'echo "[$(date)] Tasks: {" ".join(tasks)}"\n'
        f"\n"
        f"cd {OPENPI_DIR}\n"
        f"\n"
        f"{all_task_cmds}\n"
        f"\n"
        f'echo "[$(date)] Waiting for all tasks to finish..."\n'
        f"wait\n"
        f'echo "[$(date)] All training jobs complete!"\n'
    )

    job_script.write_text(script_content)
    job_script.chmod(0o755)

    log(f"Slurm script written to {job_script}")
    log(f"Tasks: {tasks} ({num_gpus} GPUs)")

    try:
        result = subprocess.run(
            ["sbatch", str(job_script)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            log(f"Submitted job: {job_id}")
            log(f"Monitor: squeue -u $(whoami)")
            log(f"Log: {log_file}")
        else:
            log(f"ERROR submitting job: {result.stderr.strip()}")
    except FileNotFoundError:
        log("ERROR: sbatch not found. Source SLURM env first:")
        log("  source /APP/u22/ai_x86/toolshs/set-XY-I.sh")
        sys.exit(1)


def cmd_train_status(args):
    """Check training status."""
    log("Pi0.5 Merged Training Status\n")

    result = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-o", "%.10i %.20j %.8T %.10M %.6D %R"],
        capture_output=True, text=True,
    )
    print("Active training jobs:")
    if result.returncode == 0:
        lines = [l for l in result.stdout.strip().split("\n") if "pi05_" in l or "JOBID" in l]
        if len(lines) <= 1:
            print("  (none)")
        else:
            for line in lines:
                print(f"  {line}")
    print()

    print("Checkpoint status:")
    for task_name in ALL_TASKS:
        cfg_name = config_name_finetune(task_name)
        ckpt_dir = OPENPI_DIR / "checkpoints" / cfg_name / "merged"
        if ckpt_dir.exists():
            steps = sorted(
                [d.name for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                key=int,
            )
            if steps:
                print(f"  {task_name}: step {steps[-1]} (latest)")
            else:
                print(f"  {task_name}: directory exists but no checkpoints")
        else:
            print(f"  {task_name}: not started")


# ═══════════════════════════════════════════════════════════════
# Step 4: eval — VLA Server + Rollout
# ═══════════════════════════════════════════════════════════════

def cmd_eval(args):
    """Run VLA evaluation for trained merged models."""
    tasks = resolve_tasks(args)
    port = args.port
    num_rollouts = args.num_rollouts

    results = {}
    if EVAL_RESULTS_FILE.exists():
        results = json.loads(EVAL_RESULTS_FILE.read_text())

    for task_name in tasks:
        cfg_name = config_name_inference(task_name)
        ckpt_dir = OPENPI_DIR / "checkpoints" / config_name_finetune(task_name) / "merged"

        if not ckpt_dir.exists():
            log(f"SKIP {task_name}: no checkpoint at {ckpt_dir}")
            continue

        steps = sorted(
            [d.name for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit()],
            key=int,
        )
        if not steps:
            log(f"SKIP {task_name}: no checkpoint steps found")
            continue

        latest_step = steps[-1]
        checkpoint_path = ckpt_dir / latest_step
        log(f"Evaluating {task_name} (step {latest_step})")

        eval_task = EVAL_TASK_MAP.get(task_name)
        if not eval_task:
            log(f"SKIP {task_name}: no eval task mapping")
            continue

        # Start VLA server
        server_cmd = [
            "conda", "run", "-n", "openpi05",
            "python", str(PROJECT_DIR / "scripts" / "start_vla_server.py"),
            "--model_type", "pi05",
            "--config_name", cfg_name,
            "--checkpoint", str(checkpoint_path),
            "--port", str(port),
            "--device", "cuda:0",
        ]
        log(f"  Starting VLA server on port {port}...")
        server_proc = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        time.sleep(30)
        if server_proc.poll() is not None:
            log(f"  ERROR: VLA server exited early")
            stderr = server_proc.stderr.read().decode()[-500:]
            log(f"  stderr: {stderr}")
            continue

        try:
            eval_cmd = [
                str(MIMICGEN_PYTHON),
                str(PROJECT_DIR / "scripts" / "1c_generate_from_policy.py"),
                "--config", str(PROJECT_DIR / "configs" / "benchmark_v4.yaml"),
                "--policy", "vla_server",
                "--mode", "natural_capture",
                "--vla_port", str(port),
                "--num_rollouts", str(num_rollouts),
                "--task", eval_task,
            ]
            env = os.environ.copy()
            env["MUJOCO_GL"] = "egl"
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["MUJOCO_EGL_DEVICE_ID"] = "0"

            log(f"  Running {num_rollouts} rollouts...")
            eval_result = subprocess.run(eval_cmd, capture_output=True, text=True, env=env,
                                         timeout=3600)

            if eval_result.returncode == 0:
                sr = _parse_success_rate(eval_result.stdout)
                results[task_name] = {
                    "step": int(latest_step),
                    "success_rate": sr,
                    "num_rollouts": num_rollouts,
                }
                log(f"  {task_name}: SR = {sr:.1f}% ({num_rollouts} rollouts)")
            else:
                log(f"  ERROR in eval for {task_name}:")
                log(f"  {eval_result.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            log(f"  TIMEOUT during eval for {task_name}")
        finally:
            _shutdown_vla_server(port)
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    EVAL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"Results saved to {EVAL_RESULTS_FILE}")


def _parse_success_rate(output: str) -> float:
    import re
    for line in reversed(output.strip().split("\n")):
        if "success" in line.lower() and "rate" in line.lower():
            m = re.search(r"(\d+\.?\d*)\s*%", line)
            if m:
                return float(m.group(1))
            m = re.search(r"(\d+)\s*/\s*(\d+)", line)
            if m:
                return 100.0 * int(m.group(1)) / int(m.group(2))
    return -1.0


def _shutdown_vla_server(port: int):
    import pickle
    import socket
    import struct
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(("127.0.0.1", port))
        msg = pickle.dumps({"cmd": "shutdown"})
        sock.sendall(struct.pack(">I", len(msg)) + msg)
        sock.close()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
# Step 5: report
# ═══════════════════════════════════════════════════════════════

def cmd_report(args):
    """Print results table for merged models."""
    results = {}
    if EVAL_RESULTS_FILE.exists():
        results = json.loads(EVAL_RESULTS_FILE.read_text())

    display_tasks = [
        ("PickPlace", "pick_place"),
        ("Coffee", "coffee"),
        ("Stack", "stack"),
        ("StackThree", "stack_three"),
        ("Threading", "threading"),
        ("ThreePieceAssembly", "three_piece_assembly"),
    ]

    header = f"{'Task':<25} {'Merged SR':>12} {'Step':>8}"
    sep = "-" * len(header)
    print(f"\n{header}")
    print(sep)

    for display_name, task_name in display_tasks:
        r = results.get(task_name, {})
        sr = r.get("success_rate")
        step = r.get("step")

        sr_str = f"{sr:.1f}%" if sr is not None and sr >= 0 else "—"
        step_str = str(step) if step is not None else "—"

        print(f"{display_name:<25} {sr_str:>12} {step_str:>8}")

    print(sep)

    print(f"\nCheckpoint status:")
    for task_name in ALL_TASKS:
        ckpt_dir = OPENPI_DIR / "checkpoints" / config_name_finetune(task_name) / "merged"
        if ckpt_dir.exists():
            steps = sorted(
                [d.name for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                key=int,
            )
            status = f"step {steps[-1]}" if steps else "no checkpoints"
        else:
            status = "not started"
        print(f"  {task_name}: {status}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pi0.5 LoRA finetuning — merged (d0+d1) datasets"
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage")

    # convert-data
    p_convert = subparsers.add_parser("convert-data", help="Merge HDF5 d0+d1 → LeRoBot")
    p_convert.add_argument("--tasks", nargs="+", default=None,
                           help=f"Tasks to convert (default: all). Choices: {ALL_TASKS}")
    p_convert.add_argument("--overwrite", action="store_true",
                           help="Overwrite existing datasets")

    # compute-norms
    p_norms = subparsers.add_parser("compute-norms", help="Compute normalization statistics")
    p_norms.add_argument("--tasks", nargs="+", default=None)

    # train
    p_train = subparsers.add_parser("train", help="Submit single N-GPU Slurm job (1 GPU per task)")
    p_train.add_argument("--tasks", nargs="+", default=None)
    p_train.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    # train-status
    subparsers.add_parser("train-status", help="Check training job status")

    # eval
    p_eval = subparsers.add_parser("eval", help="Run VLA evaluation")
    p_eval.add_argument("--tasks", nargs="+", default=None)
    p_eval.add_argument("--port", type=int, default=5555)
    p_eval.add_argument("--num-rollouts", type=int, default=50)

    # report
    subparsers.add_parser("report", help="Print results table")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    cmd_map = {
        "convert-data": cmd_convert_data,
        "compute-norms": cmd_compute_norms,
        "train": cmd_train,
        "train-status": cmd_train_status,
        "eval": cmd_eval,
        "report": cmd_report,
    }

    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
