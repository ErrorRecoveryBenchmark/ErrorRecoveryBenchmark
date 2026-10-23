#!/usr/bin/env python3
"""Run paired clean/error RecoverBench evaluation for one GR00T checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path


def wait_for_port(port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as sock:
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(2)
    return False


def shutdown_server(port: int) -> None:
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        payload = pickle.dumps({"cmd": "shutdown"}, protocol=pickle.HIGHEST_PROTOCOL)
        sock.sendall(struct.pack("!I", len(payload)) + payload)


def _validated_totals(result_dir: Path, task: str, expected: int, key: str) -> None:
    result = json.loads((result_dir / f"{task}.json").read_text(encoding="utf-8"))
    total = int(result[key]["total"])
    if total != expected:
        raise RuntimeError(f"{task} {key} total {total} != {expected}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--setting", required=True, choices=("nominal", "nominal_recovery"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenes-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--server-python", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=48)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--server-timeout", type=int, default=1200)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--clean-total", type=int, default=50)
    parser.add_argument("--error-total", type=int, required=True)
    parser.add_argument(
        "--success-streak-frames",
        type=int,
        choices=(1, 10),
        default=1,
        help="Paper-matched Pi0.5 sweep uses 1; current RecoverBench uses 10.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    server_script = Path(__file__).with_name("gr00t_vla_server.py")
    clean_script = Path(__file__).with_name("eval_clean_multi.py")
    error_script = Path(__file__).with_name("eval_pi05_error_scenes_multi.py")
    step_text = args.checkpoint.name.removeprefix("checkpoint-")
    if not step_text.isdigit():
        raise ValueError(f"checkpoint name must end in an integer step: {args.checkpoint}")
    step = int(step_text)
    output = (
        args.output_root
        / "evaluations"
        / args.task
        / args.setting
        / f"step_{step}"
    )
    clean_output = output / "clean"
    error_output = output / "error"
    output.mkdir(parents=True, exist_ok=True)

    server_log_path = output / "gr00t_server.log"
    server_env = os.environ.copy()
    server_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    server_cmd = [
        str(args.server_python),
        str(server_script),
        "--checkpoint",
        str(args.checkpoint),
        "--device",
        "cuda:0",
        "--port",
        str(args.port),
        "--max-clients",
        str(args.num_workers + 4),
        "--max-batch-size",
        str(args.max_batch_size),
    ]
    with server_log_path.open("w", encoding="utf-8") as server_log:
        server = subprocess.Popen(
            server_cmd,
            cwd=repo_root,
            env=server_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    try:
        if not wait_for_port(args.port, args.server_timeout):
            tail = server_log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"GR00T server failed to start:\n{tail}")

        eval_env = os.environ.copy()
        eval_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        eval_env["MUJOCO_GL"] = "egl"
        eval_env["MUJOCO_EGL_DEVICE_ID"] = "0"
        eval_env.setdefault("PYOPENGL_PLATFORM", "egl")
        common = [
            "--task",
            args.task,
            "--gpu",
            str(args.gpu),
            "--port",
            str(args.port),
            "--max_steps",
            str(args.max_steps),
            "--checkpoint",
            str(args.checkpoint),
            "--external_server",
            "--success-streak-frames",
            str(args.success_streak_frames),
        ]
        subprocess.run(
            [
                sys.executable,
                str(clean_script),
                *common,
                "--num_clean",
                str(args.clean_total),
                "--num_workers",
                str(args.num_workers),
                "--output_dir",
                str(clean_output),
            ],
            cwd=repo_root,
            env=eval_env,
            check=True,
        )
        _validated_totals(clean_output, args.task, args.clean_total, "clean_rollouts")

        subprocess.run(
            [
                sys.executable,
                str(error_script),
                *common,
                "--num_workers",
                str(args.num_workers),
                "--output_dir",
                str(error_output),
                "--scenes_dir",
                str(args.scenes_dir),
                "--group_by",
                "subtype_id",
                "--limit_per_group",
                "10",
                "--max_batch_size",
                str(args.max_batch_size),
            ],
            cwd=repo_root,
            env=eval_env,
            check=True,
        )
        _validated_totals(error_output, args.task, args.error_total, "error_scenes")
        (output / "evaluation_complete").touch()
    finally:
        if server.poll() is None:
            try:
                shutdown_server(args.port)
                server.wait(timeout=30)
            except Exception:
                server.terminate()
                server.wait(timeout=30)


if __name__ == "__main__":
    main()
