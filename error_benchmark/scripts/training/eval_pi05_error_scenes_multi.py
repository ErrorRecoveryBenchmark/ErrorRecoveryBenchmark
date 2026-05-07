#!/usr/bin/env python
"""Multi-worker Pi0.5 error-scene validation.

Runs one batched VLA server per task and many rollout workers against it. This
keeps only one Pi0.5 checkpoint copy on the GPU while parallelizing MuJoCo
rollouts and batching policy requests.
"""

import argparse
import json
import logging
import os
import signal
import socket
import queue
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import yaml

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - only used if tqdm is missing remotely.
    tqdm = None

import multiprocessing as mp

from error_benchmark.scripts.training.eval_pi05_error_scenes import (
    ALL_TASKS,
    CONDA_BIN,
    DEFAULT_OUTPUT_DIR,
    PROJECT_DIR,
    TASK_PROMPTS,
    VLA_SERVER_PY,
    config_name_inference,
    find_latest_checkpoint,
    shutdown_vla_server,
    wait_for_server,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

WORKER_MODES = ("round_robin", "one_per_group")
GROUP_BY_KEYS = ("subtype_id", "error_name", "degree")


@dataclass
class WorkerShard:
    worker_id: int
    paths: List[str]
    group_key: Optional[str] = None
    group_value: Optional[str] = None


class _SimpleProgress:
    """Small tqdm fallback with periodic logging."""

    def __init__(self, total: int, initial: int = 0, desc: str = ""):
        self.total = total
        self.n = initial
        self.desc = desc
        self.start = time.time()

    def __enter__(self):
        logger.info("%s progress: %d/%d", self.desc, self.n, self.total)
        return self

    def __exit__(self, *args):
        logger.info("%s progress: %d/%d", self.desc, self.n, self.total)

    def update(self, n: int = 1):
        self.n += n
        if self.n % 100 == 0 or self.n == self.total:
            elapsed = max(time.time() - self.start, 1e-6)
            rate = (self.n / elapsed) if elapsed > 0 else 0.0
            eta = (self.total - self.n) / rate if rate > 0 else float("inf")
            logger.info(
                "%s progress: %d/%d, %.2f scene/s, ETA %.1f min",
                self.desc,
                self.n,
                self.total,
                rate,
                eta / 60.0,
            )

    def set_postfix(self, **kwargs):
        return None


def progress_bar(total: int, initial: int, desc: str):
    if tqdm is None:
        return _SimpleProgress(total=total, initial=initial, desc=desc)
    return tqdm(
        total=total,
        initial=initial,
        desc=desc,
        unit="scene",
        dynamic_ncols=True,
        mininterval=5.0,
    )


def configure_runtime_env(gpu_id: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    # CUDA_VISIBLE_DEVICES narrows each validation process to one GPU, so
    # MuJoCo EGL must use the local device id inside that visible set.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    extra_paths = [
        str(PROJECT_DIR),
        str(PROJECT_DIR / "shared" / "mimicgen_workspace" / "robosuite"),
        str(PROJECT_DIR / "shared" / "mimicgen_workspace" / "mimicgen"),
    ]
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = ":".join(extra_paths + ([existing] if existing else []))


def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def ensure_port_free(port: int) -> None:
    if is_port_open(port):
        raise RuntimeError(f"Port {port} is already in use before starting validation server")


def shutdown_batched_vla_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait(timeout=10)


def start_batched_vla_server(
    task_name: str,
    checkpoint_path: Path,
    port: int,
    gpu_id: int,
    num_workers: int,
    max_batch_size: int,
    batch_timeout_ms: int,
    inference_timeout: float,
    log_dir: Path,
) -> tuple[subprocess.Popen, Path]:
    cfg_name = config_name_inference(task_name)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        str(CONDA_BIN),
        "run",
        "-n",
        "openpi05",
        "python",
        str(VLA_SERVER_PY),
        "--model_type",
        "pi05",
        "--config_name",
        cfg_name,
        "--checkpoint",
        str(checkpoint_path),
        "--port",
        str(port),
        "--device",
        "cuda:0",
        "--batched",
        "--max_clients",
        str(num_workers + 4),
        "--max_batch_size",
        str(max_batch_size),
        "--batch_timeout_ms",
        str(batch_timeout_ms),
        "--inference_timeout",
        str(inference_timeout),
    ]

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"vla_batched_server_{task_name}_{port}.log"
    log_file = open(log_path, "w")

    logger.info(
        "Starting batched VLA server: task=%s gpu=%s port=%s workers=%s batch=%s",
        task_name,
        gpu_id,
        port,
        num_workers,
        max_batch_size,
    )
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Server log: %s", log_path)
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    return proc, log_path


def read_scene_metadata(json_path: Path) -> tuple[dict, str, str, str]:
    with open(json_path) as f:
        meta = json.load(f)
    error_spec = meta.get("error_spec", {})
    labels = meta.get("labels", {})
    error_name = labels.get("error_name", error_spec.get("error_name", "unknown"))
    degree = labels.get("degree", error_spec.get("degree", "unknown"))
    subtype_id = labels.get("subtype_id", f"{error_name}_{degree}")
    return meta, error_name, degree, subtype_id


def scene_group_value(error_name: str, degree: str, subtype_id: str, group_by: str) -> str:
    if group_by == "subtype_id":
        return subtype_id
    if group_by == "error_name":
        return error_name
    if group_by == "degree":
        return degree
    raise ValueError(f"Unknown group_by '{group_by}'. Available: {', '.join(GROUP_BY_KEYS)}")


def parse_scene_group_from_name(path: Path, group_by: str) -> Optional[str]:
    stem = path.stem
    if not stem.startswith("v5_"):
        return None

    body = stem[3:]
    parts = body.rsplit("_", 2)
    if len(parts) != 3:
        return None

    error_name, degree, _scene_hash = parts
    if not degree.startswith("D"):
        return None

    return scene_group_value(error_name, degree, f"{error_name}_{degree}", group_by)


def result_group_value(result: dict, group_by: str) -> str:
    return scene_group_value(
        result.get("error_name", "unknown"),
        result.get("degree", "unknown"),
        result.get("subtype_id", "unknown"),
        group_by,
    )


def normalize_result_group(result: dict, group_by: str) -> dict:
    result.setdefault("group_key", group_by)
    result.setdefault("group_value", result_group_value(result, group_by))
    return result


def read_scene_group(json_path: Path, group_by: str) -> str:
    group_value = parse_scene_group_from_name(json_path, group_by)
    if group_value is not None:
        return group_value
    _, error_name, degree, subtype_id = read_scene_metadata(json_path)
    return scene_group_value(error_name, degree, subtype_id, group_by)


def make_result(
    scene_id: str,
    subtype_id: str,
    error_name: str,
    degree: str,
    success: bool,
    steps: int,
    worker_id: int,
    started_at: float,
    **extra,
) -> dict:
    result = {
        "scene_id": scene_id,
        "subtype_id": subtype_id,
        "error_name": error_name,
        "degree": degree,
        "success": bool(success),
        "steps": int(steps),
        "worker_id": int(worker_id),
        "duration_sec": time.time() - started_at,
    }
    result.update(extra)
    return result


def evaluate_one_scene(
    env,
    env_wrapper,
    policy,
    json_path: Path,
    max_steps: int,
    worker_id: int,
    group_by: str,
) -> dict:
    started_at = time.time()
    scene_id = json_path.stem
    npz_path = json_path.with_suffix(".npz")

    try:
        _, error_name, degree, subtype_id = read_scene_metadata(json_path)
        group_value = scene_group_value(error_name, degree, subtype_id, group_by)
    except Exception as exc:
        return make_result(
            scene_id,
            "unknown",
            "unknown",
            "unknown",
            False,
            0,
            worker_id,
            started_at,
            group_key=group_by,
            group_value="unknown",
            error=f"metadata_error: {exc}",
        )

    if not npz_path.exists():
        return make_result(
            scene_id,
            subtype_id,
            error_name,
            degree,
            False,
            0,
            worker_id,
            started_at,
            group_key=group_by,
            group_value=group_value,
            error="npz_missing",
        )

    try:
        with np.load(str(npz_path)) as npz_data:
            if "post_sim_state" in npz_data:
                sim_state = np.array(npz_data["post_sim_state"])
            else:
                sim_state = np.array(npz_data["sim_state"])

        env.reset()
        env_wrapper.set_sim_state_flat(sim_state)
        obs, _, _, _ = env_wrapper.step(env_wrapper.get_neutral_action())
        already_success = bool(env_wrapper.check_success())

        policy.start_episode()
        success = already_success
        steps_taken = 0

        for step in range(max_steps):
            policy_result = policy.predict_from_obs(obs)
            obs, _, _, _ = env_wrapper.step(policy_result.action)
            steps_taken = step + 1
            if env_wrapper.check_success():
                success = True
                break

        return make_result(
            scene_id,
            subtype_id,
            error_name,
            degree,
            success,
            steps_taken,
            worker_id,
            started_at,
            group_key=group_by,
            group_value=group_value,
            already_success=already_success,
        )
    except Exception as exc:
        return make_result(
            scene_id,
            subtype_id,
            error_name,
            degree,
            False,
            0,
            worker_id,
            started_at,
            group_key=group_by,
            group_value=group_value,
            error=str(exc),
            traceback=traceback.format_exc(limit=5),
        )


def worker_main(
    worker_id: int,
    task_name: str,
    scene_paths: List[str],
    port: int,
    gpu_id: int,
    max_steps: int,
    group_by: str,
    result_queue,
) -> None:
    configure_runtime_env(gpu_id)
    logging.getLogger().setLevel(logging.WARNING)

    env = None
    policy = None
    processed = 0
    try:
        if not scene_paths:
            return

        from error_benchmark.framework.env_wrapper import EnvWrapper
        from error_benchmark.framework.policy_adapter import PolicyServerAdapter
        from error_benchmark.scripts.utils.script_utils import create_env, load_task_registry

        task_reg = load_task_registry(task_name)
        task_config_path = PROJECT_DIR / task_reg["task_config"]
        with open(task_config_path) as f:
            task_config = yaml.safe_load(f)

        env = create_env(
            task_config,
            task_reg["dataset_path"],
            enable_camera=True,
            camera_resolution=256,
        )
        env_wrapper = EnvWrapper(env, task_config)
        policy = PolicyServerAdapter(
            host="127.0.0.1",
            port=port,
            task_prompt=TASK_PROMPTS[task_name],
            replan_interval=5,
            connection_timeout=300.0,
            seed=worker_id,
        )

        for scene_path in scene_paths:
            result = evaluate_one_scene(
                env,
                env_wrapper,
                policy,
                Path(scene_path),
                max_steps,
                worker_id,
                group_by,
            )
            processed += 1
            result_queue.put({"type": "result", "result": result})
    except BaseException as exc:
        result_queue.put(
            {
                "type": "fatal",
                "worker_id": worker_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if policy is not None:
            try:
                policy.close()
            except Exception:
                pass
        if env is not None and hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass
        result_queue.put({"type": "done", "worker_id": worker_id, "processed": processed})


def aggregate_results(results: Iterable[dict]) -> dict:
    ordered = sorted(results, key=lambda r: r["scene_id"])
    total = len(ordered)
    successes = sum(1 for r in ordered if r.get("success"))
    overall_sr = successes / total if total > 0 else 0.0

    by_subtype: Dict[str, dict] = {}
    by_degree: Dict[str, dict] = {}
    by_error_name: Dict[str, dict] = {}

    for result in ordered:
        for key, bucket in [
            ("subtype_id", by_subtype),
            ("degree", by_degree),
            ("error_name", by_error_name),
        ]:
            name = result.get(key, "unknown")
            if name not in bucket:
                bucket[name] = {"total": 0, "successes": 0}
            bucket[name]["total"] += 1
            if result.get("success"):
                bucket[name]["successes"] += 1

    for bucket in (by_subtype, by_degree, by_error_name):
        for value in bucket.values():
            value["sr"] = value["successes"] / value["total"] if value["total"] else 0.0

    return {
        "overall_sr": overall_sr,
        "total": total,
        "successes": successes,
        "by_subtype": by_subtype,
        "by_degree": by_degree,
        "by_error_name": by_error_name,
        "per_scene": ordered,
    }


def load_partial_results(partial_path: Path, group_by: Optional[str] = None) -> dict:
    if not partial_path.exists():
        return {}

    results = {}
    with open(partial_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            scene_id = result.get("scene_id")
            if scene_id:
                if group_by is not None:
                    result = normalize_result_group(result, group_by)
                results[scene_id] = result
    return results


def append_partial_result(handle, result: dict) -> None:
    json.dump(result, handle, default=str)
    handle.write("\n")
    handle.flush()


def format_duration(seconds: float) -> str:
    if seconds == float("inf") or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m"


def split_round_robin(paths: List[Path], num_workers: int) -> List[WorkerShard]:
    raw_shards: List[List[str]] = [[] for _ in range(num_workers)]
    for idx, path in enumerate(paths):
        raw_shards[idx % num_workers].append(str(path))
    return [
        WorkerShard(worker_id=worker_id, paths=shard)
        for worker_id, shard in enumerate(raw_shards)
        if shard
    ]


def group_scene_paths(paths: List[Path], group_by: str) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for path in paths:
        try:
            group_value = read_scene_group(path, group_by)
        except Exception as exc:
            logger.warning("Could not read metadata for %s; assigning unknown group: %s", path, exc)
            group_value = "unknown"
        groups.setdefault(group_value, []).append(path)
    return groups


def limit_scene_paths_per_group(
    paths: List[Path],
    group_by: str,
    limit_per_group: Optional[int],
) -> List[Path]:
    if limit_per_group is None or limit_per_group <= 0:
        return paths

    groups = group_scene_paths(paths, group_by)
    selected = set()
    for group_value in sorted(groups):
        selected.update(sorted(groups[group_value])[:limit_per_group])
    return [path for path in paths if path in selected]


def split_one_per_group(paths: List[Path], group_by: str) -> List[WorkerShard]:
    groups = group_scene_paths(paths, group_by)
    shards = []
    for worker_id, group_value in enumerate(sorted(groups)):
        group_paths = sorted(groups[group_value])
        shards.append(
            WorkerShard(
                worker_id=worker_id,
                paths=[str(path) for path in group_paths],
                group_key=group_by,
                group_value=group_value,
            )
        )
    return shards


def build_worker_shards(
    paths: List[Path],
    num_workers: int,
    worker_mode: str,
    group_by: str,
) -> List[WorkerShard]:
    if worker_mode == "round_robin":
        return split_round_robin(paths, num_workers)
    if worker_mode == "one_per_group":
        return split_one_per_group(paths, group_by)
    raise ValueError(f"Unknown worker_mode '{worker_mode}'. Available: {', '.join(WORKER_MODES)}")


def build_resume_aware_worker_shards(
    scene_jsons: List[Path],
    completed_scene_ids: set[str],
    num_workers: int,
    worker_mode: str,
    group_by: str,
) -> List[WorkerShard]:
    if worker_mode == "round_robin":
        remaining = [path for path in scene_jsons if path.stem not in completed_scene_ids]
        return split_round_robin(remaining, num_workers)
    if worker_mode == "one_per_group":
        groups = group_scene_paths(scene_jsons, group_by)
        shards = []
        for worker_id, group_value in enumerate(sorted(groups)):
            group_paths = [
                path
                for path in sorted(groups[group_value])
                if path.stem not in completed_scene_ids
            ]
            if not group_paths:
                continue
            shards.append(
                WorkerShard(
                    worker_id=worker_id,
                    paths=[str(path) for path in group_paths],
                    group_key=group_by,
                    group_value=group_value,
                )
            )
        return shards
    raise ValueError(f"Unknown worker_mode '{worker_mode}'. Available: {', '.join(WORKER_MODES)}")


def estimate_worker_count(
    scene_jsons: List[Path],
    num_workers: int,
    worker_mode: str,
    group_by: str,
) -> int:
    if worker_mode == "round_robin":
        return min(num_workers, len(scene_jsons))
    if worker_mode == "one_per_group":
        return len(group_scene_paths(scene_jsons, group_by))
    raise ValueError(f"Unknown worker_mode '{worker_mode}'. Available: {', '.join(WORKER_MODES)}")


def run_workers(
    task_name: str,
    scene_jsons: List[Path],
    port: int,
    gpu_id: int,
    max_steps: int,
    num_workers: int,
    partial_path: Path,
    resume: bool,
    worker_mode: str,
    group_by: str,
) -> dict:
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    results_by_scene = load_partial_results(partial_path, group_by=group_by) if resume else {}
    if results_by_scene:
        logger.info("Resuming from %s existing scene results in %s", len(results_by_scene), partial_path)

    completed_scene_ids = set(results_by_scene)
    remaining = [p for p in scene_jsons if p.stem not in completed_scene_ids]
    shards = build_resume_aware_worker_shards(
        scene_jsons=scene_jsons,
        completed_scene_ids=completed_scene_ids,
        num_workers=num_workers,
        worker_mode=worker_mode,
        group_by=group_by,
    )
    active_workers = len(shards)

    if worker_mode == "one_per_group":
        preview = ", ".join(f"{s.group_value}:{len(s.paths)}" for s in shards[:12])
        if len(shards) > 12:
            preview += ", ..."
        logger.info(
            "Prepared %d group workers by %s for %d remaining scenes%s%s",
            active_workers,
            group_by,
            len(remaining),
            ": " if preview else "",
            preview,
        )
    else:
        logger.info("Prepared %d round-robin workers for %d remaining scenes", active_workers, len(remaining))

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs = []
    finished_workers = set()
    failed_workers = []

    for shard in shards:
        name = f"{task_name}-worker-{shard.worker_id}"
        if shard.group_value is not None:
            safe_group = "".join(c if c.isalnum() or c in "._-" else "_" for c in shard.group_value)
            name = f"{task_name}-{safe_group}-{shard.worker_id}"
        proc = ctx.Process(
            target=worker_main,
            args=(shard.worker_id, task_name, shard.paths, port, gpu_id, max_steps, group_by, result_queue),
            name=name,
        )
        proc.start()
        procs.append((shard.worker_id, proc))

    total = len(scene_jsons)
    done = len(results_by_scene)
    successes = sum(1 for r in results_by_scene.values() if r.get("success"))
    last_log = time.time()
    progress_start = time.time()
    initial_done = done

    try:
        with open(partial_path, "a") as partial_file, progress_bar(
            total=total,
            initial=done,
            desc=f"{task_name} RSR",
        ) as pbar:
            while len(finished_workers) < active_workers:
                try:
                    msg = result_queue.get(timeout=5.0)
                except queue.Empty:
                    for worker_id, proc in procs:
                        if (
                            proc.exitcode is not None
                            and proc.exitcode != 0
                            and worker_id not in finished_workers
                        ):
                            failed_workers.append(f"{proc.name} exited with {proc.exitcode}")
                            finished_workers.add(worker_id)
                    continue

                msg_type = msg.get("type")
                if msg_type == "result":
                    result = normalize_result_group(msg["result"], group_by)
                    scene_id = result["scene_id"]
                    previous = results_by_scene.get(scene_id)
                    is_new = previous is None
                    results_by_scene[scene_id] = result
                    append_partial_result(partial_file, result)
                    if is_new:
                        done += 1
                        if result.get("success"):
                            successes += 1
                        pbar.update(1)
                    elif bool(previous.get("success")) != bool(result.get("success")):
                        successes += 1 if result.get("success") else -1
                    sr = successes / done if done else 0.0
                    elapsed = max(time.time() - progress_start, 1e-6)
                    new_done = max(done - initial_done, 0)
                    rate = new_done / elapsed if new_done else 0.0
                    eta = (total - done) / rate if rate > 0 else float("inf")
                    pbar.set_postfix(
                        sr=f"{sr:.1%}",
                        rate=f"{rate * 3600:.1f}/h",
                        eta=format_duration(eta),
                        live=f"{active_workers - len(finished_workers)}/{active_workers}",
                    )

                    if time.time() - last_log > 300:
                        logger.info(
                            "%s progress: %d/%d, RSR %.1f%%, rate %.1f scenes/h, ETA %s",
                            task_name,
                            done,
                            total,
                            sr * 100,
                            rate * 3600,
                            format_duration(eta),
                        )
                        last_log = time.time()
                elif msg_type == "fatal":
                    failed_workers.append(
                        f"worker {msg.get('worker_id')}: {msg.get('error')}\n{msg.get('traceback')}"
                    )
                    logger.error("Worker %s fatal: %s", msg.get("worker_id"), msg.get("error"))
                elif msg_type == "done":
                    finished_workers.add(msg.get("worker_id"))

        for _, proc in procs:
            proc.join(timeout=10)

        missing = total - len(results_by_scene)
        if failed_workers or missing:
            raise RuntimeError(
                f"{task_name} incomplete: missing={missing}, failed_workers={len(failed_workers)}"
            )
        aggregated = aggregate_results(results_by_scene.values())
        aggregated.update(
            {
                "worker_mode": worker_mode,
                "group_by": group_by,
                "group_worker_count": active_workers,
                "resumed_scene_count": initial_done if resume else 0,
            }
        )
        return aggregated
    finally:
        for _, proc in procs:
            if proc.is_alive():
                proc.terminate()
        for _, proc in procs:
            proc.join(timeout=5)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-worker Pi0.5 error-scene validation")
    parser.add_argument("--task", type=str, required=True, choices=ALL_TASKS)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--scenes_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--limit_scenes", type=int, default=None)
    parser.add_argument(
        "--limit_per_group",
        type=int,
        default=None,
        help="Limit selected scenes per --group_by value; <=0 disables the per-group limit.",
    )
    parser.add_argument("--worker_mode", type=str, default="round_robin", choices=WORKER_MODES)
    parser.add_argument("--group_by", type=str, default="subtype_id", choices=GROUP_BY_KEYS)
    parser.add_argument("--resume", action="store_true", help="Resume from the partial JSONL file; this is the default.")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--max_batch_size", type=int, default=None)
    parser.add_argument("--batch_timeout_ms", type=int, default=20)
    parser.add_argument("--inference_timeout", type=float, default=600.0)
    parser.add_argument("--server_timeout", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")
    if args.limit_per_group is not None and args.limit_per_group < 0:
        raise ValueError("--limit_per_group must be >= 0")

    task_name = args.task
    gpu_id = args.gpu
    port = args.port or (5560 + gpu_id)
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint(task_name)
    checkpoint_step = checkpoint_path.name
    scenes_dir = Path(args.scenes_dir)
    if not scenes_dir.exists():
        raise FileNotFoundError(f"Scenes directory not found: {scenes_dir}")

    all_scene_jsons = sorted(scenes_dir.glob("*.json"))
    scene_jsons = limit_scene_paths_per_group(
        all_scene_jsons,
        group_by=args.group_by,
        limit_per_group=args.limit_per_group,
    )
    if args.limit_scenes is not None:
        scene_jsons = scene_jsons[: args.limit_scenes]
    if not scene_jsons:
        raise RuntimeError(f"No scene JSONs found in {scenes_dir}")

    configure_runtime_env(gpu_id)
    server_worker_count = estimate_worker_count(
        scene_jsons=scene_jsons,
        num_workers=args.num_workers,
        worker_mode=args.worker_mode,
        group_by=args.group_by,
    )
    max_batch_size = args.max_batch_size or server_worker_count
    if args.limit_per_group is not None and args.limit_per_group > 0:
        partial_name = f"{task_name}_step{checkpoint_step}_{args.limit_per_group}per_{args.group_by}_partial.jsonl"
    else:
        partial_name = f"{task_name}_step{checkpoint_step}_partial.jsonl"
    partial_path = output_dir / "partials" / partial_name
    resume = False if args.no_resume else True

    logger.info("Task: %s", task_name)
    logger.info("GPU: %s, port: %s, requested workers: %s", gpu_id, port, args.num_workers)
    logger.info(
        "Worker mode: %s, group_by: %s, estimated active workers: %s",
        args.worker_mode,
        args.group_by,
        server_worker_count,
    )
    logger.info(
        "Scenes: %s selected from %s total in %s (limit_per_group=%s)",
        len(scene_jsons),
        len(all_scene_jsons),
        scenes_dir,
        args.limit_per_group,
    )
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Partial results: %s (resume=%s)", partial_path, resume)

    ensure_port_free(port)

    server_proc = None
    try:
        server_proc, server_log_path = start_batched_vla_server(
            task_name=task_name,
            checkpoint_path=checkpoint_path,
            port=port,
            gpu_id=gpu_id,
            num_workers=server_worker_count,
            max_batch_size=max_batch_size,
            batch_timeout_ms=args.batch_timeout_ms,
            inference_timeout=args.inference_timeout,
            log_dir=output_dir / "logs",
        )

        server_timeout = args.server_timeout or int(os.environ.get("VLA_SERVER_TIMEOUT", "1200"))
        logger.info("Waiting for batched VLA server for up to %ss...", server_timeout)
        if not wait_for_server(port, timeout=server_timeout):
            try:
                log_tail = server_log_path.read_text()[-4000:]
                logger.error("Server log tail:\n%s", log_tail)
            except Exception:
                pass
            raise RuntimeError("Batched VLA server failed to start")
        logger.info("Batched VLA server ready.")

        error_results = run_workers(
            task_name=task_name,
            scene_jsons=scene_jsons,
            port=port,
            gpu_id=gpu_id,
            max_steps=args.max_steps,
            num_workers=args.num_workers,
            partial_path=partial_path,
            resume=resume,
            worker_mode=args.worker_mode,
            group_by=args.group_by,
        )

        combined_results = {
            "task": task_name,
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": int(checkpoint_step) if checkpoint_step.isdigit() else checkpoint_step,
            "timestamp": datetime.now().isoformat(),
            "max_steps": args.max_steps,
            "gpu": gpu_id,
            "port": port,
            "num_workers": args.num_workers,
            "server_worker_count": server_worker_count,
            "worker_mode": args.worker_mode,
            "group_by": args.group_by,
            "limit_per_group": args.limit_per_group,
            "full_scene_count": len(all_scene_jsons),
            "selected_scene_count": len(scene_jsons),
            "resume": resume,
            "batched_server": True,
            "scenes_dir": str(scenes_dir),
            "error_scenes": error_results,
        }

        result_path = output_dir / f"{task_name}.json"
        with open(result_path, "w") as f:
            json.dump(combined_results, f, indent=2, default=str)

        logger.info(
            "Error Scene RSR: %.1f%% (%d/%d)",
            error_results["overall_sr"] * 100,
            error_results["successes"],
            error_results["total"],
        )
        logger.info("Results saved to %s", result_path)
    finally:
        if server_proc is not None:
            shutdown_batched_vla_server(server_proc)


if __name__ == "__main__":
    main()
