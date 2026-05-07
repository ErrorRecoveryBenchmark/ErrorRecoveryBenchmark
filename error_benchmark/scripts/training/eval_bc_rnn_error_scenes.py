#!/usr/bin/env python
"""
Evaluate the mixed-2000 BC-RNN baseline on:
  1. Clean rollouts via robomimic.scripts.run_trained_agent
  2. M12 validation set error scenes (outputs/v5/{task}/scenes)

Each task has one checkpoint dir bc_rnn_{task}_mixed_2000/.../models/
and we always pick the highest-epoch model_epoch_*.pth.

Usage:
  conda activate mimicgen_env
  MUJOCO_GL=egl python eval_bc_rnn_error_scenes.py --task stack --gpu 0

  # Quick smoke test (sample 100 scenes evenly across subtypes)
  python eval_bc_rnn_error_scenes.py --task stack --scenes_limit 100 --num_clean 10 --gpu 0
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ═══════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════

PROJECT_DIR = Path(
    os.environ.get("ERROR_RECOVERY_BENCHMARK_ROOT")
    or Path(__file__).resolve().parents[3]
)
CKPT_DIR = Path(
    os.environ.get("BC_RNN_CHECKPOINT_DIR")
    or "${BENCHMARK_ROOT}/checkpoints"
)
DEFAULT_SCENES_BASE = PROJECT_DIR / "error_benchmark" / "outputs" / "v5"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "error_benchmark" / "outputs" / "eval_bc_rnn_baseline"

# Ensure framework is importable
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "shared" / "mimicgen_workspace" / "robosuite"))
sys.path.insert(0, str(PROJECT_DIR / "shared" / "mimicgen_workspace" / "mimicgen"))

PYTHONPATH_IMPORT_ROOTS = [
    str(PROJECT_DIR),
    str(PROJECT_DIR / "shared" / "mimicgen_workspace" / "robosuite"),
    str(PROJECT_DIR / "shared" / "mimicgen_workspace" / "mimicgen"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def configure_runtime_env(gpu_id: int) -> None:
    """Set safe MuJoCo / EGL defaults before importing robosuite-backed modules."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    if os.environ.get("MUJOCO_GL", "").lower().strip() == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    # Respect an externally pinned GPU set by the launcher; otherwise pin to the
    # requested physical GPU for direct script invocation.
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # This script always uses one visible GPU. After CUDA remapping, EGL should
    # use local device 0.
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

# ═══════════════════════════════════════════════════════════════
# Task definitions
# ═══════════════════════════════════════════════════════════════

BC_RNN_TASK_KEYS = [
    "coffee", "pick_place", "stack",
    "stack_three", "threading", "three_piece_assembly",
]

CKPT_DIR_SUFFIX = "_mixed_2000"


# ═══════════════════════════════════════════════════════════════
# Checkpoint discovery
# ═══════════════════════════════════════════════════════════════

def find_checkpoints(task_key: str) -> list:
    """Find all checkpoint .pth files for a BC-RNN task, sorted by epoch."""
    ckpt_base = CKPT_DIR / f"bc_rnn_{task_key}{CKPT_DIR_SUFFIX}"
    if not ckpt_base.is_dir():
        return []

    pth_files = []
    for root, _dirs, files in os.walk(ckpt_base):
        for fn in files:
            if fn.startswith("model_epoch_") and fn.endswith(".pth"):
                full = Path(root) / fn
                m = re.search(r"model_epoch_(\d+)\.pth", fn)
                if m:
                    epoch = int(m.group(1))
                    pth_files.append((epoch, full))

    pth_files.sort(key=lambda x: x[0])
    return pth_files


def find_last_checkpoint(task_key: str) -> Path:
    """Return the highest-epoch checkpoint (mixed_2000 baseline only saves last)."""
    ckpts = find_checkpoints(task_key)
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoints found for {task_key} under "
            f"{CKPT_DIR}/bc_rnn_{task_key}{CKPT_DIR_SUFFIX}"
        )
    epoch, path = ckpts[-1]
    logger.info(f"Using last checkpoint for {task_key}: epoch {epoch} ({path})")
    return path


def scene_subtype_id_from_meta(meta: dict) -> str:
    """Infer subtype id from scene metadata, falling back to error_name/degree."""
    labels = meta.get("labels", {})
    spec = meta.get("error_spec", {})
    return labels.get("subtype_id") or (
        f"{labels.get('error_name', spec.get('error_name', 'unknown'))}_"
        f"{labels.get('degree', spec.get('degree', 'unknown'))}"
    )


def build_scene_catalog(scene_jsons: list[Path]) -> tuple[dict[str, list[Path]], list[tuple[Path, str]]]:
    """Group valid scene JSONs by subtype and return invalid metadata files separately."""
    by_subtype: dict[str, list[Path]] = {}
    invalid: list[tuple[Path, str]] = []

    for path in sorted(scene_jsons):
        try:
            with open(path) as f:
                meta = json.load(f)
        except Exception as exc:
            invalid.append((path, str(exc)))
            continue

        subtype_id = scene_subtype_id_from_meta(meta)
        by_subtype.setdefault(subtype_id, []).append(path)

    for paths in by_subtype.values():
        paths.sort()
    return dict(sorted(by_subtype.items())), invalid


def select_balanced_scene_subset(
    scene_jsons: list[Path], scenes_limit: int, scenes_seed: int
) -> list[Path]:
    """Select exactly ``scenes_limit`` scenes with subtype counts as even as possible."""
    scene_jsons = sorted(scene_jsons)
    if scenes_limit <= 0 or scenes_limit >= len(scene_jsons):
        return scene_jsons

    by_subtype, invalid = build_scene_catalog(scene_jsons)
    valid_total = sum(len(paths) for paths in by_subtype.values())
    if invalid:
        logger.warning(
            "Skipping %d invalid scene metadata files during sampling",
            len(invalid),
        )

    if scenes_limit >= valid_total:
        if scenes_limit > valid_total:
            logger.warning(
                "Requested %d scenes but only %d valid scene metadata files were available",
                scenes_limit,
                valid_total,
            )
        selected_all: list[Path] = []
        for subtype_id in sorted(by_subtype):
            selected_all.extend(by_subtype[subtype_id])
        return selected_all

    rng = np.random.default_rng(scenes_seed)
    subtype_order = list(by_subtype.keys())
    rng.shuffle(subtype_order)

    quotas = {subtype_id: 0 for subtype_id in by_subtype}
    remaining = scenes_limit
    while remaining > 0:
        progressed = False
        for subtype_id in subtype_order:
            if quotas[subtype_id] >= len(by_subtype[subtype_id]):
                continue
            quotas[subtype_id] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break

    selected: list[Path] = []
    for subtype_id in sorted(by_subtype):
        quota = quotas[subtype_id]
        if quota <= 0:
            continue
        paths = by_subtype[subtype_id]
        if quota == len(paths):
            chosen = paths
        else:
            chosen_idx = np.sort(rng.choice(len(paths), size=quota, replace=False))
            chosen = [paths[i] for i in chosen_idx]
        selected.extend(chosen)

    selected.sort()
    return selected


# ═══════════════════════════════════════════════════════════════
# Clean rollout evaluation using robomimic
# ═══════════════════════════════════════════════════════════════

def build_robomimic_subprocess_env() -> dict[str, str]:
    """Build an env where robomimic can import local robosuite/mimicgen packages."""
    env = os.environ.copy()
    existing_paths = [
        item for item in env.get("PYTHONPATH", "").split(os.pathsep) if item
    ]
    merged_paths: list[str] = []
    for item in PYTHONPATH_IMPORT_ROOTS + existing_paths:
        if item not in merged_paths:
            merged_paths.append(item)
    env["PYTHONPATH"] = os.pathsep.join(merged_paths)
    return env


def _parse_json_after_marker(output: str, marker: str) -> dict[str, Any] | None:
    marker_idx = output.rfind(marker)
    if marker_idx < 0:
        return None
    after_marker = output[marker_idx + len(marker):].lstrip()
    try:
        parsed, _end = json.JSONDecoder().raw_decode(after_marker)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_robomimic_success_rate(output: str, num_rollouts: int) -> dict[str, Any] | None:
    """Parse robomimic's clean-rollout average stats from stdout/stderr."""
    stats = _parse_json_after_marker(output, "Average Rollout Stats")
    if stats:
        sr = stats.get("Success_Rate")
        num_success = stats.get("Num_Success")
        if sr is not None:
            sr = float(sr)
            successes = (
                int(round(float(num_success)))
                if num_success is not None
                else int(round(sr * num_rollouts))
            )
            return {
                "sr": sr,
                "total": num_rollouts,
                "successes": successes,
                "stats": stats,
                "parse_method": "average_rollout_stats_json",
            }
        if num_success is not None:
            successes = int(round(float(num_success)))
            return {
                "sr": successes / num_rollouts if num_rollouts else None,
                "total": num_rollouts,
                "successes": successes,
                "stats": stats,
                "parse_method": "average_rollout_stats_num_success",
            }

    patterns = [
        (r'"Success_Rate"\s*:\s*([0-9.eE+-]+)', "success_rate"),
        (r"Success_Rate\s*[:=]\s*([0-9.eE+-]+)", "success_rate"),
        (r"success[_ ]rate\s*[:=]\s*([0-9.eE+-]+)", "success_rate"),
        (r'"Num_Success"\s*:\s*([0-9.eE+-]+)', "num_success"),
        (r"Num_Success\s*[:=]\s*([0-9.eE+-]+)", "num_success"),
        (r"success(?:es)?\s*[:=]\s*([0-9]+)\s*/\s*([0-9]+)", "fraction"),
    ]
    for pattern, kind in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if not match:
            continue
        if kind == "num_success":
            successes = int(round(float(match.group(1))))
            sr = successes / num_rollouts if num_rollouts else None
        elif kind == "fraction":
            successes = int(match.group(1))
            total = int(match.group(2))
            sr = successes / total if total else None
            num_rollouts = total
        else:
            sr = float(match.group(1))
            successes = int(round(sr * num_rollouts))
        return {
            "sr": sr,
            "total": num_rollouts,
            "successes": successes,
            "parse_method": f"regex:{kind}",
        }
    return None


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:] if len(text) > limit else text


def evaluate_clean_rollouts_robomimic(checkpoint_path: Path, num_rollouts: int = 50,
                                       horizon: int = 400) -> dict:
    """Run clean rollouts using robomimic's run_trained_agent."""
    logger.info(f"Running {num_rollouts} clean rollouts via robomimic...")

    # Run robomimic's run_trained_agent as subprocess
    cmd = [
        sys.executable, "-m", "robomimic.scripts.run_trained_agent",
        "--agent", str(checkpoint_path),
        "--n_rollouts", str(num_rollouts),
        "--horizon", str(horizon),
        "--seed", "0",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=build_robomimic_subprocess_env(),
    )

    # Combine stdout and stderr for parsing
    full_output = result.stdout + "\n" + result.stderr

    parsed = parse_robomimic_success_rate(full_output, num_rollouts)
    if parsed is not None:
        sr = parsed["sr"]
        successes = parsed["successes"]
        logger.info(f"Clean SR: {sr*100:.1f}% ({successes}/{parsed['total']})")
        parsed["returncode"] = result.returncode
        return parsed

    if result.returncode != 0:
        logger.error("robomimic clean rollout command failed with exit code %s", result.returncode)
        logger.error("robomimic stderr tail:\n%s", _tail(result.stderr))
        return {
            "sr": None,
            "total": num_rollouts,
            "successes": None,
            "error": "robomimic_failed",
            "returncode": result.returncode,
            "stdout_tail": _tail(result.stdout),
            "stderr_tail": _tail(result.stderr),
        }

    logger.error("Could not parse success rate from robomimic output")
    logger.error("robomimic output tail:\n%s", _tail(full_output))
    return {
        "sr": None,
        "total": num_rollouts,
        "successes": None,
        "error": "parse_failed",
        "returncode": result.returncode,
        "stdout_tail": _tail(result.stdout),
        "stderr_tail": _tail(result.stderr),
    }


# ═══════════════════════════════════════════════════════════════
# Error scene evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate_error_scenes(env, env_wrapper, policy, scenes_dir: Path,
                          task_name: str, max_steps: int = 500,
                          scenes_limit: int = 0,
                          scenes_seed: int = 0) -> dict:
    """Run policy on each error scene and measure recovery success.

    If scenes_limit > 0, sample that many scenes evenly across subtypes
    (deterministic w.r.t. scenes_seed) so each subtype gets ~equal coverage.
    """
    scene_jsons = sorted(scenes_dir.glob("*.json"))
    logger.info(f"Found {len(scene_jsons)} error scenes in {scenes_dir}")

    if scenes_limit and scenes_limit < len(scene_jsons):
        scene_jsons = select_balanced_scene_subset(scene_jsons, scenes_limit, scenes_seed)
        sampled_by_subtype, invalid = build_scene_catalog(scene_jsons)
        subtype_counts = [len(paths) for paths in sampled_by_subtype.values()]
        if invalid:
            logger.warning(
                "Selected subset still includes %d unreadable scene files",
                len(invalid),
            )
        if subtype_counts:
            logger.info(
                "Sampled down to %d scenes across %d subtypes "
                "(target=%d, min=%d/subtype, max=%d/subtype)",
                len(scene_jsons),
                len(subtype_counts),
                scenes_limit,
                min(subtype_counts),
                max(subtype_counts),
            )
        else:
            logger.info(
                "Sampled down to %d scenes (target=%d)",
                len(scene_jsons),
                scenes_limit,
            )

    results = []
    successes = 0

    for idx, json_path in enumerate(scene_jsons):
        scene_id = json_path.stem
        npz_path = json_path.with_suffix(".npz")

        try:
            with open(json_path) as f:
                meta = json.load(f)
        except Exception as e:
            logger.error(f"  [{idx+1}/{len(scene_jsons)}] {scene_id}: JSON ERROR {e}")
            results.append({
                "scene_id": scene_id,
                "subtype_id": "unknown",
                "error_name": "unknown",
                "degree": "unknown",
                "success": False,
                "steps": 0,
                "error": f"json_load_failed: {e}",
            })
            continue

        error_spec = meta.get("error_spec", {})
        labels = meta.get("labels", {})
        error_name = labels.get("error_name", error_spec.get("error_name", "unknown"))
        degree = labels.get("degree", error_spec.get("degree", "unknown"))
        subtype_id = scene_subtype_id_from_meta(meta)

        if not npz_path.exists():
            results.append({
                "scene_id": scene_id, "subtype_id": subtype_id,
                "error_name": error_name, "degree": degree,
                "success": False, "steps": 0, "error": "npz_missing",
            })
            continue

        try:
            npz_data = np.load(str(npz_path))
            if "post_sim_state" in npz_data:
                sim_state = npz_data["post_sim_state"]
            else:
                sim_state = npz_data["sim_state"]

            obs = env.reset()
            env_wrapper.set_sim_state_flat(sim_state)
            obs, _, _, _ = env_wrapper.step(env_wrapper.get_neutral_action())

            already_success = env_wrapper.check_success()
            policy.start_episode()
            success = already_success
            steps_taken = 0

            for step in range(max_steps):
                policy_result = policy.predict_from_obs(obs)
                obs, reward, done, info = env_wrapper.step(policy_result.action)
                steps_taken = step + 1
                if env_wrapper.check_success():
                    success = True
                    break

            if success:
                successes += 1

            results.append({
                "scene_id": scene_id, "subtype_id": subtype_id,
                "error_name": error_name, "degree": degree,
                "success": success, "steps": steps_taken,
                "already_success": already_success,
            })

            status = "OK" if success else "FAIL"
            logger.info(f"  [{idx+1}/{len(scene_jsons)}] {scene_id}: {status} in {steps_taken} steps")

        except Exception as e:
            logger.error(f"  [{idx+1}/{len(scene_jsons)}] {scene_id}: ERROR {e}")
            results.append({
                "scene_id": scene_id, "subtype_id": subtype_id,
                "error_name": error_name, "degree": degree,
                "success": False, "steps": 0, "error": str(e),
            })

    # Aggregate results
    total = len(results)
    overall_sr = successes / total if total > 0 else 0.0

    by_subtype = {}
    for r in results:
        sid = r["subtype_id"]
        if sid not in by_subtype:
            by_subtype[sid] = {"total": 0, "successes": 0}
        by_subtype[sid]["total"] += 1
        if r["success"]:
            by_subtype[sid]["successes"] += 1
    for v in by_subtype.values():
        v["sr"] = v["successes"] / v["total"] if v["total"] > 0 else 0.0

    return {
        "overall_sr": overall_sr,
        "total": total,
        "successes": successes,
        "by_subtype": by_subtype,
        "per_scene": results,
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate BC-RNN on error scenes")
    parser.add_argument("--task", type=str, required=True, choices=BC_RNN_TASK_KEYS)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--num_clean", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_clean", action="store_true")
    parser.add_argument("--skip_error", action="store_true")
    parser.add_argument("--scenes_dir", type=str, default=None,
                        help="Override the per-task scenes directory")
    parser.add_argument("--scenes_root", type=str,
                        default=str(DEFAULT_SCENES_BASE),
                        help="Root containing {task}/scenes/*.json (default: outputs/v5)")
    parser.add_argument("--scenes_limit", type=int, default=0,
                        help="Sample N scenes evenly across subtypes (0=use all)")
    parser.add_argument("--scenes_seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    task_key = args.task
    gpu_id = args.gpu
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_runtime_env(gpu_id)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_last_checkpoint(task_key)

    epoch_match = re.search(r"epoch_(\d+)", checkpoint_path.name)
    checkpoint_epoch = int(epoch_match.group(1)) if epoch_match else 0

    logger.info(f"Task: {task_key}, GPU: {gpu_id}")
    logger.info(f"Checkpoint: {checkpoint_path} (epoch {checkpoint_epoch})")

    base_task_name = task_key

    if args.scenes_dir:
        scenes_dir = Path(args.scenes_dir)
    else:
        scenes_dir = Path(args.scenes_root) / base_task_name / "scenes"

    result_path = output_dir / f"{task_key}.json"
    if result_path.exists():
        try:
            with open(result_path) as f:
                combined_results = json.load(f)
        except Exception as exc:
            logger.warning("Could not load existing result file %s: %s", result_path, exc)
            combined_results = {}
    else:
        combined_results = {}

    combined_results.update({
        "task": task_key,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "timestamp": datetime.now().isoformat(),
        "max_steps": args.max_steps,
        "gpu": gpu_id,
        "scenes_limit": args.scenes_limit,
        "scenes_seed": args.scenes_seed,
        "scenes_dir": str(scenes_dir),
    })

    if not args.skip_error:
        if not scenes_dir.exists():
            raise FileNotFoundError(
                f"Scene directory does not exist for task {task_key}: {scenes_dir}"
            )
        if not scenes_dir.is_dir():
            raise NotADirectoryError(
                f"Scene path is not a directory for task {task_key}: {scenes_dir}"
            )

    # Clean rollout evaluation via robomimic
    if not args.skip_clean:
        logger.info(f"\n{'='*60}")
        logger.info(f"Clean Rollout Evaluation (robomimic)")
        logger.info(f"{'='*60}")
        clean_results = evaluate_clean_rollouts_robomimic(
            checkpoint_path, args.num_clean, horizon=args.max_steps
        )
        combined_results["clean_rollouts"] = clean_results

    # Error scene evaluation
    if not args.skip_error:
        logger.info(f"\n{'='*60}")
        logger.info(f"Error Scene Evaluation: {task_key}")
        logger.info(f"{'='*60}")

        import yaml
        from error_benchmark.scripts.utils.script_utils import create_env, load_task_registry
        from error_benchmark.framework.env_wrapper import EnvWrapper
        from error_benchmark.framework.policy_adapter import RobomimicPolicyAdapter

        task_reg = load_task_registry(base_task_name)
        task_config_path = os.path.join(str(PROJECT_DIR), task_reg["task_config"])
        with open(task_config_path) as f:
            task_config = yaml.safe_load(f)

        env = create_env(task_config, task_reg["dataset_path"],
                         enable_camera=True, camera_resolution=84)
        env_wrapper = EnvWrapper(env, task_config)
        policy = RobomimicPolicyAdapter(ckpt_path=str(checkpoint_path), device="cuda:0")

        error_results = evaluate_error_scenes(
            env, env_wrapper, policy, scenes_dir, task_key,
            max_steps=args.max_steps,
            scenes_limit=args.scenes_limit,
            scenes_seed=args.scenes_seed,
        )
        combined_results["error_scenes"] = error_results
        logger.info(f"\nError Scene SR: {error_results['overall_sr']*100:.1f}% "
                    f"({error_results['successes']}/{error_results['total']})")
        env.close()

    # Save results
    with open(result_path, 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {result_path}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY: {task_key}")
    logger.info(f"{'='*60}")
    if "clean_rollouts" in combined_results:
        cr = combined_results["clean_rollouts"]
        if cr.get("sr") is None:
            logger.info(f"  Clean SR: unknown ({cr.get('error', 'missing')})")
        else:
            logger.info(f"  Clean SR: {cr['sr']*100:.1f}% ({cr['successes']}/{cr['total']})")
    if "error_scenes" in combined_results:
        es = combined_results["error_scenes"]
        logger.info(f"  Error Scene SR: {es['overall_sr']*100:.1f}% ({es['successes']}/{es['total']})")


if __name__ == "__main__":
    main()
