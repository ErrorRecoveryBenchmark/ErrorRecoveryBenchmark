#!/usr/bin/env python
"""
Stage 4A: Generate MCM (Motion Correction Module) Training Data

Converts recovery demos and augmented trajectories into LLaVA-format JSON
for Phoenix MCM fine-tuning. Each recovery step generates a training sample
with error image + motion correction instruction.

Usage:
    python error_benchmark/scripts/conversion/4a_generate_mcm_training_data.py \
        --task stack \
        --output_dir error_benchmark/outputs/recovery/phoenix_mcm/stack

Output:
    {output_dir}/
        recovery_mcm_train.json   (LLaVA JSON format)
        images/                   (Recovery camera images)
"""

import argparse
import json
import logging
import sys
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "robosuite"))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "mimicgen"))

from error_benchmark.framework.recovery_types import (
    RecoveryDemo, RecoverySubtask, AugmentedRecovery, SUBTYPE_TO_RBG, RBG_MOTION_TEMPLATES,
)
from error_benchmark.framework.error_taxonomy_v5 import ERROR_SKILL_DEFS, ErrorSkillName
from error_benchmark.framework.logger_setup import setup_logging
from error_benchmark.scripts.utils.script_utils import load_task_registry


# ─── Motion instruction mapping ───
# Maps (RBG, subtask_label) to natural language motion instructions
# These should align with Phoenix's motion codebook when available

MOTION_INSTRUCTIONS = {
    # RBG-A: Re-grasp recovery
    ("RBG_A", "retract"): "move up slowly",
    ("RBG_A", "re_orient"): "rotate gripper to align with object",
    ("RBG_A", "re_grasp"): "move down slowly, then close gripper",
    ("RBG_A", "re_lift"): "move up steadily",
    ("RBG_A", "re_transport"): "move toward target position",
    ("RBG_A", "re_place"): "move down slowly, then open gripper",

    # RBG-B: Retrieve recovery
    ("RBG_B", "navigate_to_object"): "move to the dropped object",
    ("RBG_B", "re_grasp"): "move down slowly, then close gripper",
    ("RBG_B", "re_lift"): "move up steadily",
    ("RBG_B", "re_transport"): "move toward target position",
    ("RBG_B", "re_place"): "move down slowly, then open gripper",

    # RBG-C: Retract recovery
    ("RBG_C", "retract"): "pull back from obstacle",
    ("RBG_C", "navigate_to_object"): "re-approach the target carefully",

    # RBG-D: Redirect recovery
    ("RBG_D", "release"): "open gripper to release wrong object",
    ("RBG_D", "navigate_to_object"): "move to the correct object",
    ("RBG_D", "re_grasp"): "move down slowly, then close gripper",
    ("RBG_D", "re_lift"): "move up steadily",
    ("RBG_D", "re_transport"): "move toward target position",
    ("RBG_D", "re_place"): "move down slowly, then open gripper",

    # RBG-E: Realign recovery
    ("RBG_E", "correct_position"): "adjust position to correct the error",
    ("RBG_E", "resume_task"): "continue the task normally",
}

# Error description templates for MCM conversations
ERROR_DESCRIPTIONS = {
    "grasp_misalignment": "gripper grasped the object with incorrect alignment",
    "grasp_wrong_pose": "gripper has wrong orientation for grasping",
    "drop_in_transit": "object was dropped during transport",
    "drop_at_wrong_place": "object was dropped at the wrong location",
    "collision_holding": "robot collided with obstacle while holding object",
    "collision_empty": "robot collided with obstacle (empty gripper)",
    "collision_eef_object": "robot end-effector hit a non-target object",
    "collision_self": "robot experienced a self-collision",
    "wrong_object": "robot grasped the wrong object",
    "trajectory_regression": "robot moved backward along the trajectory",
    "stuck_no_progress": "robot is stuck and not making progress",
    "position_error": "robot has a position error relative to target",
}


def get_error_description(error_name: str, degree: str) -> str:
    """Generate human-readable error description for MCM prompt."""
    base_desc = ERROR_DESCRIPTIONS.get(error_name, f"error: {error_name}")
    degree_desc = {
        "D0": "(translation error)",
        "D1": "(rotation error)",
    }
    return f"{base_desc} {degree_desc.get(degree, '')}"


def get_motion_instruction(rbg: str, subtask_label: str) -> str:
    """Get the motion instruction for a (RBG, subtask) pair."""
    instruction = MOTION_INSTRUCTIONS.get((rbg, subtask_label))
    if instruction:
        return instruction
    # Fallback: use RBG template
    return RBG_MOTION_TEMPLATES.get(rbg, "correct the error and continue")


def get_task_description(task_name: str) -> str:
    """Get human-readable task description."""
    task_descriptions = {
        "pick_place": "pick up the object and place it in the target bin",
        "stack": "stack the red cube on top of the green cube",
        "coffee": "pick up the coffee pod and insert it into the coffee machine",
        "threading": "pick up the needle and thread it through the ring",
        "stack_three": "stack three cubes on top of each other",
        "three_piece_assembly": "assemble three pieces together",
    }
    return task_descriptions.get(task_name, task_name)


def generate_mcm_samples_from_demo(
    demo: RecoveryDemo,
    images_dir: Path,
    sample_rate: int = 4,
) -> List[dict]:
    """
    Generate MCM training samples from a recovery demo.

    For each sampled timestep, creates a conversation pair:
    - Human: <image> + task context + error description + "What correction?"
    - GPT: motion instruction based on current subtask

    Args:
        demo: RecoveryDemo with camera_images and subtasks
        images_dir: Directory to save extracted images
        sample_rate: Sample every N-th step for training data

    Returns:
        List of LLaVA JSON format entries
    """
    samples = []
    rbg = demo.rbg or SUBTYPE_TO_RBG.get(demo.subtype_id, "RBG_E")
    task_desc = get_task_description(demo.task_name)
    error_desc = get_error_description(demo.error_name, demo.degree)

    if not demo.subtasks:
        return samples

    # Build step -> subtask mapping
    step_to_subtask = {}
    for subtask in demo.subtasks:
        for t in range(subtask.start_step, subtask.end_step):
            step_to_subtask[t] = subtask.label

    # Determine which steps have images
    has_images = demo.camera_images is not None and len(demo.camera_images) > 0

    for step in range(0, demo.num_steps, sample_rate):
        subtask_label = step_to_subtask.get(step, "resume_task")
        motion_instruction = get_motion_instruction(rbg, subtask_label)

        sample_id = f"recovery_{demo.demo_id}_step_{step}"

        # Save image if available
        image_path = None
        if has_images:
            # Map step to image index (images are saved at sample_rate intervals)
            img_idx = step // 4  # Images captured every 4 steps
            if img_idx < len(demo.camera_images):
                img_filename = f"{sample_id}.jpg"
                img_full_path = images_dir / img_filename
                try:
                    import cv2
                    img = demo.camera_images[img_idx]
                    cv2.imwrite(str(img_full_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    image_path = f"recovery_images/{demo.task_name}/{img_filename}"
                except ImportError:
                    # Fall back to PIL
                    try:
                        from PIL import Image
                        img = Image.fromarray(demo.camera_images[img_idx])
                        img.save(str(img_full_path))
                        image_path = f"recovery_images/{demo.task_name}/{img_filename}"
                    except Exception:
                        pass

        # Build conversation
        human_msg = f"<image>\nTask: {task_desc}. Error: {error_desc}. What correction is needed?"
        if image_path is None:
            # Without image, provide more context
            human_msg = (
                f"Task: {task_desc}. Error: {error_desc}. "
                f"Recovery phase: {subtask_label}. What correction is needed?"
            )

        sample = {
            "id": sample_id,
            "conversations": [
                {"from": "human", "value": human_msg},
                {"from": "gpt", "value": motion_instruction},
            ],
        }
        if image_path:
            sample["image"] = image_path

        # Add metadata for training filtering
        sample["metadata"] = {
            "task": demo.task_name,
            "error_name": demo.error_name,
            "degree": demo.degree,
            "subtype_id": demo.subtype_id,
            "rbg": rbg,
            "subtask": subtask_label,
            "step": step,
            "total_steps": demo.num_steps,
            "success": demo.success,
        }

        samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4A: Generate MCM training data")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--config", type=str,
                        default="error_benchmark/configs/recovery_collection.yaml")
    parser.add_argument("--demos_dir", type=str, default=None)
    parser.add_argument("--augmented_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--sample_rate", type=int, default=4,
                        help="Sample every N steps for training data")
    parser.add_argument("--include_augmented", action="store_true",
                        help="Include augmented demos (default: human demos only)")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # Load config
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        recovery_config = yaml.safe_load(f)

    phoenix_config = recovery_config.get('phoenix_conversion', {}).get('mcm', {})

    # Paths
    demos_base = args.demos_dir or recovery_config['paths']['recovery_demos_dir']
    demos_dir = PROJECT_ROOT / demos_base / args.task

    output_base = args.output_dir or recovery_config['paths']['phoenix_mcm_dir']
    output_dir = PROJECT_ROOT / output_base / args.task
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / "recovery_images" / args.task
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== MCM Training Data Generation ===")
    logger.info(f"Task: {args.task}")

    # Load manifest
    manifest_path = demos_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"No manifest found at {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    all_samples = []
    stats = {'total_demos': 0, 'total_samples': 0, 'by_subtype': {}, 'by_rbg': {},
             'human_demos': 0, 'augmented_demos': 0}

    for demo_info in manifest.get('demos', []):
        # Load demo
        demo = RecoveryDemo.from_dict(demo_info)

        # Only use successful demos for MCM training
        if not demo.success:
            continue

        # Load camera images if available
        npz_path = demo_info.get('npz_path', '')
        if npz_path and Path(npz_path).exists():
            data = np.load(npz_path, allow_pickle=True)
            if 'camera_images' in data:
                demo.camera_images = list(data['camera_images'])
            if 'eef_positions' in data:
                demo.eef_positions = data['eef_positions']
            if 'gripper_states' in data:
                demo.gripper_states = data['gripper_states']

        # Generate samples
        samples = generate_mcm_samples_from_demo(
            demo, images_dir, sample_rate=args.sample_rate)

        all_samples.extend(samples)
        stats['total_demos'] += 1
        stats['human_demos'] += 1
        stats['total_samples'] += len(samples)

        sid = demo.subtype_id
        stats['by_subtype'][sid] = stats['by_subtype'].get(sid, 0) + len(samples)
        rbg = demo.rbg
        stats['by_rbg'][rbg] = stats['by_rbg'].get(rbg, 0) + len(samples)

    # Load augmented demos if requested
    if args.include_augmented:
        aug_base = args.augmented_dir or recovery_config['paths']['augmented_demos_dir']
        aug_dir = PROJECT_ROOT / aug_base / args.task
        aug_manifest_path = aug_dir / "augmentation_manifest.json"

        if aug_manifest_path.exists():
            with open(aug_manifest_path) as f:
                aug_manifest = json.load(f)

            for subtype_id, subtype_data in aug_manifest.get('subtypes', {}).items():
                for entry in subtype_data.get('entries', []):
                    if not entry.get('success', False):
                        continue

                    npz_path = entry.get('npz_path', '')
                    if not npz_path or not Path(npz_path).exists():
                        continue

                    # Build RecoveryDemo from augmented entry + NPZ data
                    demo = RecoveryDemo(
                        demo_id=entry.get('augmented_id', ''),
                        task_name=entry.get('task_name', args.task),
                        error_name=entry.get('error_name', ''),
                        degree=entry.get('degree', ''),
                        subtype_id=entry.get('subtype_id', subtype_id),
                        rbg=entry.get('rbg', ''),
                        success=True,
                        num_steps=entry.get('num_steps', 0),
                        subtasks=[RecoverySubtask.from_dict(s) for s in entry.get('subtasks', [])],
                        metadata=entry.get('metadata', {}),
                    )

                    data = np.load(npz_path, allow_pickle=True)
                    if 'camera_images' in data:
                        demo.camera_images = list(data['camera_images'])
                    if 'eef_positions' in data:
                        demo.eef_positions = data['eef_positions']
                    if 'gripper_states' in data:
                        demo.gripper_states = data['gripper_states']

                    samples = generate_mcm_samples_from_demo(
                        demo, images_dir, sample_rate=args.sample_rate)

                    all_samples.extend(samples)
                    stats['total_demos'] += 1
                    stats['augmented_demos'] += 1
                    stats['total_samples'] += len(samples)

                    sid = demo.subtype_id
                    stats['by_subtype'][sid] = stats['by_subtype'].get(sid, 0) + len(samples)
                    rbg_key = demo.rbg
                    stats['by_rbg'][rbg_key] = stats['by_rbg'].get(rbg_key, 0) + len(samples)

            logger.info(f"Loaded augmented demos: {stats['augmented_demos']}")
        else:
            logger.warning(f"Augmented manifest not found: {aug_manifest_path}")

    # Save LLaVA JSON
    output_json = output_dir / "recovery_mcm_train.json"
    with open(output_json, 'w') as f:
        json.dump(all_samples, f, indent=2)

    # Save stats
    stats_path = output_dir / "mcm_conversion_stats.json"
    stats['created'] = datetime.now().isoformat()
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\n=== MCM Conversion Summary ===")
    logger.info(f"Demos processed: {stats['total_demos']}")
    logger.info(f"Training samples: {stats['total_samples']}")
    logger.info(f"Output: {output_json}")
    logger.info(f"By RBG:")
    for rbg, count in sorted(stats['by_rbg'].items()):
        logger.info(f"  {rbg}: {count}")


if __name__ == "__main__":
    main()
