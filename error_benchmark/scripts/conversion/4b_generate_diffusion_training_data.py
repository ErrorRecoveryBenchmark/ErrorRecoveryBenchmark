#!/usr/bin/env python
"""
Stage 4B: Generate Diffusion Policy Training Data

Converts recovery demos and augmented trajectories into HDF5 format
compatible with Phoenix's diffusion policy training (idx_speed_chunking).

Output format:
    - RGB images at 5Hz (agentview, 224x224)
    - 7-DoF action chunks (length=12)
    - Motion label (codebook index)

Usage:
    python error_benchmark/scripts/conversion/4b_generate_diffusion_training_data.py \
        --task stack \
        --output_dir error_benchmark/outputs/recovery/phoenix_diffusion/stack

Output:
    {output_dir}/
        recovery_diffusion_train.hdf5
        conversion_stats.json
"""

import argparse
import json
import logging
import sys
import yaml
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "robosuite"))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "mimicgen"))

from error_benchmark.framework.recovery_types import (
    RecoveryDemo, RecoverySubtask, SUBTYPE_TO_RBG, RBG_MOTION_TEMPLATES,
)
from error_benchmark.framework.logger_setup import setup_logging
from error_benchmark.scripts.utils.script_utils import load_task_registry


# Default motion codebook mapping (Phoenix idx_speed format)
# These map recovery behaviors to motion primitive indices.
# When Phoenix's language_idx.json is available, load from there instead.
DEFAULT_MOTION_CODEBOOK = {
    "move up slowly": 0,
    "move down slowly": 1,
    "move forward": 2,
    "move backward": 3,
    "move left": 4,
    "move right": 5,
    "move up steadily": 6,
    "move down steadily": 7,
    "open gripper": 8,
    "close gripper": 9,
    "rotate clockwise": 10,
    "rotate counter-clockwise": 11,
    "move toward target position": 12,
    "pull back from obstacle": 13,
    "re-approach the target carefully": 14,
    "adjust position to correct the error": 15,
    "continue the task normally": 16,
    "move to the dropped object": 17,
    "move to the correct object": 18,
    "rotate gripper to align with object": 19,
    "open gripper to release wrong object": 20,
    "move down slowly, then close gripper": 21,
    "move down slowly, then open gripper": 22,
    "release object, adjust gripper position, then re-grasp": 23,
    "move to dropped object, pick it up, then continue to target": 24,
    "pull back from collision, then re-approach target": 25,
    "release wrong object, move to correct object, then pick it up": 26,
    "correct the position error and resume the task": 27,
}


def load_motion_codebook(codebook_path: Optional[str] = None) -> Dict[str, int]:
    """Load motion codebook from JSON file or use defaults."""
    if codebook_path and Path(codebook_path).exists():
        with open(codebook_path) as f:
            return json.load(f)
    return DEFAULT_MOTION_CODEBOOK


def get_motion_label(rbg: str, subtask_label: str,
                     codebook: Dict[str, int]) -> int:
    """Get motion codebook index for a recovery subtask."""
    # Direct subtask-based lookup
    motion_instruction_map = {
        ("RBG_A", "retract"): "move up slowly",
        ("RBG_A", "re_orient"): "rotate gripper to align with object",
        ("RBG_A", "re_grasp"): "move down slowly, then close gripper",
        ("RBG_A", "re_lift"): "move up steadily",
        ("RBG_A", "re_transport"): "move toward target position",
        ("RBG_A", "re_place"): "move down slowly, then open gripper",
        ("RBG_B", "navigate_to_object"): "move to the dropped object",
        ("RBG_B", "re_grasp"): "move down slowly, then close gripper",
        ("RBG_B", "re_lift"): "move up steadily",
        ("RBG_B", "re_transport"): "move toward target position",
        ("RBG_B", "re_place"): "move down slowly, then open gripper",
        ("RBG_C", "retract"): "pull back from obstacle",
        ("RBG_C", "navigate_to_object"): "re-approach the target carefully",
        ("RBG_D", "release"): "open gripper to release wrong object",
        ("RBG_D", "navigate_to_object"): "move to the correct object",
        ("RBG_D", "re_grasp"): "move down slowly, then close gripper",
        ("RBG_D", "re_lift"): "move up steadily",
        ("RBG_D", "re_transport"): "move toward target position",
        ("RBG_D", "re_place"): "move down slowly, then open gripper",
        ("RBG_E", "correct_position"): "adjust position to correct the error",
        ("RBG_E", "resume_task"): "continue the task normally",
    }

    instruction = motion_instruction_map.get((rbg, subtask_label), "")
    if instruction in codebook:
        return codebook[instruction]

    # Fallback: use RBG template
    rbg_template = RBG_MOTION_TEMPLATES.get(rbg, "")
    if rbg_template in codebook:
        return codebook[rbg_template]

    return 0  # Default index


def downsample_trajectory(
    actions: np.ndarray,
    source_freq: int = 20,
    target_freq: int = 5,
) -> np.ndarray:
    """Downsample actions from source_freq to target_freq."""
    ratio = source_freq // target_freq
    if ratio <= 1:
        return actions
    return actions[::ratio]


def create_action_chunks(
    actions: np.ndarray,
    chunk_length: int = 12,
) -> np.ndarray:
    """
    Create overlapping action chunks for diffusion policy.

    Args:
        actions: (T, action_dim) array
        chunk_length: Length of each action chunk

    Returns:
        (T, chunk_length, action_dim) array where each chunk starts at timestep t
    """
    T, action_dim = actions.shape
    chunks = np.zeros((T, chunk_length, action_dim))

    for t in range(T):
        end = min(t + chunk_length, T)
        actual_len = end - t
        chunks[t, :actual_len] = actions[t:end]
        # Pad with last action if needed
        if actual_len < chunk_length:
            chunks[t, actual_len:] = actions[-1]

    return chunks


def process_demo_for_diffusion(
    demo: RecoveryDemo,
    codebook: Dict[str, int],
    target_freq: int = 5,
    chunk_length: int = 12,
    image_resolution: int = 224,
) -> Optional[Dict]:
    """
    Convert a single RecoveryDemo into diffusion policy training format.

    Returns:
        Dict with keys: actions, action_chunks, images, motion_labels, states
        or None if conversion fails
    """
    if demo.actions is None or demo.num_steps == 0:
        return None

    rbg = demo.rbg or SUBTYPE_TO_RBG.get(demo.subtype_id, "RBG_E")

    # Downsample actions
    source_freq = 20  # robosuite control_freq
    ds_actions = downsample_trajectory(demo.actions, source_freq, target_freq)
    T = len(ds_actions)

    if T < 2:
        return None

    # Create action chunks
    action_chunks = create_action_chunks(ds_actions, chunk_length)

    # Create motion labels per timestep
    motion_labels = np.zeros(T, dtype=np.int64)

    if demo.subtasks:
        # Build step -> subtask mapping at original freq
        step_to_subtask = {}
        for subtask in demo.subtasks:
            for t in range(subtask.start_step, subtask.end_step):
                step_to_subtask[t] = subtask.label

        # Map to downsampled timesteps
        ratio = source_freq // target_freq
        for t in range(T):
            orig_step = t * ratio
            subtask_label = step_to_subtask.get(orig_step, "resume_task")
            motion_labels[t] = get_motion_label(rbg, subtask_label, codebook)
    else:
        # No segmentation: use RBG template for all steps
        default_label = get_motion_label(rbg, "resume_task", codebook)
        motion_labels[:] = default_label

    # Process images
    images = None
    if demo.camera_images is not None and len(demo.camera_images) > 0:
        # Camera images are already at reduced rate (every 4 steps at 20Hz = 5Hz)
        images_list = demo.camera_images[:T]
        if images_list:
            # Resize if needed
            try:
                import cv2
                resized = []
                for img in images_list:
                    if img.shape[:2] != (image_resolution, image_resolution):
                        img = cv2.resize(img, (image_resolution, image_resolution))
                    resized.append(img)
                images = np.array(resized)
            except ImportError:
                images = np.array(images_list[:T])

    return {
        'actions': ds_actions,
        'action_chunks': action_chunks,
        'motion_labels': motion_labels,
        'images': images,
        'num_steps': T,
        'task_name': demo.task_name,
        'subtype_id': demo.subtype_id,
        'rbg': rbg,
        'success': demo.success,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4B: Generate diffusion policy training data")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--config", type=str,
                        default="error_benchmark/configs/recovery_collection.yaml")
    parser.add_argument("--demos_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--codebook", type=str, default=None,
                        help="Path to Phoenix language_idx.json codebook")
    parser.add_argument("--target_freq", type=int, default=5,
                        help="Target action frequency in Hz")
    parser.add_argument("--chunk_length", type=int, default=12,
                        help="Action chunk length for diffusion policy")
    parser.add_argument("--image_resolution", type=int, default=224)
    parser.add_argument("--augmented_dir", type=str, default=None,
                        help="Path to augmented demos directory")
    parser.add_argument("--include_augmented", action="store_true",
                        help="Include augmented demos (default: human demos only)")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # Load config
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        recovery_config = yaml.safe_load(f)

    diffusion_config = recovery_config.get('phoenix_conversion', {}).get('diffusion', {})
    target_freq = args.target_freq or diffusion_config.get('control_freq', 5)
    chunk_length = args.chunk_length or diffusion_config.get('action_chunk_length', 12)
    image_resolution = args.image_resolution or diffusion_config.get('image_resolution', 224)

    # Paths
    demos_base = args.demos_dir or recovery_config['paths']['recovery_demos_dir']
    demos_dir = PROJECT_ROOT / demos_base / args.task

    output_base = args.output_dir or recovery_config['paths']['phoenix_diffusion_dir']
    output_dir = PROJECT_ROOT / output_base / args.task
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load codebook
    codebook = load_motion_codebook(args.codebook)

    logger.info(f"=== Diffusion Policy Training Data Generation ===")
    logger.info(f"Task: {args.task}")
    logger.info(f"Target freq: {target_freq} Hz")
    logger.info(f"Chunk length: {chunk_length}")
    logger.info(f"Codebook entries: {len(codebook)}")

    # Load demos manifest
    manifest_path = demos_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"No manifest found at {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Process all successful demos
    output_hdf5 = output_dir / "recovery_diffusion_train.hdf5"
    stats = {
        'total_demos': 0, 'total_steps': 0,
        'by_subtype': {}, 'by_rbg': {},
        'human_demos': 0, 'augmented_demos': 0,
    }

    def _write_demo_to_hdf5(demo, data_group, demo_idx, stats):
        """Convert a RecoveryDemo and write to HDF5. Returns new demo_idx."""
        result = process_demo_for_diffusion(
            demo, codebook,
            target_freq=target_freq,
            chunk_length=chunk_length,
            image_resolution=image_resolution,
        )
        if result is None:
            return demo_idx

        demo_key = f"demo_{demo_idx}"
        demo_group = data_group.create_group(demo_key)

        demo_group.create_dataset('actions', data=result['actions'],
                                  compression='gzip')
        demo_group.create_dataset('action_chunks', data=result['action_chunks'],
                                  compression='gzip')
        demo_group.create_dataset('motion_labels', data=result['motion_labels'])

        if result['images'] is not None:
            demo_group.create_dataset('agentview_rgb', data=result['images'],
                                      compression='gzip')

        demo_group.attrs['task_name'] = result['task_name']
        demo_group.attrs['subtype_id'] = result['subtype_id']
        demo_group.attrs['rbg'] = result['rbg']
        demo_group.attrs['success'] = result['success']
        demo_group.attrs['num_steps'] = result['num_steps']
        demo_group.attrs['source_demo_id'] = demo.demo_id

        stats['total_demos'] += 1
        stats['total_steps'] += result['num_steps']
        sid = result['subtype_id']
        stats['by_subtype'][sid] = stats['by_subtype'].get(sid, 0) + 1
        rbg = result['rbg']
        stats['by_rbg'][rbg] = stats['by_rbg'].get(rbg, 0) + 1

        return demo_idx + 1

    with h5py.File(output_hdf5, 'w') as hf:
        # Global attributes
        hf.attrs['version'] = 'v1.0'
        hf.attrs['task'] = args.task
        hf.attrs['target_freq'] = target_freq
        hf.attrs['chunk_length'] = chunk_length
        hf.attrs['image_resolution'] = image_resolution
        hf.attrs['created'] = datetime.now().isoformat()

        data_group = hf.create_group('data')
        demo_idx = 0

        # Human demos
        for demo_info in manifest.get('demos', []):
            demo = RecoveryDemo.from_dict(demo_info)

            if not demo.success:
                continue

            npz_path = demo_info.get('npz_path', '')
            if npz_path and Path(npz_path).exists():
                data = np.load(npz_path, allow_pickle=True)
                demo.actions = data.get('actions')
                demo.states = data.get('states')
                if 'camera_images' in data:
                    demo.camera_images = list(data['camera_images'])

            prev_idx = demo_idx
            demo_idx = _write_demo_to_hdf5(demo, data_group, demo_idx, stats)
            if demo_idx > prev_idx:
                stats['human_demos'] += 1

        # Augmented demos
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

                        demo = RecoveryDemo(
                            demo_id=entry.get('augmented_id', ''),
                            task_name=entry.get('task_name', args.task),
                            error_name=entry.get('error_name', ''),
                            degree=entry.get('degree', ''),
                            subtype_id=entry.get('subtype_id', subtype_id),
                            rbg=entry.get('rbg', ''),
                            success=True,
                            num_steps=entry.get('num_steps', 0),
                            subtasks=[RecoverySubtask.from_dict(s)
                                      for s in entry.get('subtasks', [])],
                            metadata=entry.get('metadata', {}),
                        )

                        data = np.load(npz_path, allow_pickle=True)
                        demo.actions = data.get('actions')
                        demo.states = data.get('states')
                        if 'camera_images' in data:
                            demo.camera_images = list(data['camera_images'])

                        prev_idx = demo_idx
                        demo_idx = _write_demo_to_hdf5(
                            demo, data_group, demo_idx, stats)
                        if demo_idx > prev_idx:
                            stats['augmented_demos'] += 1

                logger.info(f"Loaded augmented demos: {stats['augmented_demos']}")
            else:
                logger.warning(f"Augmented manifest not found: {aug_manifest_path}")

        data_group.attrs['total_demos'] = demo_idx

    # Save stats
    stats_path = output_dir / "conversion_stats.json"
    stats['created'] = datetime.now().isoformat()
    stats['output_hdf5'] = str(output_hdf5)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\n=== Diffusion Conversion Summary ===")
    logger.info(f"Demos converted: {stats['total_demos']}")
    logger.info(f"Total steps (5Hz): {stats['total_steps']}")
    logger.info(f"Output: {output_hdf5}")
    logger.info(f"By RBG:")
    for rbg, count in sorted(stats['by_rbg'].items()):
        logger.info(f"  {rbg}: {count} demos")


if __name__ == "__main__":
    main()
