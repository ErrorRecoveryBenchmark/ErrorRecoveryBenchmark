#!/usr/bin/env python
"""Verify recovery demo NPZ files via state replay.

Loads the error scene's post_sim_state, replays recovery demo actions,
and compares the resulting trajectory with the recorded data.

Usage:
    python error_benchmark/scripts/verification/verify_recovery_replay.py --task stack
    python error_benchmark/scripts/verification/verify_recovery_replay.py --task stack --subtype collision_holding_D0
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "robosuite"))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "mimicgen"))

import yaml

from error_benchmark.framework.env_wrapper import EnvWrapper
from error_benchmark.scripts.utils.script_utils import create_env, load_task_registry

logger = logging.getLogger(__name__)


def verify_single_demo(
    demo_npz_path: str,
    scene_npz_path: str,
    env_wrapper: EnvWrapper,
) -> dict:
    """Verify a recovery demo via state-setting replay.

    Loads each recorded sim state into the environment and checks:
    1. Scene state loads without error (no version mismatch)
    2. All recorded states are loadable (dim match)
    3. Final state success matches recorded success
    4. EEF positions from state-setting match recorded positions

    Returns dict with verification results.
    """
    demo_data = np.load(demo_npz_path, allow_pickle=True)
    scene_data = np.load(scene_npz_path, allow_pickle=True)

    demo_id = str(demo_data['demo_id'])
    subtype = f"{demo_data['error_name']}_{demo_data['degree']}"
    success_recorded = bool(demo_data['success'])
    recorded_states = demo_data['states']
    recorded_eef = demo_data['eef_positions']
    num_steps = int(demo_data['num_steps'])

    result = {
        'demo_id': demo_id,
        'subtype': subtype,
        'success_recorded': success_recorded,
        'num_steps': num_steps,
        'state_dim_demo': recorded_states.shape[1] if recorded_states.ndim == 2 else 0,
    }

    # Step 1: Verify scene state loads
    if 'post_sim_state' in scene_data:
        init_state = scene_data['post_sim_state']
        result['scene_state_key'] = 'post_sim_state'
    elif 'sim_state' in scene_data:
        init_state = scene_data['sim_state']
        result['scene_state_key'] = 'sim_state'
    else:
        result['status'] = 'NO_SIM_STATE'
        return result

    result['state_dim_scene'] = len(init_state)
    result['state_dim_env'] = env_wrapper._expected_state_len

    try:
        env_wrapper.set_sim_state_flat(init_state)
        env_wrapper.forward()
        result['scene_load_ok'] = True
    except Exception as e:
        result['scene_load_ok'] = False
        result['status'] = f'SCENE_LOAD_ERROR: {e}'
        return result

    # Check: scene should NOT already be successful (error state should be broken)
    scene_already_success = env_wrapper.check_success()
    result['scene_already_success'] = scene_already_success

    # Step 2: Verify demo state-setting replay
    # Load each recorded state and check EEF position matches
    eef_errors = []
    states_loaded = 0
    state_load_errors = 0

    for i in range(len(recorded_states)):
        try:
            env_wrapper.set_sim_state_flat(recorded_states[i])
            env_wrapper.forward()
            states_loaded += 1

            # Compare EEF position
            replay_eef = env_wrapper.get_eef_pos()
            if i < len(recorded_eef):
                err = np.linalg.norm(replay_eef - recorded_eef[i])
                eef_errors.append(err)
        except Exception as e:
            state_load_errors += 1
            if state_load_errors == 1:
                result['first_state_error'] = f'frame {i}: {e}'

    result['states_loaded'] = states_loaded
    result['state_load_errors'] = state_load_errors
    result['total_states'] = len(recorded_states)

    # Step 3: Check final state success
    try:
        env_wrapper.set_sim_state_flat(recorded_states[-1])
        env_wrapper.forward()
        success_replay = env_wrapper.check_success()
    except Exception:
        success_replay = None

    result['success_replay'] = success_replay
    result['success_match'] = success_recorded == success_replay if success_replay is not None else False

    # EEF accuracy stats
    if eef_errors:
        result['max_eef_error'] = max(eef_errors)
        result['mean_eef_error'] = float(np.mean(eef_errors))
    else:
        result['max_eef_error'] = float('nan')
        result['mean_eef_error'] = float('nan')

    result['status'] = 'OK'
    return result


def main():
    parser = argparse.ArgumentParser(description="Verify recovery demo state replay")
    parser.add_argument("--task", default="stack", help="Task name")
    parser.add_argument("--subtype", default=None, help="Specific subtype to verify")
    parser.add_argument("--max_demos", type=int, default=None, help="Max demos to verify")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load task config
    task_info = load_task_registry(args.task)
    task_config_path = os.path.join(PROJECT_ROOT, task_info['task_config'])
    with open(task_config_path) as f:
        task_config = yaml.safe_load(f)
    dataset_path = task_info['dataset_path']

    # Paths
    demos_dir = PROJECT_ROOT / "error_benchmark" / "outputs" / "recovery" / "demos" / args.task
    scenes_dir = PROJECT_ROOT / "error_benchmark" / "outputs" / "v5_training" / args.task / "scenes"
    manifest_path = demos_dir / "manifest.json"

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Create environment once, reuse for all demos
    logger.info(f"Creating environment for task={args.task}...")
    env = create_env(task_config, dataset_path)
    env_wrapper = EnvWrapper(env, task_config)
    logger.info(f"Environment ready. State dim: {env_wrapper._expected_state_len}")

    # Collect demo NPZ files to verify
    demos_to_verify = []
    for demo_entry in manifest.get('demos', []):
        subtype = demo_entry['subtype_id']
        if args.subtype and subtype != args.subtype:
            continue

        demo_id = demo_entry['demo_id']
        scene_id = demo_entry['scene_id']

        # Find demo NPZ
        demo_npz = demos_dir / subtype / f"{demo_id}.npz"
        if not demo_npz.exists():
            logger.warning(f"Demo NPZ not found: {demo_npz}")
            continue

        # Find scene NPZ
        scene_npz = scenes_dir / f"{scene_id}.npz"
        if not scene_npz.exists():
            logger.warning(f"Scene NPZ not found: {scene_npz}")
            continue

        demos_to_verify.append((demo_npz, scene_npz, demo_entry))

    if args.max_demos:
        demos_to_verify = demos_to_verify[:args.max_demos]

    logger.info(f"Verifying {len(demos_to_verify)} demos...")

    # Run verification
    results = []
    for i, (demo_npz, scene_npz, entry) in enumerate(demos_to_verify):
        logger.info(f"[{i+1}/{len(demos_to_verify)}] {entry['subtype_id']} / {entry['demo_id']}")

        # Reset env before each demo
        env_wrapper._env.reset()

        result = verify_single_demo(str(demo_npz), str(scene_npz), env_wrapper)
        results.append(result)

        status = result.get('status', '?')
        if status == 'OK':
            loaded = f"{result['states_loaded']}/{result['total_states']}"
            eef_str = f"eef_err={result['max_eef_error']:.6f}m"
            success_str = f"rec={result['success_recorded']} rep={result['success_replay']}"
            match_str = "MATCH" if result['success_match'] else "MISMATCH"
            logger.info(f"  -> {match_str} | states={loaded} | {eef_str} | {success_str}")
        else:
            logger.warning(f"  -> {status}")

    # Summary table
    print("\n" + "=" * 110)
    print(f"{'SUBTYPE':<30} {'DEMO_ID':<45} {'REC':>3} {'REP':>3} {'MATCH':>5} {'STATES':>10} {'MAX_EEF_ERR':>12} {'STATUS':>8}")
    print("-" * 110)

    ok_count = 0
    match_count = 0
    all_states_ok = 0

    for r in results:
        status = r.get('status', '?')
        if status == 'OK':
            ok_count += 1
            if r['success_match']:
                match_count += 1
            if r['state_load_errors'] == 0:
                all_states_ok += 1
            print(
                f"{r['subtype']:<30} {r['demo_id']:<45} "
                f"{'Y' if r['success_recorded'] else 'N':>3} "
                f"{'Y' if r['success_replay'] else 'N':>3} "
                f"{'Y' if r['success_match'] else 'N':>5} "
                f"{r['states_loaded']}/{r['total_states']:>7} "
                f"{r['max_eef_error']:>11.6f}m "
                f"{'OK':>8}"
            )
        else:
            print(f"{r.get('subtype', '?'):<30} {r.get('demo_id', '?'):<45} {'':>3} {'':>3} {'':>5} {'':>10} {'':>12} {status}")

    print("=" * 110)
    print(f"Total: {len(results)} | Loaded OK: {ok_count} | "
          f"Success match: {match_count}/{ok_count} | "
          f"All states loadable: {all_states_ok}/{ok_count}")

    env.close()


if __name__ == "__main__":
    main()
