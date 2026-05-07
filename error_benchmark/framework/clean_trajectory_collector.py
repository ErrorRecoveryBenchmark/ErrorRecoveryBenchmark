#!/usr/bin/env python
"""
Clean Trajectory Collector - v5.0

Collects clean (successful) trajectories from:
    1. Human demonstration HDF5 datasets
    2. Successful VLA/BC policy rollouts

These trajectories are used by the OpportunityScanner for offline
error injection planning.
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from datetime import datetime
import json

if TYPE_CHECKING:
    from error_benchmark.framework.env_wrapper import EnvWrapper

from error_benchmark.framework.core import CleanTrajectory, InteractionSegment

logger = logging.getLogger(__name__)


class CleanTrajectoryCollector:
    """
    Collects clean trajectories from demos and policy rollouts.

    Usage:
        collector = CleanTrajectoryCollector(env, task_config)
        trajectories = collector.collect_from_demos(dataset_path, num_demos=20)
        trajectories += collector.collect_from_rollouts(policy_adapter, num_rollouts=10)
        collector.save(trajectories, output_dir)
    """

    def __init__(
        self,
        env: 'EnvWrapper',
        task_config: dict,
        task_name: str = "pick_place",
    ):
        self.env = env
        self.task_config = task_config
        self.task_name = task_name
        self.logger = logging.getLogger(f"{__name__}.{task_name}")

    def collect_from_demos(
        self,
        dataset_path: str,
        num_demos: int = 20,
        seed: int = 42,
    ) -> List[CleanTrajectory]:
        """
        Extract clean trajectories from HDF5 demonstration dataset.

        Args:
            dataset_path: Path to HDF5 dataset (MimicGen format)
            num_demos: Number of demos to extract
            seed: Random seed

        Returns:
            List of CleanTrajectory objects
        """
        import h5py

        trajectories = []
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            self.logger.error(f"Dataset not found: {dataset_path}")
            return trajectories

        with h5py.File(dataset_path, 'r') as f:
            demo_keys = sorted([
                k for k in f['data'].keys() if k.startswith('demo_')
            ])[:num_demos]

            for demo_key in demo_keys:
                actions = f[f'data/{demo_key}/actions'][()]
                states = f[f'data/{demo_key}/states'][()]

                trajectory_id = f"demo_{self.task_name}_{demo_key}"

                traj = CleanTrajectory(
                    trajectory_id=trajectory_id,
                    source="demo",
                    task_name=self.task_name,
                    dataset_path=str(dataset_path),
                    demo_key=demo_key,
                    num_steps=len(actions),
                    actions=actions,
                    states=states,
                )

                trajectories.append(traj)
                self.logger.info(
                    f"Collected demo trajectory: {trajectory_id} "
                    f"({len(actions)} steps)"
                )

        self.logger.info(
            f"Collected {len(trajectories)} demo trajectories from {dataset_path}"
        )
        return trajectories

    def collect_from_rollouts(
        self,
        policy_adapter,
        min_successes: int = 10,
        max_attempts: int = 100,
        max_steps: int = 400,
        policy_name: str = "vla_pi0",
        seed: int = 42,
    ) -> List[CleanTrajectory]:
        """
        Collect successful rollouts from a policy.

        Keeps rolling out until min_successes are collected or max_attempts
        is reached.

        Args:
            policy_adapter: PolicyAdapter instance
            min_successes: Minimum number of successful rollouts to collect
            max_attempts: Maximum number of rollout attempts
            max_steps: Maximum steps per rollout
            policy_name: Policy identifier
            seed: Random seed

        Returns:
            List of CleanTrajectory objects (only successful ones)
        """
        rng = np.random.RandomState(seed)
        trajectories = []
        total_attempts = 0

        while len(trajectories) < min_successes and total_attempts < max_attempts:
            total_attempts += 1
            env_seed = int(rng.randint(0, 2**31))

            # Reset environment
            obs = self.env._env.reset()
            policy_adapter.reset()

            # Reset RNN hidden state for BC-RNN policies
            if hasattr(policy_adapter, 'start_episode'):
                policy_adapter.start_episode()

            init_state = self.env.get_sim_state_flat()
            actions_list = []
            states_list = [init_state.copy()]

            success = False
            for step in range(max_steps):
                # predict() returns PolicyResult; unwrap .action
                result = policy_adapter.predict(obs)
                action = result.action if hasattr(result, 'action') else result

                obs, reward, done, info = self.env.step(action)

                actions_list.append(action.copy())
                states_list.append(self.env.get_sim_state_flat().copy())

                if self.env.check_success():
                    success = True
                    break

                if done:
                    break

            if success:
                trajectory_id = (
                    f"rollout_{self.task_name}_{policy_name}_{len(trajectories)}"
                )
                traj = CleanTrajectory(
                    trajectory_id=trajectory_id,
                    source=policy_name,
                    task_name=self.task_name,
                    num_steps=len(actions_list),
                    actions=np.array(actions_list),
                    states=np.array(states_list),
                )
                trajectories.append(traj)
                self.logger.info(
                    f"Collected successful rollout: {trajectory_id} "
                    f"({len(actions_list)} steps) "
                    f"[{len(trajectories)}/{min_successes}]"
                )
            else:
                self.logger.debug(
                    f"Rollout attempt {total_attempts} failed "
                    f"({len(actions_list)} steps)"
                )

        self.logger.info(
            f"Collected {len(trajectories)}/{total_attempts} successful "
            f"rollouts from {policy_name}"
        )
        return trajectories

    def replay_and_label_phases(
        self,
        trajectory: CleanTrajectory,
    ) -> CleanTrajectory:
        """
        Replay a trajectory in the environment and label each frame's task phase.

        Args:
            trajectory: CleanTrajectory with actions and states

        Returns:
            Same trajectory with phase_labels populated
        """
        if trajectory.states is None or trajectory.actions is None:
            self.logger.warning(
                f"Cannot label phases for {trajectory.trajectory_id}: "
                "missing states/actions"
            )
            return trajectory

        # Restore initial state
        self.env.set_sim_state_flat(trajectory.states[0])
        self.env.forward()

        phase_labels = []
        for step in range(trajectory.num_steps):
            # Get phase before executing action
            target_object = self.env.get_target_object()
            state_info = {'step': step, 'target_object': target_object}
            phase = self.env.get_task_phase(state_info)
            phase_labels.append(phase)

            # Execute action
            self.env.step(trajectory.actions[step])

        # Final phase after last action
        target_object = self.env.get_target_object()
        phase_labels.append(self.env.get_task_phase({
            'step': trajectory.num_steps, 'target_object': target_object
        }))

        trajectory.phase_labels = phase_labels
        return trajectory

    def segment_interactions(
        self,
        trajectory: CleanTrajectory,
    ) -> CleanTrajectory:
        """
        Segment a trajectory by object interaction.

        Must be called after replay_and_label_phases() or with valid states/actions.
        Populates trajectory.interaction_segments.

        Args:
            trajectory: CleanTrajectory with actions and states

        Returns:
            Same trajectory with interaction_segments populated
        """
        if trajectory.states is None or trajectory.actions is None:
            self.logger.warning(
                f"Cannot segment {trajectory.trajectory_id}: missing states/actions"
            )
            return trajectory

        from error_benchmark.framework.interaction_segmenter import InteractionSegmenter
        segmenter = InteractionSegmenter(self.task_config)
        trajectory.interaction_segments = segmenter.segment(trajectory, self.env)
        return trajectory

    @staticmethod
    def save(
        trajectories: List[CleanTrajectory],
        output_dir: str,
    ):
        """
        Save trajectories to disk (metadata JSON + data NPZ).

        Args:
            trajectories: List of CleanTrajectory objects
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        manifest = []
        for traj in trajectories:
            # Save data arrays
            npz_name = f"{traj.trajectory_id}.npz"
            npz_path = output_path / npz_name

            save_dict = {}
            if traj.actions is not None:
                save_dict['actions'] = traj.actions
            if traj.states is not None:
                save_dict['states'] = traj.states
            if traj.phase_labels is not None:
                save_dict['phase_labels'] = np.array(traj.phase_labels, dtype='U20')
            if traj.interaction_segments is not None:
                # Serialize segments as JSON string in NPZ
                save_dict['interaction_segments_json'] = json.dumps(
                    [s.to_dict() for s in traj.interaction_segments]
                )

            np.savez_compressed(npz_path, **save_dict)

            # Build manifest entry
            entry = traj.to_dict()
            entry['npz_path'] = str(npz_path)
            manifest.append(entry)

        # Save manifest
        manifest_path = output_path / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump({
                'created': datetime.now().isoformat(),
                'num_trajectories': len(trajectories),
                'trajectories': manifest,
            }, f, indent=2)

        logger.info(f"Saved {len(trajectories)} trajectories to {output_path}")

    @staticmethod
    def load(input_dir: str) -> List[CleanTrajectory]:
        """
        Load trajectories from disk.

        Args:
            input_dir: Directory containing manifest.json and NPZ files

        Returns:
            List of CleanTrajectory objects
        """
        input_path = Path(input_dir)
        manifest_path = input_path / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        trajectories = []
        for entry in manifest['trajectories']:
            traj = CleanTrajectory.from_dict(entry)

            # Load data arrays
            npz_path = entry.get('npz_path', '')
            if npz_path and Path(npz_path).exists():
                data = np.load(npz_path, allow_pickle=True)
                if 'actions' in data:
                    traj.actions = data['actions']
                if 'states' in data:
                    traj.states = data['states']
                if 'phase_labels' in data:
                    traj.phase_labels = list(data['phase_labels'])
                if 'interaction_segments_json' in data:
                    seg_list = json.loads(str(data['interaction_segments_json']))
                    traj.interaction_segments = [
                        InteractionSegment.from_dict(s) for s in seg_list
                    ]

            trajectories.append(traj)

        logger.info(f"Loaded {len(trajectories)} trajectories from {input_path}")
        return trajectories
