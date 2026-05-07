"""Generate MimicGen-style reset variants from a fixed error scene seed."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

from error_benchmark.framework.recovery_mimicgen import resolve_placement_ref

_PLACEMENT_PHASE_GROUPS = {"transfer"}
_PLACEMENT_TASK_PHASES = {"transport", "place", "done"}


@dataclass(frozen=True)
class ErrorSceneVariant:
    """A single randomized variant derived from a base error scene."""

    base_scene_id: str
    variant_id: str
    variant_index: int
    sim_state: np.ndarray
    randomized_objects: tuple[str, ...]
    anchored_objects: tuple[str, ...]
    generation_attempts: int


class ErrorSceneVariantGenerator:
    """
    Reuse the task's reset sampler on top of a fixed post-error state.

    The base error scene provides the robot state and task progress. We then
    resample only placement-initialized task objects, while keeping held or
    already-progress-critical objects anchored to the seed scene.
    """

    def __init__(
        self,
        env_wrapper,
        task_config: Optional[dict],
        rng: Optional[np.random.RandomState],
        scene_config: Optional[dict] = None,
    ) -> None:
        self.env_wrapper = env_wrapper
        self.task_config = task_config or {}
        self.rng = rng if rng is not None else np.random.RandomState(0)
        self.scene_config = scene_config or {}
        self.mode = self.scene_config.get("mode", "error_scene_reset_sampler")
        self.max_variant_attempts = int(
            self.scene_config.get("max_variant_attempts_per_scene", 8)
        )
        self.min_object_delta = float(
            self.scene_config.get("min_variant_object_delta", 1e-4)
        )
        self.allow_passthrough_on_no_variant = bool(
            self.scene_config.get("allow_passthrough_on_no_variant", True)
        )
        self._scene_variant_counts: dict[str, int] = defaultdict(int)
        self._scene_state_cache: dict[str, np.ndarray] = {}

    # Modes that enable object-position randomization (vs. passthrough).
    _RANDOMIZE_MODES = {"error_scene_reset_sampler", "source_scene_variant"}

    def generate_variant(
        self,
        base_scene: dict,
        override_base_state: Optional[np.ndarray] = None,
    ) -> Optional[ErrorSceneVariant]:
        """Generate one variant from a base error scene.

        Args:
            base_scene: Scene dict with ``_npz_path`` and optional ``labels``.
            override_base_state: If provided, use this sim state instead of the
                scene NPZ's ``post_sim_state``.  This is the demo's
                ``states[0]`` — the actual robot configuration when the human
                started teleoperating — which may differ from the scene NPZ
                snapshot (e.g. after injection replay / OSC warm-up).
        """

        if self.mode not in self._RANDOMIZE_MODES:
            return self._build_passthrough_variant(base_scene, override_base_state)

        scene_id = str(base_scene.get("scene_id", "unknown_scene"))
        for attempt in range(1, self.max_variant_attempts + 1):
            base_state = (
                override_base_state.copy()
                if override_base_state is not None
                else self._load_base_state(base_scene)
            )
            self.env_wrapper.set_sim_state_flat(base_state)
            self.env_wrapper.forward()

            sampled_objects = self._sample_object_states()
            if not sampled_objects:
                return self._build_passthrough_variant(base_scene)

            anchored_objects = self._get_anchored_objects(
                base_scene=base_scene,
                sampled_object_names=sampled_objects.keys(),
            )
            applied_objects = self._apply_object_states(
                sampled_objects=sampled_objects,
                anchored_objects=anchored_objects,
            )
            if not applied_objects:
                continue

            self.env_wrapper.forward()
            if bool(self.env_wrapper.check_success()):
                continue

            variant_index = self._scene_variant_counts[scene_id]
            self._scene_variant_counts[scene_id] += 1
            variant_id = f"{scene_id}__variant_{variant_index:04d}"
            return ErrorSceneVariant(
                base_scene_id=scene_id,
                variant_id=variant_id,
                variant_index=variant_index,
                sim_state=self.env_wrapper.get_sim_state_flat().copy(),
                randomized_objects=tuple(sorted(applied_objects)),
                anchored_objects=tuple(sorted(anchored_objects)),
                generation_attempts=attempt,
            )

        if self.allow_passthrough_on_no_variant:
            return self._build_passthrough_variant(base_scene, override_base_state)

        base_state = (
            override_base_state.copy()
            if override_base_state is not None
            else self._load_base_state(base_scene)
        )
        self.env_wrapper.set_sim_state_flat(base_state)
        self.env_wrapper.forward()
        return None

    def _build_passthrough_variant(
        self,
        base_scene: dict,
        override_base_state: Optional[np.ndarray] = None,
    ) -> ErrorSceneVariant:
        """Fallback variant that keeps the base error scene unchanged."""

        scene_id = str(base_scene.get("scene_id", "unknown_scene"))
        variant_index = self._scene_variant_counts[scene_id]
        self._scene_variant_counts[scene_id] += 1
        base_state = (
            override_base_state.copy()
            if override_base_state is not None
            else self._load_base_state(base_scene)
        )
        self.env_wrapper.set_sim_state_flat(base_state)
        self.env_wrapper.forward()
        return ErrorSceneVariant(
            base_scene_id=scene_id,
            variant_id=f"{scene_id}__variant_{variant_index:04d}",
            variant_index=variant_index,
            sim_state=self.env_wrapper.get_sim_state_flat().copy(),
            randomized_objects=tuple(),
            anchored_objects=tuple(),
            generation_attempts=1,
        )

    def _load_base_state(self, base_scene: dict) -> np.ndarray:
        npz_path = str(base_scene.get("_npz_path", ""))
        if npz_path not in self._scene_state_cache:
            if not npz_path:
                raise FileNotFoundError("Base scene is missing _npz_path")
            npz_data = np.load(npz_path)
            if "post_sim_state" in npz_data:
                state = np.array(npz_data["post_sim_state"], dtype=np.float64)
            else:
                state = np.array(npz_data["sim_state"], dtype=np.float64)
            self._scene_state_cache[npz_path] = state
        return self._scene_state_cache[npz_path].copy()

    def _sample_object_states(self) -> dict[str, dict]:
        env = self.env_wrapper._env
        if not hasattr(env, "placement_initializer") or env.placement_initializer is None:
            return {}

        placements = env.placement_initializer.sample()
        sampled: dict[str, dict] = {}
        for obj_pos, obj_quat, obj in placements.values():
            joints = list(getattr(obj, "joints", []) or [])
            if not joints:
                continue
            joint_name = joints[0]
            sampled[str(getattr(obj, "name", joint_name))] = {
                "joint_name": joint_name,
                "qpos": np.concatenate(
                    [np.asarray(obj_pos, dtype=np.float64), np.asarray(obj_quat, dtype=np.float64)]
                ),
            }
        return sampled

    def _get_anchored_objects(
        self,
        base_scene: dict,
        sampled_object_names: Iterable[str],
    ) -> set[str]:
        sampled_names = set(sampled_object_names)
        anchored = set()
        anchored.update(self._get_grasped_objects(sampled_names))
        anchored.update(self._get_goal_locked_objects(sampled_names))

        placement_ref = self._get_phase_locked_placement_ref(base_scene)
        if placement_ref in sampled_names:
            anchored.add(placement_ref)

        changed = True
        while changed:
            changed = False
            for obj_name in sorted(sampled_names - anchored):
                if any(
                    self._objects_in_contact(obj_name, anchored_name)
                    for anchored_name in anchored
                ):
                    anchored.add(obj_name)
                    changed = True

        return anchored

    def _get_grasped_objects(self, sampled_names: set[str]) -> set[str]:
        return {
            obj_name
            for obj_name in sampled_names
            if self._object_is_grasped(obj_name)
        }

    def _get_goal_locked_objects(self, sampled_names: set[str]) -> set[str]:
        env = self.env_wrapper._env
        if not hasattr(env, "objects_in_bins") or not hasattr(env, "objects"):
            return set()

        try:
            env._check_success()
        except Exception:
            return set()

        locked = set()
        objects_in_bins = np.asarray(env.objects_in_bins).astype(bool).tolist()
        for idx, in_bin in enumerate(objects_in_bins):
            if not in_bin:
                continue
            try:
                locked.add(str(env.objects[idx].name))
            except Exception:
                continue
        return locked & sampled_names

    def _get_phase_locked_placement_ref(self, base_scene: dict) -> Optional[str]:
        scene_labels = dict(base_scene.get("labels", {}) or {})
        task_phase = str(scene_labels.get("task_phase", ""))
        phase_group = str(scene_labels.get("phase_group", ""))

        explicit_ref = scene_labels.get("placement_ref")
        if explicit_ref:
            return str(explicit_ref)

        if (
            phase_group not in _PLACEMENT_PHASE_GROUPS
            and task_phase not in _PLACEMENT_TASK_PHASES
        ):
            return None

        task_name = str(self.task_config.get("task_name", ""))
        target_object = str(scene_labels.get("target_object", ""))
        if not task_name or not target_object:
            return None
        return resolve_placement_ref(
            task_name=task_name,
            task_config=self.task_config,
            target_object=target_object,
            scene_labels=scene_labels,
        )

    def _apply_object_states(
        self,
        sampled_objects: dict[str, dict],
        anchored_objects: set[str],
    ) -> set[str]:
        env = self.env_wrapper._env
        sim_data = env.sim.data
        applied: set[str] = set()

        for obj_name, payload in sampled_objects.items():
            if obj_name in anchored_objects:
                continue

            joint_name = payload["joint_name"]
            new_qpos = np.asarray(payload["qpos"], dtype=np.float64)
            cur_qpos = np.asarray(sim_data.get_joint_qpos(joint_name), dtype=np.float64).copy()
            if np.linalg.norm(new_qpos - cur_qpos) < self.min_object_delta:
                continue

            sim_data.set_joint_qpos(joint_name, new_qpos)
            cur_qvel = np.asarray(sim_data.get_joint_qvel(joint_name), dtype=np.float64)
            sim_data.set_joint_qvel(joint_name, np.zeros_like(cur_qvel))
            applied.add(obj_name)

        return applied

    def _object_is_grasped(self, obj_name: str) -> bool:
        env = self.env_wrapper._env
        if not hasattr(env, "_check_grasp"):
            return False

        object_geoms = self._resolve_object_contact_geoms(obj_name)
        if not object_geoms:
            return False

        try:
            return bool(
                env._check_grasp(
                    gripper=env.robots[0].gripper,
                    object_geoms=object_geoms,
                )
            )
        except Exception:
            return False

    def _objects_in_contact(self, obj_a: str, obj_b: str) -> bool:
        env = self.env_wrapper._env
        if not hasattr(env, "check_contact"):
            return False

        handle_a = self._resolve_object_handle(obj_a)
        handle_b = self._resolve_object_handle(obj_b)
        if handle_a is None or handle_b is None:
            return False

        try:
            return bool(env.check_contact(handle_a, handle_b))
        except Exception:
            return False

    def _resolve_object_contact_geoms(self, obj_name: str):
        env = self.env_wrapper._env
        if hasattr(env, "object_to_id") and hasattr(env, "objects"):
            object_to_id = getattr(env, "object_to_id", {})
            if obj_name in object_to_id:
                obj = env.objects[object_to_id[obj_name]]
                if hasattr(obj, "contact_geoms"):
                    return [g for g in obj.contact_geoms]

        if hasattr(env, obj_name):
            obj = getattr(env, obj_name)
            if hasattr(obj, "contact_geoms"):
                return [g for g in obj.contact_geoms]
        return None

    def _resolve_object_handle(self, obj_name: str):
        env = self.env_wrapper._env
        if hasattr(env, "object_to_id") and hasattr(env, "objects"):
            object_to_id = getattr(env, "object_to_id", {})
            if obj_name in object_to_id:
                return env.objects[object_to_id[obj_name]]

        if hasattr(env, obj_name):
            return getattr(env, obj_name)
        return None
