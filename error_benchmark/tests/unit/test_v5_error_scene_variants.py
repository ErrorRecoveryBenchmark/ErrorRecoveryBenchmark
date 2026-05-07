#!/usr/bin/env python
"""Unit tests for fixed-error-scene reset variant generation."""

from types import SimpleNamespace

import numpy as np

from error_benchmark.framework.error_scene_variants import (
    ErrorSceneVariantGenerator,
)


class FakeObject:
    def __init__(self, name: str):
        self.name = name
        self.joints = [f"{name}_joint"]
        self.contact_geoms = [f"{name}_geom"]


class FakePlacementInitializer:
    def __init__(self, placements):
        self._placements = list(placements)

    def sample(self):
        return {
            f"placement_{idx}": (
                np.asarray(pos, dtype=np.float64),
                np.asarray(quat, dtype=np.float64),
                obj,
            )
            for idx, (obj, pos, quat) in enumerate(self._placements)
        }


class FakeSimData:
    def __init__(self, objects):
        self._qpos = {
            obj.joints[0]: np.zeros(7, dtype=np.float64)
            for obj in objects
        }
        self._qvel = {
            obj.joints[0]: np.zeros(6, dtype=np.float64)
            for obj in objects
        }

    def get_joint_qpos(self, joint_name: str):
        return self._qpos[joint_name].copy()

    def set_joint_qpos(self, joint_name: str, value):
        self._qpos[joint_name] = np.asarray(value, dtype=np.float64).copy()

    def get_joint_qvel(self, joint_name: str):
        return self._qvel[joint_name].copy()

    def set_joint_qvel(self, joint_name: str, value):
        self._qvel[joint_name] = np.asarray(value, dtype=np.float64).copy()


class FakeEnv:
    def __init__(
        self,
        objects,
        placements,
        grasped_objects=None,
        contact_pairs=None,
        objects_in_bins=None,
    ):
        self.objects = list(objects)
        self.object_to_id = {
            obj.name: idx for idx, obj in enumerate(self.objects)
        }
        self.placement_initializer = FakePlacementInitializer(placements)
        self.sim = SimpleNamespace(data=FakeSimData(self.objects))
        self.robots = [SimpleNamespace(gripper=object())]
        self.grasped_objects = set(grasped_objects or [])
        self.contact_pairs = {
            frozenset(pair) for pair in (contact_pairs or [])
        }
        self.objects_in_bins = np.asarray(
            objects_in_bins or [False] * len(self.objects),
            dtype=bool,
        )

    def _check_grasp(self, gripper, object_geoms):
        del gripper
        object_names = {
            geom.replace("_geom", "")
            for geom in object_geoms
        }
        return bool(object_names & self.grasped_objects)

    def check_contact(self, handle_a, handle_b):
        return frozenset({handle_a.name, handle_b.name}) in self.contact_pairs

    def _check_success(self):
        return bool(np.any(self.objects_in_bins))


class FakeEnvWrapper:
    def __init__(self, env):
        self._env = env
        self.joint_order = [obj.joints[0] for obj in env.objects]

    def set_sim_state_flat(self, state):
        state = np.asarray(state, dtype=np.float64)
        qpos_size = 7 * len(self.joint_order)
        qpos_flat = state[:qpos_size]
        qvel_flat = state[qpos_size:]

        offset = 0
        for joint_name in self.joint_order:
            self._env.sim.data.set_joint_qpos(
                joint_name,
                qpos_flat[offset:offset + 7],
            )
            offset += 7

        offset = 0
        for joint_name in self.joint_order:
            self._env.sim.data.set_joint_qvel(
                joint_name,
                qvel_flat[offset:offset + 6],
            )
            offset += 6

    def get_sim_state_flat(self):
        chunks = []
        for joint_name in self.joint_order:
            chunks.append(self._env.sim.data.get_joint_qpos(joint_name))
        for joint_name in self.joint_order:
            chunks.append(self._env.sim.data.get_joint_qvel(joint_name))
        return np.concatenate(chunks, axis=0)

    def forward(self):
        return None

    def check_success(self):
        return False


def _build_state(wrapper, qpos_by_joint, qvel_by_joint):
    chunks = []
    for joint_name in wrapper.joint_order:
        chunks.append(np.asarray(qpos_by_joint[joint_name], dtype=np.float64))
    for joint_name in wrapper.joint_order:
        chunks.append(np.asarray(qvel_by_joint[joint_name], dtype=np.float64))
    return np.concatenate(chunks, axis=0)


def _extract_joint_state(wrapper, state, joint_name):
    state = np.asarray(state, dtype=np.float64)
    joint_idx = wrapper.joint_order.index(joint_name)
    qpos_start = joint_idx * 7
    qvel_start = (7 * len(wrapper.joint_order)) + (joint_idx * 6)
    return {
        "qpos": state[qpos_start:qpos_start + 7],
        "qvel": state[qvel_start:qvel_start + 6],
    }


def _write_scene(tmp_path, scene_id, state, labels=None):
    npz_path = tmp_path / f"{scene_id}.npz"
    np.savez_compressed(npz_path, post_sim_state=state)
    return {
        "scene_id": scene_id,
        "_npz_path": str(npz_path),
        "labels": labels or {},
    }


def test_error_scene_variant_randomizes_only_sampled_objects(tmp_path):
    cube_a = FakeObject("cubeA")
    cube_b = FakeObject("cubeB")
    env = FakeEnv(
        objects=[cube_a, cube_b],
        placements=[
            (cube_a, [0.20, 0.00, 0.80], [1.0, 0.0, 0.0, 0.0]),
            (cube_b, [0.55, -0.10, 0.80], [1.0, 0.0, 0.0, 0.0]),
        ],
    )
    wrapper = FakeEnvWrapper(env)
    base_state = _build_state(
        wrapper,
        qpos_by_joint={
            "cubeA_joint": [0.00, 0.00, 0.80, 1.0, 0.0, 0.0, 0.0],
            "cubeB_joint": [0.40, 0.00, 0.80, 1.0, 0.0, 0.0, 0.0],
        },
        qvel_by_joint={
            "cubeA_joint": [1, 2, 3, 4, 5, 6],
            "cubeB_joint": [6, 5, 4, 3, 2, 1],
        },
    )
    scene = _write_scene(tmp_path, "scene_seed", base_state)
    generator = ErrorSceneVariantGenerator(
        env_wrapper=wrapper,
        task_config={"task_name": "stack"},
        rng=np.random.RandomState(0),
        scene_config={"mode": "error_scene_reset_sampler"},
    )

    variant = generator.generate_variant(scene)

    assert variant is not None
    assert variant.base_scene_id == "scene_seed"
    assert variant.variant_id == "scene_seed__variant_0000"
    assert variant.randomized_objects == ("cubeA", "cubeB")
    assert variant.anchored_objects == ()

    cube_a_state = _extract_joint_state(wrapper, variant.sim_state, "cubeA_joint")
    cube_b_state = _extract_joint_state(wrapper, variant.sim_state, "cubeB_joint")
    np.testing.assert_allclose(
        cube_a_state["qpos"],
        np.array([0.20, 0.00, 0.80, 1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        cube_b_state["qpos"],
        np.array([0.55, -0.10, 0.80, 1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(cube_a_state["qvel"], np.zeros(6))
    np.testing.assert_allclose(cube_b_state["qvel"], np.zeros(6))


def test_error_scene_variant_keeps_grasped_object_anchored(tmp_path):
    cube_a = FakeObject("cubeA")
    cube_b = FakeObject("cubeB")
    env = FakeEnv(
        objects=[cube_a, cube_b],
        placements=[
            (cube_a, [0.25, 0.00, 0.80], [1.0, 0.0, 0.0, 0.0]),
            (cube_b, [0.60, 0.00, 0.80], [1.0, 0.0, 0.0, 0.0]),
        ],
        grasped_objects={"cubeA"},
    )
    wrapper = FakeEnvWrapper(env)
    base_state = _build_state(
        wrapper,
        qpos_by_joint={
            "cubeA_joint": [0.05, 0.00, 0.80, 1.0, 0.0, 0.0, 0.0],
            "cubeB_joint": [0.45, 0.00, 0.80, 1.0, 0.0, 0.0, 0.0],
        },
        qvel_by_joint={
            "cubeA_joint": [1, 1, 1, 1, 1, 1],
            "cubeB_joint": [2, 2, 2, 2, 2, 2],
        },
    )
    scene = _write_scene(tmp_path, "scene_grasped", base_state)
    generator = ErrorSceneVariantGenerator(
        env_wrapper=wrapper,
        task_config={"task_name": "stack"},
        rng=np.random.RandomState(1),
        scene_config={"mode": "error_scene_reset_sampler"},
    )

    variant = generator.generate_variant(scene)

    assert variant is not None
    assert variant.randomized_objects == ("cubeB",)
    assert variant.anchored_objects == ("cubeA",)

    cube_a_state = _extract_joint_state(wrapper, variant.sim_state, "cubeA_joint")
    cube_b_state = _extract_joint_state(wrapper, variant.sim_state, "cubeB_joint")
    np.testing.assert_allclose(
        cube_a_state["qpos"],
        np.array([0.05, 0.00, 0.80, 1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        cube_a_state["qvel"],
        np.array([1, 1, 1, 1, 1, 1], dtype=np.float64),
    )
    np.testing.assert_allclose(
        cube_b_state["qpos"],
        np.array([0.60, 0.00, 0.80, 1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(cube_b_state["qvel"], np.zeros(6))


def test_error_scene_variant_anchors_contact_chain_from_placement_ref(tmp_path):
    cube_a = FakeObject("cubeA")
    cube_b = FakeObject("cubeB")
    cube_c = FakeObject("cubeC")
    env = FakeEnv(
        objects=[cube_a, cube_b, cube_c],
        placements=[
            (cube_a, [0.20, 0.00, 0.80], [1.0, 0.0, 0.0, 0.0]),
            (cube_b, [0.50, 0.10, 0.80], [1.0, 0.0, 0.0, 0.0]),
            (cube_c, [0.80, 0.10, 0.80], [1.0, 0.0, 0.0, 0.0]),
        ],
        contact_pairs={("cubeB", "cubeC")},
    )
    wrapper = FakeEnvWrapper(env)
    base_state = _build_state(
        wrapper,
        qpos_by_joint={
            "cubeA_joint": [0.00, 0.00, 0.80, 1.0, 0.0, 0.0, 0.0],
            "cubeB_joint": [0.40, 0.00, 0.80, 1.0, 0.0, 0.0, 0.0],
            "cubeC_joint": [0.70, 0.00, 0.80, 1.0, 0.0, 0.0, 0.0],
        },
        qvel_by_joint={
            "cubeA_joint": [1, 1, 1, 1, 1, 1],
            "cubeB_joint": [2, 2, 2, 2, 2, 2],
            "cubeC_joint": [3, 3, 3, 3, 3, 3],
        },
    )
    scene = _write_scene(
        tmp_path,
        "scene_transfer",
        base_state,
        labels={"placement_ref": "cubeB"},
    )
    generator = ErrorSceneVariantGenerator(
        env_wrapper=wrapper,
        task_config={"task_name": "stack"},
        rng=np.random.RandomState(2),
        scene_config={"mode": "error_scene_reset_sampler"},
    )

    variant = generator.generate_variant(scene)

    assert variant is not None
    assert variant.randomized_objects == ("cubeA",)
    assert variant.anchored_objects == ("cubeB", "cubeC")

    cube_a_state = _extract_joint_state(wrapper, variant.sim_state, "cubeA_joint")
    cube_b_state = _extract_joint_state(wrapper, variant.sim_state, "cubeB_joint")
    cube_c_state = _extract_joint_state(wrapper, variant.sim_state, "cubeC_joint")
    np.testing.assert_allclose(
        cube_a_state["qpos"],
        np.array([0.20, 0.00, 0.80, 1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        cube_b_state["qpos"],
        np.array([0.40, 0.00, 0.80, 1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        cube_c_state["qpos"],
        np.array([0.70, 0.00, 0.80, 1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(cube_b_state["qvel"], np.array([2, 2, 2, 2, 2, 2]))
    np.testing.assert_allclose(cube_c_state["qvel"], np.array([3, 3, 3, 3, 3, 3]))


def test_error_scene_variant_passthrough_when_mode_disabled(tmp_path):
    cube_a = FakeObject("cubeA")
    env = FakeEnv(
        objects=[cube_a],
        placements=[
            (cube_a, [0.20, 0.00, 0.80], [1.0, 0.0, 0.0, 0.0]),
        ],
    )
    wrapper = FakeEnvWrapper(env)
    base_state = _build_state(
        wrapper,
        qpos_by_joint={
            "cubeA_joint": [0.00, 0.00, 0.80, 1.0, 0.0, 0.0, 0.0],
        },
        qvel_by_joint={
            "cubeA_joint": [1, 2, 3, 4, 5, 6],
        },
    )
    scene = _write_scene(tmp_path, "scene_passthrough", base_state)
    generator = ErrorSceneVariantGenerator(
        env_wrapper=wrapper,
        task_config={"task_name": "stack"},
        rng=np.random.RandomState(3),
        scene_config={"mode": "legacy_replay"},
    )

    variant = generator.generate_variant(scene)

    assert variant is not None
    assert variant.randomized_objects == ()
    assert variant.anchored_objects == ()
    np.testing.assert_allclose(variant.sim_state, base_state)
