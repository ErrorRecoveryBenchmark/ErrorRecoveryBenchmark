#!/usr/bin/env python
"""
EnvWrapper - Environment abstraction layer

v4.0 key improvements:
- Unified wrapping of robosuite internal API
- All modules access the environment through EnvWrapper
- Supports multi-task extension (via task_config)
- Version verification and RNG state management
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
from scipy.spatial.transform import Rotation


@dataclass
class ContactInfo:
    """Contact information"""
    geom1: str
    geom2: str
    force: float
    dist: float

    def get(self, key: str, default=None):
        """Dict-like access for compatibility with detectors"""
        return getattr(self, key, default)

    def to_dict(self) -> Dict:
        """Convert to dict for serialization"""
        return {
            'geom1': self.geom1,
            'geom2': self.geom2,
            'force': self.force,
            'dist': self.dist,
        }


class EnvironmentMismatchError(Exception):
    """Environment version/configuration mismatch exception"""
    pass


class EnvWrapper:
    """
    Environment abstraction layer. Wraps robosuite env and exposes a unified semantic interface.
    All detectors/injectors/validators/collectors interact with the environment only through this class.
    """

    def __init__(self, env, task_config: dict):
        """
        Args:
            env: robosuite environment instance
            task_config: Task-related config (object names, grasp geoms, phase definitions, etc.)
        """
        self._env = env
        self._task_config = task_config
        self._control_freq = env.control_freq
        self._timestep = env.sim.model.opt.timestep

        # Cache commonly used IDs
        self._eef_site_id = env.robots[0].eef_site_id
        self._gripper_joint_ids = self._get_gripper_joint_ids()
        self._expected_state_len = len(env.sim.get_state().flatten())

        # Cache object information
        self._object_body_ids = {}
        self._object_geom_ids = {}
        if 'objects' in task_config:
            for obj in task_config['objects']:
                name = obj['name']
                body_name = obj.get('body_name', name)
                self._object_body_ids[name] = self._sim_body_name2id(body_name)

        # Detect gripper action polarity based on robot type.
        # PandaGripper: action=+1 → close, action=-1 → open
        # RethinkGripper (Sawyer): action=-1 → close, action=+1 → open
        gripper_cls = env.robots[0].gripper.__class__.__name__
        if gripper_cls == 'PandaGripper':
            self._gripper_close_action = 1.0
            self._gripper_open_action = -1.0
        else:  # RethinkGripper / default
            self._gripper_close_action = -1.0
            self._gripper_open_action = 1.0

        # Cache joint ranges (for joint limit filter)
        self._joint_ranges = []
        for i in range(env.sim.model.njnt):
            jnt_range = env.sim.model.jnt_range[i]
            self._joint_ranges.append((jnt_range[0], jnt_range[1]))

    # ─── State reading (read-only) ───

    def get_eef_pos(self) -> np.ndarray:
        """Return end-effector position with shape (3,)"""
        return self._env.sim.data.site_xpos[self._eef_site_id].copy()

    def get_eef_quat(self) -> np.ndarray:
        """
        Return end-effector quaternion with shape (4,) as (w, x, y, z)
        Converted from site_xmat (rotation matrix) to quaternion
        """
        # site_xmat is stored as flat array (9,) in MuJoCo
        rot_mat_flat = self._env.sim.data.site_xmat[self._eef_site_id].copy()
        rot_mat = rot_mat_flat.reshape(3, 3)

        # Convert rotation matrix to quaternion using scipy
        rot = Rotation.from_matrix(rot_mat)
        xquat = rot.as_quat()  # Returns (x, y, z, w)

        # Convert to (w, x, y, z)
        return np.array([xquat[3], xquat[0], xquat[1], xquat[2]])

    def get_eef_pose(self) -> np.ndarray:
        """
        Return full eef pose (7,) as [x, y, z, qw, qx, qy, qz]
        Convenient for drift detection
        """
        pos = self.get_eef_pos()
        quat = self.get_eef_quat()
        return np.concatenate([pos, quat])

    def get_eef_pose_matrix(self) -> np.ndarray:
        """Return the current end-effector pose as a 4x4 homogeneous matrix."""
        from error_benchmark.framework.recovery_mimicgen import make_pose_from_pos_quat

        return make_pose_from_pos_quat(self.get_eef_pos(), self.get_eef_quat())

    def action_to_target_pose(self, action: np.ndarray, relative: bool = True) -> np.ndarray:
        """Infer the controller target pose corresponding to an OSC action."""
        from error_benchmark.framework.recovery_mimicgen import infer_target_pose_from_action

        max_dpos = self._env.robots[0].controller.output_max[0]
        max_drot = self._env.robots[0].controller.output_max[3]
        return infer_target_pose_from_action(
            current_pose=self.get_eef_pose_matrix(),
            action=np.asarray(action),
            max_dpos=max_dpos,
            max_drot=max_drot,
            relative=relative,
        )

    def get_robot_qpos(self) -> np.ndarray:
        """Return robot joint positions"""
        return self._env.sim.data.qpos.copy()

    def get_robot_qvel(self) -> np.ndarray:
        """Return robot joint velocities"""
        return self._env.sim.data.qvel.copy()

    def get_gripper_qpos_raw(self) -> np.ndarray:
        """Return raw gripper joint positions"""
        gripper_qpos = []
        for jnt_id in self._gripper_joint_ids:
            gripper_qpos.append(self._env.sim.data.qpos[jnt_id])
        return np.array(gripper_qpos)

    def get_gripper_closed_norm(self) -> float:
        """
        Return normalized gripper closure value [0, 1], 0=fully open, 1=fully closed.
        Encapsulates normalization logic for different grippers.
        """
        gripper_cfg = self._task_config.get('gripper', {})
        gripper_type = gripper_cfg.get('type', 'two_finger')

        if gripper_type == 'two_finger':
            qpos = self.get_gripper_qpos_raw()
            if len(qpos) >= 2:
                open_qpos = gripper_cfg.get('open_qpos', [0.0, 0.0])
                close_qpos = gripper_cfg.get('close_qpos', [0.04, -0.04])

                # For symmetric grippers (e.g. Sawyer), mean(close_qpos) == mean(open_qpos) == 0
                # which causes divide-by-zero.  Use the first finger's absolute position
                # as the reference instead:
                #   open_qpos[0]=0.0, close_qpos[0]=0.04
                #   When gripping an object, qpos[0] ≈ 0.011 → norm ≈ 0.28
                finger_open = abs(open_qpos[0])
                finger_close = abs(close_qpos[0])
                finger_current = abs(qpos[0])

                if abs(finger_close - finger_open) > 1e-8:
                    norm = (finger_current - finger_open) / (finger_close - finger_open)
                else:
                    # Fallback: use total finger spread
                    spread = abs(qpos[0] - qpos[1])
                    close_spread = abs(close_qpos[0] - close_qpos[1])
                    if close_spread > 1e-8:
                        norm = spread / close_spread
                    else:
                        norm = 0.0
                return float(np.clip(norm, 0.0, 1.0))

        return 0.0

    def get_object_pose(self, obj_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (pos (3,), quat (4,)) for the specified object
        Quaternion format (w, x, y, z)
        """
        if obj_name not in self._object_body_ids:
            raise ValueError(f"Object '{obj_name}' not found in task_config")

        body_id = self._object_body_ids[obj_name]
        pos = self._env.sim.data.body_xpos[body_id].copy()
        # MuJoCo body_xquat is already (w, x, y, z)
        quat = self._env.sim.data.body_xquat[body_id].copy()

        return pos, quat

    def get_object_velocity(self, obj_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (linvel (3,), angvel (3,))"""
        if obj_name not in self._object_body_ids:
            raise ValueError(f"Object '{obj_name}' not found in task_config")

        body_id = self._object_body_ids[obj_name]
        # MuJoCo cvel layout: [angular(3), linear(3)]
        linvel = self._env.sim.data.cvel[body_id, 3:].copy()
        angvel = self._env.sim.data.cvel[body_id, :3].copy()

        return linvel, angvel

    def get_all_object_names(self) -> List[str]:
        """Return list of all manipulable object names for the current task"""
        return list(self._object_body_ids.keys())

    def get_contact_summary(self) -> List[ContactInfo]:
        """
        Return contact information list for the current step.
        ContactInfo = {geom1: str, geom2: str, force: float, dist: float}
        """
        contacts = []
        sim = self._env.sim

        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            geom1 = self._mj_id2name(sim.model, 'geom', contact.geom1)
            geom2 = self._mj_id2name(sim.model, 'geom', contact.geom2)

            # Skip if we can't get proper geom names (e.g., robot internal contacts)
            if not geom1 or not geom2:
                continue

            # Compute contact force
            force = 0.0
            if hasattr(sim.data, 'efc_force'):
                # MuJoCo >= 2.0 uses efc_force
                address = contact.efc_address
                if address >= 0 and address < len(sim.data.efc_force):
                    force = np.linalg.norm(sim.data.efc_force[address])

            contacts.append(ContactInfo(
                geom1=geom1,
                geom2=geom2,
                force=force,
                dist=contact.dist
            ))

        return contacts

    def get_task_phase(self, state_info: dict) -> str:
        """
        Determine task phase based on current state.
        Returns: 'pre_reach' | 'reach' | 'grasp' | 'lift' | 'transport' | 'done'
        Uses actual sensor readings from get_task_completion_stages().
        """
        target_object = state_info.get('target_object')
        completion = self.get_task_completion_stages(target_object=target_object)

        # Check from highest phase downward
        if completion.get('place'):
            return 'done'
        if completion.get('transport'):
            return 'transport'
        if completion.get('lift'):
            return 'lift'
        if completion.get('grasp'):
            return 'grasp'
        if completion.get('reach'):
            return 'reach'
        return 'pre_reach'

    def get_target_object(self) -> Optional[str]:
        """Infer the target object as the graspable object closest to the EEF.

        Prefers objects with non-empty grasp_geoms (skips fixtures like 'base').
        Falls back to all objects only when no graspable objects exist.
        """
        eef_pos = self.get_eef_pos()

        # Prefer graspable objects; fall back to all if none configured
        candidates = self._get_graspable_objects()

        best_name, best_dist = None, float('inf')
        failures: List[str] = []
        for name in candidates:
            try:
                pos, _ = self.get_object_pose(name)
                dist = np.linalg.norm(pos - eef_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_name = name
            except (KeyError, AttributeError, ValueError) as e:
                failures.append(f"{name}: {type(e).__name__}: {e}")
                continue
        if best_name is None and candidates:
            import logging
            logging.getLogger(__name__).warning(
                "get_target_object: no candidate resolved from %d graspable objects; "
                "failures=%s", len(candidates), failures or "none (candidates but no valid poses)"
            )
        return best_name

    # ─── State serialization and restoration ───

    def get_sim_state_flat(self) -> np.ndarray:
        """Return full MuJoCo state (for replay)"""
        return self._env.sim.get_state().flatten()

    def set_sim_state_flat(self, state: np.ndarray):
        """Restore MuJoCo state from a flat array.

        Internally calls ``sim.forward()`` and refreshes the OSC controller's
        ``ee_pos`` / ``goal_pos`` / ``goal_ori`` caches (fix from 2026-04-12:
        bypassing the controller's step loop leaves these caches stale,
        causing drift on the next step). Callers do NOT need to call
        :meth:`forward` afterwards.
        """
        if len(state) != self._expected_state_len:
            raise EnvironmentMismatchError(
                f"State dimension mismatch: got {len(state)} elements but "
                f"environment expects {self._expected_state_len} "
                f"(nq={self._env.sim.model.nq}, nv={self._env.sim.model.nv}, "
                f"na={self._env.sim.model.na}). "
                f"This typically means the trajectory was collected in a different environment."
            )
        # Restore flat array to MuJoCo state structure
        # Note: flatten() returns [time, qpos, qvel, act, udd_state]
        from collections import namedtuple

        State = namedtuple('State', ['time', 'qpos', 'qvel', 'act', 'udd_state'])
        nq = self._env.sim.model.nq
        nv = self._env.sim.model.nv
        na = self._env.sim.model.na

        # Correct indexing: extract each part from state
        # state[0] is time
        # state[1:1+nq] is qpos
        # state[1+nq:1+nq+nv] is qvel
        # state[1+nq+nv:1+nq+nv+na] is act
        # state[1+nq+nv+na:] is udd_state
        sim_state = State(
            time=float(state[0]),  # Read time from state
            qpos=state[1:1+nq],
            qvel=state[1+nq:1+nq+nv],
            act=state[1+nq+nv:1+nq+nv+na] if na > 0 else None,
            udd_state=state[1+nq+nv+na:] if na > 0 else state[1+nq+nv:]
        )

        self._env.sim.set_state(sim_state)
        self._env.sim.forward()

        # Notify robosuite controller that state has changed:
        # set_state() bypasses the controller normal step loop,
        # need to force-refresh ee_pos and reset goal to current position,
        # otherwise controller will compute torque with stale cached values.
        for robot in self._env.robots:
            robot.controller.update(force=True)
            robot.controller.goal_pos = robot.controller.ee_pos.copy()
            robot.controller.goal_ori = robot.controller.ee_ori_mat.copy()

    def get_rng_states(self) -> dict:
        """
        Return all relevant RNG states:
        {
            'numpy': np.random.get_state(),
            'env_seed': self._env.seed (if exists),
            'model_rng': ... (if applicable)
        }
        """
        rng_states = {
            'numpy': np.random.get_state(),
        }

        # Try to get env seed
        if hasattr(self._env, '_seed'):
            rng_states['env_seed'] = self._env._seed

        return rng_states

    def set_rng_states(self, rng_states: dict):
        """Restore RNG states"""
        if 'numpy' in rng_states:
            np.random.set_state(rng_states['numpy'])

        # env seed restoration needs to be set during env reset
        # Only saved here, manual handling needed when used

    # ─── Simulation control ───

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, dict]:
        """Execute one step, return (obs, reward, done, info)"""
        return self._env.step(action)

    def get_action_dim(self) -> int:
        """Return action dimension"""
        return self._env.action_dim

    def get_neutral_action(self) -> np.ndarray:
        """
        Return a neutral action that maintains the current state.
        - OSC/IK controller: maintain current eef pose
        - Joint position controller: maintain current joint pos
        """
        controller_type = self._task_config.get('controller', 'OSC')

        if controller_type == 'OSC' or controller_type == 'IK':
            # OSC/IK: return current eef pose as action
            # Note: this depends on the specific action space definition
            # Return zero action here, letting the controller decay naturally
            return np.zeros(self._env.action_dim)
        elif controller_type == 'JOINT':
            # Joint position: maintain current joint pos
            # Usually needs to return current qpos, but varies by controller
            return np.zeros(self._env.action_dim)
        else:
            # Default: zero action
            return np.zeros(self._env.action_dim)

    def check_success(self) -> bool:
        """Unified success check: prioritize env.info['success'], otherwise use task_config definition"""
        # Check if info is available
        if hasattr(self._env, '_check_success'):
            try:
                return self._env._check_success()
            except Exception:
                pass

        # Check task completion stages
        completion = self.get_task_completion_stages()
        return completion.get('place', False)

    def get_task_completion_stages(self, target_object: Optional[str] = None) -> dict:
        """
        Return task sub-goal completion status for Recovery Progress computation.

        Args:
            target_object: Specify the target object to check. If None or not in the object list,
                           falls back to obj_names[0].

        Example return:
        {
            'reach': True,
            'grasp': True,
            'lift': True,
            'transport': False,
            'place': False
        }
        """
        completion = {}

        # Get current state
        eef_pos = self.get_eef_pos()
        gripper_closed = self.get_gripper_closed_norm()

        obj_names = self.get_all_object_names()
        if not obj_names:
            return completion

        # Use specified target_object if valid, otherwise fallback to first graspable
        if target_object and target_object in obj_names:
            obj_name = target_object
        else:
            # Prefer graspable objects (skip fixtures like 'base' with no grasp_geoms)
            graspable = self._get_graspable_objects()
            obj_name = graspable[0] if graspable else obj_names[0]
        obj_pos, _ = self.get_object_pose(obj_name)

        # Compute distances
        eef_to_obj = np.linalg.norm(eef_pos - obj_pos)

        # Determine based on configured thresholds
        thresholds = self._task_config.get('thresholds', {})

        # reach: EEF close to object
        reach_threshold = thresholds.get('reach', 0.06)
        completion['reach'] = eef_to_obj < reach_threshold

        # grasp: EEF close + gripper partially closed
        # Sawyer gripper norm: ~0.025 open, ~0.08 gripping object, ~0.28 closed on nothing
        grasp_threshold = thresholds.get('grasp_closed', 0.05)
        completion['grasp'] = completion['reach'] and gripper_closed > grasp_threshold

        # lift: object lifted (above table height)
        # Primary: grasp + height check
        # Fallback: object above table AND very close to EEF (contact hold)
        lift_threshold = thresholds.get('lift_height', 0.85)
        completion['lift'] = obj_pos[2] > lift_threshold and (
            completion['grasp'] or eef_to_obj < 0.08
        )

        # transport: object close to target
        target = self._get_target_pos(obj_name)
        if target is not None:
            obj_to_target = np.linalg.norm(obj_pos[:2] - target[:2])
            transport_threshold = thresholds.get('transport', 0.05)
            completion['transport'] = completion['lift'] and obj_to_target < transport_threshold
        else:
            completion['transport'] = False

        # place: placement successful
        # Call _check_success directly to avoid circular recursion with check_success()
        if hasattr(self._env, '_check_success'):
            try:
                completion['place'] = self._env._check_success() and completion.get('transport', False)
            except Exception:
                completion['place'] = False
        else:
            # Without native success check, transport complete counts as place success
            completion['place'] = completion.get('transport', False)

        return completion

    def _get_graspable_objects(self) -> List[str]:
        """Return objects that have grasp_geoms (exclude fixtures like 'base')."""
        objects_cfg = self._task_config.get('objects', [])
        graspable = []
        for obj_cfg in objects_cfg:
            if obj_cfg.get('grasp_geoms'):  # non-empty list
                name = obj_cfg['name']
                if name in self._object_body_ids:
                    graspable.append(name)
        return graspable if graspable else list(self._object_body_ids.keys())

    def _get_target_pos(self, obj_name: str) -> Optional[np.ndarray]:
        """Get target position for an object from the robosuite env or task config."""
        # Try robosuite env attributes (e.g., PickPlace has target_bin_placements)
        env = self._env
        if hasattr(env, 'target_bin_placements') and hasattr(env, 'objects'):
            for i, obj in enumerate(env.objects):
                if obj.name == obj_name or obj_name in obj.name:
                    return env.target_bin_placements[i]

        # Try static target_pos from task config
        if 'target_pos' in self._task_config:
            return np.array(self._task_config['target_pos'])

        # Try target.pos_range center from task config
        target_cfg = self._task_config.get('target', {})
        pos_range = target_cfg.get('pos_range')
        if pos_range:
            center = np.array([
                (pos_range['x'][0] + pos_range['x'][1]) / 2,
                (pos_range['y'][0] + pos_range['y'][1]) / 2,
                (pos_range['z'][0] + pos_range['z'][1]) / 2,
            ])
            return center

        return None

    # ─── Low-level MuJoCo access (injector use only) ───

    def get_mj_model(self):
        """Return mujoco model (injector needs body_name2id etc.)"""
        return self._env.sim.model

    def get_mj_data(self):
        """Return mujoco data (injector needs xfrc_applied etc.)"""
        return self._env.sim.data

    def forward(self):
        """Call mj_forward (needed after set_state)"""
        self._env.sim.forward()

    def geom_name_to_body_name(self, geom_name: str) -> str:
        """Convert geom name to body name"""
        model = self._env.sim.model
        geom_id = model.geom_name2id(geom_name)
        body_id = model.geom_bodyid[geom_id]
        return model.body_id2name(body_id)

    # ─── Fingerprint ───

    def get_fingerprint(self) -> dict:
        """Return env_fingerprint (containing xml_hash, versions, control_freq, etc.)"""
        import robosuite
        import mujoco
        import hashlib

        # Compute XML hash
        xml_str = self._env.sim.model.get_xml()
        xml_hash = hashlib.sha1(xml_str.encode()).hexdigest()

        return {
            'env_name': self._env.__class__.__name__,
            'env_args': self._task_config.get('env_args', {}),
            'xml_hash': f"sha1:{xml_hash}",
            'robosuite_version': robosuite.__version__,
            'mujoco_version': mujoco.__version__,
            'control_freq': self._control_freq,
            'timestep': self._timestep,
            'frame_skip': getattr(self._env, 'horizon', 1) // self._control_freq if self._control_freq > 0 else 1,
        }

    @staticmethod
    def verify_fingerprint(current: dict, expected: dict):
        """
        Version/configuration consistency check.
        Raises EnvironmentMismatchError on mismatch, does not silently continue.
        Checks: robosuite_version, mujoco_version, xml_hash, control_freq, timestep
        """
        mismatches = []

        # Check required fields
        required_fields = ['robosuite_version', 'mujoco_version', 'xml_hash',
                         'control_freq', 'timestep']

        for field in required_fields:
            if field not in expected:
                continue
            if field not in current:
                mismatches.append(f"Missing field in current: {field}")
            elif current[field] != expected[field]:
                mismatches.append(
                    f"{field}: current={current[field]}, expected={expected[field]}"
                )

        if mismatches:
            error_msg = "Environment fingerprint mismatch:\n" + "\n".join(mismatches)
            raise EnvironmentMismatchError(error_msg)

    # ─── Helper methods ───

    def _get_gripper_joint_ids(self) -> List[int]:
        """Get gripper joint IDs"""
        gripper_joints = []
        for name in self._env.sim.model.joint_names:
            if 'gripper' in name.lower() or 'finger' in name.lower():
                jnt_id = self._sim_joint_name2id(name)
                if jnt_id >= 0:
                    gripper_joints.append(jnt_id)
        return gripper_joints

    def _sim_body_name2id(self, name: str) -> int:
        """Get body ID, with fallback for common MuJoCo naming conventions."""
        candidates = [name, f"{name}_main", f"{name}_body0"]
        for candidate in candidates:
            try:
                bid = self._env.sim.model.body_name2id(candidate)
                if bid >= 0:
                    return bid
            except (ValueError, KeyError, AttributeError, TypeError):
                pass
            # dict-based fallback (older mujoco-py)
            name2id = getattr(self._env.sim.model, '_body_name2id', {})
            if candidate in name2id:
                return name2id[candidate]

        # Substring fallback: search all body names for one containing the
        # object name.  This handles e.g. ThreePieceAssembly where robosuite
        # generates body names like "piece_1_0" or "assembly_piece_1_main".
        try:
            best_bid, best_len = -1, float('inf')
            for bid in range(self._env.sim.model.nbody):
                bname = self._env.sim.model.body_id2name(bid)
                if name in bname:
                    # Prefer shortest match (most specific)
                    if len(bname) < best_len:
                        best_bid, best_len = bid, len(bname)
            if best_bid >= 0:
                import logging
                logging.getLogger(__name__).debug(
                    f"Body name '{name}' resolved via substring to "
                    f"'{self._env.sim.model.body_id2name(best_bid)}'")
                return best_bid
        except (ValueError, KeyError, AttributeError, TypeError):
            pass

        import logging
        logging.getLogger(__name__).warning(
            f"Body name '{name}' not found (tried {candidates} + substring)")
        return -1

    def _sim_joint_name2id(self, name: str) -> int:
        """Get joint ID"""
        try:
            import mujoco
            return mujoco.mj_name2id(self._env.sim.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        except (ValueError, KeyError, AttributeError, TypeError):
            if name in self._env.sim.model._joint_name2id:
                return self._env.sim.model._joint_name2id[name]
            return -1

    def _mj_id2name(self, model, obj_type, obj_id):
        """Get geom/body name"""
        try:
            import mujoco
            return mujoco.mj_id2name(model, obj_type, obj_id)
        except (ValueError, KeyError, AttributeError, TypeError):
            return None

    # ─── [v5] Action-based control for Error Skills ───

    def move_eef_to(self, target_pos: np.ndarray, target_quat: Optional[np.ndarray] = None,
                    max_steps: int = 50, pos_threshold: float = 0.01,
                    rot_threshold: float = 0.05, render_fn=None) -> int:
        """
        [v5] Move EEF to target position/orientation using OSC actions.

        Uses proportional control: action = K_p * (target - current).
        Returns the number of steps taken.

        Args:
            target_pos: Target EEF position (3,)
            target_quat: Target EEF quaternion (4,) [w,x,y,z]. None = keep current.
            max_steps: Maximum steps before giving up
            pos_threshold: Position convergence threshold (m)
            rot_threshold: Rotation convergence threshold (rad)

        Returns:
            Number of steps actually taken
        """
        from error_benchmark.framework.utils.math_utils import (
            quat_multiply, quat_conjugate
        )

        k_pos = 10.0  # Proportional gain for position
        k_rot = 5.0   # Proportional gain for rotation
        action = np.zeros(self._env.action_dim)

        for step in range(max_steps):
            current_pos = self.get_eef_pos()
            pos_error = target_pos - current_pos
            pos_dist = np.linalg.norm(pos_error)

            # Compute rotation error once (used for both convergence check and action)
            rot_converged = target_quat is None
            rot_action_vec = np.zeros(3)
            if target_quat is not None:
                current_quat = self.get_eef_quat()
                q_err = quat_multiply(target_quat, quat_conjugate(current_quat))
                angle = 2 * np.arccos(np.clip(q_err[0], -1, 1))
                rot_converged = angle < rot_threshold
                if abs(angle) > 1e-6:
                    axis = q_err[1:] / np.sin(angle / 2)
                    rot_action_vec = np.clip(k_rot * angle * axis, -1.0, 1.0)

            # Check convergence
            if pos_dist < pos_threshold and rot_converged:
                return step + 1

            # Build action
            action[:] = 0
            action[:3] = np.clip(k_pos * pos_error, -1.0, 1.0)
            if target_quat is not None and len(action) > 3:
                action[3:6] = rot_action_vec

            self.step(action)
            if render_fn is not None:
                render_fn()

        return max_steps

    def get_gripper_action_close(self) -> float:
        """Return action[-1] value that closes the gripper."""
        return self._gripper_close_action

    def get_gripper_action_open(self) -> float:
        """Return action[-1] value that opens the gripper."""
        return self._gripper_open_action

    def set_gripper_state(self, open_fraction: float, steps: int = 5, render_fn=None) -> None:
        """
        [v5] Command gripper to a target open fraction.

        Args:
            open_fraction: 0.0 = fully closed, 1.0 = fully open
            steps: Number of steps to execute the gripper command
        """
        close_val = self._gripper_close_action
        open_val = self._gripper_open_action
        for _ in range(steps):
            action = np.zeros(self._env.action_dim)
            gripper_action = close_val + open_fraction * (open_val - close_val)
            action[-1] = gripper_action
            self.step(action)
            if render_fn is not None:
                render_fn()

    def restore_state(self, sim_state: np.ndarray, rng_states: Optional[dict] = None):
        """
        [v5] Restore full simulator state and optionally RNG states.

        Convenience method combining set_sim_state_flat + set_rng_states + forward.

        Args:
            sim_state: Flat MuJoCo state array
            rng_states: Optional RNG states dict
        """
        self.set_sim_state_flat(sim_state)
        if rng_states is not None:
            self.set_rng_states(rng_states)
        self.forward()

    def apply_eef_offset(self, pos_offset: np.ndarray,
                         rot_offset: Optional[np.ndarray] = None,
                         max_steps: int = 50,
                         pos_threshold: float = 0.01,
                         rot_threshold: float = 0.05,
                         render_fn=None) -> int:
        """
        [v5] Move EEF by a relative offset from current position.

        Args:
            pos_offset: Position offset (3,) in meters
            rot_offset: Rotation offset as Euler angles (3,) in radians, or None
            max_steps: Maximum steps
            pos_threshold: Position convergence threshold (m), passed to move_eef_to
            rot_threshold: Rotation convergence threshold (rad), passed to move_eef_to

        Returns:
            Number of steps taken
        """
        from error_benchmark.framework.utils.math_utils import (
            euler_to_quat, quat_multiply
        )

        target_pos = self.get_eef_pos() + pos_offset

        target_quat = None
        if rot_offset is not None and np.any(np.abs(rot_offset) > 1e-8):
            current_quat = self.get_eef_quat()
            delta_quat = euler_to_quat(rot_offset)
            target_quat = quat_multiply(delta_quat, current_quat)

        return self.move_eef_to(target_pos, target_quat, max_steps=max_steps,
                                pos_threshold=pos_threshold, rot_threshold=rot_threshold,
                                render_fn=render_fn)

    # ─── Camera observations (needed by VLA policies) ───

    def get_camera_obs(self, camera_name: str = "agentview", resolution: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """
        Render RGB image from camera.

        Args:
            camera_name: Camera name in MuJoCo model
            resolution: (width, height) tuple

        Returns:
            (H, W, 3) uint8 RGB array
        """
        return self._env.sim.render(
            width=resolution[0],
            height=resolution[1],
            camera_name=camera_name,
        )[::-1]  # flip vertically (MuJoCo renders upside-down)

    def get_all_camera_obs(self, camera_names: List[str] = None, resolution: Tuple[int, int] = (256, 256)) -> Dict[str, np.ndarray]:
        """
        Render RGB images from multiple cameras.

        Args:
            camera_names: List of camera names. Defaults to ["agentview", "robot0_eye_in_hand"].
            resolution: (width, height) tuple

        Returns:
            Dict mapping camera_name -> (H, W, 3) uint8 RGB array
        """
        if camera_names is None:
            camera_names = ["agentview", "robot0_eye_in_hand"]

        images = {}
        for name in camera_names:
            try:
                images[name] = self.get_camera_obs(name, resolution)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to render camera '{name}': {e}")
        return images

    def get_joint_ranges(self) -> List[Tuple[float, float]]:
        """Return joint ranges (for filter)"""
        return self._joint_ranges.copy()
