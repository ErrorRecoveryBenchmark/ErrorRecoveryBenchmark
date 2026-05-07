#!/usr/bin/env python
"""
Policy Adapter - Unified policy interface

Provides a unified policy interface for the error scene generation stage.

- start_episode(): Called at the start of each rollout
- predict_from_obs(obs): Predict action from robosuite obs dict
- metadata(): Return policy metadata (for scene annotation)

Supported policy types:
- RandomPolicyAdapter: Random actions (for baseline testing)
- RobomimicPolicyAdapter: Load robomimic BC-RNN checkpoint
- PolicyServerAdapter: Connect to VLA policy server
"""

import numpy as np
import logging
import socket
import struct
import pickle
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PolicyResult:
    """Policy single-step output"""
    action: np.ndarray
    info: dict = field(default_factory=dict)


class BasePolicy:
    """Policy base class"""

    def __init__(self, name: str, seed: int = 42):
        self.name = name
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def reset(self, seed: int = None):
        """Reset policy state"""
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed=seed)

    def predict(self, obs: dict) -> PolicyResult:
        """Predict action from obs"""
        raise NotImplementedError

    def get_action_dim(self) -> int:
        """Return action dimension"""
        raise NotImplementedError


class PolicyAdapter(BasePolicy):
    """
    Policy adapter base class

    Extends BasePolicy with interfaces needed for the generation stage.
    All policies used by generate_from_policy() must inherit from this class.
    """

    def __init__(self, name: str, seed: int = 42):
        super().__init__(name, seed)
        self._episode_count = 0

    def start_episode(self):
        """
        Called at the start of each rollout.
        Subclasses may override to reset internal state (e.g., RNN hidden state).
        """
        self._episode_count += 1

    def predict_from_obs(self, obs: dict) -> PolicyResult:
        """
        Predict action from robosuite observation dict.

        This is the primary interface used during generation. Defaults to delegating to predict();
        subclasses may override to handle special obs format conversion.

        Args:
            obs: observation dict returned by robosuite env.step()

        Returns:
            PolicyResult
        """
        return self.predict(obs)

    def metadata(self) -> Dict[str, Any]:
        """
        Return policy metadata for scene annotation.

        Returns:
            Dict containing policy type, name, and other info
        """
        return {
            "policy_type": self.__class__.__name__,
            "policy_name": self.name,
            "seed": self.seed,
        }


class RandomPolicyAdapter(PolicyAdapter):
    """
    Random policy adapter

    Generates uniformly distributed random actions. Used for baseline testing and pipeline verification.
    """

    def __init__(
        self,
        action_dim: int = 7,
        action_scale: float = 1.0,
        seed: int = 42,
    ):
        super().__init__("random", seed)
        self.action_dim = action_dim
        self.action_scale = action_scale

    def predict(self, obs: dict) -> PolicyResult:
        action = self.rng.uniform(
            -self.action_scale, self.action_scale, size=self.action_dim
        )
        return PolicyResult(action=action, info={})

    def get_action_dim(self) -> int:
        return self.action_dim

    def metadata(self) -> Dict[str, Any]:
        base = super().metadata()
        base.update({
            "action_dim": self.action_dim,
            "action_scale": self.action_scale,
        })
        return base


class RobomimicPolicyAdapter(PolicyAdapter):
    """
    Robomimic BC-RNN policy adapter

    Lazily loads robomimic checkpoint, performs forward inference in predict_from_obs().
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda:0",
        name: str = "bc_rnn",
        seed: int = 42,
    ):
        super().__init__(name, seed)
        self.ckpt_path = str(ckpt_path)
        self.device = device
        self._policy = None  # Lazy loading
        self._action_dim = None

    def _ensure_loaded(self):
        """Lazily load model (triggered on first call)"""
        if self._policy is not None:
            return

        import torch
        try:
            import robomimic.utils.file_utils as FileUtils
            import robomimic.utils.torch_utils as TorchUtils
        except ImportError:
            raise ImportError(
                "robomimic is required for RobomimicPolicyAdapter. "
                "Install it or activate the correct conda environment."
            )

        logger.info(f"Loading robomimic checkpoint: {self.ckpt_path}")

        device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        self._policy, _ = FileUtils.policy_from_checkpoint(
            ckpt_path=self.ckpt_path,
            device=device,
            verbose=False,
        )
        self._policy.start_episode()
        logger.info(f"Loaded robomimic policy on device={device}")

    def start_episode(self):
        """Reset RNN hidden state"""
        super().start_episode()
        self._ensure_loaded()
        self._policy.start_episode()

    def predict(self, obs: dict) -> PolicyResult:
        return self.predict_from_obs(obs)

    def predict_from_obs(self, obs: dict) -> PolicyResult:
        """
        Infer action from robosuite obs dict.

        Directly calls robomimic policy - lets robomimic internally handle observation.
        robomimic RolloutPolicy automatically handles HWC->tensor conversion, batching, etc.
        """
        self._ensure_loaded()
        robomimic_obs = self._to_robosuite_obs(obs)
        action = self._policy(ob=robomimic_obs)
        return PolicyResult(action=np.asarray(action, dtype=np.float32), info={})

    @staticmethod
    def _sanitize_obs(obs: dict) -> dict:
        """Filter unsupported values and cast low-dim floating obs to float32."""
        result = {}
        for k, v in obs.items():
            arr = np.asarray(v)
            if arr.dtype == np.dtype("O"):
                continue
            if arr.ndim == 3 and "image" in k:
                # Keep raw HWC image layout; RolloutPolicy handles modality processing.
                result[k] = arr
                continue
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32)
            result[k] = arr
        return result

    @staticmethod
    def _to_robosuite_obs(obs: dict) -> dict:
        """Convert StateExtractor state_info to robosuite obs format for robomimic.

        If '_raw_obs' is available (injected by collector), use it directly as it
        contains the exact robosuite observation format robomimic expects.

        Note: robosuite groups sensors as "{modality}-state" (e.g. "object-state"),
        but robomimic training data uses "{modality}" (e.g. "object"). We map
        grouped keys to match what the trained model expects.

        IMPORTANT: Only include keys that the policy needs (robot0_eef_pos,
        robot0_eef_quat, robot0_gripper_qpos, agentview_image, robot0_eye_in_hand_image).
        Extra keys can cause issues with robomimic's internal processing.
        """
        # Prefer raw robosuite obs (injected by collector._get_obs)
        if '_raw_obs' in obs:
            raw = dict(obs['_raw_obs'])
            # Map "object-state" → "object" for robomimic compatibility
            if 'object-state' in raw and 'object' not in raw:
                raw['object'] = raw['object-state']
            # Filter to only np.ndarray and only the keys policy needs
            raw = {k: v for k, v in raw.items()
                   if isinstance(v, np.ndarray) and k in [
                       'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
                       'agentview_image', 'robot0_eye_in_hand_image', 'object'
                   ]}
            return RobomimicPolicyAdapter._sanitize_obs(raw)

        # If already has robosuite keys, return filtered version
        if 'robot0_eef_pos' in obs:
            result = {k: v for k, v in obs.items()
                      if isinstance(v, np.ndarray) and k in [
                          'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
                          'agentview_image', 'robot0_eye_in_hand_image', 'object'
                      ]}
            # Map "object-state" → "object" for robomimic compatibility
            if 'object-state' in obs and 'object' not in result:
                result['object'] = obs['object-state']
            return RobomimicPolicyAdapter._sanitize_obs(result)

        # Convert from StateExtractor format (best effort)
        result = {}
        if 'eef_pos' in obs:
            result['robot0_eef_pos'] = np.asarray(obs['eef_pos'], dtype=np.float32)
        if 'eef_quat' in obs:
            result['robot0_eef_quat'] = np.asarray(obs['eef_quat'], dtype=np.float32)
        if 'gripper_qpos_raw' in obs:
            result['robot0_gripper_qpos'] = np.asarray(obs['gripper_qpos_raw'], dtype=np.float32)
        # Build 'object' key from StateExtractor's object info
        if 'objects' in obs:
            obj_arrays = []
            for obj_name, obj_info in obs['objects'].items():
                if 'pos' in obj_info:
                    obj_arrays.append(np.asarray(obj_info['pos'], dtype=np.float32))
                if 'quat' in obj_info:
                    obj_arrays.append(np.asarray(obj_info['quat'], dtype=np.float32))
            if obj_arrays:
                result['object'] = np.concatenate(obj_arrays)

        return RobomimicPolicyAdapter._sanitize_obs(result)

    def get_action_dim(self) -> int:
        self._ensure_loaded()
        if self._action_dim is None:
            # Get from policy action shape
            shape = self._policy.policy.nets["policy"].output_shape
            self._action_dim = shape[0] if isinstance(shape, tuple) else shape
        return self._action_dim

    def metadata(self) -> Dict[str, Any]:
        base = super().metadata()
        base.update({
            "ckpt_path": self.ckpt_path,
            "device": self.device,
        })
        return base


class PolicyServerAdapter(PolicyAdapter):
    """
    VLA Policy Server adapter

    Connects to VLA policy server (vla_server.py) via TCP socket,
    supports action chunking (server returns action chunk, client consumes step by step).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        task_prompt: str = "pick up the object",
        replan_interval: int = 5,
        connection_timeout: float = 30.0,
        name: str = "vla_policy",
        seed: int = 42,
    ):
        super().__init__(name, seed)
        self.host = host
        self.port = port
        self.task_prompt = task_prompt
        self.replan_interval = replan_interval
        self.connection_timeout = connection_timeout
        self._sock = None
        self._action_plan = []
        self._steps_since_replan = 0
        self._server_info = None

    def _ensure_connected(self):
        """Establish TCP connection to the VLA server (lazy connect)."""
        if self._sock is not None:
            return
        logger.info(f"Connecting to VLA server at {self.host}:{self.port}...")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.connection_timeout)
        self._sock.connect((self.host, self.port))
        # Query server info
        self._server_info = self._send_command({"cmd": "info"})
        logger.info(f"Connected. Server info: {self._server_info}")

    def _send_command(self, msg: dict) -> dict:
        """Send a command and receive the response."""
        payload = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
        self._sock.sendall(struct.pack("!I", len(payload)) + payload)

        raw_len = self._recv_exactly(4)
        resp_len = struct.unpack("!I", raw_len)[0]
        resp_data = self._recv_exactly(resp_len)
        return pickle.loads(resp_data)

    def _recv_exactly(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("VLA server disconnected")
            buf.extend(chunk)
        return bytes(buf)

    def start_episode(self):
        """Reset server state and clear action plan."""
        super().start_episode()
        self._ensure_connected()
        self._send_command({"cmd": "reset"})
        self._action_plan = []
        self._steps_since_replan = 0

    def predict_from_obs(self, obs: dict) -> PolicyResult:
        """
        Predict action from observation.

        If the action plan has remaining steps and we haven't hit the replan
        interval, pop the next action. Otherwise, query the server for a new
        action (or chunk).

        The obs dict should contain robosuite state. This method preprocesses
        it into the format expected by the VLA server:
            - state: 8D vector [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]
            - agentview_image: RGB (H, W, 3)
            - robot0_eye_in_hand_image: RGB (H, W, 3) (optional)
        """
        self._ensure_connected()

        # Use buffered action if available and not time to replan
        if self._action_plan and self._steps_since_replan < self.replan_interval:
            action = self._action_plan.pop(0)
            self._steps_since_replan += 1
            return PolicyResult(action=action, info={"source": "buffer"})

        # Build observation for server
        server_obs = self._preprocess_obs(obs)

        resp = self._send_command({
            "cmd": "predict",
            "obs": server_obs,
            "prompt": self.task_prompt,
        })

        if "error" in resp:
            raise RuntimeError(f"VLA server error: {resp['error']}")

        raw_action = np.asarray(resp["action"])

        # Handle action chunks: if server returns (chunk_size, action_dim),
        # take first action and buffer the rest
        if raw_action.ndim == 2:
            self._action_plan = list(raw_action[1:])
            action = raw_action[0]
        else:
            self._action_plan = []
            action = raw_action

        self._steps_since_replan = 1

        return PolicyResult(action=action, info={"source": "server"})

    def predict(self, obs: dict) -> PolicyResult:
        return self.predict_from_obs(obs)

    def get_action_dim(self) -> int:
        self._ensure_connected()
        return self._server_info.get("action_dim", 7)

    def metadata(self) -> Dict[str, Any]:
        base = super().metadata()
        base.update({
            "host": self.host,
            "port": self.port,
            "task_prompt": self.task_prompt,
            "replan_interval": self.replan_interval,
            "server_info": self._server_info,
        })
        return base

    def close(self):
        """Close the TCP connection."""
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def __del__(self):
        self.close()

    @staticmethod
    def _preprocess_obs(obs: dict) -> dict:
        """
        Convert robosuite/framework obs dict to VLA server format.

        Handles two input formats:
          1. state_info from StateExtractor: keys eef_pos, eef_quat, gripper_qpos_raw, images
          2. raw robosuite obs: keys robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos,
             agentview_image, robot0_eye_in_hand_image

        Output: {
            'state': np.ndarray (8,),  # [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]
            'images': {cam_name: np.ndarray (H, W, 3) uint8},
        }
        """
        from scipy.spatial.transform import Rotation

        result = {"images": {}}

        # ── Build 8D state vector ──
        # Try state_info format first (from StateExtractor.extract())
        eef_pos = obs.get('eef_pos')
        eef_quat = obs.get('eef_quat')
        gripper_qpos = obs.get('gripper_qpos_raw')

        if eef_pos is not None and eef_quat is not None:
            # StateExtractor quat is (w, x, y, z) — convert to scipy (x, y, z, w)
            quat_xyzw = np.array([eef_quat[1], eef_quat[2], eef_quat[3], eef_quat[0]])
            axisangle = Rotation.from_quat(quat_xyzw).as_rotvec()

            if gripper_qpos is not None:
                grip = gripper_qpos[:2] if len(gripper_qpos) >= 2 else np.pad(
                    gripper_qpos, (0, 2 - len(gripper_qpos))
                )
            else:
                grip = np.zeros(2)

            result['state'] = np.concatenate([eef_pos, axisangle, grip]).astype(np.float32)

        # Fallback: robosuite raw obs format
        if 'state' not in result:
            r_eef_pos = obs.get('robot0_eef_pos')
            r_eef_quat = obs.get('robot0_eef_quat')
            r_grip = obs.get('robot0_gripper_qpos')

            if r_eef_pos is not None and r_eef_quat is not None:
                # robosuite quat is (x, y, z, w)
                q = np.asarray(r_eef_quat, dtype=np.float64)
                norm = np.linalg.norm(q)
                if norm > 0:
                    q = q / norm
                axisangle = Rotation.from_quat(q).as_rotvec()

                grip = r_grip[:2] if r_grip is not None and len(r_grip) >= 2 else np.zeros(2)

                result['state'] = np.concatenate([
                    np.asarray(r_eef_pos), axisangle, grip
                ]).astype(np.float32)

        # ── Images ──
        # Preprocessing: vertical flip + resize 224x224 (match training data)
        # See zhaoganlong's phoenix/main.py for reference
        RESIZE_SIZE = 224

        def preprocess_image(img: np.ndarray) -> np.ndarray:
            """Apply training-compatible preprocessing: vertical flip + resize."""
            # 1. Convert to uint8 if needed
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            # 2. Vertical flip (match training data augmentation)
            img = np.ascontiguousarray(img[::-1])
            # 3. Resize to 224x224 with padding (match model input)
            from PIL import Image
            pil_img = Image.fromarray(img)
            cur_w, cur_h = pil_img.size
            if cur_w != RESIZE_SIZE or cur_h != RESIZE_SIZE:
                ratio = max(cur_w / RESIZE_SIZE, cur_h / RESIZE_SIZE)
                new_w = int(cur_w / ratio)
                new_h = int(cur_h / ratio)
                pil_img = pil_img.resize((new_w, new_h), resample=Image.BILINEAR)
                # Pad with zeros
                padded = Image.new(pil_img.mode, (RESIZE_SIZE, RESIZE_SIZE), 0)
                pad_h = max(0, int((RESIZE_SIZE - new_h) / 2))
                pad_w = max(0, int((RESIZE_SIZE - new_w) / 2))
                padded.paste(pil_img, (pad_w, pad_h))
                img = np.asarray(padded)
            return img

        # Try multiple key patterns for each camera
        for cam in ["agentview", "robot0_eye_in_hand"]:
            for key_pattern in [f"{cam}_image", cam]:
                if key_pattern in obs:
                    img = np.asarray(obs[key_pattern])
                    result["images"][cam] = preprocess_image(img)
                    break

        # Also check images sub-dict (from StateExtractor with include_images=True)
        images_dict = obs.get('images', {})
        for cam in ["agentview", "robot0_eye_in_hand"]:
            if cam not in result["images"] and cam in images_dict:
                img = np.asarray(images_dict[cam])
                result["images"][cam] = preprocess_image(img)

        return result


def create_policy_adapter(policy_config: dict, env_action_dim: int = 7, seed: int = 42) -> PolicyAdapter:
    """
    Factory function: create policy adapter from config

    Args:
        policy_config: Policy configuration dict
            {
                'type': 'random' | 'bc_rnn',
                'ckpt_path': ... (for bc_rnn),
                'action_dim': ... (for random),
                'action_scale': ... (for random),
                'device': ... (for bc_rnn),
            }
        env_action_dim: Environment action dimension (default for random policy)
        seed: Random seed

    Returns:
        PolicyAdapter instance
    """
    policy_type = policy_config.get('type', 'random')

    if policy_type == 'random':
        return RandomPolicyAdapter(
            action_dim=policy_config.get('action_dim', env_action_dim),
            action_scale=policy_config.get('action_scale', 1.0),
            seed=seed,
        )
    elif policy_type in ('bc_rnn', 'robomimic'):
        ckpt_path = policy_config.get('ckpt_path')
        if not ckpt_path:
            raise ValueError("bc_rnn policy requires 'ckpt_path'")
        return RobomimicPolicyAdapter(
            ckpt_path=ckpt_path,
            device=policy_config.get('device', 'cuda:0'),
            name=policy_config.get('name', 'bc_rnn'),
            seed=seed,
        )
    elif policy_type == 'vla_server':
        return PolicyServerAdapter(
            host=policy_config.get('host', 'localhost'),
            port=policy_config.get('port', 5555),
            task_prompt=policy_config.get('task_prompt', 'pick up the object'),
            replan_interval=policy_config.get('replan_interval', 5),
            connection_timeout=policy_config.get('connection_timeout', 30.0),
            name=policy_config.get('name', 'vla_policy'),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown policy adapter type: {policy_type}")
