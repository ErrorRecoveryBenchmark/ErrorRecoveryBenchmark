#!/usr/bin/env python3
"""Serve GR00T checkpoints through RecoverBench's existing TCP policy protocol."""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path
from typing import Any

import numpy as np


STATE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")
ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
LANGUAGE_KEY = "annotation.human.action.task_description"


def build_gr00t_observation(
    observations: list[dict[str, Any]],
    prompts: list[str],
) -> dict[str, Any]:
    """Convert PolicyServerAdapter observations to GR00T's flat LIBERO batch."""
    if not observations:
        raise ValueError("at least one observation is required")
    if len(observations) != len(prompts):
        raise ValueError("observations and prompts must have equal length")

    states = []
    agent_images = []
    wrist_images = []
    for observation in observations:
        state = np.asarray(observation["state"], dtype=np.float32)
        if state.shape != (8,):
            raise ValueError(f"state must have shape (8,), found {state.shape}")
        images = observation["images"]
        agent = np.asarray(images["agentview"], dtype=np.uint8)
        wrist = np.asarray(images["robot0_eye_in_hand"], dtype=np.uint8)
        if agent.ndim != 3 or agent.shape[-1] != 3:
            raise ValueError(f"agentview must be HWC RGB, found {agent.shape}")
        if wrist.ndim != 3 or wrist.shape[-1] != 3:
            raise ValueError(f"robot0_eye_in_hand must be HWC RGB, found {wrist.shape}")

        states.append(state)
        # PolicyServerAdapter already applies the same vertical correction
        # used when the Pi0.5-derived training images were rendered.
        agent_images.append(np.ascontiguousarray(agent))
        wrist_images.append(np.ascontiguousarray(wrist))

    state_batch = np.stack(states)
    result: dict[str, Any] = {
        "video.image": np.stack(agent_images)[:, None],
        "video.wrist_image": np.stack(wrist_images)[:, None],
        LANGUAGE_KEY: list(prompts),
    }
    for index, key in enumerate(STATE_KEYS):
        result[f"state.{key}"] = state_batch[:, None, index : index + 1]
    result["state.gripper"] = state_batch[:, None, 6:8]
    return result


def actions_to_env_chunks(actions: dict[str, Any]) -> list[np.ndarray]:
    """Apply the raw robosuite gripper contract and split the batch."""
    arrays = [
        np.asarray(actions[f"action.{key}"], dtype=np.float32)
        for key in ACTION_KEYS
    ]
    shape = arrays[0].shape[:2]
    if any(array.ndim != 3 or array.shape[:2] != shape for array in arrays):
        raise ValueError("all GR00T action arrays must have matching (B, T, D) shapes")

    chunks = np.concatenate(arrays, axis=-1).astype(np.float32, copy=True)
    chunks[..., -1] = np.where(chunks[..., -1] > 0.0, 1.0, -1.0)
    return [chunk for chunk in chunks]


def _load_tcp_server_module(path: Path):
    spec = importlib.util.spec_from_file_location("recoverbench_tcp_vla_server", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load TCP server module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_model_wrapper(base_class):
    class Gr00tModelWrapper(base_class):
        def __init__(self, checkpoint: str, device: str):
            super().__init__(checkpoint, device)
            self.model_type = "gr00t_n1.6"
            self.chunk_size = 16
            self._policy = None

        def load(self) -> None:
            from gr00t.data.embodiment_tags import EmbodimentTag
            from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

            policy = Gr00tPolicy(
                embodiment_tag=EmbodimentTag.LIBERO_PANDA,
                model_path=self.checkpoint,
                device=self.device,
                strict=True,
            )
            self.chunk_size = len(
                policy.get_modality_config()["action"].delta_indices
            )
            self._policy = Gr00tSimPolicyWrapper(policy, strict=True)

        def predict(self, obs: dict[str, Any], prompt: str) -> np.ndarray:
            return self.predict_batch([obs], [prompt])[0]

        def predict_batch(
            self,
            obs_list: list[dict[str, Any]],
            prompts: list[str],
        ) -> list[np.ndarray]:
            if self._policy is None:
                raise RuntimeError("GR00T policy has not been loaded")
            observation = build_gr00t_observation(obs_list, prompts)
            actions, _ = self._policy.get_action(observation)
            return actions_to_env_chunks(actions)

    return Gr00tModelWrapper


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--max-clients", type=int, default=52)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--batch-timeout-ms", type=int, default=20)
    parser.add_argument("--inference-timeout", type=float, default=600.0)
    parser.add_argument(
        "--tcp-server-module",
        type=Path,
        default=repo_root / "archive" / "v4" / "framework" / "vla_server.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    server_module = _load_tcp_server_module(args.tcp_server_module)
    wrapper_class = _make_model_wrapper(server_module.BaseModelWrapper)
    model = wrapper_class(args.checkpoint, args.device)
    model.load()
    server = server_module.BatchedVLAServer(
        model=model,
        host=args.host,
        port=args.port,
        max_clients=args.max_clients,
        max_batch_size=args.max_batch_size,
        batch_timeout_ms=args.batch_timeout_ms,
        inference_timeout=args.inference_timeout,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
