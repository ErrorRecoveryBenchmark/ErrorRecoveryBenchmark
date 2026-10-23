#!/usr/bin/env python
"""Serve an OpenVLA-OFT checkpoint over the benchmark TCP protocol."""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from error_benchmark.framework.vla_server import (
    BaseModelWrapper,
    BatchedVLAServer,
    VLAServer,
)

logger = logging.getLogger(__name__)


@dataclass
class OpenVLAOFTConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: str = ""
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    unnorm_key: str = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False


def postprocess_openvla_actions(actions: np.ndarray) -> np.ndarray:
    """Convert OpenVLA's [0, 1] gripper output to robosuite convention."""
    processed = np.asarray(actions, dtype=np.float32).copy()
    if processed.ndim not in (1, 2) or processed.shape[-1] != 7:
        raise ValueError(f"Expected (..., 7) actions, got {processed.shape}")
    processed[..., -1] = -np.sign(2.0 * processed[..., -1] - 1.0)
    return processed


class OpenVLAOFTModelWrapper(BaseModelWrapper):
    def __init__(self, checkpoint: str, unnorm_key: str, device: str = "cuda:0"):
        super().__init__(checkpoint, device)
        self.model_type = "openvla_oft"
        self.action_dim = 7
        self.chunk_size = 8
        self.cfg = OpenVLAOFTConfig(
            pretrained_checkpoint=checkpoint,
            unnorm_key=unnorm_key,
        )
        self._vla = None
        self._processor = None
        self._action_head = None
        self._proprio_projector = None
        self._get_vla_action = None

    def load(self) -> None:
        if self.device != "cuda:0":
            raise ValueError(
                "OpenVLA-OFT utilities use logical cuda:0; select the physical GPU "
                "with CUDA_VISIBLE_DEVICES."
            )

        from experiments.robot.openvla_utils import (
            get_action_head,
            get_processor,
            get_proprio_projector,
            get_vla,
            get_vla_action,
        )

        logger.info("Loading OpenVLA-OFT from %s", self.checkpoint)
        self._vla = get_vla(self.cfg)
        self._processor = get_processor(self.cfg)
        self._action_head = get_action_head(self.cfg, self._vla.llm_dim)
        self._proprio_projector = get_proprio_projector(
            self.cfg,
            self._vla.llm_dim,
            proprio_dim=8,
        )
        self._get_vla_action = get_vla_action
        if self.cfg.unnorm_key not in self._vla.norm_stats:
            raise KeyError(
                f"Unnormalization key {self.cfg.unnorm_key!r} is absent from "
                f"{self.checkpoint}/dataset_statistics.json"
            )

    @staticmethod
    def _observation(obs: dict) -> dict[str, Any]:
        images = obs.get("images", {})
        missing = [
            key
            for key in ("agentview", "robot0_eye_in_hand")
            if key not in images
        ]
        if missing:
            raise KeyError(f"Missing OpenVLA image(s): {', '.join(missing)}")
        state = np.asarray(obs.get("state"), dtype=np.float32)
        if state.shape != (8,):
            raise ValueError(f"Expected 8D proprio state, got {state.shape}")
        return {
            "full_image": np.asarray(images["agentview"], dtype=np.uint8),
            "wrist_image": np.asarray(
                images["robot0_eye_in_hand"], dtype=np.uint8
            ),
            "state": state.copy(),
        }

    def predict(self, obs: dict, prompt: str) -> np.ndarray:
        actions = self._get_vla_action(
            cfg=self.cfg,
            vla=self._vla,
            processor=self._processor,
            obs=self._observation(obs),
            task_label=prompt,
            action_head=self._action_head,
            proprio_projector=self._proprio_projector,
            use_film=False,
        )
        processed = postprocess_openvla_actions(np.asarray(actions))
        if processed.shape != (self.chunk_size, self.action_dim):
            raise ValueError(
                f"Expected {(self.chunk_size, self.action_dim)} action chunk, "
                f"got {processed.shape}"
            )
        return processed

    def predict_batch(
        self, obs_list: list[dict], prompts: list[str]
    ) -> list[np.ndarray]:
        return [self.predict(obs, prompt) for obs, prompt in zip(obs_list, prompts)]

    def info(self) -> dict:
        info = super().info()
        info.update(
            {
                "unnorm_key": self.cfg.unnorm_key,
                "num_images_in_input": self.cfg.num_images_in_input,
                "use_proprio": self.cfg.use_proprio,
                "center_crop": self.cfg.center_crop,
            }
        )
        return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--unnorm-key", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batched", action="store_true")
    parser.add_argument("--max-clients", type=int, default=16)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--batch-timeout-ms", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    model = OpenVLAOFTModelWrapper(
        checkpoint=checkpoint,
        unnorm_key=args.unnorm_key,
        device=args.device,
    )
    model.load()
    if args.batched:
        server = BatchedVLAServer(
            model,
            host=args.host,
            port=args.port,
            max_clients=args.max_clients,
            max_batch_size=args.max_batch_size,
            batch_timeout_ms=args.batch_timeout_ms,
        )
    else:
        server = VLAServer(model, host=args.host, port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
