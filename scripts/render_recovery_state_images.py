#!/usr/bin/env python3
"""Render explicitly aligned derivative images without replacing source recordings."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Readable NPZ with complete states")
    parser.add_argument("destination", type=Path, help="New companion image NPZ")
    parser.add_argument("--seed-dataset", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()
    if args.destination.exists() or args.destination.with_suffix(".json").exists():
        parser.error("Destination exists; refusing to overwrite")
    if os.name == "nt":
        os.environ.setdefault("MUJOCO_GL", "glfw")
    import mujoco
    import robosuite
    import yaml
    from error_benchmark.framework.env_wrapper import EnvWrapper
    from error_benchmark.scripts.utils.script_utils import create_env

    if mujoco.__version__ != "2.3.2" or robosuite.__version__ != "1.4.1":
        parser.error("Requires pinned mujoco 2.3.2 and robosuite 1.4.1")
    config = yaml.safe_load(args.task_config.read_text(encoding="utf-8"))
    with np.load(args.source, allow_pickle=False) as source:
        states = source["states"]
        if states.ndim != 2 or not np.isfinite(states).all():
            parser.error("Invalid states")
        env = create_env(config, str(args.seed_dataset), enable_camera=True,
                         camera_resolution=args.resolution)
        try:
            env.reset()
            wrapper = EnvWrapper(env, config)
            frames, errors = [], {}
            for index, state in enumerate(states):
                wrapper.set_sim_state_flat(state)
                if not np.array_equal(wrapper.get_sim_state_flat(), state):
                    raise ValueError(f"State round trip changed values at {index}")
                comparisons = {"eef": (wrapper.get_eef_pos(), source["eef_positions"][index])}
                for obj in config.get("objects", []):
                    key = "obj_" + obj["name"]
                    if key in source:
                        comparisons[obj["name"]] = (wrapper.get_object_pose(obj["name"])[0], source[key][index])
                for name, (actual, expected) in comparisons.items():
                    distance = float(np.linalg.norm(actual - expected))
                    errors[name] = max(errors.get(name, 0.0), distance)
                frames.append(wrapper.get_camera_obs(args.camera, (args.resolution, args.resolution)))
            camera_id = env.sim.model.camera_name2id(args.camera)
            report = {
                "demo_id": str(source["demo_id"]),
                "source_npz_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "method": "Restore each saved state and forward kinematics; no action integration or image generation model",
                "origin": "state_rerender; derivative observations, not recovered original camera pixels",
                "alignment": "camera_images[i] corresponds exactly to source states[camera_state_indices[i]]",
                "frames": len(frames), "resolution": args.resolution,
                "camera": args.camera, "camera_pos": env.sim.model.cam_pos[camera_id].tolist(),
                "camera_quat_wxyz": env.sim.model.cam_quat[camera_id].tolist(),
                "geomgroups": env.sim._render_context_offscreen.vopt.geomgroup.tolist(),
                "mujoco": mujoco.__version__, "robosuite": robosuite.__version__,
                "max_position_difference_m": errors,
                "source_state_roundtrip": "All states bitwise equal after restore",
                "limitations": "Original camera settings and trimmed image timing are not fully known. Do not splice these frames into the original stream or describe them as original recordings. Position differences reflect stored observations versus forwarded saved states; this does not certify action replay or reproduce policy evaluation.",
            }
            args.destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.destination, camera_images=np.asarray(frames),
                                camera_state_indices=np.arange(len(states), dtype=np.int64),
                                state_times=states[:, 0], demo_id=source["demo_id"])
            report["image_npz_sha256"] = hashlib.sha256(args.destination.read_bytes()).hexdigest()
            args.destination.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2))
        finally:
            env.close()


if __name__ == "__main__":
    main()
