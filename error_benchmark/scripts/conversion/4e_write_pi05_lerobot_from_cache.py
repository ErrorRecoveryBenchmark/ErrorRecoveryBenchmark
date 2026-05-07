#!/usr/bin/env python
"""Write a Pi0.5 LeRobot dataset from cached rendered episode NPZ files."""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _postflight(repo_id, expected_episodes, expected_frames):
    result = {"status": "PASS", "repo_id": repo_id, "errors": []}
    try:
        meta = LeRobotDatasetMetadata(repo_id)
        dataset = LeRobotDataset(repo_id)
        result["num_tasks"] = len(meta.tasks)
        result["num_frames"] = int(len(dataset))
        result["expected_frames"] = int(expected_frames)
        result["expected_episodes"] = int(expected_episodes)

        if len(dataset) != expected_frames:
            result["errors"].append(
                f"frame count mismatch: {len(dataset)} != {expected_frames}"
            )
        if len(meta.tasks) < 1:
            result["errors"].append("no LeRobot task metadata")

        sample_indices = sorted(set([0, max(0, len(dataset) // 2), max(0, len(dataset) - 1)]))
        for idx in sample_indices:
            sample = dataset[idx]
            for key in ("image", "wrist_image", "state", "actions"):
                if key not in sample:
                    result["errors"].append(f"sample {idx} missing {key}")
            if "state" in sample and tuple(np.asarray(sample["state"]).shape)[-1:] != (8,):
                result["errors"].append(f"sample {idx} bad state shape")
            if "actions" in sample and tuple(np.asarray(sample["actions"]).shape)[-1:] != (7,):
                result["errors"].append(f"sample {idx} bad action shape")
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")

    if result["errors"]:
        result["status"] = "FAIL"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    repo_id = manifest["repo_id"]
    output_path = Path(
        os.environ.get("HF_LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot"))
    ) / repo_id
    episodes = manifest["episodes"]
    expected_frames = int(manifest["total_frames"])

    report = {
        "status": "RUNNING",
        "repo_id": repo_id,
        "output_path": str(output_path),
        "expected_episodes": len(episodes),
        "expected_frames": expected_frames,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    try:
        if output_path.exists():
            if args.overwrite:
                shutil.rmtree(output_path)
            else:
                raise RuntimeError(f"dataset exists at {output_path}; use --overwrite")

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type="panda",
            fps=20,
            features={
                "image": {
                    "dtype": "image",
                    "shape": (84, 84, 3),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image": {
                    "dtype": "image",
                    "shape": (84, 84, 3),
                    "names": ["height", "width", "channel"],
                },
                "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
                "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
            },
            image_writer_threads=int(manifest["image_writer_threads"]),
            image_writer_processes=int(manifest["image_writer_processes"]),
        )

        task_prompt = manifest["task_prompt"]
        for ep_idx, episode in enumerate(episodes, 1):
            tmp_data = np.load(episode["tmp_path"])
            n_frames = int(episode["n_frames"])
            for frame_idx in range(n_frames):
                dataset.add_frame(
                    {
                        "image": tmp_data["agent_imgs"][frame_idx],
                        "wrist_image": tmp_data["wrist_imgs"][frame_idx],
                        "state": tmp_data["state8"][frame_idx],
                        "actions": tmp_data["actions"][frame_idx],
                        "task": task_prompt,
                    }
                )
            dataset.save_episode()
            if ep_idx % 50 == 0 or ep_idx == len(episodes):
                print(f"[lerobot-writer] {ep_idx}/{len(episodes)} episodes written", flush=True)

        postflight = _postflight(repo_id, len(episodes), expected_frames)
        report.update(postflight)
        if report["status"] != "PASS":
            raise RuntimeError("LeRobot postflight failed")
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _write_json(args.report, report)


if __name__ == "__main__":
    main()
