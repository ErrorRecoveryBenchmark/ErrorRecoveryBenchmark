#!/usr/bin/env python3
"""Inventory the three frozen archives and generate Croissant CSV indexes."""
import argparse
import csv
import hashlib
import io
import json
import tarfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np

FIELDS = ["archive_member", "task_name", "scene_id", "error_name", "degree", "demo_id", "num_steps", "success", "read_error"]


def inspect_archive(path, output, kind):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    rows, fingerprints, scene_ids = [], set(), set()
    tasks, subtypes, invalid = Counter(), set(), []
    with tarfile.open(path) as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".npz"):
                continue
            parts = PurePosixPath(member.name).parts
            task = parts[1]
            row = dict.fromkeys(FIELDS, "")
            row.update(archive_member=member.name, task_name=task)
            if kind == "scene":
                sidecar = json.load(tar.extractfile(member.name[:-4] + ".json"))
                row["scene_id"] = sidecar["scene_id"]
                labels = sidecar.get("labels", {})
                row["error_name"] = labels.get("error_name", "")
                row["degree"] = labels.get("degree", "")
                with np.load(io.BytesIO(tar.extractfile(member).read()), allow_pickle=False) as data:
                    state = np.asarray(data["post_sim_state"], dtype="<f8")
                    fingerprints.add(hashlib.sha256(state.tobytes()).hexdigest())
                scene_ids.add(row["scene_id"])
                subtypes.add((task, row["error_name"], row["degree"]))
            else:
                subtype = parts[2]
                row["error_name"], row["degree"] = subtype.rsplit("_", 1)
                row["demo_id"] = PurePosixPath(member.name).stem
                try:
                    with np.load(io.BytesIO(tar.extractfile(member).read()), allow_pickle=False) as data:
                        actions, states = data["actions"], data["states"]
                        if actions.ndim != 2 or states.ndim != 2 or not len(actions):
                            raise ValueError("Empty or non-matrix actions/states")
                        if len(states) not in (len(actions), len(actions) + 1):
                            raise ValueError("Action/state length mismatch")
                        if not np.isfinite(actions).all() or not np.isfinite(states).all():
                            raise ValueError("Nonfinite actions/states")
                        row["num_steps"] = len(actions)
                        for key in ["scene_id", "success"]:
                            if key in data:
                                row[key] = str(data[key].item())
                except (ValueError, KeyError, zipfile.BadZipFile, EOFError) as exc:
                    row["read_error"] = str(exc)
                    invalid.append({"member": member.name, "error": str(exc)})
            tasks[task] += 1
            rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["archive_member"]))
    return {"filename": path.name, "sha256": digest.hexdigest(), "size": path.stat().st_size,
            "records": len(rows), "per_task": dict(tasks), "task_subtype_cells": len(subtypes),
            "invalid_records": invalid}, fingerprints, scene_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--training-scenes", type=Path, required=True)
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("metadata"))
    args = parser.parse_args()
    evaluation, ef, ei = inspect_archive(args.eval, args.output / "eval_scenes.csv", "scene")
    training, tf, ti = inspect_archive(args.training_scenes, args.output / "training_scenes.csv", "scene")
    demos, _, _ = inspect_archive(args.demos, args.output / "recovery_demos.csv", "demo")
    report = {"archives": [evaluation, training, demos],
              "train_eval_shared_scene_ids": len(ei & ti), "train_eval_shared_state_fingerprints": len(ef & tf)}
    (args.output / "release_inventory.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "SHA256SUMS").write_text("".join(f"{a['sha256']}  {a['filename']}\n" for a in report["archives"]), encoding="ascii")
    print(json.dumps(report, indent=2))
    if ei & ti or ef & tf or demos["invalid_records"]:
        raise SystemExit("Release blocked: overlap or unreadable demonstrations; see release_inventory.json")


if __name__ == "__main__":
    main()
