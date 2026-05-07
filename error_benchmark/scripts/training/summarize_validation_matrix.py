#!/usr/bin/env python3
"""Summarize validation_matrix outputs into one JSON and Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASK_EVAL_NAMES = {
    "pick_place": "PickPlace_D0",
    "coffee": "Coffee_D0",
    "stack": "Stack_D0",
    "stack_three": "StackThree_D0",
    "threading": "Threading_D0",
    "three_piece_assembly": "ThreePieceAssembly_D0",
}

DEFAULT_TASKS = list(TASK_EVAL_NAMES)
DEFAULT_VERSIONS = ["v1", "v2"]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except Exception as exc:
        return {"_load_error": str(exc)}


def newest(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def infer_task_from_artifact(path: Path, prefix: str) -> str | None:
    """Infer the task encoded in a meta/results artifact filename."""
    prefix_text = f"{prefix}_"
    stem = path.stem
    if not stem.startswith(prefix_text):
        return None

    body = stem[len(prefix_text) :]
    matches = [task for task in TASK_EVAL_NAMES if body.startswith(f"{task}_")]
    if not matches:
        return None
    return max(matches, key=len)


def task_artifacts(clean_dir: Path, prefix: str, task: str) -> list[Path]:
    paths = clean_dir.glob(f"{prefix}_*_step*.json")
    return [path for path in paths if infer_task_from_artifact(path, prefix) == task]


def sr_triplet(block: dict[str, Any] | None, sr_key: str = "sr") -> dict[str, Any] | None:
    if not block:
        return None
    error = block.get("error")
    if error:
        return {
            "sr": None,
            "successes": block.get("successes"),
            "total": block.get("total") or block.get("episodes"),
            "error": error,
        }
    sr = block.get(sr_key)
    if sr is None and sr_key != "overall_sr":
        sr = block.get("overall_sr")
    if sr is None:
        return None
    return {
        "sr": sr,
        "successes": block.get("successes"),
        "total": block.get("total") or block.get("episodes"),
        "error": block.get("error"),
    }


def format_sr(value: dict[str, Any] | None) -> str:
    if not value:
        return "-"
    sr = value.get("sr")
    successes = value.get("successes")
    total = value.get("total")
    if sr is None:
        return value.get("error") or "-"
    if successes is None or total is None:
        return f"{sr:.1%}"
    return f"{sr:.1%} ({successes}/{total})"


def compact_checkpoint(path: str | None) -> str:
    if not path:
        return "-"
    parts = Path(path).parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        return "/".join(parts[idx:])
    if "bc_rnn_checkpoints" in parts:
        idx = parts.index("bc_rnn_checkpoints")
        return "/".join(parts[idx:])
    return path


def count_scene_errors(error_block: dict[str, Any] | None) -> int:
    if not error_block:
        return 0
    per_scene = error_block.get("per_scene") or []
    return sum(1 for item in per_scene if item.get("error"))


def summarize_pi05(output_root: Path, task: str, version: str, scenes_per_group: str, group_by: str) -> dict[str, Any]:
    clean_dir = output_root / f"pi05_{version}" / "mimicgen_sr"
    error_dir = output_root / f"pi05_{version}" / f"error_scenes_{scenes_per_group}per_{group_by}"

    meta_path = newest(task_artifacts(clean_dir, "meta", task))
    meta = load_json(meta_path) if meta_path else None
    clean_path = None
    clean_data = None
    clean_result = None
    if meta and meta.get("result_file"):
        clean_path = Path(meta["result_file"])
        clean_data = load_json(clean_path)
    else:
        clean_path = newest(task_artifacts(clean_dir, "results", task))
        clean_data = load_json(clean_path) if clean_path else None

    if clean_data:
        eval_name = TASK_EVAL_NAMES[task]
        block = clean_data.get("_overall") or clean_data.get(eval_name)
        if block:
            clean_result = {
                "sr": block.get("success_rate"),
                "successes": block.get("successes"),
                "total": block.get("episodes"),
            }

    error_path = error_dir / f"{task}.json"
    error_data = load_json(error_path)
    error_block = error_data.get("error_scenes") if error_data else None
    error_result = sr_triplet(error_block, sr_key="overall_sr")

    checkpoint = None
    if error_data:
        checkpoint = error_data.get("checkpoint")
    if checkpoint is None and meta:
        checkpoint = meta.get("checkpoint")

    status_bits = []
    if clean_result:
        status_bits.append("clean")
    if error_result:
        status_bits.append("error")
    if not status_bits:
        status = "missing"
    elif len(status_bits) == 2:
        status = "done"
    else:
        status = "partial:" + ",".join(status_bits)

    scene_errors = count_scene_errors(error_block)
    if scene_errors:
        status += f"; scene_errors={scene_errors}"

    return {
        "task": task,
        "model_family": "pi05",
        "version": version,
        "checkpoint": checkpoint,
        "clean_rollouts": clean_result,
        "error_scenes": error_result,
        "status": status,
        "result_paths": {
            "clean": str(clean_path) if clean_path else None,
            "clean_meta": str(meta_path) if meta_path else None,
            "error": str(error_path) if error_path.exists() else None,
        },
    }


def summarize_bcrnn(output_root: Path, task: str, version: str, scenes_per_group: str, group_by: str) -> dict[str, Any]:
    result_dir = output_root / f"bcrnn_{version}" / f"combined_{scenes_per_group}per_{group_by}"
    result_path = result_dir / f"{task}.json"
    data = load_json(result_path)
    clean_result = None
    error_result = None
    checkpoint = None
    status = "missing"

    if data:
        checkpoint = data.get("checkpoint")
        clean_result = sr_triplet(data.get("clean_rollouts"), sr_key="sr")
        error_block = data.get("error_scenes")
        error_result = sr_triplet(error_block, sr_key="overall_sr")
        status_bits = []
        if clean_result and clean_result.get("sr") is not None:
            status_bits.append("clean")
        elif clean_result and clean_result.get("error"):
            status_bits.append(f"clean_error={clean_result['error']}")
        if error_result and error_result.get("sr") is not None:
            status_bits.append("error")
        status = "done" if status_bits == ["clean", "error"] else "partial:" + ",".join(status_bits)
        scene_errors = count_scene_errors(error_block)
        if scene_errors:
            status += f"; scene_errors={scene_errors}"

    return {
        "task": task,
        "model_family": "bcrnn",
        "version": version,
        "checkpoint": checkpoint,
        "clean_rollouts": clean_result,
        "error_scenes": error_result,
        "status": status,
        "result_paths": {
            "combined": str(result_path) if result_path.exists() else None,
        },
    }


def build_summary(output_root: Path, tasks: list[str], versions: list[str], scenes_per_group: str, group_by: str) -> dict[str, Any]:
    rows = []
    for task in tasks:
        for version in versions:
            rows.append(summarize_pi05(output_root, task, version, scenes_per_group, group_by))
            rows.append(summarize_bcrnn(output_root, task, version, scenes_per_group, group_by))

    return {
        "output_root": str(output_root),
        "tasks": tasks,
        "versions": versions,
        "scenes_per_group": scenes_per_group,
        "group_by": group_by,
        "rows": rows,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Validation Matrix Summary",
        "",
        f"- scenes_per_group: `{summary['scenes_per_group']}`",
        f"- group_by: `{summary['group_by']}`",
        "",
        "| Task | Model | Version | Checkpoint | MimicGen SR | Error Scene SR | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {task} | {model_family} | {version} | `{checkpoint}` | {clean} | {error} | {status} |".format(
                task=row["task"],
                model_family=row["model_family"],
                version=row["version"],
                checkpoint=compact_checkpoint(row.get("checkpoint")),
                clean=format_sr(row.get("clean_rollouts")),
                error=format_sr(row.get("error_scenes")),
                status=row["status"],
            )
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--versions", nargs="+", default=DEFAULT_VERSIONS)
    parser.add_argument("--scenes-per-group", default="10")
    parser.add_argument("--group-by", default="subtype_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown_tasks = sorted(set(args.tasks) - set(DEFAULT_TASKS))
    if unknown_tasks:
        raise SystemExit(f"Unknown tasks: {', '.join(unknown_tasks)}")
    unknown_versions = sorted(set(args.versions) - {"v1", "v2"})
    if unknown_versions:
        raise SystemExit(f"Unknown versions: {', '.join(unknown_versions)}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = build_summary(
        output_root=output_root,
        tasks=args.tasks,
        versions=args.versions,
        scenes_per_group=str(args.scenes_per_group),
        group_by=args.group_by,
    )

    json_path = output_root / "validation_matrix_summary.json"
    md_path = output_root / "validation_matrix_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    md_path.write_text(render_markdown(summary))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
