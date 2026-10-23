#!/usr/bin/env python3
"""Verify GR00T campaign completeness and aggregate checkpoint-sweep results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


TASKS = (
    "pick_place",
    "coffee",
    "stack",
    "stack_three",
    "threading",
    "three_piece_assembly",
)
SETTINGS = ("nominal", "nominal_recovery")
ERROR_TOTALS = {
    "pick_place": 240,
    "coffee": 200,
    "stack": 240,
    "stack_three": 240,
    "threading": 200,
    "three_piece_assembly": 240,
}


def discover_checkpoint_steps(checkpoint_dir: Path, min_step: int = 7000) -> list[int]:
    steps = []
    if not checkpoint_dir.is_dir():
        return steps
    for path in checkpoint_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("checkpoint-"):
            continue
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit() and int(suffix) >= min_step:
            steps.append(int(suffix))
    return sorted(steps)


def aggregate_best_recovery(
    rows: Iterable[dict],
    tasks: Iterable[str] = TASKS,
) -> dict:
    selected = {}
    for task in tasks:
        candidates = [row for row in rows if row["task"] == task]
        if not candidates:
            raise ValueError(f"no recovery evaluation rows for {task}")
        selected[task] = max(
            candidates,
            key=lambda row: (
                row["successes"] / row["total"],
                row["successes"],
                -row["step"],
            ),
        )

    successes = sum(row["successes"] for row in selected.values())
    total = sum(row["total"] for row in selected.values())
    return {
        "selected_steps": {task: row["step"] for task, row in selected.items()},
        "successes": successes,
        "total": total,
        "rsr": successes / total,
    }


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _evaluation_row(root: Path, task: str, setting: str, step: int) -> dict:
    base = root / "evaluations" / task / setting / f"step_{step}"
    clean_path = base / "clean" / f"{task}.json"
    error_path = base / "error" / f"{task}.json"
    if not clean_path.is_file() or not error_path.is_file():
        raise FileNotFoundError(
            f"missing paired evaluation for {task}/{setting}/step_{step}"
        )

    clean = _load_json(clean_path)["clean_rollouts"]
    error = _load_json(error_path)["error_scenes"]
    return {
        "task": task,
        "setting": setting,
        "step": step,
        "clean_successes": int(clean["successes"]),
        "clean_total": int(clean["total"]),
        "csr": float(clean["successes"]) / int(clean["total"]),
        "successes": int(error["successes"]),
        "total": int(error["total"]),
        "rsr": float(error["successes"]) / int(error["total"]),
        "clean_result": str(clean_path),
        "error_result": str(error_path),
    }


def _common_step_aggregates(rows: list[dict]) -> list[dict]:
    steps_by_task = {
        task: {row["step"] for row in rows if row["task"] == task}
        for task in TASKS
    }
    common_steps = sorted(set.intersection(*steps_by_task.values()))
    aggregates = []
    for step in common_steps:
        selected = [row for row in rows if row["step"] == step]
        successes = sum(row["successes"] for row in selected)
        total = sum(row["total"] for row in selected)
        aggregates.append(
            {
                "step": step,
                "successes": successes,
                "total": total,
                "rsr": successes / total,
            }
        )
    return aggregates


def verify(args: argparse.Namespace) -> dict:
    root = args.root
    rows = []
    model_count = 0
    for task in TASKS:
        for setting in SETTINGS:
            checkpoint_dir = root / "checkpoints" / task / setting
            steps = discover_checkpoint_steps(checkpoint_dir, args.min_eval_step)
            if checkpoint_dir.is_dir():
                model_count += 1
            for step in steps:
                row = _evaluation_row(root, task, setting, step)
                if row["clean_total"] != args.clean_episodes_per_task:
                    raise ValueError(
                        f"{task}/{setting}/step_{step} clean total "
                        f"{row['clean_total']} != {args.clean_episodes_per_task}"
                    )
                if row["total"] != ERROR_TOTALS[task]:
                    raise ValueError(
                        f"{task}/{setting}/step_{step} error total "
                        f"{row['total']} != {ERROR_TOTALS[task]}"
                    )
                rows.append(row)

    if args.require_models is not None and model_count != args.require_models:
        raise ValueError(f"model count {model_count} != {args.require_models}")
    if sum(ERROR_TOTALS.values()) != args.error_total:
        raise ValueError(
            f"registered error total {sum(ERROR_TOTALS.values())} != {args.error_total}"
        )

    recovery_rows = [row for row in rows if row["setting"] == "nominal_recovery"]
    best = aggregate_best_recovery(recovery_rows)
    common = _common_step_aggregates(recovery_rows)
    summary = {
        "model_count": model_count,
        "error_total": sum(ERROR_TOTALS.values()),
        "rows": rows,
        "paper_style_best_per_task": best,
        "common_step_aggregates": common,
    }
    root.mkdir(parents=True, exist_ok=True)
    with (root / "campaign_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    if args.require_final_recovery_rsr_above is not None:
        threshold = args.require_final_recovery_rsr_above
        if best["rsr"] <= threshold:
            raise ValueError(
                f"final recovery RSR {best['rsr']:.6f} is not above {threshold:.6f}"
            )
        print(
            f"final recovery RSR PASS: {best['rsr']:.6f} > {threshold:.6f}"
        )
    print("campaign verification passed")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--require-models", type=int)
    parser.add_argument("--min-eval-step", type=int, default=7000)
    parser.add_argument("--clean-episodes-per-task", type=int, default=50)
    parser.add_argument("--error-total", type=int, default=1360)
    parser.add_argument("--require-final-recovery-rsr-above", type=float)
    return parser.parse_args()


def main() -> None:
    verify(parse_args())


if __name__ == "__main__":
    main()
