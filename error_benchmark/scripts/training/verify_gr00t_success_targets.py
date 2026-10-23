#!/usr/bin/env python3
"""Verify final GR00T clean-success targets and error-evaluation completeness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_metric(path: Path, key: str) -> tuple[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))[key]
    return int(payload["successes"]), int(payload["total"])


def _single_step_dir(root: Path, task: str, setting: str) -> Path:
    candidates = sorted((root / task / setting).glob("step_*"))
    if len(candidates) != 1:
        raise ValueError(
            f"{task}/{setting} must have exactly one final step directory, "
            f"found {[path.name for path in candidates]}"
        )
    return candidates[0]


def verify(args: argparse.Namespace) -> dict:
    summary = {}
    for setting in args.settings:
        rows = []
        error_successes = 0
        error_total = 0
        for task in args.tasks:
            step_dir = _single_step_dir(args.results_root, task, setting)
            clean_path = step_dir / "clean" / f"{task}.json"
            if not clean_path.is_file():
                raise FileNotFoundError(clean_path)
            successes, total = _load_metric(clean_path, "clean_rollouts")
            if total != args.clean_total:
                raise ValueError(
                    f"{task}/{setting} clean total {total} != {args.clean_total}"
                )
            rate = successes / total
            if task in args.simple_tasks and rate < args.simple_min:
                raise ValueError(
                    f"{task}/{setting} clean success rate {rate:.6f} "
                    f"< {args.simple_min:.6f}"
                )
            rows.append(
                {
                    "task": task,
                    "step": step_dir.name,
                    "clean_successes": successes,
                    "clean_total": total,
                    "clean_success_rate": rate,
                }
            )

            if args.require_error_eval:
                error_path = step_dir / "error" / f"{task}.json"
                if not error_path.is_file():
                    raise FileNotFoundError(error_path)
                task_successes, task_total = _load_metric(
                    error_path,
                    "error_scenes",
                )
                error_successes += task_successes
                error_total += task_total

        clean_successes = sum(row["clean_successes"] for row in rows)
        clean_total = sum(row["clean_total"] for row in rows)
        clean_rate = clean_successes / clean_total
        if clean_rate < args.overall_min:
            raise ValueError(
                f"{setting} clean success rate {clean_rate:.6f} "
                f"< {args.overall_min:.6f}"
            )
        if (
            args.require_error_eval
            and error_total != args.expected_error_scenes_per_setting
        ):
            raise ValueError(
                f"{setting} error total {error_total} "
                f"!= {args.expected_error_scenes_per_setting}"
            )
        summary[setting] = {
            "clean_successes": clean_successes,
            "clean_total": clean_total,
            "clean_success_rate": clean_rate,
            "error_successes": error_successes,
            "error_total": error_total,
            "tasks": rows,
        }

    print("GR00T_SUCCESS_TARGETS_PASS")
    if args.require_error_eval:
        print("GR00T_ERROR_EVAL_COMPLETE")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--settings", nargs="+", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--simple-tasks", nargs="+", default=[])
    parser.add_argument("--clean-total", type=int, default=50)
    parser.add_argument("--overall-min", type=float, default=0.0)
    parser.add_argument("--simple-min", type=float, default=0.0)
    parser.add_argument("--require-error-eval", action="store_true")
    parser.add_argument("--expected-error-scenes-per-setting", type=int, default=1360)
    return parser.parse_args()


def main() -> None:
    verify(parse_args())


if __name__ == "__main__":
    main()
