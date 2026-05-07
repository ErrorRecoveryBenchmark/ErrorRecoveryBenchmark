#!/usr/bin/env python3
"""
Aggregate per-task BC-RNN baseline eval JSONs into a markdown report and summary.json.

Reads eval outputs from outputs/eval_bc_rnn_baseline/{task}.json (produced by
eval_bc_rnn_error_scenes.py) and writes:
  outputs/eval_bc_rnn_baseline/summary.json
  outputs/eval_bc_rnn_baseline/summary.md
"""

import argparse
import json
import os
from pathlib import Path

PROJECT_DIR = Path(
    os.environ.get("ERROR_RECOVERY_BENCHMARK_ROOT")
    or Path(__file__).resolve().parents[3]
)
DEFAULT_EVAL_DIR = PROJECT_DIR / "error_benchmark" / "outputs" / "eval_bc_rnn_baseline"

TASKS = ["coffee", "pick_place", "stack", "stack_three", "threading", "three_piece_assembly"]


def fmt_pct(x):
    return f"{x*100:.1f}%" if x is not None else "—"


def parse_tasks(tasks_csv: str | None) -> list[str]:
    """Parse an optional comma-separated task list."""
    if not tasks_csv:
        return list(TASKS)

    tasks = [task.strip() for task in tasks_csv.split(",") if task.strip()]
    if not tasks:
        raise ValueError("No tasks were selected")

    unknown = sorted(set(tasks) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {', '.join(unknown)}")

    return tasks


def build_summary(eval_dir: Path, tasks: list[str]) -> tuple[dict, list[str]]:
    """Build summary data and markdown for the requested tasks."""
    summary = {
        "tasks": {},
        "eval_dir": str(eval_dir),
        "selected_tasks": tasks,
    }
    rows = []
    subtype_lines = []

    for task in tasks:
        path = eval_dir / f"{task}.json"
        if not path.exists():
            rows.append((task, None, None, None, None, "missing"))
            summary["tasks"][task] = {"status": "missing"}
            continue

        with open(path) as f:
            d = json.load(f)

        clean = d.get("clean_rollouts", {})
        err = d.get("error_scenes", {})
        clean_sr = clean.get("sr")
        err_sr = err.get("overall_sr")
        n_clean = clean.get("total")
        n_err = err.get("total")
        ckpt_epoch = d.get("checkpoint_epoch")
        rows.append((task, clean_sr, n_clean, err_sr, n_err, f"epoch_{ckpt_epoch}"))
        summary["tasks"][task] = {
            "checkpoint": d.get("checkpoint"),
            "checkpoint_epoch": ckpt_epoch,
            "clean_sr": clean_sr,
            "clean_total": n_clean,
            "error_sr": err_sr,
            "error_total": n_err,
            "by_subtype": err.get("by_subtype", {}),
        }
        for sid, v in (err.get("by_subtype") or {}).items():
            subtype_lines.append((task, sid, v.get("sr"), v.get("successes"), v.get("total")))

    md = ["# BC-RNN Baseline (mixed-2000) Eval Summary", ""]
    md.append("| Task | Ckpt | Clean SR | N | Validation SR | N |")
    md.append("|---|---|---|---|---|---|")
    for task, csr, nc, esr, ne, ck in rows:
        md.append(
            f"| {task} | {ck} | {fmt_pct(csr)} | {nc or '—'} | "
            f"{fmt_pct(esr)} | {ne or '—'} |"
        )
    md.append("")
    md.append("## Per-subtype breakdown")
    md.append("")
    md.append("| Task | Subtype | SR | succ/total |")
    md.append("|---|---|---|---|")
    for task, sid, sr, s, t in sorted(subtype_lines):
        md.append(f"| {task} | {sid} | {fmt_pct(sr)} | {s}/{t} |")

    return summary, md


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", default=str(DEFAULT_EVAL_DIR))
    p.add_argument(
        "--tasks",
        default=None,
        help="Optional comma-separated subset of tasks to summarize",
    )
    args = p.parse_args()
    eval_dir = Path(args.eval_dir)
    tasks = parse_tasks(args.tasks)
    summary, md = build_summary(eval_dir, tasks)

    eval_dir.mkdir(parents=True, exist_ok=True)
    sum_json = eval_dir / "summary.json"
    sum_md = eval_dir / "summary.md"
    sum_json.write_text(json.dumps(summary, indent=2, default=str))
    sum_md.write_text("\n".join(md) + "\n")
    print(f"Wrote {sum_json}")
    print(f"Wrote {sum_md}")
    print()
    print("\n".join(md[:3 + len(tasks)]))


if __name__ == "__main__":
    main()
