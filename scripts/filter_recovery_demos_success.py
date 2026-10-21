#!/usr/bin/env python3
"""Filter strictly validated human recovery demos and report cell coverage."""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
import os
import shutil
from collections import Counter
from pathlib import Path


GATES = (
    "npz_readable",
    "required_arrays",
    "finite_arrays",
    "lengths_consistent",
    "action_dimension",
    "recorded_success",
    "state_replay",
    "state_eef_error",
    "saved_final_state_success",
    "saved_start_action_replay",
    "saved_start_final_success",
    "saved_start_success_streak",
    "saved_start_release_lift",
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(row)
    return rows


def _safe_component(value: object, label: str) -> str:
    text = str(value or "")
    if not text or text in {".", ".."} or Path(text).name != text or "/" in text or "\\" in text:
        raise ValueError(f"unsafe {label}: {text!r}")
    return text


def _selected(row: dict) -> bool:
    checks = row.get("checks")
    return isinstance(checks, dict) and all(checks.get(gate) is True for gate in GATES)


def build_selection(rows: list[dict], source_root: Path) -> list[dict]:
    selected = []
    seen = set()
    for row in rows:
        if not _selected(row):
            continue
        task = _safe_component(row.get("task"), "task")
        subtype = _safe_component(row.get("subtype"), "subtype")
        demo_id = _safe_component(row.get("demo_id") or row.get("item_id"), "demo_id")
        source_name = Path(str(row.get("source_path") or f"{demo_id}.npz")).name
        if _safe_component(source_name, "source basename") != source_name or Path(source_name).suffix.lower() != ".npz":
            raise ValueError(f"invalid source basename for {demo_id}: {source_name!r}")
        key = (task, subtype, demo_id)
        if key in seen:
            raise ValueError(f"duplicate selected demo: {'/'.join(key)}")
        seen.add(key)
        source = source_root / task / subtype / source_name
        selected.append(
            {
                "task": task,
                "subtype": subtype,
                "demo_id": demo_id,
                "basename": source_name,
                "source": source,
            }
        )
    return sorted(selected, key=lambda item: (item["task"], item["subtype"], item["demo_id"]))


def _filtered_manifests(selection: list[dict], manifest_root: Path) -> dict[str, dict]:
    by_task = {}
    for item in selection:
        by_task.setdefault(item["task"], {})[item["demo_id"]] = item
    result = {}
    for task, selected_by_id in sorted(by_task.items()):
        path = manifest_root / task / "manifest.json"
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read source manifest {path}: {exc}") from exc
        candidates_by_id = {}
        for demo in manifest.get("demos", []):
            demo_id = str(demo.get("demo_id"))
            if demo_id in selected_by_id:
                candidates_by_id.setdefault(demo_id, []).append(demo)
        demos = []
        ambiguous = []
        for demo_id in sorted(selected_by_id):
            candidates = candidates_by_id.get(demo_id, [])
            if len(candidates) == 1:
                demos.append(candidates[0])
                continue
            successful = [demo for demo in candidates if demo.get("success") is True]
            if len(successful) == 1:
                demos.append(successful[0])
            else:
                ambiguous.append(demo_id)
        if ambiguous:
            raise ValueError(f"{path}: selected demo IDs missing or ambiguously duplicated: {ambiguous}")
        filtered = []
        for demo in demos:
            copied = dict(demo)
            item = selected_by_id[copied["demo_id"]]
            copied["npz_path"] = f"{item['subtype']}/{item['basename']}"
            filtered.append(copied)
        manifest["demos"] = sorted(filtered, key=lambda demo: str(demo["demo_id"]))
        if isinstance(manifest.get("status"), dict):
            counts = Counter(item["subtype"] for item in selected_by_id.values())
            status = {}
            for subtype, count in sorted(counts.items()):
                entry = dict(manifest["status"].get(subtype, {}))
                for field in ("collected", "augmentable", "successful"):
                    if field in entry:
                        entry[field] = count
                status[subtype] = entry
            manifest["status"] = status
        manifest["selection"] = {"gates": list(GATES), "selected": len(filtered)}
        result[task] = manifest
    return result


def _json_write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _signature(selection: list[dict], mode: str) -> dict:
    return {
        "mode": mode,
        "gates": list(GATES),
        "files": [
            {key: item[key] for key in ("task", "subtype", "demo_id", "basename")}
            for item in selection
        ],
    }


def _prepare_output(output: Path, signature: dict) -> Path:
    manifest_path = output / "selection_manifest.json"
    if output.exists() and any(output.iterdir()):
        if not manifest_path.is_file():
            raise FileExistsError(f"refusing unrelated non-empty output directory: {output}")
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid existing selection manifest: {exc}") from exc
        if existing.get("signature") != signature:
            raise FileExistsError(f"existing output selection does not match this run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return manifest_path


def _stage_file(source: Path, destination: Path, mode: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        if mode == "symlink" and destination.is_symlink() and destination.resolve() == source.resolve():
            return
        if mode == "copy" and destination.is_file() and filecmp.cmp(source, destination, shallow=False):
            return
        raise FileExistsError(f"refusing to replace mismatched output file: {destination}")
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        temporary = destination.with_name(f".{destination.name}.partial")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)


def stage_selection(
    selection: list[dict],
    manifests: dict[str, dict],
    output: Path,
    mode: str,
    audit: Path,
    source_root: Path,
) -> None:
    signature = _signature(selection, mode)
    manifest_path = _prepare_output(output, signature)
    record = {
        "version": 1,
        "audit": str(audit),
        "source_root": str(source_root),
        "selected": len(selection),
        "by_cell": dict(sorted(Counter(f"{item['task']}/{item['subtype']}" for item in selection).items())),
        "signature": signature,
        "complete": False,
    }
    _json_write(manifest_path, record)
    for item in selection:
        _stage_file(
            item["source"],
            output / item["task"] / item["subtype"] / item["basename"],
            mode,
        )
    for task, manifest in manifests.items():
        _json_write(output / task / "manifest.json", manifest)
    record["complete"] = True
    _json_write(manifest_path, record)


def check_staged(selection: list[dict], output: Path, mode: str) -> None:
    for item in selection:
        destination = output / item["task"] / item["subtype"] / item["basename"]
        if mode == "symlink":
            if not destination.is_symlink() or destination.resolve() != item["source"].resolve():
                raise ValueError(f"bad staged symlink: {destination}")
        elif not destination.is_file() or not filecmp.cmp(item["source"], destination, shallow=False):
            raise ValueError(f"bad staged copy: {destination}")
    expected = Counter(item["demo_id"] for item in selection)
    actual = Counter()
    for path in output.glob("*/manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        actual.update(str(demo.get("demo_id")) for demo in manifest.get("demos", []))
    if actual != expected:
        raise ValueError("filtered manifests do not match selected demo IDs")


def _cell_universe(rows: list[dict]) -> list[str]:
    cells = set()
    for row in rows:
        if not row.get("task") or not row.get("subtype"):
            continue
        task = _safe_component(row["task"], "task")
        subtype = _safe_component(row["subtype"], "subtype")
        cells.add(f"{task}/{subtype}")
    return sorted(cells)


def _augmented_counts(root: Path | None) -> Counter:
    counts = Counter()
    if root is None or not root.exists():
        return counts
    npz_names = {path.name for path in root.rglob("*.npz")}
    seen = set()
    for path in sorted(root.rglob("augmentation_manifest*.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read augmentation manifest {path}: {exc}") from exc
        manifest_task = manifest.get("task")
        for manifest_subtype, group in (manifest.get("subtypes") or {}).items():
            for entry in group.get("entries", []):
                if entry.get("success") is not True:
                    continue
                task = _safe_component(entry.get("task_name") or manifest_task, "augmented task")
                subtype = _safe_component(entry.get("subtype_id") or manifest_subtype, "augmented subtype")
                augmented_id = _safe_component(
                    entry.get("augmented_id") or Path(str(entry.get("npz_path", ""))).stem,
                    "augmented_id",
                )
                basename = Path(str(entry.get("npz_path") or f"{augmented_id}.npz")).name
                if basename not in npz_names:
                    continue
                key = (task, subtype, augmented_id)
                if key not in seen:
                    seen.add(key)
                    counts[f"{task}/{subtype}"] += 1
    return counts


def write_coverage(
    rows: list[dict],
    selection: list[dict],
    augmented_root: Path | None,
    destination: Path,
    target: int,
) -> dict:
    human = Counter(f"{item['task']}/{item['subtype']}" for item in selection)
    augmented = _augmented_counts(augmented_root)
    cells = _cell_universe(rows)
    details = []
    for cell in cells:
        task, subtype = cell.split("/", 1)
        combined = human[cell] + augmented[cell]
        reachable = human[cell] > 0
        details.append(
            {
                "task": task,
                "subtype": subtype,
                "human": human[cell],
                "augmented": augmented[cell],
                "combined": combined,
                "target": target,
                "deficit": max(0, target - combined),
                "reachable": reachable,
                "status": "reached" if combined >= target else ("needs_augmentation" if reachable else "no_source"),
            }
        )
    summary = {
        "target": target,
        "selected": len(selection),
        "augmented": sum(augmented.values()),
        "total_cells": len(details),
        "reachable_cells": sum(item["reachable"] for item in details),
        "no_source_cells": sum(not item["reachable"] for item in details),
        "target_reached_cells": sum(item["status"] == "reached" for item in details),
        "total_deficit": sum(item["deficit"] for item in details),
        "cells": details,
    }
    _json_write(destination / "coverage.json", summary)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "coverage.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(details[0]) if details else (
            "task", "subtype", "human", "augmented", "combined", "target", "deficit", "reachable", "status"
        ))
        writer.writeheader()
        writer.writerows(details)
    os.replace(temporary, csv_path)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter strict human recovery demos into recovery_demos_success and report coverage."
    )
    parser.add_argument("--audit", type=Path, required=True, help="Qualification human.jsonl")
    parser.add_argument("--source-root", type=Path, required=True, help="Unfiltered human demo root")
    parser.add_argument("--manifest-root", type=Path, required=True, help="Root containing task/manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="Filtered output directory")
    parser.add_argument("--mode", choices=("copy", "symlink"), default="copy")
    parser.add_argument("--check", action="store_true", help="Verify all staged files and manifests")
    parser.add_argument("--augmented-root", type=Path, help="Validated MimicGen augmentation root")
    parser.add_argument("--target", type=int, default=10, help="Combined demonstrations required per cell")
    parser.add_argument("--coverage-only", action="store_true", help="Only refresh coverage.json/csv")
    args = parser.parse_args(argv)
    if args.target < 1:
        parser.error("--target must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_jsonl(args.audit)
    selection = build_selection(rows, args.source_root)
    if not args.coverage_only:
        manifests = _filtered_manifests(selection, args.manifest_root)
        stage_selection(selection, manifests, args.output, args.mode, args.audit, args.source_root)
        if args.check:
            check_staged(selection, args.output, args.mode)
            print("CHECK=PASS")
    coverage_destination = args.augmented_root if args.augmented_root is not None else args.output
    coverage = write_coverage(rows, selection, args.augmented_root, coverage_destination, args.target)
    print(f"SELECTED={coverage['selected']}")
    print(f"TOTAL_CELLS={coverage['total_cells']}")
    print(f"REACHABLE_CELLS={coverage['reachable_cells']}")
    print(f"NO_SOURCE_CELLS={coverage['no_source_cells']}")
    print(f"TARGET_REACHED_CELLS={coverage['target_reached_cells']}")
    print(f"TOTAL_DEFICIT={coverage['total_deficit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
