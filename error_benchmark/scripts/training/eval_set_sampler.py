#!/usr/bin/env python
"""Deterministic stratified sampler over the validation scene pool.

Builds the frozen evaluation set: per (task, subtype) draw N scenes
(min(N, available)) using a per-cell deterministic seed, then write a
manifest of the selected scene ids. Re-eval-only: does NOT generate scenes.

Authoritative protocol (docs/项目全景总结.md): the eval set is a sample of
the ~130k M12 validation scenes; the reported metric is SR (point estimate).
The manifest is the frozen, reproducible eval set (replaces a fixed scene count).
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

# The 6 benchmark tasks (used to infer task from a scene path).
TASKS = ["pick_place", "coffee", "stack", "stack_three", "threading", "three_piece_assembly"]
DEFAULT_N = 50
DEFAULT_SEED = 42


def _infer_task(json_path: Path, root: Path) -> str:
    try:
        parts = json_path.relative_to(root).parts
    except ValueError:
        parts = json_path.parts
    for part in parts:
        if part in TASKS:
            return part
    return "unknown"


def _subtype_from_stem(stem: str):
    """Fast path: v5 scene files are named ``v5_<subtype>_<hexhash>`` where the
    subtype already includes the degree (e.g. v5_collision_eef_object_D0_ab12 ->
    collision_eef_object_D0). Returns None if the name is not in that form, so
    the caller falls back to reading the JSON labels."""
    if stem.startswith("v5_"):
        body = stem[3:].rsplit("_", 1)[0]  # drop the trailing hash token
        if body:
            return body
    return None


def discover_records(validation_dir: str, require_npz: bool = True) -> list:
    """Scan validation_dir for paired scene JSON/NPZ; extract (task, subtype, scene_id, path).

    Subtype comes from the v5 filename when possible (no file read); otherwise
    it falls back to the JSON labels (meta['labels']['subtype_id'], else
    error_spec / error_name_degree). NPZ pairing is checked against a single
    directory walk rather than a per-file stat, so this scales to ~130k scenes.
    """
    root = Path(validation_dir)
    npz_keys = None
    if require_npz:
        npz_keys = {p.with_suffix("").as_posix() for p in root.glob("**/*.npz")}

    records = []
    for json_path in sorted(root.glob("**/*.json")):
        name = json_path.name
        if name == "manifest.json" or name.endswith("_manifest.json"):
            continue
        if npz_keys is not None and json_path.with_suffix("").as_posix() not in npz_keys:
            continue
        subtype = _subtype_from_stem(json_path.stem)
        if subtype is None:
            try:
                with open(json_path) as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                continue
            labels = meta.get("labels", {})
            spec = meta.get("error_spec", {})
            error_name = labels.get("error_name", spec.get("error_name", "unknown"))
            degree = labels.get("degree", spec.get("degree", "unknown"))
            subtype = labels.get("subtype_id", f"{error_name}_{degree}")
        records.append({
            "task": _infer_task(json_path, root),
            "subtype": subtype,
            "scene_id": json_path.stem,
            "path": str(json_path),
        })
    return records


def stratified_sample(records: list, n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> dict:
    """Per (task, subtype): select min(n, available) scenes deterministically.

    Each cell uses an independent string-seeded RNG, so adding or removing one
    cell never perturbs another cell's selection. Selections are returned in
    stable (scene_id-sorted) order.
    """
    groups = defaultdict(list)
    for r in records:
        groups[(r["task"], r["subtype"])].append(r)

    cells = {}
    selected = []
    for (task, subtype) in sorted(groups):
        items = sorted(groups[(task, subtype)], key=lambda r: r["scene_id"])
        if len(items) <= n:
            picked = list(items)
        else:
            rng = random.Random(f"{seed}:{task}:{subtype}")
            picked = sorted(rng.sample(items, n), key=lambda r: r["scene_id"])
        cells[f"{task}/{subtype}"] = {"available": len(items), "selected": len(picked)}
        selected.extend(picked)

    return {
        "seed": seed,
        "n_per_cell": n,
        "total_selected": len(selected),
        "num_cells": len(cells),
        "cells": cells,
        "scenes": [
            {"task": r["task"], "subtype": r["subtype"],
             "scene_id": r["scene_id"], "path": r.get("path", "")}
            for r in selected
        ],
    }


def write_manifest(manifest: dict, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Stratified deterministic eval-set sampler over the validation scene pool")
    parser.add_argument("--validation_dir", required=True,
                        help="Root dir of the ~130k validation scenes (organized by task)")
    parser.add_argument("--out", required=True, help="Output manifest path (eval_set_manifest.json)")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Scenes per (task, subtype) cell")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Sampling seed (freezes the eval set)")
    args = parser.parse_args()

    records = discover_records(args.validation_dir)
    manifest = stratified_sample(records, n=args.n, seed=args.seed)
    write_manifest(manifest, args.out)
    print(f"Discovered {len(records)} scenes; selected {manifest['total_selected']} "
          f"across {manifest['num_cells']} (task,subtype) cells -> {args.out}")


if __name__ == "__main__":
    main()
