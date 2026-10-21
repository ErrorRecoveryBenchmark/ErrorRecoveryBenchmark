#!/usr/bin/env python
"""
Augment MimicGen success demos via standard MimicGen data generation.

Takes success_demos.hdf5 (with datagen_info) as source dataset and generates
more success trajectories by randomizing object poses. Standard MimicGen
augmentation — no error state injection.

Requires GPU (MuJoCo EGL rendering).

Usage:
    # Augment pick_place to 1000 demos, 96 workers
    MUJOCO_GL=egl python augment_success_demos.py --task pick_place --target 1000 --num-workers 96

    # All tasks, 10x augmentation
    MUJOCO_GL=egl python augment_success_demos.py --all-tasks --multiplier 10 --num-workers 96

    # Smoke test
    MUJOCO_GL=egl python augment_success_demos.py --task stack --target 20 --num-workers 4
"""

import argparse
import json
import os
import sys
import time
import multiprocessing
import shutil
from glob import glob as glob_files
from pathlib import Path

import numpy as np

from error_benchmark.framework.mimicgen_source_validation import (
    ensure_env_interface_attrs,
    find_first_binary_transition,
)

# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MIMICGEN_ROOT = PROJECT_ROOT / "shared" / "mimicgen_workspace"

sys.path.insert(0, str(MIMICGEN_ROOT / "robosuite"))
sys.path.insert(0, str(MIMICGEN_ROOT / "mimicgen"))

# ── Lazy imports ───────────────────────────────────────────────────────────
_IMPORTS_DONE = False


def _lazy_imports():
    global _IMPORTS_DONE
    if _IMPORTS_DONE:
        return
    _IMPORTS_DONE = True

    global robomimic, get_env_metadata_from_dataset
    global mimicgen, MG_FileUtils, RobomimicUtils
    global config_factory, MG_TaskSpec, DataGenerator, make_interface

    import robomimic
    from robomimic.utils.file_utils import get_env_metadata_from_dataset as _get

    import mimicgen
    import mimicgen.utils.file_utils as _MG_FileUtils
    import mimicgen.utils.robomimic_utils as _RobomimicUtils
    from mimicgen.configs import config_factory as _cf, MG_TaskSpec as _ts
    from mimicgen.datagen.data_generator import DataGenerator as _dg
    from mimicgen.env_interfaces.base import make_interface as _mi

    get_env_metadata_from_dataset = _get
    MG_FileUtils = _MG_FileUtils
    RobomimicUtils = _RobomimicUtils
    config_factory = _cf
    MG_TaskSpec = _ts
    DataGenerator = _dg
    make_interface = _mi


# ── Task config ────────────────────────────────────────────────────────────
TASK_CONFIGS = {
    "pick_place": "pick_place.json",
    "coffee": "coffee.json",
    "stack": "stack.json",
    "stack_three": "stack_three.json",
    "threading": "threading.json",
    "three_piece_assembly": "three_piece_assembly.json",
}

SUCCESS_DEMOS_DIR = str(PROJECT_ROOT / "error_benchmark" / "outputs" / "mimicgen_success")
DEFAULT_SOURCE_FILE = "success_demos.hdf5"
COMBINED_SOURCE_FILE = "success_demos_combined.hdf5"


def resolve_source_dataset(success_dir: str, task_name: str, source_file: str | None = None) -> str:
    task_dir = os.path.join(success_dir, task_name)
    if source_file:
        return os.path.join(task_dir, source_file)

    combined = os.path.join(task_dir, COMBINED_SOURCE_FILE)
    if os.path.exists(combined):
        return combined
    return os.path.join(task_dir, DEFAULT_SOURCE_FILE)


def load_mg_config(task_name: str):
    template_dir = MIMICGEN_ROOT / "mimicgen" / "mimicgen" / "exps" / "templates" / "robosuite"
    template_file = template_dir / TASK_CONFIGS[task_name]
    with open(template_file) as f:
        ext_cfg = json.load(f)
    if "meta" in ext_cfg:
        del ext_cfg["meta"]
    mg_config = config_factory(ext_cfg["name"], config_type=ext_cfg["type"])
    with mg_config.values_unlocked():
        mg_config.update(ext_cfg)
    return mg_config


def setup_env(task_name: str, source_dataset: str, mg_config):
    """Create MimicGen env + DataGenerator from source dataset."""
    env_meta = get_env_metadata_from_dataset(dataset_path=source_dataset)
    env = RobomimicUtils.create_env(
        env_meta=env_meta, env_class=None,
        env_name=mg_config.experiment.task.name,
        robot=mg_config.experiment.task.robot,
        gripper=mg_config.experiment.task.gripper,
        camera_names=[], camera_height=84, camera_width=84,
        render=False, render_offscreen=False,
        use_image_obs=False, use_depth_obs=False,
    )
    all_demos = MG_FileUtils.get_all_demos_from_dataset(
        dataset_path=source_dataset, filter_key=None, start=None, n=None,
    )
    env_interface_name, env_interface_type = MG_FileUtils.get_env_interface_info_from_dataset(
        dataset_path=source_dataset, demo_keys=all_demos,
    )
    raw_env = getattr(env, "base_env", env.env)
    env_interface = make_interface(
        name=env_interface_name, interface_type=env_interface_type, env=raw_env,
    )
    task_spec_json = mg_config.task.task_spec.dump()
    task_spec = MG_TaskSpec.from_json(json_string=task_spec_json)
    data_generator = DataGenerator(
        task_spec=task_spec, dataset_path=source_dataset, demo_keys=all_demos,
    )
    return env, env_interface, data_generator


def _augment_worker(args):
    """Worker: generate N demos for a task using standard MimicGen augmentation."""
    task_name, num_to_generate, source_dataset, output_dir, worker_id = args

    _lazy_imports()

    try:
        mg_config = load_mg_config(task_name)
        env, env_interface, data_generator = setup_env(task_name, source_dataset, mg_config)
    except Exception as e:
        print(f"    [worker-{worker_id}] ERROR: env setup failed: {e}", flush=True)
        return {"worker_id": worker_id, "attempted": 0, "success": 0, "failed": 0}

    # Each worker saves to its own subdirectory
    worker_dir = os.path.join(output_dir, f"worker_{worker_id}")
    os.makedirs(worker_dir, exist_ok=True)

    success = 0
    failed = 0
    max_consecutive_failures = 50  # safety limit

    consecutive_failures = 0
    while success < num_to_generate and consecutive_failures < max_consecutive_failures:
        try:
            result = data_generator.generate(
                env=env, env_interface=env_interface,
                select_src_per_subtask=False,
                transform_first_robot_pose=True,
                interpolate_from_last_target_pose=True,
                render=False, video_writer=None, video_skip=5,
                camera_names=None, pause_subtask=False,
            )
        except Exception:
            failed += 1
            consecutive_failures += 1
            continue

        if result.get("success", False):
            MG_FileUtils.write_demo_to_hdf5(
                folder=worker_dir, env=env,
                initial_state=result["initial_state"],
                states=result["states"],
                observations=result["observations"],
                datagen_info=result["datagen_infos"],
                actions=result["actions"],
                src_demo_inds=result.get("src_demo_inds"),
                src_demo_labels=result.get("src_demo_labels"),
            )
            success += 1
            consecutive_failures = 0
        else:
            failed += 1
            consecutive_failures += 1

    print(f"    [worker-{worker_id}] Done: {success}/{num_to_generate} "
          f"(failed {failed})", flush=True)
    return {"worker_id": worker_id, "attempted": success + failed,
            "success": success, "failed": failed}


def filter_valid_source(source_dataset: str, task_name: str, output_path: str) -> dict:
    """Filter source HDF5 to keep only demos with valid subtask termination signals."""
    import h5py

    # Load task_spec to get required signal names
    template_dir = MIMICGEN_ROOT / "mimicgen" / "mimicgen" / "exps" / "templates" / "robosuite"
    with open(template_dir / TASK_CONFIGS[task_name]) as f:
        ext_cfg = json.load(f)
    task_spec = ext_cfg.get("task", {}).get("task_spec", {})
    signal_names = []
    for subtask_dict in task_spec.values():
        sig = subtask_dict.get("subtask_term_signal")
        if sig is not None:
            signal_names.append(sig)

    report = {
        "task": task_name,
        "source_dataset": source_dataset,
        "filtered_dataset": output_path,
        "valid_demos": [],
        "rejected": [],
        "rejection_counts": {},
    }

    with h5py.File(source_dataset, "r") as src:
        demos = sorted(
            [k for k in src["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]),
        )

        # Identify valid demos
        valid_demos = []
        for dk in demos:
            demo_group = src[f"data/{dk}"]
            reject_reason = None

            if "datagen_info" not in demo_group:
                reject_reason = "missing_datagen_info"
            else:
                di = demo_group["datagen_info"]

            if reject_reason is None:
                ok = True
            else:
                ok = False

            for sig_name in signal_names:
                if not ok:
                    break
                key = f"subtask_term_signals/{sig_name}"
                if key not in di:
                    ok = False
                    reject_reason = f"missing_signal:{sig_name}"
                    break
                signal = np.array(di[key])
                transition = find_first_binary_transition(signal)
                if transition is None:
                    ok = False
                    if task_name == "stack" and sig_name == "grasp":
                        reject_reason = "no_grasp_transition"
                    elif signal.size == 0 or signal.max() < 0.5:
                        reject_reason = f"inactive_signal:{sig_name}"
                    else:
                        reject_reason = f"no_transition:{sig_name}"
                    break
            if ok:
                valid_demos.append(dk)
                report["valid_demos"].append(dk)
            else:
                reject_reason = reject_reason or "invalid_source_demo"
                report["rejected"].append({"demo_key": dk, "reason": reject_reason})
                report["rejection_counts"][reject_reason] = (
                    report["rejection_counts"].get(reject_reason, 0) + 1
                )

        if valid_demos:
            with h5py.File(output_path, "w") as dst:
                data_grp = dst.create_group("data")
                for attr_name in src["data"].attrs:
                    data_grp.attrs[attr_name] = src["data"].attrs[attr_name]

                total_samples = 0
                for i, dk in enumerate(valid_demos):
                    new_key = f"demo_{i}"
                    src.copy(f"data/{dk}", data_grp, name=new_key)
                    total_samples += data_grp[new_key].attrs.get("num_samples", 0)

                data_grp.attrs["total"] = total_samples

    report["valid_count"] = len(report["valid_demos"])
    report["rejected_count"] = len(report["rejected"])
    if report["valid_count"] > 0:
        patch_report = ensure_env_interface_attrs(output_path, task_name=task_name)
    else:
        patch_report = {"updated_count": 0}
    report["env_interface_attrs_patched"] = patch_report["updated_count"]
    report_path = os.path.join(os.path.dirname(output_path), "filtered_source_report.json")
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)
    return report


def augment_task(task_name, source_dataset, target_count, output_base, num_workers):
    """Augment a single task's success demos."""
    import h5py
    with h5py.File(source_dataset, "r") as f:
        existing = len([k for k in f["data"].keys() if k.startswith("demo_")])

    # Filter to valid source demos
    filtered_path = os.path.join(output_base, task_name, "filtered_source.hdf5")
    os.makedirs(os.path.dirname(filtered_path), exist_ok=True)
    filter_report = filter_valid_source(source_dataset, task_name, filtered_path)
    n_valid = filter_report["valid_count"]
    print(
        f"  [{task_name}] Source: {existing} demos, {n_valid} valid for augmentation, "
        f"target: {target_count}"
    )
    if filter_report["rejection_counts"]:
        print(f"  [{task_name}] Rejections: {filter_report['rejection_counts']}")

    if n_valid == 0:
        print(f"  [{task_name}] No valid source demos, skipping")
        return {
            "task": task_name,
            "existing": existing,
            "valid_source": 0,
            "target": target_count,
            "generated": 0,
            "source_rejections": filter_report["rejection_counts"],
        }

    # Use filtered source for augmentation
    source_dataset = filtered_path
    filtered_patch_count = int(filter_report.get("env_interface_attrs_patched", 0) or 0)
    if filtered_patch_count > 0:
        print(
            f"  [{task_name}] Patched env interface attrs on "
            f"{filtered_patch_count} filtered demos"
        )

    if target_count <= 0:
        print(f"  [{task_name}] Nothing to generate")
        return {
            "task": task_name,
            "existing": existing,
            "valid_source": n_valid,
            "target": target_count,
            "generated": 0,
            "source_rejections": filter_report["rejection_counts"],
            "filtered_env_interface_attrs_patched": filtered_patch_count,
            "augmented_env_interface_attrs_patched": 0,
        }

    # Output directory
    task_output_dir = os.path.join(output_base, task_name, "aug_tmp")
    os.makedirs(task_output_dir, exist_ok=True)

    # Distribute work across workers
    actual_workers = min(num_workers, target_count)
    per_worker = target_count // actual_workers
    remainder = target_count % actual_workers

    worker_args = []
    for i in range(actual_workers):
        n = per_worker + (1 if i < remainder else 0)
        worker_args.append((task_name, n, source_dataset, task_output_dir, i))

    print(f"  [{task_name}] Generating {target_count} demos with {actual_workers} workers...",
          flush=True)

    if actual_workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(actual_workers, maxtasksperchild=1) as pool:
            results = pool.map(_augment_worker, worker_args)
    else:
        results = [_augment_worker(worker_args[0])]

    total_generated = sum(r["success"] for r in results)
    total_failed = sum(r["failed"] for r in results)

    # Flatten worker subdirectories into task_output_dir
    for worker_subdir in sorted(glob_files(os.path.join(task_output_dir, "worker_*"))):
        if os.path.isdir(worker_subdir):
            for hdf5_file in glob_files(os.path.join(worker_subdir, "*.hdf5")):
                fname = os.path.basename(hdf5_file)
                wid = os.path.basename(worker_subdir)
                dst = os.path.join(task_output_dir, f"{wid}_{fname}")
                shutil.move(hdf5_file, dst)
            try:
                os.rmdir(worker_subdir)
            except OSError:
                pass

    # Merge into single HDF5
    augmented_hdf5 = os.path.join(output_base, task_name, "augmented_demos.hdf5")
    try:
        num_merged = MG_FileUtils.merge_all_hdf5(
            folder=task_output_dir, new_hdf5_path=augmented_hdf5,
            delete_folder=True, dry_run=False,
        )
        if num_merged and num_merged > 0:
            merged_patch_report = ensure_env_interface_attrs(
                augmented_hdf5,
                task_name=task_name,
            )
            if merged_patch_report["updated_count"] > 0:
                print(
                    f"  [{task_name}] Patched env interface attrs on "
                    f"{merged_patch_report['updated_count']} augmented demos"
                )
            print(f"  [{task_name}] Merged {num_merged} augmented demos → {augmented_hdf5}")
    except Exception as e:
        print(f"  [{task_name}] WARNING: merge failed ({e})")
        merged_patch_report = {"updated_count": 0}
    else:
        if not (num_merged and num_merged > 0):
            merged_patch_report = {"updated_count": 0}

    return {
        "task": task_name,
        "existing": existing,
        "valid_source": n_valid,
        "target": target_count,
        "generated": total_generated,
        "failed": total_failed,
        "source_rejections": filter_report["rejection_counts"],
        "filtered_env_interface_attrs_patched": filtered_patch_count,
        "augmented_env_interface_attrs_patched": merged_patch_report["updated_count"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Augment success demos via standard MimicGen data generation"
    )
    parser.add_argument("--task", type=str, help="Single task")
    parser.add_argument("--all-tasks", action="store_true", help="All 6 tasks")
    parser.add_argument("--target", type=int, default=None,
                        help="Target number of augmented demos per task")
    parser.add_argument("--multiplier", type=float, default=None,
                        help="Multiplier on existing demo count (e.g., 10 = 10x)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel workers")
    parser.add_argument("--success-dir", default=None,
                        help="Directory with success_demos.hdf5 files (default: auto)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as success-dir)")
    parser.add_argument("--source-file", default=None,
                        help=("Source HDF5 filename under each task directory. "
                              "Default: use success_demos_combined.hdf5 when present, "
                              "otherwise success_demos.hdf5."))
    args = parser.parse_args()

    if not args.task and not args.all_tasks:
        parser.error("Specify --task or --all-tasks")
    if args.target is None and args.multiplier is None:
        parser.error("Specify --target or --multiplier")

    _lazy_imports()

    success_dir = args.success_dir or SUCCESS_DEMOS_DIR
    output_base = args.output_dir or success_dir

    tasks = list(TASK_CONFIGS.keys()) if args.all_tasks else [args.task]

    all_results = []
    for task_name in tasks:
        source = resolve_source_dataset(success_dir, task_name, args.source_file)
        if not os.path.exists(source):
            wanted = args.source_file or f"{COMBINED_SOURCE_FILE} or {DEFAULT_SOURCE_FILE}"
            print(f"\n[{task_name}] No source HDF5 found ({wanted}), skipping")
            continue

        # Determine target count
        if args.target is not None:
            target = args.target
        else:
            import h5py
            with h5py.File(source, "r") as f:
                n = len([k for k in f["data"].keys() if k.startswith("demo_")])
            target = int(n * args.multiplier)

        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")

        result = augment_task(
            task_name=task_name, source_dataset=source,
            target_count=target, output_base=output_base,
            num_workers=args.num_workers,
        )
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_gen = 0
    for r in all_results:
        total_gen += r["generated"]
        print(f"  {r['task']:<25} {r['existing']:>4} existing → "
              f"+{r['generated']} augmented")
    print(f"\n  Total augmented: {total_gen}")

    # Save report
    report_path = os.path.join(output_base, "augmentation_report.json")
    with open(report_path, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "results": all_results}, f, indent=2)
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    exit(main())
