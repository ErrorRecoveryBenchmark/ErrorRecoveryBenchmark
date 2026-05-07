#!/usr/bin/env python
"""
Post-collection validation helpers for human recovery demos.

This module provides:
  - manifest de-duplication helpers
  - action-replay validation for freshly collected demos
  - scene-augmentation preflight checks using the current recovery augmenter
  - persistent JSON report helpers
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from error_benchmark.framework.recovery_types import RecoveryDemo


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REPORT_FILENAME = "human_demo_test_report.json"
VALIDATION_VERSION = "v1.0"
VALID_QUOTA_RULES = {
    "augmentable_only",
    "replay_only",
    "replay_and_augmentable",
    "success_only",
}

_RECOVERY_AUGMENTER_CLASS = None
_RECOVERY_AUGMENTER_LOAD_ERROR = None

# Re-export for backward compatibility; canonical definition in script_utils
from error_benchmark.scripts.utils.script_utils import load_error_scenes


def dedupe_demo_entries(demos: List[dict]) -> Tuple[List[dict], int]:
    """Keep only the last record for each demo_id."""
    deduped: List[dict] = []
    demo_id_to_index: Dict[str, int] = {}
    duplicates_removed = 0

    for entry in demos:
        demo_id = entry.get("demo_id")
        if not demo_id:
            deduped.append(entry)
            continue
        if demo_id in demo_id_to_index:
            deduped[demo_id_to_index[demo_id]] = entry
            duplicates_removed += 1
        else:
            demo_id_to_index[demo_id] = len(deduped)
            deduped.append(entry)

    return deduped, duplicates_removed


def normalize_quota_rule(quota_rule: Optional[str]) -> str:
    """Normalize and validate the collection quota rule."""

    normalized = str(quota_rule or "replay_only").strip().lower()
    if normalized not in VALID_QUOTA_RULES:
        raise ValueError(
            f"Unsupported quota_rule '{quota_rule}'. "
            f"Expected one of {sorted(VALID_QUOTA_RULES)}"
        )
    return normalized


def _check_replay_status(validation: dict, summary: dict) -> Tuple[bool, str]:
    """Check whether action replay passed. Returns (passed, reason)."""
    replay_success = summary.get("replay_success")
    if replay_success is True:
        return True, "accepted"
    if replay_success is False:
        return False, "replay_failed"
    replay_status = (validation.get("action_replay") or {}).get("status")
    if replay_status == "skipped_disabled":
        return False, "missing_replay_validation"
    if replay_status:
        return False, f"replay_{replay_status}"
    return False, "missing_replay_validation"


def _check_augmentable_status(validation: dict, summary: dict) -> Tuple[bool, str]:
    """Check whether scene augmentation passed. Returns (passed, reason)."""
    augmentable = summary.get("augmentable")
    if augmentable is True:
        return True, "accepted"
    if augmentable is False:
        return False, "non_augmentable"
    scene_aug_status = (validation.get("scene_augmentation") or {}).get("status")
    if scene_aug_status == "skipped_disabled":
        return False, "missing_augment_validation"
    if scene_aug_status:
        return False, scene_aug_status
    return False, "missing_augment_validation"


def determine_counts_toward_target(
    entry: dict,
    quota_rule: str = "replay_only",
) -> Tuple[bool, str]:
    """
    Decide whether a manifest entry counts toward the collection target.

    Explicit manifest annotations take precedence. Legacy entries without
    collection-validation metadata are counted only under ``success_only``.
    """

    normalized_rule = normalize_quota_rule(quota_rule)
    if not entry.get("success"):
        return False, "teleop_failed"

    explicit_count = entry.get("counts_toward_target")
    if explicit_count is not None:
        reason = str(
            entry.get("quota_reason")
            or ("accepted" if bool(explicit_count) else "rejected")
        )
        return bool(explicit_count), reason

    if normalized_rule == "success_only":
        return True, "accepted"

    validation = entry.get("collection_validation") or {}
    summary = validation.get("summary") or {}

    if normalized_rule in ("replay_only", "replay_and_augmentable"):
        passed, reason = _check_replay_status(validation, summary)
        if not passed:
            return False, reason

    if normalized_rule in ("augmentable_only", "replay_and_augmentable"):
        return _check_augmentable_status(validation, summary)

    # replay_only: replay passed above
    return True, "accepted"


def get_next_demo_attempt_index(
    demos: Sequence[dict],
    task_name: str,
    subtype_id: str,
) -> int:
    """Return the next per-subtype attempt index derived from manifest history."""

    prefix = f"recovery_{task_name}_{subtype_id}_"
    max_index = -1
    for entry in demos:
        if entry.get("subtype_id") != subtype_id:
            continue
        demo_id = str(entry.get("demo_id", ""))
        if not demo_id.startswith(prefix):
            continue
        suffix = demo_id[len(prefix):]
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return max_index + 1


def recompute_collected_counts(
    demos: Sequence[dict],
    target_demos: Optional[Dict[str, int]] = None,
    quota_rule: str = "replay_only",
) -> Dict[str, int]:
    """Recompute counted-demo totals per subtype from manifest entries."""

    normalized_rule = normalize_quota_rule(quota_rule)
    counts = Counter()
    for entry in demos:
        subtype_id = entry.get("subtype_id")
        counts_toward_target, _ = determine_counts_toward_target(
            entry,
            quota_rule=normalized_rule,
        )
        if subtype_id and counts_toward_target:
            counts[subtype_id] += 1

    if target_demos:
        for subtype_id in target_demos:
            counts.setdefault(subtype_id, 0)

    return dict(counts)


def recompute_collected_scene_ids(
    demos: Sequence[dict],
    target_scene_ids: Optional[Dict[str, Sequence[str]]] = None,
    quota_rule: str = "replay_only",
) -> Dict[str, List[str]]:
    """Recompute covered scene_ids per subtype from manifest entries."""

    normalized_rule = normalize_quota_rule(quota_rule)
    covered: Dict[str, set] = {}

    for entry in demos:
        subtype_id = entry.get("subtype_id")
        scene_id = entry.get("scene_id")
        counts_toward_target, _ = determine_counts_toward_target(
            entry,
            quota_rule=normalized_rule,
        )
        if not subtype_id or not scene_id or not counts_toward_target:
            continue
        covered.setdefault(subtype_id, set()).add(str(scene_id))

    if target_scene_ids:
        for subtype_id in target_scene_ids:
            covered.setdefault(subtype_id, set())

    return {
        subtype_id: sorted(scene_ids)
        for subtype_id, scene_ids in covered.items()
    }


def _upsert_by_demo_id(items: List[dict], item: dict) -> None:
    """Insert or replace an entry in a list, keyed by demo_id."""
    demo_id = item.get("demo_id")
    if not demo_id:
        items.append(item)
        return

    for idx, existing in enumerate(items):
        if existing.get("demo_id") == demo_id:
            items[idx] = item
            return
    items.append(item)


def upsert_demo_entry(demos: List[dict], entry: dict) -> None:
    """Insert or replace a manifest entry by demo_id."""
    _upsert_by_demo_id(demos, entry)


def upsert_validation_record(records: List[dict], record: dict) -> None:
    """Insert or replace a validation record by demo_id."""
    _upsert_by_demo_id(records, record)


def load_validation_records(report_path: Path) -> List[dict]:
    """Load existing validation records from a report file if present."""
    if not report_path.exists():
        return []

    try:
        with open(report_path) as f:
            payload = json.load(f)
    except Exception:
        return []

    records = payload.get("demos", []) if isinstance(payload, dict) else []
    deduped, _ = dedupe_demo_entries(records)
    return deduped


def _build_validation_summary(records: Sequence[dict]) -> dict:
    replay_success_count = 0
    augmentable_demo_count = 0
    total_scene_pairs = 0
    total_variant_pairs = 0
    augment_success_pairs = 0
    augment_reason_counts: Counter = Counter()

    for record in records:
        summary = record.get("summary", {})
        if summary.get("replay_success") is True:
            replay_success_count += 1
        if summary.get("augmentable") is True:
            augmentable_demo_count += 1

        scene_aug = record.get("scene_augmentation", {})
        total_scene_pairs += int(scene_aug.get("tested_scene_count", 0) or 0)
        total_variant_pairs += int(
            scene_aug.get("tested_variant_count", scene_aug.get("tested_scene_count", 0))
            or 0
        )
        augment_success_pairs += int(scene_aug.get("success_count", 0) or 0)
        augment_reason_counts.update(scene_aug.get("failure_reason_counts", {}))

    return {
        "validated_demo_count": len(records),
        "replay_success_count": replay_success_count,
        "augmentable_demo_count": augmentable_demo_count,
        "total_scene_pairs": total_scene_pairs,
        "total_variant_pairs": total_variant_pairs,
        "augment_success_pairs": augment_success_pairs,
        "augment_reason_counts": dict(augment_reason_counts),
    }


def save_validation_report(
    report_path: Path,
    task_name: str,
    records: Sequence[dict],
    validation_config: Optional[dict] = None,
) -> None:
    """Save the post-collection validation report."""
    payload = {
        "version": VALIDATION_VERSION,
        "task_name": task_name,
        "updated": datetime.now().isoformat(),
        "config": validation_config or {},
        "summary": _build_validation_summary(records),
        "demos": list(records),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(payload, f, indent=2)


def tag_npz_validation(npz_path, validation_record: Optional[dict],
                       counts_toward_target: bool) -> None:
    """Write validation tags into an existing NPZ file.

    Adds three boolean fields:
      - ``validation_passed``: whether this demo counts toward the collection target
      - ``replay_passed``: whether action replay succeeded
      - ``augmentable``: whether scene augmentation succeeded

    If *validation_record* is ``None`` (validation was not run), only
    ``validation_passed`` is written; the other two are omitted so that
    downstream code can distinguish "not tested" from "tested & failed".
    """
    if not npz_path or not Path(npz_path).exists():
        return

    npz_path = Path(npz_path)
    existing = dict(np.load(npz_path, allow_pickle=True))
    existing['validation_passed'] = np.bool_(counts_toward_target)

    if validation_record is not None:
        summary = validation_record.get('summary') or {}
        replay = summary.get('replay_success')
        aug = summary.get('augmentable')
        if replay is not None:
            existing['replay_passed'] = np.bool_(replay)
        if aug is not None:
            existing['augmentable'] = np.bool_(aug)

    np.savez_compressed(npz_path, **existing)


def save_manifest(manifest_path: Path, demos: list, status) -> None:
    """Save manifest with demo metadata and collection status."""
    from error_benchmark.scripts.utils.script_utils import NumpyEncoder

    manifest = {
        'version': 'v1.2',
        'created': datetime.now().isoformat(),
        'status': status.to_dict(),
        'demos': demos,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, cls=NumpyEncoder)


def _augment_scene_worker(args: dict) -> dict:
    """Worker function for parallel scene augmentation testing.

    Each worker creates its own env + augmenter to avoid shared state.
    Must be a top-level function for pickling by multiprocessing.
    """
    import sys
    import warnings
    warnings.filterwarnings("ignore", message=".*private macro file.*")
    warnings.filterwarnings("ignore", message=".*robosuite task zoo.*")

    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "robosuite"))
    sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "mimicgen"))

    from error_benchmark.scripts.utils.script_utils import create_env, load_task_registry
    from error_benchmark.framework.env_wrapper import EnvWrapper

    task_config = args["task_config"]
    dataset_path = args["dataset_path"]
    aug_config = args["aug_config"]
    scene = args["scene"]
    demo_dict = args["demo_dict"]
    demo_npz_path = args["demo_npz_path"]
    seed = args["seed"]
    diagnose = args.get("diagnose", True)

    try:
        env = create_env(task_config, dataset_path, enable_camera=False, has_renderer=False)
        env_wrapper = EnvWrapper(env, task_config)
        rng = np.random.RandomState(seed)
        augmenter_cls = _load_recovery_augmenter_class()
        augmenter = augmenter_cls(env_wrapper, task_config, aug_config, rng)
    except Exception as exc:
        return {
            "scene_id": scene.get("scene_id", ""),
            "success": False,
            "reason": f"worker_init_error:{type(exc).__name__}",
            "detail": str(exc),
        }

    # Reconstruct demo
    demo = RecoveryDemo.from_dict(demo_dict)
    if demo_npz_path and Path(demo_npz_path).exists():
        data = np.load(demo_npz_path, allow_pickle=True)
        demo.actions = data.get("actions")
        demo.states = data.get("states")
        demo.eef_positions = data.get("eef_positions")
        demo.eef_orientations = data.get("eef_orientations")
        demo.target_poses = data.get("target_poses")
        demo.gripper_states = data.get("gripper_states")
        obj_positions = {}
        obj_orientations = {}
        for key in data.files:
            if key.startswith("obj_") and key.endswith("_quat"):
                obj_orientations[key[4:-5]] = data[key]
            elif key.startswith("obj_"):
                obj_positions[key[4:]] = data[key]
        if obj_positions:
            demo.object_positions = obj_positions
        if obj_orientations:
            demo.object_orientations = obj_orientations

    try:
        augmented = augmenter._replay_with_warping(demo, scene, diagnose=diagnose)
    except FileNotFoundError as exc:
        return {
            "scene_id": scene.get("scene_id", ""),
            "success": False,
            "reason": "missing_prepared_source",
            "detail": str(exc),
            "_is_fatal": True,
        }
    except Exception as exc:
        reason = f"exception_{type(exc).__name__}"
        if isinstance(exc, ValueError):
            reason = "exception_ValueError"
        return {
            "scene_id": scene.get("scene_id", ""),
            "success": False,
            "reason": reason,
            "detail": f"{type(exc).__name__}: {exc}",
        }

    metadata = getattr(augmented, "metadata", {}) or {}
    base_scene_id = metadata.get("base_scene_id", scene.get("scene_id", ""))
    variant_id = metadata.get("variant_id", base_scene_id)
    variant_index = metadata.get("variant_index")

    if augmented is None or not augmented.success or augmented.actions is None:
        fail_reason = metadata.get("_fail_reason", "scene_replay_not_success_or_empty")
        entry = {
            "scene_id": scene.get("scene_id", ""),
            "base_scene_id": base_scene_id,
            "variant_id": variant_id,
            "variant_index": variant_index,
            "success": False,
            "reason": fail_reason,
        }
        diagnosis = metadata.get("_diagnosis")
        if diagnosis:
            entry["diagnosis"] = diagnosis
        return entry

    return {
        "scene_id": scene.get("scene_id", ""),
        "base_scene_id": base_scene_id,
        "variant_id": variant_id,
        "variant_index": variant_index,
        "randomized_objects": metadata.get("randomized_objects", []),
        "anchored_objects": metadata.get("anchored_objects", []),
        "success": True,
        "augmented_id": augmented.augmented_id,
        "num_steps": int(augmented.num_steps),
    }


def _load_recovery_augmenter_class():
    global _RECOVERY_AUGMENTER_CLASS, _RECOVERY_AUGMENTER_LOAD_ERROR
    if _RECOVERY_AUGMENTER_CLASS is not None:
        return _RECOVERY_AUGMENTER_CLASS
    if _RECOVERY_AUGMENTER_LOAD_ERROR is not None:
        raise _RECOVERY_AUGMENTER_LOAD_ERROR

    module_path = (
        PROJECT_ROOT
        / "error_benchmark"
        / "scripts"
        / "augmentation"
        / "3_mimicgen_recovery_augment.py"
    )
    if not module_path.exists():
        _RECOVERY_AUGMENTER_LOAD_ERROR = FileNotFoundError(
            f"Missing recovery augmentation module: {module_path}"
        )
        raise _RECOVERY_AUGMENTER_LOAD_ERROR

    spec = importlib.util.spec_from_file_location(
        "error_benchmark_recovery_augment",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    _RECOVERY_AUGMENTER_CLASS = module.RecoveryAugmenter
    return _RECOVERY_AUGMENTER_CLASS


class RecoveryDemoValidator:
    """Runs post-collection validation for a human recovery demo."""

    def __init__(
        self,
        env_wrapper,
        task_config: dict,
        aug_config: Optional[dict],
        scenes_dir: Path,
        validation_config: Optional[dict],
        rng: Optional[np.random.RandomState] = None,
        augmenter_factory: Optional[Callable] = None,
        scenes_loader: Optional[Callable[[Path, str], List[dict]]] = None,
        demos_dir: Optional[Path] = None,
    ):
        self.env_wrapper = env_wrapper
        self.task_config = task_config
        self.aug_config = aug_config or {}
        self.scenes_dir = Path(scenes_dir)
        self.validation_config = validation_config or {}
        self.rng = rng or np.random.RandomState(0)
        self.augmenter_factory = augmenter_factory
        self.scenes_loader = scenes_loader or load_error_scenes
        self.demos_dir = Path(demos_dir) if demos_dir else None
        self._augmenter = None
        self._augmenter_init_error = None
        self._target_scenes_cache: Dict[str, List[dict]] = {}

    def validate_demo(self, demo: RecoveryDemo,
                      video_dir: Optional[Path] = None,
                      mimicgen_video_dir: Optional[Path] = None) -> dict:
        """Run all enabled validation checks for a collected demo.

        Args:
            demo: The recovery demo to validate.
            video_dir: If provided, record action replay to MP4 in this directory.
            mimicgen_video_dir: If provided, record MimicGen augmentation attempts
                to MP4 in this directory. Forces serial execution.
        """
        record = {
            "demo_id": demo.demo_id,
            "task_name": demo.task_name,
            "subtype_id": demo.subtype_id,
            "scene_id": demo.scene_id,
            "success_recorded": bool(demo.success),
            "validated_at": datetime.now().isoformat(),
        }

        replay_result = {"status": "skipped_disabled"}
        if self.validation_config.get("check_action_replay", True):
            recorder = None
            if video_dir is not None:
                from error_benchmark.framework.video_recorder import VideoRecorder
                video_dir = Path(video_dir)
                video_dir.mkdir(parents=True, exist_ok=True)
                video_path = video_dir / f"replay_{demo.demo_id}.mp4"
                recorder = VideoRecorder(
                    video_path, fps=20,
                    camera_names=["agentview", "robot0_eye_in_hand"],
                )
            replay_result = self._validate_action_replay(demo,
                                                         video_recorder=recorder)
        record["action_replay"] = replay_result

        scene_aug_result = {"status": "skipped_disabled"}
        if self.validation_config.get("check_scene_augmentation", True):
            scene_aug_result = self._validate_scene_augmentation(
                demo, mimicgen_video_dir=mimicgen_video_dir)
        record["scene_augmentation"] = scene_aug_result

        record["summary"] = {
            "replay_success": replay_result.get("replay_success"),
            "augmentable": scene_aug_result.get("is_augmentable"),
            "augment_success_count": scene_aug_result.get("success_count", 0),
            "augment_tested_scene_count": scene_aug_result.get(
                "tested_scene_count", 0
            ),
            "augment_tested_variant_count": scene_aug_result.get(
                "tested_variant_count", 0
            ),
        }
        return record

    def _validate_action_replay(self, demo: RecoveryDemo,
                                video_recorder=None) -> dict:
        if demo.states is None or len(demo.states) == 0:
            return {
                "status": "missing_states",
                "replay_success": None,
                "success_match": False,
            }
        if demo.actions is None or len(demo.actions) == 0:
            return {
                "status": "missing_actions",
                "replay_success": None,
                "success_match": False,
            }

        try:
            self.env_wrapper.set_sim_state_flat(np.asarray(demo.states[0]))
            self.env_wrapper.forward()
        except Exception as exc:
            return {
                "status": "init_state_error",
                "replay_success": None,
                "success_match": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        if video_recorder is not None:
            video_recorder.capture_frame(
                self.env_wrapper,
                overlay_text=[f"replay: {demo.demo_id}", "step: 0 (init)"],
            )

        actions_replayed = 0
        try:
            for action in np.asarray(demo.actions):
                self.env_wrapper.step(action)
                actions_replayed += 1
                if video_recorder is not None:
                    video_recorder.capture_frame(
                        self.env_wrapper,
                        overlay_text=[
                            f"replay: {demo.demo_id}",
                            f"step: {actions_replayed}/{len(demo.actions)}",
                        ],
                    )
            replay_success = bool(self.env_wrapper.check_success())
        except Exception as exc:
            if video_recorder is not None:
                video_recorder.capture_frame(
                    self.env_wrapper,
                    overlay_text=[f"EXCEPTION at step {actions_replayed}"],
                )
                video_recorder.close()
            return {
                "status": "replay_exception",
                "replay_success": None,
                "success_match": False,
                "actions_replayed": actions_replayed,
                "error": f"{type(exc).__name__}: {exc}",
            }

        if video_recorder is not None:
            result_text = "SUCCESS" if replay_success else "FAIL"
            video_recorder.capture_frame(
                self.env_wrapper,
                overlay_text=[
                    f"replay: {demo.demo_id}",
                    f"result: {result_text}",
                    f"steps: {actions_replayed}",
                ],
            )
            video_recorder.close()

        return {
            "status": "ok",
            "actions_replayed": actions_replayed,
            "replay_success": replay_success,
            "recorded_success": bool(demo.success),
            "success_match": replay_success == bool(demo.success),
        }

    def _validate_scene_augmentation(self, demo: RecoveryDemo,
                                      mimicgen_video_dir: Optional[Path] = None) -> dict:
        if not demo.success:
            return {
                "status": "skipped_demo_not_successful",
                "tested_scene_count": 0,
                "success_count": 0,
                "is_augmentable": False,
            }

        all_scenes = self._get_target_scenes(demo.subtype_id)
        if not all_scenes:
            return {
                "status": "no_target_scenes",
                "tested_scene_count": 0,
                "success_count": 0,
                "is_augmentable": False,
            }

        # Same-scene variant mode: only use the demo's own source scene.
        # Variants are generated by randomising non-anchored object placements
        # on top of the same error scene (analogous to MimicGen env.reset()).
        source_scene = self._find_demo_source_scene(demo, all_scenes)
        if source_scene is None:
            return {
                "status": "source_scene_not_found",
                "tested_scene_count": 0,
                "success_count": 0,
                "is_augmentable": False,
                "detail": f"scene_id={demo.scene_id} not found in {len(all_scenes)} scenes",
            }

        # Repeat the source scene N times so the validator runs N independent
        # variant attempts (each call to _replay_with_warping generates a
        # fresh variant via ErrorSceneVariantGenerator).
        max_variants = int(self.validation_config.get("max_target_scenes", 10) or 10)
        target_scenes = [source_scene] * max_variants

        if self.augmenter_factory is not None or mimicgen_video_dir is not None:
            return self._validate_scene_augmentation_serial(
                demo, target_scenes, video_dir=mimicgen_video_dir)

        # Resolve dataset_path for worker processes
        from error_benchmark.scripts.utils.script_utils import load_task_registry
        task_name = self.task_config.get("task_name", "")
        try:
            task_info = load_task_registry(task_name)
            dataset_path = task_info["dataset_path"]
        except Exception:
            dataset_path = ""

        # Find demo NPZ path for workers to load arrays
        demo_npz_path = ""
        if self.demos_dir:
            demos_dir = self.demos_dir
        else:
            # Fallback: derive from scenes_dir
            # scenes_dir = .../outputs/v5_training/{task}/scenes
            # demos are at  .../outputs/recovery/demos/{task}
            demos_dir = self.scenes_dir.parent.parent.parent / "recovery" / "demos" / task_name

        subtype_dir = demos_dir / demo.subtype_id
        candidate = subtype_dir / f"{demo.demo_id}.npz"
        if candidate.exists():
            demo_npz_path = str(candidate)
        if not demo_npz_path:
            # Search in manifest
            manifest_path = demos_dir / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                for d in manifest.get("demos", []):
                    if d.get("demo_id") == demo.demo_id and d.get("npz_path"):
                        demo_npz_path = d["npz_path"]
                        break

        # Build worker args for each scene
        demo_dict = demo.to_dict()
        max_workers = min(max(os.cpu_count() - 2, 1), len(target_scenes))
        worker_args_list = []
        for i, scene in enumerate(target_scenes):
            worker_args_list.append({
                "task_config": self.task_config,
                "dataset_path": dataset_path,
                "aug_config": self.aug_config,
                "scene": scene,
                "demo_dict": demo_dict,
                "demo_npz_path": demo_npz_path,
                "seed": int(self.rng.randint(0, 2**31)) + i,
                "diagnose": True,
            })

        # Run in parallel
        scene_results = []
        successful_scene_ids = []
        successful_variant_ids = []
        failure_reason_counts: Counter = Counter()

        logger = logging.getLogger(__name__)
        logger.debug(
            "Augmentation validation: %d scenes, %d workers",
            len(target_scenes), max_workers,
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_augment_scene_worker, args): i
                for i, args in enumerate(worker_args_list)
            }
            for future in as_completed(futures):
                result = future.result()

                # Handle fatal errors (e.g., missing prepared source)
                if result.get("_is_fatal"):
                    executor.shutdown(wait=False, cancel_futures=True)
                    return {
                        "status": "skipped_missing_prepared_source",
                        "tested_scene_count": 0,
                        "success_count": 0,
                        "is_augmentable": False,
                        "error": result.get("detail", ""),
                    }

                if result.get("success"):
                    successful_scene_ids.append(result.get("base_scene_id", ""))
                    successful_variant_ids.append(result.get("variant_id", ""))
                else:
                    reason = result.get("reason", "unknown")
                    failure_reason_counts[reason] += 1

                scene_results.append(result)

                # Early stop: cancel remaining workers on first success
                if (
                    result.get("success")
                    and self.validation_config.get("early_stop_on_success", True)
                ):
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

        # Sort by scene_id for deterministic output
        scene_results.sort(key=lambda r: r.get("scene_id", ""))

        success_count = len(successful_scene_ids)
        return {
            "status": "ok",
            "tested_scene_count": len(scene_results),
            "tested_variant_count": len(scene_results),
            "success_count": success_count,
            "is_augmentable": success_count > 0,
            "successful_scene_ids": successful_scene_ids,
            "successful_variant_ids": successful_variant_ids,
            "failure_reason_counts": dict(failure_reason_counts),
            "scene_results": scene_results,
        }

    def _validate_scene_augmentation_serial(
        self,
        demo: RecoveryDemo,
        target_scenes: List[dict],
        video_dir: Optional[Path] = None,
    ) -> dict:
        """Serial validation path. Also used when MimicGen video is requested."""
        try:
            augmenter = self._get_augmenter()
        except FileNotFoundError as exc:
            return {
                "status": "skipped_missing_prepared_source",
                "tested_scene_count": 0,
                "success_count": 0,
                "is_augmentable": False,
                "error": str(exc),
            }

        if video_dir is not None:
            video_dir = Path(video_dir)
            video_dir.mkdir(parents=True, exist_ok=True)

        scene_results = []
        successful_scene_ids = []
        successful_variant_ids = []
        failure_reason_counts: Counter = Counter()

        for scene in target_scenes:
            try:
                augmented = augmenter._replay_with_warping(
                    demo, scene, video_dir=video_dir)
            except Exception as exc:
                reason = self._classify_augment_exception(exc)
                failure_reason_counts[reason] += 1
                scene_results.append({
                    "scene_id": scene.get("scene_id", ""),
                    "success": False,
                    "reason": reason,
                    "detail": f"{type(exc).__name__}: {exc}",
                })
                continue

            metadata = getattr(augmented, "metadata", {}) or {}
            base_scene_id = metadata.get("base_scene_id", scene.get("scene_id", ""))
            variant_id = metadata.get("variant_id", base_scene_id)
            variant_index = metadata.get("variant_index")

            if augmented is None or not augmented.success or augmented.actions is None:
                fail_reason = metadata.get("_fail_reason", "scene_replay_not_success_or_empty")
                failure_reason_counts[fail_reason] += 1
                entry = {
                    "scene_id": scene.get("scene_id", ""),
                    "base_scene_id": base_scene_id,
                    "variant_id": variant_id,
                    "variant_index": variant_index,
                    "success": False,
                    "reason": fail_reason,
                }
                diagnosis = metadata.get("_diagnosis")
                if diagnosis:
                    entry["diagnosis"] = diagnosis
                scene_results.append(entry)
                continue

            successful_scene_ids.append(base_scene_id)
            successful_variant_ids.append(variant_id)
            scene_results.append({
                "scene_id": scene.get("scene_id", ""),
                "base_scene_id": base_scene_id,
                "variant_id": variant_id,
                "variant_index": variant_index,
                "randomized_objects": metadata.get("randomized_objects", []),
                "anchored_objects": metadata.get("anchored_objects", []),
                "success": True,
                "augmented_id": augmented.augmented_id,
                "num_steps": int(augmented.num_steps),
            })

            # Early stop: one success is enough to confirm augmentability
            if self.validation_config.get("early_stop_on_success", True):
                break

        scene_results.sort(key=lambda r: r.get("scene_id", ""))
        success_count = len(successful_scene_ids)
        return {
            "status": "ok",
            "tested_scene_count": len(target_scenes),
            "tested_variant_count": len(scene_results),
            "success_count": success_count,
            "is_augmentable": success_count > 0,
            "successful_scene_ids": successful_scene_ids,
            "successful_variant_ids": successful_variant_ids,
            "failure_reason_counts": dict(failure_reason_counts),
            "scene_results": scene_results,
        }

    def _get_target_scenes(self, subtype_id: str) -> List[dict]:
        if subtype_id not in self._target_scenes_cache:
            scenes = self.scenes_loader(self.scenes_dir, subtype_id)
            scenes = sorted(scenes, key=lambda scene: scene.get("scene_id", ""))
            self._target_scenes_cache[subtype_id] = scenes
        return list(self._target_scenes_cache[subtype_id])

    def _select_target_scenes(self, scenes: List[dict], source_scene_id: str) -> List[dict]:
        selected = list(scenes)
        if self.validation_config.get("exclude_source_scene", False):
            selected = [
                scene for scene in selected if scene.get("scene_id") != source_scene_id
            ]
        max_target_scenes = int(self.validation_config.get("max_target_scenes", 10) or 0)
        if max_target_scenes > 0:
            selected = selected[:max_target_scenes]
        return selected

    @staticmethod
    def _find_demo_source_scene(demo, all_scenes: list):
        """Find the error scene that *demo* was collected on."""
        for scene in all_scenes:
            if scene.get("scene_id") == demo.scene_id:
                return scene
        # Fallback: construct from demo's scene_npz_path
        if getattr(demo, "scene_npz_path", None) and Path(demo.scene_npz_path).exists():
            return {
                "scene_id": demo.scene_id,
                "_npz_path": demo.scene_npz_path,
            }
        return None

    def _get_augmenter(self):
        if self._augmenter is not None:
            return self._augmenter
        if self._augmenter_init_error is not None:
            raise self._augmenter_init_error

        augmenter_factory = self.augmenter_factory or self._default_augmenter_factory
        try:
            self._augmenter = augmenter_factory(
                self.env_wrapper,
                self.task_config,
                self.aug_config,
                self.rng,
            )
        except Exception as exc:
            self._augmenter_init_error = exc
            raise
        return self._augmenter

    @staticmethod
    def _default_augmenter_factory(env_wrapper, task_config, aug_config, rng):
        augmenter_cls = _load_recovery_augmenter_class()
        return augmenter_cls(env_wrapper, task_config, aug_config, rng)

    @staticmethod
    def _classify_augment_exception(exc: Exception) -> str:
        if isinstance(exc, FileNotFoundError):
            return "missing_prepared_source"
        if isinstance(exc, ValueError):
            return "exception_ValueError"
        return f"exception_{type(exc).__name__}"


def log_validation_result(logger: logging.Logger, record: dict) -> None:
    """Emit a compact validation summary for collection-time logs."""
    replay = record.get("action_replay", {})
    scene_aug = record.get("scene_augmentation", {})

    replay_status = replay.get("status", "unknown")
    if replay_status == "ok":
        logger.info(
            "      replay: %s | success=%s | match=%s",
            replay_status,
            replay.get("replay_success"),
            replay.get("success_match"),
        )
    else:
        logger.info("      replay: %s", replay_status)

    aug_status = scene_aug.get("status", "unknown")
    if aug_status == "ok":
        logger.info(
            "      augmentability: %s | %s/%s scenes",
            aug_status,
            scene_aug.get("success_count", 0),
            scene_aug.get("tested_scene_count", 0),
        )
        # Print failure reason breakdown when augmentation failed
        if scene_aug.get("success_count", 0) == 0:
            reason_counts = scene_aug.get("failure_reason_counts", {})
            for reason, count in sorted(reason_counts.items()):
                logger.info("        fail: %s × %d", reason, count)
            # Print first scene's detail/diagnosis
            for result in scene_aug.get("scene_results", []):
                # Print exception detail if available
                detail = result.get("detail")
                if detail:
                    logger.info("        detail: %s", detail)
                    break
                diag = result.get("diagnosis")
                if diag:
                    for key in ("obj_cubeA_pos", "obj_cubeB_pos",
                                "cubeA_cubeB_xy_dist", "cubeA_height",
                                "cubeA_above_cubeB", "eef_pos",
                                "gripper_closed"):
                        if key in diag:
                            logger.info("        %s: %s", key, diag[key])
                    subtasks = diag.get("subtasks_executed")
                    if subtasks:
                        parts = [f"{s['label']}({s['actions']})" for s in subtasks]
                        logger.info("        subtasks: %s", " → ".join(parts))
                    break  # Only show first scene's diagnosis
    else:
        logger.info("      augmentability: %s", aug_status)
