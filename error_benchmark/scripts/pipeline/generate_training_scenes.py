#!/usr/bin/env python
"""
Generate Training Error Scenes

Generates error scenes for training with uniform time-segment sampling:
- Merges injectable time segments across trajectories into a virtual timeline
- Uniformly samples N points (default 10) per subtype from the timeline
- Enforces per-object coverage: every (object × error_name) must have ≥1 scene
- Output stored in v5_training/{task}/scenes/

Usage:
    python error_benchmark/scripts/pipeline/generate_training_scenes.py \
        --task stack

    # All 6 tasks:
    python error_benchmark/scripts/pipeline/generate_training_scenes.py
"""

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import sys
import yaml
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

_MP_CTX = mp.get_context('spawn')


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ─── Setup paths ────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "robosuite"))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "mimicgen"))

from error_benchmark.framework.clean_trajectory_collector import CleanTrajectoryCollector
from error_benchmark.framework.context_replay import InjectionEngine
from error_benchmark.framework.error_skills import get_skill
from error_benchmark.framework.error_taxonomy_v5 import get_all_subtypes
from error_benchmark.framework.opportunity_scanner import OpportunityScanner
from error_benchmark.framework.quota_scheduler import QuotaScheduler
from error_benchmark.framework.utils.controlled_rng import ControlledRng
from error_benchmark.framework.logger_setup import setup_logging
from error_benchmark.scripts.utils.script_utils import create_env, load_task_registry


ALL_TASKS = ["pick_place", "coffee", "stack", "stack_three", "threading", "three_piece_assembly"]
GENERATOR_VERSION = "v5.0_training_v3"
GENERATOR_SHA256 = "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _config_sha256(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), cls=NumpyEncoder)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _record_generation_provenance(
    scene,
    *,
    mode,
    point_index,
    fraction,
    scenes_per_subtype,
    base_seed,
    config,
):
    scene.labels.update({
        "generation_mode": mode,
        "magnitude_fraction": None if fraction is None else float(fraction),
        "training_point_index": point_index,
        "scenes_per_subtype": int(scenes_per_subtype),
        "base_seed": int(base_seed),
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": GENERATOR_SHA256,
        "config_sha256": _config_sha256(config),
        "schedule_mode": "timeline_uniform" if mode == "controlled_magnitude" else "coverage_repair",
    })


def _try_inject(engine, trajectory, opp, skill, base_seed, point_idx,
                 max_retries=10, fraction=None):
    """Try injecting at one opportunity with multiple seeds.

    Args:
        fraction: If not None, wrap rng with ControlledRng for magnitude control.
            Value in [0.0, 1.0] — 0.0 = minimum magnitude, 1.0 = maximum.

    Returns:
        ErrorScene or None.
    """
    for attempt in range(max_retries):
        seed = (base_seed + hash(opp.subtype_id()) + point_idx * 1000
                + attempt) % (2**31)
        rng = np.random.RandomState(seed)
        if fraction is not None:
            rng = ControlledRng(rng, fraction)
        try:
            scene = engine.execute(trajectory, opp, skill, rng)
        except Exception:
            scene = None
        if scene is not None:
            scene.labels["injection_seed"] = int(seed)
            scene.labels["injection_attempt"] = int(attempt)
            return scene
    return None


def _worker_process(worker_id, subtype_batch, task_name, config, dataset_path,
                     task_config, traj_dir, task_output, skills_config,
                     scenes_per_subtype, base_seed, opp_path):
    """Worker process: create own env, process assigned subtypes, return results."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    wlog = logging.getLogger(f"worker-{worker_id}")
    wlog.info(f"Worker {worker_id} starting: {len(subtype_batch)} subtypes")

    # Each worker creates its own env (robosuite is not fork-safe)
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "robosuite"))
    sys.path.insert(0, str(PROJECT_ROOT / "shared" / "mimicgen_workspace" / "mimicgen"))

    from error_benchmark.scripts.utils.script_utils import create_env
    from error_benchmark.framework.env_wrapper import EnvWrapper
    from error_benchmark.framework.context_replay import InjectionEngine
    from error_benchmark.framework.clean_trajectory_collector import CleanTrajectoryCollector
    from error_benchmark.framework.opportunity_scanner import OpportunityScanner
    from error_benchmark.framework.quota_scheduler import QuotaScheduler
    from error_benchmark.framework.error_skills import get_skill

    env = create_env(task_config, dataset_path, enable_camera=False)
    env_wrapper = EnvWrapper(env, task_config)

    injection_config = config.get('injection', {})
    engine = InjectionEngine(
        env=env_wrapper, task_config=task_config,
        render_window=injection_config.get('context_window', 50),
        post_injection_steps=injection_config.get('post_injection_steps', 20),
    )

    trajectories = CleanTrajectoryCollector.load(str(traj_dir))
    traj_map = {t.trajectory_id: t for t in trajectories}

    all_opportunities = OpportunityScanner.load(str(opp_path))
    opp_by_subtype = defaultdict(list)
    for opp in all_opportunities:
        opp_by_subtype[opp.subtype_id()].append(opp)

    main_rng = np.random.RandomState(base_seed + worker_id)
    worker_results = {}
    worker_scenes = []
    worker_coverage = defaultdict(int)

    for subtype_id in subtype_batch:
        opps = opp_by_subtype.get(subtype_id, [])
        if not opps:
            worker_results[subtype_id] = {'count': 0, 'total_opps': 0}
            continue

        error_name = opps[0].error_name
        degree = opps[0].degree

        try:
            skill_params = skills_config.get(error_name, {})
            skill = get_skill(error_name, skill_params)
        except KeyError:
            worker_results[subtype_id] = {'count': 0, 'total_opps': len(opps)}
            continue

        timeline = QuotaScheduler.build_virtual_timeline(opps, merge_gap=10)
        if not timeline:
            worker_results[subtype_id] = {'count': 0, 'total_opps': len(opps)}
            continue

        sampled = QuotaScheduler.sample_from_timeline(timeline, n=scenes_per_subtype)
        scenes_done = 0
        failure_reasons = {}

        for i, opp in enumerate(sampled):
            trajectory = traj_map.get(opp.trajectory_id)
            if trajectory is None:
                continue

            fraction = i / max(scenes_per_subtype - 1, 1)
            scene = _try_inject(engine, trajectory, opp, skill, base_seed, i,
                                fraction=fraction)

            if scene is None:
                reason = getattr(engine, 'last_failure_reason', 'unknown')
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                alt_indices = main_rng.choice(
                    len(timeline), size=min(20, len(timeline)), replace=False)
                for alt_idx in alt_indices:
                    alt_opp = timeline[alt_idx]
                    alt_traj = traj_map.get(alt_opp.trajectory_id)
                    if alt_traj is None:
                        continue
                    scene = _try_inject(
                        engine, alt_traj, alt_opp, skill,
                        base_seed + 50000, alt_idx, max_retries=3,
                        fraction=fraction)
                    if scene is not None:
                        break

            if scene is None:
                continue

            _record_generation_provenance(
                scene,
                mode="controlled_magnitude",
                point_index=i,
                fraction=fraction,
                scenes_per_subtype=scenes_per_subtype,
                base_seed=base_seed,
                config=config,
            )

            target_obj = scene.error_spec.target_object or ''
            scene.labels['target_object'] = target_obj

            seq = scenes_done
            hash_input = (
                f"train_{task_name}_{subtype_id}_{seq:03d}_"
                f"{scene.clean_trajectory_ref}_{scene.injection_step}"
            )
            hash8 = hashlib.sha1(hash_input.encode()).hexdigest()[:8]
            scene.scene_id = f"v5_train_{subtype_id}_{seq:03d}_{hash8}"
            scene.labels['training_seq'] = seq

            scene_json_path = task_output / f"{scene.scene_id}.json"
            with open(scene_json_path, 'w') as f:
                json.dump(scene.to_dict(), f, indent=2, cls=NumpyEncoder)

            if scene._pre_state is not None:
                npz_path = task_output / f"{scene.scene_id}.npz"
                save_dict = {'sim_state': scene._pre_state}
                if scene._post_state is not None:
                    save_dict['post_sim_state'] = scene._post_state
                np.savez_compressed(npz_path, **save_dict)

            scenes_done += 1
            worker_coverage[(target_obj, error_name)] += 1
            worker_scenes.append({
                'scene_id': scene.scene_id,
                'subtype_id': subtype_id,
                'error_name': error_name,
                'degree': degree,
                'seq': seq,
                'injection_frame': scene.injection_step,
                'target_object': target_obj,
                'task_phase': scene.labels.get('task_phase', ''),
            })

        worker_results[subtype_id] = {
            'count': scenes_done,
            'total_opps': len(opps),
            'timeline_length': len(timeline),
            'failure_reasons': dict(failure_reasons) if failure_reasons else {},
        }

    try:
        env.close()
    except Exception:
        pass

    # Convert defaultdict and tuple keys for pickling
    coverage_serializable = {f"{k[0]}|{k[1]}": v for k, v in worker_coverage.items()}
    wlog.info(f"Worker {worker_id} done: {sum(r.get('count', 0) for r in worker_results.values())} scenes")
    return worker_results, worker_scenes, coverage_serializable


def generate_training_scenes_for_task(
    task_name,
    config,
    output_base,
    scenes_per_subtype=10,
    base_seed=42,
    logger=None,
    subtypes_filter=None,
    num_workers=1,
):
    """
    Generate training error scenes for one task.

    Algorithm per subtype:
      1. Build virtual timeline by merging injectable time segments
      2. Uniformly sample N points from the timeline
      3. For each point, try injection up to 10 times; fallback to random alt
      4. After all subtypes, enforce per-object coverage (hard constraint)

    Returns:
        dict with keys: total_scenes, subtypes, coverage, coverage_gaps
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # ─── Load task info ───
    task_info = load_task_registry(task_name)
    dataset_path = task_info.get('dataset_path', config['paths']['dataset_path'])
    task_config_path = task_info.get('task_config', config['paths']['task_config'])

    with open(str(PROJECT_ROOT / task_config_path)) as f:
        task_config = yaml.safe_load(f)

    # ─── Load clean trajectories ───
    traj_dir = PROJECT_ROOT / f"error_benchmark/outputs/v5/clean_trajectories/{task_name}"
    if not traj_dir.exists():
        logger.warning(f"No clean trajectories for {task_name}, skipping")
        return None

    trajectories = CleanTrajectoryCollector.load(str(traj_dir))
    traj_map = {t.trajectory_id: t for t in trajectories}
    logger.info(f"Loaded {len(trajectories)} trajectories for {task_name}")

    # ─── Load opportunities ───
    opp_path = PROJECT_ROOT / f"error_benchmark/outputs/v5/opportunity_maps/{task_name}/opportunities.jsonl"
    if not opp_path.exists():
        logger.warning(f"No opportunity map for {task_name}, skipping")
        return None

    all_opportunities = OpportunityScanner.load(str(opp_path))
    logger.info(f"Loaded {len(all_opportunities)} opportunities for {task_name}")

    # Group by subtype
    opp_by_subtype = defaultdict(list)
    for opp in all_opportunities:
        opp_by_subtype[opp.subtype_id()].append(opp)

    available_subtypes = sorted(opp_by_subtype.keys())
    if subtypes_filter:
        available_subtypes = [s for s in available_subtypes if s in subtypes_filter]
        if not available_subtypes:
            logger.warning(f"No matching subtypes for filter {subtypes_filter} in {task_name}")
            return None
    logger.info(f"Available subtypes for {task_name}: {len(available_subtypes)}")

    skills_config = config.get('error_skills', {})

    # ─── Task objects (for per-object coverage constraint) ───
    task_objects = [obj['name'] for obj in task_config.get('objects', [])]

    # ─── Output directory ───
    task_output = output_base / task_name / "scenes"
    task_output.mkdir(parents=True, exist_ok=True)

    # ─── Generate scenes (parallel or serial) ───
    results = {}
    total_scenes = 0
    all_scene_metas = []
    coverage = defaultdict(int)
    skill_map = {}
    main_rng = np.random.RandomState(base_seed)

    effective_workers = min(num_workers, len(available_subtypes))
    _parallel_done = False

    if effective_workers > 1:
        logger.info(f"Parallel mode: {effective_workers} workers for {len(available_subtypes)} subtypes")

        batches = [[] for _ in range(effective_workers)]
        for i, sid in enumerate(available_subtypes):
            batches[i % effective_workers].append(sid)

        worker_args = [
            (wid, batches[wid], task_name, config, dataset_path,
             task_config, str(traj_dir), task_output, skills_config,
             scenes_per_subtype, base_seed, str(opp_path))
            for wid in range(effective_workers)
        ]

        with _MP_CTX.Pool(effective_workers) as pool:
            pool_results = pool.starmap(_worker_process, worker_args)

        for w_results, w_scenes, w_coverage_ser in pool_results:
            results.update(w_results)
            all_scene_metas.extend(w_scenes)
            for key_str, cnt in w_coverage_ser.items():
                obj, err = key_str.split('|', 1)
                coverage[(obj, err)] += cnt
            total_scenes += sum(r.get('count', 0) for r in w_results.values())

        logger.info(f"Parallel generation done: {total_scenes} scenes from {effective_workers} workers")

        for sid in available_subtypes:
            opps = opp_by_subtype.get(sid, [])
            if opps:
                err = opps[0].error_name
                if err not in skill_map:
                    try:
                        skill_map[err] = get_skill(err, skills_config.get(err, {}))
                    except KeyError:
                        pass
        _parallel_done = True

    # ─── Create environment (needed for serial mode + coverage repair) ───
    env = create_env(task_config, dataset_path, enable_camera=False)
    from error_benchmark.framework.env_wrapper import EnvWrapper
    env_wrapper = EnvWrapper(env, task_config)
    injection_config = config.get('injection', {})
    engine = InjectionEngine(
        env=env_wrapper, task_config=task_config,
        render_window=injection_config.get('context_window', 50),
        post_injection_steps=injection_config.get('post_injection_steps', 20),
    )

    # ─── Serial mode (original) ───
    if not _parallel_done:
        pbar = tqdm(available_subtypes, desc=f"Subtypes ({task_name})")
        for subtype_id in pbar:
            opps = opp_by_subtype[subtype_id]
            error_name = opps[0].error_name
            degree = opps[0].degree

            try:
                skill_params = skills_config.get(error_name, {})
                skill = get_skill(error_name, skill_params)
                skill_map[error_name] = skill
            except KeyError:
                logger.warning(f"Unknown skill {error_name}, skipping {subtype_id}")
                continue

            timeline = QuotaScheduler.build_virtual_timeline(opps, merge_gap=10)
            if not timeline:
                logger.warning(f"Empty timeline for {subtype_id}")
                results[subtype_id] = {'count': 0, 'total_opps': len(opps)}
                continue

            sampled = QuotaScheduler.sample_from_timeline(timeline, n=scenes_per_subtype)

            logger.info(
                f"  {subtype_id}: {len(opps)} opps → timeline {len(timeline)} → "
                f"sampled {len(sampled)} points"
            )

            scenes_done = 0
            failure_reasons = {}

            for i, opp in enumerate(sampled):
                trajectory = traj_map.get(opp.trajectory_id)
                if trajectory is None:
                    continue

                fraction = i / max(scenes_per_subtype - 1, 1)
                scene = _try_inject(
                    engine, trajectory, opp, skill, base_seed, i,
                    fraction=fraction)

                if scene is None:
                    reason = getattr(engine, 'last_failure_reason', 'unknown')
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

                    alt_indices = main_rng.choice(
                        len(timeline), size=min(20, len(timeline)), replace=False)
                    for alt_idx in alt_indices:
                        alt_opp = timeline[alt_idx]
                        alt_traj = traj_map.get(alt_opp.trajectory_id)
                        if alt_traj is None:
                            continue
                        scene = _try_inject(
                            engine, alt_traj, alt_opp, skill,
                            base_seed + 50000, alt_idx, max_retries=3,
                            fraction=fraction)
                        if scene is not None:
                            break

                if scene is None:
                    logger.debug(f"  {subtype_id} point {i}: all retries failed")
                    continue

                _record_generation_provenance(
                    scene,
                    mode="controlled_magnitude",
                    point_index=i,
                    fraction=fraction,
                    scenes_per_subtype=scenes_per_subtype,
                    base_seed=base_seed,
                    config=config,
                )

                target_obj = scene.error_spec.target_object or ''
                scene.labels['target_object'] = target_obj

                seq = scenes_done
                hash_input = (
                    f"train_{task_name}_{subtype_id}_{seq:03d}_"
                    f"{scene.clean_trajectory_ref}_{scene.injection_step}"
                )
                hash8 = hashlib.sha1(hash_input.encode()).hexdigest()[:8]
                scene.scene_id = f"v5_train_{subtype_id}_{seq:03d}_{hash8}"
                scene.labels['training_seq'] = seq

                scene_json_path = task_output / f"{scene.scene_id}.json"
                with open(scene_json_path, 'w') as f:
                    json.dump(scene.to_dict(), f, indent=2, cls=NumpyEncoder)

                if scene._pre_state is not None:
                    npz_path = task_output / f"{scene.scene_id}.npz"
                    save_dict = {'sim_state': scene._pre_state}
                    if scene._post_state is not None:
                        save_dict['post_sim_state'] = scene._post_state
                    np.savez_compressed(npz_path, **save_dict)

                scenes_done += 1
                total_scenes += 1
                coverage[(target_obj, error_name)] += 1
                all_scene_metas.append({
                    'scene_id': scene.scene_id,
                    'subtype_id': subtype_id,
                    'error_name': error_name,
                    'degree': degree,
                    'seq': seq,
                    'injection_frame': scene.injection_step,
                    'target_object': target_obj,
                    'task_phase': scene.labels.get('task_phase', ''),
                })

            results[subtype_id] = {
                'count': scenes_done,
                'total_opps': len(opps),
                'timeline_length': len(timeline),
                'failure_reasons': dict(failure_reasons) if failure_reasons else {},
            }

            if scenes_done < scenes_per_subtype:
                logger.warning(
                    f"  {subtype_id}: {scenes_done}/{scenes_per_subtype} scenes"
                )

            pbar.set_postfix(scenes=total_scenes, subtypes=len(results))

    # ─── Per-object coverage repair ───
    all_error_names = set()
    for sid in available_subtypes:
        for d in ['D0', 'D1']:
            if sid.endswith(f'_{d}'):
                all_error_names.add(sid[:-(len(d) + 1)])
                break

    coverage_gaps = []
    if task_objects:
        logger.info(f"\n--- Coverage check: {len(task_objects)} objects × "
                     f"{len(all_error_names)} error types ---")

        for obj in task_objects:
            for err_name in sorted(all_error_names):
                if coverage[(obj, err_name)] > 0:
                    continue

                # Try to generate a scene targeting this object
                # Use opportunities for this error (any degree)
                err_opps = [o for o in all_opportunities if o.error_name == err_name]
                if not err_opps:
                    coverage_gaps.append({
                        'target_object': obj, 'error_name': err_name,
                        'reason': 'no_opportunities',
                    })
                    continue

                skill = skill_map.get(err_name)
                if skill is None:
                    coverage_gaps.append({
                        'target_object': obj, 'error_name': err_name,
                        'reason': 'skill_not_loaded',
                    })
                    continue

                repair_rng = np.random.RandomState(
                    (base_seed + hash((obj, err_name))) % (2**31))
                repair_opps = list(err_opps)
                repair_rng.shuffle(repair_opps)

                repaired = False
                for opp in repair_opps[:30]:
                    traj = traj_map.get(opp.trajectory_id)
                    if traj is None:
                        continue
                    scene = _try_inject(
                        engine, traj, opp, skill,
                        base_seed + 90000, hash((obj, err_name)) % 10000,
                        max_retries=5)
                    if scene is None:
                        continue

                    actual_obj = scene.error_spec.target_object or ''
                    if actual_obj != obj:
                        continue  # Wrong object, try next

                    # Found a scene for this (object, error) pair
                    _record_generation_provenance(
                        scene,
                        mode="coverage_repair_uncontrolled",
                        point_index=None,
                        fraction=None,
                        scenes_per_subtype=scenes_per_subtype,
                        base_seed=base_seed,
                        config=config,
                    )
                    scene.labels['target_object'] = actual_obj
                    seq = results.get(opp.subtype_id(), {}).get('count', 0)
                    hash_input = (
                        f"train_{task_name}_{opp.subtype_id()}_repair_{obj}_{err_name}"
                    )
                    hash8 = hashlib.sha1(hash_input.encode()).hexdigest()[:8]
                    scene.scene_id = f"v5_train_{opp.subtype_id()}_repair_{hash8}"
                    scene.labels['training_seq'] = seq
                    scene.labels['coverage_repair'] = True

                    scene_json_path = task_output / f"{scene.scene_id}.json"
                    with open(scene_json_path, 'w') as f:
                        json.dump(scene.to_dict(), f, indent=2, cls=NumpyEncoder)
                    if scene._pre_state is not None:
                        npz_path = task_output / f"{scene.scene_id}.npz"
                        save_dict = {'sim_state': scene._pre_state}
                        if scene._post_state is not None:
                            save_dict['post_sim_state'] = scene._post_state
                        np.savez_compressed(npz_path, **save_dict)

                    coverage[(obj, err_name)] += 1
                    total_scenes += 1
                    all_scene_metas.append({
                        'scene_id': scene.scene_id,
                        'subtype_id': opp.subtype_id(),
                        'error_name': err_name,
                        'degree': opp.degree,
                        'seq': seq,
                        'injection_frame': scene.injection_step,
                        'target_object': actual_obj,
                        'task_phase': scene.labels.get('task_phase', ''),
                        'coverage_repair': True,
                    })
                    repaired = True
                    logger.info(f"  REPAIR: ({obj}, {err_name}) covered via {opp.subtype_id()}")
                    break

                if not repaired:
                    coverage_gaps.append({
                        'target_object': obj, 'error_name': err_name,
                        'reason': 'injection_failed_or_wrong_object',
                    })
                    logger.warning(f"  COVERAGE GAP: ({obj}, {err_name}) — no scene generated")

    # ─── Save meta.json ───
    coverage_summary = {
        f"{obj}_{err}": cnt
        for (obj, err), cnt in sorted(coverage.items())
    }

    meta = {
        'version': 'v5.0_training_v2',
        'created': datetime.now().isoformat(),
        'task': task_name,
        'total_scenes': total_scenes,
        'scenes_per_subtype': scenes_per_subtype,
        'available_subtypes': len(available_subtypes),
        'generated_subtypes': len(results),
        'subtypes': results,
        'scenes': all_scene_metas,
        'coverage': coverage_summary,
        'coverage_gaps': coverage_gaps,
        'task_objects': task_objects,
    }

    if subtypes_filter:
        filter_tag = "_".join(sorted(subtypes_filter))[:60]
        meta_path = output_base / task_name / f"meta_partial_{filter_tag}.json"
    else:
        meta_path = output_base / task_name / "meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, cls=NumpyEncoder)

    logger.info(
        f"[{task_name}] Generated {total_scenes} training scenes "
        f"across {len(results)} subtypes"
    )
    if coverage_gaps:
        logger.warning(f"  {len(coverage_gaps)} coverage gaps (see meta.json)")

    # Close env
    try:
        env.close()
    except Exception:
        pass

    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Generate training error scenes with controlled magnitude distribution"
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Single task name (default: all 6 tasks)"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="error_benchmark/outputs/v5_training",
        help="Output directory (default: error_benchmark/outputs/v5_training)"
    )
    parser.add_argument(
        "--scenes_per_subtype", type=int, default=10,
        help="Number of scenes per subtype (default: 10)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)"
    )
    parser.add_argument(
        "--config", type=str,
        default="error_benchmark/configs/benchmark_v5.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "--subtypes", type=str, default=None,
        help="Comma-separated subtype IDs to process (for parallel execution)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=os.cpu_count() or 1,
        help="Number of parallel workers (default: CPU cores)"
    )
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # Load config
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_base = PROJECT_ROOT / args.output_dir
    output_base.mkdir(parents=True, exist_ok=True)

    tasks = [args.task] if args.task else ALL_TASKS
    subtypes_filter = set(args.subtypes.split(',')) if args.subtypes else None

    logger.info("=" * 60)
    logger.info("Training Scene Generation")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Scenes per subtype: {args.scenes_per_subtype}")
    logger.info(f"Workers: {args.num_workers}")
    if subtypes_filter:
        logger.info(f"Subtypes filter: {subtypes_filter}")
    logger.info(f"Output: {output_base}")
    logger.info("=" * 60)

    all_results = {}
    grand_total = 0

    for task_name in tasks:
        logger.info(f"\n{'─' * 40}")
        logger.info(f"Processing task: {task_name}")
        logger.info(f"{'─' * 40}")

        meta = generate_training_scenes_for_task(
            task_name=task_name,
            config=config,
            output_base=output_base,
            scenes_per_subtype=args.scenes_per_subtype,
            base_seed=args.seed,
            logger=logger,
            subtypes_filter=subtypes_filter,
            num_workers=args.num_workers,
        )

        if meta is not None:
            all_results[task_name] = meta
            grand_total += meta['total_scenes']

    # ─── Summary ───
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SCENE GENERATION SUMMARY")
    logger.info("=" * 60)
    for task_name, meta in all_results.items():
        logger.info(
            f"  {task_name}: {meta['total_scenes']} scenes, "
            f"{meta['generated_subtypes']}/{meta['available_subtypes']} subtypes"
        )
    logger.info(f"  TOTAL: {grand_total} scenes")
    logger.info("=" * 60)

    # Save global summary (only for full runs, not parallel subtypes)
    if subtypes_filter:
        logger.info(f"Skipping summary.json (subtypes filter active)")
        return

    summary = {
        'version': 'v5.0_training',
        'created': datetime.now().isoformat(),
        'scenes_per_subtype': args.scenes_per_subtype,
        'seed': args.seed,
        'tasks': {
            task: {
                'total_scenes': m['total_scenes'],
                'generated_subtypes': m['generated_subtypes'],
                'available_subtypes': m['available_subtypes'],
            }
            for task, m in all_results.items()
        },
        'grand_total': grand_total,
    }

    summary_path = output_base / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
