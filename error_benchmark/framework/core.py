#!/usr/bin/env python
"""
Core Data Structures

Data structures used throughout the pipeline:

    ErrorSpec          -> Complete error specification (type, target, params, direction strategy)
    DetectionResult    -> Detector output (triggered + ErrorSpec candidate list)
    ValidationResult   -> Validator output (whether error is valid)
    FilterResult       -> Filter output (whether candidate passes pre-check)
    RejectedCandidate  -> Detailed record of rejected candidates (for tuning analysis)
    ErrorScene         -> Complete error scene data (sim state + fingerprint + labels)
    PostRolloutStats   -> Post-injection trajectory statistics (for Validator judgment)
    EpisodeSummary     -> Single evaluation episode summary
    EvaluationResult   -> Multi-episode aggregated evaluation result
    DatabaseMeta       -> Scene database metadata

v4.0 key improvements:
- ErrorSpec adds direction_strategy (explicit force direction strategy)
- Added RejectedCandidate (records all rejected candidates, ~85% rejection rate is normal)
- ErrorScene adds rng_state (fully deterministic replay)

v5.0 key improvements:
- ErrorSpec adds error_name/degree/source_frame/source_trajectory fields (Error Skill injection)
- ErrorScene adds clean_trajectory_ref/injection_step/render_window fields (Context Replay)
- Added InjectionOpportunity (offline scan results)
- Added CleanTrajectory (clean trajectory data)
- Added QuotaStatus (quota status tracking)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import hashlib
import json


# ─── Phase group mapping ────────────────────────────────────

PHASE_GROUP_MAP = {
    'pre_reach': 'approach',
    'reach': 'approach',
    'pre_grasp': 'approach',
    'grasp': 'grasp',
    'lift': 'transfer',
    'transport': 'transfer',
    'place': 'place',
    'post_place': 'place',
}

PHASE_GROUPS = ['approach', 'grasp', 'transfer', 'place']


def map_phase_to_group(phase: str) -> str:
    """Map fine-grained task phase to one of 4 phase groups."""
    return PHASE_GROUP_MAP.get(phase, 'approach')


@dataclass
class ErrorSpec:
    """
    Error specification - describes complete parameters of an error

    v4 updates:
    - Added direction_strategy for explicit force direction strategy
    - All serialization methods support JSON
    """
    type: str  # "impulse" | "pose_perturb" | "friction_scale" | "gripper_bias"
    family: str  # "physics" | "execution" | "perception"
    target: Dict  # {"body": "cube_main", "geom": None} or {"eef": True}
    params: Dict  # type-specific params
    apply: Dict  # {"mode": "at_step", "t": int} or {"mode": "window", "t0": int, "len": int}
    seed: int
    direction_strategy: str = "random_unit"  # [v4] "random_unit" | "gravity_aligned" | "lateral_to_motion"
    # [v5] Error Skill fields
    error_name: str = ""         # v5 error skill name (e.g., "grasp_misalignment")
    degree: str = ""             # v5 degree: "D0" | "D1"
    source_frame: int = -1       # v5 frame index in clean trajectory where injection occurs
    source_trajectory: str = ""  # v5 identifier of the clean trajectory used
    target_object: str = ""      # v5 target object name at injection frame

    def to_dict(self) -> dict:
        """Convert to dict (supports JSON serialization)"""
        d = asdict(self)
        # Handle numpy arrays
        d['params'] = self._serialize_numpy(self.params)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'ErrorSpec':
        """Restore from dict"""
        params = d.get('params', {})
        # Restore numpy arrays
        params = cls._deserialize_numpy(params)
        return cls(
            type=d['type'],
            family=d['family'],
            target=d['target'],
            params=params,
            apply=d['apply'],
            seed=int(d.get('seed', 0)),
            direction_strategy=d.get('direction_strategy', 'random_unit'),
            error_name=d.get('error_name', ''),
            degree=d.get('degree', ''),
            source_frame=d.get('source_frame', -1),
            source_trajectory=d.get('source_trajectory', ''),
            target_object=d.get('target_object', ''),
        )

    def deterministic_hash(self) -> str:
        """
        Return deterministic hash of this spec (for dedup)
        Identical error_specs should produce the same hash
        """
        # Create a normalized string representation
        norm_str = json.dumps({
            'type': self.type,
            'family': self.family,
            'target': sorted(self.target.items()),
            'params': sorted(self._flatten_dict(self.params).items()),
            'apply': sorted(self.apply.items()),
            'seed': self.seed,
            'direction_strategy': self.direction_strategy,
        }, sort_keys=True)

        return hashlib.sha1(norm_str.encode()).hexdigest()

    @staticmethod
    def _serialize_numpy(obj: Any) -> Any:
        """Recursively serialize numpy arrays to lists"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: ErrorSpec._serialize_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ErrorSpec._serialize_numpy(v) for v in obj]
        return obj

    @staticmethod
    def _deserialize_numpy(obj: Any) -> Any:
        """Recursively restore lists to numpy arrays (heuristic)"""
        if isinstance(obj, dict):
            return {k: ErrorSpec._deserialize_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Try to determine if it should be a numpy array
            if len(obj) > 0 and all(isinstance(x, (int, float)) for x in obj):
                return np.array(obj)
            return [ErrorSpec._deserialize_numpy(v) for v in obj]
        return obj

    @staticmethod
    def _flatten_dict(d: dict, parent_key='', sep='.') -> dict:
        """Flatten nested dict"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(ErrorSpec._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, str(v)))
        return dict(items)


@dataclass
class DetectionResult:
    """
    Detection result - detector output

    v4 updates:
    - Added trigger_reason field for logging
    """
    triggered: bool
    detector: str
    score: float
    info: Dict
    proposed_error_specs: List[ErrorSpec] = field(default_factory=list)
    trigger_reason: str = ""  # [v4] Trigger reason description

    def to_dict(self) -> dict:
        d = asdict(self)
        d['proposed_error_specs'] = [spec.to_dict() for spec in self.proposed_error_specs]
        return d


@dataclass
class ValidationResult:
    """
    Validation result - validator output
    """
    ok: bool
    validator: str
    metrics: Dict
    reason: str = ""


@dataclass
class FilterResult:
    """
    Filter result - filter output
    """
    passed: bool
    reason: str
    details: Dict = field(default_factory=dict)


@dataclass
class RejectedCandidate:
    """
    [v4 new] Records discarded candidates for tuning analysis.
    """
    timestamp: str
    demo_key: str
    trigger_step: int
    error_spec: ErrorSpec
    rejection_stage: str  # "filter" | "validation" | "stability"
    rejection_reason: str
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['error_spec'] = self.error_spec.to_dict()
        d['details'] = ErrorSpec._serialize_numpy(self.details)
        return d


@dataclass
class ErrorScene:
    """
    Error scene - complete error scene data

    v4 updates:
    - Added rng_state field
    - Added version field
    v5 updates:
    - Added _post_state, post_injection fields
    - NPZ saves both pre + post states
    """
    version: str = "v5.0"
    scene_id: str = ""
    dataset: Dict = field(default_factory=dict)
    env_fingerprint: Dict = field(default_factory=dict)
    replay: Dict = field(default_factory=dict)
    error_spec: Optional[ErrorSpec] = None
    labels: Dict = field(default_factory=dict)
    detected_by: Dict = field(default_factory=dict)
    validated_by: Dict = field(default_factory=dict)
    dedup: Dict = field(default_factory=dict)
    rng_state: Dict = field(default_factory=dict)  # [v4] numpy + env rng
    # [v5] Context-replay fields
    clean_trajectory_ref: str = ""       # v5 reference to clean trajectory used
    injection_step: int = -1             # v5 step in clean trajectory where error was injected
    render_window: int = 0               # v5 number of frames rendered before injection (for video)
    _pre_state: Optional[np.ndarray] = field(default=None, repr=False)  # Temp storage: pre-injection sim state (for demo replay)
    _post_state: Optional[np.ndarray] = field(default=None, repr=False)  # Temp storage: stable post-injection sim state (for direct training/validation loading)
    post_injection: Dict = field(default_factory=dict)  # Post-injection object poses + id_eligible flags

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.error_spec is not None:
            d['error_spec'] = self.error_spec.to_dict()
        # Serialize numpy
        d['rng_state'] = ErrorSpec._serialize_numpy(self.rng_state)
        d['env_fingerprint'] = ErrorSpec._serialize_numpy(self.env_fingerprint)
        # Exclude _pre_state/_post_state (not serialized to JSON, stored in NPZ)
        d.pop('_pre_state', None)
        d.pop('_post_state', None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'ErrorScene':
        error_spec = d.get('error_spec')
        if error_spec is not None:
            error_spec = ErrorSpec.from_dict(error_spec)

        return cls(
            version=d.get('version', 'v4.0'),
            scene_id=d.get('scene_id', ''),
            dataset=d.get('dataset', {}),
            env_fingerprint=d.get('env_fingerprint', {}),
            replay=d.get('replay', {}),
            error_spec=error_spec,
            labels=d.get('labels', {}),
            detected_by=d.get('detected_by', {}),
            validated_by=d.get('validated_by', {}),
            dedup=d.get('dedup', {}),
            rng_state=d.get('rng_state', {}),
            clean_trajectory_ref=d.get('clean_trajectory_ref', ''),
            injection_step=d.get('injection_step', -1),
            render_window=d.get('render_window', d.get('context_window', 0)),
            post_injection=d.get('post_injection', {}),
        )

    def get_npz_path(self) -> str:
        """Get the corresponding npz file path"""
        return self.replay.get('pre_state_npz', '')


@dataclass
class PostRolloutStats:
    """
    Post-injection rollout trajectory statistics
    Used by validator to determine whether error actually occurred
    """
    obj_z_trajectory: List[float] = field(default_factory=list)
    obj_pos_trajectory: List[np.ndarray] = field(default_factory=list)
    obj_quat_trajectory: List[np.ndarray] = field(default_factory=list)
    min_z: float = 0.0
    max_z: float = 0.0
    max_tip_angle: float = 0.0
    max_offset_xy: float = 0.0
    gripper_contact_frames: int = 0
    time_to_min_z: int = 0
    total_steps: int = 0
    obj_contact_geoms: List[str] = field(default_factory=list)


@dataclass
class EpisodeSummary:
    """
    Single evaluation episode summary
    """
    scene_id: str
    policy_name: str
    seed: int
    success: bool
    steps_taken: int
    recovery_progress: float = 0.0  # [v4] Recovery Progress
    task_completion_final: Dict = field(default_factory=dict)
    demo_len: int = 0
    trigger_step: int = 0
    total_reward: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d['task_completion_final'] = ErrorSpec._serialize_numpy(self.task_completion_final)
        return d


@dataclass
class EvaluationResult:
    """
    Evaluation result - model overall performance across multiple scenes

    Primary evaluation metric: Success Rate (SR) + Bootstrap 95% CI
    """
    policy_name: str
    num_scenes: int = 0
    num_success: int = 0
    success_rate: float = 0.0
    confidence_intervals: Dict = field(default_factory=dict)
    episodes: List[EpisodeSummary] = field(default_factory=list)

    def compute_metrics(self, n_bootstrap: int = 10000, ci_level: float = 0.95):
        """Compute Success Rate + Bootstrap 95% CI from episodes.

        Args:
            n_bootstrap: Bootstrap resampling count (default 10000)
            ci_level: Confidence level (default 0.95)
        """
        if not self.episodes:
            return

        self.num_scenes = len(self.episodes)
        self.num_success = sum(1 for e in self.episodes if e.success)
        self.success_rate = self.num_success / self.num_scenes if self.num_scenes > 0 else 0.0

        # Bootstrap 95% CI for Success Rate
        if self.num_scenes >= 2:
            successes = np.array([1.0 if e.success else 0.0 for e in self.episodes])
            rng = np.random.RandomState(42)
            boot_srs = np.empty(n_bootstrap)
            for i in range(n_bootstrap):
                sample = rng.choice(successes, size=len(successes), replace=True)
                boot_srs[i] = sample.mean()
            alpha = 1.0 - ci_level
            lo = float(np.percentile(boot_srs, 100 * alpha / 2))
            hi = float(np.percentile(boot_srs, 100 * (1 - alpha / 2)))
            self.confidence_intervals = {
                'success_rate': {'lower': lo, 'upper': hi, 'level': ci_level},
            }
        else:
            self.confidence_intervals = {}


# Scene database meta structure
@dataclass
class DatabaseMeta:
    """
    Error scene database meta.json structure
    """
    version: str = "v4.0"
    created: str = ""
    total_scenes: int = 0
    scenes: List[Dict] = field(default_factory=list)
    statistics: Dict = field(default_factory=dict)
    blacklist: List[str] = field(default_factory=list)
    dedup_index: Dict = field(default_factory=dict)
    split_config: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now().isoformat()

    @staticmethod
    def _count_by_keys(items, key_fns: Dict[str, callable]) -> Dict[str, Dict[str, int]]:
        """Count items by multiple key functions using Counter."""
        from collections import Counter
        counters = {name: Counter() for name in key_fns}
        for item in items:
            for name, fn in key_fns.items():
                counters[name][fn(item)] += 1
        return {name: dict(c) for name, c in counters.items()}

    def update_statistics(self, scenes: List[ErrorScene]):
        """Update statistics"""
        self.total_scenes = len(scenes)

        stats = self._count_by_keys(scenes, {
            'by_error_type': lambda s: s.labels.get('error_type', 'unknown'),
            'by_severity': lambda s: s.labels.get('severity', 'unknown'),
            'by_phase': lambda s: s.labels.get('task_phase', 'unknown'),
        })
        self.statistics = stats

        # Update scenes list (lightweight version)
        self.scenes = [
            {
                'scene_id': s.scene_id,
                'error_type': s.labels.get('error_type', ''),
                'severity': s.labels.get('severity', ''),
                'task_phase': s.labels.get('task_phase', ''),
                'split': s.labels.get('split', ''),
                'status': 'active',
            }
            for s in scenes
        ]

    def update_statistics_from_info(self, scene_infos: List[dict]):
        """Update statistics from scene info dicts"""
        self.statistics = self._count_by_keys(scene_infos, {
            'by_error_type': lambda s: s.get('error_type', 'unknown'),
            'by_severity': lambda s: s.get('severity', 'unknown'),
            'by_phase': lambda s: s.get('task_phase', 'unknown'),
        })

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization"""
        return asdict(self)


# ═══════════════════════════════════════════════════════════
# v5.0 New Data Structures
# ═══════════════════════════════════════════════════════════

@dataclass
class InteractionSegment:
    """
    [v5] A segment of a trajectory where the robot interacts with a specific object.

    Each segment identifies which object is being manipulated, the task phase,
    and the frame range. This eliminates ambiguity in multi-object tasks.
    """
    target_object: str = ""       # Object being manipulated (e.g., "Milk")
    phase: str = ""               # reach | grasp | lift | transport | place
    start_step: int = 0           # Start frame (inclusive)
    end_step: int = 0             # End frame (exclusive)
    gripper_grasping: bool = False  # Whether gripper is holding the object
    other_objects: List[str] = field(default_factory=list)  # Other objects in scene

    def to_dict(self) -> dict:
        return {
            'target_object': self.target_object,
            'phase': self.phase,
            'start_step': self.start_step,
            'end_step': self.end_step,
            'gripper_grasping': self.gripper_grasping,
            'other_objects': self.other_objects,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'InteractionSegment':
        return cls(
            target_object=d.get('target_object', ''),
            phase=d.get('phase', ''),
            start_step=d.get('start_step', 0),
            end_step=d.get('end_step', 0),
            gripper_grasping=d.get('gripper_grasping', False),
            other_objects=d.get('other_objects', []),
        )

    def contains_frame(self, frame_idx: int) -> bool:
        """Check if a frame index falls within this segment."""
        return self.start_step <= frame_idx < self.end_step


@dataclass
class CleanTrajectory:
    """
    [v5] A clean (successful) trajectory for offline scanning.

    Can come from human demos (HDF5) or successful VLA/BC policy rollouts.
    """
    trajectory_id: str = ""
    source: str = ""             # "demo" | "vla_pi0" | "vla_pi05" | "bc_rnn"
    task_name: str = ""
    dataset_path: str = ""
    demo_key: str = ""           # For demo source: "demo_0", "demo_1", etc.
    num_steps: int = 0
    actions: Optional[np.ndarray] = field(default=None, repr=False)  # (T, action_dim)
    states: Optional[np.ndarray] = field(default=None, repr=False)   # (T+1, state_dim)
    observations: Optional[List[Dict]] = field(default=None, repr=False)  # Per-step obs
    phase_labels: Optional[List[str]] = field(default=None, repr=False)  # Per-step phase
    interaction_segments: Optional[List[InteractionSegment]] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Serialize (excluding large arrays)."""
        d = {
            'trajectory_id': self.trajectory_id,
            'source': self.source,
            'task_name': self.task_name,
            'dataset_path': self.dataset_path,
            'demo_key': self.demo_key,
            'num_steps': self.num_steps,
        }
        if self.interaction_segments is not None:
            d['interaction_segments'] = [s.to_dict() for s in self.interaction_segments]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'CleanTrajectory':
        segments = None
        if 'interaction_segments' in d:
            segments = [InteractionSegment.from_dict(s) for s in d['interaction_segments']]
        return cls(
            trajectory_id=d.get('trajectory_id', ''),
            source=d.get('source', ''),
            task_name=d.get('task_name', ''),
            dataset_path=d.get('dataset_path', ''),
            demo_key=d.get('demo_key', ''),
            num_steps=d.get('num_steps', 0),
            interaction_segments=segments,
        )

    def get_segment_at_frame(self, frame_idx: int) -> Optional['InteractionSegment']:
        """Get the interaction segment containing a given frame index."""
        if self.interaction_segments is None:
            return None
        for seg in self.interaction_segments:
            if seg.contains_frame(frame_idx):
                return seg
        return None


@dataclass
class InjectionOpportunity:
    """
    [v5] A single injection opportunity found during offline scanning.

    Represents: at frame N of trajectory T, error skill S with degree D can be injected.
    """
    trajectory_id: str = ""
    frame_index: int = 0
    error_name: str = ""        # Error skill name
    degree: str = ""            # "D0" | "D1"
    task_phase: str = ""        # Phase at this frame
    confidence: float = 1.0     # How suitable this frame is for injection
    metadata: Dict = field(default_factory=dict)  # Skill-specific scan info

    def to_dict(self) -> dict:
        return {
            'trajectory_id': self.trajectory_id,
            'frame_index': self.frame_index,
            'error_name': self.error_name,
            'degree': self.degree,
            'task_phase': self.task_phase,
            'confidence': self.confidence,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'InjectionOpportunity':
        valid_fields = {f.name for f in __import__('dataclasses').fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})

    def subtype_id(self) -> str:
        """Return subtype identifier like 'drop_D0'."""
        return f"{self.error_name}_{self.degree}"


@dataclass
class QuotaStatus:
    """
    [v5] Tracks quota fulfillment for balanced generation.

    Target: quota_per_subtype episodes per (task, error_name, degree).
    """
    task_name: str = ""
    quota_per_subtype: int = 100
    counts: Dict[str, int] = field(default_factory=dict)  # subtype_id -> count

    def get_count(self, error_name: str, degree: str) -> int:
        key = f"{error_name}_{degree}"
        return self.counts.get(key, 0)

    def increment(self, error_name: str, degree: str, n: int = 1):
        key = f"{error_name}_{degree}"
        self.counts[key] = self.counts.get(key, 0) + n

    def remaining(self, error_name: str, degree: str) -> int:
        return max(0, self.quota_per_subtype - self.get_count(error_name, degree))

    def is_fulfilled(self, error_name: str, degree: str) -> bool:
        return self.get_count(error_name, degree) >= self.quota_per_subtype

    def total_remaining(self) -> int:
        from error_benchmark.framework.error_taxonomy_v5 import get_all_subtypes
        total = 0
        for name, degree in get_all_subtypes():
            total += self.remaining(name, degree)
        return total

    def summary(self) -> Dict[str, Any]:
        from error_benchmark.framework.error_taxonomy_v5 import get_all_subtypes
        fulfilled = 0
        total_subtypes = 0
        total_remaining = 0
        for name, degree in get_all_subtypes():
            total_subtypes += 1
            rem = self.remaining(name, degree)
            total_remaining += rem
            if rem == 0:
                fulfilled += 1
        return {
            'task_name': self.task_name,
            'quota_per_subtype': self.quota_per_subtype,
            'fulfilled': fulfilled,
            'total_subtypes': total_subtypes,
            'total_scenes': sum(self.counts.values()),
            'total_remaining': total_remaining,
        }

    def to_dict(self) -> dict:
        return {
            'task_name': self.task_name,
            'quota_per_subtype': self.quota_per_subtype,
            'counts': self.counts.copy(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'QuotaStatus':
        return cls(
            task_name=d.get('task_name', ''),
            quota_per_subtype=d.get('quota_per_subtype', 100),
            counts=d.get('counts', {}),
        )


@dataclass
class ValidationResult_v5:
    """
    [v5] Extended validation result for Error Skills.
    Includes degree-specific sub-validation.
    """
    ok: bool = False
    error_name: str = ""
    degree: str = ""
    metrics: Dict = field(default_factory=dict)
    reason: str = ""
    post_rollout_stats: Optional[PostRolloutStats] = None

    def to_dict(self) -> dict:
        d = {
            'ok': self.ok,
            'error_name': self.error_name,
            'degree': self.degree,
            'metrics': ErrorSpec._serialize_numpy(self.metrics),
            'reason': self.reason,
        }
        return d
