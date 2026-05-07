"""
Error Framework - Error Recovery Benchmark Core Framework (v5.0)

v5.0 architecture overview (Error Skill Paradigm):
    Clean Trajectories
        -> OpportunityScanner (offline scan: every frame x every skill)
        -> QuotaScheduler (quota scheduling 100/subtype)
        -> InjectionEngine (direct injection + validation)
        -> ErrorSceneDatabase (storage, classification, query)
        -> RecoveryDataCollector (multi-policy evaluation)
        → MetricsComputer（SR/SPL/RP + Bootstrap CI）

v5.0 modules:
    error_taxonomy_v5.py  -- 12 Error Skills × D0/D1 = 24 subtypes
    error_skills/         -- 12 Error Skills (self-contained detect+inject+validate)
    clean_trajectory_collector.py -- Clean trajectory collection (demo + policy rollout)
    opportunity_scanner.py-- Offline injection opportunity scanning
    quota_scheduler.py    -- Quota scheduling
    context_replay.py     -- Direct injection engine (InjectionEngine)
    utils/                -- Extracted reusable utilities (physics, math, rollout)

Shared modules (v4/v5 common):
    core.py, env_wrapper.py, policy_adapter.py, logger_setup.py

v4 code archived to archive/v4/
"""

from .core import (
    ErrorSpec,
    ErrorScene,
    DetectionResult,
    ValidationResult,
    FilterResult,
    RejectedCandidate,
    EpisodeSummary,
    EvaluationResult,
    DatabaseMeta,
    PostRolloutStats,
    # v5 data structures
    CleanTrajectory,
    InjectionOpportunity,
    QuotaStatus,
    ValidationResult_v5,
)

from .env_wrapper import (
    EnvWrapper,
    EnvironmentMismatchError,
    ContactInfo,
)

from .policy_adapter import (
    PolicyResult,
    BasePolicy,
    PolicyAdapter,
    RandomPolicyAdapter,
    RobomimicPolicyAdapter,
    create_policy_adapter,
)

from .logger_setup import (
    setup_logging,
)

# ─── v5.0 imports ─────────────────────────────────────────

from .error_taxonomy_v5 import (
    ErrorDegree,
    ErrorSkillName,
    ERROR_SKILL_DEFS,
    get_all_subtypes,
    get_subtype_id,
    get_skill_def,
    is_valid_subtype,
    get_valid_degrees,
    get_valid_phases,
)

from .error_skills import (
    BaseErrorSkill,
    SkillConfig,
    SKILL_REGISTRY,
    get_skill,
    get_all_skills,
)

from .clean_trajectory_collector import (
    CleanTrajectoryCollector,
)

from .opportunity_scanner import (
    OpportunityScanner,
)

from .context_replay import (
    InjectionEngine,
)

from .quota_scheduler import (
    QuotaScheduler,
)

__all__ = [
    # Core (shared v4/v5 data structures)
    "ErrorSpec",
    "ErrorScene",
    "DetectionResult",
    "ValidationResult",
    "FilterResult",
    "RejectedCandidate",
    "EpisodeSummary",
    "EvaluationResult",
    "DatabaseMeta",
    "PostRolloutStats",

    # EnvWrapper
    "EnvWrapper",
    "EnvironmentMismatchError",
    "ContactInfo",

    # PolicyAdapter
    "PolicyResult",
    "BasePolicy",
    "PolicyAdapter",
    "RandomPolicyAdapter",
    "RobomimicPolicyAdapter",
    "create_policy_adapter",

    # Logger
    "setup_logging",

    # ─── v5.0 ─────────────────────────────────────────
    # Taxonomy v5
    "ErrorDegree",
    "ErrorSkillName",
    "ERROR_SKILL_DEFS",
    "get_all_subtypes",
    "get_subtype_id",
    "get_skill_def",
    "is_valid_subtype",
    "get_valid_degrees",
    "get_valid_phases",
    # Error Skills
    "BaseErrorSkill",
    "SkillConfig",
    "SKILL_REGISTRY",
    "get_skill",
    "get_all_skills",
    # v5 Core data structures
    "CleanTrajectory",
    "InjectionOpportunity",
    "QuotaStatus",
    "ValidationResult_v5",
    # v5 Pipeline
    "CleanTrajectoryCollector",
    "OpportunityScanner",
    "InjectionEngine",
    "QuotaScheduler",
]

__version__ = '5.0.0'
