"""Spec tests for eval_metrics: Fisher exact test + RBG mapping.

Authoritative protocol (docs/项目全景总结.md, M16): SR point estimate +
Fisher exact (recovery SR > normal SR, p<0.05). No Bootstrap CI.
"""
import ast
from pathlib import Path

import pytest

from error_benchmark.scripts.training import eval_metrics as em

REPO = Path(__file__).resolve().parents[3]
RECOVERY_TYPES = REPO / "error_benchmark" / "framework" / "recovery_types.py"


# ─────────────────────── RBG mapping ───────────────────────

def _canonical_subtype_to_rbg() -> dict:
    """Extract SUBTYPE_TO_RBG from recovery_types.py WITHOUT importing it
    (framework/__init__ pulls heavy robosuite deps)."""
    tree = ast.parse(RECOVERY_TYPES.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "SUBTYPE_TO_RBG":
                    return ast.literal_eval(node.value)
    raise AssertionError("SUBTYPE_TO_RBG not found in recovery_types.py")


def test_rbg_copy_matches_canonical():
    """The copied map must stay byte-for-byte in sync with the canonical one."""
    assert em.SUBTYPE_TO_RBG == _canonical_subtype_to_rbg()


def test_rbg_has_24_subtypes_5_groups():
    assert len(em.SUBTYPE_TO_RBG) == 24
    assert set(em.SUBTYPE_TO_RBG.values()) == {"RBG_A", "RBG_B", "RBG_C", "RBG_D", "RBG_E"}


def test_rbg_of_known_and_unknown():
    assert em.rbg_of("grasp_misalignment_D0") == "RBG_A"
    assert em.rbg_of("wrong_object_D1") == "RBG_D"
    assert em.rbg_of("position_error_D0") == "RBG_E"
    assert em.rbg_of("not_a_subtype") is None


# ─────────────────────── Fisher exact ───────────────────────

def test_fisher_identical_groups_not_significant():
    r = em.fisher_exact_sr(10, 100, 10, 100)
    assert r["p_value"] > 0.5
    assert r["significant"] is False


def test_fisher_separated_groups_significant():
    # recovery 40/50 vs normal 0/50 -> strongly significant, recovery > normal
    r = em.fisher_exact_sr(40, 50, 0, 50)
    assert r["p_value"] < 0.05
    assert r["significant"] is True


def test_fisher_one_sided_direction():
    # recovery LOWER than normal: a one-sided 'greater' test must NOT be significant
    r = em.fisher_exact_sr(0, 50, 40, 50)
    assert r["significant"] is False
    assert r["p_value"] > 0.5


def test_fisher_result_shape():
    r = em.fisher_exact_sr(5, 50, 1, 50)
    assert r["table"] == [[5, 45], [1, 49]]
    assert r["alternative"] == "greater"
    assert r["method"] in ("scipy", "stdlib")


def test_fisher_scipy_stdlib_parity():
    """scipy path and stdlib fallback must agree on the one-sided p-value."""
    pytest.importorskip("scipy")
    cases = [(40, 50, 0, 50), (10, 100, 5, 100), (3, 30, 1, 30),
             (0, 20, 0, 20), (7, 40, 2, 35), (50, 50, 49, 50)]
    for sa, ta, sb, tb in cases:
        _, sp = em._fisher_scipy(sa, ta, sb, tb, "greater")
        _, dp = em._fisher_stdlib(sa, ta, sb, tb, "greater")
        assert abs(sp - dp) < 1e-9, (sa, ta, sb, tb, sp, dp)


# ─────────────────────── normal-vs-recovery comparison (M16) ───────────────────────

def test_compare_error_scene_srs():
    normal = {"successes": 2, "total": 100,
              "by_rbg": {"RBG_A": {"successes": 0, "total": 50},
                         "RBG_B": {"successes": 2, "total": 50}}}
    recovery = {"successes": 40, "total": 100,
                "by_rbg": {"RBG_A": {"successes": 25, "total": 50},
                           "RBG_B": {"successes": 15, "total": 50}}}
    cmp = em.compare_error_scene_srs(normal, recovery)
    # recovery is group A (hypothesized higher); row order recovery-then-normal
    assert cmp["overall"]["table"] == [[40, 60], [2, 98]]
    assert cmp["overall"]["significant"] is True
    assert set(cmp["by_rbg"]) == {"RBG_A", "RBG_B"}
    assert cmp["by_rbg"]["RBG_A"]["significant"] is True


def test_compare_skips_empty_or_missing_cells():
    normal = {"successes": 0, "total": 0, "by_rbg": {"RBG_A": {"successes": 0, "total": 0}}}
    recovery = {"successes": 0, "total": 0, "by_rbg": {"RBG_C": {"successes": 0, "total": 0}}}
    cmp = em.compare_error_scene_srs(normal, recovery)
    assert "overall" not in cmp          # zero totals -> skipped
    assert cmp["by_rbg"] == {}           # no shared, non-empty cells
