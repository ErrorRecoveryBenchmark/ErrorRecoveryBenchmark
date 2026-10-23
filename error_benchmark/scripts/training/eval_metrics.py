#!/usr/bin/env python
"""Evaluation metrics for the authoritative RecoverBench protocol.

Authoritative metric (docs/项目全景总结.md, M16): Success Rate (point
estimate) + Fisher exact test for the recovery-vs-normal comparison
(recovery SR > normal SR, p<0.05). Bootstrap CI is NOT used (deprecated,
docs/项目全景总结.md:534).

Pure stdlib + optional scipy, so it imports without robosuite and is
unit-testable anywhere (the framework package's __init__ pulls heavy deps).
"""

from math import comb

# Copy of error_benchmark.framework.recovery_types.SUBTYPE_TO_RBG. Kept
# byte-for-byte in sync by test_eval_metrics_spec.test_rbg_copy_matches_canonical;
# we cannot import the canonical one because framework/__init__ pulls robosuite.
SUBTYPE_TO_RBG = {
    "grasp_misalignment_D0": "RBG_A",
    "grasp_misalignment_D1": "RBG_A",
    "grasp_wrong_pose_D0": "RBG_A",
    "grasp_wrong_pose_D1": "RBG_A",
    "drop_in_transit_D0": "RBG_B",
    "drop_in_transit_D1": "RBG_B",
    "drop_at_wrong_place_D0": "RBG_B",
    "drop_at_wrong_place_D1": "RBG_B",
    "collision_holding_D0": "RBG_C",
    "collision_holding_D1": "RBG_C",
    "collision_empty_D0": "RBG_C",
    "collision_empty_D1": "RBG_C",
    "collision_eef_object_D0": "RBG_C",
    "collision_eef_object_D1": "RBG_C",
    "collision_self_D0": "RBG_C",
    "collision_self_D1": "RBG_C",
    "wrong_object_D0": "RBG_D",
    "wrong_object_D1": "RBG_D",
    "trajectory_regression_D0": "RBG_E",
    "trajectory_regression_D1": "RBG_E",
    "stuck_no_progress_D0": "RBG_E",
    "stuck_no_progress_D1": "RBG_E",
    "position_error_D0": "RBG_E",
    "position_error_D1": "RBG_E",
}

SIGNIFICANCE_ALPHA = 0.05


def rbg_of(subtype: str):
    """Return the RBG ('RBG_A'..'RBG_E') for a subtype id, or None if unknown."""
    return SUBTYPE_TO_RBG.get(subtype)


def _hypergeom_pmf(x: int, r1: int, r2: int, c1: int) -> float:
    """P(X=x) for a 2x2 table with fixed margins (row totals r1, r2;
    success-column total c1)."""
    return comb(r1, x) * comb(r2, c1 - x) / comb(r1 + r2, c1)


def _fisher_stdlib(succ_a, tot_a, succ_b, tot_b, alternative="greater"):
    """Exact Fisher test via the hypergeometric distribution (no scipy).

    Table [[a, b], [c, d]] with a=succ_a. 'greater' tests SR_A > SR_B.
    Returns (sample_odds_ratio, p_value).
    """
    a = succ_a
    b = tot_a - succ_a
    c = succ_b
    d = tot_b - succ_b
    r1, r2 = a + b, c + d          # row totals
    c1 = a + c                     # success column total

    odds = float("inf") if (b * c) == 0 else (a * d) / (b * c)

    # Degenerate margins -> no information -> p = 1.0
    if r1 == 0 or r2 == 0 or c1 == 0 or c1 == (r1 + r2):
        return odds, 1.0

    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    pmf = {x: _hypergeom_pmf(x, r1, r2, c1) for x in range(lo, hi + 1)}
    if alternative == "greater":
        p = sum(v for x, v in pmf.items() if x >= a)
    elif alternative == "less":
        p = sum(v for x, v in pmf.items() if x <= a)
    else:  # two-sided
        pa = pmf[a]
        p = sum(v for v in pmf.values() if v <= pa + 1e-12)
    return odds, min(1.0, p)


def _fisher_scipy(succ_a, tot_a, succ_b, tot_b, alternative="greater"):
    """Fisher test via scipy. Returns (conditional_odds_ratio, p_value)."""
    from scipy.stats import fisher_exact
    a, b = succ_a, tot_a - succ_a
    c, d = succ_b, tot_b - succ_b
    odds, p = fisher_exact([[a, b], [c, d]], alternative=alternative)
    return float(odds), float(p)


def fisher_exact_sr(succ_a, tot_a, succ_b, tot_b, alternative="greater") -> dict:
    """Compare two success rates with Fisher's exact test.

    Group A is the hypothesized-higher group (e.g. the recovery policy);
    alternative='greater' tests SR_A > SR_B (the M16 criterion). Uses scipy
    if importable, otherwise an exact stdlib fallback.
    """
    try:
        odds, p = _fisher_scipy(succ_a, tot_a, succ_b, tot_b, alternative)
        method = "scipy"
    except Exception:
        odds, p = _fisher_stdlib(succ_a, tot_a, succ_b, tot_b, alternative)
        method = "stdlib"
    return {
        "p_value": float(p),
        "odds_ratio": float(odds),
        "significant": bool(p < SIGNIFICANCE_ALPHA),
        "table": [[succ_a, tot_a - succ_a], [succ_b, tot_b - succ_b]],
        "alternative": alternative,
        "method": method,
    }


def _succ_tot(block: dict):
    """Pull (successes, total) from an error_scenes block or a sub-bucket."""
    if not isinstance(block, dict):
        return None, None
    return block.get("successes"), block.get("total")


def compare_error_scene_srs(normal: dict, recovery: dict, alternative: str = "greater") -> dict:
    """Fisher exact comparison (recovery SR > normal SR) at the overall level
    plus per-RBG and per-subtype, for the M16 criterion.

    `normal` and `recovery` are error_scenes result blocks with keys
    {successes, total, by_rbg{...}, by_subtype{...}}. Cells are compared only
    when both sides have a non-empty total. Recovery is group A (the
    hypothesized-higher group), so a 'greater' test asks recovery > normal.
    """
    out = {}
    n_s, n_t = _succ_tot(normal)
    r_s, r_t = _succ_tot(recovery)
    if None not in (n_s, n_t, r_s, r_t) and n_t and r_t:
        out["overall"] = fisher_exact_sr(r_s, r_t, n_s, n_t, alternative)

    for level in ("by_rbg", "by_subtype"):
        n_level = normal.get(level, {}) or {}
        r_level = recovery.get(level, {}) or {}
        cells = {}
        for key in sorted(set(n_level) & set(r_level)):
            ns, nt = _succ_tot(n_level[key])
            rs, rt = _succ_tot(r_level[key])
            if None not in (ns, nt, rs, rt) and nt and rt:
                cells[key] = fisher_exact_sr(rs, rt, ns, nt, alternative)
        out[level] = cells
    return out
