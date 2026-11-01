"""Gate decision tests: CI-lower-bound-only, strata, κ (SPEC_V3 §6.5; AC-04, AC-05)."""

from slm_eval.config import ThresholdsConfig
from slm_eval.gate import GateInputs, StratumResult, classify_strata, decide_gate

T = ThresholdsConfig(target_f1=0.85, failure_threshold=0.5, min_per_stratum=50)


def _inputs(**kw):
    base = {
        "overall": (0.87, 0.855, 0.90),  # ci_lower clears target → the "healthy" baseline
        "overall_n": 200,
        "strata": [],
        "kappa": 0.9,
        "kappa_required": False,
        "pending_reviews": 0,
        "run_completed": True,
        "subjective_fields": [],
    }
    base.update(kw)
    return GateInputs(**base)


def test_promotes_when_ci_lower_clears_target():
    d = decide_gate(_inputs(overall=(0.87, 0.855, 0.90)), T)
    assert d.promote and d.status == "promote"


def test_blocks_on_ci_lower_even_when_point_clears():
    # AC-04: point 0.871 > 0.85 but ci_lower 0.842 < 0.85 → MUST block.
    d = decide_gate(_inputs(overall=(0.871, 0.842, 0.897)), T)
    assert not d.promote
    assert d.status == "blocked"
    assert any("ci_lower_below_target" in b for b in d.blockers)


def test_point_estimate_alone_never_gates():
    # A perfect point with a wide CI must not promote.
    d = decide_gate(_inputs(overall=(1.0, 0.6, 1.0)), T)
    assert not d.promote


def test_exact_boundary_ci_lower_equal_target_promotes():
    d = decide_gate(_inputs(overall=(0.86, 0.85, 0.87)), T)
    assert d.promote  # ci_lower >= target


def test_incomplete_run_is_error():
    d = decide_gate(_inputs(run_completed=False), T)
    assert d.status == "error" and not d.promote


def test_pending_review_blocks_not_errors():
    d = decide_gate(_inputs(pending_reviews=3), T)
    assert d.status == "blocked_pending_review"
    assert not d.promote and d.status != "error"


def test_stratum_blocker_when_ci_upper_at_failure_threshold():
    s = [StratumResult(key="layout=rotated", n=60, micro_f1=0.4, ci_lower=0.3, ci_upper=0.5)]
    d = decide_gate(_inputs(strata=s), T)
    assert any("stratum_blocker" in b for b in d.blockers)


def test_stratum_warning_when_below_target_but_not_failing():
    s = [StratumResult(key="lang=hi", n=60, micro_f1=0.8, ci_lower=0.7, ci_upper=0.9)]
    d = decide_gate(_inputs(strata=s), T)
    assert d.promote  # warning alone doesn't block
    assert any("stratum_below_target" in w for w in d.warnings)


def test_small_stratum_is_insufficient_n_never_blocker():
    s = [StratumResult(key="lang=mr", n=10, micro_f1=0.1, ci_lower=0.0, ci_upper=0.2)]
    d = decide_gate(_inputs(strata=s), T)
    assert d.promote  # insufficient_n is a warning, never a blocker (§6.5)
    assert any("insufficient_n" in w for w in d.warnings)


def test_classify_strata_statuses():
    strata = [
        StratumResult("ok", 100, 0.9, 0.86, 0.94),
        StratumResult("warn", 100, 0.8, 0.75, 0.85),
        StratumResult("block", 100, 0.4, 0.3, 0.5),
        StratumResult("small", 10, 0.9, 0.8, 1.0),
    ]
    got = {s.key: s.status for s in classify_strata(strata, T)}
    assert got == {"ok": "ok", "warn": "warning", "block": "blocker", "small": "insufficient_n"}


def test_kappa_missing_blocks_when_subjective_present():
    d = decide_gate(_inputs(subjective_fields=["vendor_name"], kappa=None), T)
    assert not d.promote
    assert any("kappa_missing" in b for b in d.blockers)


def test_kappa_low_blocks():
    d = decide_gate(_inputs(subjective_fields=["vendor_name"], kappa=0.7), T)
    assert not d.promote
    assert any("kappa_below_target" in b for b in d.blockers)


def test_kappa_ignored_when_no_subjective_fields():
    d = decide_gate(_inputs(subjective_fields=[], kappa=None), T)
    assert d.promote  # no subjective fields → κ not required
