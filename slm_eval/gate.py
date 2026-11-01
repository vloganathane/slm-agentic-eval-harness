"""Gate decision (SPEC_V3 §6.5): pure, CI-lower-bound-only, no point estimates.

The rule that makes this an *eval gate* and not a score reporter: promotion keys
off the 95% CI lower bound, never the point estimate. A model that scores 0.87
with CI [0.84, 0.90] against target 0.85 does NOT promote — we can't yet say it
clears the bar.

Everything here is a pure function over already-computed numbers so the decision
is auditable and unit-testable without a pipeline run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from slm_eval.config import ThresholdsConfig

Status = Literal["promote", "blocked", "blocked_pending_review", "error"]


@dataclass(frozen=True)
class StratumResult:
    key: str
    n: int
    micro_f1: float
    ci_lower: float
    ci_upper: float
    status: Literal["ok", "warning", "blocker", "insufficient_n"] = "ok"


@dataclass(frozen=True)
class GateInputs:
    overall: tuple[float, float, float]  # (micro_f1, ci_lower, ci_upper)
    overall_n: int
    strata: list[StratumResult] = field(default_factory=list)
    kappa: float | None = None
    kappa_required: bool = False
    pending_reviews: int = 0
    run_completed: bool = True
    subjective_fields: list[str] = field(default_factory=list)
    defect_rate: float | None = None      # dataset_defect (quarantined out of F1)
    defect_n: int = 0
    total_n: int = 0


@dataclass(frozen=True)
class GateDecision:
    status: Status
    promote: bool
    reason: str
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def classify_strata(
    strata: list[StratumResult], thresholds: ThresholdsConfig
) -> list[StratumResult]:
    """Label each stratum per §6.5 BLOCKER/WARNING rules."""
    out = []
    for s in strata:
        if s.n < thresholds.min_per_stratum:
            status = "insufficient_n"
        elif s.ci_upper <= thresholds.failure_threshold:
            status = "blocker"
        elif s.micro_f1 < thresholds.target_f1:
            status = "warning"
        else:
            status = "ok"
        out.append(
            StratumResult(
                key=s.key, n=s.n, micro_f1=s.micro_f1,
                ci_lower=s.ci_lower, ci_upper=s.ci_upper, status=status,
            )
        )
    return out


def dataset_defect_rate(signal_values: list[str]) -> tuple[float, int]:
    """Fraction of cases marked ``unextractable``, as (rate, defect_count).

    These are cases whose input carries no extractable signal (blank/garbage OCR).
    The gate reports them as ``dataset_defect`` so a broken input is never scored
    as a weak model — the defect rate is the dataset's own quality metric.
    """
    if not signal_values:
        return 0.0, 0
    defect_n = sum(1 for v in signal_values if v == "unextractable")
    return defect_n / len(signal_values), defect_n


def decide_gate(inputs: GateInputs, thresholds: ThresholdsConfig) -> GateDecision:
    """Apply §6.5. Order matters: error → pending review → CI → strata → κ."""
    if not inputs.run_completed:
        return GateDecision(status="error", promote=False, reason="run_incomplete")

    blockers: list[str] = []
    warnings: list[str] = []

    # --- overall CI lower bound (AC-04: never the point estimate) ---
    micro_f1, ci_lower, _ci_upper = inputs.overall
    if ci_lower < thresholds.target_f1:
        blockers.append(
            f"ci_lower_below_target: {ci_lower:.3f} < {thresholds.target_f1} "
            f"(point {micro_f1:.3f} cannot gate)"
        )

    # --- per-stratum rules ---
    classified = classify_strata(inputs.strata, thresholds)
    for s in classified:
        if s.status == "blocker":
            blockers.append(f"stratum_blocker: {s.key} n={s.n} ci_upper={s.ci_upper:.3f}")
        elif s.status == "warning":
            warnings.append(f"stratum_below_target: {s.key} micro_f1={s.micro_f1:.3f}")
        elif s.status == "insufficient_n":
            warnings.append(f"insufficient_n: {s.key} n={s.n} < {thresholds.min_per_stratum}")

    # --- dataset defect (quarantine, §2): reported, never a model blocker ---
    if inputs.defect_rate is not None and inputs.defect_rate > 0:
        warnings.append(
            f"dataset_defect: {inputs.defect_n}/{inputs.total_n} cases "
            f"({inputs.defect_rate:.0%}) unextractable — excluded from micro-F1"
        )

    # --- κ gate (§6.4, AC-05): only when subjective fields exist ---
    if inputs.subjective_fields:
        if inputs.kappa is None:
            blockers.append("kappa_missing: subjective fields present but no calibration")
        elif inputs.kappa < 0.85:
            blockers.append(f"kappa_below_target: {inputs.kappa:.3f} < 0.85")

    # --- human review queue blocks (never errors) ---
    if inputs.pending_reviews > 0:
        warnings.append(f"pending_reviews: {inputs.pending_reviews}")
        return GateDecision(
            status="blocked_pending_review", promote=False,
            reason="pending_human_review", blockers=blockers, warnings=warnings,
        )

    if blockers:
        return GateDecision(
            status="blocked", promote=False,
            reason=blockers[0].split(":")[0], blockers=blockers, warnings=warnings,
        )

    return GateDecision(status="promote", promote=True, reason="all_checks_passed",
                        blockers=[], warnings=warnings)
