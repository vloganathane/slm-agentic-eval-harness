"""Idiot Index Gate (SPEC_V3 Appendix B): meta-audit for over-engineering.

Questions every requirement, flags complexity that hasn't earned its keep.
Exit codes: 0 = lean, 1 = bloated (with report), 2 = error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from slm_eval.config import EvalConfig


@dataclass(frozen=True)
class IdiotFinding:
    severity: str          # "cut" | "question" | "keep"
    component: str         # e.g. "strata", "kappa", "bootstrap"
    message: str           # what to cut/question/keep
    savings: str           # estimated complexity reduction


@dataclass(frozen=True)
class IdiotReport:
    config_path: str
    findings: list[IdiotFinding] = field(default_factory=list)
    total_components: int = 0
    cuttable: int = 0
    questionable: int = 0
    justified: int = 0

    def summary(self) -> str:
        lines = [f"IDIOT INDEX REPORT: {self.config_path}"]
        lines.append(f"  Components audited: {self.total_components}")
        lines.append(f"  🔴 Cut:       {self.cuttable}")
        lines.append(f"  🟡 Question:  {self.questionable}")
        lines.append(f"  🟢 Justified: {self.justified}")
        lines.append("")
        for f in self.findings:
            icon = {"cut": "🔴", "question": "🟡", "keep": "🟢"}[f.severity]
            lines.append(f"  {icon} [{f.component}] {f.message}  (saves: {f.savings})")
        return "\n".join(lines)

    def exit_code(self) -> int:
        return 1 if self.cuttable > 0 else 0


# ---------------------------------------------------------------------------
# Rules: each returns (severity, message, savings) or None if justified
# ---------------------------------------------------------------------------

def _audit_strata(cfg: EvalConfig) -> IdiotFinding | None:
    if not cfg.strata.keys:
        return None
    # Strata justified only if: holdout_n >= 5 * num_strata_values AND per-stratum decisions differ
    # Heuristic: if single_key with 1 key and min_per_stratum > holdout_n / 4 → question
    if cfg.strata.mode == "single_key" and len(cfg.strata.keys) == 1:
        return IdiotFinding(
            severity="question",
            component="strata",
            message=f"Single-key strata '{cfg.strata.keys[0]}' — does the gate decision actually differ per stratum?",
            savings="~80 lines (split_stratified, _merge_small_strata, stratum metrics)",
        )
    if cfg.thresholds.min_per_stratum > 10:
        return IdiotFinding(
            severity="question",
            component="strata.min_per_stratum",
            message=f"min_per_stratum={cfg.thresholds.min_per_stratum} requires huge holdout — is per-stratum gating real?",
            savings="Simpler random split + overall CI",
        )
    return IdiotFinding(
        severity="keep",
        component="strata",
        message="Multi-key or low min_per_stratum — strata may be justified",
        savings="—",
    )


def _audit_kappa(cfg: EvalConfig) -> IdiotFinding | None:
    if not cfg.fields.subjective_fields:
        return IdiotFinding(
            severity="cut",
            component="kappa",
            message="No subjective fields declared — κ calibration is dead code",
            savings="~60 lines (CalibrationGate, cohen_kappa, calibration I/O)",
        )
    # κ justified only if: human labels exist AND judge cost > trivial AND promotion depends on it
    cal = cfg.judge.calibration
    if not Path(cal.path).exists():
        return IdiotFinding(
            severity="question",
            component="kappa",
            message=f"subjective_fields={cfg.fields.subjective_fields} but no calibration file at {cal.path} — κ will block promotion",
            savings="Either add labels or drop subjective fields",
        )
    # If judge is fast/cheap, κ may not earn its keep
    return IdiotFinding(
        severity="keep",
        component="kappa",
        message="Subjective fields + calibration present — κ justified",
        savings="—",
    )


def _audit_bootstrap(cfg: EvalConfig) -> IdiotFinding | None:
    if cfg.stats.resamples > 5000:
        return IdiotFinding(
            severity="question",
            component="bootstrap.resamples",
            message=f"B={cfg.stats.resamples} — Clopper-Pearson (analytic) is exact and 1000x faster",
            savings="~40 lines (bootstrap_micro_f1) + 99% runtime",
        )
    if cfg.stats.resamples > 1000:
        return IdiotFinding(
            severity="question",
            component="bootstrap.resamples",
            message=f"B={cfg.stats.resamples} — B=1000 gives CI width within ~0.005 of B=10000",
            savings="10x faster bootstrap",
        )
    return None  # B <= 1000 is reasonable


def _audit_judge_scope(cfg: EvalConfig) -> IdiotFinding | None:
    # If all fields are subjective → judge everything → expensive + κ required
    set(cfg.fields.exact_match_fields) | set(cfg.fields.subjective_fields)
    if len(cfg.fields.exact_match_fields) == 0 and cfg.fields.subjective_fields:
        return IdiotFinding(
            severity="cut",
            component="judge_scope",
            message="ALL fields subjective — judge handles structured fields (total, dates, IDs) that deterministic validators would score perfectly",
            savings="~90% judge calls + κ calibration + bootstrap variance",
        )
    return None


def _audit_synthetic_loop(cfg: EvalConfig) -> IdiotFinding | None:
    # Synthetic loop only justified if: real data < 500 cases AND defect_rate > 0.2
    # We can't know real data size from config alone, but we can flag the config knobs
    if cfg.cost_control.max_cost_per_round_usd > 5.0:
        return IdiotFinding(
            severity="question",
            component="synthetic_loop",
            message="cost_control.max_cost_per_round_usd > $5 — synthetic generation budgeted before real data is exhausted?",
            savings="Curate real data first; synthetic is a last resort",
        )
    return None


def _audit_advisory_lane(cfg: EvalConfig) -> IdiotFinding | None:
    if cfg.human_review.timeout_hours > 0:
        return IdiotFinding(
            severity="question",
            component="human_review",
            message=f"human_review.timeout_hours={cfg.human_review.timeout_hours} — advisory lane adds queue, reviewer UI, calibration re-runs. Does CI-lower-bound gate alone suffice?",
            savings="~200 lines (review queue, analyst categorization, recalibration triggers)",
        )
    return None


def _audit_reporting(cfg: EvalConfig) -> IdiotFinding | None:
    if len(cfg.reporting.formats) > 1:
        return IdiotFinding(
            severity="cut",
            component="reporting",
            message=f"formats={cfg.reporting.formats} — gate.json + exit code IS the contract; markdown/json/html are presentation",
            savings="Single JSON artifact; render elsewhere",
        )
    return None


AUDITORS = [
    _audit_strata,
    _audit_kappa,
    _audit_bootstrap,
    _audit_judge_scope,
    _audit_synthetic_loop,
    _audit_advisory_lane,
    _audit_reporting,
]


def run_idiot_index(config_path: str | Path) -> IdiotReport:
    """Run the Idiot Index audit on an eval_config.yaml."""
    cfg = EvalConfig.from_yaml(config_path)
    findings = []
    for auditor in AUDITORS:
        result = auditor(cfg)
        if result:
            findings.append(result)

    return IdiotReport(
        config_path=str(config_path),
        findings=findings,
        total_components=len(AUDITORS),
        cuttable=sum(1 for f in findings if f.severity == "cut"),
        questionable=sum(1 for f in findings if f.severity == "question"),
        justified=sum(1 for f in findings if f.severity == "keep"),
    )


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def _cmd_idiot_index(args) -> int:
    report = run_idiot_index(args.config)
    print(report.summary())
    return report.exit_code()


# Allow: python -m slm_eval.idiot_index -c configs/cord.yaml
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Idiot Index Gate — audit eval config for bloat")
    p.add_argument("-c", "--config", required=True)
    raise SystemExit(_cmd_idiot_index(p.parse_args()))