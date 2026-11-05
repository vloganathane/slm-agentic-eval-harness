"""GateAnalyzer — explain gate.json decisions and provide root-cause analysis.

Answers:
- Why was the gate blocked?
- Which specific fields/strata caused CI lower bound to fall below target?
- What concrete changes would flip BLOCKED → PROMOTE?
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class GateAnalysis:
    """Structured output from GateAnalyzer."""

    def __init__(
        self,
        run_id: str,
        gate_path: str | Path,
        config_path: str | Path,
        passed: bool,
        blockers: list[str],
        warnings: list[str],
        micro_f1: float,
        ci_lower: float,
        ci_upper: float,
        kappa: float | None,
        n: int,
        stratum_blockers: list[dict[str, Any]],
        root_cause: str | None,
        remediation: list[dict[str, Any]],
    ):
        self.run_id = run_id
        self.gate_path = str(gate_path)
        self.config_path = str(config_path)
        self.passed = passed
        self.blockers = blockers
        self.warnings = warnings
        self.micro_f1 = micro_f1
        self.ci_lower = ci_lower
        self.ci_upper = ci_upper
        self.kappa = kappa
        self.n = n
        self.stratum_blockers = stratum_blockers
        self.root_cause = root_cause
        self.remediation = remediation


class GateAnalyzer:
    """Analyze gate.json to explain decisions and suggest fixes.

    Args:
        gate_path: Path to gate.json
        config_path: Path to eval_config.yaml (or .json)
        score_path: Path to score JSONL (optional, for field-level drill-down)
    """

    def __init__(
        self,
        gate_path: str | Path,
        config_path: str | Path,
        score_path: str | Path | None = None,
    ):
        self.gate_path = Path(gate_path)
        self.config_path = Path(config_path)
        self.score_path = Path(score_path) if score_path else None

    def _load_gate(self) -> dict[str, Any]:
        with self.gate_path.open() as f:
            return json.load(f)

    def _load_scores(self) -> list[dict[str, Any]]:
        if not self.score_path or not self.score_path.exists():
            return []
        results = []
        with self.score_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                results.append(json.loads(line))
        return results

    def analyze(self) -> GateAnalysis:
        gate = self._load_gate()
        scores = self._load_scores()

        # Parse gate structure
        status = gate.get("status", "unknown")
        passed = status == "promote"
        gate.get("promote", False)
        blockers = gate.get("blockers", [])
        warnings = gate.get("warnings", [])
        metrics = gate.get("metrics", {})
        micro_f1 = metrics.get("micro_f1", 0.0)
        ci_lower = metrics.get("ci_lower", 0.0)
        ci_upper = metrics.get("ci_upper", 0.0)
        kappa = metrics.get("kappa")
        n = gate.get("n", gate.get("holdout", {}).get("n", 0))
        defect_rate = gate.get("dataset_defect", {}).get("rate", 0.0)
        defect_n = gate.get("dataset_defect", {}).get("defect_n", 0)

        # Parse stratum-level decisions from gate.json
        stratum_decisions = gate.get("strata", [])
        stratum_blockers: list[dict[str, Any]] = []

        for stratum in stratum_decisions:
            s_status = stratum.get("status")
            s_name = stratum.get("stratum", stratum.get("name", "unknown"))
            s_ci_lower = stratum.get("ci_lower", 0)
            s_ci_upper = stratum.get("ci_upper", 0)
            s_n = stratum.get("n", 0)
            s_blocked = s_status in ("blocked", "blocked_insufficient_n")

            if s_blocked:
                blocker = {
                    "stratum": s_name,
                    "status": s_status,
                    "ci_lower": s_ci_lower,
                    "ci_upper": s_ci_upper,
                    "n": s_n,
                    "reason": stratum.get("reason", s_status),
                    "gap_to_target": round(0.85 - s_ci_upper, 4),
                }
                stratum_blockers.append(blocker)

        # Root cause analysis
        root_cause: str | None = None
        if not passed:
            # Identify primary blocker
            if ci_lower < 0.85:
                root_cause = (
                    f"CI lower bound ({ci_lower:.4f}) below target (0.85) — "
                    f"point estimate {micro_f1:.4f} cannot gate"
                )
            elif stratum_blockers:
                root_cause = (
                    f"{len(stratum_blockers)} stratum(s) blocking promotion — "
                    f"strata require ci_upper >= 0.85 with min n per stratum"
                )
            elif defect_rate and defect_rate > 0.1:
                root_cause = (
                    f"Dataset defect rate {defect_rate:.1%} ({defect_n} cases) too high"
                )
            elif kappa is not None and kappa < 0.85:
                root_cause = f"κ ({kappa:.3f}) below 0.85 for subjective fields"
            else:
                root_cause = "Unknown — inspect gate.json for details"

        # Remediation recommendations
        remediation: list[dict[str, Any]] = []

        if root_cause and "CI lower" in root_cause:
            remediation.append({
                "type": "improve_f1",
                "action": "Address failing fields to raise micro-F1 and CI lower bound",
                "priority": 1,
                "detail": f"Need CI lower bound ≥ 0.85; currently {ci_lower:.4f}",
            })

        if stratum_blockers:
            for sb in stratum_blockers:
                remediation.append({
                    "type": "fix_stratum",
                    "action": f"Boost F1 for stratum '{sb['stratum']}'",
                    "priority": 1,
                    "detail": f"Gap: {sb['gap_to_target']:.4f} to reach ci_upper ≥ 0.85; n={sb['n']}",
                })

        if defect_n > 0:
            remediation.append({
                "type": "fix_data",
                "action": "Address defect cases",
                "priority": 2,
                "detail": f"{defect_n} cases ({defect_rate:.1%}) unextractable",
            })

        # Field-level drill-down
        if scores:
            # Aggregate by field
            field_tp: dict[str, int] = {}
            field_fp: dict[str, int] = {}
            field_fn: dict[str, int] = {}
            for s in scores:
                field = s.get("field", "unknown")
                tp = s.get("tp", 0)
                fp = s.get("fp", 0)
                fn = s.get("fn", 0)
                field_tp[field] = field_tp.get(field, 0) + tp
                field_fp[field] = field_fp.get(field, 0) + fp
                field_fn[field] = field_fn.get(field, 0) + fn

            field_f1 = {}
            for f, tp in field_tp.items():
                total = tp + field_fp.get(f, 0) + field_fn.get(f, 0)
                field_f1[f] = tp / total if total else 0.0

            sorted_fields = sorted(field_f1.items(), key=lambda x: x[1])

            for f, f1_val in sorted_fields[:3]:
                if f1_val < 0.5:
                    remediation.append({
                        "type": "fix_field",
                        "action": f"Improve scoring for field '{f}'",
                        "priority": min(3, 4 - sorted_fields.index((f, f1_val))),
                        "detail": f"Field F1 = {f1_val:.3f} (TP={field_tp.get(f, 0)}, FP={field_fp.get(f, 0)}, FN={field_fn.get(f, 0)})",
                    })

        # Synthetic data recommendation
        remediation.append({
            "type": "synthetic_data",
            "action": "Inject synthetic cases for failing fields/strata via M3 loop",
            "priority": 3,
            "detail": f"MinHash threshold 0.6; target {len(stratum_blockers)} stratum(s)",
        })

        # Judge calibration
        if kappa is not None and kappa < 0.85:
            remediation.append({
                "type": "calibrate",
                "action": "Re-run judge calibration with more pairs",
                "priority": 2,
                "detail": f"κ={kappa:.3f} < 0.85; add calibration pairs to judge.jsonl",
            })

        return GateAnalysis(
            run_id=gate.get("run_id", "unknown"),
            gate_path=self.gate_path,
            config_path=self.config_path,
            passed=passed,
            blockers=blockers,
            warnings=warnings,
            micro_f1=micro_f1,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            kappa=kappa,
            n=n,
            stratum_blockers=stratum_blockers,
            root_cause=root_cause,
            remediation=remediation,
        )

    def summary(self) -> str:
        a = self.analyze()
        lines = [
            f"GATE ANALYSIS: {a.run_id}",
            f"  Gate file: {self.gate_path}",
            f"  Status: {'PROMote' if a.passed else 'BLOCKED'}",
            f"  Micro-F1: {a.micro_f1:.4f} [{a.ci_lower:.4f}, {a.ci_upper:.4f}]",
            f"  Kappa: {a.kappa if a.kappa is not None else 'N/A'}",
            f"  Cases (n): {a.n}",
        ]
        if a.root_cause:
            lines.append(f"  Root cause: {a.root_cause}")
        if a.stratum_blockers:
            lines.append("  Stratum blockers:")
            for sb in a.stratum_blockers:
                lines.append(
                    f"    • '{sb['stratum']}': ci_upper={sb['ci_upper']:.4f}, n={sb['n']}, gap={sb['gap_to_target']:.4f} to target"
                )
        if a.blockers:
            lines.append("  Blockers:")
            for b in a.blockers:
                lines.append(f"    • {b}")
        lines.append("  Remediation recommendations:")
        for r in a.remediation:
            lines.append(f"    [{r['priority']}] {r['action']}: {r['detail']}")
        return "\n".join(lines)


def load_gate_for_analyzer(
    run_dir: str | Path,
    config_path: str | Path,
) -> GateAnalyzer:
    """Convenience: locate gate.json + scores from a run directory."""
    run_dir = Path(run_dir)
    gate_path = run_dir / "gate.json"
    score_path = run_dir / "scores.jsonl"
    if not gate_path.exists():
        raise FileNotFoundError(f"gate.json not found in {run_dir}")
    return GateAnalyzer(gate_path, config_path, score_path if score_path.exists() else None)
