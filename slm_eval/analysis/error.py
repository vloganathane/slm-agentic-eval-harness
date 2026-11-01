"""ErrorAnalyzer — after-eval analysis of scoring results.

Answers: Why did the model fail? Which fields/strata? Root cause attribution?
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from slm_eval.stats import bootstrap_micro_f1


class ErrorAnalysis:
    """Structured output from ErrorAnalyzer — JSON-serializable."""

    def __init__(
        self,
        run_id: str,
        score_path: str | Path,
        config_path: str | Path,
        n_cases: int,
        n_fields: int,
        total_tp: int,
        total_fp: int,
        total_fn: int,
        f1_point: float,
        f1_lower: float,
        f1_upper: float,
        kappa: float | None,
        error_clusters: list[dict[str, Any]],
        stratum_perf: dict[str, dict[str, float]],
        defect_rate: float,
        defect_n: int,
        recommendations: list[str],
    ):
        self.run_id = run_id
        self.score_path = str(score_path)
        self.config_path = str(config_path)
        self.n_cases = n_cases
        self.n_fields = n_fields
        self.total_tp = total_tp
        self.total_fp = total_fp
        self.total_fn = total_fn
        self.f1_point = f1_point
        self.f1_lower = f1_lower
        self.f1_upper = f1_upper
        self.kappa = kappa
        self.error_clusters = error_clusters
        self.stratum_perf = stratum_perf
        self.defect_rate = defect_rate
        self.defect_n = defect_n
        self.recommendations = recommendations


class ErrorAnalyzer:
    """Analyze scoring results after full evaluation.

    Primary inputs: score JSONL table (case_id, field, tp, fp, fn, error_category)
    and config for strata keys.
    """

    def __init__(self, score_path: str | Path, config_path: str | Path, run_id: str):
        self.score_path = Path(score_path)
        self.config_path = Path(config_path)
        self.run_id = run_id
        self._scores: list[dict[str, Any]] | None = None
        self._cases: list[dict[str, Any]] | None = None

    def _load_scores(self) -> list[dict[str, Any]]:
        if self._scores is None:
            self._scores = []
            with self.score_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self._scores.append(json.loads(line))
        return self._scores

    def _load_cases(self) -> list[dict[str, Any]]:
        if self._cases is None:
            self._cases = []
            cases_path = self.config_path.parent / "cases.jsonl"
            if not cases_path.exists() and self.score_path:
                cases_path = self.score_path.parent / "holdout" / "cases.jsonl"
            if not cases_path.exists():
                # Keep searched path; user must provide correct path
                pass
            if cases_path.exists():
                with cases_path.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        self._cases.append(json.loads(line))
        return self._cases

    def _get_strata_key(self, case: dict[str, Any]) -> str:
        strata = case.get("strata", {})
        if isinstance(strata, dict):
            return ",".join(f"{k}={v}" for k, v in sorted(strata.items()))
        return "no_strata"

    def _case_by_id(self, case_id: str, cases: list[dict[str, Any]]) -> dict[str, Any] | None:
        for c in cases:
            if c.get("case_id") == case_id:
                return c
        return None

    def analyze(self) -> ErrorAnalysis:
        scores = self._load_scores()
        cases = self._load_cases()

        # Map case_id -> case for strata lookup
        case_map = {c.get("case_id"): c for c in cases}

        total_tp = total_fp = total_fn = 0
        n_fields = set()
        field_outcomes: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
        stratum_outcomes: dict[str, list[tuple[int, int, int]]] = defaultdict(list)

        error_clusters: list[dict[str, Any]] = []

        for s in scores:
            case_id = s.get("case_id")
            field = s.get("field")
            outcome = s.get("outcome", {}) or {}

            tp = s.get("tp", outcome.get("tp", 0))
            fp = s.get("fp", outcome.get("fp", 0))
            fn = s.get("fn", outcome.get("fn", 0))

            total_tp += tp
            total_fp += fp
            total_fn += fn
            if field:
                n_fields.add(field)
                field_outcomes[field].append((tp, fp, fn))

            # Strata tracking
            case = self._case_by_id(case_id, cases)
            strata_key = self._get_strata_key(case) if case else "no_strata"
            stratum_outcomes[strata_key].append((tp, fp, fn))

            # Error clustering
            if tp == 0 and (fp > 0 or fn > 0):
                error_category = s.get("error_category")
                if not error_category:
                    if fp > 0 and fn > 0:
                        error_category = f"fp+fn:{field or 'unknown'}"
                    elif fp > 0:
                        error_category = f"fp:{field or 'unknown'}"
                    else:
                        error_category = f"fn:{field or 'unknown'}"
                error_clusters.append(
                    {
                        "case_id": case_id,
                        "field": field or "unknown",
                        "error_category": error_category,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "model_output": s.get("model_output"),
                        "gold": s.get("gold"),
                    }
                )

        n_cases = len(case_map)
        denom = total_tp + total_fp + total_fn
        f1_point = total_tp / denom if denom else 0.0

        # Bootstrap CI on micro-F1
        outcomes = [
            type("FO", (), {"case_id": s.get("case_id"), "tp": s.get("tp", 0), "fp": s.get("fp", 0), "fn": s.get("fn", 0)})()
            for s in scores
        ]
        try:
            ci_metric = bootstrap_micro_f1(outcomes, resamples=10_000)
        except Exception:  # noqa: BLE001 - bootstrap failure: return neutral CI
            ci_metric = type("M", (), {"point": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0, "resamples": 10000})()

        # Stratum performance normalized
        stratum_perf = {}
        for key, outcomes_list in stratum_outcomes.items():
            stp = sfp = sfn = 0
            for tp, fp, fn in outcomes_list:
                stp += tp
                sfp += fp
                sfn += fn
            total = stp + sfp + sfn
            stratum_perf[key] = {
                "tp": stp,
                "fp": sfp,
                "fn": sfn,
                "tp_pct": round(stp / total, 3) if total else 0.0,
                "fp_pct": round(sfp / total, 3) if total else 0.0,
                "fn_pct": round(sfn / total, 3) if total else 0.0,
            }

        # Defect rate: cases with all-zero TP (complete extraction failure)
        # Group scores by case_id and count defects
        case_correctness: dict[str, bool] = {}
        for s in scores:
            case_id = s.get("case_id")
            tp = s.get("tp", outcome.get("tp", 0)) if case_id else None
            if case_id:
                if tp == 0 and (s.get("fp", 0) > 0 or s.get("fn", 0) > 0):
                    case_correctness[case_id] = False
                elif tp is not None and tp > 0:
                    case_correctness[case_id] = True
        defect_n = sum(1 for v in case_correctness.values() if not v)
        defect_rate = defect_n / n_cases if n_cases else 0.0

        # Recommendations
        recommendations = []
        # Sort fields by FN rate
        field_stats = []
        for field, outs in field_outcomes.items():
            tp_sum = sum(x[0] for x in outs)
            fp_sum = sum(x[1] for x in outs)
            fn_sum = sum(x[2] for x in outs)
            denom_f = tp_sum + fp_sum + fn_sum
            field_stats.append((field, tp_sum / denom_f if denom_f else 0, tp_sum, fp_sum, fn_sum))
        field_stats.sort(key=lambda x: x[1])

        if field_stats:
            worst_field = field_stats[0]
            recommendations.append(
                f"Focus on '{worst_field[0]}' field — lowest field F1: {worst_field[1]:.3f} "
                f"(TP={worst_field[2]}, FP={worst_field[3]}, FN={worst_field[4]})"
            )

        high_fn_strata = sorted(
            stratum_perf.items(), key=lambda x: x[1]["fn_pct"], reverse=True
        )[:2]
        if high_fn_strata:
            recommendations.append(
                "Strata with highest FN rate: "
                + ", ".join(f"{k} ({v['fn_pct']:.1%} FN)" for k, v in high_fn_strata)
            )

        recommendations.append(
            "Re-calibrate judge if kappa < 0.85 for subjective fields"
        )
        recommendations.append(
            "Consider synthetic data injection for low-coverage strata"
        )
        recommendations.append(
            "Review error clusters for field-specific tuning"
        )

        return ErrorAnalysis(
            run_id=self.run_id,
            score_path=self.score_path,
            config_path=self.config_path,
            n_cases=n_cases,
            n_fields=len(n_fields),
            total_tp=total_tp,
            total_fp=total_fp,
            total_fn=total_fn,
            f1_point=f1_point,
            f1_lower=getattr(ci_metric, "ci_lower", 0.0),
            f1_upper=getattr(ci_metric, "ci_upper", 0.0),
            kappa=None,
            error_clusters=error_clusters,
            stratum_perf=stratum_perf,
            defect_rate=defect_rate,
            defect_n=defect_n,
            recommendations=recommendations,
        )

    def summary(self) -> str:
        a = self.analyze()
        lines = [
            f"ERROR ANALYSIS: {self.run_id}",
            f"  Run ID: {self.run_id}",
            f"  Score file: {self.score_path}",
            f"  Cases analyzed: {a.n_cases}",
            f"  Fields: {a.n_fields}",
            f"  Total TP: {a.total_tp}, FP: {a.total_fp}, FN: {a.total_fn}",
            f"  Micro-F1: {a.f1_point:.3f} [{a.f1_lower:.3f}, {a.f1_upper:.3f}]",
            f"  Kappa: {a.kappa if a.kappa is not None else 'N/A'}",
            f"  Defect rate: {a.defect_rate:.1%} ({a.defect_n} cases)",
            f"  Error clusters: {len(a.error_clusters)}",
            "  Strata performance (top 3 by FN):",
        ]
        for strata, p in sorted(
            a.stratum_perf.items(), key=lambda x: x[1]["fn_pct"], reverse=True
        )[:3]:
            lines.append(
                f"    {strata}: tp={p['tp_pct']:.1%}, fp={p['fp_pct']:.1%}, fn={p['fn_pct']:.1%}"
            )
        lines.append("  Recommendations:")
        for r in a.recommendations:
            lines.append(f"    • {r}")
        return "\n".join(lines)