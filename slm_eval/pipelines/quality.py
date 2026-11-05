"""Unified quality reporting for the dataset pipeline.

Consolidates DataQualityReport from both data_pipeline.py and dataset_validator.py
into a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataQualityReport:
    """Data quality metrics from validation pass.

    This is the single canonical quality report used by all pipeline components.
    """
    total_cases: int = 0
    valid_cases: int = 0
    invalid_cases: int = 0
    missing_fields: dict[str, int] = field(default_factory=dict)
    schema_errors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    # New fields for richer quality analysis
    dedup_removed: int = 0
    near_duplicate_pairs: list[tuple[str, str]] = field(default_factory=list)
    text_length_stats: dict[str, float] = field(default_factory=dict)
    gold_field_coverage: dict[str, float] = field(default_factory=dict)
    strata_distribution: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "valid_cases": self.valid_cases,
            "invalid_cases": self.invalid_cases,
            "missing_fields": self.missing_fields,
            "schema_errors": self.schema_errors,
            "recommendations": self.recommendations,
            "dedup_removed": self.dedup_removed,
            "near_duplicate_pairs": self.near_duplicate_pairs,
            "text_length_stats": self.text_length_stats,
            "gold_field_coverage": self.gold_field_coverage,
            "strata_distribution": self.strata_distribution,
        }

    def add_error(self, case_id: str, errors: list[str]) -> None:
        """Add schema errors for a case."""
        self.invalid_cases += 1
        self.schema_errors.append(f"case {case_id}: {'; '.join(errors)}")

    def add_valid(self) -> None:
        self.valid_cases += 1

    def finalize(self, cases: list[dict[str, Any]] = None) -> None:
        """Post-validation analysis and recommendations."""
        if self.invalid_cases > 0:
            pct = 100 * self.invalid_cases / max(self.total_cases, 1)
            self.recommendations.append(
                f"Fix or remove {self.invalid_cases} invalid cases ({pct:.1f}%)"
            )

        if cases:
            self._analyze_text_stats(cases)
            self._analyze_gold_coverage(cases)
            self._analyze_strata_distribution(cases)

    def _analyze_text_stats(self, cases: list[dict[str, Any]]) -> None:
        """Analyze text length distribution."""
        lengths = [len(c.get("text", "")) for c in cases if c.get("text")]
        if lengths:
            self.text_length_stats = {
                "min": min(lengths),
                "max": max(lengths),
                "mean": sum(lengths) / len(lengths),
                "median": sorted(lengths)[len(lengths) // 2],
            }

    def _analyze_gold_coverage(self, cases: list[dict[str, Any]]) -> None:
        """Analyze gold field coverage."""
        all_fields: dict[str, int] = {}
        for c in cases:
            gold = c.get("gold", {})
            if isinstance(gold, dict):
                for field in gold:
                    all_fields[field] = all_fields.get(field, 0) + 1

        total = len(cases)
        if total > 0:
            self.gold_field_coverage = {
                field: count / total for field, count in all_fields.items()
            }

    def _analyze_strata_distribution(self, cases: list[dict[str, Any]]) -> None:
        """Analyze strata distribution."""
        dist: dict[str, dict[str, int]] = {}
        for c in cases:
            strata = c.get("strata", {})
            if isinstance(strata, dict):
                for key, value in strata.items():
                    if key not in dist:
                        dist[key] = {}
                    dist[key][value] = dist[key].get(value, 0) + 1
        self.strata_distribution = dist


def save_report(report: DataQualityReport, path: Path) -> None:
    """Save quality report as JSON."""
    import json
    path.write_text(json.dumps(report.to_dict(), indent=2))


def load_report(path: Path) -> DataQualityReport:
    """Load quality report from JSON."""
    import json
    data = json.loads(path.read_text())
    report = DataQualityReport(**{k: v for k, v in data.items() if k in DataQualityReport.__dataclass_fields__})
    return report
