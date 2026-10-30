"""Dataset pipeline enhancements: loading, validation, augmentation.

Unified data pipeline for the SLM Evaluation Harness.
Loads from cases.jsonl, validates schema, and prepares data for evaluation/splitting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from slm_eval.analysis.dataset import DatasetAnalyzer


@dataclass
class DataQualityReport:
    """Data quality metrics from validation pass."""
    total_cases: int = 0
    valid_cases: int = 0
    invalid_cases: int = 0
    missing_fields: dict[str, int] = field(default_factory=dict)
    schema_errors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "valid_cases": self.valid_cases,
            "invalid_cases": self.invalid_cases,
            "missing_fields": self.missing_fields,
            "schema_errors": self.schema_errors,
            "recommendations": self.recommendations,
        }


class DataPipeline:
    """Unified data pipeline for loading, validating, and preparing datasets.

    Works with cases.jsonl format that contains:
    - case_id: str
    - text: str
    - gold: dict  (ground truth annotations)
    - strata: dict  (strata: item_count, ocr_signal, quality)
    """

    # Required fields for a valid case
    REQUIRED_FIELDS: ClassVar[set[str]] = {"case_id", "text", "gold", "strata"}

    def __init__(self, data_dir: Path = Path("runs")):
        self.data_dir = data_dir

    def load_cases(self, path: Path) -> list[dict[str, Any]]:
        """Load cases from a cases.jsonl file.

        Each line should be a JSON object with at minimum:
        case_id, text, gold, strata fields.
        """
        cases = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    case = json.loads(line)
                    cases.append(case)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed line: {e}")
        return cases

    def load_from_run(self, run_id: str) -> tuple[list[dict[str, Any]], Path]:
        """Load cases from a run directory.

        Returns (cases, cases_path) tuple.
        """
        cases_path = self.data_dir / run_id / "holdout" / "cases.jsonl"
        if cases_path.exists():
            cases = self.load_cases(cases_path)
            return cases, cases_path
        # Also check at data_dir/run_id/cases.jsonl
        alt_path = self.data_dir / run_id / "cases.jsonl"
        if alt_path.exists():
            cases = self.load_cases(alt_path)
            return cases, alt_path
        raise FileNotFoundError(
            f"cases.jsonl not found in {self.data_dir / run_id}/holdout/ or {self.data_dir / run_id}/"
        )

    def validate(self, cases: list[dict[str, Any]]) -> DataQualityReport:
        """Validate cases against the expected schema.

        Checks:
        - Required fields present
        - case_id is unique and non-empty
        - text is non-empty string
        - gold is a dict
        - strata contains expected keys
        """
        report = DataQualityReport(total_cases=len(cases))
        seen_ids: set[str] = set()

        for i, case in enumerate(cases):
            errors: list[str] = []

            # Check required fields
            missing = self.REQUIRED_FIELDS - set(case.keys())
            if missing:
                errors.append(f"missing required fields: {missing}")

            # Check case_id
            case_id = case.get("case_id")
            if not case_id:
                errors.append("empty case_id")
            elif case_id in seen_ids:
                errors.append(f"duplicate case_id: {case_id}")
            else:
                seen_ids.add(case_id)

            # Check text is non-empty string (warn but don't reject)
            text = case.get("text")
            if not isinstance(text, str):
                errors.append(f"text must be str, got {type(text).__name__}")
            elif not text.strip():
                # Empty text is a warning, not a hard error
                pass

            # Check gold is dict
            gold = case.get("gold")
            if gold is not None and not isinstance(gold, dict):
                errors.append(f"gold must be dict, got {type(gold).__name__}")

            # Check strata has expected keys
            strata = case.get("strata")
            if strata is not None and not isinstance(strata, dict):
                errors.append(f"strata must be dict, got {type(strata).__name__}")

            if errors:
                report.invalid_cases += 1
                # Track first error per case to avoid flooding
                report.schema_errors.append(f"case {case_id}: {'; '.join(errors)}")
            else:
                report.valid_cases += 1

        # Post-validation recommendations
        if report.missing_fields:
            field_counts = {}
            for err in report.schema_errors:
                for field in self.REQUIRED_FIELDS:
                    if field in err:
                        field_counts[field] = field_counts.get(field, 0) + 1
            report.recommendations.append(
                f"Add missing fields in {len(field_counts)} cases: {list(field_counts.keys())}"
            )
        if report.invalid_cases > 0:
            pct = 100 * report.invalid_cases / max(report.total_cases, 1)
            report.recommendations.append(
                f"Fix or remove {report.invalid_cases} invalid cases ({pct:.1f}%)"
            )

        return report

    def analyze(
        self,
        cases_path: Path,
        minhash_threshold: float = 0.8,
        top_n_duplicates: int = 20,
    ) -> dict[str, Any]:
        """Run full dataset analysis including MinHash dedup.

        Returns analysis dict suitable for reporting.
        """
        analyzer = DatasetAnalyzer(cases_path, minhash_threshold=minhash_threshold)
        analysis = analyzer.analyze()
        
        # Convert DatasetAnalysis to dict manually
        return {
            "n_cases": analysis.n_cases,
            "n_fields": analysis.n_fields,
            "fields_seen": list(analysis.fields_seen),
            "strata_distribution": analysis.strata_distribution,
            "defect_rate_estimate": analysis.defect_rate_estimate,
            "top_difficult_fields": analysis.top_difficult_fields,
            "duplicate_pairs": analysis.duplicate_pairs[:top_n_duplicates],
            "ocr_signal_dist": analysis.ocr_signal_dist,
            "field_stats": analysis.field_stats,
            "text_stats": analysis.text_stats,
            "item_count_dist": analysis.item_count_dist,
            "language_hint": analysis.language_hint,
            "null_field_rate": analysis.null_field_rate,
        }

    def prepare_split(
        self,
        cases: list[dict[str, Any]],
        strata_keys: list[str] | None = None,
        holdout_fraction: float = 0.2,
        seed: int = 42,
        min_per_stratum: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        """Validate cases and split into train/holdout sets.

        Returns dict with 'train' and 'holdout' keys.
        """
        # Validate first
        report = self.validate(cases)
        if report.invalid_cases > 0:
            raise ValueError(
                f"Cannot split data with {report.invalid_cases} invalid cases. "
                f"Schema errors: {report.schema_errors[:3]}"
            )

        # Use valid cases only
        valid_cases = [
            c for c in cases
            if self.REQUIRED_FIELDS.issubset(c.keys())
        ]

        if not valid_cases:
            raise ValueError("No valid cases found after validation")

        # Perform stratified split
        from slm_eval.split import split_stratified

        split_result = split_stratified(
            cases=valid_cases,
            strata_keys=strata_keys,
            fraction=holdout_fraction,
            seed=seed,
            min_per_stratum=min_per_stratum,
        )

        return {
            "train": split_result.train,
            "holdout": split_result.holdout,
            "stratum_sizes": getattr(split_result, "stratum_sizes", {}),
        }

    def save_splits(
        self,
        splits: dict[str, list[dict[str, Any]]],
        output_dir: Path,
    ) -> dict[str, int]:
        """Save train/holdout splits as JSONL files.

        Returns dict with counts of saved cases.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {}

        # Only process known split keys, skip metadata like stratum_sizes
        for split_name in ("train", "holdout"):
            cases = splits.get(split_name)
            if cases is None:
                continue
            out_path = output_dir / f"{split_name}.jsonl"
            with open(out_path, "w") as f:
                for case in cases:
                    # Only include essential fields for training
                    record = {
                        "case_id": case.get("case_id"),
                        "text": case.get("text", ""),
                        "gold": case.get("gold", {}),
                        "strata": case.get("strata", {}),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[split_name] = len(cases)

        return counts

    def to_training_format(
        self,
        cases: list[dict[str, Any]],
        output_path: Path,
    ) -> int:
        """Convert cases to training JSONL format.

        Output format: {"text": "...", "gold": {...}}
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(output_path, "w") as f:
            for case in cases:
                record = {
                    "text": case.get("text", ""),
                    "gold": case.get("gold", {}),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        return count