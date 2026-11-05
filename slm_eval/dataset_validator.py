"""Dataset validator agent: quality checks, dedup, cleaning."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from slm_eval.split import NEAR_DUP_THRESHOLD


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


class DatasetValidator:
    """Validate, score, and clean a dataset of cases."""

    # Required fields every case must have
    REQUIRED_FIELDS = {"case_id", "text", "gold", "strata"}

    # Optional fields that may appear
    OPTIONAL_FIELDS = {"images", "metadata"}

    def __init__(self, dedup_threshold: float = NEAR_DUP_THRESHOLD):
        self.dedup_threshold = dedup_threshold

    def _check_fields(self, case: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Return (missing, unexpected) field lists."""
        missing = []
        for f in self.REQUIRED_FIELDS:
            if f not in case or case[f] is None:
                missing.append(f)
        unexpected = [k for k in case if k not in self.REQUIRED_FIELDS | self.OPTIONAL_FIELDS]
        return missing, unexpected

    def _check_text(self, case: dict[str, Any]) -> list[str]:
        """Flag cases with blank / degraded text."""
        errors = []
        text = case.get("text", "")
        if not text or not text.strip():
            errors.append("empty_text")
        elif len(text.strip()) < 1:
            errors.append("text_too_short")
        return errors

    def _check_gold(self, case: dict[str, Any]) -> list[str]:
        """Validate gold annotations."""
        errors = []
        gold = case.get("gold")
        if gold is None:
            errors.append("missing_gold")
            return errors
        # Check that gold has some content
        if isinstance(gold, dict):
            if not gold:
                errors.append("empty_gold_dict")
        return errors

    def _check_strata(self, case: dict[str, Any]) -> list[str]:
        """Validate strata."""
        errors = []
        strata = case.get("strata")
        if strata is None:
            errors.append("missing_strata")
            return errors
        if isinstance(strata, dict):
            # Common keys
            for k in ["quality", "ocr_signal", "layout_type"]:
                if k in strata:
                    v = strata[k]
                    if v is not None and v not in ("high", "medium", "low", "usable", "degraded", "blank", "rotated_90", "clean"):
                        errors.append(f"unexpected_stratum_value:{k}={v}")
        return errors

    def _dedup_check(self, cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        """Remove exact + near-duplicate texts. Returns (kept, removed, count).
        
        Hybrid approach:
        1. Exact text dedup (very fast, O(n))
        2. Fast near-dup using prefix hashing + Jaccard on word sets (O(n) with small constant)
        """
        if not cases:
            return [], [], 0

        # Step 1: Exact text dedup
        seen_texts: set[str] = set()
        exact_kept: list[dict[str, Any]] = []
        exact_removed: list[dict[str, Any]] = []

        for case in cases:
            text = case.get("text", "")
            if text in seen_texts:
                exact_removed.append(case)
            else:
                seen_texts.add(text)
                exact_kept.append(case)

        # Step 2: Fast near-dup on unique texts
        # Use prefix hashing + Jaccard similarity on word sets
        unique_texts = [c.get("text", "") for c in exact_kept]

        # Near-dup threshold: use same as split leakage check for consistency
        jaccard_threshold = self.dedup_threshold

        buckets: dict[int, list[str]] = {}
        num_buckets = 10000
        prefix_len = 50
        max_words = 100

        kept_texts: set[str] = set()
        near_removed_count = 0

        for text in unique_texts:
            prefix = text[:prefix_len]
            bucket = hash(prefix) % num_buckets

            is_dup = False
            if bucket in buckets:
                # Compute Jaccard on word sets
                words1 = set(text.lower().split()[:max_words])
                if not words1:
                    continue
                for existing in buckets[bucket]:
                    words2 = set(existing.lower().split()[:max_words])
                    if not words2:
                        continue
                    intersection = len(words1 & words2)
                    union = len(words1 | words2)
                    if union > 0:
                        jaccard = intersection / union
                        if jaccard >= jaccard_threshold:
                            is_dup = True
                            break

            if is_dup:
                near_removed_count += 1
            else:
                kept_texts.add(text)
                if bucket not in buckets:
                    buckets[bucket] = []
                buckets[bucket].append(text)

        # Reconstruct kept cases
        final_kept = [c for c in exact_kept if c.get("text", "") in kept_texts]
        final_removed = exact_removed

        # Add near-dup removed cases
        removed_texts = set(unique_texts) - kept_texts
        for case in exact_kept:
            if case.get("text", "") in removed_texts:
                final_removed.append(case)

        return final_kept, final_removed, len(final_removed)

    def validate(self, cases: list[dict[str, Any]]) -> tuple[DataQualityReport, list[dict[str, Any]], list[dict[str, Any]]]:
        """Run all validation checks.

        Returns:
            report: DataQualityReport with counts & errors
            good_cases: cases that pass all checks
            bad_cases: cases that failed (with error notes attached)
        """
        report = DataQualityReport()
        good_cases: list[dict[str, Any]] = []
        bad_cases: list[dict[str, Any]] = []

        # Step 1: field presence
        for case in cases:
            report.total_cases += 1
            missing, unexpected = self._check_fields(case)
            if missing:
                report.missing_fields.setdefault(", ".join(missing), 0)
                report.missing_fields[", ".join(missing)] += 1
                bad_cases.append({**case, "_error": f"missing_fields:{','.join(missing)}"})
                continue

            # Step 2: text health
            text_errors = self._check_text(case)
            # Step 3: gold validation
            gold_errors = self._check_gold(case)
            # Step 4: strata validation
            strata_errors = self._check_strata(case)

            all_errors = text_errors + gold_errors + strata_errors
            if all_errors:
                bad_cases.append({**case, "_error": "; ".join(all_errors)})
                continue

            # Step 5: dedup
            good_cases.append(case)

        # Step 6: final dedup across collected good cases
        final_kept, final_removed, final_dup_count = self._dedup_check(good_cases)
        report.invalid_cases = report.total_cases - len(final_kept)
        report.valid_cases = len(final_kept)
        report.recommendations.append(f"Dedup removed {final_dup_count} near-duplicate cases")

        # Quality score: simple heuristic based on checks passed
        if report.total_cases > 0:
            report.quality_score = report.valid_cases / report.total_cases

        # Re-count missing from bad_cases
        missing_counts: dict[str, int] = Counter()
        for bad in bad_cases:
            err = bad.get("_error", "")
            for pattern, count in report.missing_fields.items():
                if pattern in err:
                    missing_counts[pattern] += 1
        report.missing_fields = dict(missing_counts)

        # Remove error tags from good cases
        clean_good = [c for c in final_kept if "_error" not in c]
        # Clean bad_cases
        clean_bad = []
        for bad in bad_cases:
            bad_copy = {k: v for k, v in bad.items() if not k.startswith("_")}
            if "_error" in bad:
                bad_copy["_error"] = bad["_error"]
            clean_bad.append(bad_copy)

        return report, clean_good, clean_bad
