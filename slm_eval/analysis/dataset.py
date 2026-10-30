"""DatasetAnalyzer — before-eval analysis of cases.jsonl.

Answers: Is this data ready? Any duplicates? Any likely defects?
What fields are hardest? Strata distribution?
What's the text quality and field coverage?
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from slm_eval.hashing import minhash_signature, minhash_similarity


class DatasetAnalysis:
    """Structured output from DatasetAnalyzer — JSON-serializable."""

    def __init__(
        self,
        n_cases: int,
        n_fields: int,
        fields_seen: set[str],
        strata_distribution: dict[str, int],
        defect_rate_estimate: float,
        top_difficult_fields: list[tuple[str, float]],
        duplicate_pairs: list[tuple[str, str, float]],
        ocr_signal_dist: dict[str, int],
        field_stats: dict[str, dict[str, Any]],
        text_stats: dict[str, Any],
        item_count_dist: dict[str, int],
        language_hint: str | None,
        null_field_rate: dict[str, float],
    ):
        self.n_cases = n_cases
        self.n_fields = n_fields
        self.fields_seen = fields_seen
        self.strata_distribution = strata_distribution
        self.defect_rate_estimate = defect_rate_estimate
        self.top_difficult_fields = top_difficult_fields  # (field_name, error_rate)
        self.duplicate_pairs = duplicate_pairs  # (case_a, case_b, similarity)
        self.ocr_signal_dist = ocr_signal_dist
        self.field_stats = field_stats
        self.text_stats = text_stats
        self.item_count_dist = item_count_dist
        self.language_hint = language_hint
        self.null_field_rate = null_field_rate


class DatasetAnalyzer:
    """Analyze a cases.jsonl dataset before running the full pipeline.

    Heuristics (tuned on CORD-v2 / Indian GST invoice extraction):
    - Defect rate: cases where ocr_signal is low/poor → likely unextractable
    - Difficulty: fraction of cases WITH the field in gold; lower coverage → harder
    - MinHash dedup at threshold 0.6 for near-duplicate text pairs
    - Text quality statistics
    - Item count distribution
    """

    def __init__(self, cases_path: str | Path, minhash_threshold: float = 0.6):
        self.cases_path = Path(cases_path)
        self.minhash_threshold = minhash_threshold
        self._cases: list[dict[str, Any]] | None = None

    def _load_cases(self) -> list[dict[str, Any]]:
        if self._cases is None:
            self._cases = []
            with self.cases_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self._cases.append(json.loads(line))
        return self._cases

    def analyze(self) -> DatasetAnalysis:
        cases = self._load_cases()
        n = len(cases)

        # --- Basic counts ---
        fields_seen: set[str] = set()
        strata_dist: Counter = Counter()
        ocr_counts: Counter = Counter()
        field_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "count": 0,
                "distinct_values": set(),
                "null_rate": 0.0,
                "example_values": [],
            }
        )
        text_lengths: list[int] = []
        total_text_chars: int = 0
        language_hint: str | None = None
        item_count_dist: Dict[str, int] = {}

        for c in cases:
            # Strata analysis
            strata = c.get("strata", {})
            if isinstance(strata, dict):
                key = ",".join(f"{k}={v}" for k, v in sorted(strata.items()))
                strata_dist[key] += 1
                ocr = strata.get("ocr_signal")
                if ocr is not None:
                    ocr_counts[str(ocr)] += 1

            # Field analysis
            gold = c.get("gold", {})
            if isinstance(gold, dict):
                for field in gold:
                    fields_seen.add(field)
                    fs = field_stats[field]
                    fs["count"] += 1
                    val = gold[field]
                    if isinstance(val, (str, int, float)):
                        val_str = str(val)
                        if len(fs["distinct_values"]) < 20:
                            fs["distinct_values"].add(val_str)
                            fs["example_values"].append(val_str)
                    else:
                        # List/dict fields - count items
                        if isinstance(val, list) and len(val) > 0:
                            fs["distinct_values"].add(f"list[{len(val)}]")
                            if len(fs["example_values"]) < 5:
                                fs["example_values"].append(
                                    str(val)[:100] if hasattr(val, "__len__") else str(val)
                                )

            # Text analysis
            text = c.get("text", "") or ""
            text_lengths.append(len(text))
            total_text_chars += len(text)

            # Language hint
            lang = self._detect_language(text)
            if lang and not language_hint:
                language_hint = lang

            # Item count distribution
            ic = strata.get("item_count", "unknown") if isinstance(strata, dict) else "unknown"
            item_count_dist[ic] = item_count_dist.get(ic, 0) + 1

        # --- MinHash dedup (near-duplicate detection) ---
        texts = [c.get("text", "") or "" for c in cases]
        signatures = [
            minhash_signature(t, k=5) if len(t) >= 10 else tuple([0] * 128) for t in texts
        ]
        duplicate_pairs: list[tuple[str, str, float]] = []

        if n > 1:
            case_ids = [c.get("case_id", "unknown") for c in cases]
            for i in range(n):
                for j in range(i + 1, n):
                    # Too-short texts get an all-zero signature, so skip them
                    # (they would otherwise all look identical to each other).
                    if len(texts[i]) < 10 or len(texts[j]) < 10:
                        continue
                    sim = minhash_similarity(signatures[i], signatures[j])
                    if sim >= self.minhash_threshold:
                        duplicate_pairs.append((case_ids[i], case_ids[j], round(sim, 4)))

        # --- Per-field analysis ---
        for field in fields_seen:
            fs = field_stats[field]
            # Calculate null rate: cases where field is missing from gold
            field_present = sum(1 for c in cases if field in c.get("gold", {}))
            fs["null_rate"] = round(1.0 - (field_present / n), 4) if n > 0 else 0.0
            # Calculate distinct value count
            fs["distinct_value_count"] = len(fs["distinct_values"])
            # Keep example values but convert sets to lists
            fs["example_values"] = list(fs["example_values"])[:10]

        # Calculate text statistics
        text_stats = {
            "mean_length": round(sum(text_lengths) / len(text_lengths), 1) if text_lengths else 0.0,
            "median_length": sorted(text_lengths)[len(text_lengths) // 2] if text_lengths else 0,
            "min_length": min(text_lengths) if text_lengths else 0,
            "max_length": max(text_lengths) if text_lengths else 0,
            "total_chars": total_text_chars,
            "median_fraction": round(sum(text_lengths) / len(text_lengths) / max(total_text_chars / max(len(text_lengths), 1), 1), 4)
            if text_lengths and total_text_chars > 0
            else 0.0,
        }

        # --- Defect rate estimate ---
        poor_ocr = ocr_counts.get("poor", 0) + ocr_counts.get("low", 0) + ocr_counts.get("bad", 0)
        defect_rate = round(poor_ocr / n, 4) if n else 0.0

        # --- Top difficult fields (lowest coverage first) ---
        field_frac: dict[str, float] = {}
        for field in fields_seen:
            count = sum(1 for c in cases if field in c.get("gold", {}))
            field_frac[field] = round(count / n, 4) if n else 0.0

        top_difficult: list[tuple[str, float]] = sorted(
            [(f, 1.0 - frac) for f, frac in field_frac.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        # --- Null field rate per field ---
        null_field_rate: dict[str, float] = {}
        for field in fields_seen:
            present = sum(1 for c in cases if field in c.get("gold", {}))
            null_field_rate[field] = round(1.0 - (present / n), 4) if n else 0.0

        return DatasetAnalysis(
            n_cases=n,
            n_fields=len(fields_seen),
            fields_seen=fields_seen,
            strata_distribution={k: v for k, v in strata_dist.items()},
            defect_rate_estimate=defect_rate,
            top_difficult_fields=top_difficult,
            duplicate_pairs=duplicate_pairs[:20],
            ocr_signal_dist=dict(ocr_counts),
            field_stats={k: {kk: (vv if not isinstance(vv, set) else list(vv)[:10]) for kk, vv in v.items()} for k, v in field_stats.items()},
            text_stats=text_stats,
            item_count_dist=item_count_dist,
            language_hint=language_hint,
            null_field_rate=null_field_rate,
        )

    @staticmethod
    def _detect_language(text: str) -> str | None:
        """Very basic language detection for heuristic purposes."""
        if not text or len(text) < 3:
            return None

        # Check for language-specific patterns
        text_lower = text.lower()

        # Indonesian markers
        if any(word in text_lower for word in ["kopi", "teh", "beras", "gula", "ikan"]):
            return "Indonesian"

        # Arabic numerals + Arabic script markers
        if any(char in text for char in "١٢٣٤٥٦٧٨٩٠"):
            if any(char in text for char in "ابتثجح خدغ عפ"):
                return "Arabic"
            # Could be English/Arabic mixed with numbers

        # Simple English heuristic
        if text.isascii() and any(c.isalpha() for c in text):
            # More English-like if common English words present
            english_markers = ["the", "and", "for", "are", "but", "not", "you", "had", "but", "from"]
            if any(word in text_lower for word in english_markers):
                return "English"

        # Chinese characters
        if any("\u4e00" <= c <= "\u9fff" for c in text):
            return "Chinese"

        # No strong signal
        return None


def quick_dataset_summary(cases_path: str | Path) -> str:
    """Quick human-readable summary of dataset characteristics."""
    analyzer = DatasetAnalyzer(cases_path)
    a = analyzer.analyze()

    lines = [
        f"DATASET SUMMARY: {analyzer.cases_path}",
        f"  Total cases: {a.n_cases}",
        f"  Fields: {sorted(a.fields_seen)} ({a.n_fields})",
        f"  Strata: {len(a.strata_distribution)} distinct combinations",
        f"  OCR signal distribution: {dict(a.ocr_signal_dist)}",
        f"  Defect rate estimate: {a.defect_rate_estimate}",
        f"  Top difficult fields: {a.top_difficult_fields[:5]}",
        f"  Duplicate pairs: {len(a.duplicate_pairs)}",
        f"  Language hint: {a.language_hint or 'undetermined'}",
        f"  Text stats: mean={a.text_stats['mean_length']} chars, "
        f"min={a.text_stats['min_length']}, max={a.text_stats['max_length']}",
        f"  Item count distribution: {dict(a.item_count_dist)}",
    ]

    # Add field stats highlights
    lines.append("  Field coverage:")
    for field in sorted(a.fields_seen)[:5]:
        fs = a.field_stats.get(field, {})
        null_pct = fs.get("null_rate", 0) * 100
        lines.append(f"    • {field}: present in {fs.get('count', 0)} cases, "
                      f"null rate {null_pct:.1f}%, "
                      f"{len(fs.get('distinct_values', set()))} distinct values")

    if len(a.fields_seen) > 5:
        lines.append(f"    ... and {len(a.fields_seen) - 5} more fields")

    return "\n".join(lines)


# --- CLI ---
if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(description="SLM Dataset Analysis Tool")
    p.add_argument("cases_path", help="Path to cases.jsonl")
    p.add_argument("--format", choices=["text", "json"], default="text",
                   help="Output format")
    p.add_argument("--field", default=None,
                   help="Show detailed stats for specific field")

    args = p.parse_args()

    analyzer = DatasetAnalyzer(args.cases_path)
    a = analyzer.analyze()

    if args.format == "json":
        # Convert sets to lists for JSON serialization
        field_stats_json = {}
        for k, v in a.field_stats.items():
            field_stats_json[k] = {
                "count": v["count"],
                "null_rate": v["null_rate"],
                "distinct_value_count": v.get("distinct_value_count", 0),
                "example_values": v.get("example_values", []),
            }
        output = {
            "n_cases": a.n_cases,
            "n_fields": a.n_fields,
            "fields_seen": sorted(a.fields_seen),
            "strata_distribution": a.strata_distribution,
            "defect_rate_estimate": a.defect_rate_estimate,
            "top_difficult_fields": a.top_difficult_fields,
            "duplicate_pairs": a.duplicate_pairs,
            "ocr_signal_distribution": a.ocr_signal_dist,
            "field_stats": field_stats_json,
            "text_stats": a.text_stats,
            "item_count_distribution": a.item_count_dist,
            "language_hint": a.language_hint,
            "null_field_rate": a.null_field_rate,
        }
        print(json.dumps(output, indent=2))
    else:
        print(quick_dataset_summary(args.cases_path))
        if args.field and args.field in a.field_stats:
            fs = a.field_stats[args.field]
            print(f"\n  Field '{args.field}' detailed stats:")
            print(f"    Present in: {fs['count']}/{a.n_cases} cases")
            print(f"    Null rate: {fs['null_rate']*100:.1f}%")
            print(f"    Distinct values: {fs.get('distinct_value_count', 0)}")
            if fs.get("example_values"):
                print(f"    Example values: {fs['example_values'][:5]}")