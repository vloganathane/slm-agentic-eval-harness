"""Unified dataset builder: download → validate → dedup → split → export → manifest.

This merges the best of both pipelines:
- data_pipeline.py: analysis, validation, stratified splitting
- dataset_builder.py: downloaders, exporters, CLI integration
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from slm_eval.hashing import sha256_json
from slm_eval.pipelines.dedup import DEFAULT_JACCARD_THRESHOLD, check_leakage, deduplicate_cases
from slm_eval.pipelines.docling_pipeline import (
    DoclingPipeline,
    integrate_with_pipeline,
    process_directory,
)
from slm_eval.pipelines.exporters import DEFAULT_TEMPLATES, get_exporter, list_formats
from slm_eval.pipelines.quality import DataQualityReport, save_report
from slm_eval.providers.data_pull import AutoDownloader
from slm_eval.split import NEAR_DUP_THRESHOLD, split_stratified


@dataclass
class BuildConfig:
    """Configuration for dataset building."""
    source: str
    out_dir: Path
    fmt: str = "sft"
    split_keys: list[str] = field(default_factory=lambda: ["quality"])
    split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)  # train, valid, test
    seed: int = 42
    validate: bool = True
    dedup: bool = True
    score: bool = True
    template: str | None = None
    custom_template: dict[str, Any] | None = None
    min_cases_per_stratum: int = 1
    skip_download: bool = False
    resume: bool = False
    progress: bool = True
    # Docling options
    use_docling: bool = False
    docling_output_dir: Path | None = None
    docling_formats: list[str] = field(default_factory=lambda: ["pdf", "docx", "pptx", "xlsx", "html", "epub", "jpg", "jpeg", "png", "tiff"])
    docling_include_subdirs: bool = True


@dataclass
class BuildResult:
    """Result of dataset build."""
    out_dir: Path
    train_count: int
    valid_count: int
    test_count: int
    total_input: int
    dedup_removed: int
    validation_report: DataQualityReport
    manifest_path: Path
    quality_report_path: Path | None = None


def load_cases(source: str, raw_dir: Path) -> list[dict[str, Any]]:
    """Load cases from various sources (local files, HF, Kaggle, URLs)."""
    # First try local path
    local_path = Path(source)
    if local_path.exists():
        return _load_from_local(local_path)

    # Parse as URL or HF/Kaggle identifier
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        return _load_from_url(source, raw_dir)
    elif "/" in source and not parsed.scheme:
        # Could be HF dataset (org/dataset) or Kaggle (org/dataset)
        return _load_from_hf_or_kaggle(source, raw_dir)
    else:
        # Shorthand HF dataset name
        return _load_from_hf_or_kaggle(source, raw_dir)


def _load_from_local(path: Path) -> list[dict[str, Any]]:
    """Load cases from local file or directory."""
    cases = []

    if path.is_file():
        if path.suffix == ".jsonl":
            cases = _load_jsonl(path)
        elif path.suffix == ".json":
            cases = _load_json(path)
        elif path.suffix in (".parquet", ".pq"):
            cases = _load_parquet([path])
        elif path.suffix == ".csv":
            cases = _load_csv([path])
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    elif path.is_dir():
        # Look for standard files
        jsonl_files = list(path.glob("*.jsonl"))
        if jsonl_files:
            for f in jsonl_files:
                cases.extend(_load_jsonl(f))
        else:
            parquet_files = list(path.glob("*.parquet"))
            if parquet_files:
                cases = _load_parquet(parquet_files)
            else:
                csv_files = list(path.glob("*.csv"))
                if csv_files:
                    cases = _load_csv(csv_files)
                else:
                    json_files = list(path.glob("*.json"))
                    if len(json_files) == 1:
                        cases = _load_json(json_files[0])
                    else:
                        raise ValueError(f"No supported files found in {path}")
    else:
        raise FileNotFoundError(f"Source not found: {path}")

    return _normalize_cases(cases)


def _load_from_url(url: str, raw_dir: Path) -> list[dict[str, Any]]:
    """Download and load from URL."""
    import urllib.request

    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = urlparse(url).path.split("/")[-1] or "download"
    dest = raw_dir / filename

    if not dest.exists():
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, dest)

    return _load_from_local(dest)


def _load_from_hf_or_kaggle(source: str, raw_dir: Path) -> list[dict[str, Any]]:
    """Download from HF or Kaggle using AutoDownloader."""
    downloader = AutoDownloader()
    downloaded_dir = downloader.download(source, raw_dir)
    return _load_from_local(downloaded_dir)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def _load_parquet(parquet_files: list[Path]) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas required for parquet. Run: pip install pandas")

    cases = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        cases.extend(_df_to_cases(df))
    return cases


def _load_csv(csv_files: list[Path]) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas required for CSV. Run: pip install pandas")

    cases = []
    for cf in csv_files:
        df = pd.read_csv(cf)
        cases.extend(_df_to_cases(df))
    return cases


def _df_to_cases(df) -> list[dict[str, Any]]:
    """Convert DataFrame rows to standard case format."""
    cases = []
    for _, row in df.iterrows():
        case = {}

        # case_id
        if "case_id" in df.columns:
            case["case_id"] = str(row.get("case_id", f"auto_{len(cases)}"))
        elif "id" in df.columns:
            case["case_id"] = str(row.get("id", f"auto_{len(cases)}"))
        else:
            case["case_id"] = f"auto_{len(cases)}"

        # Text field - try common column names
        text_val = ""
        for col in ["text", "content", "sentence", "document", "input", "prompt", "question", "context"]:
            if col in df.columns:
                text_val = str(row.get(col, ""))
                break
        if not text_val:
            # Try label/target fields
            for col in ["label", "target", "class", "answer", "response", "output"]:
                if col in df.columns:
                    text_val = str(row.get(col, ""))
                    break
        case["text"] = text_val

        # Gold - try to extract structured gold or use all other columns
        if "gold" in df.columns:
            case["gold"] = row.get("gold", {})
        else:
            # Build gold from non-standard columns
            exclude = {"case_id", "id", "text", "content", "strata", "sentence", "document",
                       "input", "prompt", "question", "context", "label", "target", "class",
                       "answer", "response", "output", "images"}
            gold = {k: v for k, v in row.items() if k not in exclude}
            case["gold"] = gold

        # Strata - provide defaults if not present
        if "strata" in df.columns:
            strata_val = row.get("strata")
            if isinstance(strata_val, str):
                try:
                    case["strata"] = json.loads(strata_val)
                except json.JSONDecodeError:
                    case["strata"] = {"domain": strata_val}
            elif isinstance(strata_val, dict):
                case["strata"] = strata_val
            else:
                case["strata"] = _default_strata()
        else:
            case["strata"] = _default_strata()

        # Images if present
        if "images" in df.columns:
            case["images"] = row.get("images", [])

        cases.append(case)
    return cases


def _default_strata() -> dict[str, str]:
    return {"quality": "high", "ocr_signal": "usable", "layout": "clean", "layout_type": "clean"}


def _normalize_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure all cases have required fields with sensible defaults."""
    normalized = []
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            continue

        # case_id
        if "case_id" not in case or not case["case_id"]:
            case["case_id"] = f"auto_{i}"

        # text
        if "text" not in case:
            case["text"] = ""

        # gold
        if "gold" not in case:
            case["gold"] = {}

        # strata
        if "strata" not in case:
            case["strata"] = _default_strata()

        normalized.append(case)
    return normalized


def validate_cases(
    cases: list[dict[str, Any]],
    dedup_threshold: float = DEFAULT_JACCARD_THRESHOLD,
) -> tuple[DataQualityReport, list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate cases and optionally deduplicate.

    Returns (report, good_cases, bad_cases)
    """
    report = DataQualityReport(total_cases=len(cases))
    required_fields = {"case_id", "text", "gold", "strata"}

    good_cases = []
    bad_cases = []
    seen_ids = set()

    for case in cases:
        case_id = case.get("case_id", f"unknown_{len(good_cases) + len(bad_cases)}")
        errors = []

        # Check required fields
        missing = required_fields - set(case.keys())
        if missing:
            errors.append(f"missing required fields: {missing}")

        # Check case_id
        if case_id in seen_ids:
            errors.append(f"duplicate case_id: {case_id}")
        else:
            seen_ids.add(case_id)

        # Check text
        text = case.get("text")
        if not isinstance(text, str):
            errors.append(f"text must be str, got {type(text).__name__}")

        # Check gold
        gold = case.get("gold")
        if gold is not None and not isinstance(gold, dict):
            errors.append(f"gold must be dict, got {type(gold).__name__}")

        # Check strata
        strata = case.get("strata")
        if strata is not None and not isinstance(strata, dict):
            errors.append(f"strata must be dict, got {type(strata).__name__}")

        if errors:
            report.add_error(case_id, errors)
            bad_cases.append(case)
        else:
            report.add_valid()
            good_cases.append(case)

    # Deduplicate if requested and we have good cases
    if good_cases:
        result = deduplicate_cases(good_cases, jaccard_threshold=dedup_threshold)
        report.dedup_removed = result.total_removed
        report.near_duplicate_pairs = result.duplicate_pairs
        if result.total_removed > 0:
            report.recommendations.append(
                f"Dedup removed {result.exact_duplicates} exact + {result.near_duplicates} near-duplicate cases"
            )
        good_cases = result.unique_cases

    report.finalize(good_cases)
    return report, good_cases, bad_cases


def split_cases(
    cases: list[dict[str, Any]],
    split_keys: list[str],
    split_fractions: tuple[float, float, float],
    seed: int,
    min_per_stratum: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    """Split cases into train/valid/test with stratification.

    Returns dict with 'train', 'valid', 'test' keys.
    """
    if len(cases) < 3:
        # Too small for 3-way split, just use train/valid
        train_frac = split_fractions[0]
        s = split_stratified(cases, strata_keys=split_keys, fraction=1-train_frac, seed=seed, min_per_stratum=min_per_stratum)
        return {"train": s.train, "valid": s.holdout, "test": []}

    # First split: train vs (valid+test)
    train_frac = split_fractions[0]
    s1 = split_stratified(cases, strata_keys=split_keys, fraction=1-train_frac, seed=seed, min_per_stratum=min_per_stratum)

    # Second split: valid vs test from holdout
    holdout = s1.holdout
    valid_frac = split_fractions[1] / (split_fractions[1] + split_fractions[2])
    s2 = split_stratified(holdout, strata_keys=split_keys, fraction=1-valid_frac, seed=seed+1, min_per_stratum=min_per_stratum)

    # Check for leakage between all splits
    train_ids = {c["case_id"] for c in s1.train}
    valid_ids = {c["case_id"] for c in s2.train}
    test_ids = {c["case_id"] for c in s2.holdout}

    if train_ids & valid_ids:
        raise ValueError("Leakage: train/valid overlap")
    if train_ids & test_ids:
        raise ValueError("Leakage: train/test overlap")
    if valid_ids & test_ids:
        raise ValueError("Leakage: valid/test overlap")

    # Also check near-duplicate leakage
    train_cases = s1.train
    valid_cases = s2.train
    test_cases = s2.holdout

    if train_cases and valid_cases:
        leaks = check_leakage(train_cases, valid_cases)
        if leaks:
            raise ValueError(f"Near-duplicate leakage train→valid: {leaks[:3]}")

    if train_cases and test_cases:
        leaks = check_leakage(train_cases, test_cases)
        if leaks:
            raise ValueError(f"Near-duplicate leakage train→test: {leaks[:3]}")

    if valid_cases and test_cases:
        leaks = check_leakage(valid_cases, test_cases)
        if leaks:
            raise ValueError(f"Near-duplicate leakage valid→test: {leaks[:3]}")

    return {"train": s1.train, "valid": s2.train, "test": s2.holdout}


def export_split(
    cases: list[dict[str, Any]],
    out_dir: Path,
    fmt: str,
    split_name: str,
    template: dict[str, Any] | None = None,
) -> int:
    """Export a single split to the specified format."""
    out_dir.mkdir(parents=True, exist_ok=True)
    exporter = get_exporter(fmt, template)
    out_file = out_dir / f"{split_name}.jsonl"

    count = 0
    with out_file.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(exporter.format(case) + "\n")
            count += 1

    print(f"  Exported {count} cases to {out_file}")
    return count


def generate_quality_report(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate comprehensive quality report."""
    total = len(cases)
    if total == 0:
        return {"total_cases": 0}

    # Basic stats
    text_lengths = [len(c.get("text", "")) for c in cases if c.get("text")]
    gold_fields: dict[str, int] = {}
    strata_dist: dict[str, dict[str, int]] = {}

    for case in cases:
        gold = case.get("gold", {})
        if isinstance(gold, dict):
            for field in gold:
                gold_fields[field] = gold_fields.get(field, 0) + 1

        strata = case.get("strata", {})
        if isinstance(strata, dict):
            for key, value in strata.items():
                if key not in strata_dist:
                    strata_dist[key] = {}
                strata_dist[key][value] = strata_dist[key].get(value, 0) + 1

    return {
        "total_cases": total,
        "text_length": {
            "min": min(text_lengths) if text_lengths else 0,
            "max": max(text_lengths) if text_lengths else 0,
            "mean": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            "median": sorted(text_lengths)[len(text_lengths) // 2] if text_lengths else 0,
        },
        "gold_field_coverage": {f: c / total for f, c in gold_fields.items()},
        "strata_distribution": strata_dist,
        "cases_with_text": sum(1 for c in cases if c.get("text", "").strip()),
        "cases_with_gold": sum(1 for c in cases if c.get("gold")),
        "cases_with_images": sum(1 for c in cases if c.get("images")),
    }


def generate_manifest(
    out_dir: Path,
    source_hash: str,
    split_seed: int,
    split_strategy: str,
    quality_dist: dict[str, Any],
    case_count: int,
    split_fractions: tuple[float, float, float],
    build_time: float,
) -> Path:
    """Write a JSON manifest summarizing the dataset."""
    manifest = {
        "source_hash": source_hash,
        "split_seed": split_seed,
        "split_strategy": split_strategy,
        "split_fractions": {
            "train": split_fractions[0],
            "valid": split_fractions[1],
            "test": split_fractions[2],
        },
        "quality_distribution": quality_dist,
        "case_count": case_count,
        "out_dir": str(out_dir),
        "build_time_seconds": build_time,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def build_dataset(config: BuildConfig) -> BuildResult:
    """Main entry point: build a clean dataset from raw source."""
    start_time = time.time()

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Download / load raw cases
    if config.progress:
        print(f"[1/6] Loading cases from {config.source}...")

    if config.skip_download:
        cases = _load_from_local(Path(config.source))
    else:
        cases = load_cases(config.source, raw_dir)

    if config.progress:
        print(f"      Loaded {len(cases)} raw cases")

    # Docling preprocessing (optional)
    docling_results = None
    docling_processed_cases = []
    if config.use_docling:
        if config.progress:
            print("[1.5/6] Processing documents with docling...")

        docling_output_dir = config.docling_output_dir or (out_dir / "docling_output")
        docling_output_dir.mkdir(parents=True, exist_ok=True)

        # Try to process source directory if it's a directory
        source_path = Path(config.source) if not config.skip_download else Path(config.source)
        if source_path.exists() and source_path.is_dir():
            try:
                pipeline = DoclingPipeline()
                docling_results, files_processed, files_failed = process_directory(
                    source_path,
                    docling_output_dir,
                    format="jsonl",
                    include_subdirs=config.docling_include_subdirs,
                )
                if config.progress:
                    print(f"      Processed {files_processed} documents, {files_failed} failed")

                # Integrate docling results with cases
                docling_processed_cases = integrate_with_pipeline(cases, docling_results)
            except Exception as e:
                if config.progress:
                    print(f"      ⚠️  Docling processing failed: {e}")
                docling_results = None
        else:
            if config.progress:
                print("      ⚠️  Docling requires a directory source. Skipping docling processing.")

    # Use docling-processed cases if available
    cases_to_use = docling_processed_cases if docling_processed_cases else cases

    # 2️⃣ Validate & dedup
    validation_report = DataQualityReport()
    bad_cases = []

    if config.validate or config.dedup:
        if config.progress:
            print("[2/6] Validating and deduplicating...")
        validation_report, cases_to_use, bad_cases = validate_cases(cases_to_use, dedup_threshold=NEAR_DUP_THRESHOLD)
        if config.progress:
            print(f"      Valid: {validation_report.valid_cases}, Invalid: {validation_report.invalid_cases}, Dedup removed: {validation_report.dedup_removed}")

        # Save reports
        save_report(validation_report, out_dir / "validation_report.json")
        if bad_cases:
            (out_dir / "bad_cases.jsonl").write_text(
                "\n".join(json.dumps(c) for c in bad_cases)
            )

    # Save docling results if processed
    if docling_results:
        docling_path = out_dir / "docling_results.jsonl"
        with docling_path.open("w", encoding="utf-8") as f:
            for result in docling_results:
                f.write(json.dumps(result) + "\n")

    # 3️⃣ Split (stratified 3-way)
    if config.progress:
        print(f"[3/6] Splitting (stratified by {config.split_keys})...")

    split = split_cases(
        cases_to_use,
        split_keys=config.split_keys,
        split_fractions=config.split_fractions,
        seed=config.seed,
        min_per_stratum=config.min_cases_per_stratum,
    )

    if config.progress:
        print(f"      Train: {len(split['train'])}, Valid: {len(split['valid'])}, Test: {len(split['test'])}")

    # 4️⃣ Export each split
    if config.progress:
        print(f"[4/6] Exporting to {config.fmt} format...")

    # Resolve template
    template = None
    if config.template and config.template in DEFAULT_TEMPLATES:
        template = DEFAULT_TEMPLATES[config.template]
    elif config.custom_template:
        template = config.custom_template

    for split_name, split_cases_list in split.items():
        if split_cases_list:
            split_dir = out_dir / split_name
            export_split(split_cases_list, split_dir, config.fmt, split_name, template)

    # 5️⃣ Quality score
    quality_report_path = None
    if config.score:
        if config.progress:
            print("[5/6] Generating quality report...")
        quality = generate_quality_report(cases_to_use)
        quality_report_path = out_dir / "quality_report.json"
        quality_report_path.write_text(json.dumps(quality, indent=2))

    # 6️⃣ Manifest
    if config.progress:
        print("[6/6] Writing manifest...")

    source_hash = sha256_json(cases_to_use)
    quality_dist = generate_quality_report(cases_to_use)
    manifest_path = generate_manifest(
        out_dir,
        source_hash,
        config.seed,
        ",".join(config.split_keys),
        quality_dist,
        len(cases_to_use),
        config.split_fractions,
        time.time() - start_time,
    )

    if config.progress:
        print(f"[DONE] Dataset written to {out_dir} in {time.time() - start_time:.1f}s")

    return BuildResult(
        out_dir=out_dir,
        train_count=len(split["train"]),
        valid_count=len(split["valid"]),
        test_count=len(split["test"]),
        total_input=len(cases_to_use),
        dedup_removed=validation_report.dedup_removed,
        validation_report=validation_report,
        manifest_path=manifest_path,
        quality_report_path=quality_report_path,
    )


# CLI entry point
def main():
    parser = argparse.ArgumentParser(description="Build a clean, fine-tuning-ready dataset from raw sources.")
    parser.add_argument("--source", required=True, help="Source: local path, HF dataset (org/name), Kaggle (org/name), or URL")
    parser.add_argument("--out", required=True, help="Output directory for the clean dataset")
    parser.add_argument("--format", choices=list_formats(), default="sft", help="Export format")
    parser.add_argument("--template", choices=list(DEFAULT_TEMPLATES.keys()), help="Predefined template for the domain")
    parser.add_argument("--split-keys", nargs="+", default=["quality"], help="Strata keys for stratified split")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Train fraction")
    parser.add_argument("--valid-frac", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Test fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--validate", action="store_true", default=True, help="Run validation")
    parser.add_argument("--no-validate", action="store_false", dest="validate", help="Skip validation")
    parser.add_argument("--dedup", action="store_true", default=True, help="Enable deduplication")
    parser.add_argument("--no-dedup", action="store_false", dest="dedup", help="Disable deduplication")
    parser.add_argument("--score", action="store_true", default=True, help="Generate quality report")
    parser.add_argument("--no-score", action="store_false", dest="score", help="Skip quality report")
    parser.add_argument("--min-per-stratum", type=int, default=1, help="Minimum cases per stratum")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, use local source directly")
    parser.add_argument("--no-progress", action="store_false", dest="progress", help="Disable progress output")

    args = parser.parse_args()

    # Validate fractions sum to ~1
    total_frac = args.train_frac + args.valid_frac + args.test_frac
    if abs(total_frac - 1.0) > 0.01:
        print(f"Warning: split fractions sum to {total_frac:.2f}, not 1.0", file=sys.stderr)

    config = BuildConfig(
        source=args.source,
        out_dir=Path(args.out),
        fmt=args.format,
        split_keys=args.split_keys,
        split_fractions=(args.train_frac, args.valid_frac, args.test_frac),
        seed=args.seed,
        validate=args.validate,
        dedup=args.dedup,
        score=args.score,
        template=args.template,
        min_cases_per_stratum=args.min_per_stratum,
        skip_download=args.skip_download,
        progress=args.progress,
    )

    try:
        result = build_dataset(config)
        print("\n✅ Success!")
        print(f"   Train: {result.train_count}")
        print(f"   Valid: {result.valid_count}")
        print(f"   Test:  {result.test_count}")
        print(f"   Dedup removed: {result.dedup_removed}")
        print(f"   Output: {result.out_dir}")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        if os.getenv("DEBUG"):
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
