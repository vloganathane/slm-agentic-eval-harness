"""Build a clean, domain‑independent dataset from raw sources.

Usage:
    python -m slm_eval.cli build-dataset \
        --source .scratch/raw_receipts/ \
        --out data/domain_independent/ \
        --format sft \
        --split-keys layout,quality \
        --min-per-stratum 50 \
        --validate --dedup --score
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from slm_eval.dataset_validator import DatasetValidator, DataQualityReport
from slm_eval.providers.data_pull import AutoDownloader
from slm_eval.exporters import SFTExporter, AlpacaExporter, ChatMLExporter
from slm_eval.hashing import sha256_json
from slm_eval.split import split_stratified, NEAR_DUP_THRESHOLD, minhash_signature, signature_similarity

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


def load_cases(source: str, dest_dir: Path) -> list[dict[str, Any]]:
    """Load cases from source (local dir, local file, or remote) into dest_dir."""
    # Ensure dest_dir exists
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Download if needed
    raw_path = auto_download(source, dest_dir)

    # Load cases.jsonl (or equivalent) into memory
    cases = []
    
    # First try cases.jsonl
    if raw_path.is_file() and raw_path.suffix == ".jsonl":
        cases_path = raw_path
        for line in cases_path.read_text().splitlines():
            if not line.strip():
                continue
            cases.append(json.loads(line))
    elif raw_path.is_dir():
        cases_path = raw_path / "cases.jsonl"
        if not cases_path.exists():
            # Try parquet files
            parquet_files = list(raw_path.glob("*.parquet"))
            if parquet_files:
                cases = load_cases_from_parquet(parquet_files)
                return cases
            # Try CSV files
            csv_files = list(raw_path.glob("*.csv"))
            if csv_files:
                cases = load_cases_from_csv(csv_files)
                return cases
            # Try single JSON file
            json_files = list(raw_path.glob("*.json"))
            if len(json_files) == 1:
                with json_files[0].open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    cases = data
                else:
                    cases = [data]
                return cases
            raise RuntimeError(f"Expected cases.jsonl, parquet files, CSV files, or single JSON file at {raw_path}, not found")
        # Read jsonl
        for line in cases_path.read_text().splitlines():
            if not line.strip():
                continue
            cases.append(json.loads(line))
    elif raw_path.is_file() and raw_path.suffix in (".parquet",):
        # Single parquet file
        cases = load_cases_from_parquet([raw_path])
    elif raw_path.is_file() and raw_path.suffix in (".json",):
        # Single JSON file
        with raw_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            cases = data
        else:
            cases = [data]
    else:
        raise RuntimeError(f"Unsupported source format: {raw_path}")
    return cases


def load_cases_from_parquet(parquet_files: list[Path]) -> list[dict[str, Any]]:
    """Load cases from parquet files. Assumes columns: case_id, text, gold, strata."""
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas required to load parquet files. Run: pip install pandas")
    
    cases = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        for _, row in df.iterrows():
            # Convert to the expected case format
            case = {}
            # case_id
            if "case_id" in df.columns:
                case["case_id"] = str(row.get("case_id", f"auto_{len(cases)}"))
            elif "id" in df.columns:
                case["case_id"] = str(row.get("id", f"auto_{len(cases)}"))
            else:
                case["case_id"] = f"auto_{len(cases)}"
            
            # Text field - try multiple common column names
            text_val = ""
            for col in ["text", "content", "sentence", "document", "input"]:
                if col in df.columns:
                    text_val = str(row.get(col, ""))
                    break
            if not text_val:
                # For image datasets, use label as text representation
                for col in ["label", "target", "class"]:
                    if col in df.columns:
                        text_val = str(row.get(col, ""))
                        break
            case["text"] = text_val
            
            # Gold - try to extract structured gold or use all other columns
            if "gold" in df.columns:
                case["gold"] = row.get("gold", {})
            elif "label" in df.columns:
                case["gold"] = {"label": row.get("label")}
            else:
                # Use all other columns as gold
                gold = {k: v for k, v in row.items() if k not in ["case_id", "id", "text", "content", "strata", "sentence", "document", "input", "label", "target", "class"]}
                case["gold"] = gold
            
            # Strata - provide defaults if not present
            if "strata" in df.columns:
                case["strata"] = row.get("strata", {})
            else:
                case["strata"] = {"quality": "high", "ocr_signal": "usable", "layout": "clean", "layout_type": "clean"}
            
            cases.append(case)
    return cases


def load_cases_from_csv(csv_files: list[Path]) -> list[dict[str, Any]]:
    """Load cases from CSV files. Assumes columns: case_id, text, gold, strata or similar."""
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas required to load CSV files. Run: pip install pandas")
    
    cases = []
    for cf in csv_files:
        df = pd.read_csv(cf)
        for _, row in df.iterrows():
            # Convert to the expected case format
            case = {}
            # case_id
            if "case_id" in df.columns:
                case["case_id"] = str(row.get("case_id", f"auto_{len(cases)}"))
            elif "id" in df.columns:
                case["case_id"] = str(row.get("id", f"auto_{len(cases)}"))
            else:
                case["case_id"] = f"auto_{len(cases)}"
            
            # Text field - try multiple common column names
            text_val = ""
            for col in ["text", "content", "sentence", "document", "input", "Passengerid"]:
                if col in df.columns:
                    text_val = str(row.get(col, ""))
                    break
            if not text_val:
                # For image datasets, use label as text representation
                for col in ["label", "target", "class", "Survived", "2urvived"]:
                    if col in df.columns:
                        text_val = str(row.get(col, ""))
                        break
            case["text"] = text_val
            
            # Gold - try to extract structured gold or use all other columns
            if "gold" in df.columns:
                case["gold"] = row.get("gold", {})
            elif "label" in df.columns:
                case["gold"] = {"label": row.get("label")}
            elif "Survived" in df.columns:
                case["gold"] = {"label": row.get("Survived")}
            elif "2urvived" in df.columns:
                case["gold"] = {"label": row.get("2urvived")}
            else:
                # Use all other columns as gold
                gold = {k: v for k, v in row.items() if k not in ["case_id", "id", "text", "content", "strata", "sentence", "document", "input", "label", "target", "class", "Passengerid", "Survived", "2urvived"]}
                case["gold"] = gold
            
            # Strata - provide defaults if not present
            if "strata" in df.columns:
                case["strata"] = row.get("strata", {})
            else:
                case["strata"] = {"quality": "high", "ocr_signal": "usable", "layout": "clean", "layout_type": "clean"}
            
            cases.append(case)
    return cases


def auto_download(source: str, dest_dir: Path) -> Path:
    """Select appropriate downloader based on source type."""
    # Local path first: if it exists on disk, use it directly
    local = Path(source)
    if local.exists():
        return local

    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        # Use AutoDownloader for all URLs (handles HF, Kaggle, and generic URLs)
        return auto_download_hf_or_kaggle(source, dest_dir)
    elif "/" in source and not parsed.scheme:
        # Assume HF or Kaggle (format: org/dataset or org/dataset@rev); AutoDownloader decides
        return auto_download_hf_or_kaggle(source, dest_dir)
    elif "/" not in source and not parsed.scheme:
        # Could be a shorthand HF dataset like "imdb" or "rotten_tomatoes"
        # Try HF first via AutoDownloader
        return auto_download_hf_or_kaggle(source, dest_dir)
    else:
        # Local path that may not exist yet
        return local


def auto_download_url(url: str, dest_dir: Path) -> Path:
    """Download a single file or archive from a URL."""
    parsed = urlparse(url)
    dest = dest_dir / parsed.path.split("/")[-1]
    dest.parent.mkdir(parents=True, exist_ok=True)
    return auto_download_url_impl(url, dest)


def auto_download_url_impl(url: str, dest: Path) -> Path:
    """Download a generic file/URL."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def auto_download_hf_or_kaggle(source: str, dest_dir: Path) -> Path:
    """Handle HF or Kaggle dataset URLs."""
    downloader = AutoDownloader()
    return downloader.download(source, dest_dir)


def load_cases_from_jsonl(cases_path: Path) -> list[dict[str, Any]]:
    """Read cases from a cases.jsonl file."""
    cases = []
    with cases_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cases.append(json.loads(line))
    return cases


def dedup_cases(cases: list[dict[str, Any]], threshold: float = NEAR_DUP_THRESHOLD) -> tuple[list[dict[str, Any]], int]:
    """Remove near-duplicate cases based on minhash signature."""
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    seen_sigs: list[tuple[int, ...]] = []

    for case in cases:
        text = case.get("text", "")
        key = hashlib.sha256(text.encode()).hexdigest()
        if text in seen_texts:
            removed.append(case)
            continue
        sig = minhash_signature(text)
        if any(signature_similarity(sig, ks) >= threshold for ks in seen_sigs):
            removed.append(case)
            continue
        seen_texts.add(text)
        seen_sigs.append(minhash_signature(text))
        kept.append(case)

    return kept, removed, len(removed)


def split_stratified_cases(
    cases: list[dict[str, Any]],
    keys: list[str],
    fraction: float,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Return train/valid/test splits."""
    from slm_eval.split import Split
    split = split_stratified(cases, strata_keys=keys, fraction=fraction, seed=seed)
    return {"train": split.train, "valid": split.holdout, "test": split.holdout}


def export_cases(
    cases: list[dict[str, Any]],
    out_dir: Path,
    fmt: str = "sft",
    split: str = "train",
) -> None:
    """Export cases to a fine‑tuning format."""
    exporter = None
    if fmt == "sft":
        exporter = SFTExporter()
    elif fmt == "alpaca":
        exporter = AlpacaExporter()
    elif fmt == "chatml":
        exporter = ChatMLExporter()
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    out_file = out_dir / f"{split}.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(exporter.format(case) + "\n")


def generate_manifest(
    out_dir: Path,
    source_hash: str,
    split_seed: int,
    split_strategy: str,
    quality_distribution: dict,
    case_count: int,
) -> None:
    """Write a JSON manifest summarizing the dataset."""
    manifest = {
        "source_hash": source_hash,
        "split_seed": split_seed,
        "split_strategy": split_strategy,
        "quality_distribution": quality_distribution,
        "case_count": case_count,
        "out_dir": str(out_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def quality_report(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate a simple quality report."""
    total = len(cases)
    valid = sum(1 for c in cases if c.get("strata", {}).get("ocr_signal") == "usable")
    low_quality = sum(1 for c in cases if c.get("strata", {}).get("quality") == "low")
    high_quality = sum(1 for c in cases if c.get("strata", {}).get("quality") == "high")
    return {
        "total_cases": total,
        "usable_cases": valid,
        "low_quality": low_quality,
        "high_quality": high_quality,
        "quality_ratio": valid / total if total else 0.0,
    }


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Build a clean dataset for SLM fine‑tuning.")
    parser.add_argument("--source", required=True, help="Path or remote source (e.g., HF dataset, Kaggle, local dir).")
    parser.add_argument("--out", required=True, help="Output directory for the clean dataset.")
    parser.add_argument("--format", choices=["sft", "alpaca", "chatml"], default="sft", help="Fine‑tuning format.")
    parser.add_argument("--split-keys", nargs="+", default=["layout", "quality"], help="Strata keys for stratified split.")
    parser.add_argument("--split-fraction", type=float, default=0.5, help="Fraction of data for validation (default 0.5).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--validate", action="store_true", help="Run validation and dedup.")
    parser.add_argument("--dedup", action="store_true", help="Enable min‑hash dedup.")
    parser.add_argument("--score", action="store_true", help="Generate quality scores.")
    args = parser.parse_args()

    # 1️⃣ Download / load raw cases
    out_dir = Path(args.out)
    raw_dir = out_dir.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.source, raw_dir)

    # 2️⃣ Optional validation & dedup
    if args.validate or args.dedup:
        validator = DatasetValidator(dedup_threshold=NEAR_DUP_THRESHOLD)
        report, good_cases, bad_cases = validator.validate(cases)
        print(report.to_dict())
        # Save reports
        (out_dir / "validation_report.json").write_text(json.dumps(report.to_dict(), indent=2))
        if bad_cases:
            (out_dir / "bad_cases.jsonl").write_text("\n".join(json.dumps(c) for c in bad_cases))
        if args.validate:
            print(f"[VALIDATION] {report.total_cases} cases, {report.valid_cases} valid, {report.invalid_cases} invalid")

    # 3️⃣ Split (stratified)
    split_fraction = args.split_fraction
    split = split_stratified_cases(cases, strata_keys=args.split_keys, fraction=split_fraction, seed=args.seed)

    # 4️⃣ Export each split
    for split_name, split_cases in split.items():
        out_subdir = out_dir / split_name
        out_subdir.mkdir(parents=True, exist_ok=True)
        export_cases(split_cases, out_subdir, fmt=args.format, split=split_name)

    # 5️⃣ Quality score (optional)
    if args.score:
        quality = quality_report(cases)
        (out_dir / "quality_report.json").write_text(json.dumps(quality, indent=2))

    # 6️⃣ Manifest
    source_hash = sha256_json(cases)
    quality_dist = quality_report(cases)
    generate_manifest(
        out_dir,
        source_hash,
        args.seed,
        ",".join(args.split_keys),
        quality_dist,
        len(cases),
    )
    print(f"[BUILD] Dataset written to {out_dir}")


if __name__ == "__main__":
    main()