"""Tests for the unified dataset pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from slm_eval.pipelines.builder import (
    BuildConfig,
    _default_strata,
    _load_json,
    _load_jsonl,
    _normalize_cases,
    build_dataset,
    export_split,
    generate_manifest,
    generate_quality_report,
    split_cases,
    validate_cases,
)
from slm_eval.pipelines.dedup import (
    check_leakage,
    compute_text_hash,
    deduplicate_cases,
    jaccard_similarity,
    word_set,
)
from slm_eval.pipelines.exporters import (
    DEFAULT_TEMPLATES,
    AlpacaExporter,
    ChatMLExporter,
    CompletionExporter,
    OpenAIChatExporter,
    SFTExporter,
    ShareGPTExporter,
    get_exporter,
    list_formats,
)
from slm_eval.pipelines.quality import DataQualityReport, load_report, save_report


# Test fixtures
def _case(case_id: str, text: str = "Sample text", gold: dict = None, strata: dict = None) -> dict:
    return {
        "case_id": case_id,
        "text": text,
        "gold": gold or {"field": "value"},
        "strata": strata or {"quality": "high"},
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


# ===== dedup.py tests =====

def test_compute_text_hash():
    text = "hello world"
    h1 = compute_text_hash(text)
    h2 = compute_text_hash(text)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_word_set():
    text = "The quick brown fox"
    ws = word_set(text, max_words=10)
    assert ws == {"the", "quick", "brown", "fox"}


def test_jaccard_similarity():
    set1 = {"a", "b", "c"}
    set2 = {"b", "c", "d"}
    assert jaccard_similarity(set1, set2) == 2/4  # 0.5


def test_deduplicate_exact_duplicates():
    cases = [
        _case("a", "identical text"),
        _case("b", "identical text"),  # exact duplicate
        _case("c", "different text"),
    ]
    result = deduplicate_cases(cases)
    assert result.exact_duplicates == 1
    assert result.total_kept == 2


def test_deduplicate_near_duplicates():
    # Use longer texts to meet min_text_len=60 requirement
    cases = [
        _case("a", "the quick brown fox jumps over the lazy dog " * 2),
        _case("b", "the quick brown fox jumps over the lazy dog extra " * 2),  # near duplicate
        _case("c", "completely different text here " * 2),
    ]
    result = deduplicate_cases(cases, jaccard_threshold=0.6)
    assert result.near_duplicates >= 1
    assert result.total_kept <= 2


def test_check_leakage():
    train = [
        _case("train1", "the quick brown fox jumps " * 3),
        _case("train2", "completely different " * 3),
    ]
    holdout = [
        _case("holdout1", "the quick brown fox jumps extra " * 3),  # near duplicate of train1
        _case("holdout2", "totally unique " * 3),
    ]
    leaks = check_leakage(train, holdout, jaccard_threshold=0.6)
    assert len(leaks) >= 1
    assert leaks[0][0] == "holdout1"


# ===== quality.py tests =====

def test_quality_report_basic():
    cases = [_case("a"), _case("b")]
    report = DataQualityReport(total_cases=2)
    report.add_valid()
    report.add_valid()
    report.finalize(cases)
    assert report.total_cases == 2
    assert report.valid_cases == 2
    assert report.invalid_cases == 0


def test_quality_report_with_errors():
    report = DataQualityReport(total_cases=2)
    report.add_error("case1", ["missing field"])
    report.add_valid()
    report.finalize([_case("case1"), _case("case2")])
    assert report.invalid_cases == 1
    assert len(report.recommendations) > 0


def test_save_load_report(tmp_path):
    report = DataQualityReport(total_cases=1, valid_cases=1)
    report.recommendations = ["test rec"]
    path = tmp_path / "report.json"
    save_report(report, path)
    loaded = load_report(path)
    assert loaded.total_cases == 1
    assert loaded.valid_cases == 1
    assert loaded.recommendations == ["test rec"]


# ===== exporters.py tests =====

def test_list_formats():
    formats = list_formats()
    assert "sft" in formats
    assert "alpaca" in formats
    assert "chatml" in formats
    assert "sharegpt" in formats
    assert "openai" in formats
    assert "completion" in formats


def test_sft_exporter():
    exporter = SFTExporter()
    case = _case("test", "input text", {"total": 100})
    output = exporter.format(case)
    data = json.loads(output)
    assert data == {"total": 100}


def test_alpaca_exporter():
    exporter = AlpacaExporter()
    case = _case("test", "input text", {"total": 100})
    output = exporter.format(case)
    data = json.loads(output)
    assert "instruction" in data
    assert "input" in data
    assert "output" in data
    assert data["input"] == "input text"


def test_chatml_exporter():
    exporter = ChatMLExporter()
    case = _case("test", "input text", {"total": 100})
    output = exporter.format(case)
    data = json.loads(output)
    assert "messages" in data
    assert len(data["messages"]) == 3
    assert data["messages"][0]["role"] == "system"
    assert data["messages"][1]["role"] == "user"
    assert data["messages"][2]["role"] == "assistant"


def test_sharegpt_exporter():
    exporter = ShareGPTExporter()
    case = _case("test", "input text", {"total": 100})
    output = exporter.format(case)
    data = json.loads(output)
    assert "conversations" in data
    assert len(data["conversations"]) >= 2
    assert data["conversations"][0]["from"] == "human"
    assert data["conversations"][1]["from"] == "gpt"


def test_openai_exporter():
    exporter = OpenAIChatExporter()
    case = _case("test", "input text", {"total": 100})
    output = exporter.format(case)
    data = json.loads(output)
    assert "messages" in data
    assert len(data["messages"]) == 3


def test_completion_exporter():
    exporter = CompletionExporter()
    case = _case("test", "input text", {"total": 100})
    output = exporter.format(case)
    data = json.loads(output)
    assert "prompt" in data
    assert "completion" in data


def test_exporter_with_template():
    template = {"system_prompt": "Custom system", "instruction": "Custom instruction"}
    exporter = AlpacaExporter(template)
    case = _case("test", "input", {"field": "value"})
    output = exporter.format(case)
    data = json.loads(output)
    assert data["instruction"] == "Custom instruction"


def test_default_templates():
    for name, template in DEFAULT_TEMPLATES.items():
        exporter = ChatMLExporter(template)
        # Provide appropriate gold fields based on template
        if name == "qa":
            case = _case("test", "input", {"question": "What is this?", "answer": "Answer"})
        elif name == "summarization":
            case = _case("test", "input", {"summary": "Summary here"})
        elif name == "classification":
            case = _case("test", "input", {"label": "positive"})
        else:
            case = _case("test", "input", {"field": "value"})
        output = exporter.format(case)
        data = json.loads(output)
        assert "messages" in data


def test_get_exporter():
    exporter = get_exporter("alpaca")
    assert isinstance(exporter, AlpacaExporter)
    with pytest.raises(ValueError):
        get_exporter("invalid_format")


# ===== builder.py tests =====

def test_normalize_cases():
    cases = [
        {"case_id": "a", "text": "hello", "gold": {}, "strata": {}},
        {"text": "world", "gold": {}},  # missing case_id, strata
        {"case_id": "c", "gold": {"x": 1}},  # missing text, strata
    ]
    normalized = _normalize_cases(cases)
    assert len(normalized) == 3
    assert normalized[0]["case_id"] == "a"
    assert normalized[1]["case_id"] == "auto_1"
    assert normalized[2]["text"] == ""
    assert "strata" in normalized[2]


def test_default_strata():
    strata = _default_strata()
    assert "quality" in strata
    assert "ocr_signal" in strata
    assert strata["quality"] == "high"


def test_validate_cases_clean():
    cases = [_case("a", "text one"), _case("b", "text two")]
    report, good, bad = validate_cases(cases)
    assert report.valid_cases == 2
    assert report.invalid_cases == 0
    assert len(bad) == 0
    assert len(good) == 2


def test_validate_cases_missing_fields():
    cases = [
        _case("a"),
        {"case_id": "b", "text": "x"},  # missing gold, strata
        {"case_id": "b", "text": "y", "gold": {}, "strata": {}},  # duplicate id
    ]
    report, good, bad = validate_cases(cases)
    assert report.invalid_cases >= 2
    assert len(good) == 1


def test_validate_cases_with_dedup():
    cases = [
        _case("a", "identical"),
        _case("b", "identical"),  # exact duplicate
        _case("c", "different"),
    ]
    report, good, bad = validate_cases(cases, dedup_threshold=0.6)
    assert report.dedup_removed >= 1
    assert len(good) == 2


def test_split_cases_three_way():
    cases = [_case(f"c{i}", strata={"quality": "high" if i % 2 == 0 else "low"}) for i in range(20)]
    split = split_cases(cases, ["quality"], (0.8, 0.1, 0.1), seed=42)
    assert "train" in split
    assert "valid" in split
    assert "test" in split
    assert len(split["train"]) > 0
    assert len(split["valid"]) > 0
    assert len(split["test"]) > 0

    # Check no overlap
    train_ids = {c["case_id"] for c in split["train"]}
    valid_ids = {c["case_id"] for c in split["valid"]}
    test_ids = {c["case_id"] for c in split["test"]}
    assert train_ids.isdisjoint(valid_ids)
    assert train_ids.isdisjoint(test_ids)
    assert valid_ids.isdisjoint(test_ids)


def test_split_cases_small_dataset():
    cases = [_case("a"), _case("b")]
    split = split_cases(cases, ["quality"], (0.8, 0.1, 0.1), seed=42)
    assert "train" in split
    assert "valid" in split
    # test might be empty for small datasets


def test_export_split(tmp_path):
    cases = [_case("a"), _case("b")]
    out_dir = tmp_path / "train"
    count = export_split(cases, out_dir, "sft", "train")
    assert count == 2
    assert (out_dir / "train.jsonl").exists()


def test_generate_quality_report():
    cases = [_case("a", "short", {"x": 1}), _case("b", "much longer text here", {"y": 2})]
    report = generate_quality_report(cases)
    assert report["total_cases"] == 2
    assert "text_length" in report
    assert "gold_field_coverage" in report
    assert report["gold_field_coverage"]["x"] == 0.5
    assert report["gold_field_coverage"]["y"] == 0.5


def test_generate_manifest(tmp_path):
    cases = [_case("a"), _case("b")]
    quality = generate_quality_report(cases)
    path = generate_manifest(tmp_path, "hash123", 42, "quality", quality, 2, (0.8, 0.1, 0.1), 1.5)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["source_hash"] == "hash123"
    assert data["split_fractions"]["train"] == 0.8


def test_load_jsonl(tmp_path):
    path = tmp_path / "test.jsonl"
    _write_jsonl(path, [_case("a"), _case("b")])
    cases = _load_jsonl(path)
    assert len(cases) == 2
    assert cases[0]["case_id"] == "a"


def test_load_json(tmp_path):
    path = tmp_path / "test.json"
    path.write_text(json.dumps([_case("a"), _case("b")]))
    cases = _load_json(path)
    assert len(cases) == 2


# ===== Integration tests =====

def test_build_dataset_local_jsonl(tmp_path):
    """End-to-end test with local JSONL file."""
    # Create input data
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_jsonl(input_dir / "cases.jsonl", [
        _case(f"c{i}", f"text {i}", {"value": i}, {"quality": "high" if i % 2 == 0 else "low"})
        for i in range(20)
    ])

    out_dir = tmp_path / "output"
    config = BuildConfig(
        source=str(input_dir),
        out_dir=out_dir,
        fmt="sft",
        split_keys=["quality"],
        split_fractions=(0.8, 0.1, 0.1),
        seed=42,
        validate=True,
        dedup=True,
        score=True,
        progress=False,
    )

    result = build_dataset(config)

    assert result.train_count > 0
    assert result.valid_count > 0
    assert result.test_count > 0
    assert result.train_count + result.valid_count + result.test_count <= 20
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "quality_report.json").exists()
    assert (out_dir / "train" / "train.jsonl").exists()
    assert (out_dir / "valid" / "valid.jsonl").exists()
    assert (out_dir / "test" / "test.jsonl").exists()


def test_build_dataset_all_formats(tmp_path):
    """Test all export formats work."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_jsonl(input_dir / "cases.jsonl", [_case(f"c{i}", f"text {i}") for i in range(20)])

    for fmt in ["sft", "alpaca", "chatml", "sharegpt", "openai", "completion"]:
        out_dir = tmp_path / f"output_{fmt}"
        config = BuildConfig(
            source=str(input_dir),
            out_dir=out_dir,
            fmt=fmt,
            split_keys=["quality"],
            split_fractions=(0.8, 0.1, 0.1),
            seed=42,
            progress=False,
        )
        result = build_dataset(config)
        assert (out_dir / "train" / "train.jsonl").exists()
        # Verify format
        line = (out_dir / "train" / "train.jsonl").read_text().splitlines()[0]
        data = json.loads(line)
        if fmt == "sft":
            assert isinstance(data, dict)
        elif fmt == "alpaca":
            assert "instruction" in data
        elif fmt == "chatml":
            assert "messages" in data
        elif fmt == "sharegpt":
            assert "conversations" in data
        elif fmt == "openai":
            assert "messages" in data
        elif fmt == "completion":
            assert "prompt" in data and "completion" in data


def test_build_dataset_with_template(tmp_path):
    """Test with predefined template."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_jsonl(input_dir / "cases.jsonl", [
        _case(f"c{i}", f"receipt text {i}", {"total": 100 + i}) for i in range(20)
    ])

    out_dir = tmp_path / "output"
    config = BuildConfig(
        source=str(input_dir),
        out_dir=out_dir,
        fmt="chatml",
        template="receipt_extraction",
        split_keys=["quality"],
        split_fractions=(0.8, 0.1, 0.1),
        seed=42,
        progress=False,
    )
    result = build_dataset(config)

    line = (out_dir / "train" / "train.jsonl").read_text().splitlines()[0]
    data = json.loads(line)
    assert "You are a receipt information extraction assistant" in data["messages"][0]["content"]


# ===== Leakage tests =====

def test_no_leakage_across_splits():
    """Ensure train/valid/test have no overlapping or near-duplicate cases."""
    cases = [
        _case(f"c{i}", f"unique text content number {i}", {"val": i}, {"quality": "high"})
        for i in range(100)
    ]
    # Add some near-duplicates to test leakage detection
    cases.append(_case("dup1", "unique text content number 5 extra", {"val": 5}, {"quality": "high"}))
    cases.append(_case("dup2", "unique text content number 10 extra", {"val": 10}, {"quality": "low"}))

    split = split_cases(cases, ["quality"], (0.8, 0.1, 0.1), seed=42)

    train_ids = {c["case_id"] for c in split["train"]}
    valid_ids = {c["case_id"] for c in split["valid"]}
    test_ids = {c["case_id"] for c in split["test"]}

    # No exact ID overlap
    assert train_ids.isdisjoint(valid_ids)
    assert train_ids.isdisjoint(test_ids)
    assert valid_ids.isdisjoint(test_ids)

    # No near-duplicate leakage
    from slm_eval.pipelines.dedup import check_leakage
    assert len(check_leakage(split["train"], split["valid"])) == 0
    assert len(check_leakage(split["train"], split["test"])) == 0
    assert len(check_leakage(split["valid"], split["test"])) == 0


# ===== Docling pipeline tests =====

def test_docling_file_type_detection(tmp_path):
    """Test file type detection logic."""
    from slm_eval.pipelines.docling_pipeline import DoclingPipeline

    pipeline = DoclingPipeline()

    assert pipeline._get_file_type(Path("file.pdf")) == "pdf"
    assert pipeline._get_file_type(Path("file.docx")) == "docx"
    assert pipeline._get_file_type(Path("file.pptx")) == "pptx"
    assert pipeline._get_file_type(Path("file.xlsx")) == "xlsx"
    assert pipeline._get_file_type(Path("file.html")) == "html"
    assert pipeline._get_file_type(Path("file.epub")) == "epub"
    assert pipeline._get_file_type(Path("file.jpg")) == "image"
    assert pipeline._get_file_type(Path("file.png")) == "image"
    assert pipeline._get_file_type(Path("file.txt")) == "text"
    # Unknown extensions default to "text"
    assert pipeline._get_file_type(Path("file.unknown_ext")) == "text"


def test_docling_process_mock():
    """Test docling processing works with mock when docling not installed."""
    from slm_eval.pipelines.docling_pipeline import DoclingPipeline

    pipeline = DoclingPipeline()
    result = pipeline.process_file("mock_file.txt")

    assert result["success"] is True
    assert "text" in result
    assert "metadata" in result
    assert result["error"] is None


def test_docling_process_directory(tmp_path):
    """Test docling processing of a directory with mock documents."""
    from slm_eval.pipelines.docling_pipeline import process_directory

    # Create mock files
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "doc1.pdf").write_text("mock pdf content")
    (input_dir / "doc2.txt").write_text("mock text content")

    output_dir = tmp_path / "output"

    results, processed, failed = process_directory(input_dir, output_dir, format="jsonl")

    assert processed == 2
    assert failed == 0
    assert len(results) == 2
    for result in results:
        assert result["success"] is True
        assert result["text"] != ""


def test_docling_process_directory_subdirs(tmp_path):
    """Test docling processing includes subdirectories."""
    from slm_eval.pipelines.docling_pipeline import process_directory

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    subdir = input_dir / "subdir"
    subdir.mkdir()
    (input_dir / "doc1.pdf").write_text("root doc")
    (subdir / "doc2.pdf").write_text("sub doc")

    output_dir = tmp_path / "output"

    results, processed, failed = process_directory(input_dir, output_dir, include_subdirs=True)

    assert processed == 2

    # Test without subdirs
    results2, processed2, failed2 = process_directory(input_dir, output_dir, include_subdirs=False)

    assert processed2 == 1


def test_docling_build_dataset_integration(tmp_path):
    """Test docling integration with build_dataset pipeline."""
    from slm_eval.pipelines.builder import BuildConfig, build_dataset

    # Create a directory with input JSONL and mock document files
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create input JSONL with cases
    _write_jsonl(input_dir / "cases.jsonl", [
        _case(f"c{i}", f"text {i}", {"value": i}, {"quality": "high" if i % 2 == 0 else "low"})
        for i in range(20)
    ])

    # Create mock document files
    (input_dir / "doc1.pdf").write_text("Receipt: Total $100, Vendor: Acme Corp")
    (input_dir / "doc2.txt").write_text("Invoice: Total $200, Vendor: Beta LLC")

    out_dir = tmp_path / "output"
    config = BuildConfig(
        source=str(input_dir),
        out_dir=out_dir,
        fmt="sft",
        use_docling=True,
        split_keys=["quality"],
        split_fractions=(0.8, 0.1, 0.1),
        seed=42,
        progress=False,
    )

    result = build_dataset(config)

    # Pipeline should complete successfully with docling enabled
    assert result.total_input > 0
    assert result.train_count + result.valid_count + result.test_count > 0
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "train" / "train.jsonl").exists()
    assert (out_dir / "valid" / "valid.jsonl").exists()
    assert (out_dir / "test" / "test.jsonl").exists()


def test_docling_integration_with_pipeline(tmp_path):
    """Test integration of docling results with existing cases."""
    from slm_eval.pipelines.docling_pipeline import integrate_with_pipeline

    cases = [
        _case("c1", "old text", {"quality": "high"}, {"quality": "high", "source": "doc1.pdf"}),
        _case("c2", "old text", {"quality": "low"}, {"quality": "low", "source": "doc2.pdf"}),
    ]

    docling_results = [
        {
            "text": "Docling extracted text from doc1",
            "metadata": {"source": "doc1.pdf", "file_type": "pdf"},
            "success": True,
            "error": None,
        },
        {
            "text": "Docling extracted text from doc2",
            "metadata": {"source": "doc2.pdf", "file_type": "pdf"},
            "success": True,
            "error": None,
        },
    ]

    integrated = integrate_with_pipeline(cases, docling_results)

    # Check that text from docling results was integrated
    assert integrated[0]["text"] == "Docling extracted text from doc1"
    assert integrated[1]["text"] == "Docling extracted text from doc2"


def test_docling_build_config_defaults():
    """Test BuildConfig defaults for docling options."""
    from slm_eval.pipelines.builder import BuildConfig

    with tempfile.TemporaryDirectory() as tmp:
        config = BuildConfig(source=tmp, out_dir=Path(tmp))

        assert config.use_docling is False
        assert config.docling_output_dir is None
        assert "pdf" in config.docling_formats
        assert "docx" in config.docling_formats
        assert config.docling_include_subdirs is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
