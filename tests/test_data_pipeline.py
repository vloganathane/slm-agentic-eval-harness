"""Dataset pipeline: load, validate, analyze, split, save (Option B)."""

from __future__ import annotations

import json

import pytest

from slm_eval.data_pipeline import DataPipeline


def _case(case_id: str, *, item_count: str = "small", text: str | None = None) -> dict:
    return {
        "case_id": case_id,
        "text": text if text is not None else f"RECEIPT {case_id}\nItem A 1.00\nTOTAL 1.00",
        "gold": {"total": 1.0, "line_items": [{"name": "Item A", "qty": 1, "price": 1.0}]},
        "strata": {"item_count": item_count, "quality": "high"},
    }


def _write_jsonl(path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_load_cases_skips_blank_and_malformed_lines(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(_case("a")) + "\n"
        "\n"
        "{not json}\n"
        + json.dumps(_case("b")) + "\n"
    )

    cases = DataPipeline().load_cases(path)

    assert [c["case_id"] for c in cases] == ["a", "b"]


def test_validate_clean_cases(tmp_path):
    cases = [_case("a"), _case("b")]
    report = DataPipeline().validate(cases)

    assert report.total_cases == 2
    assert report.valid_cases == 2
    assert report.invalid_cases == 0
    assert report.schema_errors == []


def test_validate_flags_missing_fields_duplicates_and_bad_types(tmp_path):
    cases = [
        _case("a"),
        _case("a"),  # duplicate id
        {"case_id": "b", "text": 42, "gold": "not-a-dict", "strata": "not-a-dict"},
        {"text": "missing id"},  # missing required fields
    ]

    report = DataPipeline().validate(cases)

    assert report.total_cases == 4
    assert report.valid_cases == 1
    assert report.invalid_cases == 3
    joined = " ".join(report.schema_errors)
    assert "duplicate case_id: a" in joined
    assert "text must be str" in joined
    assert "gold must be dict" in joined
    assert "strata must be dict" in joined
    assert report.recommendations


def test_prepare_split_rejects_invalid_cases():
    with pytest.raises(ValueError, match="invalid cases"):
        DataPipeline().prepare_split([{"case_id": "bad"}])


def test_prepare_split_is_stratified_disjoint_and_complete():
    cases = [
        _case(f"s{i}", item_count="small") for i in range(10)
    ] + [
        _case(f"l{i}", item_count="large") for i in range(10)
    ]

    splits = DataPipeline().prepare_split(
        cases, strata_keys=["item_count"], holdout_fraction=0.3,
        seed=42, min_per_stratum=1,
    )

    train = splits["train"]
    holdout = splits["holdout"]
    assert train and holdout
    train_ids = {c["case_id"] for c in train}
    holdout_ids = {c["case_id"] for c in holdout}
    assert not (train_ids & holdout_ids), "holdout leaked into train"
    assert train_ids | holdout_ids == {c["case_id"] for c in cases}
    assert {c["strata"]["item_count"] for c in holdout} == {"small", "large"}


def test_save_splits_and_training_format_roundtrip(tmp_path):
    pipeline = DataPipeline()
    train = [_case("a"), _case("b")]
    holdout = [_case("c")]
    splits = {"train": train, "holdout": holdout, "stratum_sizes": {"small": 3}}

    split_dir = tmp_path / "splits"
    counts = pipeline.save_splits(splits, split_dir)

    assert counts == {"train": 2, "holdout": 1}
    assert sorted(p.name for p in split_dir.iterdir()) == ["holdout.jsonl", "train.jsonl"]
    saved = pipeline.load_cases(split_dir / "train.jsonl")
    assert [c["case_id"] for c in saved] == ["a", "b"]

    training_path = tmp_path / "training_format" / "train.jsonl"
    n = pipeline.to_training_format(train, training_path)

    assert n == 2
    records = [json.loads(line) for line in training_path.read_text().splitlines()]
    assert all(set(r) == {"text", "gold"} for r in records)


def test_analyze_returns_duplicate_pairs_and_stats(tmp_path):
    dup_text = "ACME STORE\n123 Main St\nMilk 2.99\nBread 1.99\nTOTAL 4.98"
    path = tmp_path / "cases.jsonl"
    _write_jsonl(
        path,
        [
            _case("a", text=dup_text),
            _case("b", text=dup_text),
            _case("c", text="TOTALLY DIFFERENT MERCHANT\nWidget 9.99\nTOTAL 9.99"),
        ],
    )

    analysis = DataPipeline().analyze(path, minhash_threshold=0.6)

    assert analysis["n_cases"] == 3
    assert analysis["n_fields"] == 2
    assert analysis["duplicate_pairs"]
    assert analysis["duplicate_pairs"][0][0:2] == ("a", "b")
    assert analysis["text_stats"]["max_length"] == len(dup_text)


def test_load_from_run_reads_holdout(tmp_path):
    run_dir = tmp_path / "runs" / "run-x" / "holdout"
    run_dir.mkdir(parents=True)
    _write_jsonl(run_dir / "cases.jsonl", [_case("a")])

    cases, path = DataPipeline(data_dir=tmp_path / "runs").load_from_run("run-x")

    assert [c["case_id"] for c in cases] == ["a"]
    assert path == run_dir / "cases.jsonl"


def test_load_from_run_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        DataPipeline(data_dir=tmp_path / "runs").load_from_run("nope")
