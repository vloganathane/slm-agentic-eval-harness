"""Fixture tests: determinism, round-trip, strata, internal consistency (SPEC_V3 §10 M0)."""

import json

import pytest

from slm_eval import fixture
from slm_eval.hashing import sha256_json


def test_generate_is_deterministic():
    a = fixture.generate(count=50, seed=7)
    b = fixture.generate(count=50, seed=7)
    assert sha256_json(a) == sha256_json(b)


def test_generate_seed_changes_output():
    a = fixture.generate(count=50, seed=7)
    b = fixture.generate(count=50, seed=8)
    assert sha256_json(a) != sha256_json(b)


def test_write_read_round_trip(tmp_path):
    path, h = fixture.write_dataset(tmp_path / "d", count=100, seed=42)
    cases = fixture.read_dataset(path)
    assert len(cases) == 100
    assert sha256_json(cases) == h  # hash matches on re-read


def test_write_is_byte_deterministic(tmp_path):
    p1, _ = fixture.write_dataset(tmp_path / "a", count=30, seed=1)
    p2, _ = fixture.write_dataset(tmp_path / "b", count=30, seed=1)
    assert p1.read_bytes() == p2.read_bytes()


def test_case_schema_and_consistency():
    for c in fixture.generate(count=40, seed=3):
        assert set(c) == {"case_id", "text", "gold", "strata"}
        gold = c["gold"]
        # The gold values must actually appear in the source text (extraction is learnable):
        assert gold["invoice_number"] in c["text"]
        assert gold["date"] in c["text"]
        assert str(gold["total"]) in c["text"]
        assert gold["tax_id"] in c["text"]
        assert gold["vendor_name"] in c["text"]
        assert set(c["strata"]) == {"layout_type", "language", "quality"}


def test_strata_are_populated():
    cases = fixture.generate(count=200, seed=42)
    for key in ("layout_type", "language", "quality"):
        groups = dict(fixture.iter_strata(cases, key))
        assert len(groups) > 1, f"{key} has a single value — strata useless"
        assert sum(len(v) for v in groups.values()) == len(cases)


def test_line_items_total_consistent():
    for c in fixture.generate(count=20, seed=5):
        recomputed = round(sum(li["qty"] * li["rate"] for li in c["gold"]["line_items"]), 2)
        assert recomputed == pytest.approx(c["gold"]["total"])


def test_malformed_jsonl_raises(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"case_id": "a"}\nnot json\n')
    with pytest.raises(ValueError, match="malformed JSON"):
        fixture.read_dataset(p)


def test_write_dataset_creates_parent_dirs(tmp_path):
    path, _ = fixture.write_dataset(tmp_path / "deep" / "nested", count=5, seed=0)
    assert path.exists()
    assert len(json.loads(path.read_text().splitlines()[0])) == 4
