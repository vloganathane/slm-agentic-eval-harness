"""Hashing tests: content addressing for provenance chain (SPEC_V3 §4.3)."""

from slm_eval.hashing import canonical_json, sha256_dir, sha256_file, sha256_json, sha256_text


def test_sha256_file_matches_content(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hello")
    assert sha256_file(p) == sha256_text("hello")


def test_sha256_dir_is_order_independent(tmp_path):
    d1, d2 = tmp_path / "one", tmp_path / "two"
    for d in (d1, d2):
        d.mkdir()
    (d1 / "a.txt").write_text("A")
    (d1 / "b.txt").write_text("B")
    (d2 / "b.txt").write_text("B")
    (d2 / "a.txt").write_text("A")
    assert sha256_dir(d1) == sha256_dir(d2)  # same content, different creation order


def test_sha256_dir_changes_with_content(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "a.txt").write_text("A")
    before = sha256_dir(d)
    (d / "a.txt").write_text("changed")
    assert sha256_dir(d) != before


def test_canonical_json_sorts_keys():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_sha256_json_key_order_insensitive():
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})
