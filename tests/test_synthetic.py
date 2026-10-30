"""Synthetic case generation + MinHash dedup + review gating (SPEC_V3 M3; AC-12)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from slm_eval.hashing import minhash_signature, minhash_similarity
from slm_eval.llm import ScriptedClient
from slm_eval.store import Store
from slm_eval.synthetic import (
    SyntheticCase,
    generate_synthetic_cases,
    inject_synthetic_cases,
    resolve_synthetic_review,
)

HOLDOUT_TEXT = (
    "WALMART\n123 Main Street\nMilk 2.99\nBread 1.99\nEggs 3.49\n"
    "TOTAL: 8.47\nThank you for shopping with us"
)
NOVEL_TEXT = (
    "STARBUCKS COFFEE\nStore #4421\nLatte 4.95\nCroissant 3.50\n"
    "TOTAL 8.45\nPaid by VISA"
)


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(inference=SimpleNamespace(base_url="http://localhost:8000/v1"))


def _write_holdout(tmp_path, texts: list[str]):
    holdout = tmp_path / "holdout"
    holdout.mkdir(parents=True, exist_ok=True)
    holdout.joinpath("cases.jsonl").write_text(
        "".join(
            json.dumps({"case_id": f"h{i}", "text": t, "gold": {}, "strata": {}}) + "\n"
            for i, t in enumerate(texts)
        )
    )
    return holdout


def _generation_payload(*texts: str) -> str:
    return json.dumps(
        [
            {
                "text": t,
                "gold": {"total": 1.0, "line_items": []},
                "strata": {"quality": "medium", "layout": "mixed"},
            }
            for t in texts
        ]
    )


def test_generate_drops_holdout_duplicates_and_keeps_novel(tmp_path):
    """AC-12: a generated case near-identical to the holdout is rejected."""
    holdout = _write_holdout(tmp_path, [HOLDOUT_TEXT])
    client = ScriptedClient(default=_generation_payload(HOLDOUT_TEXT, NOVEL_TEXT))

    result = generate_synthetic_cases(
        _cfg(), holdout, n_propose=2, minhash_threshold=0.6, client=client
    )

    assert result.proposed == 2
    assert result.duplicate == 1
    assert result.novel == 1
    assert [c.text for c in result.cases] == [NOVEL_TEXT]

    # The surviving case must not be similar to any holdout text.
    holdout_mh = "".join(str(x) for x in minhash_signature(HOLDOUT_TEXT)[:16])
    novel_mh = "".join(str(x) for x in minhash_signature(NOVEL_TEXT)[:16])
    assert minhash_similarity(novel_mh, holdout_mh) < 0.6


def test_generated_cases_are_content_addressed_and_carry_metadata(tmp_path):
    holdout = _write_holdout(tmp_path, [HOLDOUT_TEXT])
    client = ScriptedClient(default=_generation_payload(NOVEL_TEXT))

    result = generate_synthetic_cases(
        _cfg(), holdout, n_propose=1, minhash_threshold=0.6, client=client
    )

    case = result.cases[0]
    assert case.case_id.startswith("synth_")
    assert len(case.case_id) == len("synth_") + 8
    assert case.minhash
    assert case.gold == {"total": 1.0, "line_items": []}
    assert case.strata["quality"] == "medium"
    assert case.source == "synthetic"


def test_short_generated_texts_are_skipped(tmp_path):
    holdout = _write_holdout(tmp_path, [HOLDOUT_TEXT])
    client = ScriptedClient(default=_generation_payload("too short"))

    result = generate_synthetic_cases(
        _cfg(), holdout, n_propose=1, minhash_threshold=0.6, client=client
    )

    assert result.novel == 0
    assert result.duplicate == 0
    assert result.cases == []


def test_load_holdout_falls_back_to_individual_json_files(tmp_path):
    holdout = tmp_path / "holdout-json"
    holdout.mkdir()
    holdout.joinpath("h0.json").write_text(
        json.dumps({"case_id": "h0", "text": HOLDOUT_TEXT, "gold": {}, "strata": {}})
    )
    client = ScriptedClient(default=_generation_payload(HOLDOUT_TEXT, NOVEL_TEXT))

    result = generate_synthetic_cases(
        _cfg(), holdout, n_propose=2, minhash_threshold=0.6, client=client
    )

    assert result.duplicate == 1
    assert result.novel == 1


def test_inject_and_resolve_synthetic_review(tmp_path):
    """Synthetic cases enter the review queue and stay pending until decided."""
    store = Store(tmp_path / "db.sqlite")
    try:
        cases = [
            SyntheticCase(
                case_id="synth_a", text=NOVEL_TEXT, gold={"total": 8.45},
                strata={"quality": "high"}, minhash="x",
            ),
            SyntheticCase(
                case_id="synth_b", text=NOVEL_TEXT + " #2", gold={"total": 1.0},
                strata={"quality": "low"}, minhash="y",
            ),
        ]

        assert inject_synthetic_cases(store, "run-synth", cases) == 2
        assert store.pending_review_count("run-synth") == 2

        assert resolve_synthetic_review(store, "run-synth", "synth_a", "approve") is True
        assert store.pending_review_count("run-synth") == 1
        assert resolve_synthetic_review(store, "run-synth", "synth_b", "reject") is False
        assert store.pending_review_count("run-synth") == 0

        tasks = {t["case_id"]: t for t in store.get_review_tasks("run-synth")}
        assert tasks["synth_a"]["decision"] == "approve"
        assert tasks["synth_b"]["decision"] == "reject"
    finally:
        store.close()


def test_inject_empty_list_is_a_noop(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    try:
        assert inject_synthetic_cases(store, "run-empty", []) == 0
        assert store.pending_review_count("run-empty") == 0
    finally:
        store.close()
