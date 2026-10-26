"""Store tests: optimistic locking, sealed holdout, idempotent scores (SPEC_V3 §4)."""

import pytest

from slm_eval.errors import SealedArtifactError, StateConflict
from slm_eval.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


def _mk_run(store: Store, run_id: str = "r1") -> None:
    store.create_run(
        run_id=run_id, model_path="m.gguf", model_sha256="a" * 64,
        dataset_path="d/", dataset_hash="b" * 64, config_hash="c" * 64,
        code_version="git:test",
    )


def test_create_and_get_run(store):
    _mk_run(store)
    run = store.get_run("r1")
    assert run["status"] == "created"
    assert run["version"] == 0


def test_optimistic_lock_blocks_stale_write(store):
    _mk_run(store)
    v = store.update_run("r1", 0, status="running")
    assert v == 1
    # A second writer still holding version 0 must be rejected:
    with pytest.raises(StateConflict):
        store.update_run("r1", 0, status="completed")
    assert store.get_run("r1")["status"] == "running"


def test_optimistic_lock_retry_with_fresh_version_succeeds(store):
    _mk_run(store)
    v = store.update_run("r1", 0, status="running")
    assert store.update_run("r1", v, status="completed") == v + 1
    assert store.get_run("r1")["status"] == "completed"


def test_immutable_fields_rejected(store):
    _mk_run(store)
    with pytest.raises(ValueError):
        store.update_run("r1", 0, model_path="other.gguf")


def test_invalid_status_rejected(store):
    _mk_run(store)
    with pytest.raises(ValueError):
        store.update_run("r1", 0, status="exploded")


def test_sealed_holdout_is_write_once(store):
    _mk_run(store)
    store.seal_holdout("r1", "h" * 64, expected_version=0)
    # Same hash again: idempotent no-op-ish (allowed, same value)
    store.seal_holdout("r1", "h" * 64, expected_version=1)
    # Different hash: hard error (§4.3)
    with pytest.raises(SealedArtifactError):
        store.seal_holdout("r1", "x" * 64, expected_version=1)


def test_stage_freshness_rule(store):
    _mk_run(store)
    assert not store.stage_is_fresh("r1", "split", "in1")
    store.upsert_checkpoint("r1", "split", "running", input_hash="in1", started=True)
    assert not store.stage_is_fresh("r1", "split", "in1")  # not completed yet
    store.upsert_checkpoint("r1", "split", "completed", input_hash="in1", completed=True)
    assert store.stage_is_fresh("r1", "split", "in1")      # completed + hash match
    assert not store.stage_is_fresh("r1", "split", "in2")  # input changed → redo


def test_checkpoint_preserves_original_start_time(store):
    _mk_run(store)
    store.upsert_checkpoint("r1", "infer", "running", input_hash="i", started=True)
    first = store.get_checkpoint("r1", "infer")["started_at"]
    store.upsert_checkpoint("r1", "infer", "completed", input_hash="i", completed=True)
    cp = store.get_checkpoint("r1", "infer")
    assert cp["started_at"] == first  # not overwritten on later upserts
    assert cp["completed_at"] is not None


def test_score_insert_is_idempotent(store):
    _mk_run(store)
    row = {"run_id": "r1", "round": 1, "case_id": "inv-00001", "field": "total",
           "correct": 1, "tp": 1, "fp": 0, "fn": 0,
           "judge_model": "gpt-4o-mini", "prompt_hash": "p" * 64, "latency_ms": 120}
    assert store.insert_scores([row, dict(row)]) == 1  # duplicate collapsed by UNIQUE
    assert len(store.scores("r1")) == 1


def test_proposals_dedup_by_content(store):
    _mk_run(store)
    assert store.insert_proposal("r1", 1, "analyst", "error_category", '{"k":"typo"}')
    assert not store.insert_proposal("r1", 1, "analyst", "error_category", '{"k":"typo"}')
    assert store.insert_proposal("r1", 1, "analyst", "error_category", '{"k":"layout"}')


def test_cost_ledger_totals(store):
    _mk_run(store)
    store.add_cost("r1", "score_judge", 0.5, 100, 50)
    store.add_cost("r1", "infer", 0.25, 200, 100)
    assert store.cost_total("r1") == pytest.approx(0.75)
