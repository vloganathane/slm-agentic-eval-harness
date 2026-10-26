"""E2E pipeline tests: golden path, determinism, resume (SPEC_V3 §10; AC-01, AC-03, AC-07)."""

import json
from pathlib import Path

import pytest

from slm_eval import fixture
from slm_eval.config import EvalConfig
from slm_eval.judge import CalibrationStatus
from slm_eval.llm import ScriptedClient
from slm_eval.pipeline import Orchestrator
from slm_eval.store import Store

CFG = EvalConfig.model_validate(
    {
        "task_description": "Structured field extraction from Indian GST invoices",
        "task_type": "extraction",
        "fields": {
            "subjective_fields": ["vendor_name", "address", "description"],
            "exact_match_fields": ["invoice_number", "date", "total", "tax_id", "line_items"],
            "validators": {
                "tax_id": {"type": "regex", "pattern": "^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z].*Z[0-9A-Z]$"},
                "total": {"type": "numeric", "tolerance": 0.01},
                "line_items": {"type": "list", "item_match": "exact"},
                "date": {"type": "date", "formats": ["%Y-%m-%d", "%d/%m/%Y"]},
                "invoice_number": {"type": "string", "normalize": ["trim", "collapse_ws", "casefold"]},
            },
        },
        "strata": {"keys": ["layout_type"], "mode": "single_key"},
        "split": {"fraction": 0.3, "seed": 42, "strategy": "stratified"},
        "stats": {"seed": 42, "resamples": 2000},
        "thresholds": {"target_f1": 0.85, "failure_threshold": 0.5, "min_per_stratum": 5},
        "judge": {"model": "fixture-judge", "prompt_version": "v1.0"},
    }
)

GOOD_CAL = CalibrationStatus(kappa=0.92, pairs=60, ci_lower=0.88, ci_upper=0.96, model="fixture-judge")


def _cfg_with(**overrides) -> EvalConfig:
    """Rebuild CFG with top-level section overrides (e.g. split fraction)."""
    data = CFG.model_dump()
    data.update(overrides)
    return EvalConfig.model_validate(data)


def _mk_dataset(tmp_path: Path, n: int = 120, seed: int = 42) -> str:
    path, _ = fixture.write_dataset(tmp_path / "data", count=n, seed=seed)
    return str(path)


def _judge_client(gold_by_prompt_match: bool = True) -> ScriptedClient:
    # Default judge says "match" — the fixture's own gold is the prediction,
    # so exact fields all pass and the judge accepts the (identical) subjective fields.
    return ScriptedClient(default=json.dumps({"verdict": "match", "reason": "identical"}))


def _run(tmp_path, dataset, run_id, predictions=None, resume=False, cfg=CFG):
    store = Store(tmp_path / "db.sqlite")
    try:
        orch = Orchestrator(
            cfg, store, tmp_path / "runs" / run_id, _judge_client(),
            code_version="git:fixture", resume=resume,
        )
        return orch.run(
            model_path="fixture.gguf", model_sha256="m" * 64,
            dataset_path=dataset, predictions=predictions, calibration=GOOD_CAL,
        )
    finally:
        store.close()


def test_golden_path_promotes(tmp_path):
    """AC-01: model+dataset → gate.json with promote when quality is high."""
    ds = _mk_dataset(tmp_path)
    gate = _run(tmp_path, ds, "run-perfect")
    assert gate["decision"]["status"] == "promote"
    assert gate["decision"]["promote"] is True
    assert gate["metrics"]["ci_lower"] >= CFG.thresholds.target_f1
    assert gate["holdout"]["n"] > 0
    assert gate["judge"]["kappa"] == pytest.approx(0.92)


def test_gate_json_written_and_parseable(tmp_path):
    ds = _mk_dataset(tmp_path)
    _run(tmp_path, ds, "run-artifact")
    p = tmp_path / "runs" / "run-artifact" / "gate.json"
    assert p.exists()
    gate = json.loads(p.read_text())
    for key in ("run_id", "decision", "metrics", "strata", "judge", "provenance"):
        assert key in gate


def test_low_quality_model_blocks_on_ci(tmp_path):
    """AC-04/AC-06: a bad model's CI lower bound stays under target → blocked."""
    ds = _mk_dataset(tmp_path, n=120)
    cases = fixture.read_dataset(ds)
    # Corrupt every prediction so scoring collapses.
    bad = {c["case_id"]: {"invoice_number": "WRONG", "date": "nope", "total": -1,
                          "tax_id": "bad", "line_items": [], "vendor_name": "x",
                          "address": "x", "description": "x"} for c in cases}
    gate = _run(tmp_path, ds, "run-bad", predictions=bad)
    assert gate["decision"]["promote"] is False
    assert gate["decision"]["status"] == "blocked"
    assert gate["metrics"]["ci_upper"] < CFG.thresholds.target_f1


def test_determinism_same_inputs_same_gate_hash(tmp_path):
    """AC-03: identical inputs + seeds + judge cache → identical gate minus timestamps."""
    ds = _mk_dataset(tmp_path)
    g1 = _run(tmp_path, ds, "run-det-1")
    g2 = _run(tmp_path, ds, "run-det-2")

    def canon(g):
        g = dict(g)
        g.pop("run_id", None)  # timestamp-derived
        return json.dumps(g, sort_keys=True)

    assert canon(g1) == canon(g2)


def test_resume_skips_completed_stages(tmp_path):
    """AC-07: a second run with resume=True must not recompute fresh stages."""
    ds = _mk_dataset(tmp_path)
    _run(tmp_path, ds, "run-resume")
    store = Store(tmp_path / "db.sqlite")
    try:
        n_before = store._conn.execute("SELECT COUNT(*) FROM stage_checkpoint").fetchone()[0]
        # Run again against the same run dir with resume.
        orch = Orchestrator(
            CFG, store, tmp_path / "runs" / "run-resume", _judge_client(), resume=True,
        )
        orch.run(model_path="fixture.gguf", model_sha256="m" * 64,
                 dataset_path=ds, calibration=GOOD_CAL)
        events = (tmp_path / "runs" / "run-resume" / "events.jsonl").read_text()
    finally:
        store.close()
    assert "skipped_resume" in events  # stages were skipped, not redone


def test_per_stratum_metrics_present(tmp_path):
    """AC-08: strata appear in gate with status."""
    ds = _mk_dataset(tmp_path, n=150)
    gate = _run(tmp_path, ds, "run-strata")
    assert gate["strata"], "expected per-stratum results"
    for s in gate["strata"]:
        assert set(s) >= {"key", "n", "micro_f1", "ci_lower", "ci_upper", "status"}
        assert s["status"] in ("ok", "warning", "blocker", "insufficient_n")


def test_cost_ledger_sums_into_gate(tmp_path):
    """AC-09: gate.cost_usd equals the store's ledger total."""
    ds = _mk_dataset(tmp_path, n=60)
    gate = _run(tmp_path, ds, "run-cost")
    store = Store(tmp_path / "db.sqlite")
    try:
        total = store.cost_total(gate["run_id"])
    finally:
        store.close()
    assert gate["cost_usd"] == pytest.approx(total, abs=1e-6)


def test_leakage_aborts_before_inference(tmp_path):
    """Injected overlap must abort at split (exit-2 path), never reach gate."""
    from slm_eval.errors import LeakageError

    # Force the exact-duplicate pair across the split boundary: pin the original
    # to its own stratum and the copy to another, then split hard enough that each
    # stratum yields both train and holdout members (a dup on one side is not leakage).
    ds = _mk_dataset(tmp_path, n=40)
    cases = fixture.read_dataset(ds)
    original = dict(cases[0])
    original["strata"] = dict(original["strata"])
    original["strata"]["layout_type"] = "clean"
    dupe = dict(original)
    dupe["case_id"] = "inv-dupe"
    dupe["strata"] = dict(original["strata"])
    dupe["strata"]["layout_type"] = "handwritten"  # different stratum → opposite side
    cases = [original, dupe] + [c for c in cases if c["case_id"] != cases[0]["case_id"]]

    # Overwrite the fixture dataset to include the injected cross-side duplicate.
    bad_ds = str(tmp_path / "bad_data")
    Path(bad_ds).mkdir(parents=True, exist_ok=True)
    p = Path(bad_ds) / "cases.jsonl"
    p.write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in cases))

    # fraction=0.5 makes each of the two pinned strata contribute both sides.
    # seed=0 is chosen so the two pinned cases land on opposite sides of the split
    # (the seeded shuffle must separate them for the dup to constitute leakage).
    leaking_cfg = _cfg_with(split={"fraction": 0.5, "seed": 0, "strategy": "stratified"})

    store = Store(tmp_path / "db.sqlite")
    try:
        orch = Orchestrator(leaking_cfg, store, tmp_path / "runs" / "run-leak", _judge_client())
        with pytest.raises(LeakageError):
            orch.run(model_path="m.gguf", model_sha256="m" * 64, dataset_path=bad_ds,
                     calibration=GOOD_CAL)
    finally:
        store.close()
