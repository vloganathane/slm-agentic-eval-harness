"""CLI report + inspect: provenance artifacts are readable after a run (AC-10)."""

from __future__ import annotations

import json
from argparse import Namespace

from slm_eval.cli import EXIT_ERROR, EXIT_PROMOTE, _cmd_inspect, _cmd_report
from slm_eval.config import EvalConfig

GATE = {
    "run_id": "cli-test",
    "schema_version": 1,
    "model": {"path": "model.gguf", "sha256": "abc123"},
    "holdout": {"hash": "holdout-hash", "n": 3},
    "provenance": {"config_hash": "cfg-hash", "code_version": "git:test"},
    "decision": {"status": "blocked", "promote": False, "reason": "ci_lower_below_target"},
    "metrics": {
        "micro_f1": 0.42, "ci_lower": 0.30, "ci_upper": 0.54,
        "resamples": 100, "seed": 42, "n": 3,
    },
    "thresholds": {"micro_f1": 0.85},
    "strata": [],
    "judge": {"kappa": 0.9, "pairs": 50, "model": "fixture-judge"},
    "cost_usd": 0.0123,
    "blockers": ["ci_lower_below_target"],
    "warnings": ["cost_alert"],
    "dataset_defect": {"rate": 0.0, "n": 0},
}


def _write_config(tmp_path) -> str:
    cfg = EvalConfig.model_validate(
        {
            "task_description": "field extraction",
            "fields": {"exact_match_fields": ["total"]},
        }
    )
    path = tmp_path / "eval_config.yaml"
    cfg.to_yaml(path)
    return str(path)


def _write_run(tmp_path):
    run_dir = tmp_path / "runs" / "cli-test"
    run_dir.mkdir(parents=True)
    (run_dir / "gate.json").write_text(json.dumps(GATE))
    (run_dir / "predictions.jsonl").write_text(
        json.dumps({"case_id": "c1", "prediction": {"total": "1.00"}}) + "\n"
    )
    (run_dir / "scores.jsonl").write_text(
        json.dumps({"case_id": "c1", "field": "total", "outcome": "match"}) + "\n"
    )
    (run_dir / "synthetic_proposals.jsonl").write_text(
        json.dumps({"case_id": "synth_1", "state": "proposed"}) + "\n"
    )
    (run_dir / "events.jsonl").write_text(
        json.dumps({"ts": "2026-01-01T00:00:00Z", "stage": "gate", "event": "completed"}) + "\n"
    )
    return run_dir


def test_report_writes_markdown_with_gate_and_provenance(tmp_path, capsys):
    run_dir = _write_run(tmp_path)
    config = _write_config(tmp_path)
    out = tmp_path / "report.md"

    code = _cmd_report(Namespace(run_dir=str(run_dir), config=config, output=str(out)))

    assert code == EXIT_PROMOTE
    text = out.read_text()
    assert "cli-test" in text
    assert "blocked" in text
    assert "0.42" in text
    assert "cfg-hash" in text
    assert "Report written to" in capsys.readouterr().out


def test_report_without_config_skips_review_lookup(tmp_path):
    run_dir = _write_run(tmp_path)
    out = tmp_path / "report.md"

    code = _cmd_report(Namespace(run_dir=str(run_dir), config=None, output=str(out)))

    assert code == EXIT_PROMOTE
    assert out.exists()
    assert "cli-test" in out.read_text()


def test_report_missing_gate_returns_error(tmp_path, capsys):
    run_dir = tmp_path / "runs" / "empty"
    run_dir.mkdir(parents=True)

    code = _cmd_report(Namespace(run_dir=str(run_dir), config=None, output=None))

    assert code == EXIT_ERROR
    assert "Gate file not found" in capsys.readouterr().err


def test_train_rejects_budget_below_eval_round_cap(tmp_path, capsys):
    """--max-cost is a real guard: the eval round must fit the training budget."""
    from slm_eval.cli import _cmd_train

    config = _write_config(tmp_path)  # default cost cap is $10
    code = _cmd_train(Namespace(config=config, max_cost=1.0))

    assert code == EXIT_ERROR
    assert "exceeds --max-cost" in capsys.readouterr().err


def test_inspect_prints_artifacts_and_timeline(tmp_path, capsys):
    run_dir = _write_run(tmp_path)

    code = _cmd_inspect(Namespace(run_dir=str(run_dir)))

    assert code == EXIT_PROMOTE
    out = capsys.readouterr().out
    assert "== gate.json ==" in out
    assert "== predictions.jsonl ==" in out
    assert "== scores.jsonl ==" in out
    assert "== synthetic_proposals.jsonl ==" in out
    assert "== events (1 lines) ==" in out
    assert "gate" in out and "completed" in out
