"""Shared gate parity fixture: Python is the reference for the Rust port (AC-04/05/08)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_eval.config import ThresholdsConfig
from slm_eval.gate import GateInputs, StratumResult, decide_gate

FIXTURE = Path(__file__).parent / "fixtures" / "gate_cases.json"


def _load():
    return json.loads(FIXTURE.read_text())


def _inputs(raw: dict) -> GateInputs:
    return GateInputs(
        overall=(raw["micro_f1"], raw["ci_lower"], raw["ci_upper"]),
        overall_n=raw["overall_n"],
        strata=[StratumResult(**s) for s in raw["strata"]],
        kappa=raw["kappa"],
        pending_reviews=raw["pending_reviews"],
        run_completed=raw["run_completed"],
        subjective_fields=raw["subjective_fields"],
        defect_rate=raw["defect_rate"],
        defect_n=raw["defect_n"],
        total_n=raw["total_n"],
    )


def test_fixture_matches_python_gate():
    doc = _load()
    thresholds = ThresholdsConfig(**doc["thresholds"])
    for case in doc["cases"]:
        decision = decide_gate(_inputs(case["inputs"]), thresholds)
        expected = case["expected"]
        assert decision.status == expected["status"], case["name"]
        assert decision.promote is expected["promote"], case["name"]
        assert decision.reason == expected["reason"], case["name"]
        assert decision.blockers == expected["blockers"], case["name"]
        assert decision.warnings == expected["warnings"], case["name"]


@pytest.mark.parametrize("case", _load()["cases"], ids=lambda c: c["name"])
def test_fixture_covers_gate_contract(case):
    """Guard against an empty/malformed fixture silently passing parity."""
    assert {"status", "promote", "reason", "blockers", "warnings"} <= set(case["expected"])
