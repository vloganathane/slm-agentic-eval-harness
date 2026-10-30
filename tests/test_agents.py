"""Agent contract + two-lane enforcement (SPEC_V3 §7.2; AC-11).

The rule under test: advisory agents may *propose* but never *dispose*.
A lane violation is a bug and must never be swallowed as a dropped proposal.
"""

from __future__ import annotations

import asyncio

import pytest

from slm_eval.agents import (
    AgentContext,
    AgentRunner,
    BaseAgent,
    LaneViolationError,
)
from slm_eval.config import EvalConfig
from slm_eval.errors import AgentInputError
from slm_eval.llm import ScriptedClient
from slm_eval.store import Store

CFG = EvalConfig.model_validate(
    {
        "task_description": "field extraction",
        "fields": {"exact_match_fields": ["total"]},
    }
)

SCORE_ROWS = [
    {"case_id": "c1", "field": "total", "correct": 1, "tp": 1, "fp": 0, "fn": 0},
]

GATE_PAYLOAD = {"run_id": "run-a", "decision": {"promote": False, "status": "blocked"}}


def _context(tmp_path, *, advisory: bool = False) -> AgentContext:
    store = Store(tmp_path / "db.sqlite")
    return AgentContext(
        run_id="run-a",
        round=3,
        config=CFG,
        holdout_path=str(tmp_path / "holdout" / "cases.jsonl"),
        db=store,
        llm=ScriptedClient(),
        advisory=advisory,
        run_dir=str(tmp_path / "runs" / "run-a"),
    )


class _AdvisoryScoreHijacker(BaseAgent):
    name = "hijacker"
    advisory = True

    def validate_input(self, ctx, input) -> None:
        pass

    async def execute(self, ctx, input):
        # Tries to dispose of truth despite being advisory.
        ctx.record_scores(SCORE_ROWS)
        return {"scored": True}


class _AdvisoryFailing(BaseAgent):
    name = "flaky"
    advisory = True

    def validate_input(self, ctx, input) -> None:
        pass

    async def execute(self, ctx, input):
        raise RuntimeError("analyst exploded")


class _AdvisoryProposer(BaseAgent):
    name = "proposer"
    advisory = True

    def validate_input(self, ctx, input) -> None:
        pass

    async def execute(self, ctx, input):
        return ctx.record_proposal(self.name, "risk_flag", {"case_id": input})


class _DeterministicFailing(BaseAgent):
    name = "scorer"

    def validate_input(self, ctx, input) -> None:
        pass

    async def execute(self, ctx, input):
        raise RuntimeError("scorer exploded")


class _InvalidInput(BaseAgent):
    name = "strict"
    advisory = True

    def validate_input(self, ctx, input) -> None:
        raise AgentInputError("bad input")

    async def execute(self, ctx, input):  # pragma: no cover - never reached
        return input


class _Echo(BaseAgent):
    name = "echo"

    def validate_input(self, ctx, input) -> None:
        pass

    async def execute(self, ctx, input):
        return input


def test_advisory_cannot_record_scores(tmp_path):
    ctx = _context(tmp_path, advisory=True)

    with pytest.raises(LaneViolationError, match="cannot write scores"):
        ctx.record_scores(SCORE_ROWS)

    assert ctx.db.scores("run-a") == []


def test_advisory_cannot_record_gate(tmp_path):
    ctx = _context(tmp_path, advisory=True)

    with pytest.raises(LaneViolationError, match="cannot write gate"):
        ctx.record_gate(GATE_PAYLOAD)

    assert not (tmp_path / "runs" / "run-a" / "gate.json").exists()


def test_deterministic_lane_writes_scores_and_gate(tmp_path):
    ctx = _context(tmp_path, advisory=False)

    assert ctx.record_scores(SCORE_ROWS) == 1
    rows = ctx.db.scores("run-a")
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-a"
    assert rows[0]["round"] == 3
    assert rows[0]["field"] == "total"

    path = ctx.record_gate(GATE_PAYLOAD)
    assert path == tmp_path / "runs" / "run-a" / "gate.json"
    assert "blocked" in path.read_text()
    ctx.db.close()


def test_both_lanes_may_propose(tmp_path):
    for advisory in (True, False):
        lane_dir = tmp_path / str(advisory)
        lane_dir.mkdir()
        ctx = _context(lane_dir, advisory=advisory)
        assert ctx.record_proposal("agent-x", "risk_flag", {"case_id": "c1"}) is True
        # Content-addressed: the same proposal is idempotent.
        assert ctx.record_proposal("agent-x", "risk_flag", {"case_id": "c1"}) is False
        ctx.db.close()


def test_runner_drops_advisory_failure(tmp_path):
    runner = AgentRunner()
    ctx = _context(tmp_path, advisory=True)

    result = asyncio.run(runner.run(_AdvisoryFailing(), ctx, "c1"))

    assert result is None
    assert runner.dropped == [("flaky", "analyst exploded")]
    ctx.db.close()


def test_runner_never_swallows_lane_violation(tmp_path):
    runner = AgentRunner()
    ctx = _context(tmp_path, advisory=True)

    with pytest.raises(LaneViolationError):
        asyncio.run(runner.run(_AdvisoryScoreHijacker(), ctx, "c1"))

    assert runner.dropped == []
    assert ctx.db.scores("run-a") == []
    ctx.db.close()


def test_runner_derives_lane_from_agent_declaration(tmp_path):
    """A caller cannot downgrade an advisory agent to the deterministic lane."""
    runner = AgentRunner()
    ctx = _context(tmp_path, advisory=False)
    assert ctx.advisory is False

    with pytest.raises(LaneViolationError):
        asyncio.run(runner.run(_AdvisoryScoreHijacker(), ctx, "c1"))

    ctx.db.close()


def test_runner_propagates_deterministic_failure(tmp_path):
    runner = AgentRunner()
    ctx = _context(tmp_path, advisory=False)

    with pytest.raises(RuntimeError, match="scorer exploded"):
        asyncio.run(runner.run(_DeterministicFailing(), ctx, "c1"))

    assert runner.dropped == []
    ctx.db.close()


def test_runner_drops_advisory_input_validation_failure(tmp_path):
    runner = AgentRunner()
    ctx = _context(tmp_path, advisory=True)

    result = asyncio.run(runner.run(_InvalidInput(), ctx, "c1"))

    assert result is None
    assert runner.dropped[0][0] == "strict"
    ctx.db.close()


def test_runner_returns_output_and_allows_advisory_proposals(tmp_path):
    runner = AgentRunner()
    ctx = _context(tmp_path, advisory=True)

    assert asyncio.run(runner.run(_AdvisoryProposer(), ctx, "c1")) is True
    assert runner.dropped == []
    ctx.db.close()


def test_recording_scores_requires_run_dir_for_gate(tmp_path):
    ctx = _context(tmp_path, advisory=False)
    ctx = ctx.model_copy(update={"run_dir": None})

    with pytest.raises(AgentInputError, match="run_dir"):
        ctx.record_gate(GATE_PAYLOAD)

    ctx.db.close()
