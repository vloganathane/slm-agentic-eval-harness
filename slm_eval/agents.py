"""Agent contract with two-lane enforcement (SPEC_V3 §7; AC-11).

Agents may *propose*; only the deterministic lane may *dispose*. The core rule
is enforced here, at the context boundary, so a custom advisory agent cannot
write scores or a gate even if it tries:

- deterministic lane → may write scores/gate/checkpoints; failures fail loud
- advisory lane      → may write proposals only; failures are dropped + logged

The pipeline's own analyst/synthetic stages stay in-process; this contract is
what custom, config-declared agents plug into.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from slm_eval.config import EvalConfig
from slm_eval.errors import AgentInputError, SlmEvalError
from slm_eval.llm import LLMClient
from slm_eval.store import Store

logger = logging.getLogger(__name__)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class LaneViolationError(SlmEvalError):
    """An advisory agent attempted a deterministic-lane write (AC-11)."""


class AgentContext(BaseModel):
    """Everything an agent is allowed to see and do.

    ``advisory`` is the lane marker. It is set from the agent by
    :class:`AgentRunner` and is never trusted from the agent itself.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    round: int = 0
    config: EvalConfig
    holdout_path: str  # sealed — the only data a scoring agent may read
    db: Store
    llm: LLMClient
    advisory: bool = False
    run_dir: str | None = None

    def for_agent(self, agent: BaseAgent) -> AgentContext:
        """Return a copy with the lane resolved from the agent's declaration."""
        return self.model_copy(update={"advisory": self.advisory or agent.advisory})

    # -- lane enforcement ---------------------------------------------------

    def _require_deterministic(self, write_kind: str) -> None:
        if self.advisory:
            raise LaneViolationError(
                f"advisory agent cannot write {write_kind}; emit a proposal instead"
            )

    # -- allowed writes -----------------------------------------------------

    def record_scores(self, rows: list[dict[str, Any]]) -> int:
        """Deterministic lane only: persist field-level score rows."""
        self._require_deterministic("scores")
        prepared = []
        for row in rows:
            full = {"run_id": self.run_id, "round": self.round, **row}
            full.setdefault("judge_model", None)
            full.setdefault("prompt_hash", None)
            full.setdefault("latency_ms", None)
            prepared.append(full)
        return self.db.insert_scores(prepared)

    def record_gate(self, payload: dict[str, Any]) -> Path:
        """Deterministic lane only: write ``gate.json`` for this run."""
        self._require_deterministic("gate")
        if self.run_dir is None:
            raise AgentInputError("record_gate requires AgentContext.run_dir")
        path = Path(self.run_dir) / "gate.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return path

    def record_proposal(
        self, agent: str, kind: str, payload: dict[str, Any], round_: int | None = None
    ) -> bool:
        """Both lanes may propose; proposals are never treated as truth."""
        return self.db.insert_proposal(
            self.run_id,
            self.round if round_ is None else round_,
            agent,
            kind,
            json.dumps(payload, sort_keys=True),
        )


class BaseAgent(ABC, Generic[InputT, OutputT]):
    """Minimal in-repo agent interface (SPEC_V3 §7.1)."""

    name: str = "agent"
    version: str = "0"
    advisory: bool = False

    @abstractmethod
    def validate_input(self, ctx: AgentContext, input: InputT) -> None:
        """Raise AgentInputError on invalid input (fail loud)."""

    @abstractmethod
    async def execute(self, ctx: AgentContext, input: InputT) -> OutputT:
        """Produce output. Advisory agents must only propose, never score."""

    def validate_output(self, ctx: AgentContext, output: OutputT) -> None:
        """Default: no-op. Agents may tighten, never loosen."""


class AgentRunner:
    """Run agents with lane-aware failure semantics (SPEC_V3 §7.2)."""

    def __init__(self) -> None:
        self.dropped: list[tuple[str, str]] = []

    async def run(
        self, agent: BaseAgent[InputT, OutputT], ctx: AgentContext, input: InputT
    ) -> OutputT | None:
        lane_ctx = ctx.for_agent(agent)
        try:
            agent.validate_input(lane_ctx, input)
            output = await agent.execute(lane_ctx, input)
        except LaneViolationError:
            # Never swallowed: a lane violation is a bug, not a bad proposal.
            raise
        except Exception as exc:  # lane policy decides the outcome
            if lane_ctx.advisory:
                self.dropped.append((agent.name, str(exc)))
                logger.warning(
                    "advisory agent %s failed; proposal dropped: %s", agent.name, exc
                )
                return None
            raise

        agent.validate_output(lane_ctx, output)
        return output
