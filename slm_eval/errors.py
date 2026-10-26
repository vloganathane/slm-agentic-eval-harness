"""Exception hierarchy for slm-eval (SPEC_V3 §5.4, §7.1)."""

from __future__ import annotations


class SlmEvalError(Exception):
    """Base class for all harness errors."""


class ConfigError(SlmEvalError):
    """eval_config.yaml is invalid or unreadable."""


class AgentInputError(SlmEvalError):
    """An agent received invalid input; orchestrator marks the stage FAILED (fail loud)."""


class StateConflict(SlmEvalError):
    """Optimistic-locking conflict on run.version (SPEC_V3 §4.3)."""


class SealedArtifactError(SlmEvalError):
    """A write-once artifact (sealed holdout) was modified after sealing (SPEC_V3 §4.3)."""


class LeakageError(SlmEvalError):
    """Train/holdout overlap detected; run aborts before inference (AC-02)."""


class CalibrationError(SlmEvalError):
    """Judge calibration data invalid or insufficient."""
