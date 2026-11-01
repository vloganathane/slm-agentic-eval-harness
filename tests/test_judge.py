"""Judge tests: prompt hashing cache, parse/repair, κ calibration gate (§6.3, §6.4)."""

import json

import pytest

from slm_eval.config import JudgeConfig
from slm_eval.errors import CalibrationError
from slm_eval.judge import (
    PROMPT_VERSION,
    CalibrationGate,
    Judge,
    build_prompt,
    prompt_hash,
)
from slm_eval.llm import ScriptedClient


def _verdict(v: str, reason: str = "ok") -> str:
    return json.dumps({"verdict": v, "reason": reason})


def test_prompt_hash_stable_and_sensitive():
    cfg = JudgeConfig()
    a = prompt_hash(cfg, "vendor_name", "Acme", "Acme Corp")
    b = prompt_hash(cfg, "vendor_name", "Acme", "Acme Corp")
    assert a == b
    assert a != prompt_hash(cfg, "vendor_name", "Acme", "Other")
    assert a != prompt_hash(cfg, "address", "Acme", "Acme Corp")
    # Version keys the cache: bumping template invalidates (§6.3)
    assert a != prompt_hash(JudgeConfig(prompt_version="v9.9"), "vendor_name", "Acme", "Acme Corp")


def test_build_prompt_embeds_field_gold_pred():
    p = build_prompt("task", "total", 100, 101)
    assert "task" in p and "total" in p and "100" in p and "101" in p
    assert PROMPT_VERSION  # template version exists


def test_judge_caches_by_prompt_hash():
    client = ScriptedClient(default=_verdict("match"))
    j = Judge(JudgeConfig(), client, "task")
    j.judge_field("vendor_name", "Acme", "Acme Corp")
    j.judge_field("vendor_name", "Acme", "Acme Corp")
    assert len(client.calls) == 1  # second call served from cache


def test_judge_verdict_parsing():
    client = ScriptedClient(default=_verdict("mismatch", "different entity"))
    j = Judge(JudgeConfig(), client, "task")
    v = j.judge_field("vendor_name", "Acme", "Bharat")
    assert not v.correct and v.reason == "different entity"


def test_judge_tolerates_code_fenced_json():
    client = ScriptedClient(default="```json\n" + _verdict("match") + "\n```")
    j = Judge(JudgeConfig(), client, "task")
    assert j.judge_field("vendor_name", "a", "a").correct


def test_judge_repairs_then_raises_on_garbage():
    client = ScriptedClient(default="not json at all")
    j = Judge(JudgeConfig(), client, "task")
    with pytest.raises(ValueError, match="malformed JSON"):
        j.judge_field("vendor_name", "a", "b", max_repairs=1)
    # initial call + 1 repair
    assert len(client.calls) == 2


def test_calibration_gate_passes_high_kappa():
    gate = CalibrationGate(JudgeConfig())
    human = ["match", "mismatch"] * 30
    judge = list(human)
    status = gate.evaluate(human, judge)
    blocked, _reason = gate.blocks_promotion(status)
    assert not blocked and status.kappa == pytest.approx(1.0)


def test_calibration_gate_blocks_low_kappa():
    gate = CalibrationGate(JudgeConfig())
    human = ["match", "mismatch"] * 30
    # Systematic disagreement → low/negative κ
    judge = ["mismatch" if h == "match" else "match" for h in human]
    status = gate.evaluate(human, judge)
    blocked, reason = gate.blocks_promotion(status)
    assert blocked and "kappa_below_target" in reason


def test_calibration_gate_blocks_too_few_pairs():
    gate = CalibrationGate(JudgeConfig())  # min_pairs default 50
    # Both labels present (κ defined) but only 20 pairs < 50 → blocked on count.
    human = (["match", "mismatch"] * 10)
    status = gate.evaluate(human, human)
    blocked, reason = gate.blocks_promotion(status)
    assert blocked and "below_min_pairs" in reason


def test_calibration_length_mismatch_raises():
    gate = CalibrationGate(JudgeConfig())
    with pytest.raises(CalibrationError):
        gate.evaluate(["match", "mismatch"], ["match"])


def test_calibration_constant_agreement_blocks_as_insufficient():
    gate = CalibrationGate(JudgeConfig())
    status = gate.evaluate(["match"] * 60, ["match"] * 60)  # κ undefined (nan)
    blocked, reason = gate.blocks_promotion(status)
    assert blocked and "insufficient_signal" in reason
