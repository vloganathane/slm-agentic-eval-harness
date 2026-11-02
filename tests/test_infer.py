"""Inference prompt construction (SPEC_V3 §5 stage 3).

The extract prompt is the only channel that tells the model-under-test what to
emit, so these tests pin the contract: a declared item schema must reach the
prompt, and a field's validator type must drive its scalar type.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from slm_eval.config import EvalConfig
from slm_eval.infer import _field_spec, build_extract_prompt

BASE = {
    "task_description": "Structured field extraction from receipts",
    "task_type": "extraction",
    "fields": {
        "exact_match_fields": ["total"],
        "subjective_fields": ["line_items"],
        "validators": {"total": {"type": "numeric", "tolerance": 0.01}},
    },
    "strata": {"keys": ["quality"], "mode": "single_key"},
    "split": {"fraction": 0.5, "seed": 42, "strategy": "stratified"},
    "stats": {"seed": 42, "resamples": 1000},
    "thresholds": {"target_f1": 0.85, "failure_threshold": 0.5, "min_per_stratum": 5},
    "judge": {
        "endpoint": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "prompt_version": "v1.0",
        "calibration": {
            "path": "calibration/judge.jsonl",
            "min_pairs": 50,
            "target_kappa": 0.85,
            "recalibrate_every_rounds": 5,
        },
    },
    "inference": {"backend": "llama_server", "base_url": "http://127.0.0.1:8080/v1"},
}


def _cfg(**field_overrides) -> EvalConfig:
    spec = {**BASE, "fields": {**BASE["fields"], **field_overrides}}
    return EvalConfig.model_validate(spec)


class TestItemSchemas:
    def test_schema_reaches_prompt(self):
        cfg = _cfg(item_schemas={"line_items": {"name": "string", "price": "number"}})
        prompt = build_extract_prompt(cfg, "1 COFFEE 4.50")
        assert '"line_items": [{"name": <string>, "price": <number>}]' in prompt

    def test_list_field_without_schema_falls_back_to_string(self):
        cfg = _cfg()
        assert _field_spec(cfg, "line_items") == '  "line_items": <string>'

    def test_rejects_undeclared_field(self):
        with pytest.raises(ValidationError, match="item_schemas for undeclared fields"):
            _cfg(item_schemas={"nope": {"name": "string"}})

    def test_rejects_empty_schema(self):
        with pytest.raises(ValidationError, match="at least one attribute"):
            _cfg(item_schemas={"line_items": {}})

    def test_rejects_unknown_attribute_type(self):
        with pytest.raises(ValidationError, match="not in"):
            _cfg(item_schemas={"line_items": {"name": "blob"}})

    def test_allowed_attribute_types(self):
        cfg = _cfg(
            item_schemas={"line_items": {"name": "string", "price": "number", "qty": "integer"}}
        )
        assert cfg.fields.item_schemas["line_items"]["qty"] == "integer"


class TestScalarTypes:
    def test_numeric_validator_declares_number(self):
        cfg = _cfg()
        assert _field_spec(cfg, "total") == '  "total": <number>'

    def test_field_without_validator_is_string(self):
        cfg = _cfg(exact_match_fields=["total", "note"], validators={"total": {"type": "numeric", "tolerance": 0.0}})
        assert _field_spec(cfg, "note") == '  "note": <string>'

    def test_string_validator_is_string(self):
        cfg = _cfg(
            exact_match_fields=["total", "vendor"],
            validators={
                "total": {"type": "numeric", "tolerance": 0.01},
                "vendor": {"type": "string", "normalize": ["trim"]},
            },
        )
        assert _field_spec(cfg, "vendor") == '  "vendor": <string>'


class TestPrompt:
    def test_prompt_lists_every_field(self):
        cfg = _cfg()
        prompt = build_extract_prompt(cfg, "TEXT HERE")
        assert '"line_items"' in prompt
        assert '"total"' in prompt
        assert "TEXT HERE" in prompt

    def test_prompt_orders_fields_deterministically(self):
        cfg = _cfg(
            exact_match_fields=["total", "invoice_number"],
            validators={"total": {"type": "numeric", "tolerance": 0.0}},
        )
        prompt = build_extract_prompt(cfg, "x")
        assert prompt.index('"invoice_number"') < prompt.index('"line_items"')

    def test_prompt_asks_for_empty_string_when_absent(self):
        cfg = _cfg()
        assert "empty string" in build_extract_prompt(cfg, "x")

    def test_prompt_instructs_json_only(self):
        cfg = _cfg()
        prompt = build_extract_prompt(cfg, "x")
        assert "ONLY a JSON object" in prompt

    def test_prompt_is_reproducible(self):
        cfg = _cfg()
        assert build_extract_prompt(cfg, "x") == build_extract_prompt(cfg, "x")
