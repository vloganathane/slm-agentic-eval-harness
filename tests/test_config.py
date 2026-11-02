"""Config schema tests (SPEC_V3 §8.3, §6.1)."""

import pytest
import yaml
from pydantic import ValidationError

from slm_eval.config import EvalConfig
from slm_eval.errors import ConfigError

VALID = {
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
            "invoice_number": {
                "type": "string", "normalize": ["trim", "collapse_ws", "casefold"]
            },
        },
    },
    "strata": {"keys": ["layout_type", "language", "quality"], "mode": "single_key"},
    "split": {"fraction": 0.2, "seed": 42, "strategy": "stratified"},
    "stats": {"seed": 42, "resamples": 10000},
    "thresholds": {"target_f1": 0.85, "failure_threshold": 0.5, "min_per_stratum": 50},
    "judge": {
        "endpoint": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "prompt_version": "v1.3",
        "calibration": {
            "path": "calibration/judge.jsonl", "min_pairs": 50,
            "target_kappa": 0.85, "recalibrate_every_rounds": 5,
        },
    },
    "inference": {
        "backend": "llama_server", "base_url": "http://127.0.0.1:8080/v1",
        "timeout_seconds": 1800, "retry": {"max_attempts": 3, "backoff_base": 2},
    },
    "human_review": {"timeout_hours": 24, "notify": []},
    "cost_control": {"max_cost_per_round_usd": 10.0, "alert_at_usd": 5.0},
    "reporting": {"formats": ["markdown", "json"]},
}


def _cfg(**overrides) -> EvalConfig:
    raw = {**VALID, **overrides}
    return EvalConfig.model_validate(raw)


def test_valid_config_loads():
    cfg = _cfg()
    assert cfg.task_type == "extraction"
    assert cfg.thresholds.target_f1 == 0.85
    assert cfg.strata.mode == "single_key"
    assert cfg.stats.resamples == 10000


def test_config_hash_is_stable_and_sensitive():
    a, b = _cfg(), _cfg()
    assert a.config_hash == b.config_hash  # deterministic
    c = _cfg(thresholds={"target_f1": 0.9, "failure_threshold": 0.5, "min_per_stratum": 50})
    assert a.config_hash != c.config_hash  # sensitive to change


def test_from_yaml_round_trip(tmp_path):
    p = tmp_path / "eval_config.yaml"
    p.write_text(yaml.safe_dump(VALID, sort_keys=False))
    cfg = EvalConfig.from_yaml(p)
    out = tmp_path / "snapshot.yaml"
    cfg.to_yaml(out)
    assert EvalConfig.from_yaml(out).config_hash == cfg.config_hash


def test_overlap_between_subjective_and_exact_rejected():
    raw = {**VALID, "fields": {**VALID["fields"], "subjective_fields": ["vendor_name", "date"]}}
    with pytest.raises(ValidationError):
        EvalConfig.model_validate(raw)


def test_validator_for_undeclared_field_rejected():
    raw = {
        **VALID,
        "fields": {**VALID["fields"], "validators": {"nope": {"type": "string"}}},
    }
    with pytest.raises(ValidationError):
        EvalConfig.model_validate(raw)


def test_empty_fields_rejected():
    with pytest.raises(ValidationError):
        EvalConfig.model_validate({**VALID, "fields": {"subjective_fields": [], "exact_match_fields": []}})


def test_invalid_regex_rejected():
    raw = {
        **VALID,
        "fields": {**VALID["fields"],
                   "validators": {"tax_id": {"type": "regex", "pattern": "(["}}},
    }
    with pytest.raises(ValidationError):
        EvalConfig.model_validate(raw)


def test_bad_threshold_ordering_rejected():
    with pytest.raises(ValidationError):
        _cfg(thresholds={"target_f1": 0.5, "failure_threshold": 0.8, "min_per_stratum": 50})


def test_bad_split_fraction_rejected():
    with pytest.raises(ValidationError):
        _cfg(split={"fraction": 1.5, "seed": 42, "strategy": "stratified"})


def test_low_resamples_rejected():
    with pytest.raises(ValidationError):
        _cfg(stats={"seed": 42, "resamples": 100})


def test_crossed_strata_mode_rejected():
    with pytest.raises(ValidationError):
        _cfg(strata={"keys": ["a"], "mode": "crossed"})


def test_from_yaml_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        EvalConfig.from_yaml(tmp_path / "nope.yaml")


def test_from_yaml_garbage(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("just a string")
    with pytest.raises(ConfigError):
        EvalConfig.from_yaml(p)


def test_label_fields_subset_accepted():
    cfg = _cfg(fields={**VALID["fields"], "label_fields": ["tax_id"]})
    assert cfg.fields.label_fields == ["tax_id"]


def test_label_fields_must_be_exact_match():
    # subjective fields are judge-scored — a label report over judge verdicts
    # would mix binary verdicts with class labels.
    with pytest.raises(ValidationError, match="label_fields must be exact_match_fields"):
        _cfg(fields={**VALID["fields"], "label_fields": ["vendor_name"]})


def test_label_fields_must_be_declared():
    with pytest.raises(ValidationError, match="label_fields must be exact_match_fields"):
        _cfg(fields={**VALID["fields"], "label_fields": ["nope"]})


def test_duplicate_label_fields_rejected():
    with pytest.raises(ValidationError, match="duplicate label_fields"):
        _cfg(fields={**VALID["fields"], "label_fields": ["tax_id", "tax_id"]})


def test_empty_label_fields_is_default():
    cfg = _cfg()
    assert cfg.fields.label_fields == []
