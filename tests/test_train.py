"""Self-check for slm_eval.train module."""

from inspect import Parameter, signature

from slm_eval.train import TrainResult, train_lora


def test_train_imports():
    """Verify train_lora is callable and TrainResult is a dataclass with expected fields."""
    assert callable(train_lora), "train_lora should be callable"
    assert hasattr(TrainResult, "__dataclass_fields__"), "TrainResult should be a dataclass"

    expected_fields = {
        "adapter_path",
        "model_path",
        "config_path",
        "trainable_params",
        "total_params",
        "training_config",
        "run_id",
    }
    actual_fields = set(TrainResult.__dataclass_fields__.keys())
    assert actual_fields == expected_fields, (
        f"TrainResult fields mismatch: expected {expected_fields}, got {actual_fields}"
    )


def test_train_lora_keyword_only():
    """Verify train_lora has keyword-only signature (parameters after * are keyword-only)."""
    params = list(signature(train_lora).parameters.values())
    assert params, "train_lora should have parameters"
    assert all(p.kind == Parameter.KEYWORD_ONLY for p in params), (
        "all train_lora parameters should be keyword-only"
    )
