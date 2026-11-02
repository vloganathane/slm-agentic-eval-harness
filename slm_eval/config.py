"""eval_config.yaml schema (SPEC_V3 §8.3) and field validators (§6.1).

Loading: ``EvalConfig.from_yaml(path)`` — raises ``ConfigError`` with a readable
message on any schema violation. ``config_hash`` is the provenance anchor (§4.3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from slm_eval.errors import ConfigError
from slm_eval.hashing import sha256_json

# ---------------------------------------------------------------------------
# Field validators (§6.1)
# ---------------------------------------------------------------------------


class RegexValidator(BaseModel):
    type: Literal["regex"] = "regex"
    pattern: str

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, v: str) -> str:
        import re

        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f"invalid regex: {e}") from e
        return v


class NumericValidator(BaseModel):
    type: Literal["numeric"] = "numeric"
    tolerance: float = 0.0

    @field_validator("tolerance")
    @classmethod
    def _nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError("tolerance must be >= 0")
        return v


class DateValidator(BaseModel):
    type: Literal["date"] = "date"
    formats: list[str] = Field(default_factory=lambda: ["%Y-%m-%d"])

    @field_validator("formats")
    @classmethod
    def _nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one date format required")
        return v


class StringValidator(BaseModel):
    type: Literal["string"] = "string"
    normalize: list[Literal["trim", "collapse_ws", "casefold"]] = Field(
        default_factory=lambda: ["trim", "collapse_ws"]
    )


class ListValidator(BaseModel):
    type: Literal["list"] = "list"
    item_match: Literal["exact"] = "exact"


FieldValidatorSpec = Annotated[
    RegexValidator | NumericValidator | DateValidator | StringValidator | ListValidator,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Config sections
# ---------------------------------------------------------------------------


class FieldsConfig(BaseModel):
    subjective_fields: list[str] = Field(default_factory=list)
    exact_match_fields: list[str] = Field(default_factory=list)
    validators: dict[str, FieldValidatorSpec] = Field(default_factory=dict)
    # Declares the *shape* of a field so the model-under-test is told what to
    # emit. Without this the extract prompt can only name the fields, and a
    # model guesses at the structure (e.g. a bare string instead of a list of
    # objects). Keys must be declared fields; values map attribute -> type.
    item_schemas: dict[str, dict[str, str]] = Field(default_factory=dict)

    @field_validator("item_schemas")
    @classmethod
    def _schema_types(cls, v: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
        allowed = {"string", "number", "integer"}
        for fname, attrs in v.items():
            if not attrs:
                raise ValueError(f"item_schemas[{fname!r}] must declare at least one attribute")
            for attr, typ in attrs.items():
                if typ not in allowed:
                    raise ValueError(
                        f"item_schemas[{fname!r}].{attr!r}: type {typ!r} not in {sorted(allowed)}"
                    )
        return v

    @model_validator(mode="after")
    def _consistent(self) -> FieldsConfig:
        subj, exact = set(self.subjective_fields), set(self.exact_match_fields)
        overlap = subj & exact
        if overlap:
            raise ValueError(f"fields in both subjective and exact_match: {sorted(overlap)}")
        if not subj and not exact:
            raise ValueError("at least one field required (subjective or exact_match)")
        if len(subj | exact) != len(self.subjective_fields) + len(self.exact_match_fields):
            raise ValueError("duplicate field names across lists")
        unknown = set(self.validators) - (subj | exact)
        if unknown:
            raise ValueError(f"validators for undeclared fields: {sorted(unknown)}")
        unknown_schema = set(self.item_schemas) - (subj | exact)
        if unknown_schema:
            raise ValueError(f"item_schemas for undeclared fields: {sorted(unknown_schema)}")
        return self


class StrataConfig(BaseModel):
    keys: list[str] = Field(default_factory=list)
    mode: Literal["single_key"] = "single_key"  # crossed reserved (D-09)
    definitions: str | None = None


class SplitConfig(BaseModel):
    fraction: float = 0.2
    seed: int = 42
    strategy: Literal["stratified"] = "stratified"

    @field_validator("fraction")
    @classmethod
    def _frac(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("holdout fraction must be in (0, 1)")
        return v


class StatsConfig(BaseModel):
    seed: int = 42
    resamples: int = 10_000

    @field_validator("resamples")
    @classmethod
    def _resamples(cls, v: int) -> int:
        if v < 1_000:
            raise ValueError("resamples must be >= 1000 for a usable CI")
        return v


class ThresholdsConfig(BaseModel):
    target_f1: float = 0.85
    failure_threshold: float = 0.5
    min_per_stratum: int = 50

    @field_validator("target_f1", "failure_threshold")
    @classmethod
    def _unit(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("thresholds must be in (0, 1]")
        return v

    @field_validator("min_per_stratum")
    @classmethod
    def _min_n(cls, v: int) -> int:
        if v < 1:
            raise ValueError("min_per_stratum must be >= 1")
        return v

    @model_validator(mode="after")
    def _ordering(self) -> ThresholdsConfig:
        if self.failure_threshold >= self.target_f1:
            raise ValueError("failure_threshold must be < target_f1")
        return self


class CalibrationConfig(BaseModel):
    path: str = "calibration/judge.jsonl"
    min_pairs: int = 50
    target_kappa: float = 0.85
    recalibrate_every_rounds: int = 5

    @field_validator("min_pairs")
    @classmethod
    def _pairs(cls, v: int) -> int:
        if v < 30:
            raise ValueError("min_pairs must be >= 30 (κ is noise below that)")
        return v

    @field_validator("target_kappa")
    @classmethod
    def _kappa(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("target_kappa must be in (0, 1]")
        return v


class JudgeConfig(BaseModel):
    endpoint: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    prompt_version: str = "v1.0"
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)

    @field_validator("api_key_env")
    @classmethod
    def _env_name(cls, v: str) -> str:
        if not v.isidentifier() and not v.replace("_", "").isalnum():
            raise ValueError("api_key_env must be an env var name, not a secret")
        return v


class RetryConfig(BaseModel):
    # A hosted judge under sustained load returns transient 4xx between calls;
    # 3 attempts with a short backoff lost a finished 20-minute run, so the
    # default budget is wide enough to ride out rate limiting.
    max_attempts: int = 6
    backoff_base: float = 3.0

    @field_validator("max_attempts")
    @classmethod
    def _attempts(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_attempts must be >= 1")
        return v


class InferenceConfig(BaseModel):
    backend: Literal["llama_server", "openai", "vllm"] = "llama_server"
    base_url: str = "http://127.0.0.1:8080/v1"
    timeout_seconds: int = 1800
    retry: RetryConfig = Field(default_factory=RetryConfig)
    # vLLM-specific serving options (used only when backend == "vllm")
    vllm_port: int = 8000
    vllm_gpu_memory_utilization: float = 0.9
    vllm_max_model_len: int | None = None
    vllm_dtype: str = "auto"
    vllm_tensor_parallel_size: int = 1


class HumanReviewConfig(BaseModel):
    timeout_hours: int = 24
    notify: list[str] = Field(default_factory=list)


class CostControlConfig(BaseModel):
    max_cost_per_round_usd: float = 10.0
    alert_at_usd: float = 5.0

    @model_validator(mode="after")
    def _ordering(self) -> CostControlConfig:
        if self.alert_at_usd > self.max_cost_per_round_usd:
            raise ValueError("alert_at_usd must be <= max_cost_per_round_usd")
        return self


class ReportingConfig(BaseModel):
    formats: list[Literal["markdown", "json"]] = Field(default_factory=lambda: ["markdown", "json"])


class EvalConfig(BaseModel):
    task_description: str
    task_type: Literal["extraction"] = "extraction"  # v1: extraction only (§2)

    fields: FieldsConfig
    strata: StrataConfig = Field(default_factory=StrataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    stats: StatsConfig = Field(default_factory=StatsConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    human_review: HumanReviewConfig = Field(default_factory=HumanReviewConfig)
    cost_control: CostControlConfig = Field(default_factory=CostControlConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        try:
            raw = yaml.safe_load(Path(path).read_text())
        except OSError as e:
            raise ConfigError(f"cannot read config: {e}") from e
        except yaml.YAMLError as e:
            raise ConfigError(f"invalid YAML: {e}") from e
        if not isinstance(raw, dict):
            raise ConfigError("config must be a YAML mapping")
        try:
            return cls.model_validate(raw)
        except Exception as e:  # pydantic ValidationError
            raise ConfigError(f"config validation failed: {e}") from e

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, default_flow_style=False)
        )

    @property
    def config_hash(self) -> str:
        """Provenance anchor: content hash of the canonical config (§4.3)."""
        return sha256_json(self.model_dump(mode="json"))
