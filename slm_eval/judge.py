"""Judge agent (SPEC_V3 §6.3, §6.4): versioned prompt, cached, κ-gated.

Two lanes keep the measurement honest:
- ``judge_field`` is a *deterministic lane* scorer: given (gold, pred) it returns
  a verdict, cached by prompt_hash so reruns are byte-identical (AC-03).
- ``CalibrationGate`` decides whether the judge's verdicts may gate promotion at
  all (AC-05): κ ≥ target on ≥ min_pairs human-labeled pairs, else blocked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from slm_eval.config import JudgeConfig
from slm_eval.errors import CalibrationError
from slm_eval.hashing import sha256_json, sha256_text
from slm_eval.llm import LLMClient, parse_json_response
from slm_eval.stats import cohen_kappa, kappa_ci

# Versioned template (§6.3). Bump PROMPT_VERSION on any change — it keys the cache.
PROMPT_VERSION = "v1.0"

PROMPT_TEMPLATE = """You are a strict judge for {task_description}.

Field: {field}
The human-written correct answer (gold) and the model's answer (pred) are given.
Decide whether the model's answer is an acceptable match for the gold answer.

Rules:
- Accept minor whitespace/case differences.
- For addresses/names, accept if the core entity matches.
- Reject if a key value (number, ID, date) differs at all.
- Reject if the model's answer is empty or clearly unrelated.

gold: {gold}
pred: {pred}

Reply with ONLY a JSON object: {{"verdict": "match" | "mismatch", "reason": "<short>"}}"""


def build_prompt(task_description: str, field: str, gold: Any, pred: Any) -> str:
    return PROMPT_TEMPLATE.format(
        task_description=task_description,
        field=field,
        gold=json.dumps(gold, ensure_ascii=False, sort_keys=True),
        pred=json.dumps(pred, ensure_ascii=False, sort_keys=True),
    )


def prompt_hash(cfg: JudgeConfig, field: str, gold: Any, pred: Any) -> str:
    """Cache key: template_version + field + gold + pred + model (§6.3, AC-03)."""
    return sha256_json(
        {
            "template": PROMPT_VERSION,
            "prompt_version": cfg.prompt_version,
            "model": cfg.model,
            "field": field,
            "gold": gold,
            "pred": pred,
        }
    )


@dataclass(frozen=True)
class JudgeVerdict:
    verdict: str  # "match" | "mismatch"
    reason: str
    phash: str
    latency_ms: int
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0

    @property
    def correct(self) -> bool:
        return self.verdict == "match"


class Judge:
    def __init__(self, cfg: JudgeConfig, client: LLMClient, task_description: str) -> None:
        self.cfg = cfg
        self.client = client
        self.task = task_description
        self._cache: dict[str, JudgeVerdict] = {}

    def judge_field(self, field: str, gold: Any, pred: Any, *, max_repairs: int = 2) -> JudgeVerdict:
        ph = prompt_hash(self.cfg, field, gold, pred)
        if ph in self._cache:
            return self._cache[ph]

        prompt = build_prompt(self.task, field, gold, pred)
        resp = self.client.complete(prompt, json_mode=True)
        verdict, reason = self._parse(resp.text)
        attempts = 0
        while verdict is None and attempts < max_repairs:
            # Repair path (§5.4): re-request, reminding of the schema.
            resp = self.client.complete(
                prompt + "\n\nReturn ONLY the JSON object with keys verdict and reason.",
                json_mode=True,
            )
            verdict, reason = self._parse(resp.text)
            attempts += 1
        if verdict is None:
            raise ValueError(f"judge returned malformed JSON after {max_repairs} repairs")

        out = JudgeVerdict(
            verdict=verdict,
            reason=reason,
            phash=ph,
            latency_ms=resp.latency_ms,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            usd=resp.usd,
        )
        self._cache[ph] = out
        return out

    @staticmethod
    def _parse(text: str) -> tuple[str | None, str]:
        try:
            obj = parse_json_response(text)
        except (ValueError, json.JSONDecodeError):
            return None, ""
        verdict = str(obj.get("verdict", "")).lower().strip()
        if verdict not in ("match", "mismatch"):
            return None, ""
        return verdict, str(obj.get("reason", ""))[:500]


@dataclass(frozen=True)
class CalibrationStatus:
    kappa: float
    pairs: int
    ci_lower: float
    ci_upper: float
    model: str

    @property
    def is_valid(self) -> bool:
        return not _is_nan(self.kappa)


import math


def _is_nan(x: float) -> bool:
    return math.isnan(x)


class CalibrationGate:
    """Enforces §6.4: κ ≥ target on ≥ min_pairs before any promotion (AC-05)."""

    def __init__(self, cfg: JudgeConfig) -> None:
        self.cfg = cfg.calibration
        self.model = cfg.model

    def evaluate(
        self,
        human_labels: list[str],
        judge_labels: list[str],
        resamples: int = 2_000,
    ) -> CalibrationStatus:
        pairs = len(human_labels)
        if pairs != len(judge_labels):
            raise CalibrationError(f"label count mismatch: {pairs} vs {len(judge_labels)}")
        kappa = cohen_kappa(human_labels, judge_labels)
        lo, hi = kappa_ci(human_labels, judge_labels, resamples=resamples)
        return CalibrationStatus(
            kappa=kappa, pairs=pairs, ci_lower=lo, ci_upper=hi, model=self.model
        )

    def blocks_promotion(self, status: CalibrationStatus) -> tuple[bool, str]:
        """Return (blocked, reason). Missing/expired/low calibration blocks."""
        if not status.is_valid:
            return True, "calibration_insufficient_signal"
        if status.pairs < self.cfg.min_pairs:
            return (
                True,
                f"calibration_below_min_pairs: {status.pairs} < {self.cfg.min_pairs}",
            )
        if status.kappa < self.cfg.target_kappa:
            return (
                True,
                f"kappa_below_target: {status.kappa:.3f} < {self.cfg.target_kappa}",
            )
        return False, ""


def calibration_pair_hash(field: str, gold: Any, pred: Any, human: str, judge: str) -> str:
    return sha256_text(f"{field}|{gold}|{pred}|{human}|{judge}")[:32]
