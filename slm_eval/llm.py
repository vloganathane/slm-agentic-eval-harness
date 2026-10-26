"""OpenAI-compatible client + injectable interfaces (SPEC_V3 §3.1, §6.3).

Two real backends behind one protocol so the orchestrator never imports
``openai`` directly and tests inject a scripted client instead of a live server.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    latency_ms: int = 0


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse: ...


class OpenAICompatClient:
    """Talks to llama-server / vLLM / hosted OpenAI-compatible endpoints (§3.1)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: int = 120,
        max_attempts: int = 3,
        backoff_base: float = 2.0,
        usd_per_1k_tokens: float = 0.0,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key or "local", timeout=timeout)
        self._model = model
        self._usd_per_1k = usd_per_1k_tokens
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base

    def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
        retryer = retry(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=self._backoff_base),
            reraise=True,
        )(self._once)
        kwargs: dict[str, Any] = {"model": self._model, "messages": [{"role": "user", "content": prompt}]}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return retryer(prompt, kwargs)

    def _once(self, prompt: str, kwargs: dict[str, Any]) -> LLMResponse:
        start = time.monotonic()
        resp = self._client.chat.completions.create(**kwargs)
        latency = int((time.monotonic() - start) * 1000)
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        tin = getattr(usage, "prompt_tokens", 0) or 0
        tout = getattr(usage, "completion_tokens", 0) or 0
        return LLMResponse(
            text=text,
            tokens_in=tin,
            tokens_out=tout,
            usd=(tin + tout) / 1000.0 * self._usd_per_1k,
            latency_ms=latency,
        )


class FixtureJudge:
    """Judge client for the M1 dev loop without a live endpoint (§10).

    Returns a valid match verdict so scoring, calibration, and the gate run
    end-to-end. Deterministic and free — never used for a real promotion, since
    the κ calibration gate (§6.4) is what makes a judge trustworthy.
    """

    def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
        text = json.dumps({"verdict": "match", "reason": "fixture judge"})
        return LLMResponse(text=text, tokens_in=len(prompt) // 4, tokens_out=len(text) // 4)


class ScriptedClient:
    """Deterministic client for tests/fixture runs: maps prompt → canned response."""

    def __init__(self, responses: dict[str, str] | None = None, default: str = "{}") -> None:
        self._responses = responses or {}
        self._default = default
        self.calls: list[str] = []

    def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
        self.calls.append(prompt)
        text = self._responses.get(prompt, self._default)
        return LLMResponse(text=text, tokens_in=len(prompt) // 4, tokens_out=len(text) // 4)


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a judge/analyst JSON reply, tolerating code fences (§5.4 repair path)."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s[: -len("```")]
        s = s.strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end <= start:
            raise
        obj = json.loads(s[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("expected JSON object")
    return obj
