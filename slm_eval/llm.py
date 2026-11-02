"""OpenAI-compatible client + injectable interfaces (SPEC_V3 §3.1, §6.3).

Two real backends behind one protocol so the orchestrator never imports
``openai`` directly and tests inject a scripted client instead of a live server.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from tenacity import retry, stop_after_attempt, wait_exponential

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from tavily import TavilyClient
    HAS_TAVILY = True
except ImportError:
    HAS_TAVILY = False


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    latency_ms: int = 0


@runtime_checkable
class LLMClient(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.0,
        images: list[str] | None = None,
    ) -> LLMResponse: ...
    def web_search(self, query: str, max_results: int = 5) -> list[dict[str, str]]: ...


class WebSearchProvider:
    """Web search provider for real-world data fetching."""
    
    def __init__(self, provider: str = "tavily", api_key: str | None = None):
        self.provider = provider
        self.api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY")
        self._client = None
        
        if provider == "tavily" and HAS_TAVILY and self.api_key:
            self._client = TavilyClient(api_key=self.api_key)
        elif provider == "crw":
            # crw is a CLI tool - check if it's available
            self._client = self._check_crw_available()
    
    def _check_crw_available(self) -> bool:
        """Check if crw CLI is available."""
        try:
            result = subprocess.run(
                ["crw", "--version"], capture_output=True, timeout=5, check=True
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return False
    
    def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        if self.provider == "tavily" and self._client:
            try:
                results = self._client.search(query, max_results=max_results)
                return [
                    {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
                    for r in results.get("results", [])
                ]
            except Exception:  # search failure: fall through to next provider  # noqa: S110, BLE001
                pass
        elif self.provider == "crw" and self._client:
            try:
                # crw search CLI: crw search "query" --json --limit N
                cmd = ["crw", "search", query, "--json", "--limit", str(max_results), "--format", "json"]
                env = os.environ.copy()
                if self.api_key:
                    env["CRW_API_KEY"] = self.api_key
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env, check=True)
                if result.returncode == 0 and result.stdout.strip():
                    data = json.loads(result.stdout)
                    # crw returns JSON array or object with results
                    if isinstance(data, list):
                        return [
                            {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("snippet", r.get("content", ""))}
                            for r in data
                        ]
                    elif isinstance(data, dict) and "results" in data:
                        return [
                            {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("snippet", r.get("content", ""))}
                            for r in data["results"]
                        ]
            except (subprocess.TimeoutExpired, json.JSONDecodeError, subprocess.CalledProcessError):
                pass
        # Fallback to simple HTTP fetch if no provider
        return []
    
    def fetch(self, url: str) -> str:
        if not HAS_HTTPX:
            return ""
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url)
                resp.raise_for_status()
                return resp.text[:5000]  # Limit size
        except Exception:  # network failure: return empty  # noqa: BLE001
            return ""


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

    def complete(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.0,
        images: list[str] | None = None,
    ) -> LLMResponse:
        """One chat completion. ``images`` are URLs or data: URIs, appended as
        image_url content parts after the text (OpenAI multimodal shape)."""
        retryer = retry(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=self._backoff_base),
            reraise=True,
        )(self._once)
        content: Any = prompt
        if images:
            content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": u}} for u in images
            ]
        kwargs: dict[str, Any] = {"model": self._model, "messages": [{"role": "user", "content": content}]}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if temperature > 0:
            kwargs["temperature"] = temperature
        return retryer(prompt, kwargs)

    def web_search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        """Default implementation - no web search. Override with WebSearchProvider."""
        return []

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

    def complete(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.0,
        images: list[str] | None = None,
    ) -> LLMResponse:
        text = json.dumps({"verdict": "match", "reason": "fixture judge"})
        return LLMResponse(text=text, tokens_in=len(prompt) // 4, tokens_out=len(text) // 4)
    
    def web_search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        return []


class ScriptedClient:
    """Deterministic client for tests/fixture runs: maps prompt → canned response."""

    def __init__(self, responses: dict[str, str] | None = None, default: str = "{}") -> None:
        self._responses = responses or {}
        self._default = default
        self.calls: list[str] = []
        self.image_calls: list[list[str]] = []

    def complete(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.0,
        images: list[str] | None = None,
    ) -> LLMResponse:
        self.calls.append(prompt)
        if images:
            self.image_calls.append(list(images))
        text = self._responses.get(prompt, self._default)
        return LLMResponse(text=text, tokens_in=len(prompt) // 4, tokens_out=len(text) // 4)
    
    def web_search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        return []


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a judge/analyst JSON reply, tolerating code fences (§5.4 repair path)."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        s = s.removesuffix("```")
        s = s.strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found")
        obj = json.loads(s[start : end + 1])
    if not isinstance(obj, dict):
        raise TypeError("expected JSON object")
    return obj


class WebEnabledClient(OpenAICompatClient):
    """OpenAI-compatible client with web search capability for synthetic generation."""
    
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: int = 120,
        max_attempts: int = 3,
        backoff_base: float = 2.0,
        usd_per_1k_tokens: float = 0.0,
        web_search_provider: str = "tavily",
        web_search_api_key: str | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            max_attempts=max_attempts,
            backoff_base=backoff_base,
            usd_per_1k_tokens=usd_per_1k_tokens,
        )
        self._web_search = WebSearchProvider(web_search_provider, web_search_api_key)
    
    def web_search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        return self._web_search.search(query, max_results)
    
    def web_fetch(self, url: str) -> str:
        return self._web_search.fetch(url)


class WebEnabledFixtureClient(FixtureJudge):
    """Fixture client with mock web search for testing."""
    
    def __init__(self, mock_results: dict[str, list[dict[str, str]]] | None = None):
        self._mock_results = mock_results or {}
    
    def web_search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        # Simple keyword matching for mock results
        for key, results in self._mock_results.items():
            if key.lower() in query.lower():
                return results[:max_results]
        return []
