"""Inference providers: OpenAI, Anthropic, Google, Mistral, Together, Groq, Fireworks, local."""

from __future__ import annotations

import os
from typing import Any

from .base import InferenceProvider, ProviderConfig


class OpenAIProvider(InferenceProvider):
    """OpenAI GPT models."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = None

    def _init_client(self):
        from openai import OpenAI
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        api_base = self.config.api_base or os.getenv("OPENAI_API_BASE")
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def generate(self, prompt: str, **kwargs) -> str:
        if self.client is None:
            self._init_client()
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        return response.choices[0].message.content

    def is_available(self) -> bool:
        try:
            import openai
            return True
        except ImportError:
            return False


class AnthropicProvider(InferenceProvider):
    """Anthropic Claude models."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = None

    def _init_client(self):
        import anthropic
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        if self.client is None:
            self._init_client()
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def is_available(self) -> bool:
        try:
            import anthropic
            return True
        except ImportError:
            return False


class GoogleProvider(InferenceProvider):
    """Google Gemini models."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = None

    def _init_client(self):
        import google.generativeai as genai
        api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.config.model)

    def generate(self, prompt: str, **kwargs) -> str:
        if self.client is None:
            self._init_client()
        response = self.client.generate_content(
            prompt,
            generation_config={
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_output_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            },
        )
        return response.text

    def is_available(self) -> bool:
        try:
            import google.generativeai
            return True
        except ImportError:
            return False


class MistralProvider(InferenceProvider):
    """Mistral AI models."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = None

    def _init_client(self):
        from mistralai import Mistral
        api_key = self.config.api_key or os.getenv("MISTRAL_API_KEY")
        api_base = self.config.api_base or "https://api.mistral.ai/v1"
        self.client = Mistral(api_key=api_key, server_url=api_base)

    def generate(self, prompt: str, **kwargs) -> str:
        if self.client is None:
            self._init_client()
        response = self.client.chat.complete(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        return response.choices[0].message.content

    def is_available(self) -> bool:
        try:
            import mistralai
            return True
        except ImportError:
            return False


class TogetherProvider(InferenceProvider):
    """Together AI models."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = None

    def _init_client(self):
        from openai import OpenAI
        api_key = self.config.api_key or os.getenv("TOGETHER_API_KEY")
        api_base = self.config.api_base or "https://api.together.xyz/v1"
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def generate(self, prompt: str, **kwargs) -> str:
        if self.client is None:
            self._init_client()
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        return response.choices[0].message.content

    def is_available(self) -> bool:
        try:
            import openai
            return True
        except ImportError:
            return False


class GroqProvider(InferenceProvider):
    """Groq models."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = None

    def _init_client(self):
        from openai import OpenAI
        api_key = self.config.api_key or os.getenv("GROQ_API_KEY")
        api_base = self.config.api_base or "https://api.groq.com/openai/v1"
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def generate(self, prompt: str, **kwargs) -> str:
        if self.client is None:
            self._init_client()
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        return response.choices[0].message.content

    def is_available(self) -> bool:
        try:
            import openai
            return True
        except ImportError:
            return False


class FireworksProvider(InferenceProvider):
    """Fireworks AI models."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = None

    def _init_client(self):
        from openai import OpenAI
        api_key = self.config.api_key or os.getenv("FIREWORKS_API_KEY")
        api_base = self.config.api_base or "https://api.fireworks.ai/inference/v1"
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def generate(self, prompt: str, **kwargs) -> str:
        if self.client is None:
            self._init_client()
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        return response.choices[0].message.content

    def is_available(self) -> bool:
        try:
            import openai
            return True
        except ImportError:
            return False


class LocalProvider(InferenceProvider):
    """Local inference via llama.cpp / llama-server or vLLM."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = None

    def _init_client(self):
        from openai import OpenAI
        api_base = self.config.api_base or "http://localhost:8080/v1"
        # Local servers typically use "local" as API key
        self.client = OpenAI(api_key="local", base_url=api_base)

    def generate(self, prompt: str, **kwargs) -> str:
        if self.client is None:
            self._init_client()
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        return response.choices[0].message.content

    def is_available(self) -> bool:
        # Check if any local server is running
        import urllib.request
        api_base = self.config.api_base or "http://localhost:8080/v1"
        try:
            urllib.request.urlopen(f"{api_base}/models", timeout=2)
            return True
        except Exception:
            return False


class OllamaProvider(InferenceProvider):
    """Ollama inference (local, easy setup)."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = None

    def _init_client(self):
        from openai import OpenAI
        # Ollama uses port 11434 by default
        api_base = self.config.api_base or "http://localhost:11434/v1"
        self.client = OpenAI(api_key="ollama", base_url=api_base)

    def generate(self, prompt: str, **kwargs) -> str:
        if self.client is None:
            self._init_client()
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        return response.choices[0].message.content

    def is_available(self) -> bool:
        import shutil
        if not shutil.which("ollama"):
            return False
        # Check if Ollama is running
        import urllib.request
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            return True
        except Exception:
            return False


# Provider registry
PROVIDERS: dict[str, type[InferenceProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "gemini": GoogleProvider,
    "mistral": MistralProvider,
    "together": TogetherProvider,
    "groq": GroqProvider,
    "fireworks": FireworksProvider,
    "ollama": OllamaProvider,
    "local": LocalProvider,
    "llama_server": LocalProvider,
    "vllm": LocalProvider,
}


def get_provider(name: str, config: ProviderConfig) -> InferenceProvider:
    """Get inference provider by name."""
    provider_cls = PROVIDERS.get(name.lower())
    if provider_cls is None:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown provider: {name}. Available: {available}")
    return provider_cls(config)


def list_providers() -> list[str]:
    """List available inference providers."""
    return list(PROVIDERS.keys())
