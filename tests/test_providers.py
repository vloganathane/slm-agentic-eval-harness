"""Tests for multi-provider support."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_eval.providers import (
    get_provider,
    get_serving_provider,
    get_training_provider,
    list_providers,
    list_serving_providers,
    list_training_providers,
)
from slm_eval.providers.base import ProviderConfig, ServingConfig, TrainingConfig


class TestInferenceProviders:
    """Tests for inference providers."""

    def test_list_providers(self):
        providers = list_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers
        assert "mistral" in providers
        assert "together" in providers
        assert "groq" in providers
        assert "fireworks" in providers
        assert "local" in providers

    def test_get_provider_openai(self):
        config = ProviderConfig(provider="openai", model="gpt-4")
        provider = get_provider("openai", config)
        assert provider is not None

    def test_get_provider_anthropic(self):
        config = ProviderConfig(provider="anthropic", model="claude-3-opus-20240229")
        provider = get_provider("anthropic", config)
        assert provider is not None

    def test_get_provider_google(self):
        config = ProviderConfig(provider="google", model="gemini-pro")
        provider = get_provider("google", config)
        assert provider is not None

    def test_get_provider_mistral(self):
        config = ProviderConfig(provider="mistral", model="mistral-large-latest")
        provider = get_provider("mistral", config)
        assert provider is not None

    def test_get_provider_together(self):
        config = ProviderConfig(provider="together", model="meta-llama/Llama-3-70B-chat-hf")
        provider = get_provider("together", config)
        assert provider is not None

    def test_get_provider_groq(self):
        config = ProviderConfig(provider="groq", model="llama3-70b-8192")
        provider = get_provider("groq", config)
        assert provider is not None

    def test_get_provider_fireworks(self):
        config = ProviderConfig(provider="fireworks", model="accounts/fireworks/models/llama-v3-70b-instruct")
        provider = get_provider("fireworks", config)
        assert provider is not None

    def test_get_provider_local(self):
        config = ProviderConfig(provider="local", model="model.gguf")
        provider = get_provider("local", config)
        assert provider is not None

    def test_get_provider_ollama(self):
        config = ProviderConfig(provider="ollama", model="llama3:8b")
        provider = get_provider("ollama", config)
        assert provider is not None

    def test_get_provider_unknown(self):
        config = ProviderConfig(provider="unknown", model="model")
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("unknown", config)

    def test_provider_alias_vllm(self):
        """vllm backend should map to local provider."""
        config = ProviderConfig(provider="vllm", model="model")
        provider = get_provider("vllm", config)
        assert provider is not None

    def test_provider_alias_llama_server(self):
        """llama_server backend should map to local provider."""
        config = ProviderConfig(provider="llama_server", model="model")
        provider = get_provider("llama_server", config)
        assert provider is not None


class TestTrainingProviders:
    """Tests for training providers."""

    def test_list_training_providers(self):
        providers = list_training_providers()
        assert "trl" in providers
        assert "axolotl" in providers
        assert "llama_factory" in providers
        assert "unsloth" in providers

    def test_get_training_provider_trl(self):
        provider = get_training_provider("trl")
        assert provider is not None

    def test_get_training_provider_axolotl(self):
        provider = get_training_provider("axolotl")
        assert provider is not None

    def test_get_training_provider_llama_factory(self):
        provider = get_training_provider("llama_factory")
        assert provider is not None

    def test_get_training_provider_llama_factory_alias(self):
        provider = get_training_provider("llama-factory")
        assert provider is not None

    def test_get_training_provider_unsloth(self):
        provider = get_training_provider("unsloth")
        assert provider is not None

    def test_get_training_provider_nanollama(self):
        provider = get_training_provider("nanollama", repo_path="/tmp/nonexistent")
        assert provider is not None

    def test_get_training_provider_unknown(self):
        with pytest.raises(ValueError, match="Unknown training framework"):
            get_training_provider("unknown")


class TestServingProviders:
    """Tests for serving providers."""

    def test_list_serving_providers(self):
        providers = list_serving_providers()
        assert "vllm" in providers
        assert "tgi" in providers
        assert "sglang" in providers
        assert "llama_cpp" in providers
        assert "ollama" in providers

    def test_get_serving_provider_vllm(self):
        provider = get_serving_provider("vllm")
        assert provider is not None

    def test_get_serving_provider_tgi(self):
        provider = get_serving_provider("tgi")
        assert provider is not None

    def test_get_serving_provider_sglang(self):
        provider = get_serving_provider("sglang")
        assert provider is not None

    def test_get_serving_provider_llama_cpp(self):
        provider = get_serving_provider("llama_cpp")
        assert provider is not None

    def test_get_serving_provider_llama_cpp_alias(self):
        provider = get_serving_provider("llama-cpp")
        assert provider is not None

    def test_get_serving_provider_llama_cpp_dot(self):
        provider = get_serving_provider("llama.cpp")
        assert provider is not None

    def test_get_serving_provider_ollama(self):
        provider = get_serving_provider("ollama")
        assert provider is not None

    def test_get_serving_provider_nanollama(self):
        provider = get_serving_provider("nanollama", go_binary="/tmp/nonexistent")
        assert provider is not None

    def test_get_serving_provider_nanollama_go(self):
        provider = get_serving_provider("nanollama_go", go_binary="/tmp/nonexistent")
        assert provider is not None

    def test_get_serving_provider_nanollama_llamacpp(self):
        provider = get_serving_provider("nanollama_llamacpp")
        assert provider is not None

    def test_get_serving_provider_nanollama_ollama(self):
        provider = get_serving_provider("nanollama_ollama")
        assert provider is not None

    def test_get_serving_provider_unknown(self):
        with pytest.raises(ValueError, match="Unknown serving framework"):
            get_serving_provider("unknown")


class TestProviderConfigs:
    """Tests for provider configuration dataclasses."""

    def test_provider_config_defaults(self):
        config = ProviderConfig(provider="openai", model="gpt-4")
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.api_key is None
        assert config.api_base is None
        assert config.temperature == 0.0
        assert config.max_tokens == 1024
        assert config.extra == {}

    def test_provider_config_with_values(self):
        config = ProviderConfig(
            provider="anthropic",
            model="claude-3-opus",
            api_key="test-key",
            api_base="https://api.anthropic.com",
            temperature=0.5,
            max_tokens=2048,
            extra={"top_p": 0.9},
        )
        assert config.provider == "anthropic"
        assert config.model == "claude-3-opus"
        assert config.api_key == "test-key"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048
        assert config.extra == {"top_p": 0.9}

    def test_training_config_defaults(self):
        config = TrainingConfig(
            framework="trl",
            base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dataset="data/train.jsonl",
            output_dir="checkpoints/test",
        )
        assert config.framework == "trl"
        assert config.epochs == 3
        assert config.batch_size == 4
        assert config.learning_rate == 2e-4
        assert config.lora_r == 16
        assert config.lora_alpha == 32

    def test_serving_config_defaults(self):
        config = ServingConfig(
            framework="vllm",
            model="meta-llama/Llama-3-8B-Instruct",
        )
        assert config.framework == "vllm"
        assert config.port == 8000
        assert config.host == "0.0.0.0"
        assert config.gpu_memory_utilization == 0.9
        assert config.max_model_len == 8192
        assert config.tensor_parallel_size == 1


class TestNanollama:
    """Tests for nanollama provider."""

    def test_model_tiers(self):
        from slm_eval.providers.nanollama import MODEL_TIERS

        assert "nano" in MODEL_TIERS
        assert "micro" in MODEL_TIERS
        assert "mini" in MODEL_TIERS
        assert "small" in MODEL_TIERS
        assert "goldie" in MODEL_TIERS
        assert "medium" in MODEL_TIERS
        assert "large" in MODEL_TIERS
        assert "big" in MODEL_TIERS

    def test_nano_tier_info(self):
        from slm_eval.providers.nanollama import MODEL_TIERS

        nano = MODEL_TIERS["nano"]
        assert nano["layers"] == 13
        assert nano["dim"] == 576
        assert nano["heads"] == 9
        assert nano["params"] == "89M"

    def test_training_provider_init(self):
        from slm_eval.providers.nanollama import NanollamaTrainingProvider

        provider = NanollamaTrainingProvider(repo_path="/tmp/nonexistent")
        assert provider.repo_path == Path("/tmp/nonexistent")
        assert provider.is_available() is False

    def test_go_serving_provider_init(self):
        from slm_eval.providers.nanollama import NanollamaGoServingProvider

        provider = NanollamaGoServingProvider(go_binary="/tmp/nonexistent")
        assert provider.go_binary == Path("/tmp/nonexistent")
        assert provider.is_available() is False
