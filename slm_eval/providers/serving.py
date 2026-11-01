"""Serving frameworks: vLLM, TGI, SGLang, llama.cpp, Ollama."""

from __future__ import annotations

import subprocess
import time
import urllib.request
from typing import Any

from .base import ServingConfig, ServingProvider


class OllamaServingProvider(ServingProvider):
    """Ollama serving framework (local, easy setup)."""

    def serve(self, config: ServingConfig) -> str:
        """Pull and serve model via Ollama.

        Ollama runs as a daemon, so we just pull the model and ensure it's ready.
        """
        # Pull model
        print(f"Pulling Ollama model: {config.model}")
        try:
            subprocess.run(["ollama", "pull", config.model], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: ollama pull failed: {e.stderr.decode() if e.stderr else e}")

        # Ensure Ollama daemon is running
        self._ensure_ollama_running()

        # Wait for server
        port = config.port if config.port != 8000 else 11434  # Ollama default port
        self._wait_for_server(port)

        return f"http://localhost:{port}/v1"

    def _ensure_ollama_running(self):
        """Start Ollama daemon if not running."""
        import shutil
        if not shutil.which("ollama"):
            raise RuntimeError("Ollama not installed. Install from: https://ollama.ai")

        # Check if already running
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            return
        except Exception:
            pass

        # Start Ollama daemon
        print("Starting Ollama daemon...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)

    def _wait_for_server(self, port: int, timeout: int = 60):
        start = time.time()
        while time.time() - start < timeout:
            try:
                urllib.request.urlopen(f"http://localhost:{port}/api/tags", timeout=2)
                print("Ollama server ready")
                return
            except Exception:
                time.sleep(1)
        raise TimeoutError(f"Ollama server did not start within {timeout}s")

    def is_available(self) -> bool:
        import shutil
        return shutil.which("ollama") is not None


class VLLMServingProvider(ServingProvider):
    """vLLM serving framework."""

    def serve(self, config: ServingConfig) -> str:
        cmd = [
            "vllm", "serve",
            config.model,
            "--host", config.host,
            "--port", str(config.port),
            "--gpu-memory-utilization", str(config.gpu_memory_utilization),
            "--max-model-len", str(config.max_model_len),
            "--tensor-parallel-size", str(config.tensor_parallel_size),
            "--dtype", config.dtype,
        ]

        # Add extra args
        for key, value in config.extra.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        print(f"Starting vLLM: {' '.join(cmd)}")
        subprocess.Popen(cmd)

        # Wait for server
        self._wait_for_server(config.port)
        return f"http://{config.host}:{config.port}/v1"

    def _wait_for_server(self, port: int, timeout: int = 120):
        start = time.time()
        while time.time() - start < timeout:
            try:
                urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=2)
                print("vLLM server ready")
                return
            except Exception:
                time.sleep(1)
        raise TimeoutError(f"vLLM server did not start within {timeout}s")

    def is_available(self) -> bool:
        try:
            import vllm
            return True
        except ImportError:
            return False


class TGIServingProvider(ServingProvider):
    """HuggingFace Text Generation Inference (TGI)."""

    def serve(self, config: ServingConfig) -> str:
        cmd = [
            "text-generation-launcher",
            "--model-id", config.model,
            "--host", config.host,
            "--port", str(config.port),
            "--max-input-tokens", str(config.max_model_len),
        ]

        print(f"Starting TGI: {' '.join(cmd)}")
        subprocess.Popen(cmd)

        # Wait for server
        self._wait_for_server(config.port)
        return f"http://{config.host}:{config.port}/v1"

    def _wait_for_server(self, port: int, timeout: int = 180):
        start = time.time()
        while time.time() - start < timeout:
            try:
                urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
                print("TGI server ready")
                return
            except Exception:
                time.sleep(1)
        raise TimeoutError(f"TGI server did not start within {timeout}s")

    def is_available(self) -> bool:
        try:
            import text_generation
            return True
        except ImportError:
            return False


class SGLangServingProvider(ServingProvider):
    """SGLang serving framework."""

    def serve(self, config: ServingConfig) -> str:
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", config.model,
            "--host", config.host,
            "--port", str(config.port),
            "--mem-fraction-static", str(config.gpu_memory_utilization),
            "--context-length", str(config.max_model_len),
            "--tp", str(config.tensor_parallel_size),
        ]

        print(f"Starting SGLang: {' '.join(cmd)}")
        subprocess.Popen(cmd)

        # Wait for server
        self._wait_for_server(config.port)
        return f"http://{config.host}:{config.port}/v1"

    def _wait_for_server(self, port: int, timeout: int = 120):
        start = time.time()
        while time.time() - start < timeout:
            try:
                urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
                print("SGLang server ready")
                return
            except Exception:
                time.sleep(1)
        raise TimeoutError(f"SGLang server did not start within {timeout}s")

    def is_available(self) -> bool:
        try:
            import sglang
            return True
        except ImportError:
            return False


class LlamaCppServingProvider(ServingProvider):
    """llama.cpp / llama-server serving."""

    def serve(self, config: ServingConfig) -> str:
        cmd = [
            "llama-server",
            "-m", config.model,
            "--host", config.host,
            "--port", str(config.port),
            "-c", str(config.max_model_len),
        ]

        print(f"Starting llama-server: {' '.join(cmd)}")
        subprocess.Popen(cmd)

        # Wait for server
        self._wait_for_server(config.port)
        return f"http://{config.host}:{config.port}/v1"

    def _wait_for_server(self, port: int, timeout: int = 60):
        start = time.time()
        while time.time() - start < timeout:
            try:
                urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
                print("llama-server ready")
                return
            except Exception:
                time.sleep(1)
        raise TimeoutError(f"llama-server did not start within {timeout}s")

    def is_available(self) -> bool:
        import shutil
        return shutil.which("llama-server") is not None


# Provider registry
SERVING_PROVIDERS: dict[str, type[ServingProvider]] = {
    "vllm": VLLMServingProvider,
    "tgi": TGIServingProvider,
    "sglang": SGLangServingProvider,
    "llama_cpp": LlamaCppServingProvider,
    "llama-cpp": LlamaCppServingProvider,
    "llama.cpp": LlamaCppServingProvider,
    "ollama": OllamaServingProvider,
}


def get_serving_provider(name: str) -> ServingProvider:
    """Get serving provider by name."""
    provider_cls = SERVING_PROVIDERS.get(name.lower())
    if provider_cls is None:
        available = ", ".join(SERVING_PROVIDERS.keys())
        raise ValueError(f"Unknown serving framework: {name}. Available: {available}")
    return provider_cls()


def list_serving_providers() -> list[str]:
    """List available serving frameworks."""
    return list(SERVING_PROVIDERS.keys())
