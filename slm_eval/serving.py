"""vLLM serving integration (Option C).

Manages a vLLM OpenAI-compatible server so the evaluation harness can score a
model under test without a manual server dance. vLLM is imported *lazily*: this
module only builds/launches the CLI, so importing it never requires the heavy
dependency to be installed.

OpenAI-compatible contract: vLLM exposes /v1/chat/completions and /v1/completions,
which is exactly what OpenAICompatClient talks to. No harness-side protocol work
is needed — only process lifecycle + readiness.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlparse

from slm_eval.config import EvalConfig


class ServingError(RuntimeError):
    """Raised when a vLLM server cannot be started or reached."""


@dataclass
class ServeSpec:
    """Everything needed to launch a vLLM OpenAI-compatible server."""

    model: str
    port: int = 8000
    host: str = "127.0.0.1"
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    dtype: str = "auto"
    tensor_parallel_size: int = 1
    extra_args: list[str] = field(default_factory=list)
    log_path: Path | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def build_command(self) -> list[str]:
        """Return the argv for `vllm serve`."""
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--dtype",
            self.dtype,
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
        ]
        if self.max_model_len is not None:
            cmd.extend(["--max-model-len", str(self.max_model_len)])
        cmd.extend(self.extra_args)
        return cmd


def build_spec_from_config(cfg: EvalConfig, model: str) -> ServeSpec:
    """Derive a ServeSpec from the eval config's inference section."""
    inf = cfg.inference
    return ServeSpec(
        model=model,
        port=inf.vllm_port,
        gpu_memory_utilization=inf.vllm_gpu_memory_utilization,
        max_model_len=inf.vllm_max_model_len,
        dtype=inf.vllm_dtype,
        tensor_parallel_size=inf.vllm_tensor_parallel_size,
    )


def port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    """True if a TCP connect to host:port succeeds quickly."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def endpoint_reachable(base_url: str, timeout: float = 0.4) -> bool:
    """True if the OpenAI-compatible endpoint host answers a TCP connect."""
    parsed = urlparse(base_url if "//" in base_url else f"//{base_url}")
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        return False
    return port_open(host, port, timeout=timeout)


def vllm_available() -> bool:
    """True if the vllm package is importable in this interpreter."""
    try:
        import importlib.util

        return importlib.util.find_spec("vllm") is not None
    except (ImportError, ValueError):
        return False


class VllmServer:
    """Context manager that owns a vLLM server subprocess.

    Usage::

        with VllmServer(spec) as server:
            run_inference(cfg, holdout, base_url=server.base_url, model=spec.model)

    If a server is already answering on the target port, we adopt it instead of
    spawning a second one (avoids double GPU allocation during iteration).
    """

    def __init__(self, spec: ServeSpec, startup_timeout: float = 600.0, adopt_existing: bool = True):
        self.spec = spec
        self.startup_timeout = startup_timeout
        self.adopt_existing = adopt_existing
        self._proc: subprocess.Popen | None = None
        self._adopted = False
        self._log_file: Any = None

    @property
    def base_url(self) -> str:
        return self.spec.base_url

    @property
    def adopted(self) -> bool:
        return self._adopted

    def is_running(self) -> bool:
        return endpoint_reachable(self.spec.base_url)

    def start(self) -> VllmServer:
        """Launch vLLM (or adopt an already-running one) and wait for readiness."""
        if self.is_running():
            if self.adopt_existing:
                self._adopted = True
                print(
                    f"adopting existing server at {self.spec.base_url}",
                    file=sys.stderr,
                )
                return self
            raise ServingError(
                f"port {self.spec.port} already in use; refusing to double-launch"
            )

        if not vllm_available():
            raise ServingError(
                "vllm is not installed. Install with:\n"
                "  pip install vllm\n"
                "or use backend: llama_server / openai instead."
            )

        cmd = self.spec.build_command()
        print(f"starting vLLM: {' '.join(cmd)}", file=sys.stderr)

        stdout = subprocess.PIPE
        stderr = subprocess.STDOUT
        if self.spec.log_path:
            self.spec.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self.spec.log_path, "w")  # noqa: SIM115
            stdout = self._log_file
            stderr = subprocess.STDOUT

        self._proc = subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=stderr,
            env=os.environ.copy(),
        )

        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                self._drain_log()
                raise ServingError(
                    f"vLLM exited during startup (code {self._proc.returncode}). "
                    f"See {self.spec.log_path or 'stderr'}."
                )
            if self.is_running():
                print(f"vLLM ready at {self.spec.base_url}", file=sys.stderr)
                return self
            time.sleep(1.0)

        self.stop()
        raise ServingError(
            f"vLLM did not become ready within {self.startup_timeout:.0f}s"
        )

    def stop(self) -> None:
        """Terminate the subprocess we own (never an adopted server)."""
        if self._proc is None or self._adopted:
            self._close_log()
            return
        if self._proc.poll() is None:
            self._proc.send_signal(signal.SIGTERM)
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=10)
        self._proc = None
        self._close_log()

    def _drain_log(self) -> None:
        self._close_log()

    def _close_log(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.close()
            finally:
                self._log_file = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def serve_sync(cfg: EvalConfig, model: str) -> int:
    """Blocking serve entrypoint used by the CLI `serve` command.

    Runs the server in the foreground until interrupted (SIGINT/SIGTERM).
    """
    spec = build_spec_from_config(cfg, model)
    if not vllm_available():
        raise ServingError(
            "vllm is not installed. Install with: pip install vllm"
        )
    cmd = spec.build_command()
    print(f"serving {model} at {spec.base_url}", file=sys.stderr)
    print(f"  {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.Popen(cmd, env=os.environ.copy())
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 130
