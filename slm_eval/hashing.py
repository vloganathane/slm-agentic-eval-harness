"""Content addressing: SHA-256 for files, directories, and canonical JSON (SPEC_V3 §4.3)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hash a file's bytes, streaming in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """Hash UTF-8 text (used for prompt hashing and canonical payloads)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_dir(path: str | Path) -> str:
    """Order-independent directory hash: sha256 over sorted ``relpath:filehash`` lines."""
    root = Path(path)
    h = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    for p in files:
        rel = p.relative_to(root).as_posix()
        h.update(f"{rel}:{sha256_file(p)}\n".encode("utf-8"))
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization for hashing: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(obj: Any) -> str:
    """Hash any JSON-serializable object via its canonical form."""
    return sha256_text(canonical_json(obj))
