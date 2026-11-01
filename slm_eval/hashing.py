"""Content addressing: SHA-256 for files, directories, and canonical JSON (SPEC_V3 §4.3).
MinHash for near-duplicate detection (SPEC_V3 §5.2 leakage check, M3 synthetic dedup).
"""

from __future__ import annotations

import hashlib
import json
import random
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
        h.update(f"{rel}:{sha256_file(p)}\n".encode())
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization for hashing: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(obj: Any) -> str:
    """Hash any JSON-serializable object via its canonical form."""
    return sha256_text(canonical_json(obj))


# -- MinHash (near-duplicate detection) ---------------------------------------

# Fixed permutation seeds for deterministic MinHash
_MINHASH_PERMUTATIONS = [random.Random(i).randint(1, 2**31 - 1) for i in range(128)]
_MINHASH_PRIME = 2**61 - 1  # Mersenne prime for 64-bit hashing


def _shingles(text: str, k: int = 5) -> set[int]:
    """Character k-shingles hashed to 64-bit ints."""
    text = text.lower()
    if len(text) < k:
        return {hash(text)}
    return {
        hash(text[i:i+k]) & ((1 << 61) - 1)
        for i in range(len(text) - k + 1)
    }


def minhash_signature(text: str, num_perm: int = 128, k: int = 5) -> list[int]:
    """MinHash signature for a text. Deterministic, fixed permutations."""
    shingles = _shingles(text, k)
    if not shingles:
        return [0] * num_perm

    sig = [2**63 - 1] * num_perm
    for s in shingles:
        for i, perm in enumerate(_MINHASH_PERMUTATIONS[:num_perm]):
            # Universal hashing: (a * x + b) mod p
            h = (perm * s + i) % _MINHASH_PRIME
            sig[i] = min(sig[i], h)
    return sig


def minhash_similarity(sig1: list[int] | str, sig2: list[int] | str) -> float:
    """Jaccard similarity estimate from MinHash signatures.
    Accepts either list of ints or hex string (first 16 chars = 16 perms)."""
    if isinstance(sig1, str):
        sig1 = [int(sig1[i:i+4], 16) for i in range(0, min(len(sig1), 64), 4)]
    if isinstance(sig2, str):
        sig2 = [int(sig2[i:i+4], 16) for i in range(0, min(len(sig2), 64), 4)]

    n = min(len(sig1), len(sig2))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if sig1[i] == sig2[i])
    return matches / n
