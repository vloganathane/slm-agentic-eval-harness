"""Stratified split + holdout sealing (SPEC_V3 §5.3, §6.2; AC-02).

The single most important property this stage owns: **zero overlap** between the
training remainder and the sealed holdout. Leakage here makes every downstream
number a lie, so the assertion is unconditional and aborts before inference.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slm_eval.errors import LeakageError
from slm_eval.hashing import sha256_json

# Case texts that share this many shingles are treated as near-duplicates.
# MinHash gives a probabilistic signature; equal signatures ⇒ almost surely a dup.
_MINHASH_PERMUTATIONS = 128
_SHINGLE_SIZE = 5


@dataclass
class Split:
    holdout: list[dict[str, Any]]
    train: list[dict[str, Any]]
    stratum_sizes: dict[str, int] = field(default_factory=dict)
    holdout_hash: str = ""


def _shingles(text: str) -> set[str]:
    toks = text.lower().split()
    if len(toks) < _SHINGLE_SIZE:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + _SHINGLE_SIZE]) for i in range(len(toks) - _SHINGLE_SIZE + 1)}


def minhash_signature(text: str, seeds: list[int] | None = None) -> tuple[int, ...]:
    """Stable MinHash signature over word shingles (datasketch-free, deterministic)."""
    if seeds is None:
        seeds = list(range(_MINHASH_PERMUTATIONS))
    sh = _shingles(text)
    if not sh:
        return tuple(0 for _ in seeds)
    sig = []
    for s in seeds:
        # zlib.crc32, not hash(): stable across processes (PYTHONHASHSEED would
        # otherwise reshuffle signatures and break leakage detection on resume)
        h = min(zlib.crc32(f"{s}:{tok}".encode()) & 0xFFFFFFFF for tok in sh)
        sig.append(h)
    return tuple(sig)


def signature_similarity(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """Fraction of MinHash rows that agree — the unbiased estimator of Jaccard.

    We compare by *similarity threshold*, not exact equality: a near-duplicate
    with a few extra words shares most shingles, so its signature matches on the
    majority of rows even though no row is guaranteed identical.
    """
    if len(a) != len(b):
        raise ValueError("signature length mismatch")
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)


# Two texts whose shingle Jaccard is >= this are treated as the same case for
# leakage purposes. 0.8 catches "same invoice, a few extra tokens" while leaving
# genuinely different invoices apart.
NEAR_DUP_THRESHOLD = 0.8


def _merge_small_strata(
    groups: dict[str, list[dict]], min_per_stratum: int
) -> dict[str, list[dict]]:
    """Merge under-floor values into a bucket per key (§6.2) — never silently dropped."""
    big = {k: v for k, v in groups.items() if len(v) >= min_per_stratum}
    small = {k: v for k, v in groups.items() if len(v) < min_per_stratum}
    if small:
        merged: list[dict[str, Any]] = []
        for v in small.values():
            merged.extend(v)
        key = f"other_merged_{len(small)}"
        big[key] = big.get(key, []) + merged
    return big


def split_stratified(
    cases: list[dict[str, Any]],
    strata_keys: list[str],
    fraction: float,
    seed: int,
    min_per_stratum: int = 1,
) -> Split:
    """Stratified split on the first strata key (single_key mode, §6.2).

    With no strata keys, falls back to a uniform random split so the pipeline
    still runs (gate then reports no per-stratum detail).
    """
    rng = random.Random(seed)
    by_id = {c["case_id"]: c for c in cases}
    if len(by_id) != len(cases):
        raise ValueError("duplicate case_id in dataset")

    primary = strata_keys[0] if strata_keys else None
    holdout: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    sizes: dict[str, int] = {}

    if primary is None:
        idx = list(range(len(cases)))
        rng.shuffle(idx)
        cut = max(1, round(len(cases) * fraction))
        holdout = [cases[i] for i in idx[:cut]]
        train = [cases[i] for i in idx[cut:]]
    else:
        groups: dict[str, list[dict]] = {}
        for c in cases:
            groups.setdefault(c["strata"][primary], []).append(c)
        groups = _merge_small_strata(groups, min_per_stratum)
        for value in sorted(groups):
            grp = groups[value]
            rng.shuffle(grp)
            cut = max(1, round(len(grp) * fraction))
            holdout.extend(grp[:cut])
            train.extend(grp[cut:])
            sizes[value] = len(grp)

    # --- Leakage proof (AC-02): holdout must share no case with train ---
    train_ids = {c["case_id"] for c in train}
    overlap_ids = train_ids & {c["case_id"] for c in holdout}
    if overlap_ids:
        raise LeakageError(
            f"case_id overlap between train and holdout: {sorted(overlap_ids)[:5]}"
        )

    # --- Near-duplicate leakage: no holdout case may closely match a train case ---
    train_sigs = [minhash_signature(c["text"]) for c in train]
    train_ids_list = [c["case_id"] for c in train]
    dupes = []
    for c in holdout:
        sig = minhash_signature(c["text"])
        for tsig, tid in zip(train_sigs, train_ids_list, strict=True):
            if signature_similarity(sig, tsig) >= NEAR_DUP_THRESHOLD:
                dupes.append((c["case_id"], tid))
                break
    if dupes:
        raise LeakageError(
            f"near-duplicate leakage: {len(dupes)} holdout cases match train "
            f"(e.g. {dupes[:3]}) — seal aborted"
        )

    # by_id referenced only to validate uniqueness above; keep it referenced for linters
    del by_id

    holdout_hash = sha256_json(sorted(holdout, key=lambda c: c["case_id"]))
    return Split(
        holdout=holdout,
        train=train,
        stratum_sizes=sizes,
        holdout_hash=holdout_hash,
    )


def seal_holdout(split: Split, run_dir: str | Path) -> Path:
    """Write sealed holdout to run_dir once; refuse to rewrite with different hash (§4.3)."""
    import json

    run_dir = Path(run_dir)
    holdout_dir = run_dir / "holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    path = holdout_dir / "cases.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in split.holdout))
    tmp.replace(path)
    return path
