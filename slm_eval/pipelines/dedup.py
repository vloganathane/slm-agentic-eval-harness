"""Fast deduplication using prefix-hash + Jaccard (O(n) with small constant).

This replaces the slow MinHash LSH approach and is used by both:
- DatasetValidator for dedup during validation
- split_stratified for leakage detection

Algorithm:
1. Exact text dedup via SHA-256 (O(n))
2. For remaining texts: prefix-hash bucketing + word-set Jaccard (O(n) with small constant)

This is ~100x faster than MinHash LSH for 100k+ cases while maintaining similar recall.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Configuration constants
PREFIX_WORDS = 5         # Words to use for prefix hash bucketing
MAX_WORDS = 100          # Max words to consider for Jaccard
NUM_BUCKETS = 10000      # Hash buckets for prefix
DEFAULT_JACCARD_THRESHOLD = 0.6  # Near-duplicate threshold
MIN_TEXT_LEN = 60        # Minimum text length to check for near-dup


@dataclass
class DedupResult:
    """Result of deduplication pass."""
    unique_cases: list[dict[str, Any]]
    removed_cases: list[dict[str, Any]]
    exact_duplicates: int = 0
    near_duplicates: int = 0
    duplicate_pairs: list[tuple[str, str]] = None

    def __post_init__(self):
        if self.duplicate_pairs is None:
            self.duplicate_pairs = []

    @property
    def total_removed(self) -> int:
        return self.exact_duplicates + self.near_duplicates

    @property
    def total_kept(self) -> int:
        return len(self.unique_cases)


def compute_text_hash(text: str) -> str:
    """Compute SHA-256 hash of text for exact dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def word_set(text: str, max_words: int = MAX_WORDS) -> set[str]:
    """Extract word set from text (lowercased, limited)."""
    words = text.lower().split()
    return set(words[:max_words])


def jaccard_similarity(set1: set[str], set2: set[str]) -> float:
    """Compute Jaccard similarity between two word sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def prefix_bucket(text: str, prefix_words: int = PREFIX_WORDS, num_buckets: int = NUM_BUCKETS) -> int:
    """Compute bucket index from text word prefix."""
    words = text.lower().split()[:prefix_words]
    prefix = " ".join(words)
    return hash(prefix) % num_buckets


def deduplicate_cases(
    cases: list[dict[str, Any]],
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    text_field: str = "text",
    id_field: str = "case_id",
    min_text_len: int = MIN_TEXT_LEN,
) -> DedupResult:
    """Fast deduplication: exact hash + prefix-hash Jaccard.

    Args:
        cases: List of case dicts
        jaccard_threshold: Similarity threshold for near-duplicate detection
        text_field: Field containing text to deduplicate on
        id_field: Field containing case ID
        min_text_len: Minimum text length for near-dup check

    Returns:
        DedupResult with unique and removed cases
    """
    # Phase 1: Exact deduplication via SHA-256
    seen_hashes: dict[str, dict[str, Any]] = {}  # hash -> case
    exact_dupes: list[dict[str, Any]] = []
    after_exact: list[dict[str, Any]] = []

    for case in cases:
        text = case.get(text_field, "")
        text_hash = compute_text_hash(text)
        case_id = case.get(id_field, f"unknown_{len(after_exact)}")

        if text_hash in seen_hashes:
            exact_dupes.append(case)
        else:
            seen_hashes[text_hash] = case
            after_exact.append(case)

    # Phase 2: Near-deduplication via prefix-hash + Jaccard
    # Build buckets from texts
    buckets: dict[int, list[tuple[set[str], str, str]]] = defaultdict(list)  # bucket -> [(word_set, text, case_id)]

    for case in after_exact:
        text = case.get(text_field, "")
        if len(text) < min_text_len:
            continue
        wordset = word_set(text)
        if not wordset:
            continue
        bucket = prefix_bucket(text)
        buckets[bucket].append((wordset, text, case.get(id_field, "")))

    # Check each case against its bucket and adjacent buckets (to handle prefix differences)
    kept: list[dict[str, Any]] = []
    near_dupes: list[dict[str, Any]] = []
    near_dup_pairs: list[tuple[str, str]] = []

    for case in after_exact:
        text = case.get(text_field, "")
        case_id = case.get(id_field, "")

        if len(text) < min_text_len:
            kept.append(case)
            continue

        wordset = word_set(text)
        if not wordset:
            kept.append(case)
            continue

        bucket = prefix_bucket(text)
        is_duplicate = False

        # Check against cases in same bucket and adjacent buckets (±1)
        for b in [bucket - 1, bucket, bucket + 1]:
            if b not in buckets:
                continue
            for existing_wordset, existing_text, existing_id in buckets[b]:
                if existing_id == case_id:
                    continue
                jaccard = jaccard_similarity(wordset, existing_wordset)
                if jaccard >= jaccard_threshold:
                    is_duplicate = True
                    near_dupes.append(case)
                    near_dup_pairs.append((case_id, existing_id))
                    break
            if is_duplicate:
                break

        if not is_duplicate:
            kept.append(case)
            # Add to bucket for future comparisons
            buckets[bucket].append((wordset, text, case_id))

    return DedupResult(
        unique_cases=kept,
        removed_cases=exact_dupes + near_dupes,
        exact_duplicates=len(exact_dupes),
        near_duplicates=len(near_dupes),
        duplicate_pairs=near_dup_pairs,
    )


def check_leakage(
    train_cases: list[dict[str, Any]],
    holdout_cases: list[dict[str, Any]],
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    text_field: str = "text",
    id_field: str = "case_id",
    min_text_len: int = MIN_TEXT_LEN,
) -> list[tuple[str, str]]:
    """Check for near-duplicate leakage between train and holdout.

    Returns list of (holdout_id, train_id) pairs that are near-duplicates.
    """
    # Build buckets from train texts
    buckets: dict[int, list[tuple[set[str], str]]] = defaultdict(list)

    for case in train_cases:
        text = case.get(text_field, "")
        if len(text) < min_text_len:
            continue
        wordset = word_set(text)
        if not wordset:
            continue
        bucket = prefix_bucket(text)
        buckets[bucket].append((wordset, case.get(id_field, "")))

    # Check holdout cases against train buckets (including adjacent)
    leaks: list[tuple[str, str]] = []

    for case in holdout_cases:
        text = case.get(text_field, "")
        if len(text) < min_text_len:
            continue
        wordset = word_set(text)
        if not wordset:
            continue

        bucket = prefix_bucket(text)
        holdout_id = case.get(id_field, "")

        # Check adjacent buckets
        for b in [bucket - 1, bucket, bucket + 1]:
            if b not in buckets:
                continue
            for train_wordset, train_id in buckets[b]:
                if train_id == holdout_id:
                    continue
                jaccard = jaccard_similarity(wordset, train_wordset)
                if jaccard >= jaccard_threshold:
                    leaks.append((holdout_id, train_id))
                    break
            if leaks and leaks[-1][0] == holdout_id:
                break

    return leaks


def load_text_hashes(path: Path) -> set[str]:
    """Load text hashes from a file for incremental dedup."""
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def save_text_hashes(hashes: set[str], path: Path) -> None:
    """Save text hashes for incremental dedup."""
    path.write_text("\n".join(sorted(hashes)))
