"""Deterministic text-similarity metrics for generation fields (no LLM judge).

Three metrics, all pure-Python and dependency-free (same rationale as D-04's
in-house Cohen's κ): the numbers must be reproducible on any machine without
pinning an NLP stack.

- ``rouge_l``  : ROUGE-L F1 — longest-common-subsequence based, order-aware.
- ``edit_sim`` : 1 - levenshtein(a, b) / max(len(a), len(b)) — character level.
- ``token_f1`` : bag-of-tokens F1 — order-ignorant, robust to reordering.

All return a float in [0, 1]. They score *how close* a generated string is to
gold; the SimilarityValidator turns that into a pass/fail element for the gate
(score >= threshold → TP), while the raw mean is reported alongside.
"""

from __future__ import annotations

import re
from typing import Literal

SimilarityMetric = Literal["rouge_l", "edit_sim", "token_f1"]

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(s: str) -> list[str]:
    return _TOKEN_RE.findall(s.casefold())


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def token_f1(pred: str, gold: str) -> float:
    """Bag-of-tokens F1. Order-insensitive: rewording ranks better than reordering."""
    pt, gt = _tokens(pred), _tokens(gold)
    if not pt or not gt:
        return 1.0 if pt == gt else 0.0
    from collections import Counter

    pc, gc = Counter(pt), Counter(gt)
    overlap = sum((pc & gc).values())
    precision = overlap / len(pt)
    recall = overlap / len(gt)
    return _f1(precision, recall)


def edit_sim(pred: str, gold: str) -> float:
    """1 - normalized Levenshtein distance over characters (0..1, higher = closer)."""
    a, b = pred, gold
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    # Single-row DP: O(len(a) * len(b)) time, O(len(b)) space.
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(
                min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb))
            )
        prev = curr
    return 1.0 - prev[-1] / max(len(a), len(b))


def rouge_l(pred: str, gold: str) -> float:
    """ROUGE-L F1: LCS length over token sequences (order-aware)."""
    pt, gt = _tokens(pred), _tokens(gold)
    if not pt or not gt:
        return 1.0 if pt == gt else 0.0
    # Standard LCS DP over tokens.
    m, n = len(pt), len(gt)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev_row = dp[:]
        for j in range(1, n + 1):
            dp[j] = prev_row[j - 1] + 1 if pt[i - 1] == gt[j - 1] else max(prev_row[j], dp[j - 1])
    lcs = dp[n]
    return _f1(lcs / m, lcs / n)


METRICS: dict[str, object] = {
    "rouge_l": rouge_l,
    "edit_sim": edit_sim,
    "token_f1": token_f1,
}


def similarity(metric: str, pred: str, gold: str) -> float:
    """Dispatch by metric name; unknown names raise rather than silently default."""
    fn = METRICS.get(metric)
    if fn is None:
        raise ValueError(f"unknown similarity metric: {metric!r} (choose from {sorted(METRICS)})")
    return float(fn(str(pred or ""), str(gold or "")))  # type: ignore[operator]
