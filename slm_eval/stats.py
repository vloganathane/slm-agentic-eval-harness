"""Gate statistics: micro-F1, bootstrap CI, Cohen's κ (SPEC_V3 §6.5; D-03, D-04).

D-03: bootstrap over cases, not an analytic interval — element-level micro-F1 over
mixed scalar/list fields isn't binomial, so resampling is assumption-free.

D-04: Cohen's κ in-house. Trivial math; unit-tested against textbook examples
rather than pulled in via sklearn.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FieldOutcome:
    """Element-level contribution of one (case, field) pair (§6.1, D-10)."""

    case_id: str
    tp: int
    fp: int
    fn: int

    @property
    def correct(self) -> bool:
        return self.tp > 0 and self.fp == 0 and self.fn == 0


@dataclass(frozen=True)
class Metric:
    point: float
    ci_lower: float
    ci_upper: float
    n: int
    resamples: int


def micro_f1(outcomes: Iterable[FieldOutcome]) -> float:
    """ΣTP / (ΣTP + ΣFP + ΣFN). Undefined denominators score 0.0."""
    tp = fp = fn = 0
    for o in outcomes:
        tp += o.tp
        fp += o.fp
        fn += o.fn
    denom = tp + fp + fn
    return (tp / denom) if denom else 0.0


def bootstrap_micro_f1(
    outcomes: Sequence[FieldOutcome],
    resamples: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Metric:
    """Percentile bootstrap over *cases* (SPEC_V3 §6.5).

    Cases are the resampling unit — fields within a case are correlated, so
    resampling fields would understate variance.
    """
    n = len({o.case_id for o in outcomes})
    if not outcomes:
        return Metric(point=0.0, ci_lower=0.0, ci_upper=0.0, n=0, resamples=resamples)

    # Group by case once; each case contributes fixed (tp, fp, fn) totals.
    by_case: dict[str, tuple[int, int, int]] = {}
    for o in outcomes:
        t, f, g = by_case.get(o.case_id, (0, 0, 0))
        by_case[o.case_id] = (t + o.tp, f + o.fp, g + o.fn)

    case_ids = list(by_case)
    arr = np.array([by_case[c] for c in case_ids], dtype=np.int64)
    tp_col, fp_col, fn_col = arr[:, 0], arr[:, 1], arr[:, 2]

    point = micro_f1(outcomes)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(case_ids), size=(resamples, len(case_ids)))
    tp_s = tp_col[idx].sum(axis=1)
    fp_s = fp_col[idx].sum(axis=1)
    fn_s = fn_col[idx].sum(axis=1)
    denom = tp_s + fp_s + fn_s
    with np.errstate(divide="ignore", invalid="ignore"):
        stats = np.where(denom > 0, tp_s / np.maximum(denom, 1), 0.0)

    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Metric(
        point=float(point),
        ci_lower=float(lo),
        ci_upper=float(hi),
        n=n,
        resamples=resamples,
    )


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's κ between two labelers (D-04). 1.0 = perfect, 0.0 = chance.

    Returns float('nan') when both raters are constant and agree (κ undefined:
    expected agreement is 1.0). Callers treat NaN as "insufficient signal".
    """
    if len(a) != len(b):
        raise ValueError(f"label length mismatch: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        return float("nan")

    labels = sorted(set(a) | set(b))
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    if k < 2:
        return float("nan")  # both raters gave one identical label everywhere

    confusion = np.zeros((k, k), dtype=np.int64)
    for x, y in zip(a, b, strict=True):
        confusion[idx[x], idx[y]] += 1

    p_o = np.trace(confusion) / n
    row_marg = confusion.sum(axis=1) / n
    col_marg = confusion.sum(axis=0) / n
    p_e = float(row_marg @ col_marg)

    if p_e == 1.0:
        return float("nan")
    return float((p_o - p_e) / (1.0 - p_e))


def kappa_ci(
    a: Sequence[str], b: Sequence[str], resamples: int = 2_000, seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI on κ (§6.4 reports the CI alongside the point)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    if n == 0:
        return (float("nan"), float("nan"))
    vals = []
    a_arr, b_arr = np.array(a), np.array(b)
    for _ in range(resamples):
        idx = rng.integers(0, n, size=n)
        k = cohen_kappa(a_arr[idx].tolist(), b_arr[idx].tolist())
        if not np.isnan(k):
            vals.append(k)
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))
