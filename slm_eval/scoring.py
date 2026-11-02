"""Deterministic scoring: validators + element-level TP/FP/FN (SPEC_V3 §6.1, D-10).

Exact-match fields are scored here. Subjective fields arrive pre-labeled from the
judge (score_judge stage) and are folded into outcomes as single elements.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from slm_eval.config import (
    DateValidator,
    EvalConfig,
    ListValidator,
    NumericValidator,
    RegexValidator,
    SimilarityValidator,
    StringValidator,
)
from slm_eval.similarity import similarity as _similarity
from slm_eval.stats import FieldOutcome

DEFAULT_NORMALIZE = ("trim", "collapse_ws")


def normalize(value: Any, ops: tuple[str, ...] = DEFAULT_NORMALIZE) -> str:
    s = "" if value is None else str(value)
    if "trim" in ops:
        s = s.strip()
    if "collapse_ws" in ops:
        s = re.sub(r"\s+", " ", s)
    if "casefold" in ops:
        s = s.casefold()
    return s


def _parse_date(value: Any, formats: list[str]) -> str | None:
    s = normalize(value)
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


_CURRENCY_NOISE = re.compile(r"[,\s₹$€£]")


def _to_float(value: Any) -> float | None:
    """Parse a numeric value, tolerating currency formatting.

    Real invoice amounts arrive as ``"45,500"``, ``"$1,234.50"``, ``"₹ 999"``.
    ``float()`` chokes on the thousands separators and currency symbols, so strip
    them first. Returns None if the result isn't a parseable number.
    """
    s = normalize(value)
    if s == "":
        return None
    s = _CURRENCY_NOISE.sub("", s)
    # Parenthesised negatives: "(1,500)" -> -1500
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _scalar_equal(pred: Any, gold: Any, validator: Any) -> bool:
    if isinstance(validator, SimilarityValidator):
        return _similarity(validator.metric, str(pred or ""), str(gold or "")) >= validator.threshold
    if isinstance(validator, RegexValidator):
        return re.fullmatch(validator.pattern, normalize(pred)) is not None and normalize(
            pred
        ) == normalize(gold)
    if isinstance(validator, NumericValidator):
        pv, gv = _to_float(pred), _to_float(gold)
        if pv is None or gv is None:
            return False
        return abs(pv - gv) <= validator.tolerance
    if isinstance(validator, DateValidator):
        pg, gg = _parse_date(pred, validator.formats), _parse_date(gold, validator.formats)
        return pg is not None and pg == gg
    if isinstance(validator, StringValidator):
        return normalize(pred, tuple(validator.normalize)) == normalize(
            gold, tuple(validator.normalize)
        )
    return normalize(pred) == normalize(gold)


def _list_outcome(pred: Any, gold: Any) -> tuple[int, int, int]:
    """Item-level matching for list fields (D-10): each item is an element."""
    pred_items = pred if isinstance(pred, list) else ([pred] if pred is not None else [])
    gold_items = gold if isinstance(gold, list) else ([gold] if gold is not None else [])

    def key(it: Any) -> str:
        if isinstance(it, dict):
            return "|".join(f"{k}={normalize(v)}" for k, v in sorted(it.items()))
        return normalize(it)

    remaining = [key(i) for i in gold_items]
    tp = 0
    for p in map(key, pred_items):
        if p in remaining:
            remaining.remove(p)
            tp += 1
    fp = len(pred_items) - tp
    fn = len(gold_items) - tp
    return tp, fp, fn


def _as_validator(validator: Any) -> Any:
    """Coerce a raw validator dict into its pydantic type.

    A dict reaching scoring means a caller bypassed EvalConfig; coerce it so the
    isinstance dispatch below behaves identically. Unknown types raise rather
    than silently falling through to exact-string comparison.
    """
    if validator is None or hasattr(validator, "type"):
        return validator
    if isinstance(validator, dict):
        from pydantic import TypeAdapter

        from slm_eval.config import FieldValidatorSpec

        try:
            return TypeAdapter(FieldValidatorSpec).validate_python(validator)
        except Exception as e:
            raise ValueError(f"unknown or invalid field validator: {validator!r}") from e
    raise ValueError(f"unsupported validator type: {type(validator).__name__}")


def score_field(pred: Any, gold: Any, validator: Any = None) -> tuple[int, int, int]:
    """Return (tp, fp, fn) for one field. Scalar → 1 element; list → per item."""
    validator = _as_validator(validator)
    if isinstance(validator, ListValidator) or isinstance(gold, list):
        return _list_outcome(pred, gold)
    if _scalar_equal(pred, gold, validator):
        return (1, 0, 0)
    return (0, 1, 1)


def score_case(
    case_id: str,
    prediction: dict[str, Any],
    gold: dict[str, Any],
    cfg: EvalConfig,
    judge_labels: dict[str, bool] | None = None,
) -> list[FieldOutcome]:
    """Score one case across every configured exact-match field (§6.1).

    ``judge_labels`` supplies subjective-field verdicts from the judge stage; a
    missing verdict counts as incorrect (conservative — never inflate the gate).
    """
    outcomes: list[FieldOutcome] = []
    validators = cfg.fields.validators

    for fname in cfg.fields.exact_match_fields:
        tp, fp, fn = score_field(prediction.get(fname), gold.get(fname), validators.get(fname))
        outcomes.append(FieldOutcome(case_id=case_id, tp=tp, fp=fp, fn=fn))

    for fname in cfg.fields.subjective_fields:
        if judge_labels and judge_labels.get(fname):
            outcomes.append(FieldOutcome(case_id=case_id, tp=1, fp=0, fn=0))
        else:
            outcomes.append(FieldOutcome(case_id=case_id, tp=0, fp=1, fn=1))

    return outcomes


def similarity_scores(
    case_id: str,
    prediction: dict[str, Any],
    gold: dict[str, Any],
    cfg: EvalConfig,
) -> list[dict[str, Any]]:
    """Raw similarity values (0..1) for fields declaring a SimilarityValidator.

    The gate sees the thresholded pass/fail (via score_field); these raw scores
    are reported alongside so a 0.49 and a 0.0 are distinguishable — the binary
    element alone can't show how close a miss was.
    """
    rows: list[dict[str, Any]] = []
    for fname, validator in cfg.fields.validators.items():
        if not isinstance(validator, SimilarityValidator):
            continue
        pred_v, gold_v = prediction.get(fname), gold.get(fname)
        if gold_v in (None, ""):
            continue  # nothing to compare against — not a miss, not a hit
        rows.append(
            {
                "case_id": case_id,
                "field": fname,
                "metric": validator.metric,
                "threshold": validator.threshold,
                "score": round(
                    _similarity(validator.metric, str(pred_v or ""), str(gold_v)), 6
                ),
            }
        )
    return rows
