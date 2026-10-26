"""Scoring tests: validators + element-level TP/FP/FN (SPEC_V3 §6.1, D-10)."""

import pytest

from slm_eval.config import EvalConfig
from slm_eval.scoring import normalize, score_case, score_field

VALIDATORS = {
    "tax_id": {"type": "regex", "pattern": "^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z].*Z[0-9A-Z]$"},
    "total": {"type": "numeric", "tolerance": 0.01},
    "date": {"type": "date", "formats": ["%Y-%m-%d", "%d/%m/%Y"]},
    "invoice_number": {"type": "string", "normalize": ["trim", "collapse_ws", "casefold"]},
    "line_items": {"type": "list", "item_match": "exact"},
}

CFG = EvalConfig.model_validate(
    {
        "task_description": "invoice extraction",
        "fields": {
            "subjective_fields": ["vendor_name"],
            "exact_match_fields": ["invoice_number", "total", "date", "tax_id", "line_items"],
            "validators": VALIDATORS,
        },
    }
)


def test_normalize_collapse_ws_and_casefold():
    assert normalize("  Hello   World  ") == "Hello World"
    assert normalize("  Hi There ", ("trim", "collapse_ws", "casefold")) == "hi there"


def test_numeric_tolerance():
    assert score_field(100.005, 100.0, {"type": "numeric", "tolerance": 0.01}) == (1, 0, 0)
    assert score_field(101.0, 100.0, {"type": "numeric", "tolerance": 0.01}) == (0, 1, 1)


def test_regex_validator_requires_format_and_value():
    from slm_eval.config import RegexValidator

    v = RegexValidator(pattern=r"^\d{3}$")
    assert score_field("123", "123", v) == (1, 0, 0)
    assert score_field("12", "12", v) == (0, 1, 1)   # wrong format
    assert score_field("999", "123", v) == (0, 1, 1)  # right format, wrong value


def test_date_validator_normalizes_formats():
    from slm_eval.config import DateValidator

    v = DateValidator(formats=["%Y-%m-%d", "%d/%m/%Y"])
    assert score_field("05/03/2024", "2024-03-05", v) == (1, 0, 0)
    assert score_field("2024-03-06", "2024-03-05", v) == (0, 1, 1)


def test_string_normalize_trims_and_casefolds():
    from slm_eval.config import StringValidator

    v = StringValidator(normalize=["trim", "collapse_ws", "casefold"])
    assert score_field(" INV  001 ", "inv 001", v) == (1, 0, 0)


def test_list_item_matching_counts_elements():
    # D-10: each list item is an element.
    gold = [{"sku": "A", "qty": 1}, {"sku": "B", "qty": 2}]
    pred = [{"sku": "A", "qty": 1}, {"sku": "C", "qty": 9}]
    from slm_eval.config import ListValidator

    tp, fp, fn = score_field(pred, gold, ListValidator())
    assert (tp, fp, fn) == (1, 1, 1)


def test_list_exact_match_is_all_tp():
    from slm_eval.config import ListValidator

    gold = [{"sku": "A", "qty": 1}]
    assert score_field(list(gold), gold, ListValidator()) == (1, 0, 0)


def test_list_empty_pred_is_all_fn():
    from slm_eval.config import ListValidator

    gold = [{"sku": "A"}, {"sku": "B"}]
    assert score_field([], gold, ListValidator()) == (0, 0, 2)


def test_scalar_wrong_value_is_fp_and_fn():
    assert score_field("INV-2", "INV-1", None) == (0, 1, 1)


def test_score_case_exact_only_when_no_judge_labels():
    gold = {
        "invoice_number": "INV-001", "total": 100.0, "date": "2024-01-05",
        "tax_id": "27ABCDE1234F1Z5", "line_items": [{"sku": "A", "qty": 1}],
        "vendor_name": "Acme",
    }
    pred = dict(gold)
    outcomes = score_case("c1", pred, gold, CFG, judge_labels={"vendor_name": True})
    # 5 exact + 1 subjective = 6 elements, all correct
    assert sum(o.tp for o in outcomes) == 6
    assert sum(o.fp for o in outcomes) == 0
    assert sum(o.fn for o in outcomes) == 0


def test_score_case_missing_judge_label_counts_incorrect():
    gold = {"invoice_number": "INV-001", "vendor_name": "Acme"}
    pred = {"invoice_number": "INV-001"}
    # only invoice_number declared for this minimal check
    cfg = EvalConfig.model_validate(
        {
            "task_description": "t",
            "fields": {"subjective_fields": ["vendor_name"],
                       "exact_match_fields": ["invoice_number"]},
        }
    )
    outcomes = score_case("c1", pred, gold, cfg, judge_labels=None)
    subj = [o for o in outcomes if o.case_id == "c1"][-1]
    assert subj.fp == 1 and subj.fn == 1  # conservative: no verdict → incorrect
