"""Classification reporting (macro-F1, per-class P/R/F1, confusion matrix).

The gate metric stays SPEC_V3 §6.5 micro-F1; these tests pin the *additive*
report that makes collapsed classifiers on imbalanced labels visible.
"""

from __future__ import annotations

import pytest

from slm_eval.stats import classification_report


class TestClassificationReport:
    def test_perfect_prediction(self):
        r = classification_report(["a", "b", "a"], ["a", "b", "a"])
        assert r.accuracy == pytest.approx(1.0)
        assert r.macro_f1 == pytest.approx(1.0)
        assert r.n == 3

    def test_accuracy_hides_collapse_macro_f1_exposes_it(self):
        # 90% accuracy by predicting the majority class every time.
        gold = ["majority"] * 9 + ["rare"]
        pred = ["majority"] * 10
        r = classification_report(gold, pred)
        assert r.accuracy == pytest.approx(0.9)
        # rare: P/R/F1 = 0; majority: recall 1.0, precision 9/10, F1 ~0.947
        assert r.macro_f1 < 0.55
        rare = next(row for row in r.rows if row.label == "rare")
        assert rare.f1 == 0.0
        assert rare.support == 1

    def test_macro_f1_averages_all_labels_including_predicted_only(self):
        # gold has no "ghost" but pred does -> ghost gets a support=0 row.
        r = classification_report(["a", "a"], ["a", "ghost"])
        labels = {row.label for row in r.rows}
        assert labels == {"a", "ghost"}
        ghost = next(row for row in r.rows if row.label == "ghost")
        assert ghost.support == 0
        assert ghost.precision == 0.0  # predicted once, never correct

    def test_confusion_matrix_counts(self):
        r = classification_report(["a", "b", "b"], ["a", "a", "b"])
        assert r.confusion[("a", "a")] == 1
        assert r.confusion[("b", "a")] == 1
        assert r.confusion[("b", "b")] == 1

    def test_per_class_precision_recall(self):
        # gold b,b,b ; pred b,b,a -> b: tp=2 fp=0 fn=1, a: tp=0 fp=1 fn=0
        r = classification_report(["b", "b", "b"], ["b", "b", "a"])
        b = next(row for row in r.rows if row.label == "b")
        assert b.precision == pytest.approx(1.0)
        assert b.recall == pytest.approx(2 / 3)
        assert b.f1 == pytest.approx(2 * (2 / 3) / (1 + 2 / 3))

    def test_normalizes_whitespace_by_default(self):
        r = classification_report(["a b"], ["  a   b "])
        assert r.accuracy == pytest.approx(1.0)

    def test_opt_out_of_normalization(self):
        r = classification_report(["a b"], ["  a   b "], normalize=False)
        assert r.accuracy == pytest.approx(0.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            classification_report(["a"], ["a", "b"])

    def test_empty_is_zeroed_not_crashing(self):
        r = classification_report([], [])
        assert r.accuracy == 0.0
        assert r.macro_f1 == 0.0
        assert r.n == 0
        assert r.rows == ()

    def test_single_class_perfect(self):
        r = classification_report(["only"] * 5, ["only"] * 5)
        assert r.macro_f1 == pytest.approx(1.0)

    def test_to_dict_shape_and_rounding(self):
        d = classification_report(["a", "b"], ["a", "a"]).to_dict()
        assert set(d) == {"accuracy", "macro_f1", "n", "per_class", "confusion"}
        assert d["n"] == 2
        assert d["accuracy"] == 0.5
        row = next(r for r in d["per_class"] if r["label"] == "a")
        assert set(row) == {"label", "precision", "recall", "f1", "support"}
        entry = next(c for c in d["confusion"] if c["gold"] == "b" and c["pred"] == "a")
        assert entry["count"] == 1
