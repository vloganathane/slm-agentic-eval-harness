"""Text-similarity metrics for generation fields (rouge_l, edit_sim, token_f1).

Known-value tests — these numbers are the contract: a change that moves them
changes gate outcomes, so they're pinned rather than re-derived.
"""

from __future__ import annotations

import pytest

from slm_eval.similarity import edit_sim, rouge_l, similarity, token_f1


class TestTokenF1:
    def test_identical_is_one(self):
        assert token_f1("hello world", "hello world") == pytest.approx(1.0)

    def test_disjoint_is_zero(self):
        assert token_f1("abc def", "xyz uvw") == pytest.approx(0.0)

    def test_partial_overlap(self):
        # pred 3 tokens, gold 3 tokens, overlap 2 -> P=R=2/3, F1=2/3
        assert token_f1("a b c", "a b d") == pytest.approx(2 / 3)

    def test_order_insensitive(self):
        assert token_f1("world hello", "hello world") == pytest.approx(1.0)

    def test_both_empty_is_match(self):
        assert token_f1("", "") == pytest.approx(1.0)

    def test_one_empty_is_zero(self):
        assert token_f1("a", "") == pytest.approx(0.0)

    def test_case_insensitive(self):
        assert token_f1("Hello WORLD", "hello world") == pytest.approx(1.0)

    def test_repeated_tokens_count_multiplicity(self):
        # pred "a a b" (3 tokens), gold "a c" (2 tokens), overlap 1:
        # P = 1/3, R = 1/2 -> F1 = 2*(1/3)*(1/2) / (1/3+1/2) = 0.4
        assert token_f1("a a b", "a c") == pytest.approx(0.4)


class TestEditSim:
    def test_identical_is_one(self):
        assert edit_sim("receipt total 42", "receipt total 42") == pytest.approx(1.0)

    def test_both_empty_is_one(self):
        assert edit_sim("", "") == pytest.approx(1.0)

    def test_one_empty_is_zero(self):
        assert edit_sim("abc", "") == pytest.approx(0.0)

    def test_single_substitution(self):
        # "cat" vs "car": 1 sub, len 3 -> 1 - 1/3
        assert edit_sim("cat", "car") == pytest.approx(2 / 3)

    def test_single_insertion(self):
        # "cat" vs "cats": distance 1, max len 4 -> 1 - 1/4
        assert edit_sim("cat", "cats") == pytest.approx(0.75)

    def test_completely_different(self):
        assert edit_sim("abcd", "wxyz") == pytest.approx(0.0)

    def test_symmetric(self):
        assert edit_sim("kitten", "sitting") == pytest.approx(
            edit_sim("sitting", "kitten")
        )

    def test_range_bounds(self):
        for a, b in [("x", "y"), ("long string here", "other text"), ("", "z")]:
            v = edit_sim(a, b)
            assert 0.0 <= v <= 1.0


class TestRougeL:
    def test_identical_is_one(self):
        assert rouge_l("the total is 42", "the total is 42") == pytest.approx(1.0)

    def test_order_matters(self):
        # Same words, reversed: LCS is 1 of 2 -> P=R=0.5, F1=0.5
        assert rouge_l("b a", "a b") == pytest.approx(0.5)

    def test_subsequence_scores_high(self):
        # "a b c d" is a subsequence of "a x b y c z d": LCS = 4.
        # P = 4/4 = 1, R = 4/7 -> F1 = 2*(4/7) / (1 + 4/7) = 8/11
        assert rouge_l("a b c d", "a x b y c z d") == pytest.approx(8 / 11)

    def test_disjoint_is_zero(self):
        assert rouge_l("one two", "three four") == pytest.approx(0.0)

    def test_both_empty_is_one(self):
        assert rouge_l("", "") == pytest.approx(1.0)

    def test_one_empty_is_zero(self):
        assert rouge_l("a b", "") == pytest.approx(0.0)

    def test_partial(self):
        # pred "a b c" vs gold "a x c": LCS = a,c = 2
        # P = 2/3, R = 2/3, F1 = 2/3
        assert rouge_l("a b c", "a x c") == pytest.approx(2 / 3)


class TestDispatch:
    def test_similarity_dispatches_by_name(self):
        assert similarity("token_f1", "a b", "a b") == pytest.approx(1.0)
        assert similarity("edit_sim", "cat", "cat") == pytest.approx(1.0)
        assert similarity("rouge_l", "x y", "x y") == pytest.approx(1.0)

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="unknown similarity metric"):
            similarity("bleu", "a", "a")

    def test_none_inputs_treated_as_empty(self):
        assert similarity("token_f1", None, None) == pytest.approx(1.0)

    def test_non_string_inputs_coerced(self):
        assert similarity("edit_sim", 42, 42) == pytest.approx(1.0)
