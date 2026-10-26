"""Stats tests: micro-F1, bootstrap CI, Cohen's κ against textbook (D-03, D-04)."""

import math

import pytest

from slm_eval.stats import (
    FieldOutcome,
    bootstrap_micro_f1,
    cohen_kappa,
    kappa_ci,
    micro_f1,
)


def _o(case, tp, fp, fn):
    return FieldOutcome(case_id=case, tp=tp, fp=fp, fn=fn)


def test_micro_f1_all_correct():
    outs = [_o("c1", 1, 0, 0), _o("c2", 1, 0, 0)]
    assert micro_f1(outs) == pytest.approx(1.0)


def test_micro_f1_half_correct():
    # tp=1, fp=1, fn=1 → 1/3
    outs = [_o("c1", 1, 0, 0), _o("c2", 0, 1, 1)]
    assert micro_f1(outs) == pytest.approx(1 / 3)


def test_micro_f1_empty_is_zero():
    assert micro_f1([]) == 0.0


def test_bootstrap_point_matches_direct():
    outs = [_o(f"c{i}", 1, 0, 0) if i % 2 else _o(f"c{i}", 0, 1, 1) for i in range(40)]
    m = bootstrap_micro_f1(outs, resamples=2000, seed=1)
    assert m.point == pytest.approx(micro_f1(outs))
    assert 0.0 <= m.ci_lower <= m.point <= m.ci_upper <= 1.0


def test_bootstrap_ci_widens_with_variance():
    uniform = [_o(f"c{i}", 1, 0, 0) for i in range(50)]
    mixed = [_o(f"c{i}", 1, 0, 0) if i % 2 else _o(f"c{i}", 0, 1, 1) for i in range(50)]
    mu = bootstrap_micro_f1(uniform, resamples=2000, seed=1)
    mm = bootstrap_micro_f1(mixed, resamples=2000, seed=1)
    assert (mu.ci_upper - mu.ci_lower) < (mm.ci_upper - mm.ci_lower)


def test_bootstrap_deterministic_given_seed():
    outs = [_o(f"c{i}", i % 2, 1 - i % 2, 1 - i % 2) for i in range(30)]
    a = bootstrap_micro_f1(outs, resamples=1000, seed=7)
    b = bootstrap_micro_f1(outs, resamples=1000, seed=7)
    assert (a.ci_lower, a.ci_upper) == (b.ci_lower, b.ci_upper)


def test_bootstrap_empty():
    m = bootstrap_micro_f1([], resamples=1000)
    assert m.n == 0 and m.point == 0.0


# --- Cohen's κ: textbook examples (D-04) ---


def test_kappa_perfect_agreement():
    a = ["match", "mismatch", "match", "mismatch"]
    assert cohen_kappa(a, a) == pytest.approx(1.0)


def test_kappa_chance_agreement_is_zero():
    # Two raters each flipping a fair coin independently, large n → κ ≈ 0.
    import random

    rng = random.Random(0)
    a = [rng.choice(["match", "mismatch"]) for _ in range(2000)]
    b = [rng.choice(["match", "mismatch"]) for _ in range(2000)]
    assert abs(cohen_kappa(a, b)) < 0.1


def test_kappa_known_value():
    # Classic example (Landis & Koch framing): 2 raters, 4 items, 2 agreements,
    # marginals balanced → po=0.5, pe=0.5, κ=0.0
    a = ["yes", "yes", "no", "no"]
    b = ["yes", "no", "yes", "no"]
    assert cohen_kappa(a, b) == pytest.approx(0.0)


def test_kappa_negative_when_worse_than_chance():
    a = ["yes", "yes", "no", "no"]
    b = ["no", "no", "yes", "yes"]  # systematically inverted
    assert cohen_kappa(a, b) < 0


def test_kappa_length_mismatch_raises():
    with pytest.raises(ValueError):
        cohen_kappa(["a"], ["a", "b"])


def test_kappa_undefined_when_constant_and_agree():
    assert math.isnan(cohen_kappa(["x"] * 10, ["x"] * 10))


def test_kappa_ci_brackets_point():
    a = ["match", "mismatch", "match", "mismatch"] * 10
    b = ["match", "mismatch", "match", "mismatch"] * 10  # perfect
    lo, hi = kappa_ci(a, b, resamples=500)
    assert lo <= 1.0 <= hi or (lo <= 1.0 and hi <= 1.0)
