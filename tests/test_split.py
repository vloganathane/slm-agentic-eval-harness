"""Split/seal tests: leakage proof + sealing (SPEC_V3 §5.3; AC-02, AC-12)."""

import pytest

from slm_eval import fixture
from slm_eval.errors import LeakageError
from slm_eval.split import (
    NEAR_DUP_THRESHOLD,
    minhash_signature,
    seal_holdout,
    split_stratified,
)


def _cases(n=200, seed=1):
    return fixture.generate(count=n, seed=seed)


def test_split_no_case_id_overlap():
    cases = _cases()
    s = split_stratified(cases, ["layout_type"], 0.2, seed=42)
    train_ids = {c["case_id"] for c in s.train}
    hold_ids = {c["case_id"] for c in s.holdout}
    assert train_ids.isdisjoint(hold_ids)


def test_split_covers_all_cases():
    cases = _cases()
    s = split_stratified(cases, ["layout_type"], 0.2, seed=42)
    assert len(s.train) + len(s.holdout) == len(cases)


def test_split_holdout_fraction_reasonable():
    cases = _cases(200)
    s = split_stratified(cases, ["layout_type"], 0.2, seed=42)
    frac = len(s.holdout) / len(cases)
    assert 0.10 <= frac <= 0.35  # stratified per-group rounding, not exact


def test_split_deterministic_given_seed():
    cases = _cases()
    a = split_stratified(cases, ["layout_type"], 0.2, seed=7)
    b = split_stratified(cases, ["layout_type"], 0.2, seed=7)
    assert a.holdout_hash == b.holdout_hash
    assert {c["case_id"] for c in a.holdout} == {c["case_id"] for c in b.holdout}


def test_split_seed_changes_partition():
    cases = _cases()
    a = split_stratified(cases, ["layout_type"], 0.2, seed=7)
    b = split_stratified(cases, ["layout_type"], 0.2, seed=999)
    assert a.holdout_hash != b.holdout_hash


def test_stratification_preserves_stratum_distribution():
    cases = _cases(300)
    s = split_stratified(cases, ["layout_type"], 0.2, seed=42,
                         min_per_stratum=1)
    # each holdout stratum share should roughly match its corpus share
    corpus = {}
    for c in cases:
        corpus[c["strata"]["layout_type"]] = corpus.get(c["strata"]["layout_type"], 0) + 1
    hold = {}
    for c in s.holdout:
        hold[c["strata"]["layout_type"]] = hold.get(c["strata"]["layout_type"], 0) + 1
    for val, n_hold in hold.items():
        expected = 0.2 * corpus[val]
        assert abs(n_hold - expected) <= max(3, 0.1 * expected), f"{val} off"


def test_minhash_signature_detects_duplicates():
    text = "the quick brown fox jumps over the lazy dog repeatedly for shingles"
    assert minhash_signature(text) == minhash_signature(text)
    other = ("invoice vendor address total tax number line items quantity rate "
             "amount supplier buyer ship to billing due date payment terms "
             "gst taxid serial reference order purchase order number "
             "delivery consignment warehouse receipt")
    assert minhash_signature(text) != minhash_signature(other)
    # Genuinely different texts score low similarity.
    from slm_eval.split import signature_similarity

    assert signature_similarity(
        minhash_signature(text), minhash_signature(other)
    ) < NEAR_DUP_THRESHOLD


def test_injected_exact_duplicate_leakage_aborts():
    cases = _cases(50)
    # Inject an exact duplicate of a holdout-bound case into train position.
    dupe = dict(cases[0])
    dupe["case_id"] = "inv-dupe-000"
    cases.append(dupe)
    with pytest.raises(LeakageError):
        # single_key stratification + tiny min_per_stratum forces both copies
        # into the split; the near-dup check must catch it.
        split_stratified(cases, ["layout_type"], 0.5, seed=0, min_per_stratum=1)


def test_injected_near_duplicate_text_leakage_aborts():
    # Leakage only bites across the train/holdout boundary, so force the pair onto
    # opposite sides: put the original in one stratum and an exact-worded near-copy
    # (plus a couple of stray tokens) in another, then split hard enough that each
    # stratum contributes both train and holdout members.
    cases = _cases(50)
    base = dict(cases[3])
    base["strata"] = dict(base["strata"])
    base["strata"]["layout_type"] = "clean"
    near = dict(base)
    near["case_id"] = "inv-near-000"
    near["strata"] = dict(base["strata"])
    near["strata"]["layout_type"] = "handwritten"  # different stratum
    # Same wording + a few stray tokens: shingle Jaccard stays above threshold.
    near["text"] = base["text"] + " extra trailing word"
    cases = [base, near] + [c for c in cases if c["case_id"] != cases[3]["case_id"]]

    # fraction=0.5 guarantees each of the two strata yields both train + holdout.
    with pytest.raises(LeakageError):
        split_stratified(cases, ["layout_type"], 0.5, seed=0, min_per_stratum=1)


def test_seal_holdout_writes_cases(tmp_path):
    cases = _cases(30)
    s = split_stratified(cases, ["layout_type"], 0.3, seed=42, min_per_stratum=1)
    path = seal_holdout(s, tmp_path / "run1")
    assert path.exists()
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == len(s.holdout)


def test_small_strata_merge_not_dropped():
    # Craft one rare stratum value that falls below the floor.
    cases = _cases(200)
    for c in cases[:5]:
        c["strata"]["layout_type"] = "weird_rare"
    s = split_stratified(cases, ["layout_type"], 0.2, seed=1, min_per_stratum=50)
    # the rare value must not vanish — it merges into a bucket, still represented
    all_hold_vals = {c["strata"]["layout_type"] for c in s.holdout}
    assert "weird_rare" in all_hold_vals or any(
        k.startswith("other") for k in s.stratum_sizes
    )
