import random

import pytest

from karna.indexes.learned_bloom import LearnedBloomFilter, _TraditionalBloomFilter

N = 5_000


def _keys(prefixes, n, seed):
    rng = random.Random(seed)
    return [f"{rng.choice(prefixes)}:{rng.randint(0, 10**9):010d}" for _ in range(n)]


@pytest.fixture(scope="module")
def stored():
    return _keys(["user"], N, seed=1)


def test_never_returns_false_negative(stored):
    lbf = LearnedBloomFilter()
    lbf.fit(stored, _keys(["order", "sess"], 3_000, seed=2))
    assert all(lbf.query(k) for k in stored)


def test_structured_negatives_beat_classic_filter_on_size_and_fpr(stored):
    lbf = LearnedBloomFilter()
    lbf.fit(stored, _keys(["order", "sess", "cart"], 3_000, seed=3))
    test_neg = _keys(["order", "sess", "cart"], 5_000, seed=4)

    classic = _TraditionalBloomFilter(n=N, fpr=0.01)
    for k in stored:
        classic.add(k)
    classic_fpr = sum(classic.query(k) for k in test_neg) / len(test_neg)

    assert lbf.false_positive_rate(test_neg) <= classic_fpr
    assert lbf.compare_with_traditional(N)["learned_bytes"] < classic.size_bytes


def test_unlearnable_negatives_still_meet_fpr_target(stored):
    # Absent keys look exactly like stored ones: the model can't separate them, so
    # calibration pushes almost every key into the backup filter. FPR must still hold.
    lbf = LearnedBloomFilter(model_fpr=0.005, backup_fpr=0.005)
    lbf.fit(stored, _keys(["user"], 3_000, seed=5))
    assert all(lbf.query(k) for k in stored)
    assert lbf.false_positive_rate(_keys(["user"], 5_000, seed=6)) < 0.02


def test_query_before_fit_raises():
    with pytest.raises(RuntimeError):
        LearnedBloomFilter().query("x")
