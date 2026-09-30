"""Typical-set generator: exact sum distribution, typical mask, sampling, the distribution
strategy bucket fix, API modes, and distribution checks of generated tickets vs real history."""
import itertools
import sys
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

import app as lotto
import ev_model as ev
import generator as gen

POOL = gen.CURRENT_POOL


@pytest.fixture(scope="module")
def loaded():
    lotto.load_csv_data()
    return lotto.state


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    with TestClient(lotto.app) as c:
        yield c


def _overlap(a, b):
    return len(set(a) & set(b))


# ---------------- exact sum distribution ----------------

def test_sum_pmf_matches_brute_force():
    pool, pick = 12, 4
    counts = Counter(sum(c) for c in itertools.combinations(range(1, pool + 1), pick))
    pmf = gen.sum_pmf(pool, pick)
    for s, n in counts.items():
        assert pmf[s] * comb(pool, pick) == pytest.approx(n)
    assert pmf.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("pool", [50, 52])
def test_sum_pmf_moments(pool):
    mean, sd = gen.sum_stats(pool, 7)
    assert mean == pytest.approx(7 * (pool + 1) / 2)
    # finite-population variance: n * (N^2 - 1) / 12 * (N - n) / (N - 1)
    assert sd == pytest.approx(np.sqrt(7 * (pool ** 2 - 1) / 12 * (pool - 7) / (pool - 1)))


def test_sum_bounds_hold_central_mass():
    lo, hi = gen.sum_bounds(POOL, 7)
    assert (lo, hi) == (124, 247)
    assert gen.sum_pmf(POOL, 7)[lo:hi + 1].sum() >= 0.90


# ---------------- typical-set definition ----------------

def test_typical_mask_examples():
    c = np.array([
        [5, 14, 23, 31, 38, 44, 50],     # typical: sum 205, 3 odd, 5 decade groups
        [1, 2, 3, 4, 5, 6, 7],           # sum 28
        [46, 47, 48, 49, 50, 51, 52],    # sum 343
        [10, 12, 20, 24, 30, 36, 44],    # sum in band but 0 odd numbers
        [21, 22, 23, 25, 26, 28, 30],    # sum in band, 3 odd, but a single decade group
    ], dtype=np.int16)
    assert gen.typical_mask(c, POOL).tolist() == [True, False, False, False, False]
    assert gen.decade_count(c, POOL).tolist() == [5, 1, 2, 5, 1]


def test_describe_fields():
    d = gen.describe([5, 14, 23, 31, 38, 44, 50], POOL)
    assert d["sum"] == 205 and d["sum_band"] == [124, 247] and d["sum_mean"] == 185.5
    assert d["odd_count"] == 3 and d["decades"] == 5 and d["typical"] is True
    assert gen.describe([46, 47, 48, 49, 50, 51, 52], POOL)["typical"] is False


# ---------------- sampling ----------------

def test_realistic_tickets_are_typical_and_diverse(loaded):
    tickets = [gen.sample_tickets(1, POOL, loaded["historical_sets"], "realistic", seed=i)[0]
               for i in range(60)]
    nums = [tuple(t["numbers"]) for t in tickets]
    assert all(t["typical"] for t in tickets)
    assert all(len(set(n)) == 7 and all(1 <= x <= POOL for x in n) for n in nums)
    assert len(set(nums)) >= 58                                   # not the same ticket over and over
    assert len({x for n in nums for x in n}) >= 45                # numbers spread over the whole pool
    assert abs(np.mean([t["sum"] for t in tickets]) - 185.5) < 15


def test_same_seed_same_ticket_different_seed_different(loaded):
    a = gen.sample_tickets(1, POOL, loaded["historical_sets"], "realistic", seed=5)
    b = gen.sample_tickets(1, POOL, loaded["historical_sets"], "realistic", seed=5)
    c = gen.sample_tickets(1, POOL, loaded["historical_sets"], "realistic", seed=6)
    assert a[0]["numbers"] == b[0]["numbers"] != c[0]["numbers"]


def test_balanced_respects_ev_guard_and_is_less_popular(loaded):
    hs = loaded["historical_sets"]
    bal = [gen.sample_tickets(1, POOL, hs, "balanced", seed=100 + i)[0] for i in range(12)]
    real = [gen.sample_tickets(1, POOL, hs, "realistic", seed=200 + i)[0] for i in range(40)]
    for t in bal:
        assert t["typical"]
        assert t["low_count"] <= ev.EV_MAX_LOW
        assert not lotto._has_triple_run(t["numbers"])
        assert frozenset(t["numbers"]) not in hs
    assert np.mean([t["popularity_ratio"] for t in bal]) < np.mean([t["popularity_ratio"] for t in real])


def test_batch_overlap_limit_and_uniqueness(loaded):
    for mode in ("realistic", "balanced"):
        tickets = gen.sample_tickets(6, POOL, loaded["historical_sets"], mode, seed=3)
        sets = [t["numbers"] for t in tickets]
        assert len({tuple(s) for s in sets}) == 6
        for a, b in itertools.combinations(sets, 2):
            assert _overlap(a, b) <= 3


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        gen.sample_tickets(1, POOL, None, "nope")


# ---------------- distribution strategy bucket fix ----------------

def _legacy_distribution(draws, num_range):
    """The original implementation, valid while the pool was <= 50 numbers."""
    scores = np.zeros(num_range + 1)
    recent = draws[-lotto.RECENT_WINDOW:]
    range_size = 10 if num_range <= 50 else 20
    num_ranges = (num_range + range_size - 1) // range_size
    range_counts = np.zeros(num_ranges)
    odd = even = 0
    for d in recent:
        for n in d:
            idx = (n - 1) // range_size
            if idx < num_ranges:
                range_counts[idx] += 1
            odd, even = odd + n % 2, even + 1 - n % 2
    expected = sum(len(d) for d in recent) / num_ranges
    for n in range(1, num_range + 1):
        idx = (n - 1) // range_size
        if idx < num_ranges:
            ratio = range_counts[idx] / expected
            if ratio < 1:
                scores[n] += (1 - ratio) * 0.5
        if n % 2 == 1 and odd / (odd + even) < 0.45:
            scores[n] += 0.2
        elif n % 2 == 0 and even / (odd + even) < 0.45:
            scores[n] += 0.2
    return scores


def _random_history(rng, length, pool):
    return ev.random_combos(rng, length, pool).tolist()


def test_distribution_strategy_unchanged_for_50_pool():
    rng = np.random.default_rng(11)
    for _ in range(20):
        hist = _random_history(rng, 40, 50)
        assert np.allclose(lotto.strategy_distribution(hist, 50), _legacy_distribution(hist, 50))


def test_distribution_strategy_has_no_permanent_bonus_for_top_numbers():
    """Regression: with 52 numbers the last bucket (41-52) used to look under-represented forever."""
    rng = np.random.default_rng(12)
    hi, lo = [], []
    for _ in range(300):
        s = lotto.strategy_distribution(_random_history(rng, 30, 52), 52)
        hi.append(s[41:].mean())
        lo.append(s[1:41].mean())
    assert abs(np.mean(hi) - np.mean(lo)) < 0.02
    # the legacy code gave 41-52 a constant +0.057 over 1-40 on random data
    legacy = [np.mean(_legacy_distribution(_random_history(rng, 30, 52), 52)[41:]) for _ in range(50)]
    assert np.mean(legacy) > 0.04


# ---------------- distribution checks: generated tickets vs exact null vs real history ----------------

def _binned(pmf, edges, n):
    exp = np.array([pmf[lo:hi].sum() * n for lo, hi in zip(edges[:-1], edges[1:])])
    return exp


def test_uniform_sampler_reproduces_exact_sum_distribution():
    """The random source behind both modes is uniform: its sums follow the exact pmf (chi-square)."""
    n = 200_000
    combos = ev.random_combos(np.random.default_rng(20260930), n, POOL)
    sums = combos.astype(np.int64).sum(axis=1)
    edges = [0, 100] + list(range(110, 270, 10)) + [280, 7 * POOL + 1]
    obs, _ = np.histogram(sums, bins=edges)
    exp = _binned(gen.sum_pmf(POOL, 7), edges, n)
    assert exp.min() >= 5
    assert stats.chisquare(obs, exp).pvalue > 0.001


def test_uniform_sampler_numbers_are_equiprobable():
    combos = ev.random_combos(np.random.default_rng(7), 100_000, POOL)
    counts = np.bincount(combos.ravel(), minlength=POOL + 1)[1:]
    assert stats.chisquare(counts).pvalue > 0.001


def _era2(loaded):
    draws = [d for d, p in zip(loaded["main_draws"], loaded["main_draw_pools"]) if p == 50]
    return np.array(draws, dtype=np.int16)


def test_history_sits_inside_the_typical_set_at_the_expected_rate(loaded):
    real = _era2(loaded)                                      # 7/50 era: 722+ draws, fixed pool
    rng = np.random.default_rng(1)
    rate = gen.typical_mask(ev.random_combos(rng, 300_000, 50), 50).mean()
    k = int(gen.typical_mask(real, 50).sum())
    assert stats.binomtest(k, len(real), rate).pvalue > 0.001
    lo, hi = gen.sum_bounds(50, 7)
    band = float(gen.sum_pmf(50, 7)[lo:hi + 1].sum())
    k_band = int(((real.sum(axis=1) >= lo) & (real.sum(axis=1) <= hi)).sum())
    assert stats.binomtest(k_band, len(real), band).pvalue > 0.001


def test_generated_tickets_match_real_draws_inside_the_typical_set(loaded):
    """Two-sample checks: real typical draws vs generated typical tickets (same 7/50 pool)."""
    real = _era2(loaded)
    real_typ = real[gen.typical_mask(real, 50)]
    made = gen.typical_candidates(np.random.default_rng(2), 5_000, 50)

    # sum: Kolmogorov-Smirnov
    assert stats.ks_2samp(real_typ.sum(axis=1), made.sum(axis=1)).pvalue > 0.001

    # odd count and occupied decade groups: chi-square homogeneity
    for feature in (lambda c: (c % 2).sum(axis=1), lambda c: gen.decade_count(c, 50)):
        a, b = feature(real_typ), feature(made)
        cats = sorted(set(a) | set(b))
        table = np.array([[np.sum(a == c) for c in cats], [np.sum(b == c) for c in cats]])
        assert stats.chi2_contingency(table)[1] > 0.001


def test_legacy_ensemble_top7_is_not_a_real_draw_shape(loaded):
    """Documents why the ensemble is no longer the default: top-7-of-scores piles onto one region."""
    picks = [tuple(lotto.ensemble_predict(loaded["main_draws"], POOL, 7, None, None,
                                          historical_sets=loaded["historical_sets"])["numbers"])
             for _ in range(30)]
    assert len(set(picks)) < 30                                # same tickets come back
    assert len({n for p in picks for n in p}) < 30             # only a fraction of the pool is ever used


# ---------------- API ----------------

def test_api_predict_realistic(client):
    r = client.post("/predict", json={"mode": "realistic"}).json()
    m = r["main"]
    assert r["mode"] == "realistic"
    assert len(set(m["numbers"])) == 7 and all(1 <= n <= POOL for n in m["numbers"])
    assert m["typicality"]["typical"] is True and m["confidence"] is None and m["ev_info"] is None
    assert m["typicality"]["sum_band"] == [124, 247]


def test_api_predict_realistic_varies(client):
    seen = {tuple(client.post("/predict", json={"mode": "realistic"}).json()["main"]["numbers"])
            for _ in range(8)}
    assert len(seen) >= 7


def test_api_predict_balanced(client):
    m = client.post("/predict", json={"mode": "balanced"}).json()["main"]
    assert m["typicality"]["typical"] is True
    assert m["ev_info"]["mode"] == "balanced" and m["ev_info"]["guard_applied"] is True
    assert m["ev_info"]["low_count"] <= ev.EV_MAX_LOW


def test_api_default_and_legacy_modes_report_typicality(client):
    for body in ({}, {"mode": "ensemble"}, {"mode": "smart_v2"}):
        r = client.post("/predict", json=body).json()
        assert isinstance(r["main"]["typicality"]["typical"], bool)
        assert "note" in r["main"]["typicality"]


def test_api_batch_modes(client):
    d = client.post("/predict-batch", json={"count": 3}).json()
    assert d["mode"] == "smart_v2" and len(d["tickets"]) == 3
    for mode in ("realistic", "balanced"):
        d = client.post("/predict-batch", json={"count": 5, "mode": mode}).json()
        assert d["mode"] == mode and len(d["tickets"]) == 5
        assert all(t["typical"] for t in d["tickets"])
    assert client.post("/predict-batch", json={"count": 2, "mode": "nope"}).status_code == 422
