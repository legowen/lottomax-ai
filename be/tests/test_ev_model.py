"""Popularity model, Smart Pick v2, jackpot EV math and their endpoints."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import app as lotto
import ev_model as ev

POOL = ev.CURRENT_POOL


@pytest.fixture(scope="module")
def w():
    return ev.load_weights(POOL)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    with TestClient(lotto.app) as c:
        yield c


# ---------------- popularity ratio ----------------

def test_popular_pattern_beats_guarded_high_combo(w):
    popular = ev.popularity_ratio([1, 2, 3, 4, 5, 6, 7], POOL, w=w)
    high = ev.popularity_ratio([33, 35, 39, 42, 45, 49, 52], POOL, w=w)
    assert popular > high * 100
    assert popular > 1 > high


def test_uniform_random_mean_ratio_is_one(w):
    rng = np.random.default_rng(999)   # different seed than the normalisation sample
    combos = ev.random_combos(rng, 200_000, POOL)
    r = ev.combo_scores(combos, w, POOL) / ev.e_uniform(w, POOL)
    assert r.mean() == pytest.approx(1.0, abs=0.05)


def test_pattern_detectors():
    c = np.array([[2, 6, 10, 14, 20, 33, 45],        # 2,6,10,14 arithmetic (gap 4)
                  [1, 4, 9, 16, 25, 36, 49],         # squares: none
                  [5, 6, 7, 21, 30, 38, 50],         # triple run only
                  [1, 5, 9, 13, 17, 21, 25]], dtype=np.int16)
    assert ev.has_arithmetic4(c, POOL).tolist() == [True, False, False, True]
    assert ev.has_triple_run(c).tolist() == [False, False, True, False]


def test_pattern_multipliers_apply(w):
    base = float(w[[20, 33, 37, 41, 44, 48, 50]].prod())
    combo = np.array([[20, 33, 37, 41, 44, 48, 50]], dtype=np.int16)
    assert ev.combo_scores(combo, w, POOL)[0] == pytest.approx(base)          # no pattern
    low = np.array([[2, 6, 10, 14, 20, 26, 29]], dtype=np.int16)              # AP4 + all <= 31
    expected = float(w[low[0]].prod()) * 4.0 * 2.5
    assert ev.combo_scores(low, w, POOL)[0] == pytest.approx(expected)


def test_past_winner_multiplier(w):
    combo = [20, 33, 37, 41, 44, 48, 50]
    plain = ev.popularity_ratio(combo, POOL, w=w)
    hist = {frozenset(combo)}
    assert ev.popularity_ratio(combo, POOL, historical_sets=hist, w=w) == pytest.approx(plain * 5.0)


def test_weights_and_override(tmp_path):
    d = ev.default_weights(POOL)
    assert d[5] == pytest.approx(1.6) and d[20] == pytest.approx(1.25) and d[40] == pytest.approx(0.75)
    assert d[7] == pytest.approx(1.6 * 1.25) and d[13] == pytest.approx(1.25 * 1.25)
    p = tmp_path / "popularity_override.json"
    p.write_text(json.dumps({"40": 2.0, "5": -1, "99": 3, "x": 1}))   # bad entries ignored
    o = ev.load_weights(POOL, p)
    assert o[40] == 2.0 and o[5] == pytest.approx(1.6)
    p.write_text("not json")
    assert ev.load_weights(POOL, p)[40] == pytest.approx(0.75)


# ---------------- Poisson share math ----------------

def test_share_factor_limits_and_monotonic():
    assert ev.share_factor(0.0) == 1.0
    assert ev.share_factor(1e-9) == pytest.approx(1.0, abs=1e-8)
    assert ev.share_factor(1e-6) == pytest.approx(1.0, abs=1e-5)
    lams = [1e-4, 1e-2, 0.1, 0.5, 1, 2, 5, 20, 100]
    vals = [ev.share_factor(x) for x in lams]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    assert vals[-1] == pytest.approx(1 / 100, rel=0.02)


def test_expected_share_decreases_with_lambda():
    shares = [ev.ev_estimate(5e7, n, 6, 4, 0, 1.0)["random"]["share_if_win"]
              for n in (0, 1e6, 1e7, 5e7, 2e8)]
    assert shares[0] == 5e7
    assert all(a > b for a, b in zip(shares, shares[1:]))


def test_ev_estimate_matches_formula():
    c = math.comb(52, 7)
    r = ev.ev_estimate(5e7, 25e6, 6.0, 4, 0.1)
    lam = 25e6 / c
    share = 5e7 * (1 - math.exp(-lam)) / lam
    assert r["combinations"] == c == 133_784_560
    assert r["cost_per_line"] == 1.5
    assert r["random"]["lambda"] == pytest.approx(lam)
    assert r["random"]["share_if_win"] == pytest.approx(share)
    assert r["random"]["ev_jackpot"] == pytest.approx(share / c)
    assert r["net_ev_random"] == pytest.approx(share / c + 0.1 - 1.5)
    assert r["chosen"] is None and r["net_ev_chosen"] is None


def test_low_popularity_ticket_has_better_ev():
    lo = ev.ev_estimate(5e7, 5e8, 6, 4, 0, 0.1)
    hi = ev.ev_estimate(5e7, 5e8, 6, 4, 0, 5.0)
    assert lo["net_ev_chosen"] > lo["net_ev_random"] > hi["net_ev_chosen"]


# ---------------- Smart Pick v2 sampling ----------------

def test_candidates_pass_guard_and_ratios_positive(w):
    rng = np.random.default_rng(1)
    combos, ratios = ev.candidate_pool(rng, POOL, None, w, n_candidates=3000)
    assert (combos <= 31).sum(axis=1).max() <= ev.EV_MAX_LOW
    assert not ev.has_triple_run(combos).any()
    assert (ratios > 0).all()


def test_tickets_are_low_popularity_and_valid():
    ts = ev.smart_tickets(5, POOL, None, seed=5)
    for t in ts:
        assert len(set(t["numbers"])) == 7 and all(1 <= n <= POOL for n in t["numbers"])
        assert t["popularity_ratio"] < 1.0
        assert t["low_count"] <= ev.EV_MAX_LOW


def test_batch_overlap_at_most_three_direct():
    ts = ev.smart_tickets(10, POOL, None, seed=11)
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            assert len(set(ts[i]["numbers"]) & set(ts[j]["numbers"])) <= 3


def test_same_seed_reproducible_but_default_random():
    assert ev.smart_tickets(2, POOL, None, seed=3) == ev.smart_tickets(2, POOL, None, seed=3)
    picks = {tuple(ev.smart_tickets(1, POOL, None)[0]["numbers"]) for _ in range(12)}
    assert len(picks) >= 10          # not a deterministic argmin


# ---------------- endpoints ----------------

def test_predict_batch_endpoint(client):
    r = client.post("/predict-batch", json={"count": 6}).json()
    assert len(r["tickets"]) == 6 and "추정치" in r["note"]
    sets = [set(t["numbers"]) for t in r["tickets"]]
    for i in range(6):
        for j in range(i + 1, 6):
            assert len(sets[i] & sets[j]) <= 3
    assert {"numbers", "popularity_ratio", "low_count", "sum", "share_risk"} <= set(r["tickets"][0])


def test_predict_batch_not_fixed_across_requests(client):
    outs = {tuple(client.post("/predict-batch", json={"count": 1}).json()["tickets"][0]["numbers"])
            for _ in range(8)}
    assert len(outs) >= 6


@pytest.mark.parametrize("count", [0, 11, -1, "x"])
def test_predict_batch_range_422(client, count):
    assert client.post("/predict-batch", json={"count": count}).status_code == 422


def test_ev_endpoint_basic_and_chosen(client):
    r = client.post("/ev/estimate", json={"jackpot": 50_000_000, "tickets_sold": 25_000_000,
                                          "ticket_price": 6.0, "lines_per_ticket": 4,
                                          "other_prizes_ev": 0.0, "numbers": None}).json()
    assert r["combinations"] == 133_784_560 and r["chosen"] is None
    assert r["random"]["popularity_ratio"] == 1.0 and r["net_ev_random"] < 0
    assert any("추정치" in a for a in r["assumptions"])
    ch = client.post("/ev/estimate", json={"numbers": [4, 19, 33, 38, 41, 46, 49]}).json()
    assert ch["chosen"]["popularity_ratio"] < 1 and ch["net_ev_chosen"] is not None


@pytest.mark.parametrize("body", [
    {"jackpot": -1}, {"tickets_sold": -5}, {"ticket_price": 0}, {"ticket_price": -1},
    {"lines_per_ticket": 0}, {"other_prizes_ev": -0.1},
    {"jackpot": "nan"}, {"tickets_sold": "inf"}, {"jackpot": "-inf"},
    {"numbers": [1, 2, 3]}, {"numbers": [1, 1, 2, 3, 4, 5, 6]}, {"numbers": [1, 2, 3, 4, 5, 6, 99]},
])
def test_ev_endpoint_validation_422(client, body):
    assert client.post("/ev/estimate", json=body).status_code == 422


def test_ev_endpoint_zero_tickets_sold_ok(client):
    r = client.post("/ev/estimate", json={"tickets_sold": 0}).json()
    assert r["random"]["share_if_win"] == 50_000_000
