"""Backend test suite: era handling, EV strategy, ensemble, backtest, API."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import app as lotto
from backtest import run_walk_forward, _normalize


@pytest.fixture(scope="module")
def loaded():
    lotto.load_csv_data()
    return lotto.state


# ---------------- data loading / era handling ----------------

def test_era_split(loaded):
    assert len(loaded["all_draws"]) == 1211
    assert len(loaded["main_draws"]) == 708
    assert len(loaded["main_draws_dated"]) == 708
    # era-2 starts at the 7/50 format change
    assert loaded["main_draws_dated"][0]["date"] == "2019-05-14"


def test_historical_sets_cover_full_history(loaded):
    assert len(loaded["historical_sets"]) == 1211
    assert frozenset(loaded["all_draws"][0]) in loaded["historical_sets"]


def test_draw_integrity(loaded):
    for d in loaded["main_draws"]:
        assert len(set(d)) == 7
        assert all(1 <= n <= 50 for n in d)


def test_number_50_only_in_era2(loaded):
    pre = loaded["all_draws"][: len(loaded["all_draws"]) - len(loaded["main_draws"])]
    assert max(n for d in pre for n in d) == 49


# ---------------- smart pick / EV guard ----------------

def test_smart_pick_prefers_unpopular():
    s = lotto.strategy_smart_pick(50)
    assert s[45] > s[20] > s[7]          # high > mid > lucky-low
    assert s[32] == 1.0
    assert s[7] == pytest.approx(0.30 - 0.15)
    assert s[0] == 0


def test_has_triple_run():
    assert lotto._has_triple_run([5, 6, 7])
    assert lotto._has_triple_run([1, 9, 20, 21, 22, 40, 50])
    assert not lotto._has_triple_run([1, 2, 4, 5, 7, 8, 10])
    assert not lotto._has_triple_run([10, 20, 30])


def test_ev_guard_caps_low_numbers_and_runs():
    ranked = list(range(1, 51))  # worst case: low numbers ranked best
    pick = lotto.apply_ev_guard(ranked, 7)
    assert pick == [1, 2, 4, 5, 32, 33, 35]
    assert sum(1 for n in pick if n <= 31) <= lotto.EV_MAX_LOW_NUMBERS
    assert not lotto._has_triple_run(pick)


def test_ev_guard_avoids_past_winners():
    ranked = list(range(1, 51))
    past = {frozenset([1, 2, 4, 5, 32, 33, 35])}
    pick = lotto.apply_ev_guard(ranked, 7, historical_sets=past)
    assert frozenset(pick) not in past
    assert len(set(pick)) == 7
    assert not lotto._has_triple_run(pick)
    assert sum(1 for n in pick if n <= 31) <= lotto.EV_MAX_LOW_NUMBERS


def test_ev_guard_fallback_fills_seven():
    # Constraints impossible to satisfy from a tiny pool -> still returns 7
    ranked = list(range(1, 9))
    pick = lotto.apply_ev_guard(ranked, 7)
    assert len(pick) == 7
    assert len(set(pick)) == 7


# ---------------- normalize ----------------

def test_normalize_all_zero():
    arr = np.zeros(51)
    out = lotto.normalize(arr)
    assert out.sum() == 0


def test_normalize_range():
    arr = np.zeros(51)
    arr[1], arr[2], arr[3] = 1.0, 2.0, 3.0
    out = lotto.normalize(arr)
    assert out[3] == 1.0 and out[1] == 0.0
    assert out[0] == 0


# ---------------- ensemble ----------------

def test_ensemble_predict_shape(loaded):
    r = lotto.ensemble_predict(
        loaded["main_draws"], 50, 7, None, None,
        draws_dated=None, historical_sets=loaded["historical_sets"],
    )
    nums = r["numbers"]
    assert len(nums) == 7 and len(set(nums)) == 7
    assert all(1 <= n <= 50 for n in nums)
    assert nums == sorted(nums)
    first = r["strategies"][str(nums[0])]
    assert "smart" in first and "seed" in first
    ev = r["ev_info"]
    assert ev["guard_applied"] is True
    assert ev["low_count"] <= lotto.EV_MAX_LOW_NUMBERS
    assert ev["is_past_winner"] is False


def test_ensemble_without_smart_weight(loaded):
    w = {"frequency": 0.5, "gap": 0.5}
    r = lotto.ensemble_predict(loaded["main_draws"], 50, 7, None, w,
                               historical_sets=loaded["historical_sets"])
    assert len(r["numbers"]) == 7
    assert r["ev_info"]["guard_applied"] is False


def test_ensemble_tiny_history():
    r = lotto.ensemble_predict([[1, 2, 3, 4, 5, 6, 7]], 50, 7)
    assert len(r["numbers"]) == 7
    assert r["confidence"] == 0


# ---------------- backtest engine ----------------

def test_walk_forward_sanity(loaded):
    strategies = {
        "frequency": lambda d: lotto.strategy_frequency_recency(d, 50),
        "smart": lambda d: lotto.strategy_smart_pick(50),
    }
    r = run_walk_forward(loaded["main_draws"], 50, 7, strategies,
                         window=40, n_random=20)
    assert r["window"] == 40
    assert r["expected_random"] == pytest.approx(0.98)
    assert set(r["results"]) == {"frequency", "smart", "ensemble"}
    for s in r["results"].values():
        assert 0 <= s["mean"] <= 7
        assert sum(s["dist"].values()) == 40
    assert 0.5 < r["random_baseline_mean"] < 1.5
    assert "verdict" in r


def test_backtest_normalize_matches_app():
    arr = np.zeros(51)
    arr[5], arr[10] = 2.0, 4.0
    assert np.allclose(_normalize(arr.copy()), lotto.normalize(arr.copy()))


# ---------------- API ----------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    with TestClient(lotto.app) as c:
        yield c


def test_api_health(client):
    r = client.get("/").json()
    assert r["status"] == "ok"
    assert r["main_draws"] == 708
    assert r["all_draws"] == 1211
    assert r["era_start"] == "2019-05-14"


def test_api_frequencies(client):
    r = client.get("/frequencies").json()
    assert r["total_draws"] == 708
    # era-2 counts: number 50 must NOT look artificially rare
    counts = [int(v) for v in r["main_total"].values()]
    assert min(counts) > 0.5 * (sum(counts) / len(counts))


def test_api_predict(client):
    r = client.post("/predict", json={}).json()
    nums = r["main"]["numbers"]
    assert len(nums) == 7 and all(1 <= n <= 50 for n in nums)
    assert "ev_info" in r["main"]


def test_api_predict_custom_weights(client):
    r = client.post("/predict", json={"weights": {"smart": 1.0}}).json()
    nums = r["main"]["numbers"]
    assert sum(1 for n in nums if n <= 31) <= lotto.EV_MAX_LOW_NUMBERS


def test_api_backtest(client):
    r = client.post("/backtest", json={"window": 30, "random_tickets": 20})
    assert r.status_code == 200
    body = r.json()
    assert body["window"] == 30
    assert "smart" in body["results"]
    assert "verdict" in body


def test_api_reload(client):
    r = client.post("/reload-data").json()
    assert r["main_draws"] == 708 and r["all_draws"] == 1211


# ---------------- adversarial regression tests (Stage 3) ----------------

def test_api_rejects_non_numeric_weights(client):
    r = client.post("/predict", json={"weights": {"smart": "high"}})
    assert r.status_code == 422


def test_api_rejects_nan_weight(client):
    r = client.post("/predict", json={"weights": {"gap": "nan"}})
    assert r.status_code == 422


def test_api_clamps_extreme_weights(client):
    r = client.post("/predict", json={"weights": {"smart": -5, "gap": 999}})
    assert r.status_code == 200
    assert len(r.json()["main"]["numbers"]) == 7


def test_api_all_zero_weights_confidence_zero(client):
    zero = {k: 0 for k in lotto.DEFAULT_WEIGHTS}
    r = client.post("/predict", json={"weights": zero}).json()
    assert r["main"]["confidence"] == 0
    assert "note" in r["main"]["ev_info"]


def test_api_train_rejected_while_training(client):
    lotto.state["is_training"] = True
    try:
        r = client.post("/train", json={"epochs": 1})
        assert r.status_code == 400
    finally:
        lotto.state["is_training"] = False


def test_normalize_constant_scores_capped():
    # All-equal positive scores must map to 1.0, never leak raw magnitude
    arr = np.zeros(51)
    arr[1:] = 500.0
    out = lotto.normalize(arr.copy())
    assert out[1:].max() == 1.0 and out[1:].min() == 1.0
    assert np.allclose(out, _normalize(arr.copy()))


def test_smart_selection_respects_guard(loaded):
    # The backtested smart pick must use the same EV guard production uses
    scores = lotto.strategy_smart_pick(50)
    ranked = sorted(range(1, 51), key=lambda n: -scores[n])
    pick = lotto.apply_ev_guard(ranked, 7, loaded["historical_sets"])
    assert not lotto._has_triple_run(pick)
    assert sum(1 for n in pick if n <= 31) <= lotto.EV_MAX_LOW_NUMBERS
    assert frozenset(pick) not in loaded["historical_sets"]


def test_share_risk_reflects_past_winner(loaded):
    past = next(iter(loaded["historical_sets"]))
    # Force a scenario where the pick equals a past winner via guard bypass
    r = lotto.ensemble_predict(loaded["main_draws"], 50, 7, None,
                               {"frequency": 1.0}, historical_sets=loaded["historical_sets"])
    ev = r["ev_info"]
    if ev["is_past_winner"] or lotto._has_triple_run(r["numbers"]):
        assert ev["share_risk"] == "high"
    assert ev["guard_applied"] is False  # smart weight 0 -> guard off -> risk high
    assert ev["share_risk"] == "high"
