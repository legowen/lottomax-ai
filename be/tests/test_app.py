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
    assert len(loaded["all_draws"]) == 1273
    assert len(loaded["main_draws"]) == 770
    assert len(loaded["main_draws_dated"]) == 770
    # 722 draws in the 7/50 era + 48 in the 7/52 era (since 2026-04-14)
    assert loaded["main_draw_pools"].count(50) == 722
    assert loaded["main_draw_pools"].count(52) == 48
    # era-2 starts at the 7/50 format change
    assert loaded["main_draws_dated"][0]["date"] == "2019-05-14"


def test_historical_sets_cover_full_history(loaded):
    assert len(loaded["historical_sets"]) == 1273
    assert frozenset(loaded["all_draws"][0]) in loaded["historical_sets"]


def test_draw_integrity(loaded):
    for d, pool in zip(loaded["main_draws"], loaded["main_draw_pools"]):
        assert len(set(d)) == 7
        assert all(1 <= n <= pool for n in d)
    # 51/52 only ever appear in the 7/52 era
    assert max(n for d, p in zip(loaded["main_draws"], loaded["main_draw_pools"]) if p == 50 for n in d) == 50
    assert max(n for d in loaded["main_draws"] for n in d) == 52


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
        loaded["main_draws"], lotto.LOTTO_MAX, 7, None, None,
        draws_dated=None, historical_sets=loaded["historical_sets"],
    )
    nums = r["numbers"]
    assert len(nums) == 7 and len(set(nums)) == 7
    assert all(1 <= n <= lotto.LOTTO_MAX for n in nums)
    assert nums == sorted(nums)
    first = r["strategies"][str(nums[0])]
    assert "smart" in first and "seed" in first
    ev = r["ev_info"]
    assert ev["guard_applied"] is True
    assert ev["low_count"] <= lotto.EV_MAX_LOW_NUMBERS
    assert ev["is_past_winner"] is False


def test_ensemble_without_smart_weight(loaded):
    w = {"frequency": 0.5, "gap": 0.5}
    r = lotto.ensemble_predict(loaded["main_draws"], lotto.LOTTO_MAX, 7, None, w,
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
        "frequency": lambda d: lotto.strategy_frequency_recency(d, lotto.LOTTO_MAX),
        "smart": lambda d: lotto.strategy_smart_pick(lotto.LOTTO_MAX),
    }
    r = run_walk_forward(loaded["main_draws"], lotto.LOTTO_MAX, 7, strategies,
                         window=40, n_random=20, pools=loaded["main_draw_pools"])
    assert r["window"] == 40
    assert r["expected_random"] == pytest.approx(49 / 52, abs=1e-4)
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
    assert r["main_draws"] == 770
    assert r["all_draws"] == 1273
    assert r["era_start"] == "2019-05-14"
    assert r["pool_size"] == 52
    assert "lstm_verdict" in r


def test_api_frequencies(client):
    r = client.get("/frequencies").json()
    assert r["total_draws"] == 770
    # era-2 counts: number 50 must NOT look artificially rare (51/52 are new, so excluded)
    counts = [int(v) for k, v in r["main_total"].items() if int(k) <= 50]
    assert min(counts) > 0.5 * (sum(counts) / len(counts))
    assert set(r["main_total"]) == {str(i) for i in range(1, 53)}


def test_api_predict(client):
    r = client.post("/predict", json={}).json()
    nums = r["main"]["numbers"]
    assert len(nums) == 7 and all(1 <= n <= lotto.LOTTO_MAX for n in nums)
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
    assert r["main_draws"] == 770 and r["all_draws"] == 1273


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
    r = lotto.ensemble_predict(loaded["main_draws"], lotto.LOTTO_MAX, 7, None,
                               {"frequency": 1.0}, historical_sets=loaded["historical_sets"])
    ev = r["ev_info"]
    if ev["is_past_winner"] or lotto._has_triple_run(r["numbers"]):
        assert ev["share_risk"] == "high"
    assert ev["guard_applied"] is False  # smart weight 0 -> guard off -> risk high
    assert ev["share_risk"] == "high"


# ---------------- v5 additions ----------------

def test_predict_mode_ensemble_backward_compatible(client):
    a = client.post("/predict", json={}).json()
    b = client.post("/predict", json={"mode": "ensemble"}).json()
    for r in (a, b):
        assert r["mode"] == "ensemble"
        assert len(r["main"]["numbers"]) == 7
        assert "strategies" in r["main"] and "share_risk" in r["main"]["ev_info"]
    assert client.post("/predict", json={"mode": "nope"}).status_code == 422


def test_predict_mode_smart_v2(client):
    r = client.post("/predict", json={"mode": "smart_v2"}).json()
    ev = r["main"]["ev_info"]
    assert len(set(r["main"]["numbers"])) == 7
    assert ev["popularity_ratio"] > 0 and ev["mode"] == "smart_v2"
    assert ev["low_count"] <= lotto.EV_MAX_LOW_NUMBERS


def test_history_check_exact_match_and_histogram(client, loaded):
    past = next(d for d in loaded["all_draws"][-5:])
    r = client.post("/history-check", json={"numbers": past}).json()
    assert r["exact_match"] is not None and r["max_overlap"] == 7
    assert sum(r["overlap_histogram"].values()) == 1273
    assert r["overlap_histogram"]["7"] >= 1
    # the random-ticket expectation must also sum to the number of draws
    assert sum(r["expected_histogram_random"].values()) == pytest.approx(1273, abs=0.5)


def test_history_check_no_exact_match(client):
    r = client.post("/history-check", json={"numbers": [1, 2, 3, 4, 5, 6, 7]}).json()
    assert r["exact_match"] is None
    assert r["max_overlap"] <= 6


def test_history_check_validation(client):
    assert client.post("/history-check", json={"numbers": [1, 2, 3]}).status_code == 422
    assert client.post("/history-check", json={"numbers": [1, 1, 2, 3, 4, 5, 6]}).status_code == 422
    assert client.post("/history-check", json={"numbers": [1, 2, 3, 4, 5, 6, 53]}).status_code == 422


def test_status_has_lstm_verdict(client):
    r = client.get("/status").json()
    assert "lstm_verdict" in r


def test_lstm_verdict_logic():
    base = lotto.constant_baseline_loss(7 / 52)
    assert base == pytest.approx(0.3951, abs=1e-3)
    assert lotto.constant_baseline_loss(0.14) == pytest.approx(0.4051, abs=1e-3)
    worse = lotto.make_lstm_verdict(base + 0.01, 7 / 52)
    assert worse["beats_constant_baseline"] is False
    barely = lotto.make_lstm_verdict(base - 0.0003, 7 / 52)   # inside the 0.0005 margin
    assert barely["beats_constant_baseline"] is False
    better = lotto.make_lstm_verdict(base - 0.002, 7 / 52)
    assert better["beats_constant_baseline"] is True


def test_train_defaults_skip_seed_analysis(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(lotto, "run_training", lambda epochs, seed_flag=False: seen.update(flag=seed_flag))
    assert client.post("/train", json={"epochs": 1}).status_code == 200
    assert seen["flag"] is False
    lotto.state["is_training"] = False
    assert client.post("/train", json={"epochs": 1, "run_seed_analysis": True}).status_code == 200
    assert seen["flag"] is True
    lotto.state["is_training"] = False


def test_run_training_skips_seed_analysis_by_default(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("seed analysis must not run by default")
    monkeypatch.setattr(lotto, "run_seed_analysis", boom)
    monkeypatch.setattr(lotto, "train_lstm", lambda *a, **k: None)
    lotto.run_training(1)
    assert lotto.state["is_training"] is False
    assert lotto.state["training_progress"]["status"] == "complete"


def test_ensemble_seed_is_reproducible(loaded):
    kw = dict(historical_sets=loaded["historical_sets"])
    a = lotto.ensemble_predict(loaded["main_draws"], lotto.LOTTO_MAX, 7, None, None, seed=123, **kw)
    b = lotto.ensemble_predict(loaded["main_draws"], lotto.LOTTO_MAX, 7, None, None, seed=123, **kw)
    assert a["numbers"] == b["numbers"]


def test_new_numbers_are_neutralised():
    s = np.zeros(53)
    s[1:51] = np.linspace(1, 2, 50)
    s[51], s[52] = 9.0, 0.1
    out = lotto.neutralize_new_numbers(s)
    assert out[51] == out[52] == pytest.approx(s[1:51].mean())
    assert np.array_equal(out[:51], s[:51])


def test_cors_origins_env(monkeypatch):
    monkeypatch.delenv("LOTTOMAX_CORS_ORIGINS", raising=False)
    assert lotto.cors_origins() == ["http://localhost:5173", "http://127.0.0.1:5173"]
    monkeypatch.setenv("LOTTOMAX_CORS_ORIGINS", "https://a.example, https://b.example")
    assert lotto.cors_origins() == ["https://a.example", "https://b.example"]


@pytest.mark.skipif(not lotto.TF_AVAILABLE, reason="TensorFlow not installed")
def test_lstm_is_small():
    m = lotto.build_lstm_model(lotto.LOTTO_MAX)
    assert m.count_params() < 50_000


def test_lstm_verdict_roundtrip_json():
    import json
    v = lotto.make_lstm_verdict(0.41, 7 / 52)
    assert json.loads(json.dumps(v)) == v
