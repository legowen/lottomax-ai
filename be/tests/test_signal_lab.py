"""Signal Lab: false-positive control, detection power, predictor honesty, runtime, API."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import app as lotto
import signal_lab as sl


@pytest.fixture(scope="module")
def sim_sums():
    return sl.sim_ticket_sums(np.random.default_rng(0))


@pytest.fixture(scope="module")
def loaded():
    lotto.load_csv_data()
    return lotto.state


# ---------------- unit pieces ----------------

def test_bh_adjust_matches_definition():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    q = sl.bh_adjust(p)
    assert np.all(q >= p - 1e-12) and np.all(q <= 1)
    assert q[0] == pytest.approx(0.01)            # 0.001 * 10 / 1
    assert q[4] == pytest.approx(0.042 * 10 / 5)  # monotone step
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)


def test_battery_structure(sim_sums):
    d = sl.simulate_uniform(700, np.random.default_rng(1))
    b = sl.run_battery(d, sim_sums=sim_sums)
    names = [t["name"] for t in b["tests"]]
    assert b["n_tests"] == len(names) == 1 + 5 + 1 + 1 + 1 + 50 + 50
    assert names[0] == "uniformity_chi2" and "pair_chi2" in names and "sum_ks" in names
    assert b["expected_false_positives"] == pytest.approx(b["n_tests"] * 0.05, abs=0.01)
    for t in b["tests"]:
        assert 0 <= t["p_value"] <= 1 and t["p_value"] <= t["p_adj_bh"] + 1e-9
        assert t["significant"] == (t["p_adj_bh"] < 0.05)


def test_carry_probability_gives_target_overlap():
    r = sl.carry_probability_for_overlap(1.30)
    rng = np.random.default_rng(3)
    d = sl.simulate_lag1(4000, rng, 1.30)
    ov = np.mean([len(set(a) & set(b)) for a, b in zip(d[1:], d[:-1])])
    assert 0 < r < 1 and ov == pytest.approx(1.30, abs=0.06)


# ---------------- (1) false positives on uniform data ----------------

def test_battery_false_positive_rate_on_uniform(sim_sums):
    hits = 0
    n = 100
    for seed in range(n):
        d = sl.simulate_uniform(722, np.random.default_rng(1000 + seed))
        hits += sl.run_battery(d, sim_sums=sim_sums)["n_significant_after_bh"] > 0
    assert hits / n <= 0.10, f"false-positive rate {hits}/{n}"


# ---------------- (2) planted signals are detected ----------------

def test_power_check_detects_planted_signals():
    pc = sl.run_power_check(seed=7, n_draws=722, n_rep=30, n_null=40, with_predictor=False)
    assert pc["ok"] is True
    for p in pc["planted"]:
        assert p["detected"] and p["detection_rate"] >= 0.8
    assert pc["false_positive_rate_null"] <= 0.15
    assert pc["weak_effect_info"]["effect"] == "number_17_x1.35"


def test_lag1_signal_flagged_by_battery(sim_sums):
    d = sl.simulate_lag1(722, np.random.default_rng(5), 1.30)
    b = sl.run_battery(d, sim_sums=sim_sums)
    t = next(x for x in b["tests"] if x["name"] == "overlap_lag1_z")
    assert t["significant"] and t["p_adj_bh"] < 1e-3


def test_power_check_reports_failure_when_undetectable(monkeypatch):
    monkeypatch.setattr(sl, "simulate_number_boost", lambda t, rng, number=17, factor=1.5, n=50:
                        sl.simulate_uniform(t, rng, n))     # plant nothing
    pc = sl.run_power_check(seed=1, n_draws=300, n_rep=10, n_null=5, with_predictor=False)
    assert pc["ok"] is False
    bat = {"n_tests": 109, "n_significant_after_bh": 0, "expected_false_positives": 5.45}
    pred = {"delta": 0.0, "ci95": [-0.001, 0.001], "beats_baseline": False}
    v = sl.make_verdict(bat, pc, pred, {"auc": 0.5, "ci95": [0.47, 0.54], "p_value": 0.5})
    assert v.startswith("탐지기 신뢰 불가")


# ---------------- (3) predictor honesty ----------------

def test_predictor_does_not_beat_baseline_on_random():
    for seed in (11, 12, 13, 14, 15):
        d = sl.simulate_uniform(722, np.random.default_rng(seed))
        r = sl.run_predictor_test(d, backend="numpy", n_boot=500, n_perm=500)
        assert r["beats_baseline"] is False, (seed, r)
        assert r["ci95"][1] >= 0


def test_predictor_detects_planted_lag_signal():
    # 1.30 overlap is only ~70% detectable by the predictor (reported, not gated); use a clear effect
    d = sl.simulate_lag1(722, np.random.default_rng(21), 1.60)
    r = sl.run_predictor_test(d, backend="numpy", n_boot=500, n_perm=500)
    assert r["beats_baseline"] is True and r["ci95"][1] < 0 and r["delta"] < 0
    assert r["model"] == "logreg" and r["folds"] >= 10


def test_discriminator_null_vs_planted():
    null = sl.run_discriminator(sl.simulate_uniform(722, np.random.default_rng(31)), n_boot=200, n_perm=100)
    assert 0.40 < null["auc"] < 0.60 and null["p_value"] > 0.01
    sig = sl.run_discriminator(sl.simulate_lag1(722, np.random.default_rng(32), 1.30), n_boot=200, n_perm=100)
    assert sig["auc"] > 0.55 and sig["ci95"][0] > 0.5 and sig["p_value"] < 0.05


@pytest.mark.skipif(not lotto.TF_AVAILABLE, reason="TensorFlow not installed")
def test_predictor_tf_backend_runs():
    d = sl.simulate_uniform(320, np.random.default_rng(41))
    r = sl.run_predictor_test(d, backend="tf", n_boot=200, n_perm=200, min_train=100, step=100)
    assert r["model"] in ("mlp", "logreg") and r["beats_baseline"] in (True, False)
    assert r["folds"] == 2


# ---------------- (4) quick run on the real history ----------------

def test_quick_run_under_60_seconds(loaded):
    draws, last = lotto.signal_lab_draws()
    assert len(draws) == 722 and last == "2026-04-10"
    stages = []
    t0 = time.time()
    res = sl.run_signal_lab(draws, seed=42, quick=True, last_draw_date=last,
                            progress=lambda s, d, t: stages.append(s))
    assert time.time() - t0 < 60
    assert {"battery", "power_check", "predictor", "discriminator"} <= set(stages)
    assert res["n_draws"] == 722 and res["last_draw_date"] == "2026-04-10"
    assert set(res) >= {"generated_at", "battery", "power_check", "predictor", "discriminator", "verdict"}
    assert res["battery"]["n_tests"] == len(res["battery"]["tests"])
    assert res["power_check"]["ok"] is True
    assert res["predictor"]["beats_baseline"] is False
    assert isinstance(res["verdict"], str) and res["verdict"]


def test_quick_run_is_reproducible(loaded):
    draws, last = lotto.signal_lab_draws()
    kw = dict(seed=42, quick=True, last_draw_date=last, backend="numpy")
    a = sl.run_signal_lab(draws, **kw)
    b = sl.run_signal_lab(draws, **kw)
    assert a["battery"] == b["battery"] and a["predictor"] == b["predictor"]
    assert a["discriminator"] == b["discriminator"] and a["power_check"] == b["power_check"]


# ---------------- API ----------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    with TestClient(lotto.app) as c:
        yield c


def test_api_signal_lab_flow(client):
    lotto.state["signal_lab"] = None
    assert client.get("/signal-lab/result").status_code == 404
    st = client.get("/signal-lab/status").json()
    assert st["running"] is False and st["has_result"] is False and st["progress"] is None

    assert client.post("/signal-lab/run", json={"seed": 42, "quick": True}).status_code == 200
    assert client.post("/signal-lab/run", json={"quick": True}).status_code == 400   # already running

    t0 = time.time()
    while time.time() - t0 < 90:
        st = client.get("/signal-lab/status").json()
        if not st["running"]:
            break
        assert st["progress"] is None or {"stage", "done", "total"} <= set(st["progress"])
        time.sleep(0.5)
    assert st["running"] is False and st["has_result"] is True
    res = client.get("/signal-lab/result").json()
    assert res["n_draws"] == 722 and res["power_check"]["ok"] is True
    assert lotto.state["signal_lab"] is not None
