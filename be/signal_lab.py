"""
Signal Lab — a falsifiable search for learnable structure in the draw history.

Design goals (why this cannot "discover" fake patterns):
  * statistical battery with Benjamini-Hochberg correction across ALL p-values
  * power check: plant known effects in simulated uniform data and confirm the
    detectors actually find them (otherwise the "no signal" verdict is worthless)
  * predictor judged by walk-forward log-loss against a constant 7/n baseline,
    with bootstrap CI and a sign-permutation test
  * discriminator (real next draw vs. uniform-random next draw), time-ordered
    split, AUC with bootstrap CI and label-permutation p-value

Everything runs on the fixed-pool era only (draws must be 1..n, n = 50) with
fixed seeds. Pure NumPy/SciPy; TensorFlow is used for the predictor only when
available and requested.
"""
import math
from datetime import datetime
from typing import Callable, Optional

import numpy as np
from scipy import stats

K = 7          # numbers per draw
N = 50         # pool of the analysed era
LAGS = 10      # predictor context: previous k draws
FREQ_WINDOW = 50
SIM_TICKETS = 20_000
ALPHA = 0.05

Progress = Optional[Callable[[str, int, int], None]]


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def multihot(draws, n: int = N) -> np.ndarray:
    m = np.zeros((len(draws), n), dtype=np.float64)
    for i, d in enumerate(draws):
        for x in d:
            m[i, x - 1] = 1.0
    return m


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    return out


def _two_sided_z(z: float) -> float:
    return float(math.erfc(abs(z) / math.sqrt(2)))


def hypergeom_stats(n: int = N, k: int = K):
    mean = k * k / n
    var = k * (k / n) * (1 - k / n) * ((n - k) / (n - 1))
    return mean, var


def sim_ticket_sums(rng: np.random.Generator, m: int = SIM_TICKETS, n: int = N) -> np.ndarray:
    idx = np.argpartition(rng.random((m, n)), K, axis=1)[:, :K] + 1
    return idx.sum(axis=1)


# ------------------------------------------------------------------
# A-1  statistical battery
# ------------------------------------------------------------------
def _merge_sparse(expected: np.ndarray, observed: np.ndarray, min_exp: float = 5.0):
    """Merge tail bins until every expected count >= min_exp."""
    e, o = list(expected), list(observed)
    while len(e) > 2 and e[0] < min_exp:
        e[1] += e[0]; o[1] += o[0]; e.pop(0); o.pop(0)
    while len(e) > 2 and e[-1] < min_exp:
        e[-2] += e[-1]; o[-2] += o[-1]; e.pop(); o.pop()
    return np.array(e), np.array(o)


def run_battery(draws, n: int = N, sim_sums: Optional[np.ndarray] = None,
                seed: int = 42) -> dict:
    m = multihot(draws, n)
    t = len(m)
    rng = np.random.default_rng(seed)
    raw = []  # (name, statistic, p)

    # 1. number uniformity
    counts = m.sum(axis=0)
    exp = t * K / n
    chi2 = float(((counts - exp) ** 2 / exp).sum())
    raw.append(("uniformity_chi2", chi2, float(stats.chi2.sf(chi2, n - 1))))

    # 2. overlap with the draw `lag` draws earlier (z-test vs hypergeometric)
    mean0, var0 = hypergeom_stats(n)
    for lag in range(1, 6):
        ov = (m[lag:] * m[:-lag]).sum(axis=1)
        z = (ov.mean() - mean0) / math.sqrt(var0 / len(ov))
        raw.append((f"overlap_lag{lag}_z", float(z), _two_sided_z(z)))

    # 3. pair co-occurrence
    co = m.T @ m
    iu = np.triu_indices(n, k=1)
    obs = co[iu]
    pexp = t * K * (K - 1) / (n * (n - 1))
    chi2p = float(((obs - pexp) ** 2 / pexp).sum())
    raw.append(("pair_chi2", chi2p, float(stats.chi2.sf(chi2p, len(obs) - 1))))

    # 4. odd-count distribution vs hypergeometric (sparse bins merged)
    n_odd = (n + 1) // 2
    odd_obs = np.bincount(m[:, 0::2].sum(axis=1).astype(int), minlength=K + 1)
    pmf = stats.hypergeom(n, n_odd, K).pmf(np.arange(K + 1))
    e_b, o_b = _merge_sparse(pmf * t, odd_obs)
    chi2o = float(((o_b - e_b) ** 2 / e_b).sum())
    raw.append(("odd_count_chi2", chi2o, float(stats.chi2.sf(chi2o, len(e_b) - 1))))

    # 5. draw-sum distribution vs simulated uniform tickets
    if sim_sums is None:
        sim_sums = sim_ticket_sums(rng, SIM_TICKETS, n)
    real_sums = np.array([sum(d) for d in draws])
    ks = stats.ks_2samp(real_sums, sim_sums)
    raw.append(("sum_ks", float(ks.statistic), float(ks.pvalue)))

    # 6. per-number appearance runs (Wald-Wolfowitz)
    runs = 1 + (m[1:] != m[:-1]).sum(axis=0)
    n1 = m.sum(axis=0)
    n0 = t - n1
    for j in range(n):
        if n1[j] in (0, t):
            raw.append((f"runs_n{j + 1:02d}", 0.0, 1.0))
            continue
        mu = 2 * n1[j] * n0[j] / t + 1
        var = (mu - 1) * (mu - 2) / (t - 1)
        z = (runs[j] - mu) / math.sqrt(var)
        raw.append((f"runs_n{j + 1:02d}", float(z), _two_sided_z(z)))

    # 7. per-number frequency (binomial z). Added beyond the six spec'd families:
    #    the omnibus chi-square is nearly blind to a bias on a single number.
    p0 = K / n
    sd = math.sqrt(t * p0 * (1 - p0))
    for j in range(n):
        z = (counts[j] - t * p0) / sd
        raw.append((f"freq_n{j + 1:02d}_z", float(z), _two_sided_z(z)))

    p = np.array([r[2] for r in raw])
    padj = bh_adjust(p)
    tests = [{"name": name, "statistic": round(stat, 4), "p_value": round(pv, 6),
              "p_adj_bh": round(float(pa), 6), "significant": bool(pa < ALPHA)}
             for (name, stat, pv), pa in zip(raw, padj)]
    return {
        "n_tests": len(tests),
        "n_significant_after_bh": int(sum(x["significant"] for x in tests)),
        "expected_false_positives": round(len(tests) * ALPHA, 2),
        "tests": tests,
    }


def _test(battery: dict, name: str) -> dict:
    return next(t for t in battery["tests"] if t["name"] == name)


# ------------------------------------------------------------------
# A-2  power check: planted effects in uniform simulated data
# ------------------------------------------------------------------
def simulate_uniform(t: int, rng: np.random.Generator, n: int = N) -> list:
    idx = np.argpartition(rng.random((t, n)), K, axis=1)[:, :K] + 1
    return [sorted(row.tolist()) for row in idx]


def simulate_number_boost(t: int, rng: np.random.Generator, number: int = 17,
                          factor: float = 1.35, n: int = N) -> list:
    """Number `number` appears with probability factor * 7/n; everything else uniform."""
    q = min(0.999, factor * K / n)
    r = rng.random((t, n))
    include = rng.random(t) < q
    r[include, number - 1] = -1.0
    r[~include, number - 1] = 2.0
    idx = np.argpartition(r, K, axis=1)[:, :K] + 1
    return [sorted(row.tolist()) for row in idx]


def carry_probability_for_overlap(target: float, n: int = N) -> float:
    """Per-number carry-over probability giving E[lag-1 overlap] = target (exact, bisection)."""
    binom = lambda r: stats.binom.pmf(np.arange(K + 1), K, r)

    def expected(r):
        c = np.arange(K + 1)
        return float((binom(r) * (c + (K - c) ** 2 / (n - c))).sum())

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if expected(mid) < target else (lo, mid)
    return (lo + hi) / 2


def simulate_lag1(t: int, rng: np.random.Generator, target_overlap: float = 1.30,
                  n: int = N) -> list:
    r_carry = carry_probability_for_overlap(target_overlap, n)
    draws = simulate_uniform(1, rng, n)
    for _ in range(t - 1):
        prev = np.array(draws[-1])
        carried = prev[rng.random(K) < r_carry]
        r = rng.random(n)
        r[carried - 1] = -1.0
        idx = np.argpartition(r, K)[:K] + 1
        draws.append(sorted(idx.tolist()))
    return draws


def run_power_check(seed: int = 42, n_draws: int = 708, n: int = N, n_rep: int = 40,
                    n_null: int = 100, progress: Progress = None,
                    with_predictor: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    sim_sums = sim_ticket_sums(rng, SIM_TICKETS, n)
    total = 3 * n_rep + n_null
    done = 0

    def tick():
        nonlocal done
        done += 1
        if progress:
            progress("power_check", done, total)

    # number_17_x1.35 is reported but does not gate `ok`: with ~700 draws and BH over
    # 100+ tests, a +35% bias on ONE number has only ~70% power (z ~ 3.8 vs a 3.5 cut).
    # The gating number effect is x1.5; the x1.35 detection rate is disclosed below.
    effects = {
        "number_17_x1.5": (lambda r: simulate_number_boost(n_draws, r, 17, 1.5, n), "freq_n17_z", True),
        "lag1_overlap_1.30": (lambda r: simulate_lag1(n_draws, r, 1.30, n), "overlap_lag1_z", True),
        "number_17_x1.35": (lambda r: simulate_number_boost(n_draws, r, 17, 1.35, n), "freq_n17_z", False),
    }
    planted, weak, first_data = [], None, {}
    for name, (make, target, gating) in effects.items():
        hits, padjs = 0, []
        for i in range(n_rep):
            data = make(rng)
            if i == 0:
                first_data[name] = data
            res = run_battery(data, n, sim_sums=sim_sums)
            row = _test(res, target)
            hits += int(row["significant"])
            padjs.append(row["p_adj_bh"])
            tick()
        rate = hits / n_rep
        entry = {"effect": name, "detected": rate >= 0.8, "detection_rate": round(rate, 3),
                 "p_adj_bh": float(np.median(padjs))}
        if gating:
            planted.append(entry)
        else:
            weak = entry

    fp = 0
    for _ in range(n_null):
        res = run_battery(simulate_uniform(n_draws, rng, n), n, sim_sums=sim_sums)
        fp += int(res["n_significant_after_bh"] > 0)
        tick()

    out = {
        "ok": all(p["detected"] for p in planted),
        "planted": planted,
        "weak_effect_info": weak,
        "replicates": n_rep,
        "false_positive_rate_null": round(fp / max(1, n_null), 4),
        "null_replicates": n_null,
        "criterion": "각 심은 신호의 검출률 ≥ 80% (BH 보정 후 유의)",
    }
    if with_predictor:  # can the walk-forward predictor see the planted lag effect?
        pr = run_predictor_test(first_data["lag1_overlap_1.30"], n=n, seed=seed,
                                backend="numpy", n_boot=500, n_perm=500)
        out["predictor_detects_lag1"] = bool(pr["beats_baseline"])
    return out


# ------------------------------------------------------------------
# A-3  predictor (walk-forward, per-number shared-weight model)
# ------------------------------------------------------------------
def build_features(m: np.ndarray, lags: int = LAGS, window: int = FREQ_WINDOW):
    """
    Features for every draw t >= window and every number j:
      lag-1..lag-`lags` appearance indicators, and the appearance frequency of j
      over the previous `window` draws (standardised).
    Returns X (S, n, lags+1), y (S, n), t_index (S,).
    """
    t_all, n = m.shape
    cs = np.vstack([np.zeros((1, n)), np.cumsum(m, axis=0)])
    ts = np.arange(window, t_all)
    lag_feats = np.stack([m[ts - l] for l in range(1, lags + 1)], axis=2)      # (S, n, lags)
    freq = (cs[ts] - cs[ts - window]) / window                                  # (S, n)
    p0 = K / n
    freq_z = (freq - p0) / math.sqrt(p0 * (1 - p0) / window)
    x = np.concatenate([lag_feats, freq_z[:, :, None]], axis=2)
    return x, m[ts], ts


def _fit_logreg(x: np.ndarray, y: np.ndarray, l2: float = 10.0, iters: int = 25) -> np.ndarray:
    a = np.hstack([np.ones((len(x), 1)), x])
    w = np.zeros(a.shape[1])
    ybar = min(max(y.mean(), 1e-6), 1 - 1e-6)
    w[0] = math.log(ybar / (1 - ybar))
    reg = np.full(a.shape[1], l2)
    reg[0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a @ w)))
        g = a.T @ (p - y) + reg * w
        h = (a * (p * (1 - p))[:, None]).T @ a + np.diag(reg)
        step = np.linalg.solve(h, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w


def _predict_logreg(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(np.hstack([np.ones((len(x), 1)), x]) @ w)))


def _tf_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False


def _fit_predict_mlp(xtr, ytr, xte, seed):
    """Small MLP (12 -> 16 -> 1, ~200 params, L2, early stopping), shared across numbers."""
    from tensorflow import keras
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        keras.layers.Input(shape=(xtr.shape[1],)),
        keras.layers.Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3)),
        keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=keras.regularizers.l2(1e-3)),
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="binary_crossentropy")
    model.fit(xtr, ytr, validation_split=0.15, epochs=30, batch_size=1024, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
    return model.predict(xte, verbose=0, batch_size=4096).ravel()


def _bce_per_draw(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(axis=1)


def bootstrap_ci(x: np.ndarray, rng: np.random.Generator, n_boot: int):
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def sign_permutation_p(x: np.ndarray, rng: np.random.Generator, n_perm: int) -> float:
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(x)))
    perm = (signs * x).mean(axis=1)
    return float((1 + np.sum(np.abs(perm) >= abs(x.mean()) - 1e-15)) / (n_perm + 1))


def run_predictor_test(draws, n: int = N, seed: int = 42, backend: str = "auto",
                       n_boot: int = 2000, n_perm: int = 2000, min_train: int = 100,
                       step: int = 50, progress: Progress = None) -> dict:
    """
    Expanding-window walk-forward: train on everything before the fold, test on
    the next `step` draws, retrain every fold. beats_baseline = CI upper < 0 of
    (model - constant 7/n baseline) per-draw BCE.
    """
    rng = np.random.default_rng(seed)
    m = multihot(draws, n)
    x, y, _ = build_features(m)
    s_total = len(x)
    use_tf = backend == "tf" or (backend == "auto" and _tf_available())
    model_name = "mlp" if use_tf else "logreg"

    starts = list(range(min_train, s_total, step))
    p_model = np.zeros((s_total, n))
    for fi, s0 in enumerate(starts):
        s1 = min(s0 + step, s_total)
        xtr, ytr = x[:s0].reshape(-1, x.shape[2]), y[:s0].reshape(-1)
        xte = x[s0:s1].reshape(-1, x.shape[2])
        pred = None
        if use_tf:
            try:
                pred = _fit_predict_mlp(xtr, ytr, xte, seed)
            except Exception:
                use_tf, model_name = False, "logreg"
        if pred is None:
            pred = _predict_logreg(_fit_logreg(xtr, ytr), xte)
        p_model[s0:s1] = pred.reshape(s1 - s0, n)
        if progress:
            progress("predictor", fi + 1, len(starts))

    sl = slice(min_train, s_total)
    yt = y[sl]
    loss_model = _bce_per_draw(p_model[sl], yt)
    loss_const = _bce_per_draw(np.full_like(yt, K / n), yt)

    cs = np.cumsum(m, axis=0)          # expanding empirical frequency up to (not incl.) each draw
    t_idx = np.arange(FREQ_WINDOW, len(m))[sl]
    emp = np.clip(cs[t_idx - 1] / t_idx[:, None], 1e-3, 1 - 1e-3)
    loss_freq = _bce_per_draw(emp, yt)

    delta = loss_model - loss_const
    ci = bootstrap_ci(delta, rng, n_boot)
    delta_f = loss_model - loss_freq
    return {
        "model": model_name,
        "folds": len(starts),
        "evaluated_draws": int(len(delta)),
        "mean_logloss": round(float(loss_model.mean()), 5),
        "baseline_logloss": round(float(loss_const.mean()), 5),
        "delta": round(float(delta.mean()), 5),
        "ci95": [round(ci[0], 5), round(ci[1], 5)],
        "p_value": round(sign_permutation_p(delta, rng, n_perm), 4),
        "beats_baseline": bool(ci[1] < 0),
        "freq_baseline_logloss": round(float(loss_freq.mean()), 5),
        "delta_vs_freq_baseline": round(float(delta_f.mean()), 5),
        "note": "beats_baseline = 상수 확률 7/n 대비 손실차 95% CI 상한 < 0",
    }


# ------------------------------------------------------------------
# A-4  discriminator: real (context -> next draw) vs random next draw
# ------------------------------------------------------------------
def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    ranks = stats.rankdata(scores)
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _pair_features(x: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """x: (S, n, lags+1) context features; cand: (S, n) candidate multi-hot."""
    n = cand.shape[1]
    overlaps = np.einsum("sn,snl->sl", cand, x[:, :, :LAGS])          # overlap with each lag
    freqsum = np.einsum("sn,sn->s", cand, x[:, :, LAGS])[:, None]
    nums = np.arange(1, n + 1)
    odd = (cand * (nums % 2)).sum(axis=1)[:, None]
    tot = (cand * nums).sum(axis=1)[:, None]
    return np.hstack([overlaps, freqsum, odd, tot])


def run_discriminator(draws, n: int = N, seed: int = 42, n_fake: int = 3,
                      n_boot: int = 1000, n_perm: int = 200) -> dict:
    rng = np.random.default_rng(seed)
    m = multihot(draws, n)
    x, y, _ = build_features(m)
    s = len(x)
    split = int(s * 0.7)

    def make_set(lo, hi):
        feats, labels = [_pair_features(x[lo:hi], y[lo:hi])], [np.ones(hi - lo)]
        for _ in range(n_fake):
            fake = np.zeros((hi - lo, n))
            idx = np.argpartition(rng.random((hi - lo, n)), K, axis=1)[:, :K]
            np.put_along_axis(fake, idx, 1.0, axis=1)
            feats.append(_pair_features(x[lo:hi], fake))
            labels.append(np.zeros(hi - lo))
        return np.vstack(feats), np.concatenate(labels)

    xtr, ytr = make_set(0, split)
    xte, yte = make_set(split, s)
    mu, sd = xtr.mean(axis=0), xtr.std(axis=0) + 1e-9
    w = _fit_logreg((xtr - mu) / sd, ytr, l2=1.0)
    scores = np.hstack([np.ones((len(xte), 1)), (xte - mu) / sd]) @ w
    auc = _auc(scores, yte)

    idx = rng.integers(0, len(yte), size=(n_boot, len(yte)))
    boots = []
    for row in idx:
        lab = yte[row]
        if 0 < lab.sum() < len(lab):
            boots.append(_auc(scores[row], lab))
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    perm_hits = 0
    for _ in range(n_perm):
        perm_hits += int(_auc(scores, rng.permutation(yte)) >= auc)
    return {
        "auc": round(auc, 4),
        "ci95": [round(ci[0], 4), round(ci[1], 4)],
        "p_value": round((1 + perm_hits) / (n_perm + 1), 4),
        "train_pairs": int(len(ytr)),
        "test_pairs": int(len(yte)),
    }


# ------------------------------------------------------------------
# Orchestration + verdict
# ------------------------------------------------------------------
def make_verdict(battery: dict, power: dict, predictor: dict, disc: dict) -> str:
    n_sig = battery["n_significant_after_bh"]
    pred_hit = predictor["beats_baseline"]
    disc_hit = disc["p_value"] < 0.01 and disc["ci95"][0] > 0.5
    detail = (f"검정 {battery['n_tests']}개 중 보정 후 유의 {n_sig}개 "
              f"(우연 기대 약 {battery['expected_false_positives']}개), "
              f"예측기 손실차 {predictor['delta']:+.5f} (95% CI {predictor['ci95'][0]:+.5f} ~ "
              f"{predictor['ci95'][1]:+.5f}), 판별기 AUC {disc['auc']:.3f}")
    if not power["ok"]:
        weak = [p["effect"] for p in power["planted"] if not p["detected"]]
        return (f"탐지기 신뢰 불가: 심은 신호 중 {', '.join(weak)}을(를) 안정적으로 검출하지 못했습니다. "
                f"이 상태의 '신호 없음' 결과는 증거로 쓸 수 없습니다. ({detail})")
    if n_sig > 0 or pred_hit or disc_hit:
        return (f"신호 후보가 발견되었습니다 ({detail}). 새 구간에서 재검증 필요 — "
                f"이 구간에서만 나타난 우연일 수 있습니다.")
    return (f"탐지력 검증 통과. 학습 가능한 신호가 발견되지 않았습니다 ({detail}). "
            f"딥러닝 모델은 상수 확률 7/{N}을 넘어서는 것을 학습하지 못했으므로 "
            f"당첨 확률을 바꿀 근거가 없습니다.")


def run_signal_lab(draws, seed: int = 42, quick: bool = False, backend: str = "auto",
                   progress: Progress = None, last_draw_date: Optional[str] = None,
                   n: int = N) -> dict:
    """Full Signal Lab run on fixed-pool draws (each draw: 7 ints in 1..n)."""
    if quick:
        cfg = dict(n_rep=40, n_null=100, boot=500, perm=500, d_boot=300, d_perm=100)
        backend = "numpy" if backend == "auto" else backend
    else:
        cfg = dict(n_rep=100, n_null=400, boot=2000, perm=2000, d_boot=1000, d_perm=200)

    def cb(stage, done, total):
        if progress:
            progress(stage, done, total)

    cb("battery", 0, 1)
    rng = np.random.default_rng(seed)
    battery = run_battery(draws, n, sim_sums=sim_ticket_sums(rng, SIM_TICKETS, n), seed=seed)
    cb("battery", 1, 1)

    power = run_power_check(seed, n_draws=len(draws), n=n, n_rep=cfg["n_rep"],
                            n_null=cfg["n_null"], progress=progress)
    predictor = run_predictor_test(draws, n, seed, backend, cfg["boot"], cfg["perm"],
                                   progress=progress)
    cb("discriminator", 0, 1)
    disc = run_discriminator(draws, n, seed, n_boot=cfg["d_boot"], n_perm=cfg["d_perm"])
    cb("discriminator", 1, 1)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_draws": len(draws),
        "last_draw_date": last_draw_date,
        "seed": seed,
        "quick": quick,
        "battery": battery,
        "power_check": power,
        "predictor": predictor,
        "discriminator": disc,
        "verdict": make_verdict(battery, power, predictor, disc),
    }
