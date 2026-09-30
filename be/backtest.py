"""
Walk-forward backtest engine.

For each of the last `window` draws, every strategy is fed only the draws
that came before it, its top-`pick` numbers are compared against the
actual draw, and the match count is recorded. A random-ticket baseline
and the theoretical expectation pick^2/num_range are reported alongside,
so the app can show honestly whether any strategy beats chance.
"""
import math

import numpy as np


def _normalize(arr: np.ndarray) -> np.ndarray:
    valid = arr[1:]
    positive = valid[valid > 0]
    if len(positive) == 0:
        return arr
    result = np.zeros_like(arr)
    mn, mx = positive.min(), positive.max()
    if mx == mn:
        # All positive scores equal: map to 1.0 (mirror app.normalize)
        result[1:][valid > 0] = 1.0
        return result
    nz = arr > 0
    result[nz] = (arr[nz] - mn) / (mx - mn)
    result[0] = 0
    return result


def _top_pick(scores: np.ndarray, pick: int) -> set:
    ranked = sorted(range(1, len(scores)), key=lambda n: -scores[n])
    return set(ranked[:pick])


def _summary(match_list, expected) -> dict:
    """`expected` is a scalar, or one expected-match value per evaluated draw."""
    arr = np.array(match_list, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    exp_arr = np.broadcast_to(np.asarray(expected, dtype=float), arr.shape)
    diff = arr - exp_arr
    std = float(diff.std(ddof=1)) if n > 1 else 0.0
    if std > 0:
        t = float(diff.mean()) / (std / math.sqrt(n))
        p = math.erfc(abs(t) / math.sqrt(2))  # two-sided normal approximation
    else:
        t, p = 0.0, 1.0
    dist = {"0": 0, "1": 0, "2": 0, "3+": 0}
    for m in match_list:
        key = str(int(m)) if m < 3 else "3+"
        dist[key] += 1
    return {"mean": round(mean, 4), "std": round(std, 4),
            "expected": round(float(exp_arr.mean()), 4),
            "t": round(t, 3), "p": round(p, 4), "dist": dist}


def run_walk_forward(draws: list, num_range: int, pick: int, strategies: dict,
                     window: int = 150, n_random: int = 100, seed: int = 42,
                     selectors: dict = None, ensemble_weights: dict = None,
                     ensemble_selector=None, pools=None) -> dict:
    """
    draws: full ordered draw history (list of 7-number lists)
    strategies: {name: fn(history_draws) -> score array of len num_range+1}
    selectors: optional {name: fn(scores, pick) -> set} overriding the plain
        top-N pick, so a strategy's real selection logic (e.g. the EV guard)
        is what gets backtested
    ensemble_weights: optional {name: weight}; defaults to equal weight.
        Pass the production weights so the "ensemble" row measures the
        predictor the app actually ships (minus LSTM and jitter).
    ensemble_selector: optional fn(scores, pick) -> set for the ensemble row
    pools: optional per-draw number pool (49/50/52). Numbers above a draw's pool
        cannot match, so the chance expectation of a ticket is judged per draw as
        (ticket numbers inside that pool) * pick / pool instead of a single constant.
    Returns per-strategy summaries plus the ensemble and a random baseline.
    """
    total = len(draws)
    window = max(10, min(window, total - 30))
    start = total - window
    expected = pick * pick / num_range
    rng = np.random.default_rng(seed)
    selectors = selectors or {}

    matches = {name: [] for name in strategies}
    matches["ensemble"] = []
    expects = {name: [] for name in matches}
    random_means = []

    def record(name, ticket, t):
        actual = set(draws[t])
        matches[name].append(len(ticket & actual))
        if pools is not None:
            pool = pools[t]
            expects[name].append(sum(1 for n in ticket if n <= pool) * pick / pool)
        else:
            expects[name].append(expected)

    for t in range(start, total):
        history = draws[:t]
        actual = set(draws[t])
        raw = {}
        for name, fn in strategies.items():
            scores = fn(history)
            raw[name] = scores
            select = selectors.get(name, _top_pick)
            record(name, select(scores, pick), t)

        if ensemble_weights:
            ens = sum(_normalize(raw[name].copy()) * ensemble_weights.get(name, 0)
                      for name in strategies)
        else:
            ens = sum(_normalize(raw[name].copy()) for name in strategies) / len(strategies)
        ens_select = ensemble_selector or _top_pick
        record("ensemble", ens_select(ens, pick), t)

        hits = 0
        for _ in range(n_random):
            ticket = rng.choice(num_range, pick, replace=False) + 1
            hits += len(set(ticket.tolist()) & actual)
        random_means.append(hits / n_random)

    results = {name: _summary(vals, expects[name]) for name, vals in matches.items()}
    random_mean = float(np.mean(random_means))

    beats = [name for name, r in results.items()
             if r["t"] > 0 and r["p"] < 0.05]
    worse = [name for name, r in results.items()
             if r["t"] < 0 and r["p"] < 0.05]
    if beats:
        verdict = f"{', '.join(beats)} beat(s) random (p<0.05) over this window — verify on a fresh window before believing it."
    else:
        verdict = "No strategy beats a random ticket — consistent with a fair draw. Smart Pick's value is bigger payouts when you DO win, not more wins."
    if worse:
        verdict += f" Significantly WORSE than random: {', '.join(worse)}."

    return {
        "window": window,
        "evaluated": window,
        "expected_random": round(expected, 4),
        "random_baseline_mean": round(random_mean, 4),
        "results": results,
        "verdict": verdict,
    }
