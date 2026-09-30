"""
Typical-set ticket generator ("realistic" and "balanced" modes).

HONESTY FIRST: every 7-number combination has exactly the same chance of being
drawn (1 / C(52, 7)). Nothing in this module changes that. History cannot be
"learned" (docs/RESEARCH.md: every randomness test passes, no predictor beats the
constant baseline), so instead of predicting we *generate*:

  realistic  uniform random draw restricted to the "typical set" - combinations
             that look like a real draw: sum inside the central 90 % of its exact
             distribution, 2-5 odd numbers, numbers spread over >= 4 decade groups.
             Real history sits inside this set at the expected rate (tested in
             tests/test_generator.py), so the tickets are indistinguishable from
             real draws on those features. Win probability is unchanged.
  balanced   the same typical set, plus the EV guard (<= 4 numbers <= 31, no
             3-run, never a past winner), then a random pick from the least
             popular quarter (popularity weights are ASSUMPTIONS, see ev_model).

Both modes SAMPLE at random - never "top-k of a score" - so consecutive tickets
differ, and no score noise is ever amplified into a fake signal.
"""
from math import comb
from typing import Optional

import numpy as np

import ev_model
from lotto_config import CURRENT_POOL, PICK

TYPICAL_Q = (0.05, 0.95)        # central 90 % of the exact sum distribution
ODD_RANGE = (2, 5)              # allowed count of odd numbers (inclusive)
MIN_DECADES = 4                 # occupied 10-wide groups (1-10, 11-20, ...)
CANDIDATES = {"realistic": 2_000, "balanced": 20_000}
BALANCED_QUANTILE = 0.25        # pick from the least popular 25 % of candidates
MODES = ("realistic", "balanced")

_PMF_CACHE: dict = {}


# ------------------------------------------------------------------
# Exact null distribution of the sum
# ------------------------------------------------------------------
def sum_pmf(pool: int = CURRENT_POOL, pick: int = PICK) -> np.ndarray:
    """P(sum = s) for `pick` distinct numbers drawn uniformly from 1..pool (exact DP)."""
    key = (pool, pick)
    if key not in _PMF_CACHE:
        max_sum = pick * pool
        dp = np.zeros((pick + 1, max_sum + 1))
        dp[0, 0] = 1.0
        for n in range(1, pool + 1):
            for k in range(min(n, pick), 0, -1):
                dp[k, n:] += dp[k - 1, :max_sum + 1 - n]
        _PMF_CACHE[key] = dp[pick] / comb(pool, pick)
    return _PMF_CACHE[key]


def sum_stats(pool: int = CURRENT_POOL, pick: int = PICK) -> tuple:
    pmf = sum_pmf(pool, pick)
    s = np.arange(len(pmf))
    mean = float((s * pmf).sum())
    sd = float(np.sqrt(((s - mean) ** 2 * pmf).sum()))
    return mean, sd


def sum_bounds(pool: int = CURRENT_POOL, pick: int = PICK, q: tuple = TYPICAL_Q) -> tuple:
    """Inclusive (low, high) sum band holding at least the central `q[1] - q[0]` of the mass."""
    cdf = np.cumsum(sum_pmf(pool, pick))
    return int(np.searchsorted(cdf, q[0])), int(np.searchsorted(cdf, q[1]))


# ------------------------------------------------------------------
# Typical-set definition (vectorised; combos are sorted (M, 7) int arrays)
# ------------------------------------------------------------------
def decade_count(combos: np.ndarray, pool: int = CURRENT_POOL) -> np.ndarray:
    """How many 10-wide groups (1-10, 11-20, ...) contain at least one number."""
    groups = (combos.astype(np.int64) - 1) // 10
    out = np.zeros(len(combos), dtype=np.int64)
    for g in range((pool - 1) // 10 + 1):
        out += (groups == g).any(axis=1)
    return out


def typical_mask(combos: np.ndarray, pool: int = CURRENT_POOL) -> np.ndarray:
    lo, hi = sum_bounds(pool, combos.shape[1])
    s = combos.astype(np.int64).sum(axis=1)
    odd = (combos % 2).sum(axis=1)
    return ((s >= lo) & (s <= hi)
            & (odd >= ODD_RANGE[0]) & (odd <= ODD_RANGE[1])
            & (decade_count(combos, pool) >= MIN_DECADES))


def describe(numbers, pool: int = CURRENT_POOL) -> dict:
    """Typicality report for one ticket (shown in the UI; also useful for legacy modes)."""
    nums = sorted(int(n) for n in numbers)
    arr = np.array([nums], dtype=np.int16)
    lo, hi = sum_bounds(pool, len(nums))
    mean, _ = sum_stats(pool, len(nums))
    return {
        "sum": int(sum(nums)),
        "sum_band": [lo, hi],
        "sum_mean": round(mean, 1),
        "odd_count": int((arr % 2).sum()),
        "decades": int(decade_count(arr, pool)[0]),
        "typical": bool(typical_mask(arr, pool)[0]),
    }


# ------------------------------------------------------------------
# Sampling
# ------------------------------------------------------------------
def typical_candidates(rng: np.random.Generator, n: int, pool: int = CURRENT_POOL,
                       historical_sets=None, ev_guard: bool = False,
                       max_low: int = ev_model.EV_MAX_LOW) -> np.ndarray:
    """`n` uniformly random combinations from the typical set (optionally also EV-guard-passing)."""
    batch = int(min(100_000, max(4_000, n * 4)))
    kept, total = [], 0
    for _ in range(500):
        raw = ev_model.random_combos(rng, batch, pool)
        ok = typical_mask(raw, pool)
        if ev_guard:
            ok &= ev_model.guard_pass(raw, historical_sets, max_low)
        raw = raw[ok]
        kept.append(raw)
        total += len(raw)
        if total >= n:
            break
    if total == 0:
        raise RuntimeError("no combination passes the typical-set filter for this pool")
    return np.concatenate(kept)[:n]


def ticket_info(numbers, pool: int = CURRENT_POOL, w=None, historical_sets=None) -> dict:
    nums = sorted(int(n) for n in numbers)
    r = ev_model.popularity_ratio(nums, pool, historical_sets, w=w)
    info = describe(nums, pool)
    info.update({
        "numbers": nums,
        "popularity_ratio": round(r, 4),
        "low_count": sum(1 for n in nums if n <= ev_model.LOW_MAX),
        "share_risk": ev_model.share_risk(r),
    })
    return info


def sample_tickets(count: int = 1, pool: int = CURRENT_POOL, historical_sets=None,
                   mode: str = "realistic", seed: Optional[int] = None,
                   max_overlap: int = 3, quantile: float = BALANCED_QUANTILE,
                   n_candidates: Optional[int] = None) -> list:
    """
    Draw `count` tickets at random from the typical set. `balanced` additionally applies the
    EV guard and chooses from the least popular `quantile` of the candidates. Tickets of one
    call overlap pairwise in at most `max_overlap` numbers.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    rng = np.random.default_rng(seed)
    balanced = mode == "balanced"
    w = ev_model.load_weights(pool)
    combos = typical_candidates(rng, n_candidates or CANDIDATES[mode], pool, historical_sets,
                                ev_guard=balanced)
    if balanced:
        ratios = ev_model.combo_scores(combos, w, pool) / ev_model.e_uniform(w, pool)
        order = np.argsort(ratios)
        tiers = (quantile, 0.5, 1.0)
    else:
        order = np.arange(len(combos))
        tiers = (1.0,)

    used = np.zeros(len(combos), dtype=bool)
    chosen: list = []
    for _ in range(count):
        picked = None
        for q in tiers:
            idx = order[:max(1, int(len(order) * q))]
            idx = idx[~used[idx]]
            if len(idx) == 0:
                continue
            sub = combos[idx]
            ok = np.ones(len(idx), dtype=bool)
            for c in chosen:
                ov = (sub[:, :, None] == c[None, None, :]).any(axis=2).sum(axis=1)
                ok &= ov <= max_overlap
            if ok.any():
                picked = int(rng.choice(idx[ok]))
                break
        if picked is None:  # cannot happen with thousands of candidates; stay safe
            picked = int(rng.choice(np.flatnonzero(~used)))
        used[picked] = True
        chosen.append(combos[picked])
    return [ticket_info(c.tolist(), pool, w=w, historical_sets=historical_sets) for c in chosen]
