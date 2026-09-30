"""
Smart Pick v2 — popularity model and jackpot expected-value math.

HONESTY: every 7-number combination has exactly the same chance of being
drawn. Nothing here changes that. What *can* differ between tickets is how
many other players picked the same combination, and therefore how many ways
the jackpot is split if it hits.

The popularity weights below are ASSUMPTIONS (birthday bias, lucky numbers,
pattern play), not measured from real sales data. Everything derived from
them is an estimate and must be labelled as such in any UI or API response.
Drop a data/popularity_override.json ({"1": 1.7, "2": 1.5, ...}) to replace
the per-number weights with real statistics.
"""
import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from lotto_config import CURRENT_POOL, PICK, combinations

OVERRIDE_PATH = Path(__file__).parent.parent / "data" / "popularity_override.json"

LOW_MAX = 31                      # birthday range
LUCKY_NUMBERS = {3, 7, 11, 13}
LUCKY_MULT = 1.25
BAND_WEIGHTS = ((1, 12, 1.6), (13, 31, 1.25), (32, 10**6, 0.75))

# Configurable pattern multipliers (ticket-level popularity boosts)
PATTERN_MULTIPLIERS = {
    "triple_run": 3.0,     # 3+ consecutive numbers
    "arithmetic4": 4.0,    # 4+ numbers in an arithmetic progression
    "all_low": 2.5,        # all seven <= 31
    "past_winner": 5.0,    # exact past winning combination
}

EV_MAX_LOW = 4                    # guard: at most 4 numbers <= 31
E_UNIFORM_SAMPLES = 200_000
E_UNIFORM_SEED = 20260930
CANDIDATES = 20_000
BOTTOM_QUANTILE = 0.10
_RISK_LOW, _RISK_HIGH = 0.6, 1.2


# ------------------------------------------------------------------
# Per-number popularity weights
# ------------------------------------------------------------------
def default_weights(pool: int = CURRENT_POOL) -> np.ndarray:
    w = np.ones(pool + 1)
    for n in range(1, pool + 1):
        for lo, hi, val in BAND_WEIGHTS:
            if lo <= n <= hi:
                w[n] = val
                break
        if n in LUCKY_NUMBERS:
            w[n] *= LUCKY_MULT
    w[0] = 0.0
    return w


def load_weights(pool: int = CURRENT_POOL, override_path: Optional[Path] = None) -> np.ndarray:
    """Default weights, overridden per number by data/popularity_override.json if present."""
    w = default_weights(pool)
    path = Path(override_path) if override_path else OVERRIDE_PATH
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            for k, v in raw.items():
                n, v = int(k), float(v)
                if 1 <= n <= pool and math.isfinite(v) and v > 0:
                    w[n] = v
        except (ValueError, TypeError, OSError):
            pass  # unreadable override: fall back to defaults
    return w


# ------------------------------------------------------------------
# Vectorised combination scoring; combos are sorted (M, 7) int arrays
# ------------------------------------------------------------------
def has_triple_run(combos: np.ndarray) -> np.ndarray:
    d = np.diff(combos, axis=1)
    return ((d[:, :-1] == 1) & (d[:, 1:] == 1)).any(axis=1)


def has_arithmetic4(combos: np.ndarray, pool: int = CURRENT_POOL) -> np.ndarray:
    """True if 4+ of the numbers form an arithmetic progression (any equal gap)."""
    m = combos.shape[0]
    mask = np.zeros((m, pool + 1), dtype=bool)
    rows = np.arange(m)
    for j in range(PICK):
        mask[rows, combos[:, j]] = True
    found = np.zeros(m, dtype=bool)
    for i in range(PICK):
        for j in range(i + 1, PICK):
            a = combos[:, i].astype(np.int64)
            d = combos[:, j].astype(np.int64) - a
            v2, v3 = a + 2 * d, a + 3 * d
            ok = v3 <= pool
            v2c, v3c = np.where(ok, v2, 0), np.where(ok, v3, 0)
            found |= ok & mask[rows, v2c] & mask[rows, v3c]
    return found


def combo_scores(combos: np.ndarray, w: np.ndarray, pool: int = CURRENT_POOL,
                 mults: Optional[dict] = None) -> np.ndarray:
    """s(c) = prod w[n] * prod pattern multipliers (past-winner handled by callers)."""
    mults = mults or PATTERN_MULTIPLIERS
    s = w[combos].prod(axis=1)
    s = np.where(has_triple_run(combos), s * mults["triple_run"], s)
    s = np.where(has_arithmetic4(combos, pool), s * mults["arithmetic4"], s)
    s = np.where(combos[:, -1] <= LOW_MAX, s * mults["all_low"], s)
    return s


def random_combos(rng: np.random.Generator, m: int, pool: int = CURRENT_POOL) -> np.ndarray:
    out = np.empty((m, PICK), dtype=np.int16)
    step = 50_000
    for s in range(0, m, step):
        k = min(step, m - s)
        idx = np.argpartition(rng.random((k, pool)), PICK, axis=1)[:, :PICK]
        out[s:s + k] = np.sort(idx, axis=1) + 1
    return out


_E_CACHE: dict = {}


def e_uniform(w: np.ndarray, pool: int = CURRENT_POOL, mults: Optional[dict] = None) -> float:
    """E over uniform random tickets of s(c): Monte-Carlo once (fixed seed), cached."""
    mults = mults or PATTERN_MULTIPLIERS
    key = (pool, hashlib.sha1(np.round(w, 9).tobytes()).hexdigest(),
           tuple(sorted(mults.items())))
    if key not in _E_CACHE:
        rng = np.random.default_rng(E_UNIFORM_SEED)
        combos = random_combos(rng, E_UNIFORM_SAMPLES, pool)
        _E_CACHE[key] = float(combo_scores(combos, w, pool, mults).mean())
    return _E_CACHE[key]


def popularity_ratio(numbers, pool: int = CURRENT_POOL, historical_sets=None,
                     w: Optional[np.ndarray] = None, mults: Optional[dict] = None) -> float:
    """r(c) = s(c) / E_uniform[s]; 1.0 = as popular as an average random ticket (ESTIMATE)."""
    mults = mults or PATTERN_MULTIPLIERS
    w = load_weights(pool) if w is None else w
    combo = np.array([sorted(int(n) for n in numbers)], dtype=np.int16)
    s = float(combo_scores(combo, w, pool, mults)[0])
    if historical_sets and frozenset(int(n) for n in numbers) in historical_sets:
        s *= mults["past_winner"]
    return s / e_uniform(w, pool, mults)


def share_risk(ratio: float) -> str:
    if ratio < _RISK_LOW:
        return "low"
    return "medium" if ratio < _RISK_HIGH else "high"


# ------------------------------------------------------------------
# Smart Pick v2 sampling (randomised — never a deterministic argmin)
# ------------------------------------------------------------------
def _combo_keys(combos: np.ndarray) -> np.ndarray:
    keys = np.zeros(combos.shape[0], dtype=np.int64)
    for j in range(combos.shape[1]):
        keys = keys * 64 + combos[:, j].astype(np.int64)
    return keys


def guard_pass(combos: np.ndarray, historical_sets=None, max_low: int = EV_MAX_LOW) -> np.ndarray:
    ok = (combos <= LOW_MAX).sum(axis=1) <= max_low
    ok &= ~has_triple_run(combos)
    if historical_sets:
        hist = np.array([sorted(s) for s in historical_sets if len(s) == PICK], dtype=np.int16)
        if len(hist):
            ok &= ~np.isin(_combo_keys(combos), _combo_keys(hist))
    return ok


def candidate_pool(rng: np.random.Generator, pool: int = CURRENT_POOL, historical_sets=None,
                   w: Optional[np.ndarray] = None, n_candidates: int = CANDIDATES,
                   max_low: int = EV_MAX_LOW):
    """Random guard-passing combos and their popularity ratios."""
    w = load_weights(pool) if w is None else w
    kept, total = [], 0
    while total < n_candidates:
        raw = random_combos(rng, 100_000, pool)
        raw = raw[guard_pass(raw, historical_sets, max_low)]
        kept.append(raw)
        total += len(raw)
    combos = np.concatenate(kept)[:n_candidates]
    ratios = combo_scores(combos, w, pool) / e_uniform(w, pool)
    return combos, ratios


def smart_tickets(count: int = 1, pool: int = CURRENT_POOL, historical_sets=None,
                  seed: Optional[int] = None, max_overlap: int = 3,
                  quantile: float = BOTTOM_QUANTILE) -> list:
    """
    Draw `count` tickets, each uniformly at random from the lowest-popularity
    `quantile` of guard-passing candidates, pairwise overlapping in <= max_overlap numbers.
    """
    rng = np.random.default_rng(seed)
    w = load_weights(pool)
    combos, ratios = candidate_pool(rng, pool, historical_sets, w)
    order = np.argsort(ratios)
    n = len(order)
    chosen: list = []
    for _ in range(count):
        picked = None
        for q in (quantile, 0.25, 0.5, 1.0):
            idx = order[:max(1, int(n * q))]
            sub = combos[idx]
            ok = np.ones(len(idx), dtype=bool)
            for c in chosen:
                ov = (sub[:, :, None] == c[None, None, :]).any(axis=2).sum(axis=1)
                ok &= ov <= max_overlap
            if ok.any():
                picked = int(rng.choice(idx[ok]))
                break
        if picked is None:  # cannot happen with 20k candidates; stay safe
            picked = int(rng.choice(n))
        chosen.append(combos[picked])
        # never reuse the same candidate
        ratios[picked] = np.inf
        order = np.argsort(ratios)
    return [ticket_info(c.tolist(), pool, w=w, historical_sets=historical_sets) for c in chosen]


def ticket_info(numbers, pool: int = CURRENT_POOL, w=None, historical_sets=None) -> dict:
    nums = sorted(int(n) for n in numbers)
    r = popularity_ratio(nums, pool, historical_sets, w=w)
    return {
        "numbers": nums,
        "popularity_ratio": round(r, 4),
        "low_count": sum(1 for n in nums if n <= LOW_MAX),
        "sum": int(sum(nums)),
        "share_risk": share_risk(r),
    }


# ------------------------------------------------------------------
# Jackpot expected value (Poisson co-winner model)
# ------------------------------------------------------------------
def share_factor(lam: float) -> float:
    """E[1/(1+K)] for K ~ Poisson(lam) = (1 - e^-lam)/lam; -> 1 as lam -> 0."""
    if lam < 1e-12:
        return 1.0
    return -math.expm1(-lam) / lam


def _line(jackpot: float, tickets_sold: float, ratio: float, combos: int) -> dict:
    lam = tickets_sold * ratio / combos
    share = jackpot * share_factor(lam)
    return {"popularity_ratio": round(ratio, 4), "lambda": lam,
            "share_if_win": share, "ev_jackpot": share / combos}


def ev_estimate(jackpot: float, tickets_sold: float, ticket_price: float,
                lines_per_ticket: int, other_prizes_ev: float = 0.0,
                chosen_ratio: Optional[float] = None, pool: int = CURRENT_POOL) -> dict:
    c = combinations(pool)
    cost = ticket_price / lines_per_ticket
    rnd = _line(jackpot, tickets_sold, 1.0, c)
    out = {
        "combinations": c,
        "cost_per_line": round(cost, 4),
        "random": rnd,
        "chosen": None,
        "other_prizes_ev": other_prizes_ev,
        "net_ev_random": rnd["ev_jackpot"] + other_prizes_ev - cost,
        "net_ev_chosen": None,
        "assumptions": [
            "잭팟 캡·이월·하위 등급 고정 상금은 모델링하지 않음",
            "인기도 비율은 가정 기반 추정치 (실제 판매 데이터로 검증되지 않음)",
            "다른 구매자의 당첨 조합 수는 푸아송 분포로 가정",
            "모든 조합의 당첨 확률은 동일 (1/조합 수)",
        ],
    }
    if chosen_ratio is not None:
        ch = _line(jackpot, tickets_sold, chosen_ratio, c)
        out["chosen"] = ch
        out["net_ev_chosen"] = ch["ev_jackpot"] + other_prizes_ev - cost
    return out
