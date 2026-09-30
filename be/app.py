"""
LottoMax AI - 7-Strategy Ensemble Prediction Engine
====================================================
Backend: FastAPI + TensorFlow LSTM + Statistical Analysis

Strategies:
1. LSTM Sequential Pattern - Deep learning on draw sequences
2. Frequency + Recency - Hot/cold with exponential decay
3. Gap Analysis - Overdue numbers based on gap distributions
4. Pair Correlation - Co-occurrence patterns
5. Distribution Balance - Range & odd/even equilibrium
6. Seed/RNG Analysis - Time-based PRNG reverse engineering
   (kept for transparency; backtests show no predictive power, default weight 0)
7. Smart Pick (EV) - Expected-value optimization: avoid popular
   combinations so a winning ticket shares the pari-mutuel prize
   with fewer people. Does not change win probability.

Honesty note (docs/RESEARCH.md): the draw history passes every
randomness test, and no strategy beats the 7*7/50 = 0.98 expected
matches of a random ticket. The only real lever is EV optimization.
"""

import os
import json
import math
import time
import struct
import logging
import threading
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime, timezone, timedelta
from typing import Optional, Literal
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.regularizers import l2 as l2_reg
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    keras = None
    TF_AVAILABLE = False

from backtest import run_walk_forward
import data_ingest
import ev_model
import signal_lab
from lotto_config import (
    ERA2_START_DATE, ERA3_START_DATE, ERA2_POOL, CURRENT_POOL, TICKET_PRICE,
    LINES_PER_TICKET, pool_for_date, combinations,
)

# ============================================================
# Config
# ============================================================
DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

LOTTO_MAX = CURRENT_POOL  # 52 since 2026-04-14 (was 50 in 2019-05..2026-04)
LOTTO_PICK = 7
SEQUENCE_LENGTH = 20  # How many past draws the LSTM looks at
HOT_COLD_WINDOW = 50
RECENT_WINDOW = 30

# 2019-05-14: LottoMax switched from 7/49 weekly to 7/50 twice a week.
# Statistics computed across that boundary are biased (e.g. number 50
# only exists after it), so all strategies use era-2+ draws only.
# 2026-04-14: pool grew again (7/52). Numbers 51 and 52 have only ~50 draws
# of history, so statistical strategies treat them as neutral (see
# neutralize_new_numbers). ERA2_START_DATE/ERA3_START_DATE live in lotto_config.

# Overpicked "lucky" numbers (players' favourites — bad for EV)
LUCKY_NUMBERS = {3, 7, 11, 13}
# Numbers <= 31 are birthday-biased; cap how many the final pick may contain
EV_MAX_LOW_NUMBERS = 4

DEFAULT_WEIGHTS = {
    "lstm": 0.15, "frequency": 0.15, "gap": 0.20, "pair": 0.05,
    "distribution": 0.15, "seed": 0.00, "smart": 0.30,
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lottomax-ai")

# ============================================================
# App
# ============================================================
DEFAULT_CORS = "http://localhost:5173,http://127.0.0.1:5173"


def cors_origins() -> list:
    raw = os.environ.get("LOTTOMAX_CORS_ORIGINS", DEFAULT_CORS)
    return [o.strip() for o in raw.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(_app):
    startup()
    yield


app = FastAPI(title="LottoMax AI", version="5.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
state = {
    "main_draws": [],        # era-2 draws only (7/50) — used by all strategies
    "all_draws": [],         # full history incl. 7/49 era, for reference
    "main_draws_dated": [],  # era-2 draws with dates
    "historical_sets": set(),  # frozensets of every past winning combo (full history)
    "main_model": None,
    "is_training": False,
    "training_progress": {"status": "idle", "epoch": 0, "total_epochs": 0, "loss": 0, "strategy": ""},
    "training_log": [],
    "last_trained": None,
    "seed_analysis": None,
    "backtest": None,
    "main_draw_pools": [],   # number pool (49/50/52) in force for each era-2+ draw
    "history_meta": {},      # frozenset(numbers) -> {"draw_number", "date"} (full history)
    "history_list": [],      # [{"draw_number","date","numbers"}] full history, ordered
    "lstm_verdict": None,
    "signal_lab": None,
    "signal_lab_progress": None,
    "signal_lab_running": False,
}


# ============================================================
# Data Loading
# ============================================================
def load_csv_data():
    """Load LottoMax CSV file, splitting at the 7/49 -> 7/50 era boundary."""
    main_path = DATA_DIR / "LOTTOMAX.csv"

    if not main_path.exists():
        logger.warning(f"Main CSV not found at {main_path}")
        return

    df = pd.read_csv(main_path)
    # Only main draws (SEQUENCE NUMBER == 0)
    df_main = df[df["SEQUENCE NUMBER"] == 0].sort_values("DRAW NUMBER")
    era2_ts = pd.Timestamp(ERA2_START_DATE)
    all_draws, era_draws, era_dated, era_pools = [], [], [], []
    history_meta, history_list = {}, []
    bad_dates = 0
    for _, row in df_main.iterrows():
        nums = [int(row[f"NUMBER DRAWN {i}"]) for i in range(1, 8)]
        date_str = str(row.get("DRAW DATE", "")).strip().strip('"')
        all_draws.append(nums)
        info = {"draw_number": int(row["DRAW NUMBER"]), "date": date_str}
        history_meta[frozenset(nums)] = info
        history_list.append({**info, "numbers": nums})
        parsed = pd.to_datetime(date_str, format="%Y-%m-%d", errors="coerce")
        if pd.isna(parsed):
            bad_dates += 1
            continue  # unparseable date: keep out of era-2 statistics
        if parsed >= era2_ts:
            era_draws.append(nums)
            era_dated.append({"numbers": nums, "date": date_str})
            era_pools.append(pool_for_date(date_str))
    if bad_dates:
        logger.warning(f"{bad_dates} draws had unparseable dates and were excluded from era-2 stats")

    state["all_draws"] = all_draws
    state["main_draws"] = era_draws
    state["main_draws_dated"] = era_dated
    state["historical_sets"] = {frozenset(d) for d in all_draws}
    state["main_draw_pools"] = era_pools
    state["history_meta"] = history_meta
    state["history_list"] = history_list
    logger.info(
        f"Loaded {len(all_draws)} draws total; using {len(era_draws)} "
        f"era-2+ (7/50 since {ERA2_START_DATE}, 7/52 since {ERA3_START_DATE}) draws for statistics"
    )


def signal_lab_draws():
    """Draws for Signal Lab: fixed 7/50 era only (Era 2), so 1..50 assumptions hold."""
    pairs = [(d["numbers"], d["date"]) for d, p in zip(state["main_draws_dated"], state["main_draw_pools"])
             if p == ERA2_POOL]
    draws = [p[0] for p in pairs]
    return draws, (pairs[-1][1] if pairs else None)


def load_csv_data_with_dates():
    if not state["main_draws_dated"]:
        load_csv_data()
    return state["main_draws_dated"]


# ============================================================
# STRATEGY 1: LSTM Neural Network
# ============================================================
def prepare_lstm_data(draws: list, num_range: int, seq_len: int = SEQUENCE_LENGTH):
    """
    Convert draws into multi-hot encoded sequences for LSTM training.

    Each draw becomes a binary vector of size num_range.
    Input: seq_len consecutive draws → Output: next draw
    """
    # Multi-hot encode each draw
    encoded = []
    for draw in draws:
        vec = np.zeros(num_range, dtype=np.float32)
        for n in draw:
            if 1 <= n <= num_range:
                vec[n - 1] = 1.0
        encoded.append(vec)

    encoded = np.array(encoded)

    # Create sequences
    X, y = [], []
    for i in range(len(encoded) - seq_len):
        X.append(encoded[i:i + seq_len])
        y.append(encoded[i + seq_len])

    return np.array(X), np.array(y)


LSTM_MAX_PARAMS = 50_000


def build_lstm_model(num_range: int, seq_len: int = SEQUENCE_LENGTH):
    """
    Small LSTM (~14k parameters for 52 numbers).

    ~750 training samples cannot support a deep recurrent net (the previous
    3-layer bidirectional stack had >500k parameters and memorised noise), so
    this keeps one LSTM layer with L2 + dropout. Output: an independent
    probability per number.
    """
    model = Sequential([
        LSTM(32, input_shape=(seq_len, num_range),
             kernel_regularizer=l2_reg(1e-4), recurrent_regularizer=l2_reg(1e-4)),
        Dropout(0.3),
        Dense(32, activation="relu", kernel_regularizer=l2_reg(1e-4)),
        Dropout(0.2),
        Dense(num_range, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.build(input_shape=(None, seq_len, num_range))
    assert model.count_params() < LSTM_MAX_PARAMS, model.count_params()

    return model


LSTM_VERDICT_MARGIN = 0.0005


def constant_baseline_loss(p: float) -> float:
    """BCE of always predicting the constant rate p: -(p ln p + (1-p) ln(1-p))."""
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return float(-(p * math.log(p) + (1 - p) * math.log(1 - p)))


def make_lstm_verdict(val_loss: float, train_rate: float) -> dict:
    baseline = constant_baseline_loss(train_rate)
    return {
        "val_loss": round(float(val_loss), 5),
        "baseline_loss": round(baseline, 5),
        "beats_constant_baseline": bool(val_loss < baseline - LSTM_VERDICT_MARGIN),
    }


def train_lstm(draws: list, num_range: int, model_name: str, epochs: int = 100):
    """Train LSTM model on draw history."""
    if not TF_AVAILABLE:
        log_msg("⚠️ TensorFlow not installed — skipping LSTM, statistical strategies only")
        return None

    log_msg(f"🧠 LSTM: Preparing {len(draws)} draws for training...")

    X, y = prepare_lstm_data(draws, num_range)
    if len(X) == 0:
        log_msg("❌ Not enough data for LSTM training")
        return None

    log_msg(f"  Data shape: X={X.shape}, y={y.shape}")

    model = build_lstm_model(num_range)
    log_msg(f"  Model params: {model.count_params():,}")

    # Split: last 10% for validation
    split = max(1, int(len(X) * 0.9))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
        TrainingProgressCallback(model_name, epochs),
    ]

    log_msg(f"  Training for up to {epochs} epochs...")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )

    best_val_loss = min(history.history["val_loss"])
    final_epoch = len(history.history["loss"])
    log_msg(f"  ✅ Training complete: {final_epoch} epochs, val_loss={best_val_loss:.4f}")

    # Honesty check: does the model beat "always predict the base rate"?
    val_loss = float(model.evaluate(X_val, y_val, verbose=0)[0])
    verdict = make_lstm_verdict(val_loss, float(y_train.mean()))
    state["lstm_verdict"] = verdict
    try:  # persist next to the model so the warning survives a restart
        (MODEL_DIR / f"{model_name}.verdict.json").write_text(json.dumps(verdict))
    except OSError:
        pass
    if verdict["beats_constant_baseline"]:
        log_msg(f"  📈 Validation BCE {verdict['val_loss']} beats constant baseline {verdict['baseline_loss']}")
    else:
        log_msg(f"  ⚠️ Validation BCE {verdict['val_loss']} does NOT beat constant baseline "
                f"{verdict['baseline_loss']} — no learnable signal (expected for random draws)")

    # Save model
    model_path = MODEL_DIR / f"{model_name}.keras"
    model.save(model_path)
    log_msg(f"  💾 Model saved: {model_path}")

    return model


if TF_AVAILABLE:
    class TrainingProgressCallback(keras.callbacks.Callback):
        def __init__(self, model_name, total_epochs):
            self.model_name = model_name
            self.total_epochs = total_epochs

        def on_epoch_end(self, epoch, logs=None):
            state["training_progress"] = {
                "status": "training",
                "strategy": f"LSTM ({self.model_name})",
                "epoch": epoch + 1,
                "total_epochs": self.total_epochs,
                "loss": round(logs.get("loss", 0), 4),
                "val_loss": round(logs.get("val_loss", 0), 4),
            }


def lstm_predict(model, draws: list, num_range: int) -> np.ndarray:
    """Get LSTM probability scores for each number."""
    if model is None or len(draws) < SEQUENCE_LENGTH:
        return np.zeros(num_range + 1)

    # Prepare last sequence
    encoded = []
    for draw in draws[-SEQUENCE_LENGTH:]:
        vec = np.zeros(num_range, dtype=np.float32)
        for n in draw:
            if 1 <= n <= num_range:
                vec[n - 1] = 1.0
        encoded.append(vec)

    X = np.array([encoded])
    probs = model.predict(X, verbose=0)[0]

    # Convert to 1-indexed scores
    scores = np.zeros(num_range + 1)
    scores[1:] = probs
    return scores


# ============================================================
# STRATEGY 2: Frequency + Recency (Hot/Cold with exponential decay)
# ============================================================
def strategy_frequency_recency(draws: list, num_range: int) -> np.ndarray:
    scores = np.zeros(num_range + 1)
    if len(draws) == 0:
        return scores

    recent = draws[-HOT_COLD_WINDOW:]
    for i, draw in enumerate(recent):
        recency = (i + 1) / len(recent)
        weight = np.exp(recency * 2 - 2)
        for n in draw:
            scores[n] += weight

    # "Due" bonus
    last_seen = np.full(num_range + 1, -1)
    for i, draw in enumerate(draws):
        for n in draw:
            last_seen[n] = i

    total = len(draws)
    for n in range(1, num_range + 1):
        if last_seen[n] >= 0:
            gap = total - last_seen[n]
            if gap > RECENT_WINDOW:
                scores[n] += np.log(gap / RECENT_WINDOW) * 0.3

    return scores


# ============================================================
# STRATEGY 3: Gap Analysis
# ============================================================
def strategy_gap_analysis(draws: list, num_range: int) -> np.ndarray:
    scores = np.zeros(num_range + 1)
    if len(draws) < 10:
        return scores

    for n in range(1, num_range + 1):
        gaps = []
        last_idx = -1
        for i, draw in enumerate(draws):
            if n in draw:
                if last_idx >= 0:
                    gaps.append(i - last_idx)
                last_idx = i

        if len(gaps) < 2:
            continue

        avg_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        current_gap = len(draws) - last_idx if last_idx >= 0 else 999

        if std_gap > 0:
            z = (current_gap - avg_gap) / std_gap
            if z > -0.5:
                scores[n] = 1 / (1 + np.exp(-z))
        elif current_gap >= avg_gap:
            scores[n] = 0.7

    return scores


# ============================================================
# STRATEGY 4: Pair Correlation
# ============================================================
def strategy_pair_analysis(draws: list, num_range: int) -> np.ndarray:
    scores = np.zeros(num_range + 1)
    recent = draws[-100:]
    if len(recent) == 0:
        return scores

    pair_count = {}
    for draw in recent:
        for i in range(len(draw)):
            for j in range(i + 1, len(draw)):
                key = (min(draw[i], draw[j]), max(draw[i], draw[j]))
                pair_count[key] = pair_count.get(key, 0) + 1

    top_pairs = sorted(pair_count.items(), key=lambda x: -x[1])[:50]
    last_draw = set(draws[-1]) if draws else set()

    for (a, b), count in top_pairs:
        weight = count / len(recent)
        boost = 2.0 if (a in last_draw or b in last_draw) else 1.0
        scores[a] += weight * boost
        scores[b] += weight * boost

    return scores


# ============================================================
# STRATEGY 5: Distribution Balance
# ============================================================
def strategy_distribution(draws: list, num_range: int) -> np.ndarray:
    scores = np.zeros(num_range + 1)
    recent = draws[-RECENT_WINDOW:]
    if len(recent) == 0:
        return scores

    range_size = 10 if num_range <= 50 else 20
    num_ranges = (num_range + range_size - 1) // range_size
    range_counts = np.zeros(num_ranges)
    odd_count = 0
    even_count = 0

    for draw in recent:
        for n in draw:
            idx = (n - 1) // range_size
            if idx < num_ranges:
                range_counts[idx] += 1
            if n % 2 == 0:
                even_count += 1
            else:
                odd_count += 1

    total_nums = sum(len(d) for d in recent)
    expected = total_nums / num_ranges

    for n in range(1, num_range + 1):
        idx = (n - 1) // range_size
        if idx < num_ranges:
            ratio = range_counts[idx] / expected if expected > 0 else 1
            if ratio < 1:
                scores[n] += (1 - ratio) * 0.5

        oe_total = odd_count + even_count
        if oe_total > 0:
            if n % 2 == 1 and odd_count / oe_total < 0.45:
                scores[n] += 0.2
            elif n % 2 == 0 and even_count / oe_total < 0.45:
                scores[n] += 0.2

    return scores


# ============================================================
# STRATEGY 6: Seed/RNG Analysis
# ============================================================

class MersenneTwister:
    @staticmethod
    def generate_draw(seed, num_range, pick_count):
        import random
        rng = random.Random(seed)
        pool = list(range(1, num_range + 1))
        rng.shuffle(pool)
        return sorted(pool[:pick_count])

class LCG:
    def __init__(self, seed, a=1103515245, c=12345, m=2**31):
        self.state = seed
        self.a, self.c, self.m = a, c, m
    def next(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state
    def generate_draw(self, num_range, pick_count):
        nums = set()
        while len(nums) < pick_count:
            nums.add((self.next() % num_range) + 1)
        return sorted(nums)

class XorShift:
    def __init__(self, seed):
        self.x = seed & 0xFFFFFFFF or 1
        self.y = ((seed >> 32) & 0xFFFFFFFF) or 362436069
        self.z, self.w = 521288629, 88675123
    def next(self):
        t = self.x ^ ((self.x << 11) & 0xFFFFFFFF)
        self.x, self.y, self.z = self.y, self.z, self.w
        self.w = (self.w ^ (self.w >> 19)) ^ (t ^ (t >> 8))
        self.w &= 0xFFFFFFFF
        return self.w
    def generate_draw(self, num_range, pick_count):
        nums = set()
        while len(nums) < pick_count:
            nums.add((self.next() % num_range) + 1)
        return sorted(nums)


def estimate_draw_timestamp(draw_date_str):
    try:
        draw_date = datetime.strptime(draw_date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return []
    et_offset = timedelta(hours=-5)
    base_time = draw_date.replace(hour=22, minute=0, second=0, tzinfo=timezone.utc) - et_offset
    base_ts = int(base_time.timestamp())
    return [base_ts + offset for offset in range(0, 3600)]


def run_seed_analysis(draws_with_dates, num_range, pick_count, max_draws=20):
    results = {
        "tested_draws": 0, "tested_seeds": 0,
        "best_matches": [], "partial_matches": [],
        "algo_scores": {"mt": 0, "lcg": 0, "xorshift": 0},
    }
    recent = draws_with_dates[-max_draws:]
    algos = ["mt", "lcg", "xorshift"]

    for draw_info in recent:
        draw_nums = draw_info["numbers"]
        timestamps = estimate_draw_timestamp(draw_info["date"])
        if not timestamps:
            continue
        results["tested_draws"] += 1

        for ts in timestamps:
            results["tested_seeds"] += 1
            for algo in algos:
                for seed_variant in [ts, ts * 1000, ts ^ 0x5DEECE66D]:
                    if algo == "mt":
                        predicted = MersenneTwister.generate_draw(seed_variant, num_range, pick_count)
                    elif algo == "lcg":
                        predicted = LCG(seed_variant).generate_draw(num_range, pick_count)
                    else:
                        predicted = XorShift(seed_variant).generate_draw(num_range, pick_count)

                    match_count = len(set(predicted) & set(draw_nums))
                    entry = {"match": match_count, "predicted": predicted, "seed": seed_variant,
                             "algo": algo, "date": draw_info["date"], "actual": draw_nums, "timestamp": ts}

                    if match_count == pick_count:
                        results["best_matches"].append(entry)
                        results["algo_scores"][algo] += 100
                    elif match_count >= 4:
                        results["partial_matches"].append(entry)
                        results["algo_scores"][algo] += match_count * 5

    results["partial_matches"] = sorted(results["partial_matches"], key=lambda x: -x["match"])[:20]
    return results


def strategy_seed_analysis(draws_with_dates, num_range, pick_count):
    scores = np.zeros(num_range + 1)
    if len(draws_with_dates) < 5:
        return scores

    last_date = draws_with_dates[-1]["date"]
    try:
        last_dt = datetime.strptime(last_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        return scores

    days_ahead = {0:1, 1:3, 2:2, 3:1, 4:4, 5:3, 6:2}
    next_draw = last_dt + timedelta(days=days_ahead.get(last_dt.weekday(), 1))
    next_timestamps = estimate_draw_timestamp(next_draw.strftime("%Y-%m-%d"))
    if not next_timestamps:
        return scores

    num_counts = Counter()
    algos_to_weight = {"mt": 1.0, "lcg": 0.5, "xorshift": 0.5}

    analysis = run_seed_analysis(draws_with_dates[-10:], num_range, pick_count, 10)
    total_algo_score = sum(analysis["algo_scores"].values()) + 1
    for algo in algos_to_weight:
        algos_to_weight[algo] = (analysis["algo_scores"][algo] / total_algo_score) + 0.1

    for ts in next_timestamps[::10]:
        for algo, weight in algos_to_weight.items():
            for seed_variant in [ts, ts * 1000]:
                try:
                    if algo == "mt":
                        predicted = MersenneTwister.generate_draw(seed_variant, num_range, pick_count)
                    elif algo == "lcg":
                        predicted = LCG(seed_variant).generate_draw(num_range, pick_count)
                    else:
                        predicted = XorShift(seed_variant).generate_draw(num_range, pick_count)
                    for n in predicted:
                        num_counts[n] += weight
                except Exception:
                    pass

    if num_counts:
        max_count = max(num_counts.values())
        for n, count in num_counts.items():
            if 1 <= n <= num_range:
                scores[n] = count / max_count
    return scores


# ============================================================
# STRATEGY 7: Smart Pick (Expected-Value optimization)
# ============================================================
def strategy_smart_pick(num_range: int) -> np.ndarray:
    """
    Score numbers by how UNPOPULAR they are with human players.

    Picking unpopular numbers does not change the odds of winning, but
    LottoMax jackpots and MaxMillions are pari-mutuel: winners split the
    pool. Fewer co-winners => bigger expected payout for the same odds.

    Popularity facts (docs/RESEARCH.md §5):
    - 1-31 are birthday-biased, 1-12 doubly so (month AND day)
    - 3, 7, 11, 13 are classic "lucky" favourites
    """
    scores = np.zeros(num_range + 1)
    for n in range(1, num_range + 1):
        if n >= 32:
            s = 1.0
        elif n >= 13:
            s = 0.45
        else:
            s = 0.30
        if n in LUCKY_NUMBERS:
            s -= 0.15
        scores[n] = s
    return scores


def _has_triple_run(nums) -> bool:
    """True if nums contain 3+ consecutive integers (popular pattern play)."""
    s = sorted(nums)
    return any(s[i + 1] == s[i] + 1 and s[i + 2] == s[i] + 2 for i in range(len(s) - 2))


def apply_ev_guard(ranked: list, pick_count: int, historical_sets=None,
                   max_low: int = EV_MAX_LOW_NUMBERS) -> list:
    """
    Walk the score ranking greedily, skipping candidates that would make
    the ticket popular (and thus likely to share the prize):
      1. at most `max_low` numbers <= 31 (birthday tickets)
      2. no 3+ consecutive run (sequence players)
      3. never an exact past winning combo (people replay those)
    Every 7-number combo is equally likely to be drawn, so this is
    EV-neutral on hit probability and only reduces expected sharing.
    """
    pick = []
    for n in ranked:
        if len(pick) == pick_count:
            break
        cand = pick + [n]
        if sum(1 for x in cand if x <= 31) > max_low:
            continue
        if _has_triple_run(cand):
            continue
        pick.append(n)

    # Fallback (constraints unsatisfiable with this pool): relax the
    # low-number cap first, and only as a last resort allow runs too
    if len(pick) < pick_count:
        for n in ranked:
            if len(pick) == pick_count:
                break
            if n in pick or _has_triple_run(pick + [n]):
                continue
            pick.append(n)
    for n in ranked:
        if len(pick) == pick_count:
            break
        if n not in pick:
            pick.append(n)

    if historical_sets and frozenset(pick) in historical_sets:
        for repl in ranked:
            if repl in pick:
                continue
            trial = pick[:-1] + [repl]
            if (frozenset(trial) not in historical_sets
                    and not _has_triple_run(trial)
                    and sum(1 for x in trial if x <= 31) <= max_low):
                pick = trial
                break

    return sorted(pick)


# ============================================================
# ENSEMBLE
# ============================================================
def normalize(arr: np.ndarray) -> np.ndarray:
    valid = arr[1:]
    positive = valid[valid > 0]
    if len(positive) == 0:
        return arr
    mn, mx = positive.min(), positive.max()
    result = np.zeros_like(arr)
    if mx == mn:
        # All positive scores equal: map to 1.0 so raw magnitudes
        # never leak into the weighted ensemble at their own scale
        result[1:][valid > 0] = 1.0
        return result
    for i in range(1, len(arr)):
        if arr[i] > 0:
            result[i] = (arr[i] - mn) / (mx - mn)
    return result


def neutralize_new_numbers(scores: np.ndarray, old_pool: int = ERA2_POOL) -> np.ndarray:
    """
    Numbers above `old_pool` (51, 52) exist only in the 7/52 era (~50 draws), so
    their statistics are too thin to compare with 1..50. Give them the average
    score of the established numbers instead of letting noise rank them.
    """
    if len(scores) - 1 <= old_pool:
        return scores
    out = scores.copy()
    base = out[1:old_pool + 1]
    pos = base[base > 0]
    out[old_pool + 1:] = pos.mean() if len(pos) else 0.0
    return out


def ensemble_predict(
    draws: list, num_range: int, pick_count: int,
    lstm_model=None, weights=None, draws_dated=None, historical_sets=None,
    seed: Optional[int] = None
) -> dict:
    if len(draws) < 5:
        nums = list(np.random.choice(range(1, num_range + 1), pick_count, replace=False))
        return {"numbers": sorted(nums), "confidence": 0, "strategies": {}, "ev_info": None}

    w = weights or DEFAULT_WEIGHTS

    # Run all strategies
    s1 = lstm_predict(lstm_model, draws, num_range) if lstm_model else np.zeros(num_range + 1)
    s2 = strategy_frequency_recency(draws, num_range)
    s3 = strategy_gap_analysis(draws, num_range)
    s4 = strategy_pair_analysis(draws, num_range)
    s5 = strategy_distribution(draws, num_range)
    s6 = strategy_seed_analysis(draws_dated, num_range, pick_count) if draws_dated and w.get("seed", 0) > 0 else np.zeros(num_range + 1)
    s7 = strategy_smart_pick(num_range)
    s1, s2, s3, s4, s5 = (neutralize_new_numbers(s) for s in (s1, s2, s3, s4, s5))
    jitter_rng = np.random.default_rng(seed) if seed is not None else None

    # Normalize
    n1, n2, n3, n4, n5, n6, n7 = (normalize(s) for s in (s1, s2, s3, s4, s5, s6, s7))

    # Weighted combination
    combined = np.zeros(num_range + 1)
    for i in range(1, num_range + 1):
        combined[i] = (
            n1[i] * w.get("lstm", 0) +
            n2[i] * w.get("frequency", 0) +
            n3[i] * w.get("gap", 0) +
            n4[i] * w.get("pair", 0) +
            n5[i] * w.get("distribution", 0) +
            n6[i] * w.get("seed", 0) +
            n7[i] * w.get("smart", 0)
        )
        # Small randomness for variety
        combined[i] += (jitter_rng.uniform(0, 0.03) if jitter_rng is not None
                        else np.random.uniform(0, 0.03))

    # Rank numbers by combined score
    ranked = sorted(range(1, num_range + 1), key=lambda n: -combined[n])

    # EV guard: when Smart Pick is active, veto popular-combination shapes
    guard_active = w.get("smart", 0) > 0
    if guard_active:
        selected = apply_ev_guard(ranked, pick_count, historical_sets)
    else:
        selected = sorted(ranked[:pick_count])

    # Confidence
    top_scores = [combined[n] for n in selected]
    avg_top = np.mean(top_scores)
    avg_all = np.mean(combined[1:])
    confidence = min(100, max(0, int(((avg_top - avg_all) / (avg_top + 1e-10)) * 100)))

    # Strategy breakdown per number
    strategies = {}
    for num in selected:
        strategies[str(num)] = {
            "lstm": round(float(n1[num]), 3),
            "frequency": round(float(n2[num]), 3),
            "gap": round(float(n3[num]), 3),
            "pair": round(float(n4[num]), 3),
            "distribution": round(float(n5[num]), 3),
            "seed": round(float(n6[num]), 3),
            "smart": round(float(n7[num]), 3),
            "total": round(float(combined[num]), 3),
        }

    total_weight = sum(w.get(k, 0) for k in DEFAULT_WEIGHTS)
    if total_weight <= 0:
        confidence = 0  # nothing but jitter drove the pick — it is pure random

    low_count = sum(1 for n in selected if n <= 31)
    is_past_winner = frozenset(selected) in (historical_sets or set())
    ev_info = {
        "low_count": low_count,
        "max_low": EV_MAX_LOW_NUMBERS,
        "sum": int(sum(selected)),
        "guard_applied": guard_active,
        "is_past_winner": is_past_winner,
        "share_risk": "low" if (guard_active
                                and low_count <= EV_MAX_LOW_NUMBERS
                                and not is_past_winner
                                and not _has_triple_run(selected)) else "high",
    }
    if total_weight <= 0:
        ev_info["note"] = "all strategy weights are zero — this pick is uniform random"

    return {"numbers": selected, "confidence": confidence, "strategies": strategies, "ev_info": ev_info}


# ============================================================
# Logging helper
# ============================================================
def log_msg(msg: str):
    entry = {"time": datetime.now().strftime("%H:%M:%S"), "msg": msg}
    state["training_log"].append(entry)
    if len(state["training_log"]) > 200:
        state["training_log"] = state["training_log"][-200:]
    logger.info(msg)


# ============================================================
# API Endpoints
# ============================================================
def startup():
    load_csv_data()
    # Try loading existing model
    if TF_AVAILABLE:
        model_path = MODEL_DIR / "lottomax_main.keras"
        if model_path.exists():
            try:
                model = keras.models.load_model(model_path)
                if (model.output_shape[-1] != LOTTO_MAX
                        or tuple(model.input_shape[1:]) != (SEQUENCE_LENGTH, LOTTO_MAX)):
                    log_msg(f"⚠️ Saved model targets a {model.output_shape[-1]}-number pool, "
                            f"not {LOTTO_MAX} — ignored, please retrain")
                else:
                    state["main_model"] = model
                    log_msg("✅ Loaded existing model: lottomax_main")
                    vpath = MODEL_DIR / "lottomax_main.verdict.json"
                    if vpath.exists():
                        try:
                            state["lstm_verdict"] = json.loads(vpath.read_text())
                        except (OSError, ValueError):
                            pass
            except Exception as e:
                log_msg(f"⚠️ Could not load lottomax_main: {e}")
    else:
        log_msg("⚠️ TensorFlow not installed — LSTM disabled, statistical strategies active")


@app.get("/")
def health():
    return {
        "status": "ok",
        "main_draws": len(state["main_draws"]),
        "all_draws": len(state["all_draws"]),
        "era_start": ERA2_START_DATE,
        "era3_start": ERA3_START_DATE,
        "pool_size": LOTTO_MAX,
        "tf_available": TF_AVAILABLE,
        "main_model_loaded": state["main_model"] is not None,
        "last_trained": state["last_trained"],
        "lstm_verdict": state["lstm_verdict"],
    }


@app.get("/status")
def training_status():
    return {
        "is_training": state["is_training"],
        "progress": state["training_progress"],
        "log": state["training_log"][-30:],
        "main_model_ready": state["main_model"] is not None,
        "lstm_verdict": state["lstm_verdict"],
    }


class TrainRequest(BaseModel):
    epochs: int = 100
    weights: Optional[dict] = None
    run_seed_analysis: bool = False  # ~650k seed generations; transparency demo only


@app.post("/train")
async def train(req: TrainRequest, background_tasks: BackgroundTasks):
    if state["is_training"]:
        raise HTTPException(400, "Training already in progress")
    if len(state["main_draws"]) == 0:
        raise HTTPException(400, "No data loaded")

    # Claim the flag before returning: prevents two rapid POST /train
    # requests from both passing the check and training concurrently
    state["is_training"] = True
    background_tasks.add_task(run_training, req.epochs, req.run_seed_analysis)
    return {"status": "training_started", "epochs": req.epochs,
            "run_seed_analysis": req.run_seed_analysis}


def run_training(epochs: int, do_seed_analysis: bool = False):
    state["is_training"] = True
    state["training_log"] = []

    try:
        log_msg("=" * 50)
        log_msg("🚀 Starting Ensemble Training")
        log_msg("=" * 50)

        # Train main model
        log_msg(f"\n📊 LOTTOMAX Main ({len(state['main_draws'])} draws)")
        state["training_progress"]["strategy"] = "LSTM (Main)"
        state["main_model"] = train_lstm(
            state["main_draws"], LOTTO_MAX, "lottomax_main", epochs
        )

        # Statistical strategies (instant)
        log_msg("\n📊 Statistical Strategies")
        log_msg("  ✅ Frequency + Recency: Ready")
        log_msg("  ✅ Gap Analysis: Ready")
        log_msg("  ✅ Pair Correlation: Ready")
        log_msg("  ✅ Distribution Balance: Ready")

        if do_seed_analysis:
            log_msg("\n🔑 Seed/RNG Analysis (transparency only — no predictive power)")
            load_csv_data_with_dates()
            if state.get("main_draws_dated"):
                results = run_seed_analysis(state["main_draws_dated"], LOTTO_MAX, LOTTO_PICK, 20)
                state["seed_analysis"] = results
                log_msg(f"  Tested {results['tested_seeds']} seeds across 3 algorithms")
                log_msg(f"  Perfect matches: {len(results['best_matches'])}")
                log_msg(f"  Partial matches (4+): {len(results['partial_matches'])}")
                log_msg("  ✅ Seed Analysis: Ready")
        else:
            log_msg("\n🔑 Seed/RNG Analysis skipped (POST /seed-analysis to run it on demand)")

        log_msg("\n💰 Smart Pick (EV): Ready — avoids popular combos to reduce prize splitting")

        state["last_trained"] = datetime.now().isoformat()
        log_msg("\n" + "=" * 50)
        log_msg("🎯 All 7 strategies ready!")
        log_msg("=" * 50)

        state["training_progress"] = {"status": "complete", "epoch": 0, "total_epochs": 0, "loss": 0, "strategy": ""}

    except Exception as e:
        log_msg(f"❌ Training error: {str(e)}")
        state["training_progress"] = {"status": "error", "epoch": 0, "total_epochs": 0, "loss": 0, "strategy": str(e)}
    finally:
        state["is_training"] = False


class PredictRequest(BaseModel):
    weights: Optional[dict] = None
    mode: Literal["ensemble", "smart_v2"] = "ensemble"


def sanitize_weights(weights: Optional[dict]) -> Optional[dict]:
    """Validate user-supplied strategy weights: known keys, finite numbers, clamped to [0, 10]."""
    if not weights:
        return None
    clean = {}
    for key in DEFAULT_WEIGHTS:
        if key not in weights:
            continue
        try:
            v = float(weights[key])
        except (TypeError, ValueError):
            raise HTTPException(422, f"weight '{key}' must be a number")
        if math.isnan(v) or math.isinf(v):
            raise HTTPException(422, f"weight '{key}' must be finite")
        clean[key] = min(max(v, 0.0), 10.0)
    return clean or None


@app.post("/predict")
def predict(req: PredictRequest = None):
    if len(state["main_draws"]) == 0:
        raise HTTPException(400, "No data loaded")

    req = req or PredictRequest()
    if req.mode == "smart_v2":
        t = ev_model.smart_tickets(1, LOTTO_MAX, state.get("historical_sets"))[0]
        main_result = {
            "numbers": t["numbers"],
            "confidence": 0,
            "strategies": {},
            "ev_info": {
                "mode": "smart_v2",
                "popularity_ratio": t["popularity_ratio"],
                "low_count": t["low_count"],
                "max_low": ev_model.EV_MAX_LOW,
                "sum": t["sum"],
                "guard_applied": True,
                "share_risk": t["share_risk"],
                "note": "popularity_ratio는 가정 기반 추정치입니다. 당첨 확률은 모든 조합이 동일합니다.",
            },
        }
        return {"main": main_result, "mode": "smart_v2",
                "model_trained": state["main_model"] is not None,
                "timestamp": datetime.now().isoformat()}

    weights = sanitize_weights(req.weights)

    main_result = ensemble_predict(
        state["main_draws"], LOTTO_MAX, LOTTO_PICK,
        state["main_model"], weights,
        draws_dated=state.get("main_draws_dated"),
        historical_sets=state.get("historical_sets"),
    )

    return {
        "main": main_result,
        "mode": "ensemble",
        "model_trained": state["main_model"] is not None,
        "timestamp": datetime.now().isoformat(),
    }


class BatchRequest(BaseModel):
    count: int = Field(5, ge=1, le=10)


@app.post("/predict-batch")
def predict_batch(req: BatchRequest):
    """1-10 Smart Pick v2 tickets, pairwise overlapping in at most 3 numbers."""
    tickets = ev_model.smart_tickets(req.count, LOTTO_MAX, state.get("historical_sets"))
    return {
        "tickets": tickets,
        "note": "popularity_ratio는 가정 기반 추정치입니다. 당첨 확률은 모든 조합이 동일합니다.",
    }


class TicketRequest(BaseModel):
    numbers: list[int] = Field(..., min_length=7, max_length=7)


def _check_numbers(numbers: list) -> list:
    if len(set(numbers)) != LOTTO_PICK or any(n < 1 or n > LOTTO_MAX for n in numbers):
        raise HTTPException(422, f"numbers must be {LOTTO_PICK} distinct integers in 1..{LOTTO_MAX}")
    return sorted(numbers)


@app.post("/history-check")
def history_check(req: TicketRequest):
    """Compare a ticket with every past draw (full history). Past overlap says nothing about the future."""
    from scipy import stats as sps
    nums = frozenset(_check_numbers(req.numbers))
    history = state["history_list"]
    if not history:
        raise HTTPException(400, "No data loaded")
    hist = {str(k): 0 for k in range(LOTTO_PICK + 1)}
    expected = {str(k): 0.0 for k in range(LOTTO_PICK + 1)}
    exact, max_overlap = None, 0
    for h in history:
        ov = len(nums & set(h["numbers"]))
        hist[str(ov)] += 1
        max_overlap = max(max_overlap, ov)
        if ov == LOTTO_PICK:
            exact = {"draw_number": h["draw_number"], "date": h["date"]}
    # Random-ticket expectation, draw by draw: the ticket has `m` numbers inside that
    # draw's pool (numbers 51/52 cannot exist in older pools), so overlap is hypergeometric.
    by_pool: dict = {}
    for h in history:
        pool = pool_for_date(h["date"])
        by_pool[pool] = by_pool.get(pool, 0) + 1
    for pool, cnt in by_pool.items():
        m = sum(1 for n in nums if n <= pool)
        pmf = sps.hypergeom(pool, m, LOTTO_PICK).pmf(range(LOTTO_PICK + 1))
        for k in range(LOTTO_PICK + 1):
            expected[str(k)] += cnt * float(pmf[k])
    return {
        "exact_match": exact,
        "max_overlap": max_overlap,
        "n_draws": len(history),
        "overlap_histogram": hist,
        "expected_histogram_random": {k: round(v, 2) for k, v in expected.items()},
        "note": "과거 당첨 조합과 겹친다고 미래에 유리한 것은 아닙니다. 매 추첨은 독립입니다.",
    }


class EvRequest(BaseModel):
    jackpot: float = Field(50_000_000, ge=0, allow_inf_nan=False)
    tickets_sold: float = Field(25_000_000, ge=0, allow_inf_nan=False)
    ticket_price: float = Field(TICKET_PRICE, gt=0, allow_inf_nan=False)
    lines_per_ticket: int = Field(LINES_PER_TICKET, ge=1, le=100)
    other_prizes_ev: float = Field(0.0, ge=0, allow_inf_nan=False)
    numbers: Optional[list[int]] = None


@app.post("/ev/estimate")
def ev_estimate(req: EvRequest):
    ratio = None
    if req.numbers is not None:
        if len(req.numbers) != LOTTO_PICK:
            raise HTTPException(422, f"numbers must contain {LOTTO_PICK} values")
        ratio = ev_model.popularity_ratio(_check_numbers(req.numbers), LOTTO_MAX,
                                          state.get("historical_sets"))
    return ev_model.ev_estimate(req.jackpot, req.tickets_sold, req.ticket_price,
                                req.lines_per_ticket, req.other_prizes_ev, ratio, LOTTO_MAX)


# ------------------------------------------------------------
# Data ingest
# ------------------------------------------------------------
@app.get("/data/status")
def data_status():
    last = data_ingest.read_last_draw(DATA_DIR)
    today = datetime.now().date()
    days = (today - datetime.strptime(last["date"], "%Y-%m-%d").date()).days
    return {
        "last_draw_number": last["draw_number"],
        "last_draw_date": last["date"],
        "days_since_last": days,
        "estimated_missing_draws": data_ingest.estimate_missing_draws(last["date"], today),
        "era2_draws": len(state["main_draws"]),
        "era3_draws": sum(1 for p in state["main_draw_pools"] if p == LOTTO_MAX),
        "total_draws": len(state["all_draws"]),
        "pool_size": LOTTO_MAX,
    }


class NewDraw(BaseModel):
    draw_number: int
    date: str
    numbers: list[int]
    bonus: int


class AppendRequest(BaseModel):
    draws: list[NewDraw] = Field(..., min_length=1, max_length=200)


@app.post("/data/append")
def data_append(req: AppendRequest):
    if state["is_training"]:
        raise HTTPException(400, "Training in progress — try again after it finishes")
    last = data_ingest.read_last_draw(DATA_DIR)
    valid, errors = data_ingest.validate_new_draws(
        [d.model_dump() for d in req.draws], last["draw_number"], last["date"], last["existing"])
    if not valid:
        detail = "\n".join(f"{e['row']}번째 줄 (회차 {e['draw_number']}): {e['error']}" for e in errors)
        raise HTTPException(400, detail or "추가할 유효한 회차가 없습니다")
    backup = data_ingest.append_draws_to_csv(valid, DATA_DIR)
    load_csv_data()
    state["backtest"] = None
    state["signal_lab"] = None
    return {
        "added": len(valid),
        "rejected": errors,
        "backup": backup.name if backup else None,
        "total_draws": len(state["all_draws"]),
        "era2_draws": len(state["main_draws"]),
        "last_draw_number": valid[-1]["draw_number"],
        "recommendation": "LSTM 재학습 권장 (Train LSTM Model). 백테스트·Signal Lab 결과는 초기화되었습니다.",
    }


# ------------------------------------------------------------
# Signal Lab (background thread + progress)
# ------------------------------------------------------------
class SignalLabRequest(BaseModel):
    seed: int = 42
    quick: bool = False


def _run_signal_lab(seed: int, quick: bool):
    def progress(stage, done, total):
        state["signal_lab_progress"] = {"stage": stage, "done": int(done), "total": int(total)}

    try:
        draws, last_date = signal_lab_draws()
        result = signal_lab.run_signal_lab(draws, seed=seed, quick=quick, progress=progress,
                                           last_draw_date=last_date)
        state["signal_lab"] = result
    except Exception as e:  # keep the flag honest even if something breaks
        logger.exception("Signal Lab failed")
        state["signal_lab"] = None
        state["signal_lab_error"] = str(e)
    finally:
        state["signal_lab_progress"] = None
        state["signal_lab_running"] = False


_signal_lab_lock = threading.Lock()


@app.post("/signal-lab/run")
def signal_lab_run(req: SignalLabRequest = None):
    req = req or SignalLabRequest()
    with _signal_lab_lock:
        if state["signal_lab_running"]:
            raise HTTPException(400, "Signal Lab already running")
        if len(state["main_draws"]) == 0:
            raise HTTPException(400, "No data loaded")
        state["signal_lab_running"] = True
        state["signal_lab"] = None
        state["signal_lab_error"] = None
        state["signal_lab_progress"] = {"stage": "starting", "done": 0, "total": 1}
    threading.Thread(target=_run_signal_lab, args=(req.seed, req.quick), daemon=True).start()
    return {"status": "started", "seed": req.seed, "quick": req.quick}


@app.get("/signal-lab/status")
def signal_lab_status():
    return {
        "running": state["signal_lab_running"],
        "has_result": state["signal_lab"] is not None,
        "progress": state["signal_lab_progress"],
        "error": state.get("signal_lab_error"),
    }


@app.get("/signal-lab/result")
def signal_lab_result():
    if state["signal_lab"] is None:
        raise HTTPException(404, "No Signal Lab result yet — POST /signal-lab/run first")
    return state["signal_lab"]


@app.get("/frequencies")
def frequencies():
    """Number frequency analysis."""
    if len(state["main_draws"]) == 0:
        raise HTTPException(400, "No data loaded")

    # Main frequencies
    main_freq = np.zeros(LOTTO_MAX + 1, dtype=int)
    main_recent_freq = np.zeros(LOTTO_MAX + 1, dtype=int)
    recent = state["main_draws"][-HOT_COLD_WINDOW:]

    for draw in state["main_draws"]:
        for n in draw:
            main_freq[n] += 1
    for draw in recent:
        for n in draw:
            main_recent_freq[n] += 1

    # Gap info
    last_seen = {}
    for i, draw in enumerate(state["main_draws"]):
        for n in draw:
            last_seen[n] = i
    total = len(state["main_draws"])
    gaps = {str(n): total - last_seen.get(n, 0) for n in range(1, LOTTO_MAX + 1)}

    return {
        "main_total": {str(i): int(main_freq[i]) for i in range(1, LOTTO_MAX + 1)},
        "main_recent": {str(i): int(main_recent_freq[i]) for i in range(1, LOTTO_MAX + 1)},
        "main_gaps": gaps,
        "total_draws": total,
        "recent_window": HOT_COLD_WINDOW,
    }


class SeedAnalysisRequest(BaseModel):
    max_draws: int = 20


@app.post("/seed-analysis")
def seed_analysis(req: SeedAnalysisRequest):
    draws_dated = state.get("main_draws_dated", [])
    if not draws_dated:
        raise HTTPException(400, "No dated data loaded")
    results = run_seed_analysis(draws_dated, LOTTO_MAX, LOTTO_PICK, req.max_draws)
    state["seed_analysis"] = results
    return {
        "tested_draws": results["tested_draws"],
        "tested_seeds": results["tested_seeds"],
        "perfect_matches": len(results["best_matches"]),
        "partial_matches": len(results["partial_matches"]),
        "best_matches": results["best_matches"][:5],
        "top_partial": results["partial_matches"][:10],
        "algo_scores": results["algo_scores"],
    }


class BacktestRequest(BaseModel):
    window: int = 150
    random_tickets: int = 100


@app.post("/backtest")
def backtest(req: BacktestRequest = None):
    """
    Walk-forward backtest of the statistical strategies on real history,
    against a random-ticket baseline. This is the app's honesty check:
    it shows whether any strategy actually beats chance (spoiler: no).
    """
    draws = state["main_draws"]
    if len(draws) < 60:
        raise HTTPException(400, "Not enough data for a backtest")
    req = req or BacktestRequest()
    window = max(30, min(req.window, 400))
    n_random = max(10, min(req.random_tickets, 500))

    strategies = {
        "frequency": lambda d: neutralize_new_numbers(strategy_frequency_recency(d, LOTTO_MAX)),
        "gap": lambda d: neutralize_new_numbers(strategy_gap_analysis(d, LOTTO_MAX)),
        "pair": lambda d: neutralize_new_numbers(strategy_pair_analysis(d, LOTTO_MAX)),
        "distribution": lambda d: neutralize_new_numbers(strategy_distribution(d, LOTTO_MAX)),
        "smart": lambda d: strategy_smart_pick(LOTTO_MAX),
    }

    def guarded_selector(scores, pick_count):
        # Mirror production selection: rank by score, then apply the EV guard
        ranked = sorted(range(1, len(scores)), key=lambda n: -scores[n])
        return set(apply_ev_guard(ranked, pick_count, state["historical_sets"]))

    results = run_walk_forward(
        draws, LOTTO_MAX, LOTTO_PICK, strategies,
        window=window, n_random=n_random,
        selectors={"smart": guarded_selector},
        ensemble_weights={k: DEFAULT_WEIGHTS.get(k, 0) for k in strategies},
        ensemble_selector=guarded_selector,
        pools=state["main_draw_pools"],
    )
    state["backtest"] = results
    return results


@app.post("/reload-data")
def reload_data():
    """Reload CSV data from disk."""
    load_csv_data()
    return {
        "status": "ok",
        "main_draws": len(state["main_draws"]),
        "all_draws": len(state["all_draws"]),
    }


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    import uvicorn
    # Local-only: bind to loopback, never 0.0.0.0
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("LOTTOMAX_PORT", "8000")))
