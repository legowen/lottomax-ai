"""
LottoMax AI - 6-Strategy Ensemble Prediction Engine
====================================================
Backend: FastAPI + TensorFlow LSTM + Statistical Analysis

Strategies:
1. LSTM Sequential Pattern - Deep learning on draw sequences
2. Frequency + Recency - Hot/cold with exponential decay
3. Gap Analysis - Overdue numbers based on gap distributions
4. Pair Correlation - Co-occurrence patterns
5. Distribution Balance - Range & odd/even equilibrium
6. Seed/RNG Analysis - Time-based PRNG reverse engineering
"""

import os
import json
import time
import struct
import logging
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime, timezone, timedelta
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# Config
# ============================================================
DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

LOTTO_MAX = 50
LOTTO_PICK = 7
SEQUENCE_LENGTH = 20  # How many past draws the LSTM looks at
HOT_COLD_WINDOW = 50
RECENT_WINDOW = 30

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lottomax-ai")

# ============================================================
# App
# ============================================================
app = FastAPI(title="LottoMax AI", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
state = {
    "main_draws": [],
    "main_draws_dated": [],
    "main_model": None,
    "is_training": False,
    "training_progress": {"status": "idle", "epoch": 0, "total_epochs": 0, "loss": 0, "strategy": ""},
    "training_log": [],
    "last_trained": None,
    "seed_analysis": None,
}


# ============================================================
# Data Loading
# ============================================================
def load_csv_data():
    """Load LottoMax CSV file."""
    main_path = DATA_DIR / "LOTTOMAX.csv"

    if main_path.exists():
        df = pd.read_csv(main_path)
        # Only main draws (SEQUENCE NUMBER == 0)
        df_main = df[df["SEQUENCE NUMBER"] == 0].sort_values("DRAW NUMBER")
        draws = []
        for _, row in df_main.iterrows():
            nums = [int(row[f"NUMBER DRAWN {i}"]) for i in range(1, 8)]
            draws.append(nums)
        state["main_draws"] = draws
        logger.info(f"Loaded {len(draws)} main draws")
    else:
        logger.warning(f"Main CSV not found at {main_path}")


def load_csv_data_with_dates():
    main_path = DATA_DIR / "LOTTOMAX.csv"
    main_draws_dated = []
    if main_path.exists():
        df = pd.read_csv(main_path)
        df_main = df[df["SEQUENCE NUMBER"] == 0].sort_values("DRAW NUMBER")
        for _, row in df_main.iterrows():
            nums = [int(row[f"NUMBER DRAWN {i}"]) for i in range(1, 8)]
            date_str = str(row.get("DRAW DATE", "")).strip().strip('"')
            main_draws_dated.append({"numbers": nums, "date": date_str})
    state["main_draws_dated"] = main_draws_dated
    return main_draws_dated


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


def build_lstm_model(num_range: int, seq_len: int = SEQUENCE_LENGTH) -> keras.Model:
    """
    Build a Bidirectional LSTM model for lottery prediction.

    Architecture designed to capture:
    - Short-term patterns (recent draw correlations)
    - Long-term patterns (cyclical tendencies)
    - Inter-number relationships (which numbers appear together)
    """
    model = Sequential([
        # First LSTM layer - captures broad temporal patterns
        Bidirectional(
            LSTM(128, return_sequences=True, input_shape=(seq_len, num_range)),
        ),
        BatchNormalization(),
        Dropout(0.3),

        # Second LSTM layer - refines sequential patterns
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),

        # Third LSTM layer - final temporal encoding
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),

        # Dense layers - map temporal features to number probabilities
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(num_range, activation="sigmoid"),  # Independent probability per number
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.build(input_shape=(None, seq_len, num_range))

    return model


def train_lstm(draws: list, num_range: int, model_name: str, epochs: int = 100):
    """Train LSTM model on draw history."""
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

    # Save model
    model_path = MODEL_DIR / f"{model_name}.keras"
    model.save(model_path)
    log_msg(f"  💾 Model saved: {model_path}")

    return model


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


def lstm_predict(model: keras.Model, draws: list, num_range: int) -> np.ndarray:
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
# ENSEMBLE
# ============================================================
def normalize(arr: np.ndarray) -> np.ndarray:
    valid = arr[1:]
    positive = valid[valid > 0]
    if len(positive) == 0:
        return arr
    mn, mx = positive.min(), positive.max()
    if mx == mn:
        return arr
    result = np.zeros_like(arr)
    for i in range(1, len(arr)):
        if arr[i] > 0:
            result[i] = (arr[i] - mn) / (mx - mn)
    return result


def ensemble_predict(
    draws: list, num_range: int, pick_count: int,
    lstm_model=None, weights=None, draws_dated=None
) -> dict:
    if len(draws) < 5:
        nums = list(np.random.choice(range(1, num_range + 1), pick_count, replace=False))
        return {"numbers": sorted(nums), "confidence": 0, "strategies": {}}

    default_weights = {
        "lstm": 0.25, "frequency": 0.18, "gap": 0.17,
        "pair": 0.13, "distribution": 0.12, "seed": 0.15,
    }
    w = weights or default_weights

    # Run all strategies
    s1 = lstm_predict(lstm_model, draws, num_range) if lstm_model else np.zeros(num_range + 1)
    s2 = strategy_frequency_recency(draws, num_range)
    s3 = strategy_gap_analysis(draws, num_range)
    s4 = strategy_pair_analysis(draws, num_range)
    s5 = strategy_distribution(draws, num_range)
    s6 = strategy_seed_analysis(draws_dated, num_range, pick_count) if draws_dated else np.zeros(num_range + 1)

    # Normalize
    n1, n2, n3, n4, n5, n6 = normalize(s1), normalize(s2), normalize(s3), normalize(s4), normalize(s5), normalize(s6)

    # Weighted combination
    combined = np.zeros(num_range + 1)
    for i in range(1, num_range + 1):
        combined[i] = (
            n1[i] * w.get("lstm", 0) +
            n2[i] * w.get("frequency", 0) +
            n3[i] * w.get("gap", 0) +
            n4[i] * w.get("pair", 0) +
            n5[i] * w.get("distribution", 0) +
            n6[i] * w.get("seed", 0)
        )
        # Small randomness for variety
        combined[i] += np.random.uniform(0, 0.03)

    # Select top numbers
    ranked = [(i, combined[i]) for i in range(1, num_range + 1)]
    ranked.sort(key=lambda x: -x[1])
    selected = sorted([r[0] for r in ranked[:pick_count]])

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
            "total": round(float(combined[num]), 3),
        }

    return {"numbers": selected, "confidence": confidence, "strategies": strategies}


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
@app.on_event("startup")
async def startup():
    load_csv_data()
    load_csv_data_with_dates()
    # Try loading existing model
    model_path = MODEL_DIR / "lottomax_main.keras"
    if model_path.exists():
        try:
            state["main_model"] = keras.models.load_model(model_path)
            log_msg("✅ Loaded existing model: lottomax_main")
        except Exception as e:
            log_msg(f"⚠️ Could not load lottomax_main: {e}")


@app.get("/")
def health():
    return {
        "status": "ok",
        "main_draws": len(state["main_draws"]),
        "main_model_loaded": state["main_model"] is not None,
        "last_trained": state["last_trained"],
    }


@app.get("/status")
def training_status():
    return {
        "is_training": state["is_training"],
        "progress": state["training_progress"],
        "log": state["training_log"][-30:],
        "main_model_ready": state["main_model"] is not None,
    }


class TrainRequest(BaseModel):
    epochs: int = 100
    weights: Optional[dict] = None


@app.post("/train")
async def train(req: TrainRequest, background_tasks: BackgroundTasks):
    if state["is_training"]:
        raise HTTPException(400, "Training already in progress")
    if len(state["main_draws"]) == 0:
        raise HTTPException(400, "No data loaded")

    background_tasks.add_task(run_training, req.epochs)
    return {"status": "training_started", "epochs": req.epochs}


def run_training(epochs: int):
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

        log_msg("\n🔑 Seed/RNG Analysis")
        load_csv_data_with_dates()
        if state.get("main_draws_dated"):
            results = run_seed_analysis(state["main_draws_dated"], LOTTO_MAX, LOTTO_PICK, 20)
            state["seed_analysis"] = results
            log_msg(f"  Tested {results['tested_seeds']} seeds across 3 algorithms")
            log_msg(f"  Perfect matches: {len(results['best_matches'])}")
            log_msg(f"  Partial matches (4+): {len(results['partial_matches'])}")
            log_msg("  ✅ Seed Analysis: Ready")

        state["last_trained"] = datetime.now().isoformat()
        log_msg("\n" + "=" * 50)
        log_msg("🎯 All 6 strategies trained and ready!")
        log_msg("=" * 50)

        state["training_progress"] = {"status": "complete", "epoch": 0, "total_epochs": 0, "loss": 0, "strategy": ""}

    except Exception as e:
        log_msg(f"❌ Training error: {str(e)}")
        state["training_progress"] = {"status": "error", "epoch": 0, "total_epochs": 0, "loss": 0, "strategy": str(e)}
    finally:
        state["is_training"] = False


class PredictRequest(BaseModel):
    weights: Optional[dict] = None


@app.post("/predict")
def predict(req: PredictRequest = None):
    if len(state["main_draws"]) == 0:
        raise HTTPException(400, "No data loaded")

    weights = req.weights if req else None

    main_result = ensemble_predict(
        state["main_draws"], LOTTO_MAX, LOTTO_PICK,
        state["main_model"], weights,
        draws_dated=state.get("main_draws_dated"),
    )

    return {
        "main": main_result,
        "model_trained": state["main_model"] is not None,
        "timestamp": datetime.now().isoformat(),
    }


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


@app.post("/reload-data")
def reload_data():
    """Reload CSV data from disk."""
    load_csv_data()
    load_csv_data_with_dates()
    return {
        "status": "ok",
        "main_draws": len(state["main_draws"]),
    }


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
