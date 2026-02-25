"""
LottoMax AI - 5-Strategy Ensemble Prediction Engine
====================================================
Backend: FastAPI + TensorFlow LSTM + Statistical Analysis

Strategies:
1. LSTM Sequential Pattern - Deep learning on draw sequences
2. Frequency + Recency - Hot/cold with exponential decay
3. Gap Analysis - Overdue numbers based on gap distributions
4. Pair Correlation - Co-occurrence patterns
5. Distribution Balance - Range & odd/even equilibrium
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
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
EXTRA_MAX = 99
EXTRA_PICK = 4
SEQUENCE_LENGTH = 20  # How many past draws the LSTM looks at
HOT_COLD_WINDOW = 50
RECENT_WINDOW = 30

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lottomax-ai")

# ============================================================
# App
# ============================================================
app = FastAPI(title="LottoMax AI", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
state = {
    "main_draws": [],
    "extra_draws": [],
    "main_model": None,
    "extra_model": None,
    "is_training": False,
    "training_progress": {"status": "idle", "epoch": 0, "total_epochs": 0, "loss": 0, "strategy": ""},
    "training_log": [],
    "last_trained": None,
}


# ============================================================
# Data Loading
# ============================================================
def load_csv_data():
    """Load LottoMax and Extra CSV files."""
    main_path = DATA_DIR / "LOTTOMAX.csv"
    extra_path = DATA_DIR / "LOTTOMAXExtra.csv"

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

    if extra_path.exists():
        df = pd.read_csv(extra_path)
        draws = []
        for _, row in df.iterrows():
            nums = [int(row[f"NUMBER DRAWN {i}"]) for i in range(1, 5)]
            draws.append(nums)
        state["extra_draws"] = draws
        logger.info(f"Loaded {len(draws)} extra draws")
    else:
        logger.warning(f"Extra CSV not found at {extra_path}")


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
    lstm_model=None, weights=None
) -> dict:
    if len(draws) < 5:
        nums = list(np.random.choice(range(1, num_range + 1), pick_count, replace=False))
        return {"numbers": sorted(nums), "confidence": 0, "strategies": {}}

    default_weights = {
        "lstm": 0.30,
        "frequency": 0.20,
        "gap": 0.20,
        "pair": 0.15,
        "distribution": 0.15,
    }
    w = weights or default_weights

    # Run all strategies
    s1 = lstm_predict(lstm_model, draws, num_range) if lstm_model else np.zeros(num_range + 1)
    s2 = strategy_frequency_recency(draws, num_range)
    s3 = strategy_gap_analysis(draws, num_range)
    s4 = strategy_pair_analysis(draws, num_range)
    s5 = strategy_distribution(draws, num_range)

    # Normalize
    n1, n2, n3, n4, n5 = normalize(s1), normalize(s2), normalize(s3), normalize(s4), normalize(s5)

    # Weighted combination
    combined = np.zeros(num_range + 1)
    for i in range(1, num_range + 1):
        combined[i] = (
            n1[i] * w["lstm"] +
            n2[i] * w["frequency"] +
            n3[i] * w["gap"] +
            n4[i] * w["pair"] +
            n5[i] * w["distribution"]
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
    # Try loading existing models
    for name, key in [("lottomax_main", "main_model"), ("lottomax_extra", "extra_model")]:
        model_path = MODEL_DIR / f"{name}.keras"
        if model_path.exists():
            try:
                state[key] = keras.models.load_model(model_path)
                log_msg(f"✅ Loaded existing model: {name}")
            except Exception as e:
                log_msg(f"⚠️ Could not load {name}: {e}")


@app.get("/")
def health():
    return {
        "status": "ok",
        "main_draws": len(state["main_draws"]),
        "extra_draws": len(state["extra_draws"]),
        "main_model_loaded": state["main_model"] is not None,
        "extra_model_loaded": state["extra_model"] is not None,
        "last_trained": state["last_trained"],
    }


@app.get("/status")
def training_status():
    return {
        "is_training": state["is_training"],
        "progress": state["training_progress"],
        "log": state["training_log"][-30:],
        "main_model_ready": state["main_model"] is not None,
        "extra_model_ready": state["extra_model"] is not None,
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

        # Train extra model
        if len(state["extra_draws"]) > SEQUENCE_LENGTH + 10:
            log_msg(f"\n📊 LOTTOMAX Extra ({len(state['extra_draws'])} draws)")
            state["training_progress"]["strategy"] = "LSTM (Extra)"
            state["extra_model"] = train_lstm(
                state["extra_draws"], EXTRA_MAX, "lottomax_extra", epochs
            )

        # Statistical strategies (instant)
        log_msg("\n📊 Statistical Strategies")
        log_msg("  ✅ Frequency + Recency: Ready")
        log_msg("  ✅ Gap Analysis: Ready")
        log_msg("  ✅ Pair Correlation: Ready")
        log_msg("  ✅ Distribution Balance: Ready")

        state["last_trained"] = datetime.now().isoformat()
        log_msg("\n" + "=" * 50)
        log_msg("🎯 All strategies trained and ready!")
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
        state["main_model"], weights
    )

    extra_result = None
    if len(state["extra_draws"]) > 0:
        extra_result = ensemble_predict(
            state["extra_draws"], EXTRA_MAX, EXTRA_PICK,
            state["extra_model"], weights
        )

    return {
        "main": main_result,
        "extra": extra_result,
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

    # Extra frequencies
    extra_freq = None
    if state["extra_draws"]:
        extra_freq = np.zeros(EXTRA_MAX + 1, dtype=int)
        for draw in state["extra_draws"][-HOT_COLD_WINDOW:]:
            for n in draw:
                extra_freq[n] += 1
        extra_freq = {str(i): int(extra_freq[i]) for i in range(1, EXTRA_MAX + 1)}

    return {
        "main_total": {str(i): int(main_freq[i]) for i in range(1, LOTTO_MAX + 1)},
        "main_recent": {str(i): int(main_recent_freq[i]) for i in range(1, LOTTO_MAX + 1)},
        "main_gaps": gaps,
        "extra_recent": extra_freq,
        "total_draws": total,
        "recent_window": HOT_COLD_WINDOW,
    }


@app.post("/reload-data")
def reload_data():
    """Reload CSV data from disk."""
    load_csv_data()
    return {
        "status": "ok",
        "main_draws": len(state["main_draws"]),
        "extra_draws": len(state["extra_draws"]),
    }


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
