# 🎰 LottoMax AI — Signal Lab · Smart Pick v2 · EV Calculator (v5.0)

A LottoMax toolkit with an LSTM/statistical ensemble, a **falsifiable deep-learning "Signal Lab"**, an **EV-optimizing Smart Pick v2**, a jackpot **expected-value calculator**, and a safe **data-ingest** flow.

> **Honesty note** — Every number combination has exactly the same chance of winning. According to the research ([docs/RESEARCH.md](docs/RESEARCH.md)), the draw history passes every randomness test, no deep-learning predictor, discriminator or LSTM beats a constant probability, and no strategy beats a random ticket (expected matches 7×7/N). The only thing this app can actually change is the **expected payout when you win (how many co-winners share the jackpot)**, and even that effect is an *assumption-based estimate*. The **Signal Lab** tab lets you verify for yourself that there is no learnable signal.

## Game rules covered

| Era | Period | Rules |
|---|---|---|
| Era 1 | 2009-09-25 – 2019-05-10 | 7/49, weekly |
| Era 2 | 2019-05-14 – 2026-04-10 | 7/50, twice weekly (Tue/Fri) |
| **Era 3 (current)** | **2026-04-14 –** | **7/52, twice weekly, $6 / 4 lines** (C(52,7) = 133,784,560 combinations) |

The statistical strategies and the LSTM use only Era 2+3 draws (from 2019-05-14). Numbers 51 and 52 have only ~50 draws of history, so the statistical strategies score them neutrally. Signal Lab uses only Era 2, where the number pool is fixed. (Evidence for the rule change: [RESEARCH.md §5.1](docs/RESEARCH.md), written in Korean.)

## Architecture

```
┌──────────────────────────────────────────────────────┐
│ React Frontend (:5173) — inline styles only          │
│ generate · analysis · backtest · signal · ev · settings│
└───────────────────────┬──────────────────────────────┘
                        │ API calls (CORS: LOTTOMAX_CORS_ORIGINS)
┌───────────────────────┴──────────────────────────────┐
│ FastAPI Backend (:8000)                              │
│  app.py          ensemble strategies, LSTM, endpoints │
│  signal_lab.py   battery · power check · predictor ·  │
│                  discriminator (seeded, falsifiable)  │
│  ev_model.py     popularity model, Smart Pick v2,     │
│                  Poisson jackpot-sharing EV           │
│  data_ingest.py  validate → backup → atomic CSV append│
│  backtest.py     walk-forward vs random ticket        │
└──────────────────────────────────────────────────────┘
```

## Strategies (ensemble, `mode: "ensemble"`)

| # | Strategy | Default weight | Note |
|---|---|---|---|
| 1 | LSTM | 0.15 | ~14k params. UI warns if it does not beat the constant-probability baseline |
| 2 | Frequency + Recency | 0.15 | |
| 3 | Gap Analysis | 0.20 | |
| 4 | Pair Correlation | 0.05 | significantly *worse* than random in backtests |
| 5 | Distribution Balance | 0.15 | |
| 6 | Seed/RNG Analysis | 0.00 | transparency only — cannot work on certified draw systems |
| 7 | Smart Pick (EV guard) | 0.30 | avoids popular shapes; no effect on win probability |

**Smart Pick v2** (`mode: "smart_v2"`, `/predict-batch`): estimates how popular a combination is with other players (birthday bias, lucky numbers, patterns) and draws *at random* from the least-popular 10% of guard-passing candidates — never a deterministic "best" combination, otherwise every user would pick the same ticket. Popularity weights are **assumptions**, replaceable via `data/popularity_override.json`.

## Quick Start

### Prerequisites

- **Python 3.11** (3.10+ works), **Node.js 18+**

### 1. Backend

```bash
cd be
python3 -m venv venv
source venv/bin/activate          # Mac/Linux    (Windows: venv\Scripts\activate)
pip install -r requirements.txt
python app.py                     # http://localhost:8000
```

Environment: `LOTTOMAX_CORS_ORIGINS` (comma-separated, default `http://localhost:5173,http://127.0.0.1:5173`).

### 2. Frontend

```bash
cd fe
npm ci
npm run dev                       # http://localhost:5173
```

### 3. Use the app

1. **Settings → Data update** (labelled "데이터 업데이트" in the UI): paste new official results (`draw number,date,7 numbers,bonus`, one per line, e.g. `1274,2026-09-29,3,9,14,22,31,40,47,12`). The CSV is backed up to `data/backup/` and replaced atomically.
2. **Generate** → *Train LSTM Model* (~1 min) → *Generate Numbers*.
3. **Signal Lab** → run the falsifiable deep-learning check (quick mode ≈ 3 s, full ≈ 1 min).
4. **EV** → draw Smart Pick v2 tickets and estimate jackpot EV (enter the real jackpot / sales).

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health, server info, `lstm_verdict` |
| GET | `/status` | Training progress, logs, `lstm_verdict` |
| POST | `/train` | Train LSTM (`{"epochs":100,"run_seed_analysis":false}`) |
| POST | `/predict` | `{"mode":"ensemble"\|"smart_v2","weights":{...}}` |
| POST | `/predict-batch` | `{"count":1..10}` Smart Pick v2 tickets, pairwise overlap ≤ 3 |
| POST | `/history-check` | `{"numbers":[7]}` exact-match / overlap histogram vs all past draws |
| POST | `/ev/estimate` | Jackpot EV with Poisson co-winner sharing (assumption-based) |
| GET | `/frequencies` | Number frequency analysis |
| POST | `/backtest` | Walk-forward backtest vs random ticket (pool-size aware) |
| POST | `/signal-lab/run` | `{"seed":42,"quick":false}` background job |
| GET | `/signal-lab/status` | `{running, has_result, progress}` |
| GET | `/signal-lab/result` | Full result JSON |
| GET | `/data/status` | Last draw, days since, estimated missing draws |
| POST | `/data/append` | Validate + append pasted draws (partial adds allowed, invalidates backtest/Signal Lab) |
| POST | `/seed-analysis` | PRNG seed scan (transparency demo — no predictive power) |
| POST | `/reload-data` | Reload CSV from disk |

## Signal Lab

Designed so a "discovery" cannot be an overfit accident:

1. **Battery** — 109 tests (uniformity, lag 1–5 overlap, pair co-occurrence, odd count, sum KS, per-number runs, per-number frequency) with **Benjamini–Hochberg** correction over all p-values.
2. **Power check** — plants known effects in simulated uniform data and confirms the battery finds them; otherwise the verdict is "detector not trustworthy" ("탐지기 신뢰 불가" in the Korean UI). Also reports the false-positive rate on pure noise.
3. **Predictor** — walk-forward expanding window, small MLP (TF) or NumPy logistic regression; judged by log-loss vs the constant 7/n baseline with a bootstrap CI and sign-permutation test. `beats_baseline` requires CI upper < 0.
4. **Discriminator** — real vs random next draw, time-ordered split, AUC with bootstrap CI and permutation p-value.

Current result on the real history: **no learnable signal** ([RESEARCH.md §5](docs/RESEARCH.md)).

## LSTM Architecture

```
Input: 20 consecutive draws (multi-hot, 52 numbers)
  ↓
LSTM (32 units, L2) → Dropout(0.3)
  ↓
Dense(32, ReLU, L2) → Dropout(0.2)
  ↓
Dense(52, Sigmoid) → per-number probability      (~14k parameters, < 50k enforced)
```

After training, validation BCE is compared with the constant-rate baseline `-(p·ln p + (1-p)·ln(1-p))`; the verdict is saved next to the model and shown in the UI. Weights are never changed automatically.

## Running Tests

```bash
cd be
./venv/bin/pip install -r requirements-dev.txt
./venv/bin/python -m pytest tests -q          # passes with or without TensorFlow

cd ../fe
npm run lint && npm run build
npm run e2e        # needs backend + `vite preview --port 4173`, see e2e/smoke.mjs
```

CI (`.github/workflows/ci.yml`) runs both on every push.

## Updating Data

Use **Settings → Data update** (or `POST /data/append`). Rows are validated (7 distinct numbers in the pool valid on that date, bonus not among them, increasing draw number and date, no duplicates); accepted rows are appended after a timestamped backup in `data/backup/`. Then retrain the LSTM. Nothing is scraped automatically.

---

*For entertainment purposes. Lottery outcomes are not guaranteed.*
