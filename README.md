# 🎰 LottoMax AI — Signal Lab · Smart Pick v2 · EV Calculator (v5.0)

A LottoMax toolkit with an LSTM/statistical ensemble, a **falsifiable deep-learning "Signal Lab"**, an **EV-optimizing Smart Pick v2**, a jackpot **expected-value calculator**, and a safe **data-ingest** flow.

> **정직 고지 / Honesty note** — 모든 번호 조합의 당첨 확률은 동일합니다. 리서치 결과([docs/RESEARCH.md](docs/RESEARCH.md)), 추첨 데이터는 모든 무작위성 검정을 통과했고 딥러닝 예측기·판별기·LSTM 어느 것도 상수 확률을 넘어서지 못했으며, 어떤 전략도 랜덤 티켓(기대 매치 7×7/N)을 이기지 못했습니다. 이 앱이 실제로 바꿀 수 있는 것은 **당첨 시 분배금 기대값(공동 당첨자 수)** 뿐이고, 그 효과도 *가정 기반 추정*입니다. **Signal Lab** 탭에서 "학습 가능한 신호가 없다"는 사실을 직접 검증할 수 있습니다.

## Game rules covered

| 구간 | 기간 | 규칙 |
|---|---|---|
| Era 1 | 2009-09-25 ~ 2019-05-10 | 7/49, 주 1회 |
| Era 2 | 2019-05-14 ~ 2026-04-10 | 7/50, 주 2회 (화·금) |
| **Era 3 (현재)** | **2026-04-14 ~** | **7/52, 주 2회, $6 / 4줄** (조합 수 C(52,7) = 133,784,560) |

통계 전략과 LSTM은 Era 2+3 (2019-05-14~)만 사용합니다. 51·52는 표본이 약 50회뿐이라 통계 전략에서 중립 점수로 취급합니다. Signal Lab은 풀이 고정된 Era 2만 사용합니다. (변경 근거: [RESEARCH.md §5.1](docs/RESEARCH.md))

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

1. **Settings → 데이터 업데이트**: paste new official results (`회차,날짜,번호1~7,보너스`, one per line). The CSV is backed up to `data/backup/` and replaced atomically.
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
2. **Power check** — plants known effects in simulated uniform data and confirms the battery finds them; otherwise the verdict is "탐지기 신뢰 불가". Also reports the false-positive rate on pure noise.
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

Use **Settings → 데이터 업데이트** (or `POST /data/append`). Rows are validated (7 distinct numbers in the pool valid on that date, bonus not among them, increasing draw number and date, no duplicates); accepted rows are appended after a timestamped backup in `data/backup/`. Then retrain the LSTM. Nothing is scraped automatically.

---

*For entertainment purposes. Lottery outcomes are not guaranteed.*
