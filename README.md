# 🎰 LottoMax AI — Typical-Set Generator · Signal Lab · Smart Pick v2 · EV Calculator (v5.1)

A LottoMax toolkit with a **typical-set ticket generator** (tickets that look like real draws), an LSTM/statistical ensemble (experimental), a **falsifiable deep-learning "Signal Lab"**, an **EV-optimizing Smart Pick v2**, a jackpot **expected-value calculator**, and a safe **data-ingest** flow.

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
│  generator.py    typical-set sampling (realistic,     │
│                  balanced), exact sum distribution    │
│  data_ingest.py  validate → backup → atomic CSV append│
│  backtest.py     walk-forward vs random ticket        │
└──────────────────────────────────────────────────────┘
```

## Generation modes (`POST /predict`)

| Mode | What it does | Note |
|---|---|---|
| `realistic` (UI default) | Uniform random ticket from the **typical set**: sum inside the central 90% of its exact distribution (124–247 for 7/52), 2–5 odd numbers, numbers in at least 4 of the 10-wide groups (1–10, 11–20, …) | Looks like a real draw; win probability is unchanged |
| `balanced` | Typical set **+ EV guard** (≤ 4 numbers ≤ 31, no 3-run, never a past winner), then a random pick among the least popular 25% | Fewer co-winners *if* the popularity assumptions hold |
| `smart_v2` | Random pick among the least popular 10% of guard-passing candidates | See Smart Pick v2 below |
| `ensemble` (API default) | Legacy: top 7 numbers by the weighted 7-strategy score | Experimental — repeats and clusters (reported as NOT TYPICAL) |

Every mode picks at random except `ensemble`, which ranks scores; **none of them can raise the chance of winning** — each combination has probability 1 / C(52, 7). The API default stays `ensemble` for backward compatibility; the UI sends `realistic` unless you pick another mode. Every `/predict` response carries `main.typicality` (sum, odd count, occupied decade groups, typical yes/no). Details and tests: [RESEARCH.md Stage 6](docs/RESEARCH.md).

## Strategies (ensemble, `mode: "ensemble"`)

> **Experimental.** The ensemble scores every number and takes the top 7, so it repeats the same tickets and piles onto one region of the pool (see [RESEARCH.md Stage 6](docs/RESEARCH.md)). Use `realistic` / `balanced` for tickets that look like real draws.

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
2. **Generate** → pick a mode (default *REALISTIC*) → *Generate Numbers*. Only the *ENSEMBLE* mode uses the LSTM: select it and press *Train LSTM Model* (~1 min) first.
3. **Signal Lab** → run the falsifiable deep-learning check (quick mode ≈ 3 s, full ≈ 1 min).
4. **EV** → draw Smart Pick v2 tickets and estimate jackpot EV (enter the real jackpot / sales).

## 로컬에서 실행하기 (Running locally)

이 앱은 **내 PC에서만** 쓰도록 만들어졌습니다. 백엔드는 `127.0.0.1`, 프론트엔드는 `localhost`에만 열리므로 같은 네트워크의 다른 기기나 외부에서 접속할 수 없습니다. **원격 접속은 지원하지 않습니다** (배포용이 아닙니다).

### 처음 한 번만
- **Python 3.10~3.12** (TensorFlow 때문에 3.13 이상은 안 됩니다)와 **Node.js 18+** 를 설치합니다.
- 별도 설정은 필요 없습니다. `start.bat`(또는 `start.sh`)이 처음 실행될 때 `be/venv`를 만들고 `requirements.txt`를 설치하며(TensorFlow 때문에 몇 분 걸림), `fe/node_modules`가 없으면 `npm install`도 자동으로 실행합니다.

### 실행 방법
- **Windows**: 저장소 루트의 `start.bat`을 더블클릭합니다.
- **Git Bash**: `./start.sh`
- 백엔드가 응답할 때까지(최대 60초) 기다린 뒤 프론트엔드를 켜고 기본 브라우저로 `http://localhost:5173`을 엽니다.
- 종료: 창을 닫거나 Ctrl+C (`start.bat`은 아무 키). 두 서버가 모두 함께 종료됩니다.
- 문제가 생기면 창에 원인이 출력되고, 전체 로그는 `.launcher-logs/`에 저장됩니다.
- 백엔드 포트를 바꾸려면 환경변수 `LOTTOMAX_PORT`를 지정합니다 (기본 8000). 프론트엔드 포트는 5173 고정입니다.

### 바탕화면 바로가기 만들기
1. 탐색기에서 `start.bat`을 우클릭 → **보내기 → 바탕 화면에 바로 가기 만들기**
2. 바로가기를 우클릭 → **속성**에서 이름을 "LottoMax AI"로 바꾸고, 필요하면 **아이콘 변경**을 선택합니다.
3. 바로가기 대신 파일을 옮기지 마세요. `start.bat`은 자기 위치 기준으로 동작하므로 저장소 폴더 안에 있어야 합니다.

### 포트 충돌 해결
8000 또는 5173 포트가 이미 사용 중이면 어느 포트가 문제인지 출력하고 멈춥니다.
- 이미 켜져 있는 LottoMax 창이 있으면 먼저 닫습니다.
- 그래도 사용 중이면 PowerShell/명령 프롬프트에서 `netstat -ano | findstr :8000` 으로 PID를 찾고 `taskkill /F /PID <PID>` 로 종료합니다 (5173도 동일).
- 8000만 계속 쓸 수 없다면 `set LOTTOMAX_PORT=8010` 후 실행합니다.

### CSV 데이터를 업데이트한 뒤에는
`data/LOTTOMAX.csv`를 수정하거나 Settings → 데이터 업데이트로 회차를 추가한 뒤에는 **Generate 탭에서 ENSEMBLE 모드를 고르고 Train 버튼을 다시 눌러 LSTM을 재학습**하세요. (이전 모델은 예전 데이터로 학습되어 있습니다. REALISTIC · BALANCED · SMART V2 모드는 LSTM을 쓰지 않으므로 재학습이 필요 없습니다.)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health, server info, `lstm_verdict` |
| GET | `/status` | Training progress, logs, `lstm_verdict` |
| POST | `/train` | Train LSTM (`{"epochs":100,"run_seed_analysis":false}`) |
| POST | `/predict` | `{"mode":"realistic"\|"balanced"\|"smart_v2"\|"ensemble","weights":{...}}` — `weights` only affect `ensemble`; the response has `main.typicality` |
| POST | `/predict-batch` | `{"count":1..10,"mode":"smart_v2"\|"realistic"\|"balanced"}` tickets, pairwise overlap ≤ 3 |
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

Use **Settings → Data update** (or `POST /data/append`). Rows are validated (7 distinct numbers in the pool valid on that date, bonus not among them, increasing draw number and date, no duplicates); accepted rows are appended after a timestamped backup in `data/backup/`. Then retrain the LSTM (only the ENSEMBLE mode uses it). Nothing is scraped automatically.

---

*For entertainment purposes. Lottery outcomes are not guaranteed.*
