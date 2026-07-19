# 🎰 LottoMax AI - LSTM Ensemble Prediction Engine

A deep learning-powered lottery number prediction app using a **7-strategy ensemble** approach with a real **LSTM neural network** and an **EV-optimizing Smart Pick** at its core.

> **정직 고지 / Honesty note** — 리서치 결과([docs/RESEARCH.md](docs/RESEARCH.md)), 추첨 데이터는 모든 무작위성 검정을 통과했고 어떤 전략도 랜덤 티켓(기대 매치 0.98개)을 이기지 못했습니다. 이 앱이 실제로 개선하는 것은 **당첨 시 분배금 기대값**입니다: LottoMax 잭팟은 당첨자끼리 나누므로, 남들이 안 고르는 조합(Smart Pick)을 고르면 같은 확률로 더 큰 몫을 받습니다. 앱의 **Backtest 탭**에서 이 사실을 직접 검증할 수 있습니다.

## Architecture

```
┌─────────────────────────────────────────────┐
│              React Frontend (:5173)          │
│  Casino-style UI • Ball animations • Charts │
└──────────────────────┬──────────────────────┘
                       │ API calls
┌──────────────────────┴──────────────────────┐
│           FastAPI Backend (:8000)            │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │     7-Strategy Ensemble Engine      │    │
│  │                                     │    │
│  │  1. LSTM Neural Network (15%)       │    │
│  │  2. Frequency + Recency (15%)       │    │
│  │  3. Gap Analysis (20%)              │    │
│  │  4. Pair Correlation (5%)           │    │
│  │  5. Distribution Balance (15%)      │    │
│  │  6. Seed/RNG Analysis (0%)          │    │
│  │     transparency only, no power     │    │
│  │  7. Smart Pick / EV (30%)           │    │
│  │     unpopular-combo optimizer       │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  CSV Data → Train → Predict → API Response  │
│        └→ Walk-forward Backtest (honesty)   │
└─────────────────────────────────────────────┘
```

**Era-aware statistics**: LottoMax switched from 7/49 (weekly) to 7/50 (twice a week) on **2019-05-14**. All statistics and LSTM training use only the 7/50-era draws — mixing eras would make number 50 look artificially rare (99 vs ~171 appearances).

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**

### 1. Backend Setup

```bash
cd be

# Create virtual environment
python3 -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Start server
python app.py
```

The backend runs at `http://localhost:8000`.

### 2. Frontend Setup

```bash
cd fe

# If fresh project, create with Vite:
npm create vite@latest . -- --template react
# When prompted, select React → JavaScript

# Install dependencies
npm install

# Copy LottoMaxAI.jsx to src/ and update App.jsx:
# Replace the content of src/App.jsx with:
#   import LottoMaxAI from './LottoMaxAI'
#   export default function App() { return <LottoMaxAI /> }

# Start dev server
npm run dev
```

The frontend runs at `http://localhost:5173`.

### 3. Data Files

Place your CSV files in the `data/` folder:
- `data/LOTTOMAX.csv` — Main draw history
- `data/LOTTOMAXExtra.csv` — Extra draw history

### 4. Use the App

1. Open `http://localhost:5173`
2. Click **"Train LSTM Model"** — trains the neural network (~1-3 min)
3. Click **"Generate Numbers"** — get ensemble predictions
4. Check **Analysis** tab for frequency charts and statistics
5. Adjust **Settings** for strategy weights and training epochs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check & server info |
| GET | `/status` | Training progress & logs |
| POST | `/train` | Start LSTM training (`{ "epochs": 100 }`) |
| POST | `/predict` | Get prediction (`{ "weights": {...} }`) |
| GET | `/frequencies` | Number frequency analysis (7/50 era only) |
| POST | `/backtest` | Walk-forward strategy backtest vs random baseline (`{ "window": 150 }`) |
| POST | `/seed-analysis` | PRNG seed scan (transparency demo — no predictive power) |
| POST | `/reload-data` | Reload CSV files |

## LSTM Architecture

```
Input: 20 consecutive draws (multi-hot encoded)
  ↓
Bidirectional LSTM (128 units) → BatchNorm → Dropout(0.3)
  ↓
Bidirectional LSTM (64 units) → BatchNorm → Dropout(0.3)
  ↓
LSTM (64 units) → BatchNorm → Dropout(0.3)
  ↓
Dense(128, ReLU) → Dropout(0.2)
  ↓
Dense(64, ReLU)
  ↓
Dense(50, Sigmoid) → Per-number probability
```

The model learns from sequences of 20 draws, predicting which numbers are likely to appear next based on the patterns it detects in the historical data.

## How It Works

Each strategy independently scores every number (1–50). Scores are normalized to 0–1, then combined using weighted voting. When Smart Pick is active, the final selection also passes an **EV guard**: at most 4 numbers ≤ 31 (birthday tickets), no 3+ consecutive runs, and never an exact past winning combination — shapes that many humans play and that would split the prize.

**What the research showed** ([docs/RESEARCH.md](docs/RESEARCH.md)):
- The draw history passes chi-square uniformity, serial-correlation, pair, odd/even and sum tests in both eras → nothing to learn
- Walk-forward backtest: no strategy beats the 0.98 expected matches of a random ticket; pair correlation is significantly *worse* (it chases noise)
- Seed/RNG reconstruction can't work: draws run on certified, audited draw systems, not wall-clock-seeded PRNGs
- The one real lever is **expected value**: unpopular combinations win just as often but share the pari-mutuel pool with fewer people

## Running Tests

```bash
cd be
./venv/bin/pip install pytest httpx
./venv/bin/python -m pytest tests/ -q
```

## Updating Data

1. Download latest draw data
2. Replace CSV files in `data/`
3. Click **"Reload CSV Data"** in Settings tab
4. Click **"Retrain Model"** to update the LSTM

---

*For entertainment purposes. Lottery outcomes are not guaranteed.*
