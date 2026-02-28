# 🎰 LottoMax AI - LSTM Ensemble Prediction Engine

A deep learning-powered lottery number prediction app using a **5-strategy ensemble** approach with a real **LSTM neural network** at its core.

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
│  │     5-Strategy Ensemble Engine      │    │
│  │                                     │    │
│  │  1. LSTM Neural Network (30%)       │    │
│  │     Bidirectional LSTM, 3 layers    │    │
│  │     Sequence pattern detection      │    │
│  │                                     │    │
│  │  2. Frequency + Recency (20%)       │    │
│  │     Hot/cold with exponential decay │    │
│  │                                     │    │
│  │  3. Gap Analysis (20%)              │    │
│  │     Overdue number detection        │    │
│  │                                     │    │
│  │  4. Pair Correlation (15%)          │    │
│  │     Co-occurrence patterns          │    │
│  │                                     │    │
│  │  5. Distribution Balance (15%)      │    │
│  │     Range & odd/even equilibrium    │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  CSV Data → Train → Predict → API Response  │
└─────────────────────────────────────────────┘
```

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
| GET | `/frequencies` | Number frequency analysis |
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

Each strategy independently scores every number (1-50 for main, 1-99 for extra). Scores are normalized to 0-1, then combined using weighted voting. The top-scoring numbers become the prediction.

**Why this might work if RNG isn't truly random:**
- LSTM captures hidden sequential dependencies in the number generation
- Gap analysis detects cyclical patterns in individual numbers
- Pair correlation finds numbers that the generator tends to output together
- Distribution analysis catches bias toward certain ranges

## Updating Data

1. Download latest draw data
2. Replace CSV files in `data/`
3. Click **"Reload CSV Data"** in Settings tab
4. Click **"Retrain Model"** to update the LSTM

---

*For entertainment purposes. Lottery outcomes are not guaranteed.*
