#!/bin/bash
# ============================================================
# LottoMax AI - Local Setup & Run Script
# ============================================================

set -e

echo "🎰 LottoMax AI - Setup"
echo "======================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✅ Python $PYTHON_VERSION detected"

# Check Node
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Install Node.js 18+ first."
    exit 1
fi

NODE_VERSION=$(node -v)
echo "✅ Node.js $NODE_VERSION detected"

# ============================================================
# Backend Setup
# ============================================================
echo ""
echo "📦 Setting up Backend..."

cd "$(dirname "$0")/backend"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo "  Installing Python dependencies (this may take a few minutes for TensorFlow)..."
pip install -q -r requirements.txt

echo "✅ Backend ready"

cd ..

# ============================================================
# Frontend Setup
# ============================================================
echo ""
echo "📦 Setting up Frontend..."

cd frontend

if [ ! -d "node_modules" ]; then
    # Create a simple Vite + React project if not exists
    if [ ! -f "package.json" ]; then
        echo "  Initializing React project..."
        npm create vite@latest . -- --template react 2>/dev/null || true
    fi
    echo "  Installing npm dependencies..."
    npm install
fi

echo "✅ Frontend ready"

cd ..

echo ""
echo "============================================"
echo "🎯 Setup Complete!"
echo "============================================"
echo ""
echo "To start the app, run:"
echo ""
echo "  Terminal 1 (Backend):"
echo "    cd backend"
echo "    source venv/bin/activate"
echo "    python app.py"
echo ""
echo "  Terminal 2 (Frontend):"
echo "    cd frontend"
echo "    npm run dev"
echo ""
echo "Then open http://localhost:5173 in your browser"
echo ""
