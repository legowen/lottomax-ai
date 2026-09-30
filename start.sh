#!/bin/bash
# ============================================================
# LottoMax AI - local one-click launcher (Git Bash / Linux / macOS)
# Starts backend (127.0.0.1:$LOTTOMAX_PORT) + frontend (localhost:5173),
# opens the browser, and stops both when you press Ctrl+C / close the window.
# ============================================================
ROOT="$(cd "$(dirname "$0")" && pwd)"
BE_PORT="${LOTTOMAX_PORT:-8000}"
FE_PORT=5173
LOG_DIR="$ROOT/.launcher-logs"
mkdir -p "$LOG_DIR"
BE_LOG="$LOG_DIR/backend.log"
FE_LOG="$LOG_DIR/frontend.log"

BE_PID=""
FE_PID=""
CLEANED=0

fail() {
  echo ""
  echo "[ERROR] $1"
  [ -n "$2" ] && [ -f "$2" ] && { echo "---- last lines of $2 ----"; tail -n 25 "$2"; echo "--------------------------------"; }
  cleanup
  exit 1
}

is_windows() {
  case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) return 0 ;; *) return 1 ;; esac
}

# Kill a process and all its descendants (best effort, portable).
kill_tree() {
  local pid="$1" child
  [ -z "$pid" ] && return
  if command -v pgrep >/dev/null 2>&1; then
    for child in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$child"; done
  fi
  kill "$pid" 2>/dev/null
}

# Windows: node/python children are not always killed by bash `kill`; kill whatever still listens on the port.
kill_port_windows() {
  local port="$1" pid
  is_windows || return
  for pid in $(netstat -ano 2>/dev/null | tr -d '\r' | awk -v p=":$port" '$2 ~ p"$" && $4=="LISTENING" {print $5}' | sort -u); do
    taskkill //F //T //PID "$pid" >/dev/null 2>&1
  done
}

cleanup() {
  [ "$CLEANED" = 1 ] && return
  CLEANED=1
  trap - INT TERM EXIT HUP
  echo ""
  echo "Stopping LottoMax AI..."
  kill_tree "$FE_PID"
  kill_tree "$BE_PID"
  # Only touch ports of servers THIS script started (never another running instance).
  [ -n "$FE_PID" ] && kill_port_windows "$FE_PORT"
  [ -n "$BE_PID" ] && kill_port_windows "$BE_PORT"
  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    { [ -n "$BE_PID" ] && port_busy "$BE_PORT"; } || { [ -n "$FE_PID" ] && port_busy "$FE_PORT"; } || break
    sleep 0.5
  done
  echo "Stopped."
}
trap 'cleanup; exit 130' INT TERM HUP
trap cleanup EXIT

# ---- find tools --------------------------------------------------
PY_SYS=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys" >/dev/null 2>&1; then PY_SYS="$c"; break; fi
done
[ -z "$PY_SYS" ] && fail "Python 3 not found. Install Python 3.10-3.12 first."
command -v node >/dev/null 2>&1 || fail "Node.js not found. Install Node.js 18+ first."
command -v npm  >/dev/null 2>&1 || fail "npm not found. Install Node.js 18+ first."

port_busy() {
  "$PY_SYS" - "$1" <<'PYEOF'
import socket, sys
s = socket.socket()
s.settimeout(0.5)
busy = s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0
s.close()
sys.exit(0 if busy else 1)
PYEOF
}

http_ok() {
  "$PY_SYS" - "$1" <<'PYEOF'
import sys, urllib.request
try:
    urllib.request.urlopen(sys.argv[1], timeout=2)
except Exception:
    sys.exit(1)
PYEOF
}

# ---- port check (before doing anything heavy) -------------------
for p in "$BE_PORT" "$FE_PORT"; do
  if port_busy "$p"; then
    fail "Port $p is already in use. Close the program using it (or another LottoMax window) and try again. See README 'Running locally'."
  fi
done

# ---- first-time setup -------------------------------------------
if [ -x "$ROOT/be/venv/bin/python" ]; then
  VENV_PY="$ROOT/be/venv/bin/python"
elif [ -x "$ROOT/be/venv/Scripts/python.exe" ]; then
  VENV_PY="$ROOT/be/venv/Scripts/python.exe"
else
  echo "[setup] Creating be/venv and installing requirements (first run only; TensorFlow is large)..."
  "$PY_SYS" -m venv "$ROOT/be/venv" || fail "Could not create the virtual environment."
  if [ -x "$ROOT/be/venv/bin/python" ]; then VENV_PY="$ROOT/be/venv/bin/python"; else VENV_PY="$ROOT/be/venv/Scripts/python.exe"; fi
  "$VENV_PY" -m pip install -r "$ROOT/be/requirements.txt" || fail "pip install failed (TensorFlow needs Python 3.10-3.12)."
fi

if [ ! -d "$ROOT/fe/node_modules" ]; then
  echo "[setup] Running npm install in fe/ (first run only)..."
  (cd "$ROOT/fe" && npm install) || fail "npm install failed."
fi

# ---- backend ----------------------------------------------------
echo "[1/2] Starting backend on http://127.0.0.1:$BE_PORT ..."
(cd "$ROOT/be" && LOTTOMAX_PORT="$BE_PORT" PYTHONUNBUFFERED=1 exec "$VENV_PY" app.py) >"$BE_LOG" 2>&1 &
BE_PID=$!

ok=0
for i in $(seq 1 60); do
  if http_ok "http://127.0.0.1:$BE_PORT/"; then ok=1; break; fi
  if ! kill -0 "$BE_PID" 2>/dev/null; then
    fail "Backend exited during startup." "$BE_LOG"
  fi
  sleep 1
done
[ "$ok" = 1 ] || fail "Backend did not respond on http://127.0.0.1:$BE_PORT/ within 60 seconds." "$BE_LOG"
echo "      backend is up."

# ---- frontend ---------------------------------------------------
echo "[2/2] Starting frontend on http://localhost:$FE_PORT ..."
(cd "$ROOT/fe" && VITE_API_URL="http://localhost:$BE_PORT" exec npm run dev) >"$FE_LOG" 2>&1 &
FE_PID=$!

ok=0
for i in $(seq 1 60); do
  if http_ok "http://127.0.0.1:$FE_PORT/" || http_ok "http://localhost:$FE_PORT/"; then ok=1; break; fi
  if ! kill -0 "$FE_PID" 2>/dev/null; then
    fail "Frontend exited during startup." "$FE_LOG"
  fi
  sleep 1
done
[ "$ok" = 1 ] || fail "Frontend did not respond on http://localhost:$FE_PORT within 60 seconds." "$FE_LOG"
echo "      frontend is up."

URL="http://localhost:$FE_PORT"
echo ""
echo "LottoMax AI is running: $URL"
echo "Press Ctrl+C (or close this window) to stop both servers."
if [ -z "$LOTTOMAX_NO_BROWSER" ]; then
  if   command -v cygstart  >/dev/null 2>&1; then cygstart "$URL"
  elif is_windows;                            then start "" "$URL" 2>/dev/null || cmd.exe //c start "" "$URL"
  elif command -v xdg-open  >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1
  elif command -v open      >/dev/null 2>&1; then open "$URL"
  fi
fi

# ---- wait until either server dies or the user stops us ----------
while kill -0 "$BE_PID" 2>/dev/null && kill -0 "$FE_PID" 2>/dev/null; do
  sleep 1
done
if ! kill -0 "$BE_PID" 2>/dev/null; then fail "Backend stopped unexpectedly." "$BE_LOG"; fi
fail "Frontend stopped unexpectedly." "$FE_LOG"
