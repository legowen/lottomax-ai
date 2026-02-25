import { useState, useEffect, useCallback, useRef } from "react";

// ============================================================
// LottoMax AI - Frontend
// Connects to FastAPI backend with real LSTM ensemble
// ============================================================

const API = "http://localhost:8000";

// Ball colors by number range
const getBallColor = (num, isExtra = false) => {
  if (isExtra) return { bg: "linear-gradient(135deg, #f59e0b, #d97706)", text: "#fff", glow: "rgba(245,158,11,0.5)" };
  if (num <= 10) return { bg: "linear-gradient(135deg, #ef4444, #dc2626)", text: "#fff", glow: "rgba(239,68,68,0.5)" };
  if (num <= 20) return { bg: "linear-gradient(135deg, #3b82f6, #2563eb)", text: "#fff", glow: "rgba(59,130,246,0.5)" };
  if (num <= 30) return { bg: "linear-gradient(135deg, #a855f7, #9333ea)", text: "#fff", glow: "rgba(168,85,247,0.5)" };
  if (num <= 40) return { bg: "linear-gradient(135deg, #22c55e, #16a34a)", text: "#fff", glow: "rgba(34,197,94,0.5)" };
  return { bg: "linear-gradient(135deg, #f97316, #ea580c)", text: "#fff", glow: "rgba(249,115,22,0.5)" };
};

// ============================================================
// Components
// ============================================================
function LottoBall({ number, delay = 0, isExtra = false, isRevealed = true, size = "lg" }) {
  const [revealed, setRevealed] = useState(false);
  const color = getBallColor(number, isExtra);

  useEffect(() => {
    if (isRevealed) {
      const timer = setTimeout(() => setRevealed(true), delay);
      return () => clearTimeout(timer);
    }
    setRevealed(false);
  }, [isRevealed, delay]);

  const sizeMap = { lg: "w-16 h-16 text-2xl", md: "w-12 h-12 text-lg", sm: "w-9 h-9 text-sm" };

  return (
    <div
      className={`${sizeMap[size]} rounded-full flex items-center justify-center font-black
        transition-all duration-700 ease-out select-none relative`}
      style={{
        background: revealed ? color.bg : "linear-gradient(135deg, #374151, #1f2937)",
        color: revealed ? color.text : "#6b7280",
        transform: revealed ? "scale(1) rotateY(0deg)" : "scale(0.6) rotateY(180deg)",
        opacity: revealed ? 1 : 0.4,
        boxShadow: revealed ? `0 0 20px ${color.glow}, inset 0 -3px 6px rgba(0,0,0,0.3)` : "none",
      }}
    >
      {revealed && (
        <div className="absolute inset-0 rounded-full"
          style={{ background: "radial-gradient(circle at 35% 30%, rgba(255,255,255,0.4) 0%, transparent 60%)" }} />
      )}
      <span className="relative z-10">{revealed ? number : "?"}</span>
    </div>
  );
}

function StrategyBar({ strategies, number }) {
  if (!strategies || !strategies[String(number)]) return null;
  const s = strategies[String(number)];
  const items = [
    { key: "LSTM", val: s.lstm, color: "#ef4444" },
    { key: "Freq", val: s.frequency, color: "#3b82f6" },
    { key: "Gap", val: s.gap, color: "#a855f7" },
    { key: "Pair", val: s.pair, color: "#22c55e" },
    { key: "Dist", val: s.distribution, color: "#f97316" },
  ];

  return (
    <div className="flex gap-0.5 items-end h-8 mt-1">
      {items.map((item) => (
        <div
          key={item.key}
          title={`${item.key}: ${(item.val * 100).toFixed(0)}%`}
          className="w-2 rounded-t transition-all duration-500"
          style={{ height: `${Math.max(2, item.val * 32)}px`, backgroundColor: item.color, opacity: 0.8 }}
        />
      ))}
    </div>
  );
}

function TrainingProgress({ progress, log }) {
  const logRef = useRef(null);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log]);

  return (
    <div className="bg-black/40 border border-white/10 rounded-xl overflow-hidden">
      {/* Progress bar */}
      {progress.status === "training" && (
        <div className="px-4 py-3 border-b border-white/5">
          <div className="flex justify-between text-xs text-gray-400 mb-2">
            <span>{progress.strategy}</span>
            <span>Epoch {progress.epoch}/{progress.total_epochs} • Loss: {progress.loss}</span>
          </div>
          <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{
                width: `${(progress.epoch / progress.total_epochs) * 100}%`,
                background: "linear-gradient(90deg, #3b82f6, #a855f7)",
              }}
            />
          </div>
        </div>
      )}

      {/* Log */}
      <div ref={logRef} className="p-3 font-mono text-xs max-h-56 overflow-y-auto">
        {log.map((entry, i) => (
          <div key={i} className="text-gray-500 py-0.5">
            <span className="text-gray-700 mr-2">{entry.time}</span>
            {entry.msg}
          </div>
        ))}
        {log.length === 0 && <span className="text-gray-700">Waiting for training...</span>}
      </div>
    </div>
  );
}

function FrequencyChart({ data, numRange, title }) {
  if (!data) return null;
  const maxFreq = Math.max(...Object.values(data));

  return (
    <div className="mt-6">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">{title}</h3>
      <div className="flex items-end gap-px h-24 overflow-x-auto pb-1">
        {Array.from({ length: numRange }, (_, i) => i + 1).map((n) => {
          const freq = data[String(n)] || 0;
          return (
            <div key={n} className="flex flex-col items-center flex-shrink-0" style={{ width: numRange > 50 ? "8px" : "14px" }}>
              <div
                className="w-full rounded-t transition-all duration-300"
                style={{
                  height: `${maxFreq > 0 ? (freq / maxFreq) * 80 : 0}px`,
                  background: getBallColor(n, numRange > 50).bg,
                  opacity: freq > 0 ? 0.7 : 0.15,
                }}
                title={`#${n}: ${freq} times`}
              />
              {numRange <= 50 && n % 5 === 0 && (
                <span className="text-gray-500 mt-1" style={{ fontSize: "8px" }}>{n}</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ============================================================
// Main App
// ============================================================
export default function LottoMaxAI() {
  const [connected, setConnected] = useState(false);
  const [serverInfo, setServerInfo] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState({ status: "idle" });
  const [trainingLog, setTrainingLog] = useState([]);
  const [modelReady, setModelReady] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [showStrategies, setShowStrategies] = useState(false);
  const [frequencies, setFrequencies] = useState(null);
  const [activeTab, setActiveTab] = useState("generate");
  const [epochs, setEpochs] = useState(100);
  const [weights, setWeights] = useState({
    lstm: 0.30, frequency: 0.20, gap: 0.20, pair: 0.15, distribution: 0.15,
  });

  const pollRef = useRef(null);

  // Check server connection
  const checkServer = useCallback(async () => {
    try {
      const res = await fetch(`${API}/`);
      const data = await res.json();
      setConnected(true);
      setServerInfo(data);
      setModelReady(data.main_model_loaded);
      return true;
    } catch {
      setConnected(false);
      return false;
    }
  }, []);

  useEffect(() => {
    checkServer();
    const interval = setInterval(checkServer, 10000);
    return () => clearInterval(interval);
  }, [checkServer]);

  // Poll training status
  const pollTraining = useCallback(async () => {
    try {
      const res = await fetch(`${API}/status`);
      const data = await res.json();
      setIsTraining(data.is_training);
      setTrainingProgress(data.progress);
      setTrainingLog(data.log);
      setModelReady(data.main_model_ready);

      if (!data.is_training && pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    } catch {
      // ignore
    }
  }, []);

  // Train model
  const startTraining = async () => {
    try {
      const res = await fetch(`${API}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ epochs }),
      });
      if (!res.ok) {
        const err = await res.json();
        alert(err.detail || "Training failed");
        return;
      }
      setIsTraining(true);
      setTrainingLog([]);
      // Start polling
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(pollTraining, 1000);
    } catch (e) {
      alert("Cannot connect to server");
    }
  };

  // Generate prediction
  const generate = async () => {
    setIsGenerating(true);
    setPrediction(null);

    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ weights }),
      });
      const data = await res.json();
      setPrediction(data);

      setHistory((prev) => [
        {
          id: Date.now(),
          main: data.main.numbers,
          extra: data.extra?.numbers || null,
          confidence: data.main.confidence,
          modelTrained: data.model_trained,
          time: new Date().toLocaleTimeString(),
        },
        ...prev.slice(0, 9),
      ]);
    } catch (e) {
      alert("Prediction failed: " + e.message);
    }

    setIsGenerating(false);
  };

  // Load frequencies
  const loadFrequencies = async () => {
    try {
      const res = await fetch(`${API}/frequencies`);
      const data = await res.json();
      setFrequencies(data);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    if (activeTab === "analysis" && connected) loadFrequencies();
  }, [activeTab, connected]);

  return (
    <div
      className="min-h-screen text-white"
      style={{
        background: "linear-gradient(160deg, #0a0a0f 0%, #0d1117 30%, #101820 60%, #0a0a0f 100%)",
        fontFamily: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace",
      }}
    >
      {/* Background grid */}
      <div className="fixed inset-0 opacity-5 pointer-events-none"
        style={{ backgroundImage: "radial-gradient(circle, #ffffff 1px, transparent 1px)", backgroundSize: "30px 30px" }} />

      <div className="relative max-w-4xl mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-8">
          <div className="inline-flex items-center gap-3 mb-2">
            <div className={`w-3 h-3 rounded-full ${connected ? "bg-green-500" : "bg-red-500"} animate-pulse`} />
            <h1 className="text-4xl font-black tracking-tight"
              style={{
                background: "linear-gradient(135deg, #fff 0%, #94a3b8 50%, #fff 100%)",
                WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
              }}>
              LOTTOMAX AI
            </h1>
            <div className={`w-3 h-3 rounded-full ${modelReady ? "bg-blue-500" : "bg-gray-600"} animate-pulse`} />
          </div>
          <p className="text-gray-500 text-sm tracking-widest uppercase">
            LSTM + 5-Strategy Ensemble Engine
          </p>
          <div className="flex justify-center gap-4 mt-2 text-xs">
            <span className={connected ? "text-green-500" : "text-red-500"}>
              {connected ? "● Server Connected" : "● Server Offline"}
            </span>
            {serverInfo && (
              <>
                <span className="text-gray-600">{serverInfo.main_draws} draws</span>
                <span className={modelReady ? "text-blue-400" : "text-gray-600"}>
                  {modelReady ? "● LSTM Ready" : "○ LSTM Not Trained"}
                </span>
              </>
            )}
          </div>
        </header>

        {/* Tabs */}
        <nav className="flex justify-center gap-1 mb-8">
          {["generate", "analysis", "settings"].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-5 py-2 text-xs uppercase tracking-widest font-bold rounded transition-all
                ${activeTab === tab
                  ? "bg-white/10 text-white border border-white/20"
                  : "text-gray-500 hover:text-gray-300 border border-transparent"}`}
            >
              {tab}
            </button>
          ))}
        </nav>

        {/* Not connected warning */}
        {!connected && (
          <div className="border border-red-500/30 bg-red-500/5 rounded-xl p-6 mb-8 text-center">
            <p className="text-red-400 font-bold mb-2">Server not connected</p>
            <p className="text-gray-400 text-sm mb-4">Start the backend server:</p>
            <code className="text-gray-300 bg-black/40 px-4 py-2 rounded text-sm">
              cd backend && python app.py
            </code>
          </div>
        )}

        {/* ===================== GENERATE TAB ===================== */}
        {activeTab === "generate" && connected && (
          <div>
            {/* Control Buttons */}
            <div className="flex flex-col sm:flex-row gap-3 justify-center mb-8">
              <button
                onClick={startTraining}
                disabled={isTraining}
                className={`px-6 py-3 rounded-xl text-sm font-bold uppercase tracking-wider transition-all
                  ${isTraining
                    ? "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 animate-pulse cursor-wait"
                    : modelReady
                      ? "bg-green-500/10 text-green-400 border border-green-500/30 hover:bg-green-500/20"
                      : "bg-white/5 text-gray-300 border border-white/10 hover:bg-white/10"}`}
              >
                {isTraining ? "⏳ Training LSTM..." : modelReady ? "✅ Retrain Model" : "🧠 Train LSTM Model"}
              </button>

              <button
                onClick={generate}
                disabled={isGenerating || !connected}
                className={`px-8 py-3 rounded-xl text-sm font-bold uppercase tracking-wider transition-all
                  ${isGenerating
                    ? "bg-blue-500/20 text-blue-400 border border-blue-500/30 animate-pulse"
                    : "bg-gradient-to-r from-red-500/80 to-orange-500/80 text-white border border-red-500/30 hover:from-red-500 hover:to-orange-500 shadow-lg shadow-red-500/20"}`}
              >
                {isGenerating ? "⏳ Analyzing..." : "🎰 Generate Numbers"}
              </button>
            </div>

            {!modelReady && !isTraining && (
              <p className="text-center text-gray-500 text-xs mb-6">
                💡 Train the LSTM model first for deep learning predictions, or generate with statistical strategies only.
              </p>
            )}

            {/* Training Progress */}
            {(isTraining || trainingLog.length > 0) && (
              <div className="mb-8">
                <TrainingProgress progress={trainingProgress} log={trainingLog} />
              </div>
            )}

            {/* Prediction Display */}
            {prediction && (
              <div className="mb-8">
                {/* Main Numbers */}
                <div className="bg-white/[0.03] border border-white/10 rounded-2xl p-6 mb-4">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wider">LottoMax Numbers</h2>
                    <div className="flex items-center gap-2">
                      {prediction.model_trained && (
                        <span className="text-xs bg-blue-500/20 text-blue-400 px-2 py-0.5 rounded">LSTM</span>
                      )}
                      <div className="h-2 rounded-full"
                        style={{
                          width: `${prediction.main.confidence}px`,
                          background: `linear-gradient(90deg, #22c55e, ${prediction.main.confidence > 50 ? "#22c55e" : "#ef4444"})`,
                        }} />
                      <span className="text-xs text-gray-500">{prediction.main.confidence}%</span>
                    </div>
                  </div>

                  <div className="flex justify-center gap-3 flex-wrap">
                    {prediction.main.numbers.map((num, i) => (
                      <div key={num} className="flex flex-col items-center gap-1">
                        <LottoBall number={num} delay={i * 200} isRevealed={true} />
                        {showStrategies && <StrategyBar strategies={prediction.main.strategies} number={num} />}
                      </div>
                    ))}
                  </div>

                  <button
                    onClick={() => setShowStrategies(!showStrategies)}
                    className="mt-4 text-xs text-gray-500 hover:text-gray-300 transition-colors w-full text-center"
                  >
                    {showStrategies ? "▲ Hide Strategy Breakdown" : "▼ Show Strategy Breakdown"}
                  </button>

                  {showStrategies && (
                    <div className="mt-3 flex justify-center gap-4 text-xs text-gray-500">
                      {[
                        { label: "LSTM", color: "#ef4444" },
                        { label: "Freq", color: "#3b82f6" },
                        { label: "Gap", color: "#a855f7" },
                        { label: "Pair", color: "#22c55e" },
                        { label: "Dist", color: "#f97316" },
                      ].map((s) => (
                        <span key={s.label} className="flex items-center gap-1">
                          <div className="w-2 h-2 rounded-sm" style={{ backgroundColor: s.color }} />
                          {s.label}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* Extra Numbers */}
                {prediction.extra && (
                  <div className="bg-white/[0.03] border border-amber-500/20 rounded-2xl p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h2 className="text-sm font-bold text-amber-400/80 uppercase tracking-wider">Extra Numbers</h2>
                      <span className="text-xs text-gray-500">{prediction.extra.confidence}%</span>
                    </div>
                    <div className="flex justify-center gap-3">
                      {prediction.extra.numbers.map((num, i) => (
                        <LottoBall key={num} number={num} delay={i * 200 + 1400} isExtra={true} isRevealed={true} />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* History */}
            {history.length > 0 && (
              <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-5 mb-8">
                <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-3">Generation History</h3>
                <div className="space-y-2">
                  {history.map((h, idx) => (
                    <div key={h.id}
                      className={`flex items-center gap-3 py-2 px-3 rounded-lg transition-all
                        ${idx === 0 ? "bg-white/5" : "opacity-50 hover:opacity-80"}`}>
                      <span className="text-xs text-gray-600 w-16 flex-shrink-0">{h.time}</span>
                      <div className="flex gap-1.5 flex-wrap">
                        {h.main.map((n) => (
                          <LottoBall key={n} number={n} size="sm" isRevealed={true} delay={0} />
                        ))}
                      </div>
                      {h.extra && (
                        <>
                          <span className="text-gray-600 text-xs">+</span>
                          <div className="flex gap-1.5">
                            {h.extra.map((n) => (
                              <LottoBall key={n} number={n} size="sm" isExtra={true} isRevealed={true} delay={0} />
                            ))}
                          </div>
                        </>
                      )}
                      <div className="ml-auto flex items-center gap-2">
                        {h.modelTrained && <span className="text-xs text-blue-400/50">LSTM</span>}
                        <span className="text-xs text-gray-600">{h.confidence}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ===================== ANALYSIS TAB ===================== */}
        {activeTab === "analysis" && connected && (
          <div>
            {frequencies ? (
              <>
                <FrequencyChart data={frequencies.main_recent} numRange={50}
                  title={`LottoMax Frequency (Last ${frequencies.recent_window} Draws)`} />

                {frequencies.extra_recent && (
                  <FrequencyChart data={frequencies.extra_recent} numRange={99}
                    title={`Extra Frequency (Last ${frequencies.recent_window} Draws)`} />
                )}

                {/* Hot & Cold */}
                <div className="mt-8 grid grid-cols-2 gap-4">
                  <div className="bg-white/[0.03] border border-white/10 rounded-xl p-4">
                    <h3 className="text-sm font-bold text-red-400 uppercase tracking-wider mb-3">🔥 Hot Numbers</h3>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(frequencies.main_recent)
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 10)
                        .map(([n, freq]) => (
                          <div key={n} className="flex items-center gap-1">
                            <LottoBall number={parseInt(n)} size="sm" isRevealed={true} delay={0} />
                            <span className="text-xs text-gray-500">{freq}</span>
                          </div>
                        ))}
                    </div>
                  </div>
                  <div className="bg-white/[0.03] border border-white/10 rounded-xl p-4">
                    <h3 className="text-sm font-bold text-blue-400 uppercase tracking-wider mb-3">❄️ Cold Numbers</h3>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(frequencies.main_recent)
                        .sort((a, b) => a[1] - b[1])
                        .slice(0, 10)
                        .map(([n, freq]) => (
                          <div key={n} className="flex items-center gap-1">
                            <LottoBall number={parseInt(n)} size="sm" isRevealed={true} delay={0} />
                            <span className="text-xs text-gray-500">{freq}</span>
                          </div>
                        ))}
                    </div>
                  </div>
                </div>

                {/* Overdue */}
                <div className="mt-4 bg-white/[0.03] border border-white/10 rounded-xl p-4">
                  <h3 className="text-sm font-bold text-purple-400 uppercase tracking-wider mb-3">📊 Most Overdue</h3>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(frequencies.main_gaps)
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 10)
                      .map(([n, gap]) => (
                        <div key={n} className="flex items-center gap-1">
                          <LottoBall number={parseInt(n)} size="sm" isRevealed={true} delay={0} />
                          <span className="text-xs text-gray-500">{gap}d</span>
                        </div>
                      ))}
                  </div>
                </div>
              </>
            ) : (
              <p className="text-center text-gray-500">Loading analysis...</p>
            )}
          </div>
        )}

        {/* ===================== SETTINGS TAB ===================== */}
        {activeTab === "settings" && (
          <div className="space-y-6">
            {/* Training Settings */}
            <div className="bg-white/[0.03] border border-white/10 rounded-2xl p-6">
              <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-4">Training Settings</h3>
              <div className="mb-4">
                <label className="text-sm text-gray-300 block mb-2">LSTM Training Epochs</label>
                <input
                  type="range" min="20" max="300" value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value))}
                  className="w-full h-1 rounded-lg appearance-none cursor-pointer"
                  style={{ accentColor: "#3b82f6" }}
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Fast (20)</span>
                  <span className="text-white font-bold">{epochs}</span>
                  <span>Deep (300)</span>
                </div>
              </div>
              <p className="text-xs text-gray-600">
                More epochs = deeper pattern learning but longer training time.
                Early stopping prevents overfitting.
              </p>
            </div>

            {/* Strategy Weights */}
            <div className="bg-white/[0.03] border border-white/10 rounded-2xl p-6">
              <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-4">Strategy Weights</h3>
              <p className="text-xs text-gray-500 mb-4">
                Adjust each strategy's influence. LSTM gets highest default weight for deep pattern detection.
              </p>
              {[
                { key: "lstm", label: "LSTM Deep Learning", color: "#ef4444", desc: "Neural network sequential pattern detection" },
                { key: "frequency", label: "Frequency + Recency", color: "#3b82f6", desc: "Hot/cold numbers with time decay" },
                { key: "gap", label: "Gap Analysis", color: "#a855f7", desc: "Overdue numbers based on gap distributions" },
                { key: "pair", label: "Pair Correlation", color: "#22c55e", desc: "Numbers that appear together frequently" },
                { key: "distribution", label: "Distribution Balance", color: "#f97316", desc: "Range & odd/even equilibrium" },
              ].map((s) => (
                <div key={s.key} className="mb-4">
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full" style={{ backgroundColor: s.color }} />
                      <span className="text-sm text-gray-300">{s.label}</span>
                    </div>
                    <span className="text-sm text-gray-400 font-mono">{weights[s.key].toFixed(2)}</span>
                  </div>
                  <input
                    type="range" min="0" max="100" value={weights[s.key] * 100}
                    onChange={(e) => setWeights((prev) => ({ ...prev, [s.key]: parseInt(e.target.value) / 100 }))}
                    className="w-full h-1 rounded-lg appearance-none cursor-pointer"
                    style={{ accentColor: s.color }}
                  />
                  <p className="text-xs text-gray-600 mt-0.5">{s.desc}</p>
                </div>
              ))}
              <div className="flex justify-between items-center pt-3 border-t border-white/5">
                <span className="text-xs text-gray-500">
                  Total: {Object.values(weights).reduce((a, b) => a + b, 0).toFixed(2)}
                </span>
                <button
                  onClick={() => setWeights({ lstm: 0.30, frequency: 0.20, gap: 0.20, pair: 0.15, distribution: 0.15 })}
                  className="text-xs text-gray-500 hover:text-white transition-colors"
                >
                  Reset defaults
                </button>
              </div>
            </div>

            {/* Server Info */}
            <div className="bg-white/[0.03] border border-white/10 rounded-2xl p-6">
              <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-4">Server Info</h3>
              {serverInfo ? (
                <div className="text-xs text-gray-500 space-y-1">
                  <p>Main draws: {serverInfo.main_draws}</p>
                  <p>Extra draws: {serverInfo.extra_draws}</p>
                  <p>Main model: {serverInfo.main_model_loaded ? "✅ Loaded" : "❌ Not trained"}</p>
                  <p>Extra model: {serverInfo.extra_model_loaded ? "✅ Loaded" : "❌ Not trained"}</p>
                  {serverInfo.last_trained && <p>Last trained: {new Date(serverInfo.last_trained).toLocaleString()}</p>}
                </div>
              ) : (
                <p className="text-xs text-gray-600">Not connected</p>
              )}
              <button
                onClick={async () => {
                  await fetch(`${API}/reload-data`, { method: "POST" });
                  checkServer();
                }}
                className="mt-3 px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-xs hover:bg-white/10 transition-all"
              >
                🔄 Reload CSV Data
              </button>
            </div>
          </div>
        )}

        {/* Footer */}
        <footer className="mt-12 text-center text-gray-700 text-xs">
          <p>LottoMax AI — LSTM + 5-Strategy Ensemble Engine</p>
          <p className="mt-1">For entertainment purposes. Lottery outcomes are not guaranteed.</p>
        </footer>
      </div>
    </div>
  );
}
