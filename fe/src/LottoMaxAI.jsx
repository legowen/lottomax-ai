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

  const sizes = {
    lg: { width: "56px", height: "56px", fontSize: "22px" },
    md: { width: "44px", height: "44px", fontSize: "16px" },
    sm: { width: "32px", height: "32px", fontSize: "12px" },
  };
  const s = sizes[size] || sizes.lg;

  return (
    <div
      style={{
        width: s.width,
        height: s.height,
        minWidth: s.width,
        minHeight: s.height,
        borderRadius: "50%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontWeight: 900,
        fontSize: s.fontSize,
        flexShrink: 0,
        position: "relative",
        background: revealed ? color.bg : "linear-gradient(135deg, #374151, #1f2937)",
        color: revealed ? color.text : "#6b7280",
        transform: revealed ? "scale(1)" : "scale(0.6)",
        opacity: revealed ? 1 : 0.4,
        boxShadow: revealed ? `0 0 20px ${color.glow}, inset 0 -3px 6px rgba(0,0,0,0.3)` : "none",
        transition: "all 0.7s ease-out",
        userSelect: "none",
      }}
    >
      {revealed && (
        <div style={{
          position: "absolute", inset: 0, borderRadius: "50%",
          background: "radial-gradient(circle at 35% 30%, rgba(255,255,255,0.4) 0%, transparent 60%)",
        }} />
      )}
      <span style={{ position: "relative", zIndex: 1 }}>{revealed ? number : "?"}</span>
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
    <div style={{ display: "flex", gap: "2px", alignItems: "flex-end", height: "32px", marginTop: "4px" }}>
      {items.map((item) => (
        <div
          key={item.key}
          title={`${item.key}: ${(item.val * 100).toFixed(0)}%`}
          style={{
            width: "8px",
            borderRadius: "2px 2px 0 0",
            transition: "all 0.5s",
            height: `${Math.max(2, item.val * 32)}px`,
            backgroundColor: item.color,
            opacity: 0.8,
          }}
        />
      ))}
    </div>
  );
}

function FrequencyChart({ data, numRange, title }) {
  if (!data) return null;
  const maxFreq = Math.max(...Object.values(data));

  return (
    <div style={{ marginTop: "24px" }}>
      <h3 style={{ fontSize: "14px", fontWeight: 600, color: "#e2e8f0", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "12px" }}>{title}</h3>
      <div style={{ display: "flex", alignItems: "flex-end", gap: "1px", height: "96px", overflowX: "auto", paddingBottom: "4px" }}>
        {Array.from({ length: numRange }, (_, i) => i + 1).map((n) => {
          const freq = data[String(n)] || 0;
          return (
            <div key={n} style={{ display: "flex", flexDirection: "column", alignItems: "center", flexShrink: 0, width: numRange > 50 ? "8px" : "14px" }}>
              <div
                style={{
                  width: "100%",
                  borderRadius: "2px 2px 0 0",
                  transition: "all 0.3s",
                  height: `${maxFreq > 0 ? (freq / maxFreq) * 80 : 0}px`,
                  background: getBallColor(n, numRange > 50).bg,
                  opacity: freq > 0 ? 0.7 : 0.15,
                }}
                title={`#${n}: ${freq} times`}
              />
              {numRange <= 50 && n % 5 === 0 && (
                <span style={{ color: "#94a3b8", marginTop: "4px", fontSize: "8px" }}>{n}</span>
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
  const logRef = useRef(null);

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

  // Auto-scroll training log
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [trainingLog]);

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
    <div style={{
      minHeight: "100vh",
      width: "100%",
      background: "linear-gradient(160deg, #0a0a0f 0%, #0d1117 30%, #101820 60%, #0a0a0f 100%)",
      color: "#e2e8f0",
      fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
    }}>
      {/* Background grid */}
      <div style={{
        position: "fixed", inset: 0, opacity: 0.05, pointerEvents: "none",
        backgroundImage: "radial-gradient(circle, #ffffff 1px, transparent 1px)",
        backgroundSize: "30px 30px",
      }} />

      <div style={{ position: "relative", maxWidth: "960px", margin: "0 auto", padding: "32px 24px" }}>
        {/* Header */}
        <header style={{ textAlign: "center", marginBottom: "8px" }}>
          <h1 style={{
            fontSize: "clamp(28px, 5vw, 40px)",
            fontWeight: 900,
            letterSpacing: "-0.02em",
            background: "linear-gradient(135deg, #fff 0%, #94a3b8 50%, #fff 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}>
            LOTTOMAX AI
          </h1>
          <p style={{ color: "#94a3b8", fontSize: "12px", letterSpacing: "0.15em", textTransform: "uppercase", marginTop: "4px" }}>
            LSTM + 5-Strategy Ensemble Engine
          </p>
        </header>

        {/* Status bar */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "20px", margin: "12px 0", fontSize: "12px", flexWrap: "wrap" }}>
          <span style={{ display: "flex", alignItems: "center", gap: "6px", color: connected ? "#4ade80" : "#f87171" }}>
            <span style={{
              display: "inline-block", width: "8px", height: "8px", borderRadius: "50%",
              backgroundColor: connected ? "#4ade80" : "#f87171",
              animation: "pulse 2s infinite",
            }} />
            {connected ? "Connected" : "Offline"}
          </span>
          {serverInfo && (
            <>
              <span style={{ color: "#94a3b8" }}>{serverInfo.main_draws} draws</span>
              <span style={{ display: "flex", alignItems: "center", gap: "6px", color: modelReady ? "#60a5fa" : "#94a3b8" }}>
                <span style={{
                  display: "inline-block", width: "8px", height: "8px", borderRadius: "50%",
                  backgroundColor: modelReady ? "#60a5fa" : "#6b7280",
                  animation: "pulse 2s infinite",
                }} />
                {modelReady ? "LSTM Ready" : "LSTM Not Trained"}
              </span>
            </>
          )}
        </div>

        {/* Tabs */}
        <nav style={{ display: "flex", justifyContent: "center", gap: "4px", marginBottom: "32px", borderBottom: "1px solid rgba(255,255,255,0.1)", paddingBottom: "12px" }}>
          {["generate", "analysis", "settings"].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                padding: "8px 24px",
                fontSize: "12px",
                textTransform: "uppercase",
                letterSpacing: "0.1em",
                fontWeight: 700,
                borderRadius: "8px",
                cursor: "pointer",
                border: activeTab === tab ? "1px solid rgba(255,255,255,0.2)" : "1px solid transparent",
                background: activeTab === tab ? "rgba(255,255,255,0.1)" : "transparent",
                color: activeTab === tab ? "#ffffff" : "#94a3b8",
              }}
            >
              {tab}
            </button>
          ))}
        </nav>

        {/* Not connected warning */}
        {!connected && (
          <div style={{
            border: "1px solid rgba(239,68,68,0.3)",
            background: "rgba(239,68,68,0.1)",
            borderRadius: "12px",
            padding: "24px",
            marginBottom: "32px",
            textAlign: "center",
          }}>
            <p style={{ color: "#f87171", fontWeight: 700, marginBottom: "8px" }}>Server not connected</p>
            <p style={{ color: "#cbd5e1", fontSize: "14px", marginBottom: "16px" }}>Start the backend server:</p>
            <code style={{ color: "#e2e8f0", background: "rgba(0,0,0,0.4)", padding: "8px 16px", borderRadius: "6px", fontSize: "14px" }}>
              cd backend && python app.py
            </code>
          </div>
        )}

        {/* ===================== GENERATE TAB ===================== */}
        {activeTab === "generate" && connected && (
          <div>
            {/* Control Buttons */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "24px", margin: "32px 0" }}>
              <button
                onClick={startTraining}
                disabled={isTraining}
                style={{
                  padding: "12px 24px",
                  borderRadius: "12px",
                  fontSize: "14px",
                  fontWeight: 700,
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  cursor: isTraining ? "wait" : "pointer",
                  border: "1px solid",
                  borderColor: isTraining ? "#eab308" : modelReady ? "#22c55e" : "#64748b",
                  background: isTraining ? "rgba(234,179,8,0.15)" : modelReady ? "rgba(34,197,94,0.1)" : "rgba(255,255,255,0.05)",
                  color: isTraining ? "#facc15" : modelReady ? "#4ade80" : "#e2e8f0",
                }}
              >
                {isTraining ? "\u23F3 Training LSTM..." : modelReady ? "\u2705 Retrain Model" : "\uD83E\uDDE0 Train LSTM Model"}
              </button>

              <button
                onClick={generate}
                disabled={isGenerating}
                style={{
                  padding: "12px 32px",
                  borderRadius: "12px",
                  fontSize: "14px",
                  fontWeight: 700,
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  cursor: isGenerating ? "wait" : "pointer",
                  border: "1px solid rgba(239,68,68,0.3)",
                  background: isGenerating ? "rgba(59,130,246,0.15)" : "linear-gradient(135deg, rgba(239,68,68,0.8), rgba(249,115,22,0.8))",
                  color: "#ffffff",
                  boxShadow: isGenerating ? "none" : "0 4px 15px rgba(239,68,68,0.2)",
                }}
              >
                {isGenerating ? "\u23F3 Analyzing..." : "\uD83C\uDFB0 Generate Numbers"}
              </button>
            </div>

            {!modelReady && !isTraining && (
              <p style={{ textAlign: "center", color: "#94a3b8", fontSize: "12px", marginBottom: "24px" }}>
                Train the LSTM model first for deep learning predictions, or generate with statistical strategies only.
              </p>
            )}

            {/* Training Progress / Log */}
            {(isTraining || trainingLog.length > 0) && (
              <div style={{ marginBottom: "24px" }}>
                {/* Progress bar */}
                {trainingProgress.status === "training" && (
                  <div style={{ padding: "12px 16px", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", color: "#cbd5e1", marginBottom: "8px" }}>
                      <span>{trainingProgress.strategy}</span>
                      <span>Epoch {trainingProgress.epoch}/{trainingProgress.total_epochs} &bull; Loss: {trainingProgress.loss}</span>
                    </div>
                    <div style={{ width: "100%", height: "6px", background: "rgba(255,255,255,0.05)", borderRadius: "3px", overflow: "hidden" }}>
                      <div style={{
                        height: "100%",
                        borderRadius: "3px",
                        transition: "all 0.3s",
                        width: `${(trainingProgress.epoch / trainingProgress.total_epochs) * 100}%`,
                        background: "linear-gradient(90deg, #3b82f6, #a855f7)",
                      }} />
                    </div>
                  </div>
                )}
                {/* Log terminal box */}
                <div
                  ref={logRef}
                  style={{
                    background: "rgba(0,0,0,0.4)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: "12px",
                    padding: "12px",
                    maxHeight: "200px",
                    overflowY: "auto",
                    fontFamily: "monospace",
                    fontSize: "12px",
                  }}
                >
                  {trainingLog.map((entry, i) => (
                    <div key={i} style={{ color: "#94a3b8", padding: "2px 0" }}>
                      <span style={{ color: "#475569", marginRight: "8px" }}>{entry.time}</span>
                      <span style={{ color: "#cbd5e1" }}>{entry.msg}</span>
                    </div>
                  ))}
                  {trainingLog.length === 0 && <span style={{ color: "#94a3b8" }}>Waiting for training...</span>}
                </div>
              </div>
            )}

            {/* Prediction Display */}
            {prediction && (
              <div style={{ marginBottom: "32px" }}>
                {/* Main Numbers */}
                <div style={{
                  background: "rgba(255,255,255,0.03)",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: "16px",
                  padding: "24px",
                  marginBottom: "16px",
                }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "20px" }}>
                    <h2 style={{ fontSize: "14px", fontWeight: 700, color: "#f1f5f9", textTransform: "uppercase", letterSpacing: "0.05em" }}>LottoMax Numbers</h2>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      {prediction.model_trained && (
                        <span style={{ fontSize: "12px", background: "rgba(59,130,246,0.2)", color: "#60a5fa", padding: "2px 8px", borderRadius: "4px" }}>LSTM</span>
                      )}
                      <div style={{
                        height: "8px",
                        borderRadius: "4px",
                        width: `${prediction.main.confidence}px`,
                        background: `linear-gradient(90deg, #22c55e, ${prediction.main.confidence > 50 ? "#22c55e" : "#ef4444"})`,
                      }} />
                      <span style={{ fontSize: "12px", color: "#cbd5e1" }}>{prediction.main.confidence}%</span>
                    </div>
                  </div>

                  {/* Main ball row - horizontal, no wrapping */}
                  <div style={{ display: "flex", flexDirection: "row", alignItems: "center", justifyContent: "center", gap: "12px", flexWrap: "nowrap" }}>
                    {prediction.main.numbers.map((num, i) => (
                      <div key={num} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "4px" }}>
                        <LottoBall number={num} delay={i * 200} isRevealed={true} />
                        {showStrategies && <StrategyBar strategies={prediction.main.strategies} number={num} />}
                      </div>
                    ))}
                  </div>

                  <button
                    onClick={() => setShowStrategies(!showStrategies)}
                    style={{
                      marginTop: "16px",
                      fontSize: "12px",
                      color: "#94a3b8",
                      background: "none",
                      border: "none",
                      cursor: "pointer",
                      width: "100%",
                      textAlign: "center",
                    }}
                  >
                    {showStrategies ? "Hide Strategy Breakdown" : "Show Strategy Breakdown"}
                  </button>

                  {showStrategies && (
                    <div style={{ marginTop: "12px", display: "flex", justifyContent: "center", gap: "16px", fontSize: "12px", color: "#cbd5e1" }}>
                      {[
                        { label: "LSTM", color: "#ef4444" },
                        { label: "Freq", color: "#3b82f6" },
                        { label: "Gap", color: "#a855f7" },
                        { label: "Pair", color: "#22c55e" },
                        { label: "Dist", color: "#f97316" },
                      ].map((s) => (
                        <span key={s.label} style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                          <div style={{ width: "8px", height: "8px", borderRadius: "2px", backgroundColor: s.color }} />
                          {s.label}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* Extra Numbers */}
                {prediction.extra && (
                  <div style={{
                    background: "rgba(255,255,255,0.03)",
                    border: "1px solid rgba(245,158,11,0.3)",
                    borderRadius: "16px",
                    padding: "24px",
                  }}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "20px" }}>
                      <h2 style={{ fontSize: "14px", fontWeight: 700, color: "#fbbf24", textTransform: "uppercase", letterSpacing: "0.05em" }}>Extra Numbers</h2>
                      <span style={{ fontSize: "12px", color: "#cbd5e1" }}>{prediction.extra.confidence}%</span>
                    </div>
                    <div style={{ display: "flex", flexDirection: "row", alignItems: "center", justifyContent: "center", gap: "12px", flexWrap: "nowrap" }}>
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
              <div style={{
                background: "rgba(255,255,255,0.02)",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: "16px",
                padding: "16px 20px",
                marginBottom: "32px",
              }}>
                <h3 style={{ fontSize: "14px", fontWeight: 700, color: "#f1f5f9", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "12px" }}>Generation History</h3>
                <div>
                  {history.map((h, idx) => (
                    <div key={h.id} style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      padding: "8px 12px",
                      borderRadius: "8px",
                      background: idx === 0 ? "rgba(255,255,255,0.05)" : "transparent",
                      opacity: idx === 0 ? 1 : 0.6,
                    }}>
                      <span style={{ fontSize: "12px", color: "#94a3b8", width: "70px", flexShrink: 0 }}>{h.time}</span>
                      <div style={{ display: "flex", flexDirection: "row", gap: "4px", flexWrap: "nowrap" }}>
                        {h.main.map((n) => (
                          <LottoBall key={n} number={n} size="sm" isRevealed={true} delay={0} />
                        ))}
                      </div>
                      {h.extra && (
                        <>
                          <span style={{ color: "#64748b", fontSize: "12px" }}>+</span>
                          <div style={{ display: "flex", flexDirection: "row", gap: "4px", flexWrap: "nowrap" }}>
                            {h.extra.map((n) => (
                              <LottoBall key={n} number={n} size="sm" isExtra={true} isRevealed={true} delay={0} />
                            ))}
                          </div>
                        </>
                      )}
                      <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "8px", flexShrink: 0 }}>
                        {h.modelTrained && <span style={{ fontSize: "12px", color: "#60a5fa" }}>LSTM</span>}
                        <span style={{ fontSize: "12px", color: "#94a3b8" }}>{h.confidence}%</span>
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
                <div style={{ marginTop: "32px", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
                  <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "12px", padding: "16px" }}>
                    <h3 style={{ fontSize: "14px", fontWeight: 700, color: "#f87171", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "12px" }}>Hot Numbers</h3>
                    <div style={{ display: "flex", flexDirection: "row", flexWrap: "wrap", gap: "8px", alignItems: "center" }}>
                      {Object.entries(frequencies.main_recent)
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 10)
                        .map(([n, freq]) => (
                          <div key={n} style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                            <LottoBall number={parseInt(n)} size="sm" isRevealed={true} delay={0} />
                            <span style={{ fontSize: "11px", color: "#94a3b8" }}>{freq}</span>
                          </div>
                        ))}
                    </div>
                  </div>
                  <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "12px", padding: "16px" }}>
                    <h3 style={{ fontSize: "14px", fontWeight: 700, color: "#60a5fa", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "12px" }}>Cold Numbers</h3>
                    <div style={{ display: "flex", flexDirection: "row", flexWrap: "wrap", gap: "8px", alignItems: "center" }}>
                      {Object.entries(frequencies.main_recent)
                        .sort((a, b) => a[1] - b[1])
                        .slice(0, 10)
                        .map(([n, freq]) => (
                          <div key={n} style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                            <LottoBall number={parseInt(n)} size="sm" isRevealed={true} delay={0} />
                            <span style={{ fontSize: "11px", color: "#94a3b8" }}>{freq}</span>
                          </div>
                        ))}
                    </div>
                  </div>
                </div>

                {/* Overdue */}
                <div style={{ marginTop: "16px", background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "12px", padding: "16px" }}>
                  <h3 style={{ fontSize: "14px", fontWeight: 700, color: "#c084fc", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "12px" }}>Most Overdue</h3>
                  <div style={{ display: "flex", flexDirection: "row", flexWrap: "wrap", gap: "8px", alignItems: "center" }}>
                    {Object.entries(frequencies.main_gaps)
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 10)
                      .map(([n, gap]) => (
                        <div key={n} style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                          <LottoBall number={parseInt(n)} size="sm" isRevealed={true} delay={0} />
                          <span style={{ fontSize: "11px", color: "#94a3b8" }}>{gap}d</span>
                        </div>
                      ))}
                  </div>
                </div>
              </>
            ) : (
              <p style={{ textAlign: "center", color: "#94a3b8" }}>Loading analysis...</p>
            )}
          </div>
        )}

        {/* ===================== SETTINGS TAB ===================== */}
        {activeTab === "settings" && (
          <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
            {/* Training Settings */}
            <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "16px", padding: "24px" }}>
              <h3 style={{ fontSize: "14px", fontWeight: 700, color: "#f1f5f9", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "16px" }}>Training Settings</h3>
              <div style={{ marginBottom: "16px" }}>
                <label style={{ fontSize: "14px", color: "#e2e8f0", display: "block", marginBottom: "8px" }}>LSTM Training Epochs</label>
                <input
                  type="range" min="20" max="300" value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value))}
                  style={{ width: "100%", height: "4px", borderRadius: "4px", appearance: "none", cursor: "pointer", accentColor: "#3b82f6" }}
                />
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", color: "#94a3b8", marginTop: "4px" }}>
                  <span>Fast (20)</span>
                  <span style={{ color: "#ffffff", fontWeight: 700 }}>{epochs}</span>
                  <span>Deep (300)</span>
                </div>
              </div>
              <p style={{ fontSize: "12px", color: "#94a3b8" }}>
                More epochs = deeper pattern learning but longer training time.
                Early stopping prevents overfitting.
              </p>
            </div>

            {/* Strategy Weights */}
            <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "16px", padding: "24px" }}>
              <h3 style={{ fontSize: "14px", fontWeight: 700, color: "#f1f5f9", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "16px" }}>Strategy Weights</h3>
              <p style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "16px" }}>
                Adjust each strategy&apos;s influence. LSTM gets highest default weight for deep pattern detection.
              </p>
              {[
                { key: "lstm", label: "LSTM Deep Learning", color: "#ef4444", desc: "Neural network sequential pattern detection" },
                { key: "frequency", label: "Frequency + Recency", color: "#3b82f6", desc: "Hot/cold numbers with time decay" },
                { key: "gap", label: "Gap Analysis", color: "#a855f7", desc: "Overdue numbers based on gap distributions" },
                { key: "pair", label: "Pair Correlation", color: "#22c55e", desc: "Numbers that appear together frequently" },
                { key: "distribution", label: "Distribution Balance", color: "#f97316", desc: "Range & odd/even equilibrium" },
              ].map((s) => (
                <div key={s.key} style={{ marginBottom: "16px" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "4px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <div style={{ width: "8px", height: "8px", borderRadius: "50%", backgroundColor: s.color }} />
                      <span style={{ fontSize: "14px", color: "#e2e8f0" }}>{s.label}</span>
                    </div>
                    <span style={{ fontSize: "14px", color: "#cbd5e1", fontFamily: "monospace" }}>{weights[s.key].toFixed(2)}</span>
                  </div>
                  <input
                    type="range" min="0" max="100" value={weights[s.key] * 100}
                    onChange={(e) => setWeights((prev) => ({ ...prev, [s.key]: parseInt(e.target.value) / 100 }))}
                    style={{ width: "100%", height: "4px", borderRadius: "4px", appearance: "none", cursor: "pointer", accentColor: s.color }}
                  />
                  <p style={{ fontSize: "12px", color: "#94a3b8", marginTop: "2px" }}>{s.desc}</p>
                </div>
              ))}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingTop: "12px", borderTop: "1px solid rgba(255,255,255,0.1)" }}>
                <span style={{ fontSize: "12px", color: "#94a3b8" }}>
                  Total: {Object.values(weights).reduce((a, b) => a + b, 0).toFixed(2)}
                </span>
                <button
                  onClick={() => setWeights({ lstm: 0.30, frequency: 0.20, gap: 0.20, pair: 0.15, distribution: 0.15 })}
                  style={{ fontSize: "12px", color: "#94a3b8", background: "none", border: "none", cursor: "pointer" }}
                >
                  Reset defaults
                </button>
              </div>
            </div>

            {/* Server Info */}
            <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "16px", padding: "24px" }}>
              <h3 style={{ fontSize: "14px", fontWeight: 700, color: "#f1f5f9", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "16px" }}>Server Info</h3>
              {serverInfo ? (
                <div style={{ fontSize: "12px", color: "#cbd5e1", display: "flex", flexDirection: "column", gap: "4px" }}>
                  <p>Main draws: {serverInfo.main_draws}</p>
                  <p>Extra draws: {serverInfo.extra_draws}</p>
                  <p>Main model: <span style={{ color: serverInfo.main_model_loaded ? "#4ade80" : "#f87171" }}>{serverInfo.main_model_loaded ? "Loaded" : "Not trained"}</span></p>
                  <p>Extra model: <span style={{ color: serverInfo.extra_model_loaded ? "#4ade80" : "#f87171" }}>{serverInfo.extra_model_loaded ? "Loaded" : "Not trained"}</span></p>
                  {serverInfo.last_trained && <p>Last trained: {new Date(serverInfo.last_trained).toLocaleString()}</p>}
                </div>
              ) : (
                <p style={{ fontSize: "12px", color: "#94a3b8" }}>Not connected</p>
              )}
              <button
                onClick={async () => {
                  await fetch(`${API}/reload-data`, { method: "POST" });
                  checkServer();
                }}
                style={{
                  marginTop: "12px",
                  padding: "8px 16px",
                  background: "rgba(255,255,255,0.05)",
                  border: "1px solid rgba(255,255,255,0.15)",
                  borderRadius: "8px",
                  fontSize: "12px",
                  color: "#e2e8f0",
                  cursor: "pointer",
                }}
              >
                Reload CSV Data
              </button>
            </div>
          </div>
        )}

        {/* Footer */}
        <footer style={{ marginTop: "48px", textAlign: "center", fontSize: "12px", color: "#94a3b8" }}>
          <p>LottoMax AI — LSTM + 5-Strategy Ensemble Engine</p>
          <p style={{ marginTop: "4px", color: "#94a3b8" }}>For entertainment purposes. Lottery outcomes are not guaranteed.</p>
        </footer>
      </div>
    </div>
  );
}
