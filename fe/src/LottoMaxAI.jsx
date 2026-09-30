import { useState, useEffect, useCallback, useRef } from "react";
import SignalLab from "./SignalLab";
import EvTab from "./EvTab";
import DataPanel from "./DataPanel";
import Sprite from "./pixel/Sprite";
import PixelBall from "./pixel/PixelBall";
import PixelButton from "./pixel/PixelButton";
import PixelTabs from "./pixel/PixelTabs";
import CatReveal from "./pixel/CatReveal";
import { CAT_FRAMES, CAT_PALETTE } from "./pixel/sprites";
import { C, FONT, pixelBox } from "./pixel/theme";
import { card, h3, muted, body, th, td } from "./pixel/styles";

// ============================================================
// LottoMax AI - Frontend
// Connects to FastAPI backend with real LSTM 7-Strategy ensemble
// ============================================================

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Ball colors by number range (solid hex; same ranges as before)
const getBallColor = (num) => {
  if (num <= 10) return "#ef4444";
  if (num <= 20) return "#3b82f6";
  if (num <= 30) return "#a855f7";
  if (num <= 40) return "#22c55e";
  return "#f97316";
};

// Strategy legend colours (ball hues + palette)
const STRATEGIES = [
  { key: "lstm", label: "LSTM", color: C.bad },
  { key: "frequency", label: "Freq", color: "#3b82f6" },
  { key: "gap", label: "Gap", color: "#a855f7" },
  { key: "pair", label: "Pair", color: C.good },
  { key: "distribution", label: "Dist", color: "#f97316" },
  { key: "seed", label: "Seed", color: C.accent },
  { key: "smart", label: "Smart", color: C.ink },
];

const TABS = ["generate", "analysis", "backtest", "signal", "ev", "settings"];

// ============================================================
// Components
// ============================================================
function StrategyBar({ strategies, number }) {
  if (!strategies || !strategies[String(number)]) return null;
  const s = strategies[String(number)];

  return (
    <div style={{ display: "flex", gap: "2px", alignItems: "flex-end", height: "32px", marginTop: "4px" }}>
      {STRATEGIES.map((item) => {
        const val = s[item.key] ?? 0;
        return (
          <div
            key={item.key}
            title={`${item.label}: ${(val * 100).toFixed(0)}%`}
            style={{
              width: "8px",
              transition: "height 300ms steps(4, end)",
              height: `${Math.max(2, val * 32)}px`,
              backgroundColor: item.color,
            }}
          />
        );
      })}
    </div>
  );
}

function FrequencyChart({ data, numRange, title }) {
  if (!data) return null;
  const maxFreq = Math.max(...Object.values(data));

  return (
    <div style={{ marginTop: "24px" }}>
      <h3 style={h3}>{title}</h3>
      <div style={{ display: "flex", alignItems: "flex-end", gap: "2px", height: "104px", overflowX: "auto", paddingBottom: "4px" }}>
        {Array.from({ length: numRange }, (_, i) => i + 1).map((n) => {
          const freq = data[String(n)] || 0;
          return (
            <div key={n} style={{ display: "flex", flexDirection: "column", alignItems: "center", flexShrink: 0, width: "14px" }}>
              <div
                style={{
                  width: "100%",
                  transition: "height 300ms steps(4, end)",
                  height: `${maxFreq > 0 ? (freq / maxFreq) * 80 : 0}px`,
                  background: freq > 0 ? getBallColor(n) : C.panelHi,
                }}
                title={`#${n}: ${freq} times`}
              />
              {n % 5 === 0 && (
                <span style={{ fontFamily: FONT, color: C.dim, marginTop: "4px", fontSize: "8px" }}>{n}</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Small labelled ball list used by Hot / Cold / Overdue
function BallList({ entries, suffix = "" }) {
  return (
    <div style={{ display: "flex", flexDirection: "row", flexWrap: "wrap", gap: "12px", alignItems: "center" }}>
      {entries.map(([n, v]) => (
        <div key={n} style={{ display: "flex", alignItems: "center", gap: "4px" }}>
          <PixelBall n={parseInt(n)} color={getBallColor(parseInt(n))} scale={3} />
          <span style={{ fontFamily: FONT, fontSize: "8px", color: C.dim }}>{v}{suffix}</span>
        </div>
      ))}
    </div>
  );
}

const sectionTitle = (color) => ({ ...h3, color });
const btnRow = { display: "flex", alignItems: "center", justifyContent: "center", gap: "24px", margin: "32px 0", flexWrap: "wrap" };
const noticeBox = { ...pixelBox(C.shadow, C.good, 2, false), padding: "12px 14px", fontFamily: FONT, fontSize: 8, color: C.good, lineHeight: 1.9 };
const rangeStyle = { width: "100%", height: "8px", cursor: "pointer", accentColor: C.accent };

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
  const [seedAnalysis, setSeedAnalysis] = useState(null);
  const [backtest, setBacktest] = useState(null);
  const [isBacktesting, setIsBacktesting] = useState(false);
  const [revealDone, setRevealDone] = useState(false);
  const [weights, setWeights] = useState({
    lstm: 0.15, frequency: 0.15, gap: 0.20, pair: 0.05, distribution: 0.15, seed: 0.0, smart: 0.30,
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
    // eslint-disable-next-line react-hooks/set-state-in-effect
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
    } catch {
      alert("Cannot connect to server");
    }
  };

  // Generate prediction
  const generate = async () => {
    setIsGenerating(true);
    setPrediction(null);
    setRevealDone(false);

    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ weights }),
      });
      const data = await res.json();
      if (!res.ok) {
        alert(data.detail || "Prediction failed");
        setIsGenerating(false);
        return;
      }
      setPrediction(data);

      setHistory((prev) => [
        {
          id: Date.now(),
          main: data.main.numbers,
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
      if (!res.ok) return;
      const data = await res.json();
      setFrequencies(data);
    } catch {
      // ignore
    }
  };

  // Run walk-forward backtest
  const runBacktest = async () => {
    setIsBacktesting(true);
    try {
      const res = await fetch(`${API}/backtest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ window: 150, random_tickets: 100 }),
      });
      const data = await res.json();
      if (res.ok) setBacktest(data);
    } catch {
      // ignore
    }
    setIsBacktesting(false);
  };

  // Run seed analysis
  const runSeedAnalysis = async () => {
    try {
      const res = await fetch(`${API}/seed-analysis`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ max_draws: 20 }),
      });
      const data = await res.json();
      if (res.ok) setSeedAnalysis(data);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (activeTab === "analysis" && connected) loadFrequencies();
  }, [activeTab, connected]);

  // the newest history row stays hidden until Bomi has revealed it
  const visibleHistory = prediction && !revealDone ? history.slice(1) : history;

  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.ink, fontFamily: FONT }}>
      <div style={{ position: "relative", maxWidth: "960px", margin: "0 auto", padding: "24px 20px 32px" }}>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", gap: 16, padding: "16px 8px" }}>
          <Sprite rows={CAT_FRAMES.idle} palette={CAT_PALETTE} scale={3} />
          <div>
            <div style={{ fontFamily: FONT, fontSize: 20, color: C.accent, textShadow: `3px 3px 0 ${C.shadow}` }}>
              LOTTO MAX
            </div>
            <div style={{ fontFamily: FONT, fontSize: 8, color: C.dim, marginTop: 8 }}>BOMI&apos;S NUMBER PICKER</div>
          </div>
        </div>

        {/* Status bar */}
        <div style={{ display: "flex", alignItems: "center", gap: "20px", margin: "8px 8px 12px", fontFamily: FONT, fontSize: "8px", flexWrap: "wrap" }}>
          <span style={{ display: "flex", alignItems: "center", gap: "8px", color: connected ? C.good : C.bad }}>
            <span style={{ display: "inline-block", width: "8px", height: "8px", backgroundColor: connected ? C.good : C.bad }} />
            {connected ? "CONNECTED" : "OFFLINE"}
          </span>
          {serverInfo && (
            <>
              <span style={{ color: C.dim }}>{serverInfo.main_draws} DRAWS</span>
              <span style={{ display: "flex", alignItems: "center", gap: "8px", color: modelReady ? C.ink : C.dim }}>
                <span style={{ display: "inline-block", width: "8px", height: "8px", backgroundColor: modelReady ? C.accent : C.panelHi }} />
                {modelReady ? "LSTM READY" : "LSTM NOT TRAINED"}
              </span>
            </>
          )}
        </div>

        {/* Tabs */}
        <PixelTabs tabs={TABS} active={activeTab} onChange={setActiveTab} />

        {/* Not connected warning */}
        {!connected && (
          <div style={{ ...pixelBox(C.panel, C.bad, 4), padding: "24px", margin: "8px 4px 32px", textAlign: "center" }}>
            <p style={{ fontFamily: FONT, fontSize: 12, color: C.bad, marginBottom: "16px" }}>SERVER NOT CONNECTED</p>
            <p style={{ ...body, marginBottom: "16px" }}>Start the backend server:</p>
            <code style={{ ...pixelBox(C.shadow, C.dim, 2, false), display: "inline-block", fontFamily: FONT, color: C.ink, padding: "8px 16px", fontSize: "8px" }}>
              cd be && python app.py
            </code>
          </div>
        )}

        {/* ===================== GENERATE TAB ===================== */}
        {activeTab === "generate" && connected && (
          <div>
            {/* Honesty notice */}
            <div style={{ ...noticeBox, marginTop: "8px", textAlign: "center" }}>
              모든 번호 조합의 당첨 확률은 동일합니다. Smart Pick은 당첨 시 공동 당첨자를 줄이는 방식이며 당첨 확률을 높이지 않습니다.
            </div>

            {/* LSTM vs constant-probability baseline */}
            {serverInfo && serverInfo.lstm_verdict && serverInfo.lstm_verdict.beats_constant_baseline === false && (
              <div style={{
                ...pixelBox(C.shadow, C.bad, 2, false), display: "inline-block", marginTop: "16px", padding: "8px 12px",
                color: C.bad, fontFamily: FONT, fontSize: "8px", lineHeight: 1.8,
              }} title={`검증 손실 ${serverInfo.lstm_verdict.val_loss} vs 상수 기준선 ${serverInfo.lstm_verdict.baseline_loss}`}>
                LSTM: 상수 확률 대비 개선 없음
              </div>
            )}

            {/* Control Buttons */}
            <div style={btnRow}>
              <PixelButton
                onClick={startTraining}
                disabled={isTraining}
                tone={modelReady && !isTraining ? "good" : "plain"}
              >
                {isTraining ? "TRAINING LSTM..." : modelReady ? "RETRAIN MODEL" : "TRAIN LSTM MODEL"}
              </PixelButton>

              <PixelButton onClick={generate} disabled={isGenerating} tone="accent">
                {isGenerating ? "ANALYZING..." : "GENERATE NUMBERS"}
              </PixelButton>
            </div>

            {!modelReady && !isTraining && (
              <p style={{ ...muted, textAlign: "center", marginBottom: "24px" }}>
                Train the LSTM model first for deep learning predictions, or generate with statistical strategies only.
              </p>
            )}

            {/* Training Progress / Log */}
            {(isTraining || trainingLog.length > 0) && (
              <div style={{ marginBottom: "24px" }}>
                {/* Progress bar */}
                {trainingProgress.status === "training" && (
                  <div style={{ padding: "12px 16px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: "8px", fontFamily: FONT, fontSize: "8px", color: C.ink, marginBottom: "12px" }}>
                      <span>{trainingProgress.strategy}</span>
                      <span>EPOCH {trainingProgress.epoch}/{trainingProgress.total_epochs} &bull; LOSS: {trainingProgress.loss}</span>
                    </div>
                    <div style={{ ...pixelBox(C.shadow, C.dim, 2, false), width: "calc(100% - 4px)", height: "12px", overflow: "hidden" }}>
                      <div style={{
                        height: "100%",
                        transition: "width 300ms steps(6, end)",
                        width: `${(trainingProgress.epoch / trainingProgress.total_epochs) * 100}%`,
                        background: C.accent,
                      }} />
                    </div>
                  </div>
                )}
                {/* Log terminal box */}
                <div
                  ref={logRef}
                  style={{
                    ...pixelBox(C.shadow, C.dim, 2, false),
                    padding: "12px",
                    maxHeight: "200px",
                    overflowY: "auto",
                    marginBottom: "24px",
                    fontFamily: FONT,
                    fontSize: "8px",
                  }}
                >
                  {trainingLog.map((entry, i) => (
                    <div key={i} style={{ color: C.dim, padding: "4px 0", lineHeight: 1.6 }}>
                      <span style={{ color: C.panelHi, marginRight: "8px" }}>{entry.time}</span>
                      <span style={{ color: C.ink }}>{entry.msg}</span>
                    </div>
                  ))}
                  {trainingLog.length === 0 && <span style={{ color: C.dim }}>WAITING FOR TRAINING...</span>}
                </div>
              </div>
            )}

            {/* Prediction Display */}
            {prediction && prediction.main && (
              <div style={{ marginBottom: "32px" }}>
                <div style={{ ...card, marginBottom: "24px" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "12px", marginBottom: "8px" }}>
                    <h2 style={{ ...h3, margin: 0 }}>LOTTOMAX NUMBERS</h2>
                    {revealDone && (
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        {prediction.model_trained && (
                          <span style={{ fontFamily: FONT, fontSize: "8px", background: C.panelHi, color: C.ink, padding: "4px 8px" }}>LSTM</span>
                        )}
                        <div style={{
                          height: "8px",
                          width: `${prediction.main.confidence}px`,
                          background: prediction.main.confidence > 50 ? C.good : C.bad,
                        }} />
                        <span style={{ fontFamily: FONT, fontSize: "8px", color: C.ink }}>{prediction.main.confidence}%</span>
                      </div>
                    )}
                  </div>

                  <CatReveal
                    numbers={prediction.main.numbers}
                    runKey={prediction.timestamp}
                    colorFor={getBallColor}
                    onDone={() => setRevealDone(true)}
                  />

                  {revealDone && (
                    <>
                      {showStrategies && (
                        <div style={{ display: "flex", flexDirection: "row", alignItems: "flex-start", justifyContent: "center", gap: "12px", flexWrap: "wrap", marginTop: "8px" }}>
                          {prediction.main.numbers.map((num) => (
                            <div key={num} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "4px" }}>
                              <PixelBall n={num} color={getBallColor(num)} scale={3} />
                              <StrategyBar strategies={prediction.main.strategies} number={num} />
                            </div>
                          ))}
                        </div>
                      )}

                      {/* EV info: why this combo shares less prize money */}
                      {prediction.main.ev_info && (
                        <div style={{
                          ...noticeBox, marginTop: "16px",
                          display: "flex", justifyContent: "center", gap: "18px", flexWrap: "wrap",
                        }}>
                          <span>SMART PICK {prediction.main.ev_info.guard_applied ? "ON" : "OFF"}</span>
                          <span>1–31 numbers: {prediction.main.ev_info.low_count}/{prediction.main.ev_info.max_low}</span>
                          <span>Sum: {prediction.main.ev_info.sum}</span>
                          <span>Share risk: {prediction.main.ev_info.share_risk === "low" ? "LOW" : "HIGH"}</span>
                        </div>
                      )}

                      <div style={{ marginTop: "16px", textAlign: "center" }}>
                        <PixelButton tone="plain" onClick={() => setShowStrategies(!showStrategies)}>
                          {showStrategies ? "HIDE STRATEGY BREAKDOWN" : "SHOW STRATEGY BREAKDOWN"}
                        </PixelButton>
                      </div>

                      {showStrategies && (
                        <div style={{ marginTop: "16px", display: "flex", justifyContent: "center", gap: "16px", fontFamily: FONT, fontSize: "8px", color: C.ink, flexWrap: "wrap" }}>
                          {STRATEGIES.map((s) => (
                            <span key={s.label} style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                              <div style={{ width: "8px", height: "8px", backgroundColor: s.color }} />
                              {s.label.toUpperCase()}
                            </span>
                          ))}
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            )}

            {/* History */}
            {visibleHistory.length > 0 && (
              <div style={{ ...card, marginBottom: "32px" }}>
                <h3 style={h3}>GENERATION HISTORY</h3>
                <div>
                  {visibleHistory.map((h, idx) => {
                    return (
                      <div key={h.id} style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "8px",
                        flexWrap: "wrap",
                        padding: "8px 12px",
                        background: idx === 0 ? C.panelHi : "transparent",
                        opacity: idx === 0 ? 1 : 0.7,
                      }}>
                        <span style={{ fontFamily: FONT, fontSize: "8px", color: C.dim, width: "90px", flexShrink: 0 }}>{h.time}</span>
                        <div style={{ display: "flex", flexDirection: "row", gap: "4px", flexWrap: "nowrap" }}>
                          {h.main.map((n) => (
                            <PixelBall key={n} n={n} color={getBallColor(n)} scale={3} />
                          ))}
                        </div>
                        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "8px", flexShrink: 0 }}>
                          {h.modelTrained && <span style={{ fontFamily: FONT, fontSize: "8px", color: C.accent }}>LSTM</span>}
                          <span style={{ fontFamily: FONT, fontSize: "8px", color: C.dim }}>{h.confidence}%</span>
                        </div>
                      </div>
                    );
                  })}
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
                <FrequencyChart data={frequencies.main_recent} numRange={serverInfo?.pool_size || 52}
                  title={`LottoMax Frequency (Last ${frequencies.recent_window} Draws)`} />

                {/* Hot & Cold */}
                <div style={{ marginTop: "32px", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "16px" }}>
                  <div style={{ ...card, marginBottom: "4px" }}>
                    <h3 style={sectionTitle(C.bad)}>HOT NUMBERS</h3>
                    <BallList entries={Object.entries(frequencies.main_recent).sort((a, b) => b[1] - a[1]).slice(0, 10)} />
                  </div>
                  <div style={{ ...card, marginBottom: "4px" }}>
                    <h3 style={sectionTitle("#3b82f6")}>COLD NUMBERS</h3>
                    <BallList entries={Object.entries(frequencies.main_recent).sort((a, b) => a[1] - b[1]).slice(0, 10)} />
                  </div>
                </div>

                {/* Overdue */}
                <div style={{ ...card, marginTop: "16px" }}>
                  <h3 style={sectionTitle("#a855f7")}>MOST OVERDUE</h3>
                  <BallList suffix=" draws" entries={Object.entries(frequencies.main_gaps).sort((a, b) => b[1] - a[1]).slice(0, 10)} />
                </div>
              </>
            ) : (
              <p style={{ ...muted, textAlign: "center" }}>LOADING ANALYSIS...</p>
            )}

            {/* Seed Analysis Section */}
            <div style={{ ...card, marginTop: "24px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "12px", marginBottom: "16px" }}>
                <h3 style={{ ...sectionTitle(C.accent), margin: 0 }}>SEED/RNG ANALYSIS</h3>
                <PixelButton onClick={runSeedAnalysis}>RUN ANALYSIS</PixelButton>
              </div>
              {seedAnalysis && (
                <div>
                  <p style={body}>
                    Tested {seedAnalysis.tested_seeds} seeds &bull; {seedAnalysis.perfect_matches} perfect &bull; {seedAnalysis.partial_matches} partial (4+)
                  </p>
                  {/* Algo scores */}
                  <div style={{ display: "flex", gap: "16px", marginTop: "12px", flexWrap: "wrap" }}>
                    {Object.entries(seedAnalysis.algo_scores).map(([algo, score]) => (
                      <div key={algo} style={{ flex: 1, minWidth: "120px" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontFamily: FONT, fontSize: "8px", color: C.dim, marginBottom: "8px" }}>
                          <span>{algo.toUpperCase()}</span><span>{score}</span>
                        </div>
                        <div style={{ height: "10px", background: C.shadow }}>
                          <div style={{ height: "100%", background: C.accent, width: `${Math.min(100, score)}%`, transition: "width 300ms steps(6, end)" }} />
                        </div>
                      </div>
                    ))}
                  </div>
                  {/* Top partial matches */}
                  {seedAnalysis.top_partial && seedAnalysis.top_partial.length > 0 && (
                    <div style={{ marginTop: "16px" }}>
                      <h4 style={{ ...muted, textTransform: "uppercase", marginBottom: "8px" }}>TOP PARTIAL MATCHES</h4>
                      <div style={{ maxHeight: "160px", overflowY: "auto" }}>
                        {seedAnalysis.top_partial.map((m, i) => (
                          <div key={i} style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: "8px", padding: "4px 0", fontFamily: FONT, fontSize: "8px", borderBottom: `2px solid ${C.panelHi}` }}>
                            <span style={{ color: C.accent, width: "32px" }}>{m.match}/{7}</span>
                            <span style={{ color: C.dim, width: "40px" }}>{m.algo.toUpperCase()}</span>
                            <span style={{ color: C.ink }}>[{m.predicted.join(", ")}]</span>
                            <span style={{ color: C.dim, marginLeft: "auto" }}>{m.date}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
              {!seedAnalysis && (
                <p style={muted}>Click &quot;RUN ANALYSIS&quot; to test PRNG seeds against historical draws.</p>
              )}
            </div>
          </div>
        )}

        {/* ===================== BACKTEST TAB ===================== */}
        {activeTab === "backtest" && connected && (
          <div>
            <div style={card}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "12px" }}>
                <div style={{ flex: "1 1 320px" }}>
                  <h3 style={{ ...h3, marginBottom: "8px" }}>WALK-FORWARD BACKTEST</h3>
                  <p style={muted}>
                    최근 150회차에 대해 각 전략의 top-7이 실제 당첨번호와 몇 개나 일치했는지 검증합니다.
                    우연 기대값은 7×7/50 = 0.98개입니다.
                  </p>
                </div>
                <PixelButton onClick={runBacktest} disabled={isBacktesting} tone="good">
                  {isBacktesting ? "RUNNING..." : "> RUN BACKTEST"}
                </PixelButton>
              </div>

              {backtest && (
                <div style={{ marginTop: "20px" }}>
                  <div style={{ overflowX: "auto" }}>
                    <table style={{ width: "100%", borderCollapse: "collapse" }}>
                      <thead>
                        <tr>
                          <th style={th}>Strategy</th>
                          <th style={th}>Avg matches</th>
                          <th style={th}>vs random ({backtest.expected_random})</th>
                          <th style={th}>p-value</th>
                          <th style={th}>0 / 1 / 2 / 3+</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(backtest.results).map(([name, r]) => {
                          const diff = r.mean - (r.expected ?? backtest.expected_random);
                          const sig = r.p < 0.05;
                          return (
                            <tr key={name}>
                              <td style={td}>{name}</td>
                              <td style={td}>{r.mean.toFixed(3)}</td>
                              <td style={{ ...td, color: sig ? (diff > 0 ? C.good : C.bad) : C.dim }}>
                                {diff >= 0 ? "+" : ""}{diff.toFixed(3)}{sig ? " *" : ""}
                              </td>
                              <td style={{ ...td, color: C.dim }}>{r.p.toFixed(3)}</td>
                              <td style={{ ...td, color: C.dim }}>
                                {r.dist["0"]} / {r.dist["1"]} / {r.dist["2"]} / {r.dist["3+"]}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <div style={{ ...noticeBox, marginTop: "16px" }}>
                    {backtest.verdict}
                  </div>
                </div>
              )}
              {!backtest && !isBacktesting && (
                <p style={{ ...muted, marginTop: "16px" }}>
                  &quot;Run Backtest&quot;를 눌러 전략별 실제 성능을 확인하세요. (약 5초 소요)
                </p>
              )}
            </div>
          </div>
        )}

        {/* ===================== SIGNAL LAB TAB ===================== */}
        {activeTab === "signal" && connected && <SignalLab API={API} />}

        {/* ===================== EV TAB ===================== */}
        {activeTab === "ev" && connected && <EvTab API={API} />}

        {/* ===================== SETTINGS TAB ===================== */}
        {activeTab === "settings" && (
          <div style={{ display: "flex", flexDirection: "column" }}>
            {connected && <DataPanel API={API} />}

            {/* Training Settings */}
            <div style={card}>
              <h3 style={h3}>TRAINING SETTINGS</h3>
              <div style={{ marginBottom: "16px" }}>
                <label style={{ ...body, display: "block", marginBottom: "12px" }}>LSTM TRAINING EPOCHS</label>
                <input
                  type="range" min="20" max="300" value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value))}
                  style={rangeStyle}
                />
                <div style={{ display: "flex", justifyContent: "space-between", fontFamily: FONT, fontSize: "8px", color: C.dim, marginTop: "8px" }}>
                  <span>FAST (20)</span>
                  <span style={{ color: C.accent }}>{epochs}</span>
                  <span>DEEP (300)</span>
                </div>
              </div>
              <p style={muted}>
                More epochs = deeper pattern learning but longer training time.
                Early stopping prevents overfitting.
              </p>
            </div>

            {/* Strategy Weights */}
            <div style={card}>
              <h3 style={h3}>STRATEGY WEIGHTS</h3>
              <p style={{ ...muted, marginBottom: "16px" }}>
                Adjust each strategy&apos;s influence. Defaults follow the backtest (docs/RESEARCH.md):
                Smart Pick highest, pair lowered, seed off.
              </p>
              {[
                { key: "smart", label: "Smart Pick (EV)", color: C.ink, desc: "인기 조합 회피 — 당첨 확률은 동일, 당첨 시 분배금 기대값 ↑" },
                { key: "gap", label: "Gap Analysis", color: "#a855f7", desc: "Overdue numbers based on gap distributions" },
                { key: "lstm", label: "LSTM Deep Learning", color: C.bad, desc: "Neural network sequential pattern detection" },
                { key: "frequency", label: "Frequency + Recency", color: "#3b82f6", desc: "Hot/cold numbers with time decay" },
                { key: "distribution", label: "Distribution Balance", color: "#f97316", desc: "Range & odd/even equilibrium" },
                { key: "pair", label: "Pair Correlation", color: C.good, desc: "백테스트에서 랜덤보다 유의하게 나빴음 — 기본 가중치 최소화" },
                { key: "seed", label: "Seed/RNG Analysis", color: C.accent, desc: "예측력 없음 검증됨 (기본 0) — 투명성을 위해 유지" },
              ].map((s) => (
                <div key={s.key} style={{ marginBottom: "20px" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "8px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <div style={{ width: "8px", height: "8px", backgroundColor: s.color }} />
                      <span style={{ fontFamily: FONT, fontSize: "10px", color: C.ink }}>{s.label}</span>
                    </div>
                    <span style={{ fontFamily: FONT, fontSize: "10px", color: C.ink }}>{weights[s.key].toFixed(2)}</span>
                  </div>
                  <input
                    type="range" min="0" max="100" value={weights[s.key] * 100}
                    onChange={(e) => setWeights((prev) => ({ ...prev, [s.key]: parseInt(e.target.value) / 100 }))}
                    style={{ ...rangeStyle, accentColor: s.color }}
                  />
                  <p style={{ ...muted, marginTop: "4px" }}>{s.desc}</p>
                </div>
              ))}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "12px", paddingTop: "12px", borderTop: `4px solid ${C.panelHi}` }}>
                <span style={{ fontFamily: FONT, fontSize: "8px", color: C.dim }}>
                  TOTAL: {Object.values(weights).reduce((a, b) => a + b, 0).toFixed(2)}
                </span>
                <PixelButton
                  tone="plain"
                  onClick={() => setWeights({ lstm: 0.15, frequency: 0.15, gap: 0.20, pair: 0.05, distribution: 0.15, seed: 0.0, smart: 0.30 })}
                >
                  RESET DEFAULTS
                </PixelButton>
              </div>
            </div>

            {/* Server Info */}
            <div style={card}>
              <h3 style={h3}>SERVER INFO</h3>
              {serverInfo ? (
                <div style={{ ...muted, display: "flex", flexDirection: "column", gap: "4px" }}>
                  <p>Draws used for statistics: {serverInfo.main_draws} (7/50 era since {serverInfo.era_start}, 7/{serverInfo.pool_size} since {serverInfo.era3_start})</p>
                  <p>Total draws in CSV: {serverInfo.all_draws}</p>
                  <p>TensorFlow: <span style={{ color: serverInfo.tf_available ? C.good : C.bad }}>{serverInfo.tf_available ? "Available" : "Not installed (LSTM disabled)"}</span></p>
                  <p>Main model: <span style={{ color: serverInfo.main_model_loaded ? C.good : C.bad }}>{serverInfo.main_model_loaded ? "Loaded" : "Not trained"}</span></p>
                  {serverInfo.last_trained && <p>Last trained: {new Date(serverInfo.last_trained).toLocaleString()}</p>}
                </div>
              ) : (
                <p style={muted}>NOT CONNECTED</p>
              )}
              <div style={{ marginTop: "16px" }}>
                <PixelButton
                  tone="plain"
                  onClick={async () => {
                    await fetch(`${API}/reload-data`, { method: "POST" });
                    checkServer();
                  }}
                >
                  RELOAD CSV DATA
                </PixelButton>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <footer style={{ marginTop: "48px", textAlign: "center", fontFamily: FONT, fontSize: "8px", color: C.dim, lineHeight: 2 }}>
          <p>LottoMax AI — LSTM + 7-Strategy Ensemble Engine</p>
          <p style={{ marginTop: "4px" }}>
            정직 고지: 추첨은 완전한 무작위이며 어떤 전략도 번호 적중 확률을 높일 수 없습니다 (Backtest 탭에서 직접 확인 가능).
            Smart Pick은 당첨 시 <em>분배금 기대값</em>을 높이는 전략입니다.
          </p>
          <p style={{ marginTop: "4px" }}>For entertainment purposes. Lottery outcomes are not guaranteed.</p>
        </footer>
      </div>
    </div>
  );
}
