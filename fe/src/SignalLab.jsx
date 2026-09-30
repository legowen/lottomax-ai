import { useState, useEffect, useRef } from "react";
import PixelButton from "./pixel/PixelButton";
import { C } from "./pixel/theme";
import { card, muted, body, errorText, h2, h3, th, td, badge } from "./pixel/styles";

export default function SignalLab({ API }) {
  const [status, setStatus] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [quick, setQuick] = useState(true);
  const timer = useRef(null);

  const stopPolling = () => {
    if (timer.current) { clearInterval(timer.current); timer.current = null; }
  };

  const poll = async () => {
    try {
      const res = await fetch(`${API}/signal-lab/status`);
      const s = await res.json();
      setStatus(s);
      if (s.running) {
        if (!timer.current) timer.current = setInterval(poll, 1500);
      } else {
        stopPolling();
        if (s.has_result) {
          const r = await fetch(`${API}/signal-lab/result`);
          if (r.ok) setResult(await r.json());
        }
      }
    } catch (e) {
      stopPolling();
      setError(String(e.message || e));
    }
  };

  useEffect(() => {
    poll();
    return stopPolling;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const run = async () => {
    setError("");
    setResult(null);
    try {
      const res = await fetch(`${API}/signal-lab/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seed: 42, quick }),
      });
      if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
      poll();
    } catch (e) {
      setError(String(e.message || e));
    }
  };

  const running = !!(status && status.running);
  const progress = status && status.progress ? status.progress : null;

  return (
    <div>
      <div style={card}>
        <h2 style={h2}>SIGNAL LAB</h2>
        <p style={{ ...muted, margin: "0 0 16px" }}>
          딥러닝과 통계 검정으로 역대 추첨에 학습 가능한 패턴이 있는지 검증합니다.
          먼저 가짜 신호를 심어 탐지기가 실제로 작동하는지 확인한 뒤(탐지력 검증), 실제 데이터를 검사합니다.
        </p>
        <label style={{ ...muted, display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
          <input type="checkbox" checked={quick} onChange={(e) => setQuick(e.target.checked)} style={{ accentColor: C.accent }} />
          빠른 실행 (부트스트랩/순열 횟수 축소)
        </label>
        <PixelButton disabled={running} onClick={run}>
          {running ? "분석 중..." : "Signal Lab 실행"}
        </PixelButton>
        {progress && running && (
          <div style={{ ...muted, marginTop: 12 }}>
            {progress.stage} — {progress.done}/{progress.total}
          </div>
        )}
        {error && <div style={{ ...errorText, marginTop: 12 }}>{error}</div>}
      </div>

      {result && (
        <>
          <div style={card}>
            <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8, flexWrap: "wrap" }}>
              <span style={badge(result.power_check.ok)}>
                탐지력 검증 {result.power_check.ok ? "통과" : "실패"}
              </span>
              <span style={badge(!result.predictor.beats_baseline && result.battery.n_significant_after_bh === 0)}>
                {result.predictor.beats_baseline || result.battery.n_significant_after_bh > 0 ? "신호 후보 발견 — 재검증 필요" : "학습 가능한 신호 없음"}
              </span>
            </div>
            <p style={{ ...body, margin: 0 }}>{result.verdict}</p>
            <div style={{ ...muted, marginTop: 8 }}>
              Era 2 · {result.n_draws}회 · 마지막 추첨 {result.last_draw_date}
            </div>
          </div>

          <div style={card}>
            <h3 style={h3}>통계 검정 배터리</h3>
            <div style={{ ...muted, marginBottom: 8 }}>
              검정 {result.battery.n_tests}개 중 BH 보정 후 유의 {result.battery.n_significant_after_bh}개
              (우연히 기대되는 수 약 {result.battery.expected_false_positives})
            </div>
            <div style={{ maxHeight: 280, overflowY: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr>
                    <th style={th}>검정</th>
                    <th style={th}>통계량</th>
                    <th style={th}>p</th>
                    <th style={th}>p (BH)</th>
                    <th style={th}>유의</th>
                  </tr>
                </thead>
                <tbody>
                  {result.battery.tests.map((t) => (
                    <tr key={t.name}>
                      <td style={td}>{t.name}</td>
                      <td style={td}>{Number(t.statistic).toFixed(3)}</td>
                      <td style={td}>{Number(t.p_value).toFixed(3)}</td>
                      <td style={td}>{Number(t.p_adj_bh).toFixed(3)}</td>
                      <td style={td}>{t.significant ? "예" : "아니오"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div style={card}>
            <h3 style={h3}>딥러닝 예측기 · 판별기</h3>
            <div style={{ ...muted }}>
              예측기({result.predictor.model}): 손실 {Number(result.predictor.mean_logloss).toFixed(4)} vs 상수 기준선{" "}
              {Number(result.predictor.baseline_logloss).toFixed(4)} · 차이 {Number(result.predictor.delta).toFixed(4)}{" "}
              (95% CI {Number(result.predictor.ci95[0]).toFixed(4)} ~ {Number(result.predictor.ci95[1]).toFixed(4)}, p={Number(result.predictor.p_value).toFixed(3)})
              <br />
              판별기 AUC {Number(result.discriminator.auc).toFixed(3)} (95% CI {Number(result.discriminator.ci95[0]).toFixed(3)} ~{" "}
              {Number(result.discriminator.ci95[1]).toFixed(3)}, p={Number(result.discriminator.p_value).toFixed(3)}) — 0.5에 가까울수록 패턴 없음
            </div>
          </div>

          <div style={card}>
            <h3 style={h3}>탐지력 검증 (가짜 신호 심기)</h3>
            {result.power_check.planted.map((p) => (
              <div key={p.effect} style={{ ...muted, marginBottom: 4 }}>
                {p.effect}: <span style={badge(p.detected)}>{p.detected ? "검출됨" : "놓침"}</span> (p_BH={Number(p.p_adj_bh).toFixed(4)})
              </div>
            ))}
            <div style={{ ...muted, marginTop: 8 }}>
              순수 무작위 데이터 오탐률 {(Number(result.power_check.false_positive_rate_null) * 100).toFixed(1)}%
            </div>
          </div>
        </>
      )}
    </div>
  );
}
