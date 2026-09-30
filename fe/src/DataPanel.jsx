import { useState, useEffect } from "react";

const card = { background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 16, padding: 20, marginBottom: 16 };
const muted = { color: "#94a3b8", fontSize: 13 };
const btnStyle = (disabled) => ({
  padding: "10px 18px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.2)",
  background: disabled ? "rgba(255,255,255,0.05)" : "linear-gradient(135deg,#f59e0b,#ef4444)",
  color: "#ffffff", fontWeight: 700, cursor: disabled ? "not-allowed" : "pointer", opacity: disabled ? 0.6 : 1,
});

// One draw per line: draw number, date, numbers 1..7, bonus  e.g. 1274,2026-09-29,3,9,14,22,31,40,47,12
function parseLines(text) {
  const draws = [];
  const errors = [];
  text.split("\n").map((l) => l.trim()).filter(Boolean).forEach((line, idx) => {
    const p = line.split(/[,\s]+/).filter(Boolean);
    if (p.length !== 10) { errors.push(`${idx + 1}번째 줄: 값이 10개여야 합니다 (회차, 날짜, 번호 7개, 보너스)`); return; }
    const nums = p.slice(2, 9).map(Number);
    if (nums.some((n) => !Number.isInteger(n))) { errors.push(`${idx + 1}번째 줄: 번호가 정수가 아닙니다`); return; }
    draws.push({ draw_number: Number(p[0]), date: p[1], numbers: nums, bonus: Number(p[9]) });
  });
  return { draws, errors };
}

export default function DataPanel({ API }) {
  const [info, setInfo] = useState(null);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  const load = async () => {
    try {
      const res = await fetch(`${API}/data/status`);
      if (res.ok) setInfo(await res.json());
    } catch (e) {
      setMessage(String(e.message || e));
    }
  };
  useEffect(() => { load(); // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const submit = async () => {
    const { draws, errors } = parseLines(text);
    if (errors.length) { setMessage(errors.join("\n")); return; }
    if (!draws.length) { setMessage("추가할 회차가 없습니다."); return; }
    setBusy(true);
    setMessage("");
    try {
      const res = await fetch(`${API}/data/append`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ draws }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body));
      const rejected = (body.rejected || []).map((r) => `${r.row}번째 줄 (회차 ${r.draw_number}) 제외: ${r.error}`);
      setMessage([`${body.added ?? draws.length}회차를 추가했습니다. LSTM 재학습을 권장합니다.`, ...rejected].join("\n"));
      setText("");
      load();
    } catch (e) {
      setMessage(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const stale = info && info.estimated_missing_draws > 0;

  return (
    <div style={card}>
      <h3 style={{ margin: "0 0 8px", color: "#ffffff", fontSize: 16 }}>데이터 업데이트</h3>
      {info && (
        <div style={{ ...muted, marginBottom: 8, lineHeight: 1.7 }}>
          마지막 회차 {info.last_draw_number} ({info.last_draw_date}) · {info.days_since_last}일 경과 ·{" "}
          <span style={{ color: stale ? "#fbbf24" : "#4ade80" }}>
            {stale ? `누락 추정 약 ${info.estimated_missing_draws}회` : "최신 상태"}
          </span>
        </div>
      )}
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={"회차,날짜,번호1~7,보너스 (한 줄에 한 회차)\n1274,2026-09-29,3,9,14,22,31,40,47,12"}
        rows={5}
        style={{
          width: "100%", padding: "10px 12px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.15)",
          background: "rgba(0,0,0,0.3)", color: "#ffffff", fontSize: 13, fontFamily: "monospace", boxSizing: "border-box",
        }}
      />
      <div style={{ marginTop: 10 }}>
        <button style={btnStyle(busy)} disabled={busy} onClick={submit}>{busy ? "추가 중..." : "회차 추가"}</button>
      </div>
      {message && <pre style={{ ...muted, whiteSpace: "pre-wrap", marginTop: 10 }}>{message}</pre>}
    </div>
  );
}
