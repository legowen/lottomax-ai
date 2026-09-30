import { useState } from "react";

const card = { background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 16, padding: 20, marginBottom: 16 };
const muted = { color: "#94a3b8", fontSize: 13 };
const labelStyle = { display: "block", color: "#94a3b8", fontSize: 12, marginBottom: 4 };
const inputStyle = {
  width: "100%", padding: "10px 12px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.15)",
  background: "rgba(0,0,0,0.3)", color: "#ffffff", fontSize: 14, boxSizing: "border-box",
};
const btnStyle = (disabled) => ({
  padding: "10px 18px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.2)",
  background: disabled ? "rgba(255,255,255,0.05)" : "linear-gradient(135deg,#f59e0b,#ef4444)",
  color: "#ffffff", fontWeight: 700, cursor: disabled ? "not-allowed" : "pointer", opacity: disabled ? 0.6 : 1,
});
const money = (v) =>
  v === null || v === undefined ? "-" : "$" + Number(v).toLocaleString("en-CA", { maximumFractionDigits: 2 });

export default function EvTab({ API }) {
  const [count, setCount] = useState(5);
  const [tickets, setTickets] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState(null);
  const [ev, setEv] = useState(null);
  const [hist, setHist] = useState(null);
  const [form, setForm] = useState({
    jackpot: 50000000, tickets_sold: 25000000, ticket_price: 6, lines_per_ticket: 4, other_prizes_ev: 0,
  });

  const post = async (path, body) => {
    const res = await fetch(`${API}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    return res.json();
  };

  const drawTickets = async () => {
    setBusy(true);
    setError("");
    try {
      const r = await post("/predict-batch", { count });
      setTickets(r.tickets);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const calc = async (numbers) => {
    setBusy(true);
    setError("");
    setSelected(numbers);
    try {
      const body = {
        jackpot: Number(form.jackpot), tickets_sold: Number(form.tickets_sold),
        ticket_price: Number(form.ticket_price), lines_per_ticket: Number(form.lines_per_ticket),
        other_prizes_ev: Number(form.other_prizes_ev), numbers,
      };
      const [e, h] = await Promise.all([
        post("/ev/estimate", body),
        numbers ? post("/history-check", { numbers }) : Promise.resolve(null),
      ]);
      setEv(e);
      setHist(h);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setBusy(false);
    }
  };

  const setField = (key) => (e) => setForm({ ...form, [key]: e.target.value });

  return (
    <div>
      <div style={card}>
        <h2 style={{ margin: "0 0 8px", color: "#ffffff", fontSize: 20 }}>Smart Pick v2</h2>
        <p style={{ ...muted, margin: "0 0 12px", lineHeight: 1.6 }}>
          모든 7개 조합의 당첨 확률은 동일합니다. 이 도구는 남들이 덜 고르는 조합을 뽑아 당첨 시 공동 당첨자를 줄이는 것을 목표로 하며,
          인기도 비율은 가정 기반 추정치입니다.
        </p>
        <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
          <div style={{ width: 100 }}>
            <label style={labelStyle}>티켓 수 (1~10)</label>
            <input style={inputStyle} type="number" min={1} max={10} value={count}
              onChange={(e) => setCount(Math.max(1, Math.min(10, Number(e.target.value) || 1)))} />
          </div>
          <button style={btnStyle(busy)} disabled={busy} onClick={drawTickets}>번호 뽑기</button>
        </div>
        {error && <div style={{ color: "#f87171", marginTop: 12, fontSize: 13 }}>{error}</div>}

        {tickets.map((t, i) => (
          <div key={i} style={{
            display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap",
            marginTop: 12, padding: "12px 14px", borderRadius: 12,
            background: selected && selected.join() === t.numbers.join() ? "rgba(245,158,11,0.12)" : "rgba(0,0,0,0.25)",
            border: "1px solid rgba(255,255,255,0.08)",
          }}>
            <div>
              <div style={{ color: "#ffffff", fontWeight: 700, fontSize: 18, letterSpacing: 1 }}>
                {t.numbers.map((n) => String(n).padStart(2, "0")).join("  ")}
              </div>
              <div style={muted}>
                랜덤 티켓 대비 예상 공동당첨 ×{Number(t.popularity_ratio).toFixed(2)} (추정) · ≤31 {t.low_count}개 · 합 {t.sum}
              </div>
            </div>
            <button style={btnStyle(busy)} disabled={busy} onClick={() => calc(t.numbers)}>이 번호로 EV 계산</button>
          </div>
        ))}
      </div>

      <div style={card}>
        <h2 style={{ margin: "0 0 8px", color: "#ffffff", fontSize: 20 }}>잭팟 EV 계산기</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12 }}>
          <div><label style={labelStyle}>잭팟 (CAD)</label><input style={inputStyle} type="number" value={form.jackpot} onChange={setField("jackpot")} /></div>
          <div><label style={labelStyle}>예상 판매 티켓 수(줄)</label><input style={inputStyle} type="number" value={form.tickets_sold} onChange={setField("tickets_sold")} /></div>
          <div><label style={labelStyle}>티켓 가격</label><input style={inputStyle} type="number" value={form.ticket_price} onChange={setField("ticket_price")} /></div>
          <div><label style={labelStyle}>티켓당 줄 수</label><input style={inputStyle} type="number" value={form.lines_per_ticket} onChange={setField("lines_per_ticket")} /></div>
          <div><label style={labelStyle}>하위 등급 기대값(줄당)</label><input style={inputStyle} type="number" value={form.other_prizes_ev} onChange={setField("other_prizes_ev")} /></div>
        </div>
        <div style={{ marginTop: 12 }}>
          <button style={btnStyle(busy)} disabled={busy} onClick={() => calc(selected)}>
            {selected ? "선택한 번호로 다시 계산" : "랜덤 티켓 기준 계산"}
          </button>
        </div>
        <p style={{ ...muted, marginTop: 12, lineHeight: 1.6 }}>
          기본값은 예시입니다. 잭팟 금액, 판매량, 가격은 실제 값으로 바꿔 입력하세요. 잭팟 상한, 이월, 하위 등급 고정 상금은 모델링하지 않습니다.
        </p>

        {ev && (
          <div style={{ marginTop: 8, color: "#e2e8f0", fontSize: 14, lineHeight: 1.9 }}>
            <div>조합 수: {Number(ev.combinations).toLocaleString()} · 줄당 비용 {money(ev.cost_per_line)}</div>
            <div>
              랜덤 티켓: 당첨 시 기대 몫 {money(ev.random.share_if_win)} · 잭팟 EV {money(ev.random.ev_jackpot)} · 순 EV {money(ev.net_ev_random)}
            </div>
            {ev.chosen && (
              <div>
                선택 번호: 당첨 시 기대 몫 {money(ev.chosen.share_if_win)} · 잭팟 EV {money(ev.chosen.ev_jackpot)} · 순 EV {money(ev.net_ev_chosen)}
              </div>
            )}
            {hist && (
              <div style={muted}>
                역대 최대 겹침 {hist.max_overlap}개 ·{" "}
                {hist.exact_match ? `${hist.exact_match.draw_number}회(${hist.exact_match.date})와 완전 일치` : "과거 당첨 조합과 일치하지 않음"}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
