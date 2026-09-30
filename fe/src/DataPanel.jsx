import { useState, useEffect } from "react";
import PixelButton from "./pixel/PixelButton";
import { C, FONT, pixelBox } from "./pixel/theme";
import { card, muted, h3 } from "./pixel/styles";

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
      <h3 style={h3}>데이터 업데이트</h3>
      {info && (
        <div style={{ ...muted, marginBottom: 12 }}>
          마지막 회차 {info.last_draw_number} ({info.last_draw_date}) · {info.days_since_last}일 경과 ·{" "}
          <span style={{ color: stale ? C.accent : C.good }}>
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
          ...pixelBox(C.shadow, C.dim, 2, false),
          width: "calc(100% - 4px)", padding: "10px 12px", border: "none", outline: "none",
          color: C.ink, fontSize: 8, lineHeight: 1.8, fontFamily: FONT, boxSizing: "border-box",
        }}
      />
      <div style={{ marginTop: 12 }}>
        <PixelButton disabled={busy} onClick={submit}>{busy ? "추가 중..." : "회차 추가"}</PixelButton>
      </div>
      {message && <pre style={{ ...muted, whiteSpace: "pre-wrap", marginTop: 12 }}>{message}</pre>}
    </div>
  );
}
