import { useEffect, useRef, useState } from "react";
import Sprite from "./Sprite";
import PixelBall from "./PixelBall";
import PixelButton from "./PixelButton";
import { CAT_FRAMES, CAT_PALETTE } from "./sprites";
import { C, FONT, pixelBox } from "./theme";

const STEP_MS = 1100; // time per ball
const BALL_SCALE = 5;
const BALL_PX = 12 * BALL_SCALE;
const SLOT = BALL_PX + 12;
const OFF = (SLOT - BALL_PX) / 2;

// 0 hidden · 1 above the slot · 2 dropping in · 3 small bounce up · 4 settled
const BALL_PHASE = {
  0: { opacity: 0, transform: "translateY(-64px)" },
  1: { opacity: 1, transform: "translateY(-64px)" },
  2: { opacity: 1, transform: "translateY(0px)", transition: "transform 280ms steps(5, end)" },
  3: { opacity: 1, transform: "translateY(-10px)", transition: "transform 120ms steps(2, end)" },
  4: { opacity: 1, transform: "translateY(0px)", transition: "transform 120ms steps(2, end)" },
};

export default function CatReveal({ numbers, runKey, colorFor, scale = 6, onDone }) {
  const total = numbers.length;
  const [frame, setFrame] = useState("idle");
  const [phases, setPhases] = useState(() => Array(total).fill(0));
  const [count, setCount] = useState(0);
  const timers = useRef([]);
  const finished = useRef(false);
  const doneRef = useRef(onDone);

  useEffect(() => {
    doneRef.current = onDone;
  });

  const later = (fn, ms) => {
    timers.current.push(setTimeout(fn, ms));
  };
  const clearAll = () => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
  };
  const setPhase = (i, p) =>
    setPhases((prev) => {
      const next = prev.slice();
      next[i] = p;
      return next;
    });
  const finish = () => {
    if (finished.current) return;
    finished.current = true;
    if (doneRef.current) doneRef.current();
  };

  const skip = () => {
    clearAll();
    setPhases(Array(total).fill(4));
    setCount(total);
    setFrame("happy");
    later(() => setFrame("idle"), 1400);
    finish();
  };

  // (re)start the reveal whenever a new prediction arrives
  useEffect(() => {
    clearAll();
    finished.current = false;
    setPhases(Array(total).fill(0));
    setCount(0);
    setFrame("idle");

    const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) {
      setPhases(Array(total).fill(4));
      setCount(total);
      setFrame("happy");
      finish();
      return clearAll;
    }

    const START = 500;
    for (let i = 0; i < total; i += 1) {
      const t0 = START + i * STEP_MS;
      later(() => setFrame("swipe"), t0);
      later(() => setPhase(i, 1), t0 + 200);
      later(() => {
        setPhase(i, 2);
        setCount(i + 1);
      }, t0 + 240);
      later(() => setPhase(i, 3), t0 + 560);
      later(() => {
        setPhase(i, 4);
        setFrame("idle");
      }, t0 + 700);
    }
    const END = START + total * STEP_MS;
    later(() => {
      setFrame("happy");
      finish();
    }, END);
    later(() => setFrame("idle"), END + 1600);
    return clearAll;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runKey]);

  // idle blinking
  useEffect(() => {
    let id;
    const loop = () => {
      id = setTimeout(() => {
        setFrame((f) => (f === "idle" ? "blink" : f));
        id = setTimeout(() => {
          setFrame((f) => (f === "blink" ? "idle" : f));
          loop();
        }, 140);
      }, 2200 + Math.random() * 2000);
    };
    loop();
    return () => clearTimeout(id);
  }, []);

  const label =
    count >= total ? "ALL SET! GOOD LUCK!" : count === 0 ? "BOMI IS THINKING..." : `BOMI PICKS #${count}...`;

  return (
    <div style={{ display: "flex", gap: 24, alignItems: "flex-end", flexWrap: "wrap", justifyContent: "center", padding: 16 }}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
        <Sprite rows={CAT_FRAMES[frame]} palette={CAT_PALETTE} scale={scale} />
        <div style={{ fontFamily: FONT, fontSize: 10, color: C.ink }}>BOMI</div>
      </div>

      <div style={{ flex: "1 1 340px", minWidth: 280 }}>
        <div
          style={{
            ...pixelBox(C.panel, C.ink, 4),
            padding: "12px 14px",
            marginBottom: 20,
            fontFamily: FONT,
            fontSize: 10,
            color: C.ink,
            lineHeight: 1.6,
          }}
        >
          {label}
        </div>

        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          {numbers.map((n, i) => (
            <div
              key={`${runKey}-${i}`}
              style={{
                ...pixelBox(C.shadow, C.dim, 2, false),
                margin: 2,
                width: SLOT,
                height: SLOT,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              {(phases[i] || 0) === 0 && (
                <span style={{ fontFamily: FONT, fontSize: 14, color: C.dim }}>?</span>
              )}
              <PixelBall
                n={n}
                color={colorFor(n)}
                scale={BALL_SCALE}
                style={{ position: "absolute", left: OFF, top: OFF, ...BALL_PHASE[phases[i] || 0] }}
              />
            </div>
          ))}
        </div>

        <div style={{ marginTop: 20, minHeight: 48 }}>
          {count < total && (
            <PixelButton tone="plain" onClick={skip}>
              SKIP &gt;&gt;
            </PixelButton>
          )}
        </div>
      </div>
    </div>
  );
}
