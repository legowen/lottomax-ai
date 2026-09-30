// NES-style limited palette + pixel helpers (inline styles only)
export const FONT = "'Press Start 2P', 'Malgun Gothic', 'Apple SD Gothic Neo', monospace";

export const C = {
  bg: "#14143a",
  panel: "#25256b",
  panelHi: "#3a3a8c",
  ink: "#f8f8f8",
  dim: "#a8a8d8",
  accent: "#fcc03c",
  accentDk: "#3a2a00",
  good: "#3ad880",
  bad: "#f86060",
  shadow: "#0a0a20",
};

// Notched pixel border (NES dialog-box look) + hard drop shadow. No border-radius, ever.
export const pixelBox = (bg = C.panel, border = C.ink, t = 4, drop = true) => ({
  background: bg,
  margin: t,
  position: "relative",
  borderRadius: 0,
  boxShadow: [
    `${t}px 0 0 0 ${border}`,
    `-${t}px 0 0 0 ${border}`,
    `0 ${t}px 0 0 ${border}`,
    `0 -${t}px 0 0 ${border}`,
    ...(drop ? [`${t * 2}px ${t * 2}px 0 0 ${C.shadow}`] : []),
  ].join(", "),
});

// Fallback ball colours by number range. If the existing code already has a
// per-range colour mapping, REUSE THAT (converted to a solid hex) instead.
export const ballColor = (n) =>
  n <= 10 ? "#e84a4a" : n <= 20 ? "#f2a23a" : n <= 30 ? "#e8d43a" : n <= 40 ? "#3ac86a" : "#3a8ff2";
