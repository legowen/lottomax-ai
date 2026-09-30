// Shared pixel-UI style fragments for the tab components (inline styles only).
import { C, FONT, pixelBox } from "./theme";

export const card = { ...pixelBox(C.panel, C.ink, 4), margin: "4px 4px 24px", padding: 20 };

export const h2 = { margin: "0 0 12px", fontFamily: FONT, fontSize: 16, color: C.accent, lineHeight: 1.5, textTransform: "uppercase" };
export const h3 = { margin: "0 0 12px", fontFamily: FONT, fontSize: 12, color: C.ink, lineHeight: 1.6, textTransform: "uppercase" };

export const muted = { fontFamily: FONT, fontSize: 8, color: C.dim, lineHeight: 1.9 };
export const body = { fontFamily: FONT, fontSize: 10, color: C.ink, lineHeight: 1.9 };
export const errorText = { fontFamily: FONT, fontSize: 8, color: C.bad, lineHeight: 1.8 };

export const labelStyle = { display: "block", fontFamily: FONT, fontSize: 8, color: C.dim, marginBottom: 8, lineHeight: 1.6 };

export const inputStyle = {
  ...pixelBox(C.shadow, C.dim, 2, false),
  width: "calc(100% - 4px)",
  boxSizing: "border-box",
  padding: "10px 12px",
  border: "none",
  outline: "none",
  color: C.ink,
  fontFamily: FONT,
  fontSize: 10,
};

export const th = { textAlign: "left", padding: "8px", fontFamily: FONT, fontSize: 8, color: C.dim, borderBottom: `2px solid ${C.dim}`, textTransform: "uppercase" };
export const td = { padding: "8px", fontFamily: FONT, fontSize: 8, color: C.ink, borderBottom: `2px solid ${C.panelHi}`, lineHeight: 1.6 };

export const badge = (ok) => ({
  display: "inline-block",
  padding: "4px 8px",
  fontFamily: FONT,
  fontSize: 8,
  lineHeight: 1.5,
  background: ok ? C.good : C.bad,
  color: ok ? "#00301a" : "#3a0000",
});
