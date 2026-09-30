import { useState } from "react";
import { C, FONT, pixelBox } from "./theme";

const TONES = {
  accent: [C.accent, C.accentDk],
  good: [C.good, "#00301a"],
  bad: [C.bad, "#3a0000"],
  plain: [C.panelHi, C.ink],
};

export default function PixelButton({ children, onClick, disabled = false, tone = "accent", style, ...rest }) {
  const [down, setDown] = useState(false);
  const [bg, fg] = TONES[tone] || TONES.accent;
  const pressed = down && !disabled;
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      onMouseDown={() => setDown(true)}
      onMouseUp={() => setDown(false)}
      onMouseLeave={() => setDown(false)}
      {...rest}
      style={{
        ...pixelBox(disabled ? "#4a4a70" : bg, C.ink, 3, !pressed),
        color: disabled ? C.dim : fg,
        fontFamily: FONT,
        fontSize: 11,
        lineHeight: 1.4,
        padding: "12px 16px",
        border: "none",
        cursor: disabled ? "not-allowed" : "pointer",
        transform: pressed ? "translate(3px, 3px)" : "none",
        ...style,
      }}
    >
      {children}
    </button>
  );
}
