import { C, FONT, pixelBox } from "./theme";

export default function PixelTabs({ tabs, active, onChange }) {
  return (
    <div style={{ display: "flex", gap: 12, flexWrap: "wrap", margin: "16px 4px" }}>
      {tabs.map((t) => {
        const on = t === active;
        return (
          <button
            key={t}
            type="button"
            onClick={() => onChange(t)}
            style={{
              ...pixelBox(on ? C.accent : C.panel, C.ink, 3),
              color: on ? C.accentDk : C.ink,
              fontFamily: FONT,
              fontSize: 10,
              padding: "10px 12px",
              border: "none",
              cursor: "pointer",
              textTransform: "uppercase",
            }}
          >
            {on ? "> " : ""}
            {t}
          </button>
        );
      })}
    </div>
  );
}
