import Sprite from "./Sprite";
import { BALL, ballPalette } from "./sprites";
import { FONT } from "./theme";

const OUTLINE = "-2px 0 #15151f, 2px 0 #15151f, 0 -2px #15151f, 0 2px #15151f";

export default function PixelBall({ n, color, scale = 5, style }) {
  const size = BALL.length * scale;
  return (
    <div style={{ position: "relative", width: size, height: size, ...style }}>
      <Sprite rows={BALL} palette={ballPalette(color)} scale={scale} />
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: size,
          height: size,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: FONT,
          fontSize: Math.round(scale * 2.6),
          color: "#ffffff",
          textShadow: OUTLINE,
        }}
      >
        {n}
      </div>
    </div>
  );
}
