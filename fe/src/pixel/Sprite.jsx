// Renders a character grid as crisp SVG rects (horizontal runs merged).
export default function Sprite({ rows, palette, scale = 6, style }) {
  const h = rows.length;
  const w = rows[0].length;
  const rects = [];
  rows.forEach((row, y) => {
    let x = 0;
    while (x < w) {
      const c = row[x];
      if (c === ".") {
        x += 1;
        continue;
      }
      let run = 1;
      while (x + run < w && row[x + run] === c) run += 1;
      rects.push(<rect key={`${x}-${y}`} x={x} y={y} width={run} height={1} fill={palette[c]} />);
      x += run;
    }
  });
  return (
    <svg
      width={w * scale}
      height={h * scale}
      viewBox={`0 0 ${w} ${h}`}
      shapeRendering="crispEdges"
      style={{ display: "block", imageRendering: "pixelated", ...style }}
    >
      {rects}
    </svg>
  );
}
