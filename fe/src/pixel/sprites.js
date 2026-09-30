// Bomi: black cat, small white chest tuft, emerald eyes.
// NO pink anywhere (nose is grey, inner ears are dark grey, no tongue, no paw pads).
export const CAT_PALETTE = {
  K: "#15151f", // black fur
  G: "#3d3d5c", // fur highlight / inner ear / mouth
  W: "#f4f4f4", // white chest
  E: "#22d67f", // emerald eye
  P: "#08080d", // pupil
  H: "#ffffff", // eye shine
  N: "#57577a", // nose (grey)
  w: "#8a8aa8", // whiskers
};

const CAT_IDLE = [
  "...K..............K...",
  "...KK............KK...",
  "...KGK..........KGK...",
  "...KGGK........KGGK...",
  "...KKKKGGGKKGGGKKKK...",
  "..KKKKKKKKKKKKKKKKKK..",
  "..KKKHEEKKKKKKHEEKKK..",
  "..KKKEPEKKKKKKEPEKKK..",
  "wwKKKEEEKKKKKKEEEKKKww",
  "..KKKKKKKKNNKKKKKKKK..",
  "wwKKKKKKKKGGKKKKKKKKww",
  "...KKKKKKGKKGKKKKKK...",
  ".....KKKKKKKKKKKK.....",
  "......KKKWWWWKKK......",
  "GK...KGKKWWWWKKGK.....",
  "KK...KKKKKWWKKKKK.....",
  "KK...KKKKKKKKKKKK.....",
  ".KK..KKKKKKKKKKKK.....",
  ".KKK.KKKKK..KKKKK.....",
  "..KKKKKKKK..KKKKK.....",
  ".....KKGKK..KKGKK.....",
  "......................",
];

const patch = (base, cells) => {
  const g = base.map((r) => r.split(""));
  cells.forEach(([y, x, c]) => {
    g[y][x] = c;
  });
  return g.map((r) => r.join(""));
};

const eyeCells = (kind) =>
  [5, 14].flatMap((x0) =>
    kind === "blink"
      ? [
          [6, x0, "K"], [6, x0 + 1, "K"], [6, x0 + 2, "K"],
          [7, x0, "G"], [7, x0 + 1, "G"], [7, x0 + 2, "G"],
          [8, x0, "K"], [8, x0 + 1, "K"], [8, x0 + 2, "K"],
        ]
      : [
          [6, x0, "K"], [6, x0 + 1, "E"], [6, x0 + 2, "K"],
          [7, x0, "E"], [7, x0 + 1, "K"], [7, x0 + 2, "E"],
          [8, x0, "K"], [8, x0 + 1, "K"], [8, x0 + 2, "K"],
        ]
  );

// right arm raised toward the ball tray
const SWIPE_CELLS = [
  [16, 17, "K"], [16, 18, "K"],
  [15, 17, "K"], [15, 18, "K"], [15, 19, "K"],
  [14, 18, "K"], [14, 19, "K"], [14, 20, "K"],
  [13, 19, "K"], [13, 20, "K"], [13, 21, "K"],
  [12, 19, "K"], [12, 20, "K"], [12, 21, "K"],
  [11, 20, "K"], [11, 21, "G"],
];

export const CAT_FRAMES = {
  idle: CAT_IDLE,
  blink: patch(CAT_IDLE, eyeCells("blink")),
  happy: patch(CAT_IDLE, eyeCells("happy")),
  swipe: patch(CAT_IDLE, SWIPE_CELLS),
};

// 12x12 ball: O outline, L base, D shade, H highlight
export const BALL = [
  "....OOOO....",
  "..OOLLLLOO..",
  ".OLHHLLLLLO.",
  ".OHHLLLLLLO.",
  "OLHLLLLLLLDO",
  "OLLLLLLLLDDO",
  "OLLLLLLLDDDO",
  "OLLLLLLDDDDO",
  ".OLLLLDDDDO.",
  ".OLLLDDDDDO.",
  "..OODDDDOO..",
  "....OOOO....",
];

const clamp = (v) => Math.max(0, Math.min(255, Math.round(v)));
const parse = (hex) => {
  const h = hex.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
};
const toHex = (rgb) => "#" + rgb.map((v) => clamp(v).toString(16).padStart(2, "0")).join("");
export const shade = (hex, f) => toHex(parse(hex).map((v) => v * f));
export const mix = (hex, f) => toHex(parse(hex).map((v) => v + (255 - v) * f));

export const ballPalette = (hex) => ({
  O: shade(hex, 0.4),
  L: hex,
  D: shade(hex, 0.75),
  H: mix(hex, 0.75),
});
