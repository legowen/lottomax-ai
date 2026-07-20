// E2E smoke test: drives the built frontend against a live backend.
//
// Prereqs:  backend on :8000  (cd be && python -m uvicorn app:app --port 8000)
//           preview on :4173  (cd fe && npm run build && npx vite preview --port 4173)
// Run:      node e2e/smoke.mjs
// Optional: CHROMIUM_PATH=/path/to/chromium  SHOT_DIR=/tmp
import { chromium } from "@playwright/test";

const SHOT_DIR = process.env.SHOT_DIR || ".";
const launchOpts = process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {};
const browser = await chromium.launch(launchOpts);
const page = await browser.newPage({ viewport: { width: 1100, height: 900 } });
const errors = [];
page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
page.on("console", (m) => { if (m.type() === "error") errors.push("console: " + m.text()); });

await page.goto("http://localhost:4173", { waitUntil: "networkidle" });
await page.waitForSelector("text=Connected", { timeout: 15000 });
console.log("✅ connected to backend");

// Generate numbers -> prediction + EV panel render
await page.click("text=Generate Numbers");
await page.waitForSelector("text=Smart Pick ON", { timeout: 20000 });
console.log("✅ prediction rendered, EV panel visible");
await page.screenshot({ path: `${SHOT_DIR}/e2e_generate.png` });

// Backtest tab -> results table + verdict
await page.click("nav >> text=backtest");
await page.click("text=Run Backtest");
await page.waitForSelector("table", { timeout: 60000 });
const rows = await page.locator("tbody tr").count();
if (rows < 6) throw new Error(`expected 6 backtest rows, got ${rows}`);
console.log("✅ backtest table rows:", rows);
await page.screenshot({ path: `${SHOT_DIR}/e2e_backtest.png` });

// Analysis tab renders without crash
await page.click("nav >> text=analysis");
await page.waitForSelector("text=Hot Numbers", { timeout: 15000 });
console.log("✅ analysis tab ok");

// Settings shows all 7 strategies incl. Smart Pick
await page.click("nav >> text=settings");
await page.waitForSelector("text=Smart Pick (EV)", { timeout: 10000 });
console.log("✅ settings shows Smart Pick");
await page.screenshot({ path: `${SHOT_DIR}/e2e_settings.png` });

if (errors.length) { console.log("❌ page errors:", errors); process.exit(1); }
console.log("✅ E2E PASS — no page errors");
await browser.close();
