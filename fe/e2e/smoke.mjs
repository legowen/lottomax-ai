// E2E smoke test: drives the built frontend against a live backend.
//
// Prereqs:  backend on :8000  (cd be && LOTTOMAX_CORS_ORIGINS=http://localhost:4173 python -m uvicorn app:app --port 8000)
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
await page.waitForSelector("text=CONNECTED", { timeout: 15000 });
console.log("✅ connected to backend");

// Tabs are PixelTabs buttons; the active one is prefixed with "> "
const tab = (name) => page.getByRole("button", { name: new RegExp(`^(> )?${name}$`, "i") });

// Generate numbers -> Bomi reveals the balls, SKIP finishes, EV panel renders afterwards
await page.click("text=GENERATE NUMBERS");
await page.waitForSelector("text=BOMI", { timeout: 20000 });
await page.waitForSelector("text=/BOMI PICKS #\\d/", { timeout: 10000 });
console.log("✅ Bomi reveal started");
await page.click("text=/^SKIP/");
await page.waitForSelector("text=ALL SET! GOOD LUCK!", { timeout: 5000 });
await page.waitForSelector("text=SMART PICK ON", { timeout: 20000 });
console.log("✅ SKIP works, prediction rendered, EV panel visible");
await page.screenshot({ path: `${SHOT_DIR}/e2e_generate.png` });

// Backtest tab -> results table + verdict
await tab("backtest").click();
await page.click("text=RUN BACKTEST");
await page.waitForSelector("table", { timeout: 60000 });
const rows = await page.locator("tbody tr").count();
if (rows < 6) throw new Error(`expected 6 backtest rows, got ${rows}`);
console.log("✅ backtest table rows:", rows);
await page.screenshot({ path: `${SHOT_DIR}/e2e_backtest.png` });

// Analysis tab renders without crash
await tab("analysis").click();
await page.waitForSelector("text=HOT NUMBERS", { timeout: 15000 });
console.log("✅ analysis tab ok");

// Signal Lab tab renders (run button + honesty copy), without starting the long job
await tab("signal").click();
await page.waitForSelector("text=Signal Lab 실행", { timeout: 10000 });
await page.waitForSelector("text=탐지력 검증", { timeout: 10000 });
console.log("✅ signal tab renders");
await page.screenshot({ path: `${SHOT_DIR}/e2e_signal.png` });

// EV tab: draw Smart Pick v2 tickets, compute EV for the first one
await tab("ev").click();
await page.waitForSelector("text=SMART PICK V2", { timeout: 10000 });
await page.click("text=번호 뽑기");
await page.waitForSelector("text=이 번호로 EV 계산", { timeout: 20000 });
const ticketRows = await page.locator("text=이 번호로 EV 계산").count();
if (ticketRows !== 5) throw new Error(`expected 5 tickets, got ${ticketRows}`);
await page.locator("text=이 번호로 EV 계산").first().click();
await page.waitForSelector("text=잭팟 EV", { timeout: 20000 });
console.log("✅ ev tab: tickets + EV calculation rendered");
await page.screenshot({ path: `${SHOT_DIR}/e2e_ev.png` });

// Settings shows all 7 strategies incl. Smart Pick
await tab("settings").click();
await page.waitForSelector("text=Smart Pick (EV)", { timeout: 10000 });
console.log("✅ settings shows Smart Pick");
await page.waitForSelector("text=데이터 업데이트", { timeout: 10000 });
await page.waitForSelector("text=마지막 회차", { timeout: 10000 });
console.log("✅ settings shows data panel");
await page.screenshot({ path: `${SHOT_DIR}/e2e_settings.png` });

if (errors.length) { console.log("❌ page errors:", errors); process.exit(1); }
console.log("✅ E2E PASS — no page errors");
await browser.close();
