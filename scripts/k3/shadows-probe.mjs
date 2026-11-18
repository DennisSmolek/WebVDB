#!/usr/bin/env node
// K3.6 dev helper: loads examples/src/harness/k3-frame-shadows.html with a query against a running examples dev server and
// prints the result JSON (without the image fields). Usage:
//   pnpm --filter @webvdb/examples dev --host 127.0.0.1 --port 5467 &
//   node scripts/k3/shadows-probe.mjs 5467 "?case=emission&twins=1" [--executable <chrome>] [--hardware]
// Default browser: Playwright's Chromium with SwiftShader (the e2e flags); --hardware drops the SwiftShader flags.
import { chromium } from "@playwright/test";

const [port = "5467", query = "", ...rest] = process.argv.slice(2);
const exe = rest.includes("--executable") ? rest[rest.indexOf("--executable") + 1] : process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
const hardware = rest.includes("--hardware");
const args = hardware ? ["--enable-unsafe-webgpu", "--enable-gpu", "--ignore-gpu-blocklist", "--enable-webgpu-developer-features"] : ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--enable-unsafe-swiftshader"];
const browser = await chromium.launch({ headless: !hardware, args, ...(exe ? { executablePath: exe } : {}) });
const page = await browser.newPage();
page.on("console", (m) => (m.type() === "error" ? console.error(`[page] ${m.text()}`) : undefined));
await page.goto(`http://127.0.0.1:${port}/src/harness/k3-frame-shadows.html${query}`);
await page.waitForFunction(() => window.__VDB_K3_SHADOWS__?.done === true, undefined, { timeout: 600_000 });
const r = await page.evaluate(() => window.__VDB_K3_SHADOWS__);
await browser.close();
for (const k of ["rgba", "rgbaOff", "debugRgba"]) delete r[k];
console.log(JSON.stringify(r, null, 1));
