import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { chromium } from "@playwright/test";
import { exactStateValidator } from "./lib/hardware-validation.mjs";

//* Stable Chrome Hardware Checks ==============================================

const ROOT = path.resolve(import.meta.dirname, "..");
const BASE_URL = process.env.WEBVDB_BASE_URL ?? "http://127.0.0.1:5173";
const CHROME =
  process.env.CHROME_EXECUTABLE_PATH ??
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const INCLUDE_PERF = process.env.WEBVDB_PERF === "1";
const PERF_STEP_SIZE = process.env.WEBVDB_PERF_STEP_SIZE ?? "4";
const PERF_SHADOW_STEPS = process.env.WEBVDB_PERF_SHADOW_STEPS ?? "4";
const PERF_SAMPLE_BUDGET = process.env.WEBVDB_PERF_SAMPLE_BUDGET ?? "8192";
const EXPECTED_ADAPTER = process.env.WEBVDB_EXPECTED_ADAPTER ?? "apple metal-3";
const EMBERGEN_MANIFEST =
  process.env.WEBVDB_EMBERGEN_MANIFEST ?? "/fixtures/embergen/Fire_01/manifest.json";
const OUTPUT = path.join(ROOT, "artifacts", "hardware-report.json");

if (!existsSync(CHROME)) throw new Error(`Stable Chrome executable not found: ${CHROME}`);

const checks = [
  {
    name: "demo-02-direct-presentation",
    path:
      "/src/demos/02-cloud/index.html" +
      "?test=1&fixture=primitive&presentation=direct&swizzle=off&keepAlive=off",
    state: "__DEMO02__",
    timeout: 180_000,
    validate: exactStateValidator({
      fixture: "sphere_fog_fp8.nvdb",
      gridType: "Fp8",
      adapter: EXPECTED_ADAPTER,
      presentation: "direct",
      swizzleShim: false,
      keepAliveShim: false,
      resolution: "640x480",
      stepSize: 0.45,
      shadowSteps: 16,
      sampleBudgetCap: 8192,
    }),
  },
  {
    name: "demo-04-stock-3d-texture",
    path:
      "/src/demos/04-atlas-fallback/index.html" +
      "?test=1&texture=3d&presentation=direct&swizzle=off&keepAlive=off",
    state: "__DEMO04__",
    timeout: 180_000,
    validate: exactStateValidator({
      fixture: "/fixtures/primitives/sphere_fog_fp8.nvdb",
      statsFixture: "/fixtures/primitives/box_fog_float.nvdb",
      gridType: "Fp8",
      adapter: EXPECTED_ADAPTER,
      presentation: "direct",
      textureMode: "3d",
      swizzleShim: false,
      keepAliveShim: false,
      resolution: "640x480",
      atlasMaxDim: 96,
      atlasFilter: "trilinear",
      atlasFormat: "uint8",
      atlasNormalize: true,
      statsOk: true,
      transformOk: true,
    }),
  },
  {
    name: "demo-04-storage-bridge-no-shims",
    path:
      "/src/demos/04-atlas-fallback/index.html" +
      "?test=1&texture=storage&presentation=direct&swizzle=off&keepAlive=off",
    state: "__DEMO04__",
    timeout: 180_000,
    validate: exactStateValidator({
      fixture: "/fixtures/primitives/sphere_fog_fp8.nvdb",
      statsFixture: "/fixtures/primitives/box_fog_float.nvdb",
      gridType: "Fp8",
      adapter: EXPECTED_ADAPTER,
      presentation: "direct",
      textureMode: "storage",
      swizzleShim: false,
      keepAliveShim: false,
      resolution: "640x480",
      atlasMaxDim: 96,
      atlasFilter: "trilinear",
      atlasFormat: "uint8",
      atlasNormalize: true,
      statsOk: true,
      transformOk: true,
    }),
  },
  {
    name: "demo-05-embergen-direct",
    path:
      "/src/demos/05-embergen-sequence/index.html" +
      `?manifest=${encodeURIComponent(EMBERGEN_MANIFEST)}` +
      "&presentation=direct&swizzle=off&keepAlive=off&validate=cycle",
    state: "__DEMO05__",
    timeout: 600_000,
    validate: (state) => {
      const errors = exactStateValidator({
        adapter: EXPECTED_ADAPTER,
        validationMode: "full-cycle",
        cyclesCompleted: 1,
        finalFrame: 0,
        stalls: 0,
        presentation: "direct",
        swizzleShim: false,
        keepAliveShim: false,
      })(state);
      if (!(state?.frameCount > 1)) errors.push(`frameCount must be > 1, got ${state?.frameCount}`);
      if (state?.framesValidated !== state?.frameCount)
        errors.push(`framesValidated ${state?.framesValidated} != frameCount ${state?.frameCount}`);
      if (state?.framesPlayed !== state?.frameCount + 1)
        errors.push(`framesPlayed ${state?.framesPlayed} != frameCount + 1`);
      if (state?.rebinds !== state?.frameCount + 1)
        errors.push(`rebinds ${state?.rebinds} != frameCount + 1`);
      return errors;
    },
  },
];

if (INCLUDE_PERF)
  checks.push({
    name: "demo-02-wdas-1080p",
    path:
      "/src/demos/02-cloud/index.html" +
      `?perf=1&fixture=wdas&presentation=direct&stepSize=${PERF_STEP_SIZE}` +
      `&shadowSteps=${PERF_SHADOW_STEPS}&sampleBudgetCap=${PERF_SAMPLE_BUDGET}` +
      "&swizzle=off&keepAlive=off",
    state: "__DEMO02__",
    timeout: 600_000,
    validate: exactStateValidator({
      fixture: "wdas_cloud_quarter_fp8.nvdb",
      gridType: "Fp8",
      adapter: EXPECTED_ADAPTER,
      presentation: "direct",
      swizzleShim: false,
      keepAliveShim: false,
      resolution: "1920x1080",
      stepSize: Number(PERF_STEP_SIZE),
      shadowSteps: Number(PERF_SHADOW_STEPS),
      sampleBudgetCap: Number(PERF_SAMPLE_BUDGET),
    }),
  });

const browser = await chromium.launch({
  executablePath: CHROME,
  headless: false,
  args: ["--enable-unsafe-webgpu"],
});

try {
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 1,
  });
  const screenshotDir = path.join(ROOT, "artifacts", "hardware");
  mkdirSync(screenshotDir, { recursive: true });
  const results = [];

  for (const check of checks) {
    const page = await context.newPage();
    const consoleErrors = [];
    page.on("console", (message) => {
      if (message.type() === "error") consoleErrors.push(message.text());
    });
    page.on("pageerror", (error) => consoleErrors.push(error.message));

    const url = `${BASE_URL}${check.path}`;
    await page.goto(url);
    await page.waitForFunction(
      (stateName) => Reflect.get(window, stateName) !== undefined,
      check.state,
      { timeout: check.timeout },
    );
    const state = await page.evaluate((stateName) => Reflect.get(window, stateName), check.state);
    const perfPassed = check.name !== "demo-02-wdas-1080p" || state?.fps >= 60;
    const validationErrors = check.validate?.(state) ?? [];
    const validationPassed = validationErrors.length === 0;
    const passed =
      state?.ready === true &&
      !state?.error &&
      perfPassed &&
      validationPassed &&
      consoleErrors.length === 0;
    const screenshot = path.join(screenshotDir, `${check.name}.png`);
    await page.screenshot({ path: screenshot });
    results.push({
      name: check.name,
      url,
      passed,
      state,
      validationPassed,
      validationErrors,
      consoleErrors,
      screenshot: path.relative(ROOT, screenshot),
    });
    console.log(`${passed ? "✓" : "✗"} ${check.name}`, state);
    await page.close();
  }

  mkdirSync(path.dirname(OUTPUT), { recursive: true });
  writeFileSync(
    OUTPUT,
    `${JSON.stringify(
      {
        capturedAt: new Date().toISOString(),
        chromeExecutable: CHROME,
        chromeVersion: browser.version(),
        viewport: "1920x1080",
        results,
      },
      null,
      2,
    )}\n`,
  );

  if (results.some((result) => !result.passed)) process.exitCode = 1;
  console.log(`Hardware report: ${path.relative(ROOT, OUTPUT)}`);
} finally {
  await browser.close();
}
