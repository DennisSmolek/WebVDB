#!/usr/bin/env node
// K3 reference capture (R1 §6 Step 0): renders mjurczyk's renderer at the pinned commit to PNG, for the
// owner to run on a machine with a real GPU. Agents must NOT run this script (CLAUDE.md: his code is read,
// not run, in the cloud sandbox); it starts his Vite dev server and drives his examples page.
//
// ── How to run (owner's machine) ────────────────────────────────────────────────────────────────────
//
//   # 1. Clone his repo into a temporary directory OUTSIDE this repo, at the pinned commit:
//   tmp="$(mktemp -d)"
//   git clone https://github.com/mjurczyk/openvdb "$tmp/openvdb"
//   git -C "$tmp/openvdb" checkout --detach 75e73454f2d6f4384c9ffd57b704319767499360
//
//   # 2. Install his dependencies with lifecycle scripts disabled (no install/postinstall hooks run):
//   (cd "$tmp/openvdb" && npm ci --ignore-scripts)
//   #    If Vite then fails with an esbuild binary error, allow only esbuild's own install step:
//   #    (cd "$tmp/openvdb" && npm rebuild esbuild)
//
//   # 3. From this repo (needs `pnpm install` here for Playwright), capture:
//   node scripts/k3/capture-reference.mjs --openvdb "$tmp/openvdb"
//   #    --dry-run         check prerequisites and print the plan; start nothing
//   #    --scenarios a,b   only these scenario names (see SCENARIOS below)
//   #    --port 5406       his dev server port (default 5406; 127.0.0.1 only)
//   #    --executable <p>  Chrome/Chromium binary (default: Playwright's Chromium)
//   #    --headed          show the browser window
//   #    --force           overwrite an existing capture directory
//
//   # 4. Compare our renders with the captures (Step 1 bar: PSNR >= 40 dB, <= 0.5% pixels > 8/255):
//   node scripts/k3/compare.mjs --ours artifacts/k3/ours --reference fixtures/k3-reference/75e7345 --heatmap artifacts/k3/diff
//
//   # 5. Delete the clone when done: rm -rf "$tmp"
//
// Output (git-ignored): fixtures/k3-reference/75e7345/<scenario>.png (parity capture: jitter off, temporal
// off, light animation off, his default 1280x720 viewport at device scale 1), <scenario>.asis.jpg (his own
// validation settings, q90 JPEG, for the sanity check against his committed JPEG) and manifest.json (commit,
// machine, browser, adapter, camera and volume transforms, every GUI field value, SHA-256 of each file, and
// the sanity result).
//
// Parity notes: his data path is inherently `normalize: 'clamp01'` (8-bit textures, values clamped to [0, 1]),
// so our dense source is set to the same; the matched resolution is his `.wvdb` finest LOD (`resolution` is
// baked into the file) and the 1280x720 frame. The captures are display-referred (sRGB, his tone mapping,
// MSAA as his examples page sets it): his page has no linear pre-tonemap output without changing his code.

import { execFileSync, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const PINNED_COMMIT = "75e73454f2d6f4384c9ffd57b704319767499360";
const PINNED_SHORT = PINNED_COMMIT.slice(0, 7);
const PINNED_THREE = "0.187.0";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

/** Fields every parity capture adds to its scenario (GUI field ids of his frame demo). */
const PARITY_OVERRIDES = Object.freeze({ jitter: false, temporal: false, animateLights: false });

/**
 * Scenarios (R1 §6). `fields` are his GUI field ids, as in his tests/validation-set/*.json; `sanity` is his
 * committed JPEG for the same settings (his validation.spec.js threshold: <= 1% pixels with a channel
 * difference > 48), checked on the `asis` capture.
 */
const SCENARIOS = [
  // Step 0: the dense-source assets, his demo-1 sources in the default .wvdb format.
  { name: "step0-bunny", step: 0, fields: { demo: "frame" }, sanity: "demo-1.jpg" },
  { name: "step0-smoke", step: 0, fields: { demo: "frame", source: "smoke" }, sanity: null },
  // Step 1: froxels with 'volume' lighting; one directional light; multiple scattering off, then true.
  { name: "step1-lighting-volume", step: 1, fields: { demo: "frame", lighting: "volume" }, sanity: "demo-1-lighting-volume.jpg" },
  { name: "step1-multiple-scattering-off", step: 1, fields: { demo: "frame", multipleScattering: false }, sanity: "demo-1-multiple-scattering-off.jpg" },
  { name: "step1-lights-directional", step: 1, fields: { demo: "frame", lightSetup: "dir" }, sanity: "demo-1-lights-directional.jpg" },
  { name: "step1-volume-dir-ms-off", step: 1, fields: { demo: "frame", lighting: "volume", lightSetup: "dir", multipleScattering: false }, sanity: null },
  { name: "step1-volume-dir-ms-on", step: 1, fields: { demo: "frame", lighting: "volume", lightSetup: "dir", multipleScattering: true }, sanity: null },
  // K3.1: the same two, isolated for the Step 1 parity bar. His 'dir' light setup keeps scene.environment (the HDR
  // map) and every FrameVolume lights itself from it too (lights bitmask: all), which Step 1 does not port (K3.4).
  // `isolate` uses his public API only, after the GUI fields: every FrameVolume gets lights = 8
  // (frameLights.useDirectionalLights), every other mesh (grounded skybox, ground, props, light helpers) is hidden,
  // and the background is his init colour 0x598eff (scene.background null + renderer clear colour).
  { name: "step1-volume-dir-ms-off-isolated", step: 1, fields: { demo: "frame", lighting: "volume", lightSetup: "dir", multipleScattering: false }, sanity: null, isolate: true },
  { name: "step1-volume-dir-ms-on-isolated", step: 1, fields: { demo: "frame", lighting: "volume", lightSetup: "dir", multipleScattering: true }, sanity: null, isolate: true },
];

// His tests/playwright.config.js and tests/validation.spec.js, mirrored.
const VIEWPORT = { width: 1280, height: 720 };
const BROWSER_ARGS = ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-gpu"];
const FRAME_TIME = 1000 / 60;
const JPEG_QUALITY = 90;
const SANITY = { channelThreshold: 48, maxChangedPixelRatio: 0.01 };
const HIDE_OVERLAYS = `
  .lil-gui, #loading, #dropzone, .bottom-tab, body > div[style*="z-index: 10000"] {
    display: none !important;
  }
`;

// ── CLI ──────────────────────────────────────────────────────────────────────────────────────────────

function fail(message) {
  console.error(`\ncapture-reference: ${message}\n`);
  process.exit(1);
}

const argv = process.argv.slice(2);
const opt = { port: "5406" };
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === "--dry-run" || a === "--headed" || a === "--force") opt[a.slice(2)] = true;
  else if (a === "--help" || a === "-h") {
    console.log(readFileSync(fileURLToPath(import.meta.url), "utf8").split("\n").filter((l) => l.startsWith("//")).join("\n"));
    process.exit(0);
  } else if (a.startsWith("--")) {
    const v = argv[++i];
    if (v === undefined) fail(`${a} needs a value`);
    opt[a.slice(2)] = v;
  } else fail(`unexpected argument ${a} (see --help)`);
}

// ── Prerequisites (all checked before anything of his runs) ─────────────────────────────────────────

if (!opt.openvdb) fail("pass --openvdb <path to his clone> (see the header of this file for the clone + install steps)");
const his = path.resolve(opt.openvdb);
if (!existsSync(path.join(his, "package.json"))) fail(`${his} has no package.json; is it a clone of mjurczyk/openvdb?`);
const rel = path.relative(repoRoot, his);
if (!rel.startsWith("..") && !path.isAbsolute(rel)) fail(`${his} is inside this repo; clone his repo into a temporary directory outside it`);
const pkg = JSON.parse(readFileSync(path.join(his, "package.json"), "utf8"));
if (pkg.name !== "openvdb-workspace") fail(`${his}/package.json is "${pkg.name}", expected "openvdb-workspace"`);

let head;
try {
  head = execFileSync("git", ["-C", his, "rev-parse", "HEAD"], { encoding: "utf8" }).trim();
} catch {
  fail(`git rev-parse failed in ${his}; it must be a git clone`);
}
if (head !== PINNED_COMMIT) fail(`${his} is at ${head}; check out the pinned commit: git -C "${his}" checkout --detach ${PINNED_COMMIT}`);
const dirty = execFileSync("git", ["-C", his, "status", "--porcelain", "--untracked-files=no"], { encoding: "utf8" }).trim();
if (dirty) fail(`${his} has local changes to tracked files; captures must come from the pinned commit as published:\n${dirty}`);

const examplesDir = path.join(his, "src", "docs", "examples");
const viteCandidates = [path.join(his, "node_modules", "vite", "bin", "vite.js"), path.join(examplesDir, "node_modules", "vite", "bin", "vite.js")];
const viteBin = viteCandidates.find((p) => existsSync(p));
if (!viteBin) fail(`no Vite in ${his}/node_modules; install his dependencies first: (cd "${his}" && npm ci --ignore-scripts)`);
const threePkg = [path.join(his, "node_modules", "three", "package.json"), path.join(examplesDir, "node_modules", "three", "package.json")].find((p) => existsSync(p));
if (!threePkg) fail(`three is not installed in ${his}; run npm ci --ignore-scripts there`);
const threeVersion = JSON.parse(readFileSync(threePkg, "utf8")).version;
if (threeVersion !== PINNED_THREE) fail(`his install has three ${threeVersion}; his lockfile pins ${PINNED_THREE}. Use npm ci (not npm install)`);
const validationSet = path.join(his, "tests", "validation-set");
if (!existsSync(validationSet)) fail(`${validationSet} is missing`);

let chromium;
try {
  ({ chromium } = await import("@playwright/test"));
} catch {
  fail("Playwright is not installed in this repo; run pnpm install here");
}

const wanted = opt.scenarios ? new Set(String(opt.scenarios).split(",")) : null;
const scenarios = SCENARIOS.filter((s) => !wanted || wanted.has(s.name));
if (wanted) for (const n of wanted) if (!SCENARIOS.some((s) => s.name === n)) fail(`unknown scenario ${n}; known: ${SCENARIOS.map((s) => s.name).join(", ")}`);
for (const s of scenarios) if (s.sanity && !existsSync(path.join(validationSet, s.sanity))) fail(`his reference ${s.sanity} is missing from ${validationSet}`);

const outDir = path.join(repoRoot, "fixtures", "k3-reference", PINNED_SHORT);
if (existsSync(outDir) && readdirSync(outDir).length && !opt.force) fail(`${outDir} is not empty; pass --force to overwrite`);
const port = Number(opt.port);
if (!Number.isInteger(port) || port < 1024) fail(`bad --port ${opt.port}`);
const baseURL = `http://127.0.0.1:${port}/`;

console.log(`his clone      ${his} @ ${head} (three ${threeVersion})`);
console.log(`dev server     ${viteBin} --config ./dev.config.js (cwd ${examplesDir}) on ${baseURL}`);
console.log(`output         ${outDir}`);
console.log(`scenarios      ${scenarios.map((s) => s.name).join(", ")}`);
console.log(`parity fields  ${JSON.stringify(PARITY_OVERRIDES)}; viewport ${VIEWPORT.width}x${VIEWPORT.height} @1x`);
if (opt["dry-run"]) {
  console.log("\n--dry-run: prerequisites OK; nothing started.");
  process.exit(0);
}

// ── His dev server ───────────────────────────────────────────────────────────────────────────────────

try {
  await fetch(baseURL);
  fail(`something already listens on ${baseURL}; pick another --port`);
} catch {
  // free
}

const server = spawn(process.execPath, [viteBin, "--config", "./dev.config.js", "--host", "127.0.0.1", "--port", String(port), "--strictPort"], {
  cwd: examplesDir,
  stdio: ["ignore", "pipe", "pipe"],
  env: { ...process.env, BROWSER: "none" },
});
let serverLog = "";
server.stdout.on("data", (d) => (serverLog += d));
server.stderr.on("data", (d) => (serverLog += d));
const stopServer = () => {
  if (server.exitCode === null) server.kill("SIGTERM");
};
process.on("exit", stopServer);
process.on("SIGINT", () => {
  stopServer();
  process.exit(130);
});

async function waitForServer() {
  const started = Date.now();
  while (Date.now() - started < 90_000) {
    if (server.exitCode !== null) fail(`his dev server exited (${server.exitCode}):\n${serverLog}`);
    try {
      const r = await fetch(baseURL);
      if (r.ok) return;
    } catch {
      // not yet
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  fail(`his dev server did not answer on ${baseURL} within 90 s:\n${serverLog}`);
}

// ── Capture (his validation.spec.js procedure) ───────────────────────────────────────────────────────

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

async function setFields(page, fields) {
  await page.evaluate(async (f) => {
    const guiModule = await import("/src/utils/gui.js");
    for (const [id, value] of Object.entries(f)) {
      const controller = guiModule.gui.controllersRecursive().find((each) => each.property === id);
      if (!controller) throw new Error(`No GUI field "${id}"`);
      controller.setValue(value);
    }
  }, fields);
}

async function waitForIdle(page, pending, errors) {
  let lastState = null;
  let stableSince = Date.now();
  const started = Date.now();
  while (Date.now() - stableSince < 1000) {
    if (Date.now() - started > 170_000) throw new Error("the page did not go idle within 170 s");
    await page.evaluate(() => window.__flushLoadingFrames());
    await page.clock.runFor(0);
    await page.waitForTimeout(50);
    if (errors.length) throw new Error(`page errors: ${errors.join(" | ")}`);
    const state = await page.evaluate(() => {
      const loading = document.querySelector("#loading").style.display !== "none";
      const status = [...document.querySelectorAll(".lil-gui input[disabled]")].map((input) => input.value).join("|");
      return JSON.stringify({ loading, status });
    });
    const idle = !JSON.parse(state).loading && pending.size === 0;
    if (!idle || state !== lastState) stableSince = Date.now();
    lastState = state;
  }
}

/** Camera, renderer, volumes and GUI values, read from his page modules after the capture. */
async function readMetadata(page) {
  return page.evaluate(async () => {
    const out = {};
    try {
      const main = await import("/src/main.jsx");
      const { camera, renderer, scene } = main;
      out.camera = {
        type: camera.type,
        fov: camera.fov,
        aspect: camera.aspect,
        near: camera.near,
        far: camera.far,
        position: camera.position.toArray(),
        quaternion: camera.quaternion.toArray(),
        matrixWorld: camera.matrixWorld.toArray(),
        projectionMatrix: camera.projectionMatrix.toArray(),
      };
      out.renderer = {
        canvas: renderer.domElement ? [renderer.domElement.width, renderer.domElement.height] : null,
        pixelRatio: renderer.getPixelRatio?.(),
        toneMapping: renderer.toneMapping,
        toneMappingExposure: renderer.toneMappingExposure,
        outputColorSpace: renderer.outputColorSpace,
        isWebGPURenderer: !!renderer.isWebGPURenderer,
        backend: renderer.backend?.isWebGPUBackend ? "webgpu" : renderer.backend ? "webgl2-fallback" : "webgl",
      };
      out.background = scene.background?.isColor ? scene.background.toArray() : scene.background ? String(scene.background.type ?? "texture") : null;
      out.volumes = [];
      scene.updateMatrixWorld(true);
      scene.traverse((o) => {
        if (!o.isFrameVolume) return;
        out.volumes.push({
          matrixWorld: o.matrixWorld.toArray(),
          meanDensity: o.meanDensity,
          densityScale: o.getDensityScale?.(),
          extinction: o.extinction,
          normalizeDensity: o.normalizeDensity,
          mipmaps: o.mipmaps,
          mipmapFilter: o.mipmapFilter,
          textures: o.volumes.map((v) => ({
            matrix: v.matrix.toArray(),
            density: v.densityMap3D ? [v.densityMap3D.image.width, v.densityMap3D.image.height, v.densityMap3D.image.depth] : null,
          })),
        });
      });
    } catch (e) {
      out.metadataError = String(e?.message ?? e);
    }
    try {
      const guiModule = await import("/src/utils/gui.js");
      out.gui = {};
      for (const c of guiModule.gui.controllersRecursive()) {
        const v = c.getValue();
        if (v === null || ["number", "string", "boolean"].includes(typeof v)) out.gui[c.property] = v;
      }
    } catch (e) {
      out.guiError = String(e?.message ?? e);
    }
    try {
      const adapter = await navigator.gpu?.requestAdapter();
      out.adapter = adapter?.info ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device, description: adapter.info.description } : null;
    } catch (e) {
      out.adapterError = String(e?.message ?? e);
    }
    return out;
  });
}

/** K3.1 `isolate`: only the FrameVolumes and the directional light reach the frame (see SCENARIOS). His public API only. */
async function isolateVolumes(page) {
  await page.evaluate(async () => {
    const { scene, renderer } = await import("/src/main.jsx");
    const DIRECTIONAL_ONLY = 0b001000; // three-openvdb frame/core/frameLights.js useDirectionalLights
    const insideVolume = (o) => {
      for (let p = o; p; p = p.parent) if (p.isFrameVolume) return true;
      return false;
    };
    scene.traverse((o) => {
      if (o.isFrameVolume) o.lights = DIRECTIONAL_ONLY;
      else if ((o.isMesh || o.isPoints || o.isLine || o.isSprite) && !insideVolume(o)) o.visible = false;
    });
    scene.background = null;
    renderer.setClearColor(0x598eff, 1.0);
  });
}

async function capture(context, scenario, fields, type) {
  const page = await context.newPage();
  const pending = new Set();
  const errors = [];
  page.on("request", (r) => pending.add(r));
  page.on("requestfinished", (r) => pending.delete(r));
  page.on("requestfailed", (r) => pending.delete(r));
  page.on("pageerror", (e) => errors.push(e.message));
  await page.clock.install({ time: new Date("2026-01-01T00:00:00Z") });
  await page.clock.pauseAt(new Date("2026-01-01T00:00:01Z"));
  await page.addInitScript(() => {
    const requestFrame = window.requestAnimationFrame.bind(window);
    const cancelFrame = window.cancelAnimationFrame.bind(window);
    const loadingFrames = new Map();
    window.requestAnimationFrame = (callback) => {
      const id = requestFrame((time) => {
        loadingFrames.delete(id);
        callback(time);
      });
      if (/\/src\/utils\/resources\.js/.test(new Error().stack)) loadingFrames.set(id, callback);
      return id;
    };
    window.cancelAnimationFrame = (id) => {
      loadingFrames.delete(id);
      cancelFrame(id);
    };
    window.__flushLoadingFrames = () => {
      const frames = [...loadingFrames];
      loadingFrames.clear();
      frames.forEach(([id, callback]) => {
        cancelFrame(id);
        callback(performance.now());
      });
    };
  });
  await page.goto(baseURL);
  await page.waitForFunction(() => !!document.querySelector(".lil-gui"));
  const { demo, ...rest } = fields;
  await setFields(page, { demo, ...rest });
  await waitForIdle(page, pending, errors);
  if (scenario.isolate) await isolateVolumes(page);
  await page.clock.runFor(FRAME_TIME);
  await page.waitForTimeout(10);
  const bytes = await page.screenshot(type === "png" ? { type: "png", style: HIDE_OVERLAYS } : { type: "jpeg", quality: JPEG_QUALITY, style: HIDE_OVERLAYS });
  if (errors.length) throw new Error(`${scenario.name}: page errors: ${errors.join(" | ")}`);
  const metadata = await readMetadata(page);
  await page.close();
  return { bytes, metadata };
}

/** His countChangedPixels on two JPEGs, decoded by the browser. */
async function sanityCheck(context, actual, expected) {
  const page = await context.newPage();
  await page.goto(baseURL);
  const r = await page.evaluate(
    async ({ a, b, threshold }) => {
      const decode = async (b64) => {
        const blob = await (await fetch(`data:image/jpeg;base64,${b64}`)).blob();
        const bmp = await createImageBitmap(blob);
        const canvas = new OffscreenCanvas(bmp.width, bmp.height);
        const ctx = canvas.getContext("2d");
        ctx.drawImage(bmp, 0, 0);
        return ctx.getImageData(0, 0, bmp.width, bmp.height);
      };
      const x = await decode(a);
      const y = await decode(b);
      if (x.width !== y.width || x.height !== y.height) return { sizeMismatch: [x.width, x.height, y.width, y.height] };
      let changed = 0;
      for (let i = 0; i < x.data.length; i += 4) {
        const d = Math.max(Math.abs(x.data[i] - y.data[i]), Math.abs(x.data[i + 1] - y.data[i + 1]), Math.abs(x.data[i + 2] - y.data[i + 2]));
        if (d > threshold) changed++;
      }
      return { changedRatio: changed / (x.width * x.height) };
    },
    { a: Buffer.from(actual).toString("base64"), b: Buffer.from(expected).toString("base64"), threshold: SANITY.channelThreshold },
  );
  await page.close();
  if (r.sizeMismatch) return { pass: false, error: `size mismatch ${r.sizeMismatch.join(",")}` };
  return { changedRatio: r.changedRatio, threshold: SANITY.channelThreshold, maxRatio: SANITY.maxChangedPixelRatio, pass: r.changedRatio <= SANITY.maxChangedPixelRatio };
}

await waitForServer();
const browser = await chromium.launch({
  headless: !opt.headed,
  args: BROWSER_ARGS,
  ...(opt.executable ? { executablePath: opt.executable } : {}),
});
const context = await browser.newContext({ viewport: VIEWPORT, deviceScaleFactor: 1 });
mkdirSync(outDir, { recursive: true });

const manifest = {
  commit: PINNED_COMMIT,
  commitShort: PINNED_SHORT,
  capturedAt: new Date().toISOString(),
  tool: "scripts/k3/capture-reference.mjs",
  machine: { platform: process.platform, arch: process.arch, release: os.release(), cpu: os.cpus()[0]?.model ?? null, cpus: os.cpus().length },
  node: process.version,
  browser: { name: "chromium", version: browser.version(), args: BROWSER_ARGS, headless: !opt.headed },
  three: threeVersion,
  viewport: { ...VIEWPORT, deviceScaleFactor: 1 },
  parityOverrides: PARITY_OVERRIDES,
  notes: [
    "PNG captures are display-referred (sRGB output of his examples page, MSAA as his page sets it).",
    "Dense data on his side is 8-bit clamp(value/scale, 0, 1): compare against our dense source with normalize 'clamp01'.",
    "Timings are not captured (D25).",
  ],
  scenarios: [],
};

let failed = false;
try {
  for (const s of scenarios) {
    process.stdout.write(`${s.name} … `);
    const fields = { ...s.fields, ...PARITY_OVERRIDES };
    const parity = await capture(context, s, fields, "png");
    const file = `${s.name}.png`;
    writeFileSync(path.join(outDir, file), parity.bytes);
    const entry = { name: s.name, step: s.step, fields, isolate: !!s.isolate, file, sha256: sha256(parity.bytes), metadata: parity.metadata, asis: null, sanity: null };
    const description = parity.metadata.adapter?.description ?? "";
    if (/swiftshader|llvmpipe|software/i.test(`${description} ${parity.metadata.adapter?.vendor ?? ""}`)) {
      throw new Error(`the browser used a software adapter (${description}); references must come from a hardware GPU (try --executable with an installed Chrome)`);
    }
    if (s.sanity) {
      const asis = await capture(context, s, s.fields, "jpeg");
      const asisFile = `${s.name}.asis.jpg`;
      writeFileSync(path.join(outDir, asisFile), asis.bytes);
      entry.asis = { file: asisFile, sha256: sha256(asis.bytes), fields: s.fields };
      entry.sanity = { reference: s.sanity, ...(await sanityCheck(context, asis.bytes, readFileSync(path.join(validationSet, s.sanity)))) };
      if (!entry.sanity.pass) failed = true;
    }
    manifest.scenarios.push(entry);
    console.log(entry.sanity ? `ok (sanity vs ${s.sanity}: ${(entry.sanity.changedRatio * 100).toFixed(3)}% > 48, ${entry.sanity.pass ? "pass" : "FAIL"})` : "ok");
  }
} catch (e) {
  failed = true;
  console.error(`\ncapture failed: ${e?.stack ?? e}`);
} finally {
  writeFileSync(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  await browser.close();
  stopServer();
}
console.log(`\nwrote ${manifest.scenarios.length} scenario(s) and manifest.json to ${outDir}`);
if (failed) {
  console.error("one or more scenarios failed or missed his own JPEG sanity bar; see manifest.json before using these as references");
  process.exit(1);
}
