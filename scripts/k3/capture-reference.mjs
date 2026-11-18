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
//   #    --out <dir>       output directory (default fixtures/k3-reference/75e7345)
//   #    --merge           add to an existing capture directory: the captured scenarios replace theirs in
//   #                      manifest.json, every other scenario (and file) stays (no --force needed)
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
  // K3.3 (R1 §6 Step 3): 'frame' lighting, as is (his default 'environment' light setup: three-environment-shadows
  // scene lights found in the UltraHDR map, the map without them as scene.environment, the grounded skybox; our
  // harness renders none of that, so these are sanity + INFO only). MS true first (R1: 'neural' is Step 4).
  { name: "step3-default", step: 3, fields: { demo: "frame", multipleScattering: true }, sanity: null },
  { name: "step3-shadow-cascades-4", step: 3, fields: { demo: "frame", multipleScattering: true, shadowCascades: 4 }, sanity: null },
  { name: "step3-shadow-grid-high", step: 3, fields: { demo: "frame", multipleScattering: true, shadowResolution: 160 }, sanity: null },
  { name: "step3-two-bunnies", step: 3, fields: { demo: "frame", multipleScattering: true, source: "twoBunnies" }, sanity: null },
  { name: "step3-environment-sky-midday", step: 3, fields: { demo: "frame", multipleScattering: true, environment: "sky-midday" }, sanity: null },
  // His own JPEGs for the same GUI fields (default MS 'neural'): the sanity check of the capture rig only.
  { name: "step3-shadow-cascades-4-asis", step: 3, fields: { demo: "frame", shadowCascades: 4 }, sanity: "demo-1-shadow-cascades-4.jpg" },
  { name: "step3-shadow-grid-high-asis", step: 3, fields: { demo: "frame", shadowResolution: 160 }, sanity: "demo-1-shadow-grid-high.jpg" },
  { name: "step3-environment-sky-midday-asis", step: 3, fields: { demo: "frame", environment: "sky-midday" }, sanity: "demo-1-environment-sky-midday.jpg" },
  // K3.3 parity scenes. `isolate: 'frame'`: his sun only ('dir' setup), scene.environment null, every other mesh hidden,
  // background 0x598eff; FrameVolume `lights` untouched ('frame' lighting does not read it). `isolate: 'environment'`: no
  // scene light ('none' setup), scene.environment kept (his UltraHDR sky-midday map, intensity 0.4: the harness loads
  // fixtures/k3/sky-midday-HDR.jpg), the skybox and other meshes hidden. The two-bunnies second bunny turns on a timer:
  // the manifest's volume matrices place it.
  { name: "step3-default-dir-isolated", step: 3, fields: { demo: "frame", lightSetup: "dir", multipleScattering: true }, sanity: null, isolate: "frame" },
  { name: "step3-shadow-cascades-4-dir-isolated", step: 3, fields: { demo: "frame", lightSetup: "dir", multipleScattering: true, shadowCascades: 4 }, sanity: null, isolate: "frame" },
  { name: "step3-shadow-grid-high-dir-isolated", step: 3, fields: { demo: "frame", lightSetup: "dir", multipleScattering: true, shadowResolution: 160 }, sanity: null, isolate: "frame" },
  { name: "step3-two-bunnies-dir-isolated", step: 3, fields: { demo: "frame", lightSetup: "dir", multipleScattering: true, source: "twoBunnies" }, sanity: null, isolate: "frame" },
  { name: "step3-environment-sky-midday-env-isolated", step: 3, fields: { demo: "frame", lightSetup: "none", multipleScattering: true, environment: "sky-midday" }, sanity: null, isolate: "environment" },
  // K3.4 (R1 §6 Step 4): phase functions and multiple scattering modes, his tests/validation-set demo-1-frame-phase-*,
  // -multiple-scattering-*, -octave* and -lighting-volume-phase-g-low fields (his default multipleScattering 'neural').
  // `*-asis`: his GUI fields as is ('environment' light setup), checked against his JPEG where he has one (INFO for
  // our parity). `*-dir-isolated`: + his sun only, `isolate: 'frame'` ('volume' lighting: `isolate: true`), the gate.
  ...[
    ["step4-default-neural", {}, "demo-1.jpg"],
    ["step4-multiple-scattering-two-sided", { multipleScattering: "two-sided" }, null],
    ["step4-multiple-scattering-octaves", { multipleScattering: "octaves" }, "demo-1-multiple-scattering-octaves.jpg"],
    ["step4-octaves-1", { multipleScattering: "octaves", scatteringOctaves: 1 }, "demo-1-octaves-1.jpg"],
    ["step4-octaves-8", { multipleScattering: "octaves", scatteringOctaves: 8 }, "demo-1-octaves-8.jpg"],
    ["step4-octave-extinction-high", { multipleScattering: "octaves", scatteringOctaveExtinction: 1 }, "demo-1-octave-extinction-high.jpg"],
    ["step4-octave-contribution-high", { multipleScattering: "octaves", scatteringOctaveContribution: 1 }, "demo-1-octave-contribution-high.jpg"],
    ["step4-octave-phase-low", { multipleScattering: "octaves", scatteringOctavePhase: 0 }, "demo-1-octave-phase-low.jpg"],
    ["step4-frame-phase-isotropic", { "frame-phase": "isotropic" }, "demo-1-frame-phase-isotropic.jpg"],
    ["step4-frame-phase-g-low", { "frame-phaseG": -0.5 }, "demo-1-frame-phase-g-low.jpg"],
    ["step4-frame-phase-dual-lobe", { "frame-phase": "dual-lobe" }, "demo-1-frame-phase-dual-lobe.jpg"],
    ["step4-frame-phase-dual-lobe-back-g-high", { "frame-phase": "dual-lobe", "frame-phaseBackG": 0.5 }, "demo-1-frame-phase-dual-lobe-back-g-high.jpg"],
    ["step4-frame-phase-dual-lobe-back-weight-high", { "frame-phase": "dual-lobe", "frame-phaseBackWeight": 1 }, "demo-1-frame-phase-dual-lobe-back-weight-high.jpg"],
    ["step4-frame-phase-lorenz-mie", { "frame-phase": "lorenz-mie" }, "demo-1-frame-phase-lorenz-mie.jpg"],
    ["step4-frame-phase-lorenz-mie-rgb", { "frame-phase": "lorenz-mie-rgb" }, null],
    ["step4-frame-phase-g-exponent", { "frame-phaseGExponent": 1 }, null],
    ["step4-lighting-volume-phase-g-low", { lighting: "volume", phaseG: -0.5 }, "demo-1-lighting-volume-phase-g-low.jpg"],
  ].flatMap(([name, extra, sanity]) => [
    { name: `${name}-asis`, step: 4, fields: { demo: "frame", ...extra }, sanity },
    { name: `${name}-dir-isolated`, step: 4, fields: { demo: "frame", lightSetup: "dir", ...extra }, sanity: null, isolate: extra.lighting === "volume" ? true : "frame" },
  ]),
  // K3.5 (R1 §6 Step 5): spectral media and presets, his Spectral folder fields (utils/medium.js): `spectral`, `medium`
  // (its onChange turns spectral on and sets fogColor / baseColor (exact linear albedo), angstromExponent, phaseG,
  // phaseGExponent and the frame-phase fields) and `angstromExponent`. His default multipleScattering 'neural' (spectral
  // runs it as 1D diffusion, his isNeural). He has no validation JPEGs for them (sanity null). As Step 4: `*-asis` (INFO)
  // and `*-dir-isolated` (his sun only, the gate; 'volume' lighting `isolate: true`).
  ...[
    ["step5-spectral", { spectral: true }],
    ...["cloud", "fog", "ice", "soot", "white-smoke", "dark-smoke", "haze", "dust"].map((m) => [`step5-medium-${m}`, { medium: m }]),
    ["step5-angstrom-air", { spectral: true, angstromExponent: 4.09 }],
    ["step5-lighting-volume-medium-dark-smoke", { lighting: "volume", medium: "dark-smoke" }],
  ].flatMap(([name, extra]) => [
    { name: `${name}-asis`, step: 5, fields: { demo: "frame", ...extra }, sanity: null },
    { name: `${name}-dir-isolated`, step: 5, fields: { demo: "frame", lightSetup: "dir", ...extra }, sanity: null, isolate: extra.lighting === "volume" ? true : "frame" },
  ]),
  // K3.5: 'volume' lighting with his sky-midday environment and no scene light (`isolate: 'environment'`: the map kept,
  // FrameVolume `lights` untouched): the 'volume'-lighting environment terms (his inject pass's environment lights).
  { name: "step5-lighting-volume-sky-midday-env-isolated", step: 5, fields: { demo: "frame", lighting: "volume", lightSetup: "none", environment: "sky-midday" }, sanity: null, isolate: "environment" },
  { name: "step5-lighting-volume-medium-cloud-sky-midday-env-isolated", step: 5, fields: { demo: "frame", lighting: "volume", lightSetup: "none", environment: "sky-midday", medium: "cloud" }, sanity: null, isolate: "environment" },
  // K3.7 (R1 §6 Step 7): level sets (demo 5, `levelset`) and points (demo 7, `points`). His frame-demo parity fields
  // (jitter, temporal, animateLights) do not exist on these demos: `parityOverrides: false`. `*-asis`: his GUI fields
  // as in his tests/validation-set JSON, checked against his JPEG (INFO for our parity: his 'environment' lights).
  // `isolate: 'normals'`: his public three API only, after the build: every light hidden, scene.environment and
  // background null, clear colour black, every mesh but the level set hidden, and the level set drawn unlit as
  // view-space normal * 0.5 + 0.5 (LevelSetSurface: its MeshStandardNodeMaterial gets emissiveNode = normalView * 0.5
  // + 0.5, colour black, toneMapped false; LevelSetMesh: a MeshBasicNodeMaterial with that colourNode). `format: 'vdb'`
  // reads the same .vdb bytes as ours (utahteapot.vdb SHA-1 4ecf3a74; venusstatue.vdb from his assets).
  // `isolate: 'points'`: VDBPoints white (toneMapped false) on black, every other object and light hidden.
  { name: "step7-utah-teapot-surface-asis", step: 7, fields: { demo: "levelset", model: "utahteapot", render: "surface" }, sanity: "demo-5-utah-teapot-render-surface.jpg", parityOverrides: false },
  { name: "step7-venus-statue-surface-asis", step: 7, fields: { demo: "levelset", render: "surface" }, sanity: "demo-5-render-surface.jpg", parityOverrides: false },
  { name: "step7-utah-teapot-mesh-asis", step: 7, fields: { demo: "levelset", model: "utahteapot" }, sanity: "demo-5-model-utah-teapot.jpg", parityOverrides: false },
  { name: "step7-utah-teapot-surface-normals", step: 7, fields: { demo: "levelset", model: "utahteapot", format: "vdb", render: "surface" }, sanity: null, isolate: "normals", parityOverrides: false },
  { name: "step7-venus-statue-surface-normals", step: 7, fields: { demo: "levelset", format: "vdb", render: "surface" }, sanity: null, isolate: "normals", parityOverrides: false },
  { name: "step7-sphere-points-asis", step: 7, fields: { demo: "points", model: "sphere_points", format: "vdb" }, sanity: "demo-7.jpg", parityOverrides: false },
  { name: "step7-sphere-points-isolated", step: 7, fields: { demo: "points", model: "sphere_points", format: "vdb" }, sanity: null, isolate: "points", parityOverrides: false },
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
  if (a === "--dry-run" || a === "--headed" || a === "--force" || a === "--merge") opt[a.slice(2)] = true;
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

const outDir = opt.out ? path.resolve(String(opt.out)) : path.join(repoRoot, "fixtures", "k3-reference", PINNED_SHORT);
if (existsSync(outDir) && readdirSync(outDir).length && !opt.force && !opt.merge) fail(`${outDir} is not empty; pass --merge to add scenarios or --force to overwrite`);
// --merge: the existing manifest's other scenarios are kept (K3.4: Step 4 captures added to the Step 0-3 set)
const previousManifest = opt.merge && existsSync(path.join(outDir, "manifest.json")) ? JSON.parse(readFileSync(path.join(outDir, "manifest.json"), "utf8")) : null;
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
      // K3.3: what lights 'frame' lighting besides the scene lights
      out.environment = scene.environment
        ? { name: scene.environment.name ?? null, intensity: scene.environmentIntensity, rotation: scene.environmentRotation.toArray(), mapping: scene.environment.mapping }
        : null;
      out.lights = [];
      scene.traverseVisible((o) => {
        if (o.isLight) out.lights.push({ type: o.type, color: o.color?.toArray(), intensity: o.intensity, matrixWorld: o.matrixWorld.toArray() });
      });
      out.volumes = [];
      scene.updateMatrixWorld(true);
      // K3.7: his level sets (LevelSetSurface / LevelSetMesh: local space = the grid's world) and VDBPoints.
      out.levelSets = [];
      out.points = [];
      scene.traverse((o) => {
        if (o.isLevelSetSurface || o.isLevelSetMesh) {
          const t = o.distanceTexture?.image;
          out.levelSets.push({
            kind: o.isLevelSetSurface ? "surface" : "mesh",
            matrixWorld: o.matrixWorld.toArray(),
            bbox: o.bbox ? [o.bbox.min.toArray(), o.bbox.max.toArray()] : null,
            resolution: o.resolution ?? null,
            distanceTexture: t ? [t.width, t.height, t.depth] : null,
            background: o.material?.background?.value ?? null,
            vertices: o.geometry?.attributes?.position?.count ?? null,
          });
        }
        if (o.isVDBPoints) {
          out.points.push({ matrixWorld: o.matrixWorld.toArray(), pointCount: o.pointCount, count: o.count, spacing: o.spacing, size: o.size, pixelScale: o.pixelScale?.value ?? null });
        }
      });
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

/**
 * `isolate` (see SCENARIOS), his public API only. `true` (K3.1, 'volume' lighting): FrameVolumes light only from
 * directional lights. `'frame'` (K3.3): no scene.environment. `'environment'` (K3.3): scene.environment kept. In
 * every mode the meshes outside the volumes (skybox, ground, props, helpers) are hidden and the background is 0x598eff.
 */
async function isolateVolumes(page, mode) {
  await page.evaluate(async (m) => {
    const { scene, renderer } = await import("/src/main.jsx");
    const DIRECTIONAL_ONLY = 0b001000; // three-openvdb frame/core/frameLights.js useDirectionalLights
    const insideVolume = (o) => {
      for (let p = o; p; p = p.parent) if (p.isFrameVolume) return true;
      return false;
    };
    scene.traverse((o) => {
      if (o.isFrameVolume) {
        if (m === true) o.lights = DIRECTIONAL_ONLY;
      } else if ((o.isMesh || o.isPoints || o.isLine || o.isSprite) && !insideVolume(o)) o.visible = false;
    });
    if (m === "frame") scene.environment = null;
    scene.background = null;
    renderer.setClearColor(0x598eff, 1.0);
  }, mode);
}

/**
 * K3.7 `isolate: 'normals' | 'points'` (see SCENARIOS). three's own API on his scene objects. TSL comes from his dev
 * server's pre-bundled deps: the URL of three/webgpu that his main.jsx imports, with three_tsl.js beside it.
 */
async function isolateStep7(page, mode) {
  await page.evaluate(async (m) => {
    const { scene, renderer } = await import("/src/main.jsx");
    const source = await (await fetch("/src/main.jsx")).text();
    const webgpuUrl = source.match(/\/node_modules\/\.vite\/deps\/three_webgpu\.js(\?v=[0-9a-z]+)?/)?.[0];
    if (!webgpuUrl) throw new Error("isolateStep7: three/webgpu's pre-bundled URL is not in his main.jsx");
    const webgpu = await import(webgpuUrl);
    const tsl = await import(webgpuUrl.replace("three_webgpu.js", "three_tsl.js"));
    const normalColor = () => tsl.normalView.mul(0.5).add(0.5);
    let kept = 0;
    scene.traverse((o) => {
      if (o.isLight) o.visible = false;
      else if (m === "normals" && o.isLevelSetSurface) {
        o.material.color.setRGB(0, 0, 0);
        o.material.emissiveNode = normalColor();
        o.material.toneMapped = false;
        o.material.needsUpdate = true;
        kept++;
      } else if (m === "normals" && o.isLevelSetMesh) {
        o.material = new webgpu.MeshBasicNodeMaterial({ toneMapped: false });
        o.material.colorNode = normalColor();
        kept++;
      } else if (m === "points" && o.isVDBPoints) {
        o.material.color.setRGB(1, 1, 1);
        o.material.toneMapped = false;
        o.material.needsUpdate = true;
        kept++;
      } else if (o.isMesh || o.isPoints || o.isLine || o.isSprite) o.visible = false;
    });
    if (kept !== 1) throw new Error(`isolateStep7(${m}): expected one level set / points object, found ${kept}`);
    scene.environment = null;
    scene.background = null;
    renderer.setClearColor(0x000000, 1.0);
  }, mode);
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
  if (scenario.isolate === "normals" || scenario.isolate === "points") {
    await isolateStep7(page, scenario.isolate);
    await waitForIdle(page, pending, errors);
    // The replaced materials build their pipelines on the next renders.
    for (let i = 0; i < 3; i++) {
      await page.clock.runFor(FRAME_TIME);
      await page.waitForTimeout(150);
    }
  } else if (scenario.isolate) {
    await isolateVolumes(page, scenario.isolate);
    // 'frame' lighting reads the environment lights from an async readback; let a changed environment settle.
    if (scenario.isolate !== true) {
      await waitForIdle(page, pending, errors);
      // His environment lights are made on the update() after the async map readback lands, and the clock is paused
      // (waitForIdle never renders a frame), so without frames here the captured frame still takes the 'outward'
      // path. Run frames until the readback is consumed (real time passes for the GPU between them).
      for (let i = 0; i < 6; i++) {
        await page.clock.runFor(FRAME_TIME);
        await page.waitForTimeout(150);
      }
    }
  }
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
    const fields = s.parityOverrides === false ? { ...s.fields } : { ...s.fields, ...PARITY_OVERRIDES };
    const parity = await capture(context, s, fields, "png");
    const file = `${s.name}.png`;
    writeFileSync(path.join(outDir, file), parity.bytes);
    const entry = { name: s.name, step: s.step, fields, isolate: s.isolate ?? false, file, sha256: sha256(parity.bytes), metadata: parity.metadata, asis: null, sanity: null };
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
  if (previousManifest) {
    const captured = new Set(manifest.scenarios.map((e) => e.name));
    manifest.scenarios = [...(previousManifest.scenarios ?? []).filter((e) => !captured.has(e.name)), ...manifest.scenarios];
    manifest.merged = [...(previousManifest.merged ?? [{ capturedAt: previousManifest.capturedAt, browser: previousManifest.browser }]), { capturedAt: manifest.capturedAt, browser: manifest.browser, scenarios: [...captured] }];
  }
  writeFileSync(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  await browser.close();
  stopServer();
}
console.log(`\nwrote ${manifest.scenarios.length} scenario(s) and manifest.json to ${outDir}`);
if (failed) {
  console.error("one or more scenarios failed or missed his own JPEG sanity bar; see manifest.json before using these as references");
  process.exit(1);
}
