// K3.2c: hardware timings of the K3 froxel renderer on installed Google Chrome (the real GPU, not
// SwiftShader). Runs examples/src/harness/k3-timing.html once per configuration (one page load each, so
// shader builds never land in a timed frame), k3-pyramid.html for the GPU vs CPU pyramid build, and
// k3-frame.html's own timing for the K3.3 scene (step3-default-dir-isolated) as a cross-check. Writes
// artifacts/k3/hardware/timings-<stamp>.json and .md (+ timings-latest.*).
//
//   node scripts/k3/time-hardware.mjs                 # starts its own Vite server
//   node scripts/k3/time-hardware.mjs --only=hdda     # one group: hdda | gap | pyramid | k3frame | bench, sampling (K3.2d, opt-in)
//   options: --frames=30 --warmup=10 --port=5443 --base=http://127.0.0.1:5173 (reuse a server)
//            --quiet (you assert nothing else is using the GPU: drops the 'provisional' label)
//            --note="..." (recorded with the results) --allow-swiftshader --small (debug the script itself
//            on SwiftShader at 320x180; never a timing)
//
// Per-pass GPU ms come from three's render-pass timestamps (`timestamp-query`; Chrome is started with
// --enable-webgpu-developer-features so they are not quantized to 100 µs); without the feature only the
// queue-synchronised frame time (render, then queue.onSubmittedWorkDone(): v1's hardware gate,
// docs/archive/handoffs/PHASE-3.md) is reported. Medians of --frames after --warmup. D25: results are provisional
// unless --quiet; the machine's load average and the busiest processes are recorded either way.

import { spawn, execSync } from "node:child_process";
import { mkdirSync, writeFileSync, existsSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const OUT = path.join(ROOT, "artifacts", "k3", "hardware");
const arg = (name, fallback) => {
  const hit = process.argv.find((a) => a === `--${name}` || a.startsWith(`--${name}=`));
  if (!hit) return fallback;
  return hit.includes("=") ? hit.slice(hit.indexOf("=") + 1) : true;
};
const FRAMES = Number(arg("frames", 30));
const WARMUP = Number(arg("warmup", 10));
const PORT = Number(arg("port", 5443));
const ONLY = arg("only", "");
const QUIET = arg("quiet", false) === true;
const NOTE = String(arg("note", process.env.K3_TIMING_NOTE ?? ""));
const ALLOW_SWIFTSHADER = arg("allow-swiftshader", false) === true;
let BASE = arg("base", "");

const fixture = (rel) => existsSync(path.join(ROOT, "fixtures", rel));
const group = (g) => !ONLY || String(ONLY).split(",").includes(g);

// --- the configurations --------------------------------------------------------------------------------
// --small: a SwiftShader-sized smoke test of the script itself (not a timing).
const SMALL = arg("small", false) === true;
const HD = SMALL ? "w=320&h=180&froxels=64x64x16&substeps=16&shadowSteps=16" : "w=1920&h=1080&froxels=512x512x24&substeps=32&shadowSteps=32";
const assets = ["cloud04", fixture("k3/bunny_cloud.wvdb") && "bunny", fixture("vdb-samples/smoke.vdb") && "smoke", fixture("wdas/wdas_cloud_quarter_fp8.nvdb") && "wdas"].filter(Boolean);
const runs = [];
if (group("hdda")) {
  // (1) HDDA on vs off: the 'volume' lighting shadow march (inject pass), every sparse LOD strategy.
  for (const asset of assets) for (const lod of ["none", "pyramid", "node"]) for (const hdda of [1, 0]) {
    runs.push({ group: "hdda", key: `${asset}/volume/sparse-${lod}/hdda${hdda}`, page: "k3-timing", query: `asset=${asset}&source=sparse&lod=${lod}&hdda=${hdda}&lighting=volume&${HD}` });
  }
}
if (group("gap")) {
  // (2) Sparse ('coarse-pyramid', HDDA on) vs dense, 'frame' and 'volume' lighting, 1080p, his froxel defaults.
  for (const asset of assets) for (const lighting of ["frame", "volume"]) for (const source of ["dense", "sparse"]) {
    runs.push({ group: "gap", key: `${asset}/${lighting}/${source}`, page: "k3-timing", query: `asset=${asset}&source=${source}&lod=pyramid&hdda=1&lighting=${lighting}&${HD}` });
  }
}
if (group("k3frame") && fixture("k3/bunny_cloud.wvdb")) {
  // The K3.3 timing scene (his bunny, step3-default-dir-isolated, 1080p), k3-frame's own timing: the SwiftShader
  // gap (~5.6×: 292.6 vs 51.6 ms froxels) was measured here.
  for (const source of ["dense", "sparse"]) {
    runs.push({ group: "k3frame", key: `step3-default-dir-isolated/${source}`, page: "k3-frame", state: "__VDB_K3_FRAME__", query: `scene=step3-default-dir-isolated&source=${source}&timing=1&${SMALL ? "w=320&h=180&froxels=64x64x16" : "w=1920&h=1080"}&timingFrames=${FRAMES}&timingWarmup=${WARMUP}` });
  }
}
if (group("sampling") && ONLY) {
  // K3.2d: the sparse level-0 sampling candidates (fastTrilinear, cacheAccessor) in the real passes, HDDA off, vs dense.
  // Opt-in (--only=sampling); --sampling-assets= limits the assets, --sampling-variants= the variants (base,fast,cache,fastcache).
  const sa = String(arg("sampling-assets", "")).split(",").filter(Boolean);
  const sv = String(arg("sampling-variants", "base,fast,cache,fastcache")).split(",").filter(Boolean);
  const Q = { base: "fast=0&cache=0", fast: "fast=1&cache=0", cache: "fast=0&cache=1", fastcache: "fast=1&cache=1" };
  for (const asset of sa.length ? sa : assets) for (const lighting of ["frame", "volume"]) {
    runs.push({ group: "sampling", key: `${asset}/${lighting}/dense`, page: "k3-timing", query: `asset=${asset}&source=dense&lod=pyramid&hdda=0&lighting=${lighting}&${HD}` });
    for (const v of sv) runs.push({ group: "sampling", key: `${asset}/${lighting}/${v}`, page: "k3-timing", query: `asset=${asset}&source=sparse&lod=pyramid&hdda=0&lighting=${lighting}&${Q[v]}&${HD}` });
  }
}
if (group("custom") && ONLY) {
  // Ad-hoc k3-timing configurations: --custom="key=query;key=query" (query without the HD part, which is appended).
  for (const entry of String(arg("custom", "")).split(";").filter(Boolean)) {
    const at = entry.indexOf("=");
    runs.push({ group: "custom", key: entry.slice(0, at), page: "k3-timing", query: `${entry.slice(at + 1)}&${HD}` });
  }
}
if (group("bench") && ONLY) {
  // K3.2d: the sampling microbenchmark (compute kernels: dense texture fetch vs sparse vdbSample variants). Opt-in (--only=bench).
  runs.push({ group: "bench", key: "sample-bench", page: "k3-sample-bench", state: "__VDB_K3_SAMPLE_BENCH__", query: `assets=${["cloud04", ...assets.filter((a) => a !== "cloud04")].join(",")}&repeat=${Math.max(5, Math.min(FRAMES, 15))}${SMALL ? "&threads=4096&steps=8" : ""}` });
}
if (group("pyramid")) {
  // (3) GPU vs CPU coarse-pyramid build.
  const list = ["cloud04", ...assets.filter((a) => a !== "cloud04"), fixture("primitives/sphere_fog_fp8.nvdb") && "sphere8"].filter(Boolean);
  runs.push({ group: "pyramid", key: "pyramid", page: "k3-pyramid", state: "__VDB_K3_PYRAMID__", query: `assets=${list.join(",")}&repeat=${Math.max(5, Math.min(FRAMES, 15))}` });
}

// --- machine state (D25) -------------------------------------------------------------------------------
function machineState() {
  let busiest = [];
  try {
    busiest = execSync("ps -A -o %cpu=,comm= | sort -rn | head -8", { encoding: "utf8" })
      .trim()
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean)
      .map((l) => {
        const [cpu, ...rest] = l.split(/\s+/);
        return { cpu: Number(cpu), command: path.basename(rest.join(" ")) };
      });
  } catch {
    // ps unavailable: leave it empty
  }
  let otherBrowsers = 0;
  try {
    otherBrowsers = Number(execSync("pgrep -f 'chrome-headless-shell|Chrome for Testing' | wc -l", { encoding: "utf8" }).trim()) || 0;
  } catch {
    otherBrowsers = 0;
  }
  return { at: new Date().toISOString(), loadavg: os.loadavg().map((v) => Math.round(v * 100) / 100), cpus: os.cpus().length, busiest, otherHeadlessBrowserProcesses: otherBrowsers };
}

// --- server --------------------------------------------------------------------------------------------
async function startServer() {
  if (BASE) return null;
  BASE = `http://127.0.0.1:${PORT}`;
  const child = spawn("pnpm", ["--filter", "@webvdb/examples", "dev", "--host", "127.0.0.1", "--port", String(PORT), "--strictPort"], { cwd: ROOT, stdio: ["ignore", "pipe", "pipe"] });
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(BASE);
      if (r.ok) return child;
    } catch {
      // not up yet
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  child.kill();
  throw new Error(`Vite did not start on ${BASE}`);
}

async function runPage(browser, run) {
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  const consoleErrors = [];
  page.on("console", (m) => {
    // k3-frame asks for the optional reference manifest first; its 404 is expected.
    if (m.type() === "error" && !(run.page === "k3-frame" && /404/.test(m.text()))) consoleErrors.push(m.text());
  });
  page.on("pageerror", (e) => consoleErrors.push(e.message));
  const state = run.state ?? "__VDB_K3_TIMING__";
  const extra = run.page === "k3-timing" ? `&frames=${FRAMES}&warmup=${WARMUP}` : "";
  const t0 = Date.now();
  await page.goto(`${BASE}/src/harness/${run.page}.html?${run.query}${extra}`);
  await page.waitForFunction((s) => window[s]?.done === true, state, { timeout: 900_000 });
  const result = await page.evaluate((s) => {
    const r = window[s];
    // Drop image payloads.
    return JSON.parse(JSON.stringify(r, (k, v) => (k === "rgba" || k === "pixels" || k === "image" ? undefined : v)));
  }, state);
  await page.close();
  return { ...run, wallSeconds: (Date.now() - t0) / 1000, consoleErrors, result };
}

// --- report --------------------------------------------------------------------------------------------
const f2 = (v) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(2) : "—");
const f1 = (v) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(1) : "—");
const ratio = (a, b) => (typeof a === "number" && typeof b === "number" && b > 0 ? `${(a / b).toFixed(2)}×` : "—");
const top = (passes) => {
  const e = Object.entries(passes ?? {}).filter(([k]) => k !== "froxels" && k !== "other").sort((a, b) => b[1] - a[1]);
  return e.length ? `${e[0][0]} ${f2(e[0][1])}` : "—";
};

function markdown(report) {
  const lines = [];
  const { machine, adapter, provisional } = report;
  lines.push(`# K3.2c hardware timings`, "");
  lines.push(`- Adapter: **${adapter}**; Chrome ${report.browserVersion}; timestamp-query: ${report.timestampQuery ? "yes (per-pass GPU ms)" : "no (queue-synchronised frame time only)"}`);
  lines.push(`- ${provisional ? "**PROVISIONAL** (D25): not asserted quiet" : "Quiet machine asserted (--quiet)"}; load average ${machine.before.loadavg.join(" / ")} before, ${machine.after.loadavg.join(" / ")} after (${machine.before.cpus} CPUs); other headless browser processes: ${machine.before.otherHeadlessBrowserProcesses} before, ${machine.after.otherHeadlessBrowserProcesses} after`);
  if (report.note) lines.push(`- Note: ${report.note}`);
  lines.push(`- Medians of ${report.frames} frames after ${report.warmup} warm-up; ${report.size}; ms. GPU ms are three's render-pass timestamps; "frame wall" is a whole frame from render() to queue.onSubmittedWorkDone().`, "");
  const by = (g) => report.runs.filter((r) => r.group === g);

  const hdda = by("hdda");
  if (hdda.length) {
    lines.push(`## (1) HDDA on vs off ('volume' lighting, sparse; the inject pass holds the shadow march)`, "");
    lines.push(`| asset | LOD | inject on | inject off | on/off | froxels on | froxels off | frame wall on | frame wall off |`, `|---|---|---:|---:|---:|---:|---:|---:|---:|`);
    const keys = [...new Set(hdda.map((r) => r.key.replace(/\/hdda[01]$/, "")))];
    for (const k of keys) {
      const on = hdda.find((r) => r.key === `${k}/hdda1`)?.result ?? {};
      const off = hdda.find((r) => r.key === `${k}/hdda0`)?.result ?? {};
      const [asset, , lod] = k.split("/");
      lines.push(`| ${asset} | ${lod.replace("sparse-", "")} | ${f2(on.passesMs?.inject)} | ${f2(off.passesMs?.inject)} | ${ratio(on.passesMs?.inject, off.passesMs?.inject)} | ${f2(on.froxelsMs)} | ${f2(off.froxelsMs)} | ${f2(on.frameWallMs)} | ${f2(off.frameWallMs)} |`);
    }
    lines.push("");
  }

  const gap = by("gap");
  if (gap.length) {
    lines.push(`## (2) Sparse ('coarse-pyramid', HDDA on) vs dense`, "");
    lines.push(`| asset | lighting | dense froxels | sparse froxels | sparse/dense | dense top pass | sparse top pass | grid redraw dense / sparse | integrate | frame wall dense / sparse |`, `|---|---|---:|---:|---:|---|---|---:|---:|---:|`);
    const keys = [...new Set(gap.map((r) => r.key.replace(/\/(dense|sparse)$/, "")))];
    for (const k of keys) {
      const d = gap.find((r) => r.key === `${k}/dense`)?.result ?? {};
      const s = gap.find((r) => r.key === `${k}/sparse`)?.result ?? {};
      const [asset, lighting] = k.split("/");
      lines.push(`| ${asset} | ${lighting} | ${f2(d.froxelsMs)} | ${f2(s.froxelsMs)} | ${ratio(s.froxelsMs, d.froxelsMs)} | ${top(d.passesMs)} | ${top(s.passesMs)} | ${f2(d.gridRedrawMs)} / ${f2(s.gridRedrawMs)} | ${f2(s.integrateMs)} | ${f2(d.frameWallMs)} / ${f2(s.frameWallMs)} |`);
    }
    lines.push("", "Per-pass medians (ms):", "");
    for (const r of gap) lines.push(`- ${r.key}: ${JSON.stringify(r.result?.passesMs ?? {})}${r.result?.gridRedrawPassesMs ? `; grid redraw ${JSON.stringify(r.result.gridRedrawPassesMs)}` : ""}${r.result?.error ? ` ERROR ${r.result.error.split("\n")[0]}` : ""}`);
    lines.push("");
  }

  const k3 = by("k3frame");
  if (k3.length) {
    lines.push(`## K3.3 timing scene (step3-default-dir-isolated, his bunny, k3-frame's timing)`, "");
    lines.push(`| source | froxels | grid redraw | integrate | scene |`, `|---|---:|---:|---:|---:|`);
    for (const r of k3) {
      const t = r.result?.timings ?? {};
      lines.push(`| ${r.key.split("/")[1]} | ${f2(t.froxelsMs)} | ${f2(t.passes?.gridRedrawMs)} | ${f2(t.integrateMs)} | ${f2(t.sceneMs)} |`);
    }
    lines.push("");
  }

  const sampling = by("sampling");
  if (sampling.length) {
    lines.push(`## K3.2d sparse level-0 sampling ('coarse-pyramid', HDDA off): the source pass (density for 'frame', inject for 'volume')`, "");
    const variants = [...new Set(sampling.map((r) => r.key.split("/")[2]))];
    lines.push(`| asset | lighting | ${variants.map((v) => `${v} pass`).join(" | ")} | ${variants.map((v) => `${v} wall`).join(" | ")} |`, `|---|---|${variants.map(() => "---:").join("|")}|${variants.map(() => "---:").join("|")}|`);
    const keys = [...new Set(sampling.map((r) => r.key.split("/").slice(0, 2).join("/")))];
    for (const k of keys) {
      const [asset, lighting] = k.split("/");
      const res = (v) => sampling.find((r) => r.key === `${k}/${v}`)?.result ?? {};
      const pass = (v) => (lighting === "frame" ? res(v).passesMs?.density : res(v).passesMs?.inject);
      lines.push(`| ${asset} | ${lighting} | ${variants.map((v) => f2(pass(v))).join(" | ")} | ${variants.map((v) => f2(res(v).frameWallMs)).join(" | ")} |`);
    }
    lines.push("", "Per-pass medians (ms):", "");
    for (const r of sampling) lines.push(`- ${r.key}: ${JSON.stringify(r.result?.passesMs ?? {})}${r.result?.error ? ` ERROR ${r.result.error.split("\n")[0]}` : ""}`);
    lines.push("");
  }
  const custom = by("custom");
  if (custom.length) {
    lines.push(`## Custom configurations`, "", `| key | query | froxels | frame wall | passes |`, `|---|---|---:|---:|---|`);
    for (const r of custom) lines.push(`| ${r.key} | ${r.query.split("&w=")[0]} | ${f2(r.result?.froxelsMs)} | ${f2(r.result?.frameWallMs)} | ${JSON.stringify(r.result?.passesMs ?? {})}${r.result?.error ? ` ERROR ${r.result.error.split("\n")[0]}` : ""} |`);
    lines.push("");
  }
  const bench = by("bench")[0]?.result;
  if (bench?.rows) {
    lines.push(`## Sampling microbenchmark (compute; ns per sample, median; mismatches vs 'init' bit for bit)`, "");
    const variants = [...new Set(bench.rows.map((r) => r.variant))];
    lines.push(`| asset | stride | ${variants.join(" | ")} |`, `|---|---:|${variants.map(() => "---:").join("|")}|`);
    const keys = [...new Set(bench.rows.map((r) => `${r.asset}|${r.stride}`))];
    for (const k of keys) {
      const [asset, stride] = k.split("|");
      const cell = (v) => {
        const r = bench.rows.find((x) => x.asset === asset && String(x.stride) === stride && x.variant === v);
        return r ? `${r.nsPerSample.toFixed(3)}${r.mismatches ? ` (≠${r.mismatches})` : ""}` : "—";
      };
      lines.push(`| ${asset} | ${stride} | ${variants.map(cell).join(" | ")} |`);
    }
    lines.push("");
  }
  const pyr = by("pyramid")[0]?.result;
  if (pyr?.assets) {
    lines.push(`## (3) Coarse-pyramid build: GPU (vdbCreateBoxPyramid) vs CPU (vdbBoxPyramid + f16 conversion)`, "");
    lines.push(`| asset | type | L₀ | base | atlas | GPU (timestamps) | GPU (queue wall) | CPU | CPU/GPU | textures | GPU scratch | parity (max ulp, texels at 1 ulp) |`, `|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|`);
    for (const a of pyr.assets) {
      const gpu = a.gpuMs ?? a.gpuWallMs;
      lines.push(`| ${a.name} | ${a.type} | ${a.level} | ${a.dimensions.join("x")} | ${a.atlasDimensions?.join("x") ?? "—"} | ${f2(a.gpuMs)} | ${f2(a.gpuWallMs)} | ${f1(a.cpuMs)} | ${ratio(a.cpuMs, gpu)} | ${(a.gpuBytes.textures / 1024).toFixed(0)} KiB | ${(a.gpuBytes.scratch / 1024).toFixed(0)} KiB | ${Math.max(a.base.maxUlp, a.atlas?.maxUlp ?? 0)}, ${a.base.ulp1 + (a.atlas?.ulp1 ?? 0)} |`);
    }
    lines.push("");
  }
  const errors = report.runs.filter((r) => r.result?.error || r.consoleErrors.length || r.result?.uncaptured?.length);
  if (errors.length) {
    lines.push(`## Errors`, "");
    for (const r of errors) lines.push(`- ${r.key}: ${(r.result?.error ?? "").split("\n")[0]} ${r.consoleErrors.slice(0, 2).join(" | ")} ${(r.result?.uncaptured ?? []).slice(0, 2).join(" | ")}`);
    lines.push("");
  }
  return lines.join("\n");
}

// --- main ----------------------------------------------------------------------------------------------
const server = await startServer();
const before = machineState();
const browser = await chromium.launch({
  channel: ALLOW_SWIFTSHADER && process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ? undefined : "chrome",
  ...(ALLOW_SWIFTSHADER && process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {}),
  // Headed: macOS Chrome picks the hardware adapter only with a window server session; headless may fall back.
  headless: ALLOW_SWIFTSHADER,
  args: ["--enable-unsafe-webgpu", "--enable-webgpu-developer-features", ...(ALLOW_SWIFTSHADER ? ["--enable-features=Vulkan", "--enable-unsafe-swiftshader"] : [])],
});
const report = { at: new Date().toISOString(), size: SMALL ? "320x180, froxels 64x64x16, 16 sub-steps, 16 shadow steps (--small: not a timing)" : "1920x1080, froxels 512x512x24, 32 sub-steps, 32 shadow steps", frames: FRAMES, warmup: WARMUP, note: NOTE, provisional: !QUIET, browserVersion: browser.version(), machine: { before }, runs: [] };
try {
  // The adapter first: refuse SwiftShader (and print what the GPU is).
  const probe = await browser.newPage();
  await probe.goto(`${BASE}/src/harness/index.html`);
  const adapterInfo = await probe.evaluate(async () => {
    const a = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (!a) return null;
    const i = a.info ?? {};
    return { vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description, isFallbackAdapter: a.isFallbackAdapter ?? i.isFallbackAdapter ?? false, timestampQuery: a.features.has("timestamp-query") };
  });
  await probe.close();
  if (!adapterInfo) throw new Error("no WebGPU adapter in Chrome");
  const name = [adapterInfo.vendor, adapterInfo.architecture, adapterInfo.device, adapterInfo.description].filter(Boolean).join(" ");
  console.log(`adapter: ${name} (fallback ${adapterInfo.isFallbackAdapter}, timestamp-query ${adapterInfo.timestampQuery}); Chrome ${browser.version()}`);
  if (!ALLOW_SWIFTSHADER && (/swiftshader/i.test(name) || adapterInfo.isFallbackAdapter)) throw new Error(`refusing to time on a software adapter: ${name}`);
  report.adapter = name;
  report.adapterInfo = adapterInfo;
  report.timestampQuery = adapterInfo.timestampQuery;
  console.log(`machine: load ${before.loadavg.join(" / ")}, busiest ${before.busiest.slice(0, 4).map((p) => `${p.command} ${p.cpu}%`).join(", ")}; ${runs.length} runs`);

  for (const [i, run] of runs.entries()) {
    process.stdout.write(`[${i + 1}/${runs.length}] ${run.key} … `);
    try {
      const r = await runPage(browser, run);
      report.runs.push(r);
      const res = r.result ?? {};
      const summary =
        run.page === "k3-timing"
          ? `froxels ${f2(res.froxelsMs)} ms, wall ${f2(res.frameWallMs)} ms, passes ${JSON.stringify(res.passesMs ?? {})}`
          : run.page === "k3-sample-bench"
            ? `${(res.rows ?? []).length} rows`
          : run.page === "k3-frame"
            ? `timings ${JSON.stringify(res.timings ?? {})}`
            : `${(res.assets ?? []).map((a) => `${a.name} GPU ${f2(a.gpuMs ?? a.gpuWallMs)} / CPU ${f1(a.cpuMs)} ms`).join("; ")}`;
      console.log(`${res.error ? `ERROR ${String(res.error).split("\n")[0]}` : summary} (${r.wallSeconds.toFixed(0)} s)`);
    } catch (e) {
      console.log(`FAILED ${e.message}`);
      report.runs.push({ ...run, consoleErrors: [], result: { error: e.message } });
    }
  }
} finally {
  report.machine.after = machineState();
  await browser.close();
  server?.kill();
}

mkdirSync(OUT, { recursive: true });
const stamp = report.at.replace(/[:.]/g, "-");
const md = markdown(report);
writeFileSync(path.join(OUT, `timings-${stamp}.json`), `${JSON.stringify(report, null, 2)}\n`);
writeFileSync(path.join(OUT, `timings-${stamp}.md`), `${md}\n`);
writeFileSync(path.join(OUT, "timings-latest.json"), `${JSON.stringify(report, null, 2)}\n`);
writeFileSync(path.join(OUT, "timings-latest.md"), `${md}\n`);
console.log(`\n${md}\n\nwrote ${path.relative(ROOT, OUT)}/timings-${stamp}.{json,md}`);
