// GPU parity for upstream's WGSL read accessor (vendor/picovdb.wgsl, verbatim).
//
// For each fixture x variant: convert with src/convert.ts, draw deterministic
// probes with probeCoords (packages/nanovdb-wgsl/src/cpu/probe-coords.ts),
// run picovdbReadAccessorGetLevelIndex + picovdbGetValue on a real WebGPU
// device (headless Chromium, SwiftShader Vulkan), and compare against:
//   (1) the CPU NanoVDB reference `readValue` on the SOURCE grid, mapped
//       through PicoVDB's value rules (src/compare.ts expectedPico), which is
//       the parity gate;
//   (2) the spike's CPU mirror of the WGSL (src/reader.ts), which should match
//       bit for bit and checks the mirror itself.
// Two dispatch modes: "fresh" (new accessor per probe) and "chunk32" (one
// accessor walks 32 consecutive probes), so the accessor's cached paths run too.
//
// Bundle-free, like scripts/check-wgsl-compile.mjs: a throwaway localhost
// HTTP server (WebGPU needs a secure context) serves a blank page plus the
// binding blobs. The WGSL is inlined via page.evaluate.
//
// Run:  node experiments/picovdb-spike/scripts/gpu-parity.mjs [--count N] [--only substr]
// Exit code: 0 when every gated case passes, 1 on parity failures, 2+ on setup errors.

import { createServer } from "node:http";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { chromium } from "@playwright/test";
import { SPIKE_DIR, tsLoader } from "./lib/load-ts.mjs";

const CHROMIUM_EXECUTABLE = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ?? "/opt/pw-browsers/chromium";
const args = process.argv.slice(2);
const argVal = (name, dflt) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : dflt;
};
const COUNT = Number(argVal("--count", "20000"));
const ONLY = argVal("--only", "");
const CHUNK = 32;

const BINDINGS = ["grids", "roots", "uppers", "lowers", "leaves", "data"];

// Upstream's header comment lists these six bindings (commented out); the
// harness supplies them plus a probe input and a result output.
const HARNESS = /* wgsl */ `
// ---- gpu-parity harness (prepended/appended; vendored file untouched) ----
@group(0) @binding(0) var<storage, read> picovdb_grids: array<PicoVDBGrid>;
@group(0) @binding(1) var<storage, read> picovdb_roots: array<PicoVDBRoot>;
@group(0) @binding(2) var<storage, read> picovdb_uppers: array<PicoVDBUpper>;
@group(0) @binding(3) var<storage, read> picovdb_lowers: array<PicoVDBLower>;
@group(0) @binding(4) var<storage, read> picovdb_leaves: array<PicoVDBLeaf>;
@group(0) @binding(5) var<storage, read> picovdb_buffer: array<u32>;
@group(0) @binding(6) var<storage, read> parity_probes: array<vec4i>;
@group(0) @binding(7) var<storage, read_write> parity_out: array<vec4u>;

override CHUNK: u32 = 1u;

@compute @workgroup_size(64)
fn parity_main(@builtin(global_invocation_id) gid: vec3u) {
    let n = arrayLength(&parity_probes);
    let first = gid.x * CHUNK;
    if (first >= n) { return; }
    var acc: PicoVDBReadAccessor;
    picovdbReadAccessorInit(&acc, 0u);
    for (var k = 0u; k < CHUNK; k++) {
        let i = first + k;
        if (i >= n) { break; }
        let r = picovdbReadAccessorGetLevelIndex(&acc, parity_probes[i].xyz);
        let v = picovdbGetValue(0u, r.index);
        parity_out[i] = vec4u(bitcast<u32>(v), r.level, r.index, u32(r.isSurface));
    }
}
`;

// 6 (scale) x 2.5 ULP of a value in [0.5, 1) = 6 * 2.5 * 2^-24 ~= 9e-7.
const MIRROR_U8_ABS = 1e-6;

function f32bits(x) {
  const b = new Float32Array([x]);
  return new Uint32Array(b.buffer)[0];
}

async function main() {
  const wgsl = readFileSync(path.join(SPIKE_DIR, "vendor/picovdb.wgsl"), "utf8") + HARNESS;
  const ts = await tsLoader();
  const fx = await ts.load("src/fixtures.ts");
  const cv = await ts.load("src/convert.ts");
  const cmp = await ts.load("src/compare.ts");
  const rd = await ts.load("src/reader.ts");
  const { readValue } = await ts.load("../../packages/nanovdb-wgsl/src/cpu/read-value.ts");
  const { probeCoords } = await ts.load("../../packages/nanovdb-wgsl/src/cpu/probe-coords.ts");

  // Build every case up front (CPU side).
  const cases = [];
  for (const f of fx.loadAll()) {
    if (ONLY && !f.id.includes(ONLY)) continue;
    const variants = [];
    variants.push({ valueType: "f32", layout: "native", gated: f.id.startsWith("vdb-samples") });
    if (f.id.startsWith("primitives")) variants.push({ valueType: "f32", layout: "fixed", gated: true });
    if (f.kind === "sdf") variants.push({ valueType: "u8", layout: "native", gated: true });
    const coords = probeCoords({ seed: 0x7069636fn, count: COUNT, bboxMin: f.indexBBox.min, bboxMax: f.indexBBox.max });
    const probes = new Int32Array(coords.length * 4);
    coords.forEach((c, i) => probes.set([c[0], c[1], c[2], 0], i * 4));
    const src = coords.map((c) => readValue(f.image, c));
    for (const v of variants) {
      const { bytes } = cv.convertToPicoVDB([{ image: f.image, gridClassId: f.gridClassId }], v);
      const view = new rd.PicoVDBView(bytes);
      const slices = view.bindingSlices();
      cases.push({ f, v, coords, probes, src, bytes, view, slices, gridType: view.grid(0).gridType });
    }
  }

  // Hazard case (ungated): one root (odd count -> zero-key padding root) away
  // from the origin; probes in [0,4096)^3 match the padding root in the WGSL
  // root search (arrayLength of the padded roots binding) and index upper[1],
  // which does not exist.
  if (!ONLY || "hazard/padding-root".includes(ONLY)) {
    const { buildFromDense } = await ts.load("../../packages/vdb-web-tools/src/index.ts");
    const image = buildFromDense(new Float32Array(512).fill(0.5), [8, 8, 8], { origin: [-100, -100, -100] });
    const f = { id: "hazard/padding-root", kind: "fog", image, voxelSize: 1, gridClassId: 2 };
    const coords = [];
    for (let i = 0; i < 4096; i++) coords.push(i % 2 ? [(i * 37) % 4096, (i * 101) % 4096, (i * 13) % 4096] : [-100 + (i % 8), -100 + ((i >> 3) % 8), -100 + ((i >> 6) % 8)]);
    const probes = new Int32Array(coords.length * 4);
    coords.forEach((c, i) => probes.set([c[0], c[1], c[2], 0], i * 4));
    const { bytes } = cv.convertToPicoVDB([{ image, gridClassId: 2 }]);
    const view = new rd.PicoVDBView(bytes);
    cases.push({ f, v: { valueType: "f32", layout: "native", gated: false }, coords, probes, src: coords.map((c) => readValue(image, c)), bytes, view, slices: view.bindingSlices(), gridType: view.grid(0).gridType });
  }

  // Serve blobs: /c/<caseIndex>/<binding>, /c/<caseIndex>/probes
  const server = createServer((req, res) => {
    const m = /^\/c\/(\d+)\/(\w+)$/.exec(req.url ?? "");
    if (m) {
      const c = cases[Number(m[1])];
      const body = m[2] === "probes" ? new Uint8Array(c.probes.buffer) : c.slices[m[2]];
      res.writeHead(200, { "content-type": "application/octet-stream" });
      res.end(Buffer.from(body.buffer, body.byteOffset, body.byteLength));
      return;
    }
    res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
    res.end("<!doctype html><meta charset=utf-8><title>picovdb gpu parity</title><body>ok</body>");
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const base = `http://localhost:${server.address().port}`;

  const browser = await chromium.launch({
    executablePath: CHROMIUM_EXECUTABLE,
    args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--enable-unsafe-swiftshader"],
  });
  let exitCode = 0;
  const report = { count: COUNT, chunk: CHUNK, device: null, compile: null, cases: [] };
  try {
    const page = await browser.newPage();
    page.on("console", (msg) => { if (msg.type() === "error") console.log(`[page:error] ${msg.text()}`); });
    await page.goto(base + "/");

    const init = await page.evaluate(async (source) => {
      if (!("gpu" in navigator)) return { fatal: "navigator.gpu unavailable" };
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return { fatal: "no WebGPU adapter" };
      const device = await adapter.requestDevice({
        requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize },
      });
      const t0 = performance.now();
      device.pushErrorScope("validation");
      const module = device.createShaderModule({ code: source });
      const info = await module.getCompilationInfo();
      const pipelines = {};
      for (const chunk of [1, 32]) {
        pipelines[chunk] = await device.createComputePipelineAsync({
          layout: "auto",
          compute: { module, entryPoint: "parity_main", constants: { CHUNK: chunk } },
        });
      }
      const err = await device.popErrorScope();
      window.__pv = { device, pipelines };
      return {
        adapterInfo: { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture, description: adapter.info?.description },
        limits: {
          maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
          defaultMaxStorageBuffersPerShaderStage: 8,
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        },
        compileMs: Math.round(performance.now() - t0),
        messages: info.messages.map((m) => ({ type: m.type, line: m.lineNum, message: m.message })),
        validation: err ? err.message : null,
      };
    }, wgsl);
    if (init.fatal) throw new Error(init.fatal);
    report.device = { adapterInfo: init.adapterInfo, limits: init.limits };
    report.compile = { ms: init.compileMs, messages: init.messages, validation: init.validation };
    const errors = init.messages.filter((m) => m.type === "error");
    console.log(`compile: ${init.messages.length} message(s), ${errors.length} error(s), ${init.compileMs} ms; validation: ${init.validation ?? "clean"}`);
    console.log(`adapter limits: maxStorageBuffersPerShaderStage=${init.limits.maxStorageBuffersPerShaderStage} (spec default 8; this harness binds 8: picovdb's 6 + probes + output)`);
    for (const m of init.messages) console.log(`  [${m.type}] line ${m.line}: ${m.message}`);
    if (errors.length || init.validation) throw new Error("WGSL compile/pipeline failed");

    for (let ci = 0; ci < cases.length; ci++) {
      const c = cases[ci];
      const label = `${c.f.id} ${c.v.valueType}/${c.v.layout}`;
      const isFog = c.gridType === cv.GRID_TYPE_FOG_FLOAT;
      const u8 = c.gridType === cv.GRID_TYPE_SDF_UINT8;
      const tol = u8 ? 3 / 255 + 1e-6 : 0;
      // Expected from the NanoVDB source (mapped), and from the CPU mirror per mode.
      const expected = c.src.map((s) => {
        let e = cmp.expectedPico(s, { isFog, voxelSize: c.f.voxelSize, background: cmp.sourceBackground(c.f.image) });
        if (u8) e = Math.min(3, Math.max(-3, e));
        return e;
      });
      const mirror = {};
      for (const chunk of [1, CHUNK]) {
        const out = new Float32Array(c.coords.length);
        for (let s = 0; s < c.coords.length; s += chunk) {
          const acc = new rd.PicoVDBAccessor(c.view);
          for (let k = s; k < Math.min(s + chunk, c.coords.length); k++) out[k] = acc.read(c.coords[k]).value;
        }
        mirror[chunk] = out;
      }

      const caseReport = { fixture: c.f.id, valueType: c.v.valueType, layout: c.v.layout, gated: c.v.gated, modes: {} };
      for (const chunk of [1, CHUNK]) {
        const t0 = performance.now();
        const gpu = await page.evaluate(async ({ ci, chunk, bindings }) => {
          const { device, pipelines } = window.__pv;
          const fetchBuf = async (name) => new Uint8Array(await (await fetch(`/c/${ci}/${name}`)).arrayBuffer());
          const mk = (bytes, usage) => {
            const size = Math.max(16, Math.ceil(bytes.byteLength / 16) * 16);
            const b = device.createBuffer({ size, usage, mappedAtCreation: true });
            new Uint8Array(b.getMappedRange()).set(bytes);
            b.unmap();
            return b;
          };
          const bufs = [];
          for (const name of bindings) bufs.push(mk(await fetchBuf(name), GPUBufferUsage.STORAGE));
          const probes = await fetchBuf("probes");
          bufs.push(mk(probes, GPUBufferUsage.STORAGE));
          const n = probes.byteLength / 16;
          const outBuf = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
          const read = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
          const pipeline = pipelines[chunk];
          device.pushErrorScope("validation");
          const bg = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [...bufs, outBuf].map((buffer, binding) => ({ binding, resource: { buffer } })),
          });
          const enc = device.createCommandEncoder();
          const pass = enc.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(Math.ceil(Math.ceil(n / chunk) / 64));
          pass.end();
          enc.copyBufferToBuffer(outBuf, 0, read, 0, n * 16);
          device.queue.submit([enc.finish()]);
          const err = await device.popErrorScope();
          await read.mapAsync(GPUMapMode.READ);
          const out = Array.from(new Uint32Array(read.getMappedRange().slice(0)));
          read.unmap();
          for (const b of [...bufs, outBuf, read]) b.destroy();
          return { out, validation: err ? err.message : null };
        }, { ci, chunk, bindings: BINDINGS });
        const ms = Math.round(performance.now() - t0);
        if (gpu.validation) throw new Error(`${label}: validation error: ${gpu.validation}`);

        let paddingRootHits = 0;
        let failures = 0, maxDelta = 0, mirrorExact = 0, mirrorMaxAbs = 0, activeFail = 0, active = 0, maxActiveRawDelta = 0;
        const firstFailures = [];
        const f32 = new Float32Array(1);
        const u32v = new Uint32Array(f32.buffer);
        for (let i = 0; i < c.coords.length; i++) {
          u32v[0] = gpu.out[i * 4];
          const g = f32[0];
          const d = Math.abs(g - expected[i]);
          // Coords in the padding root's key range that resolve below root level on a 1-root grid.
          if (c.f.id === "hazard/padding-root" && c.coords[i][0] >= 0 && gpu.out[i * 4 + 1] !== 3) paddingRootHits++;
          if (!(d <= tol)) {
            failures++;
            if (c.src[i].active) activeFail++;
            if (firstFailures.length < 4) firstFailures.push({ ijk: c.coords[i], gpu: g, expected: expected[i], level: gpu.out[i * 4 + 1] });
          }
          if (Number.isFinite(d)) maxDelta = Math.max(maxDelta, d);
          const mb = f32bits(mirror[chunk][i]);
          if (gpu.out[i * 4] === mb) mirrorExact++;
          else mirrorMaxAbs = Math.max(mirrorMaxAbs, Math.abs(g - mirror[chunk][i]));
          if (c.src[i].active) {
            active++;
            // Raw delta vs the NanoVDB value in PicoVDB units (fog: identity; SDF: / voxelSize).
            const raw = isFog ? c.src[i].value : Math.fround(c.src[i].value / Math.fround(c.f.voxelSize));
            if (Number.isFinite(g - raw)) maxActiveRawDelta = Math.max(maxActiveRawDelta, Math.abs(g - raw));
          }
        }
        const mode = chunk === 1 ? "fresh" : `chunk${chunk}`;
        caseReport.modes[mode] = { total: c.coords.length, failures, active, activeFail, maxDelta, maxActiveRawDelta, mirrorExact, mirrorMaxAbs, paddingRootHits, ms, firstFailures };
        // The mirror must be bit-exact, except u8 dequantization: WGSL allows
        // 2.5 ULP for unpack4x8unorm's /255, and fma may be unfused; near zero
        // (codes ~127) the "- 3" cancels, so bound the absolute error instead.
        const mirrorOk = mirrorExact === c.coords.length || (u8 && mirrorMaxAbs <= MIRROR_U8_ABS);
        const pass = failures === 0 && mirrorOk;
        if (c.v.gated && !pass) exitCode = 1;
        console.log(
          `${pass ? "PASS" : c.v.gated ? "FAIL" : "DIFF"} ${label.padEnd(36)} ${mode.padEnd(7)} failures ${failures}/${c.coords.length} ` +
            `(active ${activeFail}/${active})  max|d| ${maxDelta.toExponential(2)}  mirror bit-exact ${mirrorExact}/${c.coords.length}${mirrorMaxAbs ? ` (max |d| ${mirrorMaxAbs.toExponential(1)})` : ""}  ${ms} ms` +
            (c.f.id === "hazard/padding-root" ? `  padding-root hits ${paddingRootHits}` : "") +
            (c.v.gated ? "" : "  [ungated: documents an upstream defect]"),
        );
      }
      report.cases.push(caseReport);
    }
  } finally {
    await browser.close();
    server.close();
    await ts.close();
  }
  mkdirSync(path.join(SPIKE_DIR, ".out"), { recursive: true });
  writeFileSync(path.join(SPIKE_DIR, ".out/gpu-parity.json"), JSON.stringify(report, null, 2));
  console.log(`\nwrote .out/gpu-parity.json; ${exitCode === 0 ? "all gated cases PASS" : "gated FAILURES"}`);
  process.exit(exitCode);
}

main().catch((err) => {
  console.error(err);
  process.exit(3);
});
