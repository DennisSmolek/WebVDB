/**
 * GPU parity harness — the Phase 2 gate (docs/PLAN.md): replay thousands of
 * deterministic voxel probes and trilinear samples across all nine fixtures
 * through the extended vendored WGSL module (`nanovdb-wgsl/pnanovdb.wgsl`),
 * comparing against the CPU reference (`nanovdb-wgsl`'s own `readValue`/
 * `sampleTrilinear` — Phase 1-proven `packages/nanovdb-wgsl/src/cpu/*`
 * descent, package-exported since Phase 5's dedup, see
 * docs/handoffs/PHASE-5.md's "Known debts").
 *
 * Architecture (deliberately raw WebGPU, no three.js/TSL — see the T2 brief):
 * the TSL binding pattern was already proven by demo 01; this harness's job
 * is validating the WGSL module itself with minimal machinery.
 *
 * Pipeline, once per page load:
 *   1. Device-first bootstrap (adapter -> device with a raised
 *      `maxStorageBufferBindingSize`, matching demo 01's D4 pattern).
 *   2. Compile ONE shader module = vendored WGSL + a small harness footer
 *      (`shader-footer.ts`) declaring `nanovdb_buffer` and two `@compute`
 *      entry points (`harness_main_probe`, `harness_main_trilinear`).
 *      `getCompilationInfo()` is checked immediately — WGSL errors are
 *      surfaced verbatim into `window.__GPU_PARITY__.error`, the debugging
 *      channel.
 *   3. Build two explicit (non-'auto') pipelines once, so per-fixture work
 *      is just buffers + bind groups + dispatch + readback.
 *
 * Per fixture (all nine under fixtures/primitives/, git-ignored — missing
 * files are skipped gracefully):
 *   - Parse with `NanoVDBFile`, read the grid's numeric type id straight out
 *     of the grid image (byte PNANOVDB_GRID_OFF_GRID_TYPE) rather than
 *     trusting a name string.
 *   - Generate 2000 deterministic integer probes + 500 continuous trilinear
 *     points (`nanovdb-wgsl`'s `probeCoords`/`probePoints`, seed = fixture
 *     index).
 *   - Compute CPU-truth values/active flags in-page.
 *   - Dispatch both entry points, read back, compare
 *     (probes |Δ| ≤ 1e-5 + active exact; trilinear |Δ| ≤ 1e-3 — GPU is pure
 *     f32, CPU accumulates in f64).
 *   - Publish per-fixture pass/fail + max deltas into
 *     `window.__GPU_PARITY__` (the e2e gate's read channel) and an on-page
 *     table.
 */
import { NanoVDBFile, defineNumber, probeCoords, probePoints, readValue, sampleTrilinear } from "nanovdb-wgsl";
import pnanovdbWgsl from "nanovdb-wgsl/pnanovdb.wgsl?raw";

import { HARNESS_FOOTER_WGSL } from "./shader-footer";

// ---------------------------------------------------------------------------
// Types shared with the e2e gate (window.__GPU_PARITY__).
// ---------------------------------------------------------------------------

interface FixtureCounts {
  total: number;
  failed: number;
  maxDelta: number;
}

interface FixtureResult {
  name: string;
  gridType: string;
  gridTypeId: number;
  probes: FixtureCounts;
  trilinear: FixtureCounts;
  timingMs: number;
  error?: string;
}

interface GpuParityResult {
  fixtures: FixtureResult[];
  done: boolean;
  error?: string;
}

declare global {
  interface Window {
    __GPU_PARITY__?: GpuParityResult;
  }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const FIXTURE_NAMES = [
  "box_fog_float",
  "box_fog_fp8",
  "box_fog_fpn",
  "sphere_fog_float",
  "sphere_fog_fp8",
  "sphere_fog_fpn",
  "torus_fog_float",
  "torus_fog_fp8",
  "torus_fog_fpn",
] as const;

const PROBE_COUNT = 2000;
const TRILINEAR_COUNT = 500;
const PROBE_VALUE_EPS = 1e-5;
const TRILINEAR_VALUE_EPS = 1e-3;
const WORKGROUP_SIZE = 64;
const DEFAULT_STORAGE_LIMIT = 128 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Page scaffolding
// ---------------------------------------------------------------------------

const root = document.querySelector<HTMLDivElement>("#demo-root");
if (!root) throw new Error("missing #demo-root element");

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  if (cls) p.className = cls;
  p.textContent = text;
  root!.appendChild(p);
}

function publish(result: GpuParityResult): void {
  window.__GPU_PARITY__ = result;
}

function renderTable(fixtures: FixtureResult[]): void {
  const table = document.createElement("table");
  table.innerHTML =
    "<thead><tr><th>fixture</th><th>type</th><th>probes</th><th>probe maxΔ</th>" +
    "<th>trilinear</th><th>tri maxΔ</th><th>ms</th></tr></thead>";
  const tbody = document.createElement("tbody");
  for (const f of fixtures) {
    const ok = f.probes.failed === 0 && f.trilinear.failed === 0 && !f.error;
    const tr = document.createElement("tr");
    tr.className = ok ? "row-ok" : "row-fail";
    tr.innerHTML =
      `<td>${f.name}</td><td>${f.gridType}</td>` +
      `<td>${f.probes.total - f.probes.failed}/${f.probes.total}</td>` +
      `<td>${f.probes.maxDelta.toExponential(2)}</td>` +
      `<td>${f.trilinear.total - f.trilinear.failed}/${f.trilinear.total}</td>` +
      `<td>${f.trilinear.maxDelta.toExponential(2)}</td>` +
      `<td>${f.timingMs.toFixed(1)}</td>`;
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  root!.appendChild(table);
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/** Smallest power of two >= n (tidy storage-binding-size limit request). */
function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

function ceilDiv(a: number, b: number): number {
  return Math.ceil(a / b);
}

function alignTo4(n: number): number {
  return Math.ceil(n / 4) * 4;
}

function makeReadOnlyStorage(device: GPUDevice, data: Uint32Array | Int32Array | Float32Array): GPUBuffer {
  const buf = device.createBuffer({
    size: Math.max(4, alignTo4(data.byteLength)),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  // `.slice()` copies into a fresh, byteOffset-0 ArrayBuffer of the same
  // element type, sidestepping both (a) `data.buffer`'s `ArrayBufferLike`
  // typing (lib.dom.d.ts allows SharedArrayBuffer there, which
  // GPUAllowSharedBufferSource's TS type rejects) and (b) any ambiguity
  // about whether a raw-ArrayBuffer overload's offset/size args are bytes
  // or elements — passing the view directly always means "the whole view".
  device.queue.writeBuffer(buf, 0, data.slice());
  return buf;
}

function makeOutputStorage(device: GPUDevice, byteLength: number): GPUBuffer {
  return device.createBuffer({
    size: Math.max(4, alignTo4(byteLength)),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
}

function makeReadback(device: GPUDevice, byteLength: number): GPUBuffer {
  return device.createBuffer({
    size: Math.max(4, alignTo4(byteLength)),
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
}

interface FixtureFile {
  name: string;
  buffer: ArrayBuffer | null; // null = 404 / missing
}

async function fetchFixtures(): Promise<FixtureFile[]> {
  return Promise.all(
    FIXTURE_NAMES.map(async (name): Promise<FixtureFile> => {
      const url = `/fixtures/primitives/${name}.nvdb`;
      try {
        const res = await fetch(url);
        if (!res.ok) return { name, buffer: null };
        return { name, buffer: await res.arrayBuffer() };
      } catch {
        return { name, buffer: null };
      }
    }),
  );
}

// ---------------------------------------------------------------------------
// Bind group layouts (explicit — NOT 'auto' — so group(0) can be reused
// across both pipelines; auto-derived layouts are opaque per-pipeline and
// not meant to be shared, see docs/handoffs — group(0) holds nanovdb_buffer +
// harness_grid_type, common to both entry points).
// ---------------------------------------------------------------------------

function createGroup0Layout(device: GPUDevice): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ],
  });
}

function createGroup1Layout(device: GPUDevice): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
}

function createGroup2Layout(device: GPUDevice): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function run(): Promise<void> {
  root!.textContent = "";
  log("GPU Parity Harness — Phase 2 gate (docs/PLAN.md)", "h");

  if (!navigator.gpu) {
    const msg = "WebGPU unavailable (navigator.gpu undefined) — needs a secure context + adapter.";
    log(msg, "fail");
    publish({ fixtures: [], done: true, error: msg });
    return;
  }

  // -------------------------------------------------------------------------
  // 1. Layout constants for the byte this harness reads straight out of every
  //    grid image (the numeric grid-type id, for logging + the GPU
  //    `harness_grid_type` buffer). `nanovdb-wgsl`'s own `readValue`/
  //    `sampleTrilinear` (used below as this harness's CPU truth) detect the
  //    grid type themselves from the same bytes, so this is the only offset
  //    this file needs directly.
  // -------------------------------------------------------------------------
  const gridOffGridType = defineNumber("PNANOVDB_GRID_OFF_GRID_TYPE");

  // -------------------------------------------------------------------------
  // 2. Fixtures (git-ignored; skip gracefully per-file).
  // -------------------------------------------------------------------------
  const fixtureFiles = await fetchFixtures();
  const missing = fixtureFiles.filter((f) => f.buffer === null).map((f) => f.name);
  if (missing.length > 0) {
    log(`Missing fixtures (skipped): ${missing.join(", ")}`, "note");
  }
  const present = fixtureFiles.filter(
    (f): f is { name: string; buffer: ArrayBuffer } => f.buffer !== null,
  );
  if (present.length === 0) {
    const msg = "No fixtures found under /fixtures/primitives/ (git-ignored assets not present on this machine).";
    log(msg, "fail");
    publish({ fixtures: [], done: true, error: msg });
    return;
  }

  // -------------------------------------------------------------------------
  // 3. Device-first bootstrap (D4): our adapter, our device, our limits.
  // -------------------------------------------------------------------------
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    const msg = "No WebGPU adapter (headless without SwiftShader?).";
    log(msg, "fail");
    publish({ fixtures: [], done: true, error: msg });
    return;
  }

  const adapterMax = adapter.limits.maxStorageBufferBindingSize;
  const largestGrid = Math.max(...present.map((f) => f.buffer.byteLength));
  const needed = nextPow2(largestGrid);
  const requestedBinding = Math.min(adapterMax, Math.max(DEFAULT_STORAGE_LIMIT, needed));
  log(
    `Adapter maxStorageBufferBindingSize = ${(adapterMax / 1024 / 1024).toFixed(0)} MiB; ` +
      `requesting ${(requestedBinding / 1024 / 1024).toFixed(0)} MiB ` +
      `(largest of ${present.length} fixtures needs ${(needed / 1024 / 1024).toFixed(2)} MiB).`,
  );

  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: requestedBinding,
      maxBufferSize: requestedBinding,
    },
  });

  // -------------------------------------------------------------------------
  // 4. Compile once: vendored WGSL + harness footer. Surface compile errors
  //    verbatim — this is the debugging channel for WGSL authoring mistakes.
  // -------------------------------------------------------------------------
  const shaderSource = `${pnanovdbWgsl}\n${HARNESS_FOOTER_WGSL}`;
  const shaderModule = device.createShaderModule({ code: shaderSource });
  const info = await shaderModule.getCompilationInfo();
  const compileErrors = info.messages.filter((m) => m.type === "error");
  if (compileErrors.length > 0) {
    const detail = compileErrors.map((m) => `line ${m.lineNum}:${m.linePos}: ${m.message}`).join("\n");
    log(`WGSL compile error:\n${detail}`, "fail");
    publish({ fixtures: [], done: true, error: `WGSL compilation failed:\n${detail}` });
    return;
  }
  log("Shader module compiled clean (no getCompilationInfo() errors).", "ok");

  const group0Layout = createGroup0Layout(device);
  const group1Layout = createGroup1Layout(device);
  const group2Layout = createGroup2Layout(device);
  // `GPUPipelineLayout.bindGroupLayouts` is positional — index i describes
  // @group(i). The trilinear entry point only uses @group(0) and @group(2)
  // (skipping @group(1), which is probe-only), so its layout needs an
  // explicit empty placeholder at index 1; passing a 2-element array here
  // silently mislabels group2Layout as the layout for @group(1) and leaves
  // @group(2) undeclared, which (found the hard way — see the harness
  // report) fails pipeline creation and silently invalidates the whole
  // command buffer, so probe results downstream are just the buffers'
  // zero-initialized bytes, not a real dispatch.
  const emptyLayout = device.createBindGroupLayout({ entries: [] });
  const emptyBindGroup = device.createBindGroup({ layout: emptyLayout, entries: [] });

  const probePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [group0Layout, group1Layout] }),
    compute: { module: shaderModule, entryPoint: "harness_main_probe" },
  });
  const triPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [group0Layout, emptyLayout, group2Layout] }),
    compute: { module: shaderModule, entryPoint: "harness_main_trilinear" },
  });

  // -------------------------------------------------------------------------
  // 5. Per-fixture probe + trilinear dispatch, compared against CPU truth.
  // -------------------------------------------------------------------------
  const fixtures: FixtureResult[] = [];

  for (let n = 0; n < present.length; n++) {
    const { name, buffer } = present[n]!;
    const t0 = performance.now();
    try {
      const file = NanoVDBFile.fromArrayBuffer(buffer);
      const gridImage = file.gridImage(0);
      const gridMeta = file.grids[0]!;
      const gridTypeId = gridImage[gridOffGridType >>> 2]!;

      const bboxMin = gridMeta.indexBBox.min;
      const bboxMax = gridMeta.indexBBox.max;
      const seed = BigInt(n);

      const coords = probeCoords({ seed, count: PROBE_COUNT, bboxMin, bboxMax });
      const points = probePoints({ seed, count: TRILINEAR_COUNT, bboxMin, bboxMax });

      // ---- CPU truth ----
      const expectedValues = new Float32Array(PROBE_COUNT);
      const expectedActive = new Uint8Array(PROBE_COUNT);
      for (let i = 0; i < PROBE_COUNT; i++) {
        const r = readValue(gridImage, coords[i]!);
        expectedValues[i] = r.value;
        expectedActive[i] = r.active ? 1 : 0;
      }
      const expectedTri = new Float32Array(TRILINEAR_COUNT);
      for (let i = 0; i < TRILINEAR_COUNT; i++) {
        expectedTri[i] = sampleTrilinear(gridImage, points[i]!);
      }

      // ---- flat GPU inputs ----
      const coordFlat = new Int32Array(PROBE_COUNT * 3);
      for (let i = 0; i < PROBE_COUNT; i++) {
        const [x, y, z] = coords[i]!;
        coordFlat[i * 3] = x;
        coordFlat[i * 3 + 1] = y;
        coordFlat[i * 3 + 2] = z;
      }
      const pointFlat = new Float32Array(TRILINEAR_COUNT * 3);
      for (let i = 0; i < TRILINEAR_COUNT; i++) {
        const [x, y, z] = points[i]!;
        pointFlat[i * 3] = x;
        pointFlat[i * 3 + 1] = y;
        pointFlat[i * 3 + 2] = z;
      }

      // ---- GPU buffers ----
      const gridBuf = makeReadOnlyStorage(device, gridImage);
      const gridTypeBuf = makeReadOnlyStorage(device, new Uint32Array([gridTypeId]));
      const group0 = device.createBindGroup({
        layout: group0Layout,
        entries: [
          { binding: 0, resource: { buffer: gridBuf } },
          { binding: 1, resource: { buffer: gridTypeBuf } },
        ],
      });

      const coordBuf = makeReadOnlyStorage(device, coordFlat);
      const probeValueOut = makeOutputStorage(device, PROBE_COUNT * 4);
      const probeActiveOut = makeOutputStorage(device, PROBE_COUNT * 4);
      const group1 = device.createBindGroup({
        layout: group1Layout,
        entries: [
          { binding: 0, resource: { buffer: coordBuf } },
          { binding: 1, resource: { buffer: probeValueOut } },
          { binding: 2, resource: { buffer: probeActiveOut } },
        ],
      });

      const triBuf = makeReadOnlyStorage(device, pointFlat);
      const triValueOut = makeOutputStorage(device, TRILINEAR_COUNT * 4);
      const group2 = device.createBindGroup({
        layout: group2Layout,
        entries: [
          { binding: 0, resource: { buffer: triBuf } },
          { binding: 1, resource: { buffer: triValueOut } },
        ],
      });

      const probeValueReadback = makeReadback(device, PROBE_COUNT * 4);
      const probeActiveReadback = makeReadback(device, PROBE_COUNT * 4);
      const triValueReadback = makeReadback(device, TRILINEAR_COUNT * 4);

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(probePipeline);
      pass.setBindGroup(0, group0);
      pass.setBindGroup(1, group1);
      pass.dispatchWorkgroups(ceilDiv(PROBE_COUNT, WORKGROUP_SIZE));
      pass.setPipeline(triPipeline);
      pass.setBindGroup(0, group0);
      pass.setBindGroup(1, emptyBindGroup);
      pass.setBindGroup(2, group2);
      pass.dispatchWorkgroups(ceilDiv(TRILINEAR_COUNT, WORKGROUP_SIZE));
      pass.end();
      encoder.copyBufferToBuffer(probeValueOut, 0, probeValueReadback, 0, probeValueReadback.size);
      encoder.copyBufferToBuffer(probeActiveOut, 0, probeActiveReadback, 0, probeActiveReadback.size);
      encoder.copyBufferToBuffer(triValueOut, 0, triValueReadback, 0, triValueReadback.size);
      device.queue.submit([encoder.finish()]);

      await Promise.all([
        probeValueReadback.mapAsync(GPUMapMode.READ),
        probeActiveReadback.mapAsync(GPUMapMode.READ),
        triValueReadback.mapAsync(GPUMapMode.READ),
      ]);
      const gotProbeValues = new Float32Array(probeValueReadback.getMappedRange().slice(0));
      const gotProbeActive = new Uint32Array(probeActiveReadback.getMappedRange().slice(0));
      const gotTriValues = new Float32Array(triValueReadback.getMappedRange().slice(0));
      probeValueReadback.unmap();
      probeActiveReadback.unmap();
      triValueReadback.unmap();

      // ---- compare vs CPU truth ----
      let probeFailed = 0;
      let probeMaxDelta = 0;
      for (let i = 0; i < PROBE_COUNT; i++) {
        const dv = Math.abs(gotProbeValues[i]! - expectedValues[i]!);
        if (dv > probeMaxDelta) probeMaxDelta = dv;
        const gotActive = gotProbeActive[i] !== 0;
        const wantActive = expectedActive[i] !== 0;
        if (dv > PROBE_VALUE_EPS || gotActive !== wantActive) {
          probeFailed++;
        }
      }
      let triFailed = 0;
      let triMaxDelta = 0;
      for (let i = 0; i < TRILINEAR_COUNT; i++) {
        const dv = Math.abs(gotTriValues[i]! - expectedTri[i]!);
        if (dv > triMaxDelta) triMaxDelta = dv;
        if (dv > TRILINEAR_VALUE_EPS) triFailed++;
      }

      const timingMs = performance.now() - t0;

      fixtures.push({
        name,
        gridType: gridMeta.gridType,
        gridTypeId,
        probes: { total: PROBE_COUNT, failed: probeFailed, maxDelta: probeMaxDelta },
        trilinear: { total: TRILINEAR_COUNT, failed: triFailed, maxDelta: triMaxDelta },
        timingMs,
      });

      log(
        `${name} (${gridMeta.gridType}): probes ${PROBE_COUNT - probeFailed}/${PROBE_COUNT} ` +
          `(maxΔ ${probeMaxDelta.toExponential(2)}), trilinear ${TRILINEAR_COUNT - triFailed}/${TRILINEAR_COUNT} ` +
          `(maxΔ ${triMaxDelta.toExponential(2)}), ${timingMs.toFixed(1)} ms`,
        probeFailed === 0 && triFailed === 0 ? "ok" : "fail",
      );

      gridBuf.destroy();
      gridTypeBuf.destroy();
      coordBuf.destroy();
      probeValueOut.destroy();
      probeActiveOut.destroy();
      triBuf.destroy();
      triValueOut.destroy();
      probeValueReadback.destroy();
      probeActiveReadback.destroy();
      triValueReadback.destroy();
    } catch (err) {
      const message = err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err);
      log(`${name}: ERROR ${message}`, "fail");
      fixtures.push({
        name,
        gridType: "unknown",
        gridTypeId: -1,
        probes: { total: PROBE_COUNT, failed: PROBE_COUNT, maxDelta: Number.POSITIVE_INFINITY },
        trilinear: { total: TRILINEAR_COUNT, failed: TRILINEAR_COUNT, maxDelta: Number.POSITIVE_INFINITY },
        timingMs: performance.now() - t0,
        error: message,
      });
    }
  }

  publish({ fixtures, done: true });
  renderTable(fixtures);

  const totalFailed = fixtures.reduce((acc, f) => acc + f.probes.failed + f.trilinear.failed, 0);
  log(
    `Done: ${fixtures.length}/${FIXTURE_NAMES.length} fixtures exercised, ${totalFailed} total mismatches.`,
    totalFailed === 0 ? "ok" : "fail",
  );
}

run().catch((err: unknown) => {
  const message = err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err);
  log(`ERROR: ${message}`, "fail");
  const prev = window.__GPU_PARITY__;
  window.__GPU_PARITY__ = {
    fixtures: prev?.fixtures ?? [],
    done: true,
    error: message,
  };
});
