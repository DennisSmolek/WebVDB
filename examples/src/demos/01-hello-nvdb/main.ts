/**
 * Demo 01 — hello-nvdb: the Phase 1 de-risking spike.
 *
 * Goal (docs/PLAN.md Phase 1, docs/FEASIBILITY.md §9 risk #1): prove that a
 * hand-written WGSL NanoVDB traversal can run inside three.js TSL, reading a
 * NanoVDB grid image from a `storage` buffer, and that its per-voxel results
 * match native ground truth.
 *
 * Pipeline:
 *   1. Device-first bootstrap (decision D4): request the adapter and create
 *      the GPUDevice ourselves with a raised `maxStorageBufferBindingSize`,
 *      then hand that device to `WebGPURenderer`.
 *   2. Load `sphere_fog_float.nvdb` -> a flat u32 grid image, upload it as a
 *      read-only storage buffer. Upload the 73 sidecar probe coords as a
 *      second storage buffer.
 *   3. Dispatch a compute pass of 73 threads. Each thread reads its coord,
 *      calls the raw WGSL `nvdb_probe_float` (bound via `wgslFn` with a
 *      `ptr<storage, array<u32>, read>` param), and writes `value` + `active`
 *      into two output storage buffers.
 *   4. Read the outputs back, compare against the sidecar in JS, render a
 *      pass/fail table, and publish `window.__DEMO01__` for the e2e gate.
 */
import { NanoVDBFile } from "nanovdb-wgsl";
import { WebGPURenderer, StorageBufferAttribute } from "three/webgpu";
import { Fn, instanceIndex, ivec3, storage, uint, vec2, wgslFn } from "three/tsl";

import { NVDB_PROBE_FLOAT_WGSL } from "./traversal";

// ---------------------------------------------------------------------------
// Types shared with the e2e harness (window.__DEMO01__).
// ---------------------------------------------------------------------------

interface Mismatch {
  ijk: [number, number, number];
  expectedValue: number;
  gotValue: number;
  expectedActive: boolean;
  gotActive: boolean;
}

interface Demo01Result {
  total: number;
  passed: number;
  failed: number;
  mismatches: Mismatch[];
  gridName: string;
  binding: string;
  computeMs?: number;
  error?: string;
}

interface Sample {
  ijk: [number, number, number];
  value: number;
  active: boolean;
}

interface Sidecar {
  grid: { name: string };
  samples: Sample[];
}

declare global {
  interface Window {
    __DEMO01__?: Demo01Result;
  }
}

const VALUE_EPS = 1e-5;
const FIXTURE_NVDB = "/fixtures/primitives/sphere_fog_float.nvdb";
const FIXTURE_SIDECAR = "/fixtures/primitives/sphere_fog_float.sidecar.json";

// One-line description of the binding pattern that worked — the primary
// deliverable of the spike, echoed into window.__DEMO01__.binding.
const BINDING_DESCRIPTION =
  "wgslFn fn(grid: ptr<storage, array<u32>, read>, ijk: vec3<i32>) called from " +
  "Fn().compute(N); grid arg = storage(StorageBufferAttribute,'uint',count).toReadOnly() " +
  "which TSL emits as `&NodeBufferN.value` (a `ptr<storage, array<u32>, read>`).";

const root = document.querySelector<HTMLDivElement>("#demo-root");
if (!root) throw new Error("missing #demo-root element");

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  if (cls) p.className = cls;
  p.textContent = text;
  root!.appendChild(p);
}

/** Smallest power of two >= n (for a tidy storage-binding-size limit). */
function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

async function run(): Promise<void> {
  root!.textContent = "";
  log("Demo 01 — hello-nvdb (Phase 1 spike)", "h");

  if (!navigator.gpu) {
    const msg = "WebGPU unavailable (navigator.gpu undefined) — needs a secure context + adapter.";
    log(msg, "fail");
    window.__DEMO01__ = {
      total: 0,
      passed: 0,
      failed: 0,
      mismatches: [],
      gridName: "",
      binding: BINDING_DESCRIPTION,
      error: msg,
    };
    return;
  }

  // -------------------------------------------------------------------------
  // 1. Load fixture + sidecar (CPU side).
  // -------------------------------------------------------------------------
  const [file, sidecar] = await Promise.all([
    NanoVDBFile.fromURL(FIXTURE_NVDB),
    fetch(FIXTURE_SIDECAR).then((r) => {
      if (!r.ok) throw new Error(`sidecar fetch failed: ${r.status}`);
      return r.json() as Promise<Sidecar>;
    }),
  ]);

  const gridImage = file.gridImage(0); // Uint32Array — the flat grid image
  const gridMeta = file.grids[0]!;
  const samples = sidecar.samples;
  const total = samples.length;
  log(
    `Grid "${gridMeta.name}" (${gridMeta.gridType}) — ${gridImage.length} u32 words ` +
      `(${(gridImage.byteLength / 1024 / 1024).toFixed(2)} MiB); ${total} probes.`,
  );

  // -------------------------------------------------------------------------
  // 2. Device-first bootstrap (D4): our adapter, our device, our limits.
  // -------------------------------------------------------------------------
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    const msg = "No WebGPU adapter (headless without SwiftShader?).";
    log(msg, "fail");
    window.__DEMO01__ = {
      total,
      passed: 0,
      failed: total,
      mismatches: [],
      gridName: gridMeta.name,
      binding: BINDING_DESCRIPTION,
      error: msg,
    };
    return;
  }

  const adapterMax = adapter.limits.maxStorageBufferBindingSize;
  // Raise the binding limit to cover this grid (rounded up to a power of two),
  // but never below the 128 MiB default, and never above what the adapter
  // offers. For big cinema grids this is where the ceiling gets lifted.
  const DEFAULT_LIMIT = 128 * 1024 * 1024;
  const needed = nextPow2(gridImage.byteLength);
  const requestedBinding = Math.min(adapterMax, Math.max(DEFAULT_LIMIT, needed));
  log(
    `Adapter maxStorageBufferBindingSize = ${(adapterMax / 1024 / 1024).toFixed(0)} MiB; ` +
      `requesting ${(requestedBinding / 1024 / 1024).toFixed(0)} MiB (grid needs ` +
      `${(needed / 1024 / 1024).toFixed(0)} MiB).`,
  );

  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: requestedBinding,
      maxBufferSize: requestedBinding,
    },
  });

  // r185 accepts an externally-created device: WebGPURenderer forwards
  // `parameters.device` to WebGPUBackend, which uses it verbatim instead of
  // calling requestAdapter/requestDevice itself (see WebGPUBackend.js ~L250).
  const renderer = new WebGPURenderer({ device, antialias: false });
  await renderer.init();
  log("WebGPURenderer running on our externally-created GPUDevice (D4).", "ok");

  // -------------------------------------------------------------------------
  // 3. Storage buffers.
  // -------------------------------------------------------------------------
  // Grid image: flat u32 words, read-only.
  const gridAttr = new StorageBufferAttribute(gridImage, 1);
  const gridStorage = storage(gridAttr, "uint", gridImage.length).toReadOnly();

  // Probe coords: flat i32 triples (x,y,z per thread), read-only.
  const coordData = new Int32Array(total * 3);
  for (let i = 0; i < total; i++) {
    const [x, y, z] = samples[i]!.ijk;
    coordData[i * 3 + 0] = x;
    coordData[i * 3 + 1] = y;
    coordData[i * 3 + 2] = z;
  }
  const coordAttr = new StorageBufferAttribute(coordData, 1);
  const coordStorage = storage(coordAttr, "int", total * 3).toReadOnly();

  // Outputs: one f32 value + one u32 active flag per thread (read_write).
  const outValueAttr = new StorageBufferAttribute(new Float32Array(total), 1);
  const outValueStorage = storage(outValueAttr, "float", total);
  const outActiveAttr = new StorageBufferAttribute(new Uint32Array(total), 1);
  const outActiveStorage = storage(outActiveAttr, "uint", total);

  // -------------------------------------------------------------------------
  // 4. THE SPIKE — bind the raw WGSL traversal via wgslFn with a storage
  //    pointer param, and drive it from a TSL compute kernel.
  // -------------------------------------------------------------------------
  const probeFloat = wgslFn(NVDB_PROBE_FLOAT_WGSL);

  const kernel = Fn(() => {
    const i = instanceIndex;
    const base = i.mul(3);
    const ijk = ivec3(
      coordStorage.element(base),
      coordStorage.element(base.add(1)),
      coordStorage.element(base.add(2)),
    );
    // `gridStorage` (a read-only StorageBufferNode) is passed to the
    // `ptr<storage, array<u32>, read>` parameter. TSL's FunctionCallNode
    // detects the pointer input and emits `&<buffer>.value`.
    // `wgslFn(...)` is typed as returning a bare `Node`; the WGSL actually
    // returns a vec2f, so re-type it as a vec2 node to reach `.x`/`.y`.
    const result = probeFloat({ grid: gridStorage, ijk }) as unknown as ReturnType<typeof vec2>;
    outValueStorage.element(i).assign(result.x);
    outActiveStorage.element(i).assign(uint(result.y));
  })().compute(total);

  const t0 = performance.now();
  await renderer.computeAsync(kernel);
  // Read back the two output buffers.
  const valueBytes = await renderer.getArrayBufferAsync(outValueAttr);
  const activeBytes = await renderer.getArrayBufferAsync(outActiveAttr);
  const computeMs = performance.now() - t0;
  const gotValues = new Float32Array(valueBytes);
  const gotActives = new Uint32Array(activeBytes);

  // -------------------------------------------------------------------------
  // 5. Compare against the sidecar ground truth.
  // -------------------------------------------------------------------------
  const mismatches: Mismatch[] = [];
  for (let i = 0; i < total; i++) {
    const s = samples[i]!;
    const gotValue = gotValues[i]!;
    const gotActive = gotActives[i] !== 0;
    const dv = Math.abs(gotValue - s.value);
    if (dv > VALUE_EPS || gotActive !== s.active) {
      mismatches.push({
        ijk: s.ijk,
        expectedValue: s.value,
        gotValue,
        expectedActive: s.active,
        gotActive,
      });
    }
  }
  const failed = mismatches.length;
  const passed = total - failed;

  window.__DEMO01__ = {
    total,
    passed,
    failed,
    mismatches: mismatches.slice(0, 20),
    gridName: gridMeta.name,
    binding: BINDING_DESCRIPTION,
    computeMs,
  };

  // -------------------------------------------------------------------------
  // 6. Render the pass/fail table.
  // -------------------------------------------------------------------------
  log(
    `${passed}/${total} probes matched native ground truth ` +
      `(|Δ| ≤ ${VALUE_EPS}); compute ${computeMs.toFixed(1)} ms.`,
    failed === 0 ? "ok" : "fail",
  );

  const table = document.createElement("table");
  table.innerHTML =
    "<thead><tr><th>#</th><th>ijk</th><th>expected</th><th>got</th>" +
    "<th>act(exp/got)</th><th></th></tr></thead>";
  const tbody = document.createElement("tbody");
  for (let i = 0; i < total; i++) {
    const s = samples[i]!;
    const gotValue = gotValues[i]!;
    const gotActive = gotActives[i] !== 0;
    const ok = Math.abs(gotValue - s.value) <= VALUE_EPS && gotActive === s.active;
    const tr = document.createElement("tr");
    tr.className = ok ? "row-ok" : "row-fail";
    tr.innerHTML =
      `<td>${i}</td><td>[${s.ijk.join(", ")}]</td>` +
      `<td>${s.value.toFixed(6)}</td><td>${gotValue.toFixed(6)}</td>` +
      `<td>${s.active ? "1" : "0"}/${gotActive ? "1" : "0"}</td>` +
      `<td>${ok ? "✓" : "✗"}</td>`;
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  root!.appendChild(table);

  log(`Binding pattern: ${BINDING_DESCRIPTION}`, "note");
}

run().catch((err: unknown) => {
  const message = err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err);
  log(`ERROR: ${message}`, "fail");
  // Publish a result even on failure so the e2e test reports meaningfully.
  const prev = window.__DEMO01__;
  window.__DEMO01__ = {
    total: prev?.total ?? 0,
    passed: prev?.passed ?? 0,
    failed: prev?.failed ?? prev?.total ?? 0,
    mismatches: prev?.mismatches ?? [],
    gridName: prev?.gridName ?? "",
    binding: BINDING_DESCRIPTION,
    error: message,
  };
});
