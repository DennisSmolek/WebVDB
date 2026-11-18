/**
 * Demo 01 — hello-nvdb: read voxels of a NanoVDB grid in a three.js TSL
 * compute pass, through the K2 adapter (`VDBGridPool` + `VDBGridNode`).
 *
 * v1 proved this with a hand-written WGSL traversal bound as a
 * `ptr<storage, array<u32>, read>` parameter (the Phase 1 spike). Since K2.3
 * the grid lives in a pool and the generated `vdb*` module reads it:
 *
 *   1. The host makes the device with `VDBGridPool.requiredLimits(adapter)`
 *      and hands it to `WebGPURenderer` (so the pool outlives the renderer).
 *   2. `sphere_fog_float.nvdb` -> `pool.add(image)`; `pool.flush()` uploads it.
 *   3. `new VDBGridNode(renderer, pool, { profile: 'read', isActive: true })`.
 *      A TSL compute kernel of 73 threads reads each probe's value with the
 *      `vdbReadValue` TSL helper, and its active flag with a one-line `wgslFn`
 *      that includes `grids.module` and calls `vdbIsActive` (the WGSL route).
 *   4. The outputs are read back and compared with the native sidecar
 *      (|Δ| ≤ 1e-5, active flag exact); `window.__DEMO01__` feeds the e2e gate.
 */
import { VDBGridPool } from "@pmndrs/vdb";
import { NvdbFile } from "@pmndrs/vdb/tools";
import { VDBGridNode, getDevice, vdbReadValue } from "@pmndrs/vdb/three";
import { WebGPURenderer, StorageBufferAttribute } from "three/webgpu";
import { Fn, instanceIndex, ivec3, storage, uint, wgslFn } from "three/tsl";

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

// How the grid reaches the shader, echoed into window.__DEMO01__.binding.
const BINDING_DESCRIPTION =
  "VDBGridNode(renderer, pool): the pool's GPUBuffer bound by three as a read-only storage node; " +
  "values via the vdbReadValue TSL helper, active flags via wgslFn(..., [grids.module]) calling vdbIsActive.";

const root = document.querySelector<HTMLDivElement>("#demo-root");
if (!root) throw new Error("missing #demo-root element");

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  if (cls) p.className = cls;
  p.textContent = text;
  root!.appendChild(p);
}

function publishError(message: string, total = 0, gridName = ""): void {
  log(message, "fail");
  window.__DEMO01__ = { total, passed: 0, failed: total, mismatches: [], gridName, binding: BINDING_DESCRIPTION, error: message };
}

async function run(): Promise<void> {
  root!.textContent = "";
  log("Demo 01 — hello-nvdb (read voxels in TSL through VDBGridNode)", "h");

  if (!navigator.gpu) {
    publishError("WebGPU unavailable (navigator.gpu undefined) — needs a secure context + adapter.");
    return;
  }

  // -------------------------------------------------------------------------
  // 1. Load fixture + sidecar (CPU side).
  // -------------------------------------------------------------------------
  const [file, sidecar] = await Promise.all([
    NvdbFile.fromURL(FIXTURE_NVDB),
    fetch(FIXTURE_SIDECAR).then((r) => {
      if (!r.ok) throw new Error(`sidecar fetch failed: ${r.status}`);
      return r.json() as Promise<Sidecar>;
    }),
  ]);

  const gridImage = file.gridImage(0);
  const gridMeta = file.grids[0]!;
  const samples = sidecar.samples;
  const total = samples.length;
  log(
    `Grid "${gridMeta.name}" (${gridMeta.gridType}) — ${gridImage.length} u32 words ` +
      `(${(gridImage.byteLength / 1024 / 1024).toFixed(2)} MiB); ${total} probes.`,
  );

  // -------------------------------------------------------------------------
  // 2. The host owns the device (D8): the pool's limits, then the renderer on it.
  // -------------------------------------------------------------------------
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    publishError("No WebGPU adapter (headless without SwiftShader?).", total, gridMeta.name);
    return;
  }
  const device = await adapter.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(adapter) });
  const renderer = new WebGPURenderer({ device, antialias: false });
  await renderer.init();

  // -------------------------------------------------------------------------
  // 3. Pool + node. One module per shader: the kernel below uses one VDBGridNode.
  // -------------------------------------------------------------------------
  const pool = await VDBGridPool.create(getDevice(renderer), { label: "demo 01 pool" });
  const gridId = pool.add(gridImage, { label: gridMeta.name });
  pool.flush();
  await pool.settled();
  const grids = new VDBGridNode(renderer, pool, { profile: "read", isActive: true });
  log(`Pool: ${(pool.byteLength / 1024 / 1024).toFixed(2)} MiB, grid id ${gridId}, module types [${grids.options.types.join(", ")}].`, "ok");

  // Probe coords: flat i32 triples, read-only.
  const coordData = new Int32Array(total * 3);
  for (let i = 0; i < total; i++) coordData.set(samples[i]!.ijk, i * 3);
  const coordStorage = storage(new StorageBufferAttribute(coordData, 1), "int", total * 3).toReadOnly();

  // Outputs: one f32 value + one u32 active flag per thread.
  const outValueAttr = new StorageBufferAttribute(new Float32Array(total), 1);
  const outValueStorage = storage(outValueAttr, "float", total);
  const outActiveAttr = new StorageBufferAttribute(new Uint32Array(total), 1);
  const outActiveStorage = storage(outActiveAttr, "uint", total);

  // The WGSL route: a function over the generated module (vdbIsActive needs isActive: true).
  const isActive = wgslFn(
    /* wgsl */ `fn demo01IsActive(g: u32, ijk: vec3i) -> u32 {
    var a = vdbAccessorInit(g);
    return select(0u, 1u, vdbIsActive(&a, ijk));
}`,
    [grids.module] as never,
  );

  const kernel = Fn(() => {
    const i = instanceIndex;
    const base = i.mul(3);
    const ijk = ivec3(coordStorage.element(base), coordStorage.element(base.add(1)), coordStorage.element(base.add(2)));
    outValueStorage.element(i).assign(vdbReadValue(grids, gridId, ijk));
    outActiveStorage.element(i).assign(isActive({ g: uint(gridId), ijk }));
  })().compute(total);

  // Per frame (here once): after the pool's update was submitted, rebind if it grew.
  grids.update();
  const t0 = performance.now();
  await renderer.computeAsync(kernel);
  const gotValues = new Float32Array(await renderer.getArrayBufferAsync(outValueAttr));
  const gotActives = new Uint32Array(await renderer.getArrayBufferAsync(outActiveAttr));
  const computeMs = performance.now() - t0;

  // -------------------------------------------------------------------------
  // 4. Compare against the sidecar ground truth.
  // -------------------------------------------------------------------------
  const mismatches: Mismatch[] = [];
  for (let i = 0; i < total; i++) {
    const s = samples[i]!;
    const gotValue = gotValues[i]!;
    const gotActive = gotActives[i] !== 0;
    if (Math.abs(gotValue - s.value) > VALUE_EPS || gotActive !== s.active) {
      mismatches.push({ ijk: s.ijk, expectedValue: s.value, gotValue, expectedActive: s.active, gotActive });
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
  // 5. Render the pass/fail table.
  // -------------------------------------------------------------------------
  log(
    `${passed}/${total} probes matched native ground truth ` + `(|Δ| ≤ ${VALUE_EPS}); compute ${computeMs.toFixed(1)} ms.`,
    failed === 0 ? "ok" : "fail",
  );

  const table = document.createElement("table");
  table.innerHTML =
    "<thead><tr><th>#</th><th>ijk</th><th>expected</th><th>got</th>" + "<th>act(exp/got)</th><th></th></tr></thead>";
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

  log(`Binding: ${BINDING_DESCRIPTION}`, "note");
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
