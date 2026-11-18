/**
 * K2.2 browser gate: `VDBGridNode` + `vdbReadValue` / `vdbSample` in a three
 * r187 TSL compute shader, against the `tools` CPU twins.
 *
 * 1. A host-made device (so `renderer.dispose()` can't destroy it) shared by a
 *    `WebGPURenderer` and a `VDBGridPool` (`getDevice(renderer)`).
 * 2. Four grids: box Float, sphere Fp8, torus FpN, and sphere Float again as an
 *    instance with a rotation + scale + translation transform. One compute
 *    node reads every grid by a per-probe id: `vdbReadValue` (index coords),
 *    `vdbSample` in index space and in world space. Phase-1 slots for grids not
 *    added yet read the dead id 99 (expect 0).
 * 3. Two more grids force pool growth (`version` bump); `grids.update()`
 *    rebinds; the same compute node (no new shader module) reads all six.
 * 4. `renderer.dispose()`; the pool's buffer must survive: raw WebGPU readback
 *    of every image range must be byte-identical, and the pool keeps working.
 *
 * Results go to `window.__VDB_THREE_GRID_NODE__` for e2e/three-grid-node.spec.ts.
 */
import { Matrix4, Quaternion, Vector3 } from "three";
import { StorageBufferAttribute, WebGPURenderer } from "three/webgpu";
import { Fn, If, float, instanceIndex, ivec3, storage, uint } from "three/tsl";
import { VDBGridPool } from "@pmndrs/vdb";
import { VDBGridNode, getDevice, vdbReadValue, vdbSample } from "@pmndrs/vdb/three";
import { NvdbFile, probeCoords, probePoints, vdbReadValueCpu, vdbReadValueF32, vdbSampleTrilinear } from "../../../packages/vdb/src/tools/internal.js";

type Mode = 0 | 1 | 2; // read, sample index, sample world
const MODE_NAMES = ["read", "sampleIndex", "sampleWorld"] as const;
const PER_MODE = 256;
const DEAD_ID = 99;
const READ_EPS = 1e-5;
const SAMPLE_EPS = 1e-3;

interface ModeStats {
  grid: string;
  mode: (typeof MODE_NAMES)[number];
  count: number;
  maxAbs: number;
  exact: number;
  /** Bit-exact against the f32 twin (read: `vdbReadValueF32`; sampleIndex: f32 taps + f32 lerps, see `f32Twin`); -1 = no f32 twin. */
  exactF32: number;
  bad: number;
  firstBad?: { i: number; gpu: number; cpu: number; at: number[] };
}

interface PhaseResult {
  version: number;
  rebound?: boolean;
  stats: ModeStats[];
  dead: { count: number; bad: number };
  validationError?: string;
}

interface Result {
  done: boolean;
  error?: string;
  missing?: string[];
  sameDevice?: boolean;
  /** `adapter.info` vendor / architecture / description. */
  adapter?: string;
  phase1?: PhaseResult;
  phase2?: PhaseResult;
  shaderModulesPhase1?: number;
  shaderModulesPhase2Delta?: number;
  computePipelinesPhase2Delta?: number;
  bindingVerified?: boolean;
  /** The storage node's property references three handed the module (expected: `vdbPool.value`). */
  bufferRefs?: string[];
  retiredStandIns?: number;
  afterDispose?: { identical: boolean[]; poolUsable: boolean; validationError?: string };
  uncaptured: string[];
}

declare global {
  interface Window {
    __VDB_THREE_GRID_NODE__?: Result;
  }
}

const FIXTURES = ["box_fog_float", "sphere_fog_fp8", "torus_fog_fpn", "sphere_fog_float", "torus_fog_float", "box_fog_fpn"] as const;
const root = document.getElementById("demo-root")!;

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
}

type V3 = [number, number, number];
const fround3 = (p: readonly number[]): V3 => [Math.fround(p[0]!), Math.fround(p[1]!), Math.fround(p[2]!)];
/** Applies a 3x4 row-major affine (f64). */
const affine = (m: Float64Array, p: readonly number[]): V3 => [
  m[0]! * p[0]! + m[1]! * p[1]! + m[2]! * p[2]! + m[3]!,
  m[4]! * p[0]! + m[5]! * p[1]! + m[6]! * p[2]! + m[7]!,
  m[8]! * p[0]! + m[9]! * p[1]! + m[10]! * p[2]! + m[11]!,
];

interface Probe {
  grid: number; // index into `slots` (-1 = dead)
  id: number;
  mode: Mode;
  at: V3;
  expected: number;
  /** f32 twin, when there is one (read, sampleIndex). */
  f32?: number;
}

/**
 * f32 twin of the vendored trilinear (`sample_trilinear_typed`): taps through
 * `vdbReadValueF32`, `mix(a, b, t)` as `a + t * (b - a)`, each op rounded to f32.
 * WGSL leaves mix's formula and FMA contraction to the implementation, so this
 * is diagnostic: exact on SwiftShader is the expectation, not the gate.
 */
function f32Twin(image: Uint32Array, xyz: V3): number {
  const f = Math.fround;
  const b = xyz.map(Math.floor) as V3;
  const u = xyz.map((v, k) => f(v - b[k]!)) as V3;
  const c = (dx: number, dy: number, dz: number) => vdbReadValueF32(image, [b[0] + dx, b[1] + dy, b[2] + dz]);
  const mix = (a: number, bb: number, t: number) => f(a + f(t * f(bb - a)));
  const x00 = mix(c(0, 0, 0), c(1, 0, 0), u[0]);
  const x10 = mix(c(0, 1, 0), c(1, 1, 0), u[0]);
  const x01 = mix(c(0, 0, 1), c(1, 0, 1), u[0]);
  const x11 = mix(c(0, 1, 1), c(1, 1, 1), u[0]);
  return mix(mix(x00, x10, u[1]), mix(x01, x11, u[1]), u[2]);
}

/** 3 x PER_MODE probes for one grid: integer coords, index points, world points. */
function probesFor(pool: VDBGridPool, slot: number, id: number, image: Uint32Array, seed: bigint): Probe[] {
  const info = pool.info(id);
  const bbox = { bboxMin: [...info.indexBBox.min] as V3, bboxMax: [...info.indexBBox.max] as V3 };
  const out: Probe[] = [];
  for (const ijk of probeCoords({ seed, count: PER_MODE, ...bbox })) {
    out.push({ grid: slot, id, mode: 0, at: ijk, expected: Math.fround(vdbReadValueCpu(image, ijk).value), f32: vdbReadValueF32(image, ijk) });
  }
  for (const p of probePoints({ seed: seed + 1n, count: PER_MODE, ...bbox })) {
    const at = fround3(p);
    out.push({ grid: slot, id, mode: 1, at, expected: Math.fround(vdbSampleTrilinear(image, at)), f32: f32Twin(image, at) });
  }
  for (const p of probePoints({ seed: seed + 2n, count: PER_MODE, ...bbox })) {
    const at = fround3(affine(info.indexToWorld, p));
    out.push({ grid: slot, id, mode: 2, at, expected: Math.fround(vdbSampleTrilinear(image, affine(info.worldToIndex, at))) });
  }
  return out;
}

function deadProbes(count: number): Probe[] {
  return Array.from({ length: count }, (_, k) => ({ grid: -1, id: DEAD_ID, mode: (k % 3) as Mode, at: [k, k, k] as V3, expected: 0 }));
}

function score(probes: Probe[], gpu: Float32Array, names: string[]): Pick<PhaseResult, "stats" | "dead"> {
  const stats = new Map<string, ModeStats>();
  const dead = { count: 0, bad: 0 };
  probes.forEach((p, i) => {
    const got = gpu[i]!;
    if (p.grid < 0) {
      dead.count++;
      if (got !== 0) dead.bad++;
      return;
    }
    const key = `${p.grid}:${p.mode}`;
    let s = stats.get(key);
    if (!s) stats.set(key, (s = { grid: names[p.grid]!, mode: MODE_NAMES[p.mode], count: 0, maxAbs: 0, exact: 0, exactF32: p.f32 === undefined ? -1 : 0, bad: 0 }));
    const d = Math.abs(got - p.expected);
    s.count++;
    if (Number.isNaN(d) || d > (p.mode === 0 ? READ_EPS : SAMPLE_EPS)) {
      s.bad++;
      s.firstBad ??= { i, gpu: got, cpu: p.expected, at: p.at };
    }
    if (got === p.expected) s.exact++;
    if (p.f32 !== undefined && got === p.f32) s.exactF32++;
    s.maxAbs = Math.max(s.maxAbs, Number.isNaN(d) ? Infinity : d);
  });
  return { stats: [...stats.values()], dead };
}

async function run(result: Result): Promise<Result> {
  if (!navigator.gpu) return { ...result, done: true, error: "WebGPU unavailable (navigator.gpu undefined)." };
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { ...result, done: true, error: "No WebGPU adapter." };
  result.adapter = [adapter.info.vendor, adapter.info.architecture, adapter.info.description].filter(Boolean).join(" / ");
  const device = await adapter.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(adapter) });
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));

  // Count shader modules / pipelines / retired stand-ins three and the adapter make.
  const counts = { shaderModules: 0, computePipelines: 0, retiredStandIns: 0 };
  const createShaderModule = device.createShaderModule.bind(device);
  device.createShaderModule = (d) => (counts.shaderModules++, createShaderModule(d));
  const createComputePipeline = device.createComputePipeline.bind(device);
  device.createComputePipeline = (d) => (counts.computePipelines++, createComputePipeline(d));
  const createComputePipelineAsync = device.createComputePipelineAsync.bind(device);
  device.createComputePipelineAsync = (d) => (counts.computePipelines++, createComputePipelineAsync(d));
  const createBuffer = device.createBuffer.bind(device);
  device.createBuffer = (d) => {
    if (d.label === "@pmndrs/vdb/three retired stand-in") counts.retiredStandIns++;
    return createBuffer(d);
  };

  const files = await Promise.all(
    FIXTURES.map(async (name) => {
      const res = await fetch(`/fixtures/primitives/${name}.nvdb`);
      return res.ok ? NvdbFile.fromArrayBuffer(await res.arrayBuffer()).gridImage(0) : null;
    }),
  );
  const missing = FIXTURES.filter((_, k) => files[k] === null);
  if (missing.length > 0) return { ...result, done: true, missing };
  const images = files as Uint32Array[];

  const renderer = new WebGPURenderer({ device });
  await renderer.init();
  result.sameDevice = getDevice(renderer) === device;

  // ---- Phase 1: four grids, one with an instance transform.
  device.pushErrorScope("validation");
  const pool = await VDBGridPool.create(getDevice(renderer), { initialBytes: 4096, label: "three grid node" });
  const instance = new Matrix4().compose(
    new Vector3(10, -5, 3),
    new Quaternion().setFromAxisAngle(new Vector3(1, 1, 0).normalize(), Math.PI / 6),
    new Vector3(0.5, 0.5, 0.5),
  );
  const names = ["box_fog_float", "sphere_fog_fp8", "torus_fog_fpn", "sphere_fog_float (instance)", "torus_fog_float", "box_fog_fpn"];
  const ids: number[] = [];
  for (let k = 0; k < 4; k++) ids.push(pool.add(images[k]!, k === 3 ? { transform: instance.elements, label: names[k]! } : { label: names[k]! }));
  pool.flush();
  await pool.settled();

  const grids = new VDBGridNode(renderer, pool, { types: ["float", "fp8", "fpn"], profile: "sample" });

  const phase1Probes = [0, 1, 2, 3].flatMap((k) => probesFor(pool, k, ids[k]!, images[k]!, BigInt(1000 + k * 10)));
  const capacity = Math.ceil((phase1Probes.length + 2 * 3 * PER_MODE + 64) / 64) * 64;
  const pad = (probes: Probe[]) => [...probes, ...deadProbes(capacity - probes.length)];
  let probes = pad(phase1Probes);

  const posAttr = new StorageBufferAttribute(new Float32Array(capacity * 4), 4);
  const idAttr = new StorageBufferAttribute(new Uint32Array(capacity), 1);
  const modeAttr = new StorageBufferAttribute(new Uint32Array(capacity), 1);
  const outAttr = new StorageBufferAttribute(new Float32Array(capacity), 1);
  const upload = (list: Probe[]) => {
    const pos = posAttr.array as Float32Array;
    const idArr = idAttr.array as Uint32Array;
    const modeArr = modeAttr.array as Uint32Array;
    list.forEach((p, i) => {
      pos.set([...p.at, 1], i * 4);
      idArr[i] = p.id;
      modeArr[i] = p.mode;
    });
    posAttr.needsUpdate = true;
    idAttr.needsUpdate = true;
    modeAttr.needsUpdate = true;
  };
  upload(probes);

  const pos = storage(posAttr, "vec4", capacity).toReadOnly();
  const gid = storage(idAttr, "uint", capacity).toReadOnly();
  const mode = storage(modeAttr, "uint", capacity).toReadOnly();
  const out = storage(outAttr, "float", capacity);
  const kernel = Fn(() => {
    const i = instanceIndex;
    const g = gid.element(i);
    const p = pos.element(i).xyz;
    const m = mode.element(i);
    const r = float(0).toVar();
    If(m.equal(uint(0)), () => {
      r.assign(vdbReadValue(grids, g, ivec3(p)));
    })
      .ElseIf(m.equal(uint(1)), () => {
        r.assign(vdbSample(grids, g, p, { space: "index" }));
      })
      .Else(() => {
        r.assign(vdbSample(grids, g, p));
      });
    out.element(i).assign(r);
  })().compute(capacity);

  const dispatch = async (): Promise<Float32Array> => {
    await renderer.computeAsync(kernel);
    return new Float32Array(await renderer.getArrayBufferAsync(outAttr));
  };

  const gpu1 = await dispatch();
  result.shaderModulesPhase1 = counts.shaderModules;
  // update() verifies three bound the pool's buffer to the stand-in it used (throws otherwise).
  const reboundEarly = grids.update();
  result.bindingVerified = !reboundEarly;
  result.bufferRefs = [...(grids.module as unknown as { _byRef: Map<string, string> })._byRef.keys()];
  const v1 = await device.popErrorScope();
  result.phase1 = { version: pool.version, ...score(probes, gpu1, names), ...(v1 ? { validationError: v1.message } : {}) };

  // ---- Phase 2: two more grids force a growth; rebind; same shader.
  device.pushErrorScope("validation");
  const versionBefore = pool.version;
  for (let k = 4; k < 6; k++) ids.push(pool.add(images[k]!, { label: names[k]! }));
  const encoder = device.createCommandEncoder();
  pool.update(encoder);
  device.queue.submit([encoder.finish()]);
  await pool.settled();
  if (pool.version === versionBefore) throw new Error(`expected a growth (version ${versionBefore} unchanged, ${pool.byteLength} B)`);
  const rebound = grids.update();

  probes = pad([...phase1Probes, ...[4, 5].flatMap((k) => probesFor(pool, k, ids[k]!, images[k]!, BigInt(1000 + k * 10)))]);
  upload(probes);
  const modulesBefore = counts.shaderModules;
  const pipelinesBefore = counts.computePipelines;
  const gpu2 = await dispatch();
  result.shaderModulesPhase2Delta = counts.shaderModules - modulesBefore;
  result.computePipelinesPhase2Delta = counts.computePipelines - pipelinesBefore;
  grids.update(); // verifies the new stand-in's binding
  // Two more updates retire (and later destroy) the pre-growth buffer; reads must not care.
  pool.flush();
  pool.flush();
  const gpu3 = await dispatch();
  const v2 = await device.popErrorScope();
  const s2 = score(probes, gpu2, names);
  const s3 = score(probes, gpu3, names);
  result.phase2 = {
    version: pool.version,
    rebound,
    stats: s2.stats.map((s, k) => ({ ...s, bad: s.bad + s3.stats[k]!.bad })),
    dead: { count: s2.dead.count, bad: s2.dead.bad + s3.dead.bad },
    ...(v2 ? { validationError: v2.message } : {}),
  };

  // ---- Phase 3: renderer.dispose() must leave the pool's buffer alone.
  await renderer.dispose();
  result.retiredStandIns = counts.retiredStandIns;
  device.pushErrorScope("validation");
  const readBack = async (): Promise<Uint8Array> => {
    const size = pool.byteLength;
    const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(pool.buffer, 0, staging, 0, size);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return bytes;
  };
  const same = (bytes: Uint8Array, k: number): boolean => {
    const info = pool.info(ids[k]!);
    const src = new Uint8Array(images[k]!.buffer, images[k]!.byteOffset, images[k]!.byteLength);
    const got = bytes.subarray(info.imageOffset!, info.imageOffset! + info.imageBytes);
    return got.length === src.length && got.every((b, i) => b === src[i]);
  };
  const bytes = await readBack();
  const identical = ids.map((_, k) => same(bytes, k));
  // Still usable: stage a replace + a remove and read back again.
  pool.remove(ids[5]!);
  pool.replace(ids[4]!, images[4]!);
  pool.flush();
  await pool.settled();
  const again = await readBack();
  const poolUsable = same(again, 0) && same(again, 4) && pool.lastError === null;
  const v3 = await device.popErrorScope();
  result.afterDispose = { identical, poolUsable, ...(v3 ? { validationError: v3.message } : {}) };

  grids.dispose();
  pool.dispose();
  device.destroy();
  return { ...result, done: true };
}

root.textContent = "";
const initial: Result = { done: false, uncaptured: [] };
run(initial)
  .catch((e: unknown): Result => ({ ...initial, done: true, error: e instanceof Error ? `${e.name}: ${e.message}\n${e.stack ?? ""}` : String(e) }))
  .then((r) => {
    window.__VDB_THREE_GRID_NODE__ = r;
    if (r.error || r.missing) {
      log(r.error ?? `missing fixtures: ${r.missing!.join(", ")}`, "fail");
      return;
    }
    for (const phase of [r.phase1, r.phase2]) {
      if (!phase) continue;
      for (const s of phase.stats) {
        log(`v${phase.version} ${s.grid} ${s.mode}: ${s.count} probes, ${s.exact} exact (${s.exactF32} vs f32 twin), max |Δ| ${s.maxAbs.toExponential(2)}, ${s.bad} bad`, s.bad ? "fail" : "");
      }
      log(`v${phase.version} dead id: ${phase.dead.count} probes, ${phase.dead.bad} bad`);
    }
    log(JSON.stringify({ ...r, phase1: undefined, phase2: undefined }), "ok");
  });
