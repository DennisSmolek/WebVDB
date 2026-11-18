/**
 * K2 review gate: the `./three` adapter's guards on a real three r187 + GPU.
 *
 * 1. Self-test: the first `VDBGridNode` on a backend puts one 4-byte buffer
 *    through three's attribute path (cached per backend).
 * 2. A node used by a shader that another renderer builds (two renderers on
 *    one device) throws `VDB_BAD_OPTION` instead of silently reading 0; a
 *    node made for that renderer reads the grid.
 * 3. Two VDB modules in one shader throw `VDB_BAD_OPTION`; one node used
 *    twice in a shader is fine.
 * 4. `VDBVolumeMaterial` refuses a node made for another renderer.
 * 5. A three whose `Attributes.delete` bypasses `backend.destroyAttribute`
 *    (simulated on a third renderer) fails the self-test with
 *    `VDB_ENGINE_MISMATCH` before any pool buffer reaches it, and the pool's
 *    buffer survives that renderer's dispose(). If the constructor did not
 *    throw, the page reproduces the damage (shader use + dispose) and reports
 *    whether the pool survived.
 * 6. A renderer that made its own device warns once.
 *
 * Results go to `window.__VDB_THREE_GUARDS__` for e2e/three-guards.spec.ts.
 */
import { WebGPURenderer } from "three/webgpu";
import { Fn, float, instanceIndex, ivec3, storage, uint, vec3 } from "three/tsl";
import { StorageBufferAttribute } from "three/webgpu";
import { VDBGridPool } from "@pmndrs/vdb";
import { VDBGridNode, VDBVolumeMaterial, getDevice, vdbReadValue, vdbSample } from "@pmndrs/vdb/three";
import { NanoVDBFile, readValueF32 } from "@pmndrs/vdb/tools";

interface Thrown {
  threw: boolean;
  code?: string;
  message?: string;
}

interface Result {
  done: boolean;
  error?: string;
  missing?: string[];
  adapter?: string;
  /** Self-test buffers made after constructing nodes on A, A again, B. */
  selfTests?: number[];
  sameNodeTwice?: { ok: boolean; read: number; expected: number; error?: string };
  otherRenderer?: Thrown & { read?: number };
  ownNodeOnB?: { read: number; expected: number };
  twoModules?: Thrown;
  materialOtherRenderer?: Thrown;
  bypass?: Thrown & { poolIntact: boolean; validationError?: string };
  warnings?: string[];
  uncaptured: string[];
}

declare global {
  interface Window {
    __VDB_THREE_GUARDS__?: Result;
  }
}

const SELF_TEST_LABEL = "@pmndrs/vdb/three self-test";
const root = document.getElementById("demo-root")!;
function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
}

async function thrown(fn: () => unknown): Promise<Thrown> {
  try {
    await fn();
    return { threw: false };
  } catch (e) {
    return { threw: true, code: (e as { code?: string }).code ?? "not a VDBError", message: e instanceof Error ? e.message : String(e) };
  }
}

async function run(result: Result): Promise<Result> {
  if (!navigator.gpu) return { ...result, done: true, error: "WebGPU unavailable (navigator.gpu undefined)." };
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { ...result, done: true, error: "No WebGPU adapter." };
  result.adapter = [adapter.info.vendor, adapter.info.architecture, adapter.info.description].filter(Boolean).join(" / ");
  const device = await adapter.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(adapter) });
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));
  let selfTests = 0;
  const createBuffer = device.createBuffer.bind(device);
  device.createBuffer = (d) => {
    if (d.label === SELF_TEST_LABEL) selfTests++;
    return createBuffer(d);
  };
  const warnings: string[] = [];
  const warn = console.warn.bind(console);
  console.warn = (...args: unknown[]) => {
    warnings.push(args.map(String).join(" "));
    warn(...args);
  };

  const res = await fetch("/fixtures/primitives/box_fog_float.nvdb");
  if (!res.ok) return { ...result, done: true, missing: ["box_fog_float"] };
  const image = NanoVDBFile.fromArrayBuffer(await res.arrayBuffer()).gridImage(0);

  const rendererA = new WebGPURenderer({ device });
  await rendererA.init();
  const rendererB = new WebGPURenderer({ device });
  await rendererB.init();
  const pool = await VDBGridPool.create(getDevice(rendererA), { label: "three guards" });
  const id = pool.add(image);
  pool.flush();
  await pool.settled();
  const info = pool.info(id);
  const mid = [0, 1, 2].map((a) => Math.floor((info.indexBBox.min[a]! + info.indexBBox.max[a]!) / 2)) as [number, number, number];
  const expected = readValueF32(image, mid);

  const counts: number[] = [];
  const gridsA = new VDBGridNode(rendererA, pool, { types: ["float"], helpers: ["rayClipSupport"] });
  counts.push(selfTests);
  const gridsA2 = new VDBGridNode(rendererA, pool, { types: ["float"], name: "vdbPoolA2" });
  counts.push(selfTests);
  const gridsB = new VDBGridNode(rendererB, pool, { types: ["float"] });
  counts.push(selfTests);
  result.selfTests = counts;

  const out = new StorageBufferAttribute(new Float32Array(4), 1);
  const outNode = storage(out, "float", 4);
  const p = vec3(...mid);
  const read = async (renderer: WebGPURenderer, kernel: unknown): Promise<number> => {
    await renderer.computeAsync(kernel as never);
    return new Float32Array(await renderer.getArrayBufferAsync(out))[0]!;
  };

  // One node used twice in one shader: fine.
  {
    const kernel = Fn(() => {
      outNode.element(instanceIndex).assign(vdbReadValue(gridsA, id, ivec3(p)).add(vdbSample(gridsA, uint(id), p, { space: "index" }).mul(float(0))));
    })().compute(1);
    try {
      const v = await read(rendererA, kernel);
      gridsA.update();
      result.sameNodeTwice = { ok: v === expected, read: v, expected };
    } catch (e) {
      result.sameNodeTwice = { ok: false, read: NaN, expected, error: String(e) };
    }
  }

  // gridsA in a shader that renderer B builds.
  {
    const kernel = Fn(() => {
      outNode.element(instanceIndex).assign(vdbReadValue(gridsA, id, ivec3(p)));
    })().compute(1);
    let value: number | undefined;
    const t = await thrown(async () => {
      value = await read(rendererB, kernel);
    });
    result.otherRenderer = { ...t, ...(value !== undefined ? { read: value } : {}) };
  }

  // The advice: a node made for B reads the grid on B.
  {
    const kernel = Fn(() => {
      outNode.element(instanceIndex).assign(vdbReadValue(gridsB, id, ivec3(p)));
    })().compute(1);
    result.ownNodeOnB = { read: await read(rendererB, kernel), expected };
    gridsB.update();
  }

  // Two modules (two nodes) in one shader.
  {
    const kernel = Fn(() => {
      outNode.element(instanceIndex).assign(vdbReadValue(gridsA, id, ivec3(p)).add(vdbReadValue(gridsA2, id, ivec3(p))));
    })().compute(1);
    result.twoModules = await thrown(() => read(rendererA, kernel));
  }

  result.materialOtherRenderer = await thrown(() => new VDBVolumeMaterial(rendererB, gridsA, id));

  // A three that destroys attribute buffers behind backend.destroyAttribute.
  {
    const rendererC = new WebGPURenderer({ device });
    await rendererC.init();
    const backendC = (rendererC as unknown as { backend: { attributeUtils: { destroyAttribute(a: object): void } } }).backend;
    type DataMapLike = { delete(a: object): unknown; has(a: object): boolean };
    const attrs = (rendererC as unknown as { _attributes: DataMapLike })._attributes;
    const dataMapDelete = (Object.getPrototypeOf(Object.getPrototypeOf(attrs)) as DataMapLike).delete;
    attrs.delete = (attribute: object) => {
      if (!attrs.has(attribute)) return null;
      backendC.attributeUtils.destroyAttribute(attribute); // buffer.destroy() + forget, no guard
      return dataMapDelete.call(attrs, attribute); // three's own bookkeeping (DataMap)
    };
    device.pushErrorScope("validation");
    const t = await thrown(async () => {
      const gridsC = new VDBGridNode(rendererC, pool, { types: ["float"] });
      // Reached only without the self-test: reproduce the damage.
      const kernel = Fn(() => {
        outNode.element(instanceIndex).assign(vdbReadValue(gridsC, id, ivec3(p)));
      })().compute(1);
      await read(rendererC, kernel);
      gridsC.dispose();
    });
    await rendererC.dispose();
    // The pool's buffer must still work on A.
    pool.flush();
    const kernel = Fn(() => {
      outNode.element(instanceIndex).assign(vdbReadValue(gridsA, id, ivec3(p)));
    })().compute(1);
    let v = NaN;
    try {
      v = await read(rendererA, kernel);
    } catch {
      /* reported through poolIntact */
    }
    const validation = await device.popErrorScope();
    result.bypass = { ...t, poolIntact: v === expected && validation === null, ...(validation ? { validationError: validation.message } : {}) };
  }

  // A renderer that owns its device warns once.
  {
    const rendererD = new WebGPURenderer();
    await rendererD.init();
    const poolD = await VDBGridPool.create(getDevice(rendererD), { label: "own device" });
    poolD.add(image);
    poolD.flush();
    const d1 = new VDBGridNode(rendererD, poolD, { types: ["float"] });
    const d2 = new VDBGridNode(rendererD, poolD, { types: ["float"] });
    d1.dispose();
    d2.dispose();
    poolD.dispose();
    await rendererD.dispose();
  }
  result.warnings = warnings;

  gridsA.dispose();
  gridsA2.dispose();
  gridsB.dispose();
  await rendererA.dispose();
  await rendererB.dispose();
  pool.dispose();
  device.destroy();
  return { ...result, done: true };
}

root.textContent = "";
const initial: Result = { done: false, uncaptured: [] };
run(initial)
  .catch((e: unknown): Result => ({ ...initial, done: true, error: e instanceof Error ? `${e.name}: ${e.message}\n${e.stack ?? ""}` : String(e) }))
  .then((r) => {
    window.__VDB_THREE_GUARDS__ = r;
    log(JSON.stringify(r, null, 2), r.error ? "fail" : "ok");
  });
