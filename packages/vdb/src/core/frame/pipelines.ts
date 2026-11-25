/**
 * The per-device pipeline cache of the core frame renderer (K6.1, design §4.1, §5.2): pipelines, layouts
 * and modules live as long as their device (a WeakMap keyed by the device), found by a key of the WGSL
 * text's hash, the bind-group layouts and the target formats. Every creation runs inside
 * `pushErrorScope('validation')`; `prepare()` uses the async creators. Bind-group layouts are positional
 * and must have no gaps (Traps): callers pass one descriptor per group, 0..n−1.
 */

import { VDBError } from "../errors.js";

/** FNV-1a (two 32-bit lanes) of a string: the WGSL text's key. */
export function vdbHashText(text: string): string {
  let a = 0x811c9dc5;
  let b = 0x01000193 ^ text.length;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    a = Math.imul(a ^ c, 0x01000193);
    b = Math.imul(b ^ c, 0x5bd1e995) ^ (b >>> 13);
  }
  return `${(a >>> 0).toString(16).padStart(8, "0")}${(b >>> 0).toString(16).padStart(8, "0")}`;
}

/** A compute kernel to build. */
export interface VDBComputeSpec {
  readonly label: string;
  readonly code: string;
  readonly entryPoint: string;
  /** One layout per group, 0..n−1 (no gaps). */
  readonly groups: readonly GPUBindGroupLayoutDescriptor[];
}

/** A render pipeline to build (the composite). */
export interface VDBRenderSpec {
  readonly label: string;
  readonly code: string;
  readonly vertexEntryPoint: string;
  readonly fragmentEntryPoint: string;
  readonly groups: readonly GPUBindGroupLayoutDescriptor[];
  readonly targets: readonly GPUColorTargetState[];
  /** K6.6 (level sets, points): optional depth state, primitive state and vertex buffers (default: none, triangle-list). */
  readonly depthStencil?: GPUDepthStencilState;
  readonly primitive?: GPUPrimitiveState;
  readonly buffers?: readonly GPUVertexBufferLayout[];
}

/** The pipeline-state part of a render spec's key and descriptor (K6.6); empty for the composite. */
const renderState = (s: VDBRenderSpec) => (s.depthStencil || s.primitive || s.buffers ? `:${vdbHashText(JSON.stringify([s.depthStencil ?? null, s.primitive ?? null, s.buffers ?? null]))}` : "");

function renderDescriptor(spec: VDBRenderSpec, module: GPUShaderModule, layout: GPUPipelineLayout): GPURenderPipelineDescriptor {
  return {
    label: spec.label,
    layout,
    vertex: { module, entryPoint: spec.vertexEntryPoint, ...(spec.buffers ? { buffers: [...spec.buffers] } : {}) },
    fragment: { module, entryPoint: spec.fragmentEntryPoint, targets: [...spec.targets] },
    primitive: spec.primitive ?? { topology: "triangle-list" },
    ...(spec.depthStencil ? { depthStencil: spec.depthStencil } : {}),
  };
}

/** A built pipeline with its group layouts and the validation result of its creation. */
export interface VDBCachedPipeline<P> {
  readonly pipeline: P;
  readonly layouts: readonly GPUBindGroupLayout[];
  /** Resolves null when valid, else the first validation error (also logged once). */
  readonly error: Promise<GPUError | null>;
}

type AnyPipeline = VDBCachedPipeline<GPUComputePipeline> | VDBCachedPipeline<GPURenderPipeline>;
const caches = new WeakMap<GPUDevice, Map<string, AnyPipeline | Promise<AnyPipeline>>>();

const layoutKey = (groups: readonly GPUBindGroupLayoutDescriptor[]) =>
  JSON.stringify(groups.map((g) => [...g.entries].sort((x, y) => x.binding - y.binding)));

/** The cache key of a compute spec. */
export const computeKey = (s: VDBComputeSpec) => `c:${s.entryPoint}:${vdbHashText(s.code)}:${vdbHashText(layoutKey(s.groups))}`;
/** The cache key of a render spec. */
export const renderKey = (s: VDBRenderSpec) =>
  `r:${s.vertexEntryPoint}:${s.fragmentEntryPoint}:${vdbHashText(s.code)}:${vdbHashText(layoutKey(s.groups))}:${vdbHashText(JSON.stringify(s.targets))}${renderState(s)}`;

function cacheOf(device: GPUDevice): Map<string, AnyPipeline | Promise<AnyPipeline>> {
  let c = caches.get(device);
  if (!c) caches.set(device, (c = new Map()));
  return c;
}

function checkGroups(label: string, groups: readonly GPUBindGroupLayoutDescriptor[]): void {
  if (groups.length > 4) throw new VDBError("VDB_BAD_OPTION", `${label}: ${groups.length} bind groups exceed maxBindGroups (4).`);
}

function logged(label: string, error: Promise<GPUError | null>): Promise<GPUError | null> {
  return error.then((e) => {
    if (e) console.error(`@pmndrs/vdb frame renderer: ${label}: ${e.message}`);
    return e;
  });
}

function layoutsOf(device: GPUDevice, label: string, groups: readonly GPUBindGroupLayoutDescriptor[]): GPUBindGroupLayout[] {
  return groups.map((g, i) => device.createBindGroupLayout({ ...g, label: `${label} group ${i}` }));
}

/** The compute pipeline of `spec` on `device`, built once (synchronously) per device and key. */
export function computePipeline(device: GPUDevice, spec: VDBComputeSpec): VDBCachedPipeline<GPUComputePipeline> {
  const cache = cacheOf(device);
  const key = computeKey(spec);
  const hit = cache.get(key);
  if (hit && !(hit instanceof Promise)) return hit as VDBCachedPipeline<GPUComputePipeline>;
  checkGroups(spec.label, spec.groups);
  device.pushErrorScope("validation");
  const layouts = layoutsOf(device, spec.label, spec.groups);
  const module = device.createShaderModule({ label: spec.label, code: spec.code });
  const pipeline = device.createComputePipeline({
    label: spec.label,
    layout: device.createPipelineLayout({ label: spec.label, bindGroupLayouts: layouts }),
    compute: { module, entryPoint: spec.entryPoint },
  });
  const built = { pipeline, layouts, error: logged(spec.label, device.popErrorScope()) };
  cache.set(key, built);
  return built;
}

/** The render pipeline of `spec` on `device`, built once (synchronously) per device and key. */
export function renderPipeline(device: GPUDevice, spec: VDBRenderSpec): VDBCachedPipeline<GPURenderPipeline> {
  const cache = cacheOf(device);
  const key = renderKey(spec);
  const hit = cache.get(key);
  if (hit && !(hit instanceof Promise)) return hit as VDBCachedPipeline<GPURenderPipeline>;
  checkGroups(spec.label, spec.groups);
  device.pushErrorScope("validation");
  const layouts = layoutsOf(device, spec.label, spec.groups);
  const module = device.createShaderModule({ label: spec.label, code: spec.code });
  const pipeline = device.createRenderPipeline(renderDescriptor(spec, module, device.createPipelineLayout({ label: spec.label, bindGroupLayouts: layouts })));
  const built = { pipeline, layouts, error: logged(spec.label, device.popErrorScope()) };
  cache.set(key, built);
  return built;
}

/** Builds `spec` with `createComputePipelineAsync` into the cache (no-op when cached). */
export async function prepareCompute(device: GPUDevice, spec: VDBComputeSpec): Promise<void> {
  const cache = cacheOf(device);
  const key = computeKey(spec);
  const hit = cache.get(key);
  if (hit) {
    await hit;
    return;
  }
  checkGroups(spec.label, spec.groups);
  const job = (async (): Promise<AnyPipeline> => {
    device.pushErrorScope("validation");
    const layouts = layoutsOf(device, spec.label, spec.groups);
    const module = device.createShaderModule({ label: spec.label, code: spec.code });
    const layout = device.createPipelineLayout({ label: spec.label, bindGroupLayouts: layouts });
    const scope = device.popErrorScope();
    const pipeline = await device.createComputePipelineAsync({ label: spec.label, layout, compute: { module, entryPoint: spec.entryPoint } });
    return { pipeline, layouts, error: logged(spec.label, scope) };
  })();
  await settle(cache, key, job);
}

/** Caches `job` under `key`, then its result; a failed build leaves no entry and rethrows. */
async function settle(cache: Map<string, AnyPipeline | Promise<AnyPipeline>>, key: string, job: Promise<AnyPipeline>): Promise<void> {
  cache.set(key, job);
  try {
    cache.set(key, await job);
  } catch (e) {
    cache.delete(key);
    throw e;
  }
}

/** Builds `spec` with `createRenderPipelineAsync` into the cache (no-op when cached). */
export async function prepareRender(device: GPUDevice, spec: VDBRenderSpec): Promise<void> {
  const cache = cacheOf(device);
  const key = renderKey(spec);
  const hit = cache.get(key);
  if (hit) {
    await hit;
    return;
  }
  checkGroups(spec.label, spec.groups);
  const job = (async (): Promise<AnyPipeline> => {
    device.pushErrorScope("validation");
    const layouts = layoutsOf(device, spec.label, spec.groups);
    const module = device.createShaderModule({ label: spec.label, code: spec.code });
    const layout = device.createPipelineLayout({ label: spec.label, bindGroupLayouts: layouts });
    const scope = device.popErrorScope();
    const pipeline = await device.createRenderPipelineAsync(renderDescriptor(spec, module, layout));
    return { pipeline, layouts, error: logged(spec.label, scope) };
  })();
  await settle(cache, key, job);
}
