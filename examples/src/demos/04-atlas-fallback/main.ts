/**
 * Demo 04 — atlas-fallback: the Phase 4 gate (SPEC §3.3/§5, docs/PLAN.md).
 *
 * Proves the compute toolset end to end: `decodeToAtlas` turns
 * `sphere_fog_fp8.nvdb` into a dense array, rendered with three's OWN stock
 * `VolumeNodeMaterial` (NOT `NanoVDBVolumeMaterial`'s from-scratch fragment
 * raymarch) — the mobile/compat fallback path for a GPU that doesn't need
 * (or can't do) the full storage-buffer traversal every fragment. Also runs
 * `gridStats` and `valueTransform` against a second (Float) fixture and
 * cross-checks both against an in-page CPU reference, publishing
 * verification booleans.
 *
 * The SPEC mini-design pictures the atlas landing in a `THREE.Data3DTexture`
 * sampled via TSL `texture3D()`; this sandbox's Dawn/SwiftShader build
 * rejects that combination outright (a real WebGPU validation error, not a
 * mistake in this file — see the "decodeToAtlas -> VolumeNodeMaterial"
 * section below for the full finding), so this demo instead binds the atlas
 * as a `StorageBufferAttribute` and samples it with a hand-written WGSL
 * function via `wgslFn` — the exact Phase 1-3 binding mechanism every other
 * demo already relies on.
 *
 * ## Why a second fixture for stats/transform
 *
 * `valueTransform` is Float-only in v1 (SPEC §3.3 mini-design) and the atlas
 * fixture (`sphere_fog_fp8`) is Fp8, so `box_fog_float.nvdb` — small (81 x 41
 * x 61 index bbox, every voxel active, ~200K voxels) and Float-typed —
 * exercises both `gridStats` (full CPU cross-check, cheap at this size) and
 * `valueTransform` (sidecar-sample cross-check) instead.
 *
 * ## CPU truth: `nanovdb-wgsl`'s package-exported CPU reference
 *
 * `packages/nanovdb-wgsl/src/cpu/*` used to pull its layout constants from
 * `vendor/stride-tables.json` via `node:fs`, which has no browser
 * equivalent (a documented debt — see docs/handoffs/PHASE-5.md's "Known
 * debts"). It now reads a baked, browser-safe generated module instead, so
 * this demo imports the package's own `readValue` directly rather than
 * duplicating the descent logic.
 *
 * ## Presentation & determinism
 *
 * Same offscreen-RT -> 2D-canvas blit as demo 02 (canvas *presentation*
 * drops the Dawn instance under this sandbox's SwiftShader — see
 * docs/handoffs/PHASE-3.md), the same swizzle/adapter-keepalive shims, and
 * the same `?test=1` deterministic mode (fixed pose, one settled render).
 * `window.__DEMO04__ = { ready, statsOk, transformOk, error? }` is the e2e
 * read channel.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Fn, storage, uniform, wgslFn } from "three/tsl";
import { StorageBufferAttribute, VolumeNodeMaterial } from "three/webgpu";
import type { Node, WebGPURenderer } from "three/webgpu";
import { NanoVDBFile, readValue } from "nanovdb-wgsl";
import pnanovdbSource from "nanovdb-wgsl/pnanovdb.wgsl?raw";
import { NanoVDBGrid, buildComputeShaderSource, createVolumeRenderer, decodeToAtlas, gridStats, valueTransform } from "three-nanovdb";

/**
 * Hand-rolled trilinear sample over the `decodeToAtlas` output, packed 4
 * bytes/u32 (`StorageBufferAttribute`, bound the same way demo 01's spike
 * and `material.ts` bind the grid image — a `ptr<storage, array<u32>, read>`
 * parameter to `wgslFn`). `uvw` is clamped and per-axis clamped-to-edge at
 * the integer-coordinate level (matching a real `ClampToEdgeWrapping` 3D
 * texture's behavior) rather than wrapping. Nearest/trilinear filter choice
 * isn't exposed here (this demo only asks `decodeToAtlas` for `trilinear`
 * data); this WGSL always does an 8-tap trilinear reconstruction over that
 * data for a smoother look than nearest would give at this resolution.
 */
const ATLAS_SAMPLE_WGSL = /* wgsl */ `
fn nvdbx_sample_atlas(atlas_ptr : ptr<storage, array<u32>, read>, uvw : vec3<f32>, dims : vec3<f32>) -> f32 {
  let dims_i = vec3<i32>(dims);
  let p = clamp(uvw, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0)) * dims - vec3<f32>(0.5, 0.5, 0.5);
  let p0 = floor(p);
  let f = p - p0;
  var result = 0.0;
  for (var dz = 0; dz < 2; dz = dz + 1) {
    for (var dy = 0; dy < 2; dy = dy + 1) {
      for (var dx = 0; dx < 2; dx = dx + 1) {
        let ix = clamp(i32(p0.x) + dx, 0, dims_i.x - 1);
        let iy = clamp(i32(p0.y) + dy, 0, dims_i.y - 1);
        let iz = clamp(i32(p0.z) + dz, 0, dims_i.z - 1);
        let flat = u32(ix + iy * dims_i.x + iz * dims_i.x * dims_i.y);
        let word = (*atlas_ptr)[flat / 4u];
        let shift = (flat % 4u) * 8u;
        let v = f32((word >> shift) & 0xFFu) / 255.0;
        let wx = select(1.0 - f.x, f.x, dx == 1);
        let wy = select(1.0 - f.y, f.y, dy == 1);
        let wz = select(1.0 - f.z, f.z, dz == 1);
        result = result + v * wx * wy * wz;
      }
    }
  }
  return result;
}
`;

interface Demo04State {
  ready: boolean;
  statsOk: boolean;
  transformOk: boolean;
  error?: string;
}

declare global {
  interface Window {
    __DEMO04__?: Demo04State;
  }
}

interface Sample {
  ijk: [number, number, number];
  value: number;
  active: boolean;
}
interface Sidecar {
  grid: { name: string; activeVoxelCount: number };
  samples: Sample[];
}

const ATLAS_FIXTURE = "/fixtures/primitives/sphere_fog_fp8.nvdb";
const STATS_FIXTURE_NVDB = "/fixtures/primitives/box_fog_float.nvdb";
const STATS_FIXTURE_SIDECAR = "/fixtures/primitives/box_fog_float.sidecar.json";

const STATS_VALUE_EPS = 1e-3; // FP8 atlas isn't involved here, but keep parity with demo 01/harness tolerances
const TRANSFORM_VALUE_EPS = 1e-4;
const MAX_TRANSFORM_SAMPLES = 50;

/** Same r185 x SwiftShader compat shim as demo 02 (see its module header). */
function installSwizzleCompatShim(): void {
  if (typeof GPUTexture === "undefined") return;
  const proto = GPUTexture.prototype as unknown as {
    createView: (d?: GPUTextureViewDescriptor) => GPUTextureView;
    __swizzleShim?: boolean;
  };
  if (proto.__swizzleShim) return;
  const orig = proto.createView;
  proto.createView = function (this: GPUTexture, descriptor?: GPUTextureViewDescriptor): GPUTextureView {
    if (descriptor && typeof (descriptor as { swizzle?: unknown }).swizzle === "string") {
      const { swizzle: _drop, ...rest } = descriptor as GPUTextureViewDescriptor & { swizzle?: unknown };
      return orig.call(this, rest);
    }
    return orig.call(this, descriptor);
  };
  proto.__swizzleShim = true;
}

const _pinnedAdapters: unknown[] = [];
function installAdapterKeepAliveShim(): void {
  if (typeof navigator === "undefined" || !navigator.gpu) return;
  const gpu = navigator.gpu as GPU & { __keepAliveShim?: boolean };
  if (gpu.__keepAliveShim) return;
  const orig = gpu.requestAdapter.bind(gpu);
  gpu.requestAdapter = async (options?: GPURequestAdapterOptions): Promise<GPUAdapter | null> => {
    const adapter = await orig(options);
    if (adapter) _pinnedAdapters.push(adapter);
    return adapter;
  };
  gpu.__keepAliveShim = true;
}

const params = new URLSearchParams(location.search);
const TEST_MODE = params.get("test") === "1";

const statsEl = document.querySelector<HTMLDivElement>("#stats")!;
const errEl = document.querySelector<HTMLDivElement>("#err")!;
const panelEl = document.querySelector<HTMLDivElement>("#panel")!;
const appEl = document.querySelector<HTMLDivElement>("#app")!;

function fail(message: string): void {
  errEl.textContent = message;
  statsEl.textContent = "failed";
  window.__DEMO04__ = { ready: false, statsOk: false, transformOk: false, error: message };
}

async function fetchOk(url: string): Promise<ArrayBuffer | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return await res.arrayBuffer();
  } catch {
    return null;
  }
}

/**
 * Full CPU pass over `grid`'s dense index bbox (fine at this fixture's ~200K
 * voxel scale — see docs/PLAN.md's test-strategy note that a full pass is
 * fine for the small fixtures). Mirrors `gridStats`'s own definitions
 * exactly: mean/min/max are over ACTIVE voxels only.
 */
function cpuGridStats(
  grid: NanoVDBGrid,
): { min: number; max: number; mean: number; activeVoxelCount: number } {
  const [minX, minY, minZ] = grid.metadata.indexBBox.min;
  const [maxX, maxY, maxZ] = grid.metadata.indexBBox.max;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sum = 0;
  let activeVoxelCount = 0;
  for (let x = minX; x <= maxX; x++) {
    for (let y = minY; y <= maxY; y++) {
      for (let z = minZ; z <= maxZ; z++) {
        const r = readValue(grid.image, [x, y, z]);
        if (!r.active) continue;
        activeVoxelCount++;
        sum += r.value;
        if (r.value < min) min = r.value;
        if (r.value > max) max = r.value;
      }
    }
  }
  return { min, max, mean: activeVoxelCount > 0 ? sum / activeVoxelCount : 0, activeVoxelCount };
}

/**
 * Resolves, for each `ijk`, whether it is an individually-addressable LEAF
 * voxel (`level === 0`, per `pnanovdb_readaccessor_get_value_address_and_
 * level` — the exact primitive `valueTransform`'s WGSL uses to decide what
 * to mutate) or a shared upper/lower/root tile constant / background value
 * (`level >= 1`) — plus the value stored there. A tiny ad hoc raw-WebGPU pass
 * built on top of `buildComputeShaderSource` (the same footer `gridStats`/
 * `decodeToAtlas` share), demo-scoped rather than exported from the package:
 * it's a verification-only primitive, not part of the compute toolset's
 * public surface.
 *
 * Needed because `box_fog_float`'s solid interior collapses into a handful
 * of constant LOWER-level tiles despite EVERY voxel in its dense bbox being
 * active (see this module's `run()` doc) — so validating `valueTransform`
 * against "all active sidecar samples double" is simply the wrong
 * expectation for this fixture's topology. The right check is: LEAF-level
 * samples double, tile-level samples are byte-for-byte unchanged (proving
 * the documented "leaf only" limitation is actually enforced, not just
 * documented).
 */
async function resolveVoxelLevels(
  device: GPUDevice,
  grid: NanoVDBGrid,
  ijks: ReadonlyArray<readonly [number, number, number]>,
): Promise<Array<{ level: number; value: number }>> {
  const source = `${buildComputeShaderSource(pnanovdbSource)}
@group(1) @binding(0) var<storage, read> nvdbx_probe_coords : array<i32>;
@group(1) @binding(1) var<storage, read_write> nvdbx_probe_out : array<f32>; // [level, value] per probe
@compute @workgroup_size(1)
fn nvdbx_probe_level(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= arrayLength(&nvdbx_probe_out) / 2u) { return; }
  let base = gid.x * 3u;
  let ijk = vec3<i32>(nvdbx_probe_coords[base], nvdbx_probe_coords[base + 1u], nvdbx_probe_coords[base + 2u]);
  let grid_type = nvdbx_grid_type();
  let buf = pnanovdb_make_buffer(0u, arrayLength(&nanovdb_buffer));
  let root = nvdbx_root();
  var acc : pnanovdb_readaccessor_t;
  pnanovdb_readaccessor_init(&acc, root);
  let resolved = pnanovdb_readaccessor_get_value_address_and_level(grid_type, buf, &acc, ijk);
  nvdbx_probe_out[gid.x * 2u] = f32(resolved.level);
  nvdbx_probe_out[gid.x * 2u + 1u] = pnanovdb_read_float(buf, resolved.address);
}
`;
  const module = device.createShaderModule({ code: source });

  const [minX, minY, minZ] = grid.metadata.indexBBox.min;
  const [maxX, maxY, maxZ] = grid.metadata.indexBBox.max;
  const paramsI32 = new Int32Array(12);
  paramsI32[0] = grid.gridTypeId;
  paramsI32[1] = minX;
  paramsI32[2] = minY;
  paramsI32[3] = minZ;
  paramsI32[4] = maxX - minX + 1;
  paramsI32[5] = maxY - minY + 1;
  paramsI32[6] = maxZ - minZ + 1;

  const coords = new Int32Array(ijks.length * 3);
  ijks.forEach(([x, y, z], i) => {
    coords[i * 3] = x;
    coords[i * 3 + 1] = y;
    coords[i * 3 + 2] = z;
  });

  const group0Layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ],
  });
  const group1Layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });

  const gridBuf = device.createBuffer({
    size: grid.image.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(gridBuf, 0, grid.image.slice().buffer);
  const paramsBuf = device.createBuffer({
    size: paramsI32.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuf, 0, paramsI32.slice().buffer);
  const paramsF32Buf = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const coordBuf = device.createBuffer({
    size: Math.max(4, coords.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(coordBuf, 0, coords.slice().buffer);
  const outBuf = device.createBuffer({
    size: Math.max(4, ijks.length * 2 * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const group0 = device.createBindGroup({
    layout: group0Layout,
    entries: [
      { binding: 0, resource: { buffer: gridBuf } },
      { binding: 1, resource: { buffer: paramsBuf } },
      { binding: 2, resource: { buffer: paramsF32Buf } },
    ],
  });
  const group1 = device.createBindGroup({
    layout: group1Layout,
    entries: [
      { binding: 0, resource: { buffer: coordBuf } },
      { binding: 1, resource: { buffer: outBuf } },
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [group0Layout, group1Layout] }),
    compute: { module, entryPoint: "nvdbx_probe_level" },
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group0);
  pass.setBindGroup(1, group1);
  pass.dispatchWorkgroups(Math.max(1, ijks.length));
  pass.end();
  const readBuf = device.createBuffer({
    size: outBuf.size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBuf.size);
  device.queue.submit([encoder.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();

  return ijks.map((_, i) => ({ level: out[i * 2]!, value: out[i * 2 + 1]! }));
}

async function run(): Promise<void> {
  if (!navigator.gpu) {
    fail("WebGPU unavailable (navigator.gpu undefined) — needs a secure context + adapter.");
    return;
  }
  installSwizzleCompatShim();
  installAdapterKeepAliveShim();

  const [atlasBuf, statsBuf, sidecarRes] = await Promise.all([
    fetchOk(ATLAS_FIXTURE),
    fetchOk(STATS_FIXTURE_NVDB),
    fetch(STATS_FIXTURE_SIDECAR).catch(() => null),
  ]);
  if (!atlasBuf) {
    fail(`Missing atlas fixture: ${ATLAS_FIXTURE}`);
    return;
  }
  if (!statsBuf) {
    fail(`Missing stats fixture: ${STATS_FIXTURE_NVDB}`);
    return;
  }
  if (!sidecarRes || !sidecarRes.ok) {
    fail(`Missing sidecar: ${STATS_FIXTURE_SIDECAR}`);
    return;
  }
  const sidecar = (await sidecarRes.json()) as Sidecar;

  const atlasGrid = NanoVDBGrid.fromFile(NanoVDBFile.fromArrayBuffer(atlasBuf), 0);
  const statsGrid = NanoVDBGrid.fromFile(NanoVDBFile.fromArrayBuffer(statsBuf), 0);

  const { renderer, device } = await createVolumeRenderer({
    gridBytes: Math.max(atlasGrid.byteLength, statsGrid.byteLength),
  });

  // -------------------------------------------------------------------------
  // gridStats: GPU result vs. full CPU pass over box_fog_float.
  // -------------------------------------------------------------------------
  const gpuStats = await gridStats(device, statsGrid, pnanovdbSource);
  const cpuStats = cpuGridStats(statsGrid);
  const statsOk =
    gpuStats.activeVoxelCount === sidecar.grid.activeVoxelCount &&
    gpuStats.activeVoxelCount === cpuStats.activeVoxelCount &&
    Math.abs(gpuStats.min - cpuStats.min) <= STATS_VALUE_EPS &&
    Math.abs(gpuStats.max - cpuStats.max) <= STATS_VALUE_EPS &&
    Math.abs(gpuStats.mean - cpuStats.mean) <= STATS_VALUE_EPS &&
    gpuStats.histogram.reduce((a, b) => a + b, 0) === gpuStats.activeVoxelCount;

  // -------------------------------------------------------------------------
  // valueTransform: "return v * 2.0;", then CPU-readValue the RESULT image at
  // up to 50 active sidecar coords. `box_fog_float`'s solid interior turns
  // out to collapse into constant LOWER-level tiles despite every voxel in
  // its dense bbox being active (found via `resolveVoxelLevels` while
  // debugging this demo) — so the correct expectation per sample is:
  //   - LEAF-resolved (level 0): value doubles (valueTransform touched it).
  //   - tile/background-resolved (level >= 1): value is BYTE-IDENTICAL
  //     (valueTransform intentionally skips shared tile storage — see its
  //     doc comment in packages/three-nanovdb/src/compute.ts).
  // Both branches are asserted, which also positively proves the
  // leaf-vs-tile distinction is enforced rather than merely documented.
  // -------------------------------------------------------------------------
  const { image: transformedImage } = await valueTransform(device, statsGrid, pnanovdbSource, "return v * 2.0;");
  const transformedGrid = new NanoVDBGrid({ image: transformedImage, metadata: statsGrid.metadata });
  const activeSamples = sidecar.samples.filter((s) => s.active).slice(0, MAX_TRANSFORM_SAMPLES);
  const levels = await resolveVoxelLevels(
    device,
    statsGrid,
    activeSamples.map((s) => s.ijk),
  );

  let sawLeaf = false;
  let sawTile = false;
  let transformOk = activeSamples.length > 0;
  activeSamples.forEach((s, i) => {
    const got = readValue(transformedGrid.image, s.ijk);
    const isLeaf = levels[i]!.level === 0;
    if (isLeaf) {
      sawLeaf = true;
      if (Math.abs(got.value - 2 * s.value) > TRANSFORM_VALUE_EPS) transformOk = false;
    } else {
      sawTile = true;
      if (Math.abs(got.value - s.value) > TRANSFORM_VALUE_EPS) transformOk = false;
    }
  });
  // A vacuously-true result (no leaf samples at all, so nothing actually got
  // exercised) doesn't count as verified.
  if (!sawLeaf) transformOk = false;
  void sawTile; // exercised opportunistically; not required for the gate

  // -------------------------------------------------------------------------
  // decodeToAtlas -> VolumeNodeMaterial (the fallback path).
  //
  // SPEC's mini-design pictures `decodeToAtlas` feeding a `THREE.Data3DTexture`
  // sampled via TSL `texture3D()`. That combination hits a genuine WebGPU
  // validation error in THIS sandbox's Chromium/SwiftShader (Dawn) build —
  // "The dimension (TextureViewDimension::e2D) of the texture view is not
  // compatible with the dimension (TextureDimension::e3D)" — thrown for
  // EVERY z-slice upload AND at render Submit time, for both `RedFormat` and
  // `RGBAFormat` 3D textures alike (ruled out a format-specific cause by
  // testing both), so it looks like a genuine 3D-texture support gap in this
  // environment rather than a mistake in this file (matches the project's
  // running list of environment-specific WebGPU quirks — see the swizzle
  // shim above and docs/handoffs/PHASE-3.md). Re-test `Data3DTexture` +
  // `texture3D()` directly on real hardware; the package's own
  // `decodeToAtlas` output (a plain typed array) is unaffected either way.
  //
  // Demo-level workaround: skip `Data3DTexture` entirely and reuse the
  // ALREADY-PROVEN Phase 1-3 binding mechanism instead — a `StorageBuffer
  // Attribute` bound via `storage()` and read from a hand-written WGSL
  // function through `wgslFn` (exactly demo 01's spike / `material.ts`'s
  // pattern) — packing the atlas bytes 4-per-u32 and trilinearly sampling
  // them by hand in `ATLAS_SAMPLE_WGSL` below.
  // -------------------------------------------------------------------------
  const atlas = await decodeToAtlas(device, atlasGrid, pnanovdbSource, {
    maxDim: 96,
    filter: "trilinear",
    format: "uint8",
    normalize: true,
  });
  const [dx, dy, dz] = atlas.dims;
  const voxelCount = dx * dy * dz;
  const packedWordCount = Math.ceil(voxelCount / 4);
  const packedBytes = new Uint8Array(packedWordCount * 4);
  packedBytes.set(atlas.data as Uint8Array);
  const packedAtlas = new Uint32Array(packedBytes.buffer);
  const atlasAttr = new StorageBufferAttribute(packedAtlas, 1);
  const atlasStorage = storage(atlasAttr, "uint", packedAtlas.length).toReadOnly();
  const dimsUniform = uniform(new THREE.Vector3(dx, dy, dz));

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b1020);

  const worldBox = atlasGrid.worldBBox();
  const center = worldBox.getCenter(new THREE.Vector3());
  const size = worldBox.getSize(new THREE.Vector3());
  const diag = size.length();

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, diag * 100);

  // Map world position -> the box's normalized [0,1]^3 texture space. The
  // proxy geometry sits exactly on worldBox with no extra mesh transform
  // (NanoVDBGrid.proxyGeometry()'s own contract), so this is a plain
  // translate + per-axis inverse-scale, no matrix inverse needed.
  const boxMin = uniform(worldBox.min.clone());
  const boxSizeInv = uniform(new THREE.Vector3(1 / size.x, 1 / size.y, 1 / size.z));
  const tintColor = uniform(new THREE.Vector3(0.85, 0.9, 1.0));
  const densityGain = uniform(0.9);

  const material = new VolumeNodeMaterial();
  material.steps = 48;
  // `scatteringEmissiveNode` is a real, runtime-supported `VolumeNodeMaterial`
  // property (`VolumetricLightingModel.start()` reads it directly, adding its
  // result independent of any scene lights) — used here instead of
  // `scatteringNode` (which only contributes via the lighting pipeline's
  // `direct()` callback, i.e. only when a real `PointLight`/`SpotLight` with
  // a `.distance` is present in the scene) so this demo needs no light setup
  // at all, just the density texture. `@types/three@0.185.1` hasn't caught up
  // with this field yet (only `scatteringNode` is typed), hence the cast.
  const sampleAtlas = wgslFn(ATLAS_SAMPLE_WGSL);
  (material as VolumeNodeMaterial & { scatteringEmissiveNode?: (p: { positionRay: Node<"vec3"> }) => Node }).scatteringEmissiveNode = Fn(
    ({ positionRay }: { positionRay: Node<"vec3"> }) => {
      const uvw = positionRay.sub(boxMin).mul(boxSizeInv);
      // wgslFn is typed as returning a bare Node; the WGSL returns f32 (demo
      // 01's same cast-around-the-generic-type pattern).
      const density = sampleAtlas({ atlas_ptr: atlasStorage, uvw, dims: dimsUniform }) as unknown as Node<"float">;
      return tintColor.mul(density).mul(densityGain);
    },
  );

  const mesh = new THREE.Mesh(atlasGrid.proxyGeometry(), material);
  scene.add(mesh);

  // `NodeMaterial.setupLighting()` (three's own WebGPU node-material pipeline)
  // skips the ENTIRE lighting-model codepath — meaning `VolumetricLighting
  // Model.start()`/`finish()`, and therefore `scatteringEmissiveNode` above —
  // unless the scene has at least one light (`lightsNode.getScope().
  // hasLights`, see NodeMaterial.js). `scatteringEmissiveNode` itself needs
  // no actual light contribution (it adds to `stepLight` unconditionally,
  // independent of any light's `direct()` callback), so a real light merely
  // needs to be PRESENT; zero intensity keeps it a no-op otherwise.
  scene.add(new THREE.AmbientLight(0xffffff, 0));

  const display = document.createElement("canvas");
  display.id = "view";
  display.style.width = "100%";
  display.style.height = "100%";
  display.style.display = "block";
  appEl.appendChild(display);
  const ctx = display.getContext("2d")!;

  const controls = new OrbitControls(camera, display);
  controls.target.copy(center);
  controls.enableDamping = !TEST_MODE;

  const rt = new THREE.RenderTarget(2, 2);
  rt.texture.colorSpace = THREE.SRGBColorSpace;
  let flip = new Uint8ClampedArray(0);
  let vw = 0;
  let vh = 0;

  function setViewport(w: number, h: number): void {
    vw = Math.max(1, w);
    vh = Math.max(1, h);
    display.width = vw;
    display.height = vh;
    rt.setSize(vw, vh);
    flip = new Uint8ClampedArray(vw * vh * 4);
    camera.aspect = vw / vh;
    camera.updateProjectionMatrix();
  }

  function placeCamera(azimuth: number, elevation: number, radiusFactor: number): void {
    const r = diag * radiusFactor;
    camera.position.set(
      center.x + r * Math.cos(elevation) * Math.cos(azimuth),
      center.y + r * Math.sin(elevation),
      center.z + r * Math.cos(elevation) * Math.sin(azimuth),
    );
    camera.lookAt(center);
    controls.update();
  }

  async function presentFrame(): Promise<void> {
    (renderer as WebGPURenderer).setRenderTarget(rt);
    await (renderer as WebGPURenderer).renderAsync(scene, camera);
    const pixels = (await (renderer as WebGPURenderer).readRenderTargetPixelsAsync(rt, 0, 0, vw, vh)) as Uint8Array;
    (renderer as WebGPURenderer).setRenderTarget(null);
    const stride = vw * 4;
    for (let y = 0; y < vh; y++) {
      const src = (vh - 1 - y) * stride;
      flip.set(pixels.subarray(src, src + stride), y * stride);
    }
    ctx.putImageData(new ImageData(flip, vw, vh), 0, 0);
  }

  statsEl.textContent =
    `atlas: ${dx}x${dy}x${dz} (${atlas.range.min.toFixed(3)}..${atlas.range.max.toFixed(3)})\n` +
    `gridStats: min ${gpuStats.min.toFixed(4)} max ${gpuStats.max.toFixed(4)} mean ${gpuStats.mean.toFixed(4)} ` +
    `active ${gpuStats.activeVoxelCount} (fast path: ${gpuStats.usedAtomicFastPath})\n` +
    `statsOk: ${statsOk}  transformOk: ${transformOk}`;

  if (TEST_MODE) {
    panelEl.classList.add("test-hidden");
    setViewport(640, 480);
    placeCamera(Math.PI * 0.25, Math.PI * 0.16, 1.7);
    try {
      await presentFrame();
      await presentFrame();
      window.__DEMO04__ = { ready: true, statsOk, transformOk };
    } catch (err) {
      fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));
    }
    return;
  }

  setViewport(window.innerWidth, window.innerHeight);
  placeCamera(Math.PI * 0.25, Math.PI * 0.18, 1.8);
  window.addEventListener("resize", () => setViewport(window.innerWidth, window.innerHeight));

  let published = false;
  let busy = false;
  const loop = (): void => {
    requestAnimationFrame(loop);
    if (busy || window.__DEMO04__?.error) return;
    busy = true;
    controls.update();
    presentFrame()
      .then(() => {
        busy = false;
        if (!published) {
          published = true;
          window.__DEMO04__ = { ready: true, statsOk, transformOk };
        }
      })
      .catch((err: unknown) => {
        busy = false;
        fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));
      });
  };
  requestAnimationFrame(loop);
}

run().catch((err: unknown) => {
  fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));
});
