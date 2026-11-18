/**
 * K3.0 harness (R1 §6 Step 0): our side of the renderer parity rig.
 *
 * 1. Reads a `.wvdb` (default: the committed CC0 `cloud_04_variant_0000.wvdb`;
 *    `?file=<url>` for another, e.g. `/fixtures/k3/bunny_cloud.wvdb`), builds the
 *    dense volume his renderer would upload (`wvdbDenseVolume`, `normalize: 'clamp01'`),
 *    and uploads it as three `Data3DTexture`s (`vdbDenseTextures`).
 * 2. Renders it with a minimal emission-absorption march (a three r187
 *    `NodeMaterial`) that reads only through the `VDBVolumeSource` node shape
 *    (`vdbVolumeNodes`: `boxSegment`, `getLod`, `sampleExtinction`,
 *    `sampleAlbedo`, `sampleEmission`). Fixed camera, fixed step count, no
 *    jitter, no temporal: deterministic. Step 1 replaces this material with the
 *    ported froxel pipeline. Offscreen `RenderTarget` (sRGB), read back as RGBA8.
 * 3. Raw-WebGPU check of the core `vdbDenseSource` on the same device: a compute
 *    kernel calls the generated `vdbSrcDensity` at probe points and LODs; the
 *    values are compared with the CPU twin `sampleWvdbDenseTexture`.
 *
 * Results go to `window.__VDB_K3_DENSE__` for e2e/k3-dense.spec.ts (which writes
 * the PNG). Query: `?w=&h=` (default 320x240), `?mips=0` (no mip atlas), `?lod=`,
 * `?steps=` (default 96), `?extinction=` (default 20), `?timing=1` (GPU timestamp hook, provisional per D25).
 */
import * as THREE from "three";
import { NodeMaterial, WebGPURenderer } from "three/webgpu";
import { Fn, Loop, cameraPosition, exp, float, length, modelWorldMatrixInverse, positionLocal, uniform, vec3, vec4 } from "three/tsl";
import { vdbDenseSource } from "@pmndrs/vdb";
import { getDevice, vdbDenseTextures, vdbVolumeNodes, vdbVolumeUniforms } from "@pmndrs/vdb/three";
import { readWvdb, sampleWvdbDenseTexture, wvdbDenseVolume } from "../../../packages/vdb/src/tools/internal.js";
import defaultWvdbUrl from "../../../packages/vdb/test/tools/fixtures/wvdb/cloud_04_variant_0000.wvdb?url";

interface ProbeStats {
  count: number;
  maxAbs: number;
  meanAbs: number;
  over: number;
  tolerance: number;
  lodLevels: number;
  texels: number[];
  firstBad?: { uvw: number[]; lod: number; gpu: number; cpu: number };
}

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  file?: string;
  volume?: { dimensions: number[]; lod: number; meanDensity: number; maxDensity: number; activeTexels: number; mipLevels: number; statsSource: string };
  width?: number;
  height?: number;
  /** RGBA8, top row first, base64. */
  rgba?: string;
  /** The same frame rendered twice in this page gave identical bytes. */
  inPageIdentical?: boolean;
  nonBackground?: number;
  probes?: ProbeStats;
  /** GPU timing hook (D25: provisional; only with ?timing=1 and timestamp-query). */
  timings?: { provisional: true; renderMs: number | null; note: string };
  uncaptured: string[];
}

declare global {
  interface Window {
    __VDB_K3_DENSE__?: Result;
  }
}

const params = new URLSearchParams(location.search);
const W = Number(params.get("w") ?? 320);
const H = Number(params.get("h") ?? 240);
const STEPS = Number(params.get("steps") ?? 96);
const MIPS = params.get("mips") !== "0";
const TIMING = params.get("timing") === "1";
// His default is 100 (opaque at this size); 20 keeps the cloud's structure visible in the harness image.
const EXTINCTION = Number(params.get("extinction") ?? 20);
const FILE = params.get("file") ?? defaultWvdbUrl;
const BACKGROUND = new THREE.Color(0.12, 0.16, 0.22);

const root = document.getElementById("demo-root")!;
const result: Result = { done: false, uncaptured: [] };
window.__VDB_K3_DENSE__ = result;

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
}

function toBase64(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

/** Deterministic xorshift in [0, 1). */
function rng(seed: number): () => number {
  let x = seed >>> 0 || 1;
  return () => {
    x ^= x << 13;
    x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5;
    x >>>= 0;
    return x / 4294967296;
  };
}

async function main(): Promise<void> {
  root.textContent = "";
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU: no adapter");
  const info = adapter.info;
  result.adapter = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(" ");
  const features: GPUFeatureName[] = TIMING && adapter.features.has("timestamp-query") ? ["timestamp-query"] : [];
  const device = await adapter.requestDevice({ requiredFeatures: features });
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));

  // --- the dense volume (his bytes) -------------------------------------------------------
  const response = await fetch(FILE);
  if (!response.ok) throw new Error(`fetch ${FILE}: ${response.status}`);
  const file = readWvdb(new Uint8Array(await response.arrayBuffer()));
  const lodParam = params.get("lod");
  const volume = wvdbDenseVolume(file, { mipmaps: MIPS, ...(lodParam !== null ? { lod: Number(lodParam) } : {}) });
  result.file = FILE;
  result.volume = {
    dimensions: [...volume.dimensions],
    lod: volume.lod,
    meanDensity: volume.meanDensity,
    maxDensity: volume.maxDensity,
    activeTexels: volume.activeTexels,
    mipLevels: volume.density.mip?.levels ?? 0,
    statsSource: volume.densityStatsSource,
  };
  log(`${FILE.split("/").pop()}: LOD ${volume.lod}, ${volume.dimensions.join("x")} texels, mean ${volume.meanDensity.toFixed(4)}, ${result.volume.mipLevels} mip levels`);

  // --- three: renderer, source, minimal march --------------------------------------------
  const renderer = new WebGPURenderer({ device, trackTimestamp: TIMING && features.length > 0 });
  await renderer.init();
  const source = vdbDenseTextures(volume);
  const bbox = volume.bbox ?? [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]];
  const center = new THREE.Vector3(...bbox[0]).add(new THREE.Vector3(...bbox[1])).multiplyScalar(0.5);
  const size = new THREE.Vector3(...bbox[1]).sub(new THREE.Vector3(...bbox[0]));
  const worldScale = Math.max(size.x, size.y, size.z);
  // His FrameVolume default is extinction 100 with normalizeDensity (frame.js L52, L100–102); see EXTINCTION.
  const u = vdbVolumeUniforms(source, { extinction: EXTINCTION, normalizeDensity: true, worldScale });
  const nodes = vdbVolumeNodes(source, u, { useMipmaps: MIPS });

  const lightColor = uniform(new THREE.Color(1.0, 0.97, 0.92));
  const background = uniform(BACKGROUND.clone());
  const material = new NodeMaterial();
  material.side = THREE.BackSide;
  material.depthWrite = false;
  material.fragmentNode = Fn(() => {
    // Ray in the local unit box: camera → this back-face point; boxSegment clips it to the box.
    const start = modelWorldMatrixInverse.mul(vec4(cameraPosition, 1.0)).xyz.toVar();
    const delta = positionLocal.sub(start).toVar();
    const seg = nodes.boxSegment(start, positionLocal).toVar();
    const dt = seg.y.sub(seg.x).div(float(STEPS)).toVar();
    const stepLocal = length(delta).mul(dt).toVar();
    const stepWorld = stepLocal.mul(u.worldScale).toVar();
    const lod = nodes.getLod(stepLocal).toVar();
    const transmittance = float(1.0).toVar();
    const radiance = vec3(0.0).toVar();
    Loop(STEPS, ({ i }) => {
      const t = seg.x.add(float(i).add(0.5).mul(dt));
      const p = start.add(delta.mul(t)).toVar();
      const tau = nodes.sampleExtinction(p, lod).mul(stepWorld);
      const absorbed = float(1.0).sub(exp(tau.negate()));
      // Emission-absorption: albedo-tinted self-light plus the (here absent) emission channel.
      radiance.addAssign(transmittance.mul(absorbed).mul(nodes.sampleAlbedo(p, lod)).mul(lightColor));
      radiance.addAssign(transmittance.mul(nodes.sampleEmission(p, lod)).mul(stepWorld));
      transmittance.mulAssign(exp(tau.negate()));
    });
    return vec4(radiance.add(background.mul(transmittance)), 1.0);
  })();

  const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
  mesh.position.copy(center);
  mesh.scale.copy(size);
  const scene = new THREE.Scene();
  scene.background = BACKGROUND.clone();
  scene.add(mesh);

  const camera = new THREE.PerspectiveCamera(40, W / H, worldScale * 0.01, worldScale * 20);
  const diag = size.length();
  camera.position.copy(center).add(new THREE.Vector3(0.9, 0.45, 1.1).normalize().multiplyScalar(diag * 1.25));
  camera.lookAt(center);
  camera.updateMatrixWorld();

  const rt = new THREE.RenderTarget(W, H);
  rt.texture.colorSpace = THREE.SRGBColorSpace;

  async function frame(): Promise<Uint8Array> {
    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
    const pixels = (await renderer.readRenderTargetPixelsAsync(rt, 0, 0, W, H)) as Uint8Array;
    renderer.setRenderTarget(null);
    // three r187 returns rows top first (row 0 = screenCoordinate.y 0), padded to 256 bytes: strip the padding only
    // (K3.1b: this used to flip the rows, which made the PNGs upside down).
    const tight = W * 4;
    const stride = H > 1 ? (pixels.byteLength - tight) / (H - 1) : tight;
    if (!Number.isInteger(stride) || stride < tight) throw new Error(`readback layout: ${pixels.byteLength} bytes for ${W}x${H}`);
    const out = new Uint8Array(W * H * 4);
    for (let y = 0; y < H; y++) out.set(pixels.subarray(y * stride, y * stride + tight), y * tight);
    return out;
  }

  const first = await frame();
  const second = await frame();
  result.inPageIdentical = first.length === second.length && first.every((v, i) => v === second[i]);
  if (TIMING && features.length > 0) {
    const ms = await renderer.resolveTimestampsAsync("render");
    result.timings = { provisional: true, renderMs: typeof ms === "number" ? ms : null, note: "D25: not from a quiet machine; never gate on it" };
  } else {
    result.timings = { provisional: true, renderMs: null, note: "timing off (?timing=1 needs timestamp-query); D25" };
  }
  const bg = [BACKGROUND.clone().convertLinearToSRGB()].map((c) => [c.r, c.g, c.b].map((v) => Math.round(v * 255)))[0]!;
  let nonBackground = 0;
  for (let i = 0; i < first.length; i += 4) {
    if (Math.abs(first[i]! - bg[0]!) + Math.abs(first[i + 1]! - bg[1]!) + Math.abs(first[i + 2]! - bg[2]!) > 6) nonBackground++;
  }
  result.width = W;
  result.height = H;
  result.rgba = toBase64(first);
  result.nonBackground = nonBackground;

  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  canvas.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(first), W, H), 0, 0);
  root.append(canvas);
  log(`rendered ${W}x${H}, ${nonBackground} volume pixels; in-page repeat identical: ${result.inPageIdentical}`, result.inPageIdentical ? "ok" : "fail");

  // --- core: vdbDenseSource WGSL vs the CPU twin --------------------------------------------
  result.probes = await probeCore(getDevice(renderer), volume);
  log(
    `core vdbDenseSource: ${result.probes.count} probes, max |GPU - CPU twin| ${result.probes.maxAbs.toExponential(2)} (tolerance ${result.probes.tolerance}), ${result.probes.over} over`,
    result.probes.over === 0 ? "ok" : "fail",
  );
  source.dispose();
}

async function probeCore(device: GPUDevice, volume: ReturnType<typeof wvdbDenseVolume>): Promise<ProbeStats> {
  const src = vdbDenseSource(device, volume, { label: "k3 probe" });
  const slot = { group: 0, firstBinding: 0 };
  const code = `${src.wgsl({ slot, profile: "march" })}
@group(1) @binding(0) var<storage, read> probes: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> outValues: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let n = arrayLength(&probes);
  if (id.x == 0u) {
    let t = vdbSrcTexels(0u);
    outValues[n] = t.x;
    outValues[n + 1u] = t.y;
    outValues[n + 2u] = t.z;
    outValues[n + 3u] = vdbSrcLodLevels(0u);
  }
  if (id.x >= n) {
    return;
  }
  let p = probes[id.x];
  outValues[id.x] = vdbSrcDensity(0u, p.xyz, p.w) * vdbSrcMask(0u, p.xyz, p.w) + vdbSrcEmission(0u, p.xyz, p.w);
}`;
  const random = rng(0x6b33);
  const lods = [0, 0, 0.25, 0.5, 1, 1.5, 2, 2.75, 4, 20];
  const N = 2048;
  const probes = new Float32Array(N * 4);
  for (let i = 0; i < N; i++) {
    for (let a = 0; a < 3; a++) probes[i * 4 + a] = -0.05 + 1.1 * random();
    probes[i * 4 + 3] = lods[i % lods.length]!;
  }
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code, label: "k3 probe" });
  const layout0 = device.createBindGroupLayout({ entries: src.layoutEntries(slot, 0x4) });
  const layout1 = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: 0x4, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: 0x4, buffer: { type: "storage" } },
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout0, layout1] }),
    compute: { module, entryPoint: "main" },
  });
  const probeBuf = device.createBuffer({ size: probes.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(probeBuf, 0, probes);
  const outBytes = (N + 4) * 4;
  const outBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({ layout: layout0, entries: src.bindEntries(slot) }));
  pass.setBindGroup(1, device.createBindGroup({ layout: layout1, entries: [{ binding: 0, resource: { buffer: probeBuf } }, { binding: 1, resource: { buffer: outBuf } }] }));
  pass.dispatchWorkgroups(Math.ceil(N / 64));
  pass.end();
  encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBytes);
  device.queue.submit([encoder.finish()]);
  const err = await device.popErrorScope();
  if (err) throw new Error(`probe kernel: ${err.message}`);
  await readBuf.mapAsync(GPUMapMode.READ);
  const gpu = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();

  // Hardware filtering quantizes the weights (8 fractional bits is common); allow one 8-bit step plus that.
  const tolerance = 2 / 255;
  let maxAbs = 0;
  let sum = 0;
  let over = 0;
  let firstBad: ProbeStats["firstBad"];
  for (let i = 0; i < N; i++) {
    const uvw = [probes[i * 4]!, probes[i * 4 + 1]!, probes[i * 4 + 2]!];
    const lod = probes[i * 4 + 3]!;
    const cpu = sampleWvdbDenseTexture(volume.density, uvw, lod);
    const d = Math.abs(gpu[i]! - cpu);
    maxAbs = Math.max(maxAbs, d);
    sum += d;
    if (d > tolerance) {
      over++;
      firstBad ??= { uvw, lod, gpu: gpu[i]!, cpu };
    }
  }
  src.dispose();
  [probeBuf, outBuf, readBuf].forEach((b) => b.destroy());
  return {
    count: N,
    maxAbs,
    meanAbs: sum / N,
    over,
    tolerance,
    lodLevels: gpu[N + 3]!,
    texels: [gpu[N]!, gpu[N + 1]!, gpu[N + 2]!],
    ...(firstBad ? { firstBad } : {}),
  };
}

main()
  .catch((e: unknown) => {
    result.error = e instanceof Error ? `${e.message}\n${e.stack ?? ""}` : String(e);
    log(result.error, "fail");
  })
  .finally(() => {
    result.done = true;
  });
