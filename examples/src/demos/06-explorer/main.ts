/**
 * Demo 06 — explorer: drag-drop a `.vdb` OR `.nvdb` and inspect it (Phase 5/8
 * demo gate half 1, docs/archive/v1/PLAN.md — "drag-drop a .vdb and render it").
 *
 * Pipeline (SPEC §4/§3.3):
 *   - `.vdb` -> `parseVdb` (pure-TS OpenVDB container reader) -> `buildFromVdb`
 *     (streams parser leaves straight into the NanoVDB serializer, FLOAT).
 *   - `.nvdb` -> `NvdbFile` (the proven Phase 1 loader).
 *
 * Either way the result is one NanoVDB image plus its metadata (FLOAT/Fp8/FpN,
 * SPEC §2.1), which feeds every panel below:
 *   - metadata table (name/type/class/voxel count/bboxes/voxel size/sizes)
 *   - `vdbInspect()`'s node-count + memory-by-section bars
 *   - a 256-bin histogram (`gridStats`, the core compute pass over the grid's
 *     `VDBGridPool` copy through the three wrapper; a ~100k-sample CPU scan if
 *     it fails)
 *   - LEAF-node bbox wireframes (`./leaf-walk.ts`, capped at 5000, drawn over
 *     the rendered volume)
 *   - a CPU-only axial Z-slice view (`readValue`, no GPU involved)
 *
 * ## Device, pool and material (K2)
 *
 * The page creates the `GPUDevice` and passes it to `WebGPURenderer` (D8). The
 * loaded grid lives in one `VDBGridPool` slot (`pool.replace` on each load),
 * which `gridStats` reads and `VDBVolumeMaterial` marches through one
 * `VDBGridNode` (profile `sample`, helper `rayClipSupport`, module types fixed
 * up front so any reload reuses it). The proxy box and the wireframe's
 * index-to-world affine come from `pool.info(id)`.
 *
 * ## Presentation
 *
 * Default (`?presentation=direct`): three renders into the page's canvas.
 * `?presentation=readback` renders into an offscreen `RenderTarget`, reads it
 * back and blits it to a 2D canvas; the e2e spec uses it because headless
 * SwiftShader drops the Dawn instance when a WebGPU canvas context is
 * configured (DIVERGES V7). The swizzle-string and adapter keep-alive shims
 * below are the same as demos 02/04.
 *
 * ## Deterministic test mode (`?test=1&sample=vdb|nvdb`)
 *
 * Auto-loads the named sample (`sphere.vdb` or `sphere_fog_fp8.nvdb`),
 * renders one settled frame, fills every panel, and publishes
 * `window.__DEMO06__`.
 */
import "../webgpu-gate.js";
import { fixtureUrl } from "../assets.js";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { NvdbFile, vdbReadValueCpu } from "@pmndrs/vdb/tools";
import type { NvdbGridMetadata } from "@pmndrs/vdb/tools";
import { VDBGridPool } from "@pmndrs/vdb";
import type { VDBGridInfo } from "@pmndrs/vdb";
import { VDBGridNode, VDBVolumeMaterial, getDevice, vdbGridStats, vdbProxyGeometry } from "@pmndrs/vdb/three";
import { WebGPURenderer } from "three/webgpu";
import { buildFromVdbDetailed, vdbInspect, parseVdb } from "@pmndrs/vdb/tools";
import type { VDBInspectReport } from "@pmndrs/vdb/tools";

import { walkLeafOrigins } from "./leaf-walk";

/** One loaded grid: its NanoVDB image and the file/builder metadata the panels show. */
interface ExplorerGrid {
  image: Uint32Array;
  metadata: NvdbGridMetadata;
}

interface Demo06State {
  ready: boolean;
  presentation?: "direct" | "readback";
  source?: "vdb" | "nvdb";
  voxelCount?: number;
  leafCount?: number;
  histogramNonEmpty?: boolean;
  /** Where the histogram came from: the GPU `gridStats` pass or the CPU sampler fallback. */
  histogramFrom?: "gpu" | "cpu";
  sliceNonEmpty?: boolean;
  error?: string;
}

declare global {
  interface Window {
    __DEMO06__?: Demo06State;
  }
}

// ---------------------------------------------------------------------------
// Compatibility shims (identical to demo 02/04 — see their module headers).
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Constants / DOM
// ---------------------------------------------------------------------------

const SAMPLE_URLS = {
  vdb: fixtureUrl("vdb-samples/sphere.vdb"),
  nvdb: fixtureUrl("primitives/sphere_fog_fp8.nvdb"),
} as const;

const HISTOGRAM_BINS = 256;
const CPU_HISTOGRAM_SAMPLES = 100_000;
const SLICE_MAX_DIM = 256;
const LEAF_CAP = 5000;

const params = new URLSearchParams(location.search);
const TEST_MODE = params.get("test") === "1";
/** `?presentation=readback` renders offscreen and blits through a 2D canvas (the e2e spec uses it: SwiftShader has no WebGPU canvas). */
const PRESENTATION: "direct" | "readback" = params.get("presentation") === "readback" ? "readback" : "direct";
const TEST_SAMPLE: "vdb" | "nvdb" = params.get("sample") === "vdb" ? "vdb" : "nvdb";

const appEl = document.querySelector<HTMLDivElement>("#app")!;
const panelEl = document.querySelector<HTMLDivElement>("#panel")!;
const statsEl = document.querySelector<HTMLDivElement>("#stats")!;
const errEl = document.querySelector<HTMLDivElement>("#err")!;
const dropzoneEl = document.querySelector<HTMLDivElement>("#dropzone")!;
const fileInputEl = document.querySelector<HTMLInputElement>("#file-input")!;
const sampleVdbBtn = document.querySelector<HTMLButtonElement>("#sample-vdb")!;
const sampleNvdbBtn = document.querySelector<HTMLButtonElement>("#sample-nvdb")!;
const metaEl = document.querySelector<HTMLDivElement>("#meta")!;
const inspectEl = document.querySelector<HTMLDivElement>("#inspect-bars")!;
const histCanvas = document.querySelector<HTMLCanvasElement>("#hist-canvas")!;
const sliceCanvas = document.querySelector<HTMLCanvasElement>("#slice-canvas")!;
const sliceZInput = document.querySelector<HTMLInputElement>("#slice-z")!;
const leafNoteEl = document.querySelector<HTMLDivElement>("#leaf-note")!;

function fail(message: string): void {
  errEl.textContent = message;
  statsEl.textContent = "failed";
  window.__DEMO06__ = { ready: false, error: message };
}

// ---------------------------------------------------------------------------
// Load + route by extension/magic (SPEC §4 / §2.2)
// ---------------------------------------------------------------------------

function detectFormat(buffer: ArrayBuffer, filename: string): "vdb" | "nvdb" {
  const lower = filename.toLowerCase();
  if (lower.endsWith(".vdb")) return "vdb";
  if (lower.endsWith(".nvdb")) return "nvdb";
  const head = new Uint8Array(buffer, 0, Math.min(8, buffer.byteLength));
  const ascii = String.fromCharCode(...head);
  return ascii.startsWith("NanoVDB") ? "nvdb" : "vdb";
}

interface LoadedSource {
  grid: ExplorerGrid;
  source: "vdb" | "nvdb";
}

/** `.vdb` -> parseVdb -> buildFromVdb (FLOAT); `.nvdb` -> NvdbFile — either way, one `ExplorerGrid`. */
function loadFromArrayBuffer(buffer: ArrayBuffer, format: "vdb" | "nvdb"): LoadedSource {
  if (format === "vdb") {
    const vdbFile = parseVdb(buffer);
    const vdbGrid = vdbFile.grids[0];
    if (!vdbGrid) throw new Error("this .vdb file contains no grids");
    const built = buildFromVdbDetailed(vdbGrid);
    const metadata: NvdbGridMetadata = {
      name: built.gridName,
      gridType: "Float",
      gridClass: "FogVolume",
      worldBBox: built.worldBBox,
      indexBBox: built.indexBBox,
      voxelSize: [built.voxelSize, built.voxelSize, built.voxelSize],
      voxelCount: built.voxelCount,
      gridByteSize: built.image.byteLength,
    };
    return { grid: { image: built.image, metadata }, source: "vdb" };
  }
  const file = NvdbFile.fromArrayBuffer(buffer);
  const metadata = file.grids[0];
  if (!metadata) throw new Error("this .nvdb file contains no grids");
  return { grid: { image: file.gridImage(0), metadata }, source: "nvdb" };
}

// ---------------------------------------------------------------------------
// Panel: metadata table
// ---------------------------------------------------------------------------

function fmtVec3(v: readonly [number, number, number], digits = 3): string {
  return `[${v.map((x) => x.toFixed(digits)).join(", ")}]`;
}

function renderMeta(grid: ExplorerGrid, fileByteLength: number): void {
  const m = grid.metadata;
  const rows: Array<[string, string]> = [
    ["name", m.name || "(unnamed)"],
    ["grid type", m.gridType],
    ["grid class", m.gridClass],
    ["voxel count", m.voxelCount.toLocaleString()],
    ["index bbox", `${fmtVec3(m.indexBBox.min, 0)} .. ${fmtVec3(m.indexBBox.max, 0)}`],
    ["world bbox", `${fmtVec3(m.worldBBox.min)} .. ${fmtVec3(m.worldBBox.max)}`],
    ["voxel size", fmtVec3(m.voxelSize, 5)],
    ["file size", `${(fileByteLength / 1024).toFixed(1)} KB`],
    ["grid size", `${(m.gridByteSize / 1024).toFixed(1)} KB`],
  ];
  metaEl.innerHTML = "";
  const table = document.createElement("table");
  for (const [k, v] of rows) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = k;
    const td = document.createElement("td");
    td.textContent = v;
    tr.append(th, td);
    table.appendChild(tr);
  }
  metaEl.appendChild(table);
}

// ---------------------------------------------------------------------------
// Panel: inspect() breakdown as a plain-DOM bar list
// ---------------------------------------------------------------------------

function barRow(label: string, value: number, total: number): HTMLDivElement {
  const row = document.createElement("div");
  row.className = "bar-row";
  const bar = document.createElement("div");
  bar.className = "bar";
  bar.style.width = `${total > 0 ? Math.max(1, (value / total) * 100) : 1}%`;
  const span = document.createElement("span");
  span.textContent = `${label}: ${value.toLocaleString()}`;
  row.append(bar, span);
  return row;
}

function renderInspect(report: VDBInspectReport): void {
  inspectEl.innerHTML = "";

  const nodeTitle = document.createElement("div");
  nodeTitle.className = "bar-title";
  nodeTitle.textContent = "node counts";
  inspectEl.appendChild(nodeTitle);
  const nodeTotal = report.nodeCounts.upper + report.nodeCounts.lower + report.nodeCounts.leaf;
  inspectEl.appendChild(barRow("upper", report.nodeCounts.upper, nodeTotal));
  inspectEl.appendChild(barRow("lower", report.nodeCounts.lower, nodeTotal));
  inspectEl.appendChild(barRow("leaf", report.nodeCounts.leaf, nodeTotal));

  const memTitle = document.createElement("div");
  memTitle.className = "bar-title";
  memTitle.textContent = "memory (bytes)";
  inspectEl.appendChild(memTitle);
  const memTotal = report.memoryBreakdown["total"] ?? 1;
  for (const [key, value] of Object.entries(report.memoryBreakdown)) {
    if (key === "total") continue;
    inspectEl.appendChild(barRow(key, value, memTotal));
  }
}

// ---------------------------------------------------------------------------
// Panel: histogram (GPU gridStats, CPU sample fallback)
// ---------------------------------------------------------------------------

function renderHistogram(histogram: Uint32Array): void {
  const ctx = histCanvas.getContext("2d")!;
  const w = histCanvas.width;
  const h = histCanvas.height;
  ctx.clearRect(0, 0, w, h);
  let max = 1;
  for (const v of histogram) if (v > max) max = v;
  const barW = w / histogram.length;
  ctx.fillStyle = "#9db4ff";
  for (let i = 0; i < histogram.length; i++) {
    const barH = (histogram[i]! / max) * h;
    ctx.fillRect(i * barW, h - barH, Math.max(1, barW - 0.5), barH);
  }
}

/** ~100k-sample CPU scan, used when WebGPU (or `gridStats` itself) is unavailable. */
function cpuHistogramFallback(grid: ExplorerGrid): Uint32Array {
  const [minX, minY, minZ] = grid.metadata.indexBBox.min;
  const [maxX, maxY, maxZ] = grid.metadata.indexBBox.max;
  const sizeX = maxX - minX + 1;
  const sizeY = maxY - minY + 1;
  const sizeZ = maxZ - minZ + 1;

  let vmin = Number.POSITIVE_INFINITY;
  let vmax = Number.NEGATIVE_INFINITY;
  const values: number[] = [];
  for (let i = 0; i < CPU_HISTOGRAM_SAMPLES; i++) {
    const x = minX + Math.floor(Math.random() * sizeX);
    const y = minY + Math.floor(Math.random() * sizeY);
    const z = minZ + Math.floor(Math.random() * sizeZ);
    const r = vdbReadValueCpu(grid.image, [x, y, z]);
    if (!r.active) continue;
    values.push(r.value);
    if (r.value < vmin) vmin = r.value;
    if (r.value > vmax) vmax = r.value;
  }

  const histogram = new Uint32Array(HISTOGRAM_BINS);
  if (values.length === 0) return histogram;
  const range = vmax > vmin ? vmax - vmin : 1;
  for (const v of values) {
    const bin = Math.min(HISTOGRAM_BINS - 1, Math.floor(((v - vmin) / range) * HISTOGRAM_BINS));
    histogram[bin]!++;
  }
  return histogram;
}

/** GPU `gridStats` on the pool's copy of the grid; the CPU sampler if that fails. */
async function computeHistogram(grid: ExplorerGrid): Promise<{ histogram: Uint32Array; from: "gpu" | "cpu" }> {
  if (renderer && pool && gridId !== undefined) {
    try {
      const result = await vdbGridStats(renderer, pool, gridId, { histogramBins: HISTOGRAM_BINS });
      return { histogram: result.histogram, from: "gpu" };
    } catch (err) {
      // Fall through to the CPU sampler below (defensive; gridStats is
      // expected to succeed whenever the pool holds the grid).
      console.warn("gridStats failed; using the CPU histogram sampler", err);
    }
  }
  return { histogram: cpuHistogramFallback(grid), from: "cpu" };
}

// ---------------------------------------------------------------------------
// Panel: axial Z-slice (pure CPU, no GPU involved)
// ---------------------------------------------------------------------------

function renderSlice(grid: ExplorerGrid, z: number): boolean {
  const [minX, minY] = grid.metadata.indexBBox.min;
  const [maxX, maxY] = grid.metadata.indexBBox.max;
  const w = maxX - minX + 1;
  const h = maxY - minY + 1;
  const strideX = Math.max(1, Math.ceil(w / SLICE_MAX_DIM));
  const strideY = Math.max(1, Math.ceil(h / SLICE_MAX_DIM));
  const outW = Math.max(1, Math.ceil(w / strideX));
  const outH = Math.max(1, Math.ceil(h / strideY));

  sliceCanvas.width = outW;
  sliceCanvas.height = outH;
  const ctx = sliceCanvas.getContext("2d")!;
  const img = ctx.createImageData(outW, outH);

  const values = new Float32Array(outW * outH);
  const actives = new Uint8Array(outW * outH);
  let vmin = Number.POSITIVE_INFINITY;
  let vmax = Number.NEGATIVE_INFINITY;
  let nonEmpty = false;

  for (let oy = 0; oy < outH; oy++) {
    const y = minY + oy * strideY;
    for (let ox = 0; ox < outW; ox++) {
      const x = minX + ox * strideX;
      const r = vdbReadValueCpu(grid.image, [x, y, z]);
      const idx = oy * outW + ox;
      values[idx] = r.value;
      if (r.active) {
        actives[idx] = 1;
        nonEmpty = true;
        if (r.value < vmin) vmin = r.value;
        if (r.value > vmax) vmax = r.value;
      }
    }
  }

  const range = vmax > vmin ? vmax - vmin : 1;
  for (let i = 0; i < outW * outH; i++) {
    // Flip vertically (canvas rows go top-down; +Y should read "up").
    const row = outH - 1 - Math.floor(i / outW);
    const col = i % outW;
    const srcIdx = row * outW + col;
    const gray = actives[srcIdx] ? Math.round(((values[srcIdx]! - vmin) / range) * 255) : 0;
    img.data[i * 4 + 0] = gray;
    img.data[i * 4 + 1] = gray;
    img.data[i * 4 + 2] = gray;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return nonEmpty;
}

// ---------------------------------------------------------------------------
// Panel: LEAF-node bbox wireframes over the rendered volume
// ---------------------------------------------------------------------------

const WIREFRAME_EDGES: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 3],
  [3, 2],
  [2, 0],
  [4, 5],
  [5, 7],
  [7, 6],
  [6, 4],
  [0, 4],
  [1, 5],
  [2, 6],
  [3, 7],
];

function buildLeafWireframe(
  grid: ExplorerGrid,
  info: VDBGridInfo,
  cap: number,
): { lines: THREE.LineSegments; material: THREE.LineBasicMaterial; shown: number; total: number } {
  const { origins, total } = walkLeafOrigins(grid.image, info.gridTypeId, cap);
  const LEAF_DIM = 8;
  const positions = new Float32Array(origins.length * WIREFRAME_EDGES.length * 2 * 3);
  const a = info.indexToWorld; // 3x4 row-major index -> world
  const m = new THREE.Matrix4().set(
    a[0]!, a[1]!, a[2]!, a[3]!,
    a[4]!, a[5]!, a[6]!, a[7]!,
    a[8]!, a[9]!, a[10]!, a[11]!,
    0, 0, 0, 1,
  );
  let ptr = 0;

  const corners: THREE.Vector3[] = Array.from({ length: 8 }, () => new THREE.Vector3());
  for (const [ox, oy, oz] of origins) {
    const x1 = ox + LEAF_DIM;
    const y1 = oy + LEAF_DIM;
    const z1 = oz + LEAF_DIM;
    corners[0]!.set(ox, oy, oz).applyMatrix4(m);
    corners[1]!.set(x1, oy, oz).applyMatrix4(m);
    corners[2]!.set(ox, y1, oz).applyMatrix4(m);
    corners[3]!.set(x1, y1, oz).applyMatrix4(m);
    corners[4]!.set(ox, oy, z1).applyMatrix4(m);
    corners[5]!.set(x1, oy, z1).applyMatrix4(m);
    corners[6]!.set(ox, y1, z1).applyMatrix4(m);
    corners[7]!.set(x1, y1, z1).applyMatrix4(m);
    for (const [a, b] of WIREFRAME_EDGES) {
      const ca = corners[a]!;
      const cb = corners[b]!;
      positions[ptr++] = ca.x;
      positions[ptr++] = ca.y;
      positions[ptr++] = ca.z;
      positions[ptr++] = cb.x;
      positions[ptr++] = cb.y;
      positions[ptr++] = cb.z;
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({ color: 0x5ee6c9, transparent: true, opacity: 0.4 });
  return { lines: new THREE.LineSegments(geometry, material), material, shown: origins.length, total };
}

// ---------------------------------------------------------------------------
// Renderer bootstrap (device-first, created once and reused across loads)
// ---------------------------------------------------------------------------

let renderer: WebGPURenderer | undefined;
let pool: VDBGridPool | undefined;
let gridId: number | undefined;
let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let controls: OrbitControls;
let display: HTMLCanvasElement;
let ctx2d: CanvasRenderingContext2D;
let rt: THREE.RenderTarget;
let flip = new Uint8ClampedArray(0);
let vw = 0;
let vh = 0;

let currentMesh: THREE.Mesh | undefined;
let currentMeshMaterial: VDBVolumeMaterial | undefined;
let grids: VDBGridNode | undefined;
let currentWireframe: THREE.LineSegments | undefined;
let currentWireframeMaterial: THREE.LineBasicMaterial | undefined;
let currentGrid: ExplorerGrid | undefined;

/** Grid types the one shared module decodes: everything the explorer loads (`.vdb` builds Float). */
const MODULE_TYPES = ["float", "fp8", "fpn"] as const;

/** Optional device features requested when the adapter has them. */
const WANTED_FEATURES: readonly GPUFeatureName[] = ["shader-f16", "float32-filterable"];

async function initRenderer(): Promise<void> {
  // Device first (D8): the page owns the device; three and the pool share it.
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter.");
  const device = await adapter.requestDevice({
    requiredLimits: VDBGridPool.requiredLimits(adapter),
    requiredFeatures: WANTED_FEATURES.filter((f) => adapter.features.has(f)),
  });
  display = document.createElement("canvas");
  renderer = new WebGPURenderer({ device, antialias: false, ...(PRESENTATION === "direct" ? { canvas: display } : {}) });
  await renderer.init();
  pool = await VDBGridPool.create(getDevice(renderer), { label: "demo 06" });
  // One module for the page (types fixed up front so a reload of any type reuses it).
  grids = new VDBGridNode(renderer, pool, { types: MODULE_TYPES, profile: "sample", helpers: ["rayClipSupport"] });
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b1020);
  camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);

  display.id = "view";
  display.style.width = "100%";
  display.style.height = "100%";
  display.style.display = "block";
  appEl.appendChild(display);
  if (PRESENTATION === "readback") ctx2d = display.getContext("2d")!;

  controls = new OrbitControls(camera, display);
  controls.enableDamping = !TEST_MODE;

  rt = new THREE.RenderTarget(2, 2);
  rt.texture.colorSpace = THREE.SRGBColorSpace;
}

function setViewport(w: number, h: number): void {
  vw = Math.max(1, w);
  vh = Math.max(1, h);
  if (PRESENTATION === "direct") renderer!.setSize(vw, vh, false);
  else {
    display.width = vw;
    display.height = vh;
    rt.setSize(vw, vh);
    flip = new Uint8ClampedArray(vw * vh * 4);
  }
  camera.aspect = vw / vh;
  camera.updateProjectionMatrix();
}

function placeCamera(
  center: THREE.Vector3,
  diag: number,
  azimuth: number,
  elevation: number,
  radiusFactor: number,
): void {
  camera.near = Math.max(0.001, diag * 0.001);
  camera.far = diag * 100;
  const r = diag * radiusFactor;
  camera.position.set(
    center.x + r * Math.cos(elevation) * Math.cos(azimuth),
    center.y + r * Math.sin(elevation),
    center.z + r * Math.cos(elevation) * Math.sin(azimuth),
  );
  camera.updateProjectionMatrix();
  camera.lookAt(center);
  controls.target.copy(center);
  controls.update();
}

async function presentFrame(): Promise<void> {
  // The pool's update is already submitted (flush); rebind after any growth.
  grids!.update();
  if (PRESENTATION === "direct") {
    renderer!.setRenderTarget(null);
    await renderer!.renderAsync(scene, camera);
    return;
  }
  renderer!.setRenderTarget(rt);
  await renderer!.renderAsync(scene, camera);
  const pixels = (await renderer!.readRenderTargetPixelsAsync(rt, 0, 0, vw, vh)) as Uint8Array;
  renderer!.setRenderTarget(null);
  const stride = vw * 4;
  for (let y = 0; y < vh; y++) {
    const src = (vh - 1 - y) * stride;
    flip.set(pixels.subarray(src, src + stride), y * stride);
  }
  ctx2d.putImageData(new ImageData(flip, vw, vh), 0, 0);
}

// ---------------------------------------------------------------------------
// Orchestration: load a file (or sample), fill every panel, render
// ---------------------------------------------------------------------------

async function loadAndDisplay(buffer: ArrayBuffer, filename: string): Promise<void> {
  errEl.textContent = "";
  statsEl.textContent = `loading ${filename}…`;

  const format = detectFormat(buffer, filename);
  const { grid, source } = loadFromArrayBuffer(buffer, format);
  const report = vdbInspect(grid.image);

  renderMeta(grid, buffer.byteLength);
  renderInspect(report);

  // One pool slot for the loaded grid: replaced in place when it fits.
  if (gridId === undefined) gridId = pool!.add(grid.image, { label: filename });
  else pool!.replace(gridId, grid.image);
  pool!.flush();
  await pool!.settled();
  const info = pool!.info(gridId);

  const { histogram, from: histogramFrom } = await computeHistogram(grid);
  renderHistogram(histogram);
  const histogramNonEmpty = histogram.some((v) => v > 0);

  const { lines, material: wireframeMaterial, shown, total } = buildLeafWireframe(grid, info, LEAF_CAP);
  leafNoteEl.textContent =
    total > shown
      ? `leaf bboxes: showing ${shown.toLocaleString()} of ${total.toLocaleString()} (capped at ${LEAF_CAP.toLocaleString()})`
      : `leaf bboxes: ${shown.toLocaleString()}`;

  if (currentMesh) {
    scene.remove(currentMesh);
    currentMesh.geometry.dispose();
    currentMeshMaterial?.dispose();
  }
  if (currentWireframe) {
    scene.remove(currentWireframe);
    currentWireframe.geometry.dispose();
    currentWireframeMaterial?.dispose();
  }

  grids!.update();
  const volumeMaterial = new VDBVolumeMaterial(renderer!, grids!, gridId, { jitter: !TEST_MODE });
  const mesh = new THREE.Mesh(vdbProxyGeometry(info), volumeMaterial);
  scene.add(mesh);
  scene.add(lines);
  currentMesh = mesh;
  currentMeshMaterial = volumeMaterial;
  currentWireframe = lines;
  currentWireframeMaterial = wireframeMaterial;
  currentGrid = grid;

  const worldBox = new THREE.Box3(
    new THREE.Vector3(...grid.metadata.worldBBox.min),
    new THREE.Vector3(...grid.metadata.worldBBox.max),
  );
  const center = worldBox.getCenter(new THREE.Vector3());
  const size = worldBox.getSize(new THREE.Vector3());
  const diag = Math.max(size.length(), 1e-3);
  placeCamera(center, diag, Math.PI * 0.25, Math.PI * 0.2, 1.8);

  const [, , minZ] = grid.metadata.indexBBox.min;
  const [, , maxZ] = grid.metadata.indexBBox.max;
  sliceZInput.min = String(minZ);
  sliceZInput.max = String(maxZ);
  const midZ = Math.round((minZ + maxZ) / 2);
  sliceZInput.value = String(midZ);
  sliceZInput.disabled = false;
  const sliceNonEmpty = renderSlice(grid, midZ);

  statsEl.textContent =
    `${filename} (${source})\n` +
    `type: ${grid.metadata.gridType}  ·  voxels: ${grid.metadata.voxelCount.toLocaleString()}  ·  ` +
    `leaves: ${report.nodeCounts.leaf.toLocaleString()}`;

  const state: Demo06State = {
    ready: true,
    presentation: PRESENTATION,
    source,
    voxelCount: grid.metadata.voxelCount,
    leafCount: report.nodeCounts.leaf,
    histogramNonEmpty,
    histogramFrom,
    sliceNonEmpty,
  };

  if (TEST_MODE) {
    setViewport(640, 480);
    await presentFrame();
    await presentFrame();
  }

  window.__DEMO06__ = state;
}

// ---------------------------------------------------------------------------
// Interactive wiring (drag-drop, file input, sample buttons, slice slider)
// ---------------------------------------------------------------------------

let loading = false;

async function handleFile(file: File): Promise<void> {
  if (loading) return;
  loading = true;
  try {
    const buffer = await file.arrayBuffer();
    await loadAndDisplay(buffer, file.name);
  } catch (err) {
    fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));
  } finally {
    loading = false;
  }
}

async function handleSample(url: string): Promise<void> {
  if (loading) return;
  loading = true;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`failed to fetch ${url} (${res.status})`);
    const buffer = await res.arrayBuffer();
    await loadAndDisplay(buffer, url.split("/").pop() ?? url);
  } catch (err) {
    fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));
  } finally {
    loading = false;
  }
}

function wireInteractiveControls(): void {
  dropzoneEl.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzoneEl.classList.add("drag-over");
  });
  dropzoneEl.addEventListener("dragleave", () => {
    dropzoneEl.classList.remove("drag-over");
  });
  dropzoneEl.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzoneEl.classList.remove("drag-over");
    const file = e.dataTransfer?.files?.[0];
    if (file) void handleFile(file);
  });
  fileInputEl.addEventListener("change", () => {
    const file = fileInputEl.files?.[0];
    if (file) void handleFile(file);
  });
  sampleVdbBtn.addEventListener("click", () => void handleSample(SAMPLE_URLS.vdb));
  sampleNvdbBtn.addEventListener("click", () => void handleSample(SAMPLE_URLS.nvdb));
  sliceZInput.addEventListener("input", () => {
    if (!currentGrid) return;
    renderSlice(currentGrid, Number(sliceZInput.value));
  });
}

let frameBusy = false;
function startRenderLoop(): void {
  const loop = (): void => {
    requestAnimationFrame(loop);
    if (frameBusy || !currentMesh || window.__DEMO06__?.error) return;
    frameBusy = true;
    controls.update();
    presentFrame()
      .then(() => {
        frameBusy = false;
      })
      .catch((err: unknown) => {
        frameBusy = false;
        fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));
      });
  };
  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function run(): Promise<void> {
  if (!navigator.gpu) {
    fail("WebGPU unavailable (navigator.gpu undefined) — needs a secure context + adapter.");
    return;
  }
  installSwizzleCompatShim();
  installAdapterKeepAliveShim();
  await initRenderer();

  if (TEST_MODE) {
    panelEl.classList.add("test-hidden");
    const url = SAMPLE_URLS[TEST_SAMPLE];
    const res = await fetch(url);
    if (!res.ok) {
      fail(`fixture missing: ${url} (${res.status})`);
      return;
    }
    const buffer = await res.arrayBuffer();
    await loadAndDisplay(buffer, url.split("/").pop() ?? url);
    return;
  }

  wireInteractiveControls();
  setViewport(window.innerWidth, window.innerHeight);
  window.addEventListener("resize", () => setViewport(window.innerWidth, window.innerHeight));
  startRenderLoop();
}

run().catch((err: unknown) => {
  fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));
});
