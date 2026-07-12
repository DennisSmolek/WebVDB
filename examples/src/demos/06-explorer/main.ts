/**
 * Demo 06 — explorer: drag-drop a `.vdb` OR `.nvdb` and inspect it (Phase 5/8
 * demo gate half 1, docs/PLAN.md — "drag-drop a .vdb and render it").
 *
 * Pipeline (SPEC §4/§3.3):
 *   - `.vdb` -> `parseVdb` (pure-TS OpenVDB container reader) -> `buildFromVdb`
 *     (streams parser leaves straight into the NanoVDB serializer, FLOAT) ->
 *     wrapped in a `NanoVDBGrid` for GPU use.
 *   - `.nvdb` -> `NanoVDBFile` (the proven Phase 1 loader) -> `NanoVDBGrid`.
 *
 * Either way the result is one `NanoVDBGrid` (FLOAT/Fp8/FpN, SPEC §2.1), which
 * feeds every panel below:
 *   - metadata table (name/type/class/voxel count/bboxes/voxel size/sizes)
 *   - `inspect()`'s node-count + memory-by-section bars
 *   - a 256-bin histogram (`gridStats` on the GPU; a ~100k-sample CPU scan if
 *     WebGPU isn't available)
 *   - LEAF-node bbox wireframes (`./leaf-walk.ts`, capped at 5000, drawn over
 *     the rendered volume)
 *   - a CPU-only axial Z-slice view (`readValue`, no GPU involved)
 *
 * ## Presentation & CPU reference: same decisions as demos 02/04
 *
 * Offscreen `RenderTarget` -> `readRenderTargetPixelsAsync` -> 2D-canvas blit
 * (canvas *presentation* drops the Dawn instance under this sandbox's
 * SwiftShader — see docs/handoffs/PHASE-3.md), the same r185 swizzle-string
 * shim + adapter keep-alive shim, and `nanovdb-wgsl`'s own package-exported
 * `readValue` (its `cpu/*` used to load `stride-tables.json` via `node:fs` —
 * not importable into a Vite page; see docs/handoffs/PHASE-5.md's "Known
 * debts" for the dedup that fixed this). `NodeMaterial` still needs >=1
 * light in the scene for its lighting codepath to run at all (see
 * docs/handoffs/PHASE-4.md) — moot here since `NanoVDBVolumeMaterial` is a
 * from-scratch fragment raymarch, not a stock `NodeMaterial` lighting model,
 * but noted for anyone extending this demo toward `VolumeNodeMaterial`.
 *
 * ## Deterministic test mode (`?test=1&sample=vdb|nvdb`)
 *
 * Auto-loads the named sample (`smoke.vdb` or `sphere_fog_fp8.nvdb`),
 * renders one settled frame, fills every panel, and publishes
 * `window.__DEMO06__`.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { NanoVDBFile, readValue } from "nanovdb-wgsl";
import type { GridMetadata } from "nanovdb-wgsl";
import pnanovdbSource from "nanovdb-wgsl/pnanovdb.wgsl?raw";
import { NanoVDBGrid, NanoVDBVolumeMaterial, createVolumeRenderer, gridStats } from "three-nanovdb";
import type { WebGPURenderer } from "three/webgpu";
import { buildFromVdbDetailed, inspect, parseVdb } from "vdb-web-tools";
import type { InspectReport } from "vdb-web-tools";

import { walkLeafOrigins } from "./leaf-walk";

interface Demo06State {
  ready: boolean;
  source?: "vdb" | "nvdb";
  voxelCount?: number;
  leafCount?: number;
  histogramNonEmpty?: boolean;
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
  vdb: "/fixtures/vdb-samples/smoke.vdb",
  nvdb: "/fixtures/primitives/sphere_fog_fp8.nvdb",
} as const;

const HISTOGRAM_BINS = 256;
const CPU_HISTOGRAM_SAMPLES = 100_000;
const SLICE_MAX_DIM = 256;
const LEAF_CAP = 5000;

const params = new URLSearchParams(location.search);
const TEST_MODE = params.get("test") === "1";
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
  grid: NanoVDBGrid;
  source: "vdb" | "nvdb";
}

/** `.vdb` -> parseVdb -> buildFromVdb (FLOAT); `.nvdb` -> NanoVDBFile — either way, one `NanoVDBGrid`. */
function loadFromArrayBuffer(buffer: ArrayBuffer, format: "vdb" | "nvdb"): LoadedSource {
  if (format === "vdb") {
    const vdbFile = parseVdb(buffer);
    const vdbGrid = vdbFile.grids[0];
    if (!vdbGrid) throw new Error("this .vdb file contains no grids");
    const built = buildFromVdbDetailed(vdbGrid);
    const metadata: GridMetadata = {
      name: built.gridName,
      gridType: "Float",
      gridClass: "FogVolume",
      worldBBox: built.worldBBox,
      indexBBox: built.indexBBox,
      voxelSize: [built.voxelSize, built.voxelSize, built.voxelSize],
      voxelCount: built.voxelCount,
      gridByteSize: built.image.byteLength,
    };
    return { grid: new NanoVDBGrid({ image: built.image, metadata }), source: "vdb" };
  }
  const file = NanoVDBFile.fromArrayBuffer(buffer);
  return { grid: NanoVDBGrid.fromFile(file, 0), source: "nvdb" };
}

// ---------------------------------------------------------------------------
// Panel: metadata table
// ---------------------------------------------------------------------------

function fmtVec3(v: readonly [number, number, number], digits = 3): string {
  return `[${v.map((x) => x.toFixed(digits)).join(", ")}]`;
}

function renderMeta(grid: NanoVDBGrid, fileByteLength: number): void {
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

function renderInspect(report: InspectReport): void {
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
function cpuHistogramFallback(grid: NanoVDBGrid): Uint32Array {
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
    const r = readValue(grid.image, [x, y, z]);
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

async function computeHistogram(device: GPUDevice | undefined, grid: NanoVDBGrid): Promise<Uint32Array> {
  if (device) {
    try {
      const result = await gridStats(device, grid, pnanovdbSource, { histogramBins: HISTOGRAM_BINS });
      return result.histogram;
    } catch {
      // Fall through to the CPU sampler below (defensive; gridStats is expected
      // to succeed whenever `device` exists).
    }
  }
  return cpuHistogramFallback(grid);
}

// ---------------------------------------------------------------------------
// Panel: axial Z-slice (pure CPU, no GPU involved)
// ---------------------------------------------------------------------------

function renderSlice(grid: NanoVDBGrid, z: number): boolean {
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
      const r = readValue(grid.image, [x, y, z]);
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
  grid: NanoVDBGrid,
  cap: number,
): { lines: THREE.LineSegments; material: THREE.LineBasicMaterial; shown: number; total: number } {
  const { origins, total } = walkLeafOrigins(grid.image, grid.gridTypeId, cap);
  const LEAF_DIM = 8;
  const positions = new Float32Array(origins.length * WIREFRAME_EDGES.length * 2 * 3);
  const m = grid.indexToWorld();
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
let device: GPUDevice | undefined;
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
let currentMeshMaterial: NanoVDBVolumeMaterial | undefined;
let currentWireframe: THREE.LineSegments | undefined;
let currentWireframeMaterial: THREE.LineBasicMaterial | undefined;
let currentGrid: NanoVDBGrid | undefined;

async function initRenderer(): Promise<void> {
  ({ renderer, device } = await createVolumeRenderer({}));
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b1020);
  camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);

  display = document.createElement("canvas");
  display.id = "view";
  display.style.width = "100%";
  display.style.height = "100%";
  display.style.display = "block";
  appEl.appendChild(display);
  ctx2d = display.getContext("2d")!;

  controls = new OrbitControls(camera, display);
  controls.enableDamping = !TEST_MODE;

  rt = new THREE.RenderTarget(2, 2);
  rt.texture.colorSpace = THREE.SRGBColorSpace;
}

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
  const report = inspect(grid.image);

  renderMeta(grid, buffer.byteLength);
  renderInspect(report);

  const histogram = await computeHistogram(device, grid);
  renderHistogram(histogram);
  const histogramNonEmpty = histogram.some((v) => v > 0);

  const { lines, material: wireframeMaterial, shown, total } = buildLeafWireframe(grid, LEAF_CAP);
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

  const volumeMaterial = new NanoVDBVolumeMaterial({ grid, pnanovdbSource, jitter: !TEST_MODE });
  const mesh = new THREE.Mesh(grid.proxyGeometry(), volumeMaterial);
  scene.add(mesh);
  scene.add(lines);
  currentMesh = mesh;
  currentMeshMaterial = volumeMaterial;
  currentWireframe = lines;
  currentWireframeMaterial = wireframeMaterial;
  currentGrid = grid;

  const worldBox = grid.worldBBox();
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
    source,
    voxelCount: grid.metadata.voxelCount,
    leafCount: report.nodeCounts.leaf,
    histogramNonEmpty,
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
