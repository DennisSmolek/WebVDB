/**
 * Gallery demo 15 — full resolution vs 128³. One asset twice, split by a draggable divider: on the left our sparse
 * NanoVDB grid at full resolution (from the pool, `vdbSparseSource`), on the right a dense `.wvdb` bake of it at his
 * default resolution (128 on the longest side; 64 and 200 on the control), the format trade-off made visible. Each
 * side lists its resolution, GPU bytes, file size and the mean error against the source (`vdbResampleError`, the
 * comparison report's metric, docs/dev/research/INDEX), measured in a worker. Public API only.
 *
 * Each side renders with its own `VDBFrameRenderer` ('volume' lighting by default, see `controls.lighting`; neural
 * scattering, temporal) into its own
 * target; a quad composites them at the divider (direct), or the CPU does after the readback (`?presentation=readback`).
 *
 * Assets: `smoke.vdb` (`pnpm fixtures:vdb-samples`) and the WDAS quarter cloud (`pnpm fixtures:wdas`, Fp8; its 68 MB
 * `.vdb` is the error reference when present). The gallery's `sky-midday-HDR.jpg` lights the scene when present
 * (else a hemisphere light and a gradient sky).
 *
 * Query: the gallery shell's, plus `?asset=wdas|smoke`, `?res=64|128|200`, `?divider=0..1`, `?view=overview|detail`,
 * `?lighting=volume|frame`.
 * `window.__DEMO15__` is the e2e state; `window.__DEMO15_API__.capture()` summarizes the last read-back frame.
 */
import * as THREE from "three/webgpu";
import { select, texture, uniform, uv } from "three/tsl";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { UltraHDRLoader } from "three/addons/loaders/UltraHDRLoader.js";
import { VDBGridPool } from "@pmndrs/vdb";
import { readWvdb, wvdbDenseVolume, type WvdbDenseVolume } from "@pmndrs/vdb/tools";
import {
  VDBFrameRenderer,
  VDBFrameVolume,
  VDBGridNode,
  getDevice,
  vdbDenseTextures,
  vdbSparseSource,
  type VDBThreeDenseSource,
  type VDBThreeSparseSource,
} from "@pmndrs/vdb/three";
import { createDeviceRenderer } from "../device-renderer.js";
import { GALLERY_DIR, mountShowcase, PRESENTATION, Presenter, autoOrbit, formatBytes, params, probe, qualityParam, type FrameSnapshot, type Quality } from "../showcase-shell.js";
import { ASSETS, type AssetId, type ErrorStat, type WorkerRequest, type WorkerResponse } from "./protocol.js";

// --- State the e2e spec reads ------------------------------------------------------------------------------

interface SideStats {
  label: string;
  resolution: string;
  gpuBytes?: number;
  fileBytes?: number;
  fileGz?: number;
  errorPercent?: number;
}

interface Demo15State {
  ready: boolean;
  error?: string;
  frames: number;
  frameMs?: number;
  adapter?: string;
  presentation: "direct" | "readback";
  available?: AssetId[];
  asset?: AssetId;
  resolution?: number;
  divider?: number;
  view?: View;
  hdr?: boolean;
  sparse?: SideStats;
  dense?: SideStats;
  loading: boolean;
  uncaptured: string[];
}

declare global {
  interface Window {
    __DEMO15__?: Demo15State;
    __DEMO15_API__?: { capture(): FrameSnapshot | null };
  }
}

type View = "overview" | "detail";
const RESOLUTIONS = [64, 128, 200] as const;

const state: Demo15State = { ready: false, frames: 0, loading: true, uncaptured: [], presentation: PRESENTATION };
window.__DEMO15__ = state;

const shell = mountShowcase({
  number: "15",
  title: "Full-res vs 128³",
  blurb:
    "The same volume twice. Left: our sparse NanoVDB grid at full resolution, every voxel of the source. Right: a dense .wvdb bake at 128 on the longest side, his default: far smaller because it throws the detail away. The cards give the price of each; drag the divider or zoom in to see it.",
});
shell.onFail((m) => {
  state.error = m;
  state.loading = false;
});

const HDR_URL = `${GALLERY_DIR}sky-midday-HDR.jpg`;
const QUALITY: Record<Quality, { width: number; height: number; depth: number; subSteps: number; shadowSteps: number; shadowResolution: number; pixelRatio: number }> = {
  low: { width: 128, height: 128, depth: 16, subSteps: 8, shadowSteps: 8, shadowResolution: 32, pixelRatio: 1 },
  medium: { width: 256, height: 256, depth: 24, subSteps: 16, shadowSteps: 16, shadowResolution: 64, pixelRatio: 1 },
  high: { width: 512, height: 512, depth: 32, subSteps: 32, shadowSteps: 32, shadowResolution: 96, pixelRatio: Math.min(2, window.devicePixelRatio || 1) },
};

const resParam = Number(params.get("res"));
const viewParam = params.get("view");
const controls = {
  asset: (params.get("asset") === "smoke" ? "smoke" : "wdas") as AssetId,
  resolution: (RESOLUTIONS as readonly number[]).includes(resParam) ? resParam : 128,
  divider: Math.min(0.95, Math.max(0.05, Number(params.get("divider") ?? 0.5))),
  view: (viewParam === "detail" ? "detail" : "overview") as View,
  quality: qualityParam(PRESENTATION === "readback" ? "low" : "medium"),
  /**
   * 'volume' by default here, not the gallery's 'frame': the comparison is about the data, and in 'frame' lighting
   * both sides are lit from the same coarse shadow grid (64 voxels across), so only the silhouettes would differ.
   * In 'volume' lighting each side's shadow march reads its own data: the full grid on the left, the bake on the right.
   */
  lighting: (params.get("lighting") === "frame" ? "frame" : "volume") as "volume" | "frame",
  temporal: params.has("temporal") ? params.get("temporal") === "1" : PRESENTATION === "direct",
};

/**
 * Camera presets per asset, in the placed frame (longest side 100, centred on the origin): the whole volume, and a
 * close look at an edge where the dense bake loses its detail.
 */
const VIEWS: Record<AssetId, Record<View, { position: [number, number, number]; target: [number, number, number] }>> = {
  wdas: { overview: { position: [-76, 18, 128], target: [0, 6, 0] }, detail: { position: [-30, 34, 58], target: [-9, 21, 11] } },
  smoke: { overview: { position: [-66, 24, 138], target: [0, 8, 0] }, detail: { position: [-24, 50, 46], target: [-6, 38, 0] } },
};

const pct = (e: ErrorStat | undefined): string =>
  !e ? "measuring…" : e.percent < 1e-6 ? "0 % (exact)" : `${e.percent.toFixed(e.percent < 1 ? 2 : 1)} %`;

// --- Side cards and the divider (DOM over the canvas) --------------------------------------------------------

const SIDE_CSS = `
.g-side { position: fixed; top: 12px; width: 260px; padding: 9px 12px; font-size: 12px; line-height: 1.45;
  background: rgba(12, 16, 32, 0.72); color: #e6ebff; border-radius: 10px; border: 1px solid rgba(255,255,255,0.12);
  backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); pointer-events: none; font-variant-numeric: tabular-nums; }
.g-side h2 { font-size: 12.5px; font-weight: 600; margin: 0 0 3px; }
.g-side .k { opacity: 0.6; display: inline-block; width: 50px; }
.g-side .dim { opacity: 0.5; font-size: 11px; }
.g-side.left { text-align: left; }
#g-divider { position: fixed; top: 0; bottom: 0; width: 2px; margin-left: -1px; background: rgba(255,255,255,0.85);
  box-shadow: 0 0 6px rgba(0,0,0,0.35); pointer-events: none; }
#g-handle { position: fixed; top: 50%; width: 34px; height: 34px; margin: -17px 0 0 -17px; border-radius: 50%;
  background: rgba(12,16,32,0.8); border: 2px solid rgba(255,255,255,0.9); cursor: ew-resize; touch-action: none;
  display: grid; place-items: center; color: #fff; font: 600 13px system-ui, sans-serif; user-select: none; }
`;

class SideUi {
  readonly divider: HTMLDivElement;
  readonly handle: HTMLDivElement;
  readonly left: HTMLDivElement;
  readonly right: HTMLDivElement;

  constructor(onDrag: (fraction: number) => void) {
    const style = document.createElement("style");
    style.textContent = SIDE_CSS;
    document.head.append(style);
    const make = (tag: string, id: string, cls = ""): HTMLDivElement => {
      const el = document.createElement(tag) as HTMLDivElement;
      if (id) el.id = id;
      if (cls) el.className = cls;
      document.body.append(el);
      return el;
    };
    this.divider = make("div", "g-divider");
    this.handle = make("div", "g-handle");
    this.handle.textContent = "⇔";
    this.handle.dataset.testid = "divider-handle";
    this.left = make("div", "", "g-side left");
    this.left.dataset.testid = "side-sparse";
    this.right = make("div", "", "g-side right");
    this.right.dataset.testid = "side-dense";
    let dragging = false;
    this.handle.addEventListener("pointerdown", (e) => {
      dragging = true;
      this.handle.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    this.handle.addEventListener("pointermove", (e) => {
      if (dragging) onDrag(e.clientX / window.innerWidth);
    });
    const stop = (e: PointerEvent): void => {
      dragging = false;
      if (this.handle.hasPointerCapture(e.pointerId)) this.handle.releasePointerCapture(e.pointerId);
    };
    this.handle.addEventListener("pointerup", stop);
    this.handle.addEventListener("pointercancel", stop);
    if (params.get("panel") === "0") {
      for (const el of [this.left, this.right]) el.style.display = "none";
    }
  }

  place(fraction: number): void {
    const x = fraction * window.innerWidth;
    this.divider.style.left = `${x}px`;
    this.handle.style.left = `${x}px`;
    const gap = 14;
    this.left.style.left = `${Math.max(12, x - gap - 260)}px`;
    this.right.style.left = `${Math.min(window.innerWidth - 272, x + gap)}px`;
  }

  fill(el: HTMLDivElement, title: string, rows: [string, string][], foot: string): void {
    el.replaceChildren();
    const h = document.createElement("h2");
    h.textContent = title;
    el.append(h);
    for (const [k, v] of rows) {
      const row = document.createElement("div");
      const key = document.createElement("span");
      key.className = "k";
      key.textContent = k;
      row.append(key, document.createTextNode(v));
      el.append(row);
    }
    const f = document.createElement("div");
    f.className = "dim";
    f.textContent = foot;
    el.append(f);
  }
}

// --- The page --------------------------------------------------------------------------------------------

interface SparseSide {
  asset: AssetId;
  pool: VDBGridPool;
  grids: VDBGridNode;
  source: VDBThreeSparseSource;
  volume: VDBFrameVolume;
  gridType: string;
  activeVoxels: number;
  indexDims: [number, number, number];
  fileName: string;
  fileBytes: number;
  fileGz?: number;
  error?: ErrorStat;
}

interface DenseSide {
  asset: AssetId;
  resolution: number;
  dense: WvdbDenseVolume;
  source: VDBThreeDenseSource;
  volume: VDBFrameVolume;
  fileBytes: number;
  error?: ErrorStat;
}

async function run(): Promise<void> {
  // --- what is here -------------------------------------------------------------------------------------
  const ids = Object.keys(ASSETS) as AssetId[];
  const found = await Promise.all(ids.map((id) => probe(ASSETS[id].url)));
  const available = ids.filter((_, i) => found[i]);
  state.available = available;
  const hdr = await probe(HDR_URL);
  state.hdr = hdr;
  const missing = [
    ...ids.filter((_, i) => !found[i]).map((id) => `${ASSETS[id].label} (${ASSETS[id].url.replace(/^\//, "")}: ${id === "wdas" ? "pnpm fixtures:wdas, then Fp8" : "pnpm fixtures:vdb-samples"})`),
    ...(hdr ? [] : ["HDR sky (fixtures/gallery/sky-midday-HDR.jpg): hemisphere light and a gradient instead"]),
  ];
  shell.note(missing.length ? `Missing, so not shown:\n${missing.map((m) => `• ${m}`).join("\n")}` : "");
  if (available.length === 0) throw new Error("Neither smoke.vdb nor the WDAS grid is present (pnpm fixtures:vdb-samples, pnpm fixtures:wdas).");
  if (!available.includes(controls.asset)) controls.asset = available[0]!;

  // --- renderer, two frame renderers, two targets --------------------------------------------------------
  const directCanvas = PRESENTATION === "direct" ? document.createElement("canvas") : undefined;
  const created = await createDeviceRenderer(directCanvas ? { canvas: directCanvas } : {});
  const { renderer, device } = created;
  device.addEventListener("uncapturederror", (e) => {
    const message = (e as GPUUncapturedErrorEvent).error.message;
    state.uncaptured.push(message);
    console.error(`WebGPU: ${message}`);
  });
  void device.lost.then((info) => {
    if (info.reason !== "destroyed") shell.fail(`WebGPU device lost: ${info.message}`);
  });
  state.adapter = created.adapterName || "unreported adapter";
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.85;
  const presenter = new Presenter(shell.app, renderer, directCanvas);
  const rtSparse = presenter.target();
  const rtDense = presenter.target();
  // Each side its own froxel renderer: its own settings and its own temporal history.
  const frSparse = new VDBFrameRenderer();
  const frDense = new VDBFrameRenderer();

  // Direct presentation: a quad picks a target per pixel at the divider.
  const dividerUniform = uniform(controls.divider);
  const compositeMaterial = new THREE.MeshBasicNodeMaterial();
  compositeMaterial.colorNode = select(uv().x.lessThan(dividerUniform), texture(rtSparse.texture), texture(rtDense.texture));
  const composite = new THREE.QuadMesh(compositeMaterial);

  // --- scene ----------------------------------------------------------------------------------------------
  const scene = new THREE.Scene();
  const sun = new THREE.DirectionalLight(0xfff4e6, 3.2);
  sun.position.set(120, 160, 60);
  scene.add(sun, sun.target);
  const hemisphere = new THREE.HemisphereLight(0xa9c4ff, 0x4a4036, 0.9);
  scene.add(hemisphere);
  if (hdr) {
    const tex = await new UltraHDRLoader().loadAsync(HDR_URL);
    tex.mapping = THREE.EquirectangularReflectionMapping;
    scene.environment = tex;
    scene.background = tex;
    scene.environmentIntensity = 0.35;
    scene.backgroundBlurriness = 0.35;
    scene.backgroundIntensity = 0.55;
    hemisphere.visible = false;
  } else {
    scene.background = gradientSky();
  }
  const placed = new THREE.Group();
  scene.add(placed);

  // --- camera ---------------------------------------------------------------------------------------------
  const camera = new THREE.PerspectiveCamera(45, 1, 0.5, 5000);
  const orbit = new OrbitControls(camera, presenter.canvas);
  orbit.enableDamping = PRESENTATION === "direct";
  orbit.minDistance = 12;
  orbit.maxDistance = 400;
  autoOrbit(orbit, 0.4);
  const applyView = (): void => {
    const v = VIEWS[controls.asset][controls.view];
    camera.position.set(...v.position);
    orbit.target.set(...v.target);
    orbit.update();
    state.view = controls.view;
  };
  const resize = (): void => {
    presenter.resize(QUALITY[controls.quality].pixelRatio, camera);
    ui.place(controls.divider);
  };

  // --- side cards -----------------------------------------------------------------------------------------
  const ui = new SideUi((f) => setDivider(f));
  let dividerRow: { set(v: number): void } | null = null;
  function setDivider(f: number): void {
    controls.divider = Math.min(0.95, Math.max(0.05, f));
    dividerUniform.value = controls.divider;
    state.divider = controls.divider;
    ui.place(controls.divider);
    dividerRow?.set(controls.divider);
  }

  // --- sides ----------------------------------------------------------------------------------------------
  let sparse: SparseSide | null = null;
  let dense: DenseSide | null = null;
  let renderedSinceLoad = 0;
  const common = { extinction: 100, angstromExponent: 0, normalizeDensity: true } as const;
  const worker = new Worker(new URL("./bake-worker.ts", import.meta.url), { type: "module" });
  const send = (m: WorkerRequest): void => worker.postMessage(m);
  let pending: { resolve: () => void; reject: (e: Error) => void; want: "loaded" | "baked"; asset: AssetId; resolution?: number } | null = null;

  /** Place the shared group by the sparse grid's world box: longest side 100, centred on the origin. */
  function placeGroup(volume: VDBFrameVolume): void {
    const box = volume.bbox;
    const size = box.getSize(new THREE.Vector3());
    const s = 100 / Math.max(size.x, size.y, size.z);
    placed.scale.setScalar(s);
    placed.position.copy(box.getCenter(new THREE.Vector3())).multiplyScalar(-s);
    placed.updateMatrixWorld(true);
  }

  async function onLoaded(m: Extract<WorkerResponse, { type: "loaded" }>): Promise<void> {
    const pool = await VDBGridPool.create(getDevice(renderer), { label: "demo-15" });
    const id = pool.add(m.image);
    pool.flush();
    await pool.settled();
    const grids = new VDBGridNode(renderer, pool, { profile: "march", helpers: ["cellBoundTrilinear", "hddaCellEnd", "nodeAverage", "fastTrilinear"] });
    const source = vdbSparseSource(grids, { density: id }, { lod: "coarse-pyramid", images: { density: m.image } });
    const volume = new VDBFrameVolume(source, { ...common, frameRenderer: frSparse });
    placed.add(volume);
    placeGroup(volume);
    sparse = { asset: m.asset, pool, grids, source, volume, gridType: m.gridType, activeVoxels: m.activeVoxels, indexDims: m.indexDims, fileName: m.fileName, fileBytes: m.fileBytes };
  }

  function onBaked(m: Extract<WorkerResponse, { type: "baked" }>): void {
    const d = wvdbDenseVolume(readWvdb(m.wvdb));
    const source = vdbDenseTextures(d);
    const volume = new VDBFrameVolume(source, { ...common, frameRenderer: frDense });
    const old = dense;
    placed.add(volume);
    dense = { asset: m.asset, resolution: m.resolution, dense: d, source, volume, fileBytes: m.wvdb.byteLength };
    if (old) {
      placed.remove(old.volume);
      old.volume.dispose();
      old.source.dispose();
    }
  }

  worker.addEventListener("message", (ev: MessageEvent<WorkerResponse>) => {
    const m = ev.data;
    switch (m.type) {
      case "status":
        shell.status(m.text);
        break;
      case "error":
        shell.fail(`bake worker: ${m.message}`);
        pending?.reject(new Error(m.message));
        pending = null;
        break;
      case "loaded":
        if (m.asset !== controls.asset) return;
        onLoaded(m).then(
          () => settle("loaded", m.asset),
          (err: unknown) => shell.fail(err),
        );
        break;
      case "baked":
        if (m.asset !== controls.asset || m.resolution !== controls.resolution) return;
        onBaked(m);
        settle("baked", m.asset, m.resolution);
        break;
      case "sparse-stats":
        if (sparse && sparse.asset === m.asset) {
          sparse.fileGz = m.fileGz;
          sparse.error = m.error;
          refreshCards();
        }
        break;
      case "dense-stats":
        if (dense && dense.asset === m.asset && dense.resolution === m.resolution) {
          dense.error = m.error;
          refreshCards();
        }
        break;
    }
  });
  worker.addEventListener("error", (e) => shell.fail(`bake worker: ${e.message}`));

  function settle(want: "loaded" | "baked", asset: AssetId, resolution?: number): void {
    if (pending && pending.want === want && pending.asset === asset && (resolution === undefined || pending.resolution === resolution)) {
      const p = pending;
      pending = null;
      p.resolve();
    }
  }

  const ask = (m: WorkerRequest, want: "loaded" | "baked"): Promise<void> =>
    new Promise<void>((resolve, reject) => {
      pending = { resolve, reject, want, asset: m.asset, ...(m.type === "bake" ? { resolution: m.resolution } : {}) };
      send(m);
    });

  function disposeSides(): void {
    if (sparse) {
      placed.remove(sparse.volume);
      sparse.volume.dispose();
      sparse.source.dispose();
      sparse.grids.dispose();
      sparse.pool.dispose();
      sparse = null;
    }
    if (dense) {
      placed.remove(dense.volume);
      dense.volume.dispose();
      dense.source.dispose();
      dense = null;
    }
  }

  async function loadAsset(): Promise<void> {
    state.ready = false;
    state.loading = true;
    disposeSides();
    state.asset = controls.asset;
    await ask({ type: "load", asset: controls.asset }, "loaded");
    await bakeDense();
    applyView();
  }

  async function bakeDense(): Promise<void> {
    state.ready = false;
    state.loading = true;
    state.resolution = controls.resolution;
    await ask({ type: "bake", asset: controls.asset, resolution: controls.resolution }, "baked");
    send({ type: "measure", asset: controls.asset, resolution: controls.resolution });
    applySettings();
    refreshCards();
    renderedSinceLoad = 0;
    state.loading = false;
  }

  function applySettings(): void {
    const q = QUALITY[controls.quality];
    for (const fr of [frSparse, frDense]) {
      const s = fr.settings;
      s.lighting = controls.lighting;
      s.width = q.width;
      s.height = q.height;
      s.depth = q.depth;
      s.subSteps = q.subSteps;
      s.shadowResolution = q.shadowResolution;
      s.multipleScattering = "neural";
      s.temporal = controls.temporal;
      s.jitter = controls.temporal;
      s.phase = "dual-lobe";
      s.phaseG = 0.8;
      s.phaseBackG = -0.3;
      s.phaseBackWeight = 0.2;
    }
    for (const v of [sparse?.volume, dense?.volume]) if (v) v.shadowSteps = q.shadowSteps;
  }

  function refreshCards(): void {
    if (sparse) {
      const g = sparse.source.gpuBytes;
      const gpu = g.pool + g.pyramid + g.pyramidScratch;
      const st: SideStats = {
        label: "sparse",
        resolution: `${sparse.indexDims.join("×")} (${(sparse.activeVoxels / 1e6).toFixed(sparse.activeVoxels < 1e7 ? 2 : 1)} M active)`,
        gpuBytes: gpu,
        fileBytes: sparse.fileBytes,
        ...(sparse.fileGz !== undefined ? { fileGz: sparse.fileGz } : {}),
        ...(sparse.error ? { errorPercent: sparse.error.percent } : {}),
      };
      state.sparse = st;
      ui.fill(
        ui.left,
        `Sparse NanoVDB · full resolution`,
        [
          ["grid", `${sparse.indexDims.join("×")}, ${sparse.gridType}`],
          ["voxels", `${(sparse.activeVoxels / 1e6).toFixed(sparse.activeVoxels < 1e7 ? 2 : 1)} M active, sparse`],
          ["GPU", `${formatBytes(g.pool)} pool + ${formatBytes(g.pyramid + g.pyramidScratch)} LOD`],
          ["file", `.nvdb ${formatBytes(sparse.fileBytes)}${sparse.fileGz !== undefined ? ` · ${formatBytes(sparse.fileGz)} gz` : ""}`],
          ["error", `${pct(sparse.error)}${sparse.error ? ` vs ${sparse.error.against}` : ""}`],
        ],
        "every active voxel, read through the pool",
      );
    }
    if (dense) {
      const d = dense.dense;
      const texels = d.dimensions[0] * d.dimensions[1] * d.dimensions[2];
      const gpu = texels + (d.density.mip?.data.byteLength ?? 0);
      const st: SideStats = {
        label: "dense",
        resolution: `${d.dimensions.join("×")} texels`,
        gpuBytes: gpu,
        fileBytes: dense.fileBytes,
        ...(dense.error ? { errorPercent: dense.error.percent } : {}),
      };
      state.dense = st;
      ui.fill(
        ui.right,
        `Dense .wvdb · ${dense.resolution}${dense.resolution === 128 ? " (his default)" : ""}`,
        [
          ["grid", `${d.dimensions.join("×")}, 8-bit`],
          ["voxels", `${(texels / 1e6).toFixed(2)} M texels, dense`],
          ["GPU", `${formatBytes(gpu)} r8 3D texture + mips`],
          ["file", `.wvdb ${formatBytes(dense.fileBytes)} (deflated)`],
          ["error", `${pct(dense.error)}${dense.error ? ` vs ${dense.error.against}` : ""}`],
        ],
        "point-sampled, clamped to 0–1",
      );
    }
  }

  // --- controls -------------------------------------------------------------------------------------------
  shell.select<AssetId>("asset", "asset", available.map((id) => ({ value: id, label: id === "wdas" ? "WDAS cloud" : "smoke.vdb" })), controls.asset, (v) => {
    controls.asset = v;
    void loadAsset().catch((err: unknown) => shell.fail(err));
  });
  shell.select<string>("res", "dense side", RESOLUTIONS.map((r) => ({ value: String(r), label: `${r}${r === 128 ? " (his default)" : ""}` })), String(controls.resolution), (v) => {
    controls.resolution = Number(v);
    void bakeDense().catch((err: unknown) => shell.fail(err));
  });
  dividerRow = shell.range("divider", "divider", 0.05, 0.95, 0.01, controls.divider, (v) => setDivider(v), (v) => `${Math.round(v * 100)}%`);
  shell.select<View>("view", "camera", [
    { value: "overview", label: "overview" },
    { value: "detail", label: "zoom: detail" },
  ], controls.view, (v) => {
    controls.view = v;
    applyView();
  });
  shell.select<"volume" | "frame">("lighting", "lighting", [
    { value: "volume", label: "own shadows" },
    { value: "frame", label: "shared grid" },
  ], controls.lighting, (v) => {
    controls.lighting = v;
    applySettings();
  });
  shell.select<Quality>("quality", "quality", [
    { value: "low", label: "low (128² × 16)" },
    { value: "medium", label: "medium (256² × 24)" },
    { value: "high", label: "high (512² × 32)" },
  ], controls.quality, (v) => {
    controls.quality = v;
    resize();
    applySettings();
  });

  resize();
  window.addEventListener("resize", resize);
  setDivider(controls.divider);
  window.__DEMO15_API__ = { capture: () => presenter.capture() };
  await loadAsset();

  // --- loop -----------------------------------------------------------------------------------------------
  let busy = false;
  let frameMs = 0;
  let lastStats = 0;
  const renderSide = (show: VDBFrameVolume | undefined, hide: VDBFrameVolume | undefined, rt: THREE.RenderTarget): void => {
    if (show) show.visible = true;
    if (hide) hide.visible = false;
    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
  };
  const frame = async (): Promise<void> => {
    renderSide(sparse?.volume, dense?.volume, rtSparse);
    renderSide(dense?.volume, sparse?.volume, rtDense);
    renderer.setRenderTarget(null);
    if (PRESENTATION === "direct") {
      composite.render(renderer);
      await device.queue.onSubmittedWorkDone();
      return;
    }
    // Readback: both sides back to the CPU, joined at the divider column.
    const [a, b] = await Promise.all([presenter.readTarget(rtSparse), presenter.readTarget(rtDense)]);
    const w = presenter.width;
    const cut = Math.round(controls.divider * w) * 4;
    for (let y = 0; y < presenter.height; y++) {
      const row = y * w * 4;
      a.set(b.subarray(row + cut, row + w * 4), row + cut);
    }
    presenter.show(a);
  };
  const loop = (now: number): void => {
    requestAnimationFrame(loop);
    if (busy || state.error || state.loading || !sparse || !dense) return;
    busy = true;
    orbit.update();
    const t0 = performance.now();
    frame()
      .then(() => {
        const ms = performance.now() - t0;
        frameMs = frameMs === 0 ? ms : frameMs * 0.9 + ms * 0.1;
        state.frames++;
        state.frameMs = frameMs;
        renderedSinceLoad++;
        if (renderedSinceLoad >= 2) state.ready = true;
        if (now - lastStats > 500 || renderedSinceLoad <= 2) {
          lastStats = now;
          const q = QUALITY[controls.quality];
          shell.stats(
            `${state.adapter} · ${presenter.pixelWidth}×${presenter.pixelHeight} · froxels ${q.width}×${q.height}×${q.depth}\n` +
              `${frameMs.toFixed(1)} ms/frame for both sides (provisional, D25)\n` +
              `GPU: sparse ${formatBytes(state.sparse?.gpuBytes ?? 0)} · dense ${formatBytes(state.dense?.gpuBytes ?? 0)}`,
          );
        }
      })
      .catch((err: unknown) => shell.fail(err))
      .finally(() => {
        busy = false;
      });
  };
  requestAnimationFrame(loop);
  window.addEventListener("pagehide", () => {
    disposeSides();
    worker.terminate();
  });
}

/** A soft vertical gradient sky (equirect, linear HalfFloat) when the HDR is absent. */
function gradientSky(): THREE.DataTexture {
  const h = 64;
  const data = new Uint16Array(4 * h * 4);
  const top = [0.12, 0.2, 0.42];
  const horizon = [0.55, 0.62, 0.72];
  const ground = [0.16, 0.16, 0.18];
  for (let y = 0; y < h; y++) {
    const v = (y + 0.5) / h; // 0 = down, 1 = up
    const up = Math.max(0, (v - 0.5) * 2);
    const c = v < 0.5 ? horizon.map((x, i) => x + (ground[i]! - x) * Math.min(1, (0.5 - v) * 6)) : horizon.map((x, i) => x + (top[i]! - x) * Math.pow(up, 0.6));
    for (let x = 0; x < 4; x++) {
      const i = (y * 4 + x) * 4;
      data[i] = THREE.DataUtils.toHalfFloat(c[0]!);
      data[i + 1] = THREE.DataUtils.toHalfFloat(c[1]!);
      data[i + 2] = THREE.DataUtils.toHalfFloat(c[2]!);
      data[i + 3] = THREE.DataUtils.toHalfFloat(1);
    }
  }
  const t = new THREE.DataTexture(data, 4, h, THREE.RGBAFormat, THREE.HalfFloatType);
  t.mapping = THREE.EquirectangularReflectionMapping;
  t.magFilter = THREE.LinearFilter;
  t.minFilter = THREE.LinearFilter;
  t.needsUpdate = true;
  return t;
}

run().catch((err: unknown) => shell.fail(err));
