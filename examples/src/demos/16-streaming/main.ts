/**
 * Gallery demo 16 — Streaming. A coarse-first `.nvdb` (`vdbBakeCoarseFirst`: `[density.lod48, density]`, ZIP) is
 * streamed through a throttled link into the froxel renderer with `vdbStreamCoarseFirst`: the 48-voxel preview is
 * drawn once its ~14 KiB are in, and the full-resolution grid replaces it in place (the same pool id) when its last
 * byte lands. A byte counter and a timeline (bytes received against time, with the preview and full markers) show
 * it; the caption flashes at each swap.
 *
 * Assets: the WDAS cloud (quarter, Fp8 full grid) from `fixtures/wdas/wdas_cloud_quarter.coarse.nvdb` (baked at dev
 * time: `pnpm build && node scripts/bake-coarse-first.mjs fixtures/wdas/wdas_cloud_quarter_fp8.nvdb --out
 * fixtures/wdas/wdas_cloud_quarter.coarse.nvdb`), else baked in the page from `wdas_cloud_quarter_fp8.nvdb`; and
 * `smoke.vdb` (openvdb.org, CC BY-SA 4.0) baked in the page as demo 09. Sky: `fixtures/gallery/sky-midday-HDR.jpg`
 * when present, else a procedural sky. Public API only.
 *
 * The froxel renderer's sparse source maps its unit box onto the grid's index box when it is made, and the
 * `VDBGridNode` decodes the grid types the pool holds when it is made, so each swap makes a new node, source and
 * `VDBFrameVolume` over the same pool id (the frame renderer integrates in world space, so a preview needs no
 * `voxelRatio` density scale, unlike `VDBVolumeMaterial`).
 *
 * Query (plus the shell's `presentation`, `panel`, `still`, `gallery`): `?asset=wdas|smoke`, `?speed=<KiB/s>`
 * (0: unthrottled), `?quality=low|medium|high`, `?temporal=0|1`, `?mode=direct` (no stream: the full grid added
 * directly, the reference the e2e compares the streamed result with), `?test=1` (with `presentation=readback`: still, unthrottled
 * 1 KiB chunks, the stream held until each grid's frame is read back). `window.__DEMO16__` is the e2e state.
 *
 * Frame times are submit-to-done wall times on this page: provisional (D25), not a benchmark.
 */
import "../webgpu-gate.js";
import { fixtureUrl } from "../assets.js";
import * as THREE from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { VDBGridPool } from "@pmndrs/vdb";
import { NvdbFile, parseVdb, vdbBakeCoarseFirst, type NvdbGridMetadata } from "@pmndrs/vdb/tools";
import { VDBFrameRenderer, VDBFrameVolume, VDBGridNode, getDevice, vdbSparseSource, vdbStreamCoarseFirst, type VDBCoarseFirstEvent, type VDBThreeVolumeSource } from "@pmndrs/vdb/three";
import { createDeviceRenderer } from "../device-renderer.js";
import { KiB, MiB, PRESENTATION as SHELL_PRESENTATION, Presenter, QUALITY, autoOrbit, initialQuality, mountShowcase, params, probe, type Quality, type Snapshot } from "../showcase-shell.js";
import { loadHdrSky, proceduralSky } from "../gallery-sky.js";

type AssetId = "wdas" | "smoke";
type Mode = "stream" | "direct";

interface PhaseRecord {
  phase: "preview" | "full";
  grid: string;
  gridType: string;
  /** File bytes received when the grid completed (the wire bytes: no HTTP compression on this link). */
  transferBytes: number;
  /** Seconds from the start of the stream to the grid's arrival. */
  seconds: number;
  /** Pool image bytes of the grid. */
  imageBytes: number;
  voxelRatio: number;
  /** The first settled frame with this grid (readback only). */
  snapshot: Snapshot | null;
}

interface Demo16State {
  ready: boolean;
  error?: string;
  adapter?: string;
  presentation: "direct" | "readback";
  mode: Mode;
  asset?: AssetId;
  available: AssetId[];
  missing: string[];
  sky?: string;
  totalBytes?: number;
  transferred: number;
  phases: PhaseRecord[];
  /** Same pool id for every grid shown, and no pool growth after the preview. */
  sameId?: boolean;
  inPlace?: boolean;
  done: boolean;
  frames: number;
  uncaptured: string[];
  /** `?mode=direct`: the reference frame. */
  reference?: Snapshot | null;
}

declare global {
  interface Window {
    __DEMO16__?: Demo16State;
    __DEMO16_API__?: { capture(): Snapshot | null; restart(): void };
  }
}

const TEST = params.get("test") === "1";
const PRESENTATION = SHELL_PRESENTATION;
const MODE: Mode = params.get("mode") === "direct" ? "direct" : "stream";
const STILL_MODE = TEST || params.get("still") === "1";
const SPEEDS = [256, 1024, 2048, 4096, 10240] as const;
const speedLabel = (k: number) => (k === 0 ? "unthrottled" : k < 1024 ? `${k} KiB/s` : `${k / 1024} MiB/s`);
const SKY_URL = fixtureUrl("gallery/sky-midday-HDR.jpg");
const WDAS_BAKED = fixtureUrl("wdas/wdas_cloud_quarter.coarse.nvdb");
const WDAS_FP8 = fixtureUrl("wdas/wdas_cloud_quarter_fp8.nvdb");
const SMOKE_VDB = fixtureUrl("vdb-samples/smoke.vdb");
/** Longest side of the volume in scene units. */
const SIZE = 100;

const shell = mountShowcase({ strip: "row",
  number: "16",
  title: "Streaming",
  blurb:
    "A coarse-first .nvdb streams into the froxel renderer: a 48-voxel preview draws from the first ~14 KiB, and the full-resolution grid swaps in place when its last byte lands.",
});
const state: Demo16State = { ready: false, presentation: PRESENTATION, mode: MODE, available: [], missing: [], transferred: 0, phases: [], done: false, frames: 0, uncaptured: [] };
window.__DEMO16__ = state;
shell.onFail((message) => {
  state.error = message;
});

// --- the caption's counter, phase line and timeline ---------------------------------------------------------

const phaseEl = document.createElement("div");
phaseEl.dataset.testid = "phase";
phaseEl.style.cssText = "margin:8px 0 2px;font-weight:650;font-size:13.5px;letter-spacing:0.02em";
const counterEl = document.createElement("div");
counterEl.dataset.testid = "counter";
counterEl.style.cssText = "font-size:12.5px;opacity:0.85;font-variant-numeric:tabular-nums;white-space:pre-line";
const timeline = document.createElement("canvas");
timeline.style.cssText = "display:block;width:100%;height:68px;margin-top:6px;border-radius:6px;background:rgba(255,255,255,0.04)";
shell.extra.append(phaseEl, counterEl, timeline);
const PHASE_COLOR = { loading: "#c9d1e6", preview: "#ffd479", full: "#7be3a1" };
const setPhase = (kind: keyof typeof PHASE_COLOR, text: string) => {
  phaseEl.textContent = text;
  phaseEl.style.color = PHASE_COLOR[kind];
  phaseEl.dataset.phase = kind;
};

interface Timeline {
  samples: { t: number; bytes: number }[];
  total: number;
  expectedSeconds: number;
  markers: { t: number; bytes: number; kind: "preview" | "full" }[];
}
function drawTimeline(tl: Timeline): void {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.max(1, Math.round(timeline.clientWidth * dpr));
  const h = Math.max(1, Math.round(timeline.clientHeight * dpr));
  if (timeline.width !== w || timeline.height !== h) {
    timeline.width = w;
    timeline.height = h;
  }
  const g = timeline.getContext("2d")!;
  g.clearRect(0, 0, w, h);
  const pad = { l: 6 * dpr, r: 6 * dpr, t: 14 * dpr, b: 6 * dpr };
  const last = tl.samples[tl.samples.length - 1];
  const tMax = Math.max(0.5, tl.expectedSeconds, last?.t ?? 0, ...tl.markers.map((m) => m.t)) * 1.04;
  const x = (t: number) => pad.l + (t / tMax) * (w - pad.l - pad.r);
  const y = (b: number) => h - pad.b - (tl.total > 0 ? b / tl.total : 0) * (h - pad.t - pad.b);
  // the link's ideal line
  g.strokeStyle = "rgba(255,255,255,0.12)";
  g.setLineDash([3 * dpr, 3 * dpr]);
  g.beginPath();
  g.moveTo(x(0), y(0));
  g.lineTo(x(tl.expectedSeconds), y(tl.total));
  g.stroke();
  g.setLineDash([]);
  // bytes received
  if (tl.samples.length) {
    g.strokeStyle = "#a9bcff";
    g.lineWidth = 1.6 * dpr;
    g.beginPath();
    g.moveTo(x(0), y(0));
    for (const s of tl.samples) g.lineTo(x(s.t), y(s.bytes));
    g.stroke();
    g.fillStyle = "rgba(169,188,255,0.12)";
    g.lineTo(x(last!.t), y(0));
    g.closePath();
    g.fill();
  }
  // markers
  g.font = `${10.5 * dpr}px system-ui, sans-serif`;
  for (const m of tl.markers) {
    const color = m.kind === "full" ? PHASE_COLOR.full : PHASE_COLOR.preview;
    g.strokeStyle = color;
    g.lineWidth = 1.2 * dpr;
    g.beginPath();
    g.moveTo(x(m.t), pad.t - 2 * dpr);
    g.lineTo(x(m.t), h - pad.b);
    g.stroke();
    g.fillStyle = color;
    g.beginPath();
    g.arc(x(m.t), y(m.bytes), 3 * dpr, 0, Math.PI * 2);
    g.fill();
    const label = `${m.kind} · ${m.bytes < 1048576 ? KiB(m.bytes) : MiB(m.bytes)}`;
    const tw = g.measureText(label).width;
    const lx = m.kind === "preview" ? Math.min(x(m.t) + 4 * dpr, w - tw - 2 * dpr) : Math.max(2 * dpr, x(m.t) - tw - 4 * dpr);
    g.fillText(label, lx, 10.5 * dpr);
  }
}

// --- the throttled link (demo 09's pull-driven stream) -------------------------------------------------------

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
function throttle(response: Response, opts: { chunk: number; bytesPerSec: number; gate: () => Promise<void>; onBytes: (sent: number) => void; signal: AbortSignal }): Response {
  const upstream = response.body!.getReader();
  let pending = new Uint8Array(0);
  let sent = 0;
  const body = new ReadableStream<Uint8Array>(
    {
      async pull(controller) {
        await opts.gate();
        if (opts.signal.aborted) {
          controller.close();
          return;
        }
        while (pending.length < opts.chunk) {
          const { done, value } = await upstream.read();
          if (done) break;
          const merged = new Uint8Array(pending.length + value.length);
          merged.set(pending);
          merged.set(value, pending.length);
          pending = merged;
        }
        if (pending.length === 0) {
          controller.close();
          return;
        }
        const chunk = pending.slice(0, opts.chunk);
        pending = pending.slice(chunk.length);
        if (opts.bytesPerSec > 0) await sleep((chunk.length / opts.bytesPerSec) * 1000);
        sent += chunk.length;
        opts.onBytes(sent);
        controller.enqueue(chunk);
      },
      cancel: () => upstream.cancel(),
    },
    { highWaterMark: 0 },
  );
  return new Response(body);
}

// --- assets -------------------------------------------------------------------------------------------------

interface AssetDef {
  label: string;
  /** The coarse-first file as a fresh Response each run (a real fetch, or a page bake behind a Blob URL). */
  open(): Promise<{ response: Response; total: number; note: string }>;
}

async function run(): Promise<void> {
  const [bakedOk, fp8Ok, smokeOk, skyOk] = await Promise.all([probe(WDAS_BAKED), probe(WDAS_FP8), probe(SMOKE_VDB), probe(SKY_URL)]);
  const bakes = new Map<AssetId, Uint8Array>();
  const bakeOnce = async (id: AssetId, make: () => Promise<Uint8Array>): Promise<Uint8Array> => {
    let bytes = bakes.get(id);
    if (!bytes) {
      bytes = await make();
      bakes.set(id, bytes);
    }
    return bytes;
  };
  const blobResponse = async (bytes: Uint8Array): Promise<Response> => fetch(URL.createObjectURL(new Blob([bytes as BlobPart], { type: "application/octet-stream" })));
  const ASSETS: Partial<Record<AssetId, AssetDef>> = {};
  if (bakedOk || fp8Ok) {
    ASSETS.wdas = {
      label: "WDAS cloud (quarter)",
      async open() {
        if (bakedOk) {
          const response = await fetch(WDAS_BAKED);
          if (!response.ok) throw new Error(`${WDAS_BAKED}: ${response.status}`);
          return { response, total: Number(response.headers.get("content-length") ?? 0), note: "wdas_cloud_quarter.coarse.nvdb (baked at dev time)" };
        }
        const bytes = await bakeOnce("wdas", async () => {
          shell.status("baking the WDAS cloud coarse-first in the page (a few seconds; bake it once with scripts/bake-coarse-first.mjs)…");
          const image = NvdbFile.fromArrayBuffer(await (await fetch(WDAS_FP8)).arrayBuffer()).gridImage(0);
          await sleep(30); // paint the status before the bake blocks
          return vdbBakeCoarseFirst(image, { previewResolution: 48 });
        });
        shell.status("");
        return { response: await blobResponse(bytes), total: bytes.length, note: "wdas_cloud_quarter_fp8.nvdb, baked in the page" };
      },
    };
  } else state.missing.push("WDAS cloud: pnpm fixtures:wdas, then nanovdb_convert --fp8 (fixtures/README.md)");
  if (smokeOk) {
    ASSETS.smoke = {
      label: "smoke.vdb",
      async open() {
        const bytes = await bakeOnce("smoke", async () => {
          shell.status("baking smoke.vdb coarse-first in the page…");
          const vdb = parseVdb(await (await fetch(SMOKE_VDB)).arrayBuffer());
          await sleep(30);
          return vdbBakeCoarseFirst(vdb, { previewResolution: 48 });
        });
        shell.status("");
        return { response: await blobResponse(bytes), total: bytes.length, note: "smoke.vdb, baked in the page" };
      },
    };
  } else state.missing.push("smoke.vdb: pnpm fixtures:vdb-samples");
  state.available = Object.keys(ASSETS) as AssetId[];
  if (!skyOk) state.missing.push("sky-midday-HDR.jpg (fixtures/gallery/): procedural sky");
  if (state.missing.length) shell.note(`Not found:\n${state.missing.map((m) => `• ${m}`).join("\n")}`);
  if (state.available.length === 0) throw new Error("No streaming asset is present: pnpm fixtures:vdb-samples (smoke.vdb) or pnpm fixtures:wdas.");

  const asked = params.get("asset") as AssetId | null;
  const controls = {
    asset: asked && ASSETS[asked] ? asked : state.available[0]!,
    speed: params.has("speed") ? Math.max(0, Number(params.get("speed"))) : TEST ? 0 : 2048,
    quality: TEST ? ("low" as Quality) : initialQuality(),
    temporal: params.has("temporal") ? params.get("temporal") === "1" : PRESENTATION === "direct",
  };

  // --- renderer and stage -----------------------------------------------------------------------------------
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
  renderer.toneMappingExposure = 0.9;
  const presenter = new Presenter(shell.app, renderer, directCanvas, device);
  const frameRenderer = new VDBFrameRenderer();

  const scene = new THREE.Scene();
  const sun = new THREE.DirectionalLight(0xfff4e6, 1.8);
  sun.position.set(250, 110, 60); // from the side: the cloud shades across its face
  scene.add(sun, sun.target);
  if (skyOk) {
    const sky = await loadHdrSky(SKY_URL);
    scene.background = sky;
    scene.environment = sky;
    scene.environmentIntensity = 0.3;
    state.sky = "sky-midday-HDR.jpg";
  } else {
    const sky = proceduralSky({ zenith: "#2f6fd6", horizon: "#b8d4f2", ground: "#59636e" });
    scene.background = sky;
    scene.environment = sky;
    scene.environmentIntensity = 0.6;
    state.sky = "procedural";
  }

  const camera = new THREE.PerspectiveCamera(40, 1, 1, 8000);
  const orbit = new OrbitControls(camera, presenter.canvas);
  orbit.enableDamping = PRESENTATION === "direct";
  orbit.minDistance = 60;
  orbit.maxDistance = 900;
  /** The start view; `far` > 1 backs off for a tall asset (the smoke plume). */
  const home = (far = 1) => {
    camera.position.set(-95 * far, -15, 100 * far);
    orbit.target.set(0, -5, 0);
    orbit.update();
  };
  home();
  if (!STILL_MODE) autoOrbit(orbit, 0.4);
  const resize = () => presenter.resize(QUALITY[controls.quality].pixelRatio, camera);
  resize();
  window.addEventListener("resize", resize);

  /** The volume's parent: places the grid's world box (longest side SIZE) at the origin. */
  const holder = new THREE.Group();
  scene.add(holder);
  const place = (box: NvdbGridMetadata["worldBBox"]) => {
    const size = box.max.map((v, i) => v - box.min[i]!);
    const s = SIZE / Math.max(...size);
    holder.scale.setScalar(s);
    holder.position.set(...(box.min.map((v, i) => -(v + size[i]! / 2) * s) as [number, number, number]));
  };

  function applySettings(volume: VDBFrameVolume | null): void {
    const q = QUALITY[controls.quality];
    const st = frameRenderer.settings;
    st.width = q.width;
    st.height = q.height;
    st.depth = q.depth;
    st.subSteps = q.subSteps;
    st.shadowResolution = q.shadowResolution;
    st.lighting = "frame";
    st.multipleScattering = "neural";
    st.temporal = controls.temporal;
    st.jitter = controls.temporal;
    st.phase = "dual-lobe";
    st.phaseG = 0.8;
    st.phaseBackG = -0.3;
    st.phaseBackWeight = 0.2;
    if (volume) volume.shadowSteps = q.shadowSteps;
  }

  // --- one run: a pool, the stream, a node + source + volume per grid shown ---------------------------------
  interface Shown {
    grids: VDBGridNode;
    source: VDBThreeVolumeSource;
    volume: VDBFrameVolume;
    id: number;
  }
  let pool: VDBGridPool | null = null;
  let shown: Shown | null = null;
  let abort: AbortController | null = null;
  let generation = 0;

  // One render at a time: the swap work and the loop both draw.
  let chain: Promise<unknown> = Promise.resolve();
  const serial = <T>(fn: () => Promise<T>): Promise<T> => {
    const next = chain.then(fn, fn);
    chain = next.catch(() => undefined);
    return next;
  };

  let frameMs = 0;
  const present = async (): Promise<void> => {
    const t0 = performance.now();
    await presenter.present(scene, camera);
    const ms = performance.now() - t0;
    frameMs = frameMs === 0 ? ms : frameMs * 0.9 + ms * 0.1;
    state.frames++;
  };

  /** Show the grid now in pool id `id`: a new node (the pool's types now), source and volume; the old ones go. */
  function showGrid(p: VDBGridPool, id: number, image: Uint32Array): void {
    const grids = new VDBGridNode(renderer, p, { profile: "march", helpers: ["cellBoundTrilinear", "hddaCellEnd", "nodeAverage", "fastTrilinear"] });
    const source = vdbSparseSource(grids, { density: id }, { lod: "coarse-pyramid", images: { density: image } });
    const volume = new VDBFrameVolume(source, { frameRenderer, extinction: 100, angstromExponent: 0, normalizeDensity: true });
    applySettings(volume);
    holder.add(volume);
    const old = shown;
    shown = { grids, source, volume, id };
    if (old) disposeShown(old);
  }
  function disposeShown(s: Shown): void {
    holder.remove(s.volume);
    s.volume.dispose();
    s.source.dispose();
    s.grids.dispose();
  }

  const tl: Timeline = { samples: [], total: 0, expectedSeconds: 1, markers: [] };
  let t0 = 0;
  let total = 0;
  const showCounter = () => {
    const pct = total > 0 ? Math.min(100, (state.transferred / total) * 100) : 0;
    const unit = total < 1048576 ? KiB : MiB;
    counterEl.textContent = `received ${unit(state.transferred)} of ${unit(total)} (${pct.toFixed(pct < 10 ? 1 : 0)}%) · link ${speedLabel(controls.speed)}`;
  };

  async function startRun(): Promise<void> {
    const gen = ++generation;
    abort?.abort();
    abort = new AbortController();
    const signal = abort.signal;
    // Let the previous run's swap work drain, then clear its pieces.
    await serial(async () => {
      if (shown) disposeShown(shown);
      shown = null;
      pool?.dispose();
      pool = null;
    });
    state.ready = false;
    state.done = false;
    state.phases = [];
    state.transferred = 0;
    tl.samples = [];
    tl.markers = [];
    const asset = ASSETS[controls.asset]!;
    if (state.asset !== controls.asset) home(controls.asset === "smoke" ? 1.5 : 1);
    state.asset = controls.asset;
    setPhase("loading", MODE === "direct" ? "loading the full grid…" : "connecting…");
    const opened = await asset.open();
    if (gen !== generation) return;
    total = opened.total;
    state.totalBytes = total;
    tl.total = total;
    tl.expectedSeconds = controls.speed > 0 ? total / (controls.speed * 1024) : 1;
    const p = await VDBGridPool.create(getDevice(renderer), { label: "demo-16" });
    pool = p;

    if (MODE === "direct") {
      // The reference: the full grid added directly, placed by the preview's box as the stream places it.
      const file = NvdbFile.fromArrayBuffer(await opened.response.arrayBuffer());
      const fullIndex = file.grids.findIndex((g) => !/\.lod\d+$/.test(g.name));
      place(file.grids[0]!.worldBBox);
      const image = file.gridImage(fullIndex);
      const id = p.add(image);
      p.flush();
      await p.settled();
      await serial(async () => {
        showGrid(p, id, image);
        await present();
        await present();
      });
      state.reference = presenter.capture();
      state.transferred = total;
      setPhase("full", `FULL RESOLUTION · ${file.grids[fullIndex]!.gridType} (direct, no stream)`);
      showCounter();
      state.done = true;
      state.ready = true;
      return;
    }

    let held: Promise<void> | null = null;
    let versionAfterPreview = -1;
    const speed = controls.speed * 1024;
    t0 = performance.now();
    const throttled = throttle(opened.response, {
      chunk: TEST ? 1024 : speed > 0 ? Math.max(1024, Math.round(speed / 60)) : 65536,
      bytesPerSec: speed,
      gate: async () => {
        if (held) await held;
      },
      onBytes: (sent) => {
        if (gen !== generation) return;
        state.transferred = sent;
        tl.samples.push({ t: (performance.now() - t0) / 1000, bytes: sent });
        showCounter();
      },
      signal,
    });
    setPhase("loading", "streaming… waiting for the preview grid");
    showCounter();

    const onGrid = (e: VDBCoarseFirstEvent): void => {
      // Arrival: the file bytes the reader had when the grid completed (no HTTP compression here: wire = file bytes).
      const arrived = (performance.now() - t0) / 1000;
      const bytes = e.bytesReceived;
      const phase = e.final ? "full" : "preview";
      tl.markers.push({ t: arrived, bytes, kind: phase });
      held = serial(async () => {
        if (gen !== generation) return;
        // The new grid is in the pool's staging: write it, then build the node, source and volume for it.
        p.flush();
        await p.settled();
        if (!e.final) versionAfterPreview = p.version;
        if (e.action === "add") place(e.grid.metadata.worldBBox);
        showGrid(p, e.id, e.grid.image);
        await present(); // compiles
        await present(); // a settled frame
        const info = p.info(e.id);
        state.phases.push({ phase, grid: e.grid.metadata.name, gridType: e.grid.metadata.gridType, transferBytes: bytes, seconds: arrived, imageBytes: info.imageBytes, voxelRatio: e.voxelRatio, snapshot: presenter.capture() });
        if (e.final) {
          setPhase("full", `FULL RESOLUTION · ${e.grid.metadata.gridType}, ${MiB(info.imageBytes)} · in at ${MiB(bytes)}, ${arrived.toFixed(1)} s`);
          shell.flash("full", "full-resolution grid swapped in");
        } else {
          const res = Math.max(...e.grid.metadata.indexBBox.max.map((v, i) => v - e.grid.metadata.indexBBox.min[i]! + 1));
          setPhase("preview", `PREVIEW · ${res}-voxel ${e.grid.metadata.gridType} · in at ${KiB(bytes)}, ${(arrived * 1000).toFixed(0)} ms`);
          shell.flash("preview", `preview from the first ${KiB(bytes)}`);
        }
      });
      held.catch((err: unknown) => shell.fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err)));
    };

    try {
      const result = await vdbStreamCoarseFirst(p, throttled, { onGrid, signal });
      await held;
      if (gen !== generation) return;
      state.sameId = result.shown.every((s) => s.id === result.id);
      state.inPlace = p.version === versionAfterPreview;
      state.done = true;
      state.ready = true;
    } catch (err) {
      if (signal.aborted || gen !== generation) return;
      throw err;
    }
  }

  // --- controls ---------------------------------------------------------------------------------------------
  const restart = () => {
    startRun().catch((err: unknown) => shell.fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err)));
  };
  shell.select<AssetId>("asset", "asset", state.available.map((id) => ({ value: id, label: ASSETS[id]!.label })), controls.asset, (v) => {
    controls.asset = v;
    restart();
  });
  const speeds = [...new Set([...SPEEDS, controls.speed])].sort((a, b) => a - b);
  shell.select("speed", "link", speeds.map((k) => ({ value: String(k), label: speedLabel(k) })), String(controls.speed), (v) => {
    controls.speed = Number(v);
    restart();
  });
  shell.select<Quality>("quality", "quality", (["low", "medium", "high"] as const).map((q) => ({ value: q, label: QUALITY[q].label })), controls.quality, (v) => {
    controls.quality = v;
    resize();
    applySettings(shown?.volume ?? null);
  });
  shell.button("restart", "↻ restart", restart);
  window.__DEMO16_API__ = { capture: () => presenter.capture(), restart };
  // `?debug=1`: the scene's pieces, for tuning from the console.
  if (params.get("debug") === "1") {
    Object.assign(window, {
      __demo16: { THREE, scene, camera, orbit, sun, controls, applySettings, frameRenderer, renderer, shown: () => shown },
    });
  }

  // --- loop -------------------------------------------------------------------------------------------------
  let inFlight = false;
  let lastStats = 0;
  const loop = (now: number): void => {
    requestAnimationFrame(loop);
    drawTimeline(tl);
    if (!shown || state.error || inFlight || TEST) return;
    inFlight = true;
    orbit.update();
    serial(present)
      .catch((err: unknown) => shell.fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err)))
      .finally(() => {
        inFlight = false;
      });
    if (now - lastStats > 400) {
      lastStats = now;
      const info = pool && shown ? pool.info(shown.id) : null;
      shell.stats(
        `${state.adapter} · ${presenter.width}×${presenter.height} px · ${frameMs.toFixed(1)} ms/frame (provisional, D25)\n` +
        (info ? `pool grid ${info.type}, ${MiB(info.imageBytes)} (capacity ${MiB(info.capacity)}) · ` : "") +
        `${ASSETS[controls.asset]!.label}`,
      );
    }
  };
  requestAnimationFrame(loop);
  applySettings(null);
  await startRun();
}

run().catch((err: unknown) => shell.fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err)));
