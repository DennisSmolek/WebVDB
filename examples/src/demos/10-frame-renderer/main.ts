/**
 * Demo 10 — the froxel renderer (K3), interactive: mjurczyk's FrameRenderer ported to TSL over dense
 * and sparse sources (`VDBFrameVolume`, `VDBFrameRenderer`), with the level-set and point scenes of his
 * examples page (`VDBLevelSetMaterial`, `VDBPoints`). Public API only.
 *
 * A scene picker offers the scenes whose fixtures exist (HEAD-probed); the panel names the fixture
 * script for the rest. The controls set `frameRenderer.settings` and the volume's props live; the
 * source kind and the sparse LOD rebuild the source and the volume.
 *
 * Query: `?scene=wdas|smoke|bunny|explosion|teapot|points`, `?quality=low|medium|high`,
 * `?presentation=direct|readback` (readback: an offscreen HalfFloat target read back to a 2D canvas, for
 * headless SwiftShader e2e, as demos 02 and 08), `?stage=0` (no ground or obstacles), `?temporal=0|1`,
 * `?play=0` and `?frame=N` (the explosion sequence), `?source=dense|sparse` (first scene), `?panel=0` (hide the panel). `window.__DEMO10__` is the e2e state;
 * `window.__DEMO10_API__.capture()` summarizes the last read-back frame.
 *
 * Frame times are submit-to-done wall times of one frame on this page: provisional (D25), not a
 * benchmark.
 */
import * as THREE from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { UltraHDRLoader } from "three/addons/loaders/UltraHDRLoader.js";
import { vdbFrameMediumNames } from "@pmndrs/vdb/tools";
import { VDBFrameRenderer, vdbFrameMedium, type VDBFrameMultipleScattering, type VDBFramePhase, type VDBSparseLod } from "@pmndrs/vdb/three";
import { createDeviceRenderer } from "../device-renderer.js";
import { Panel, type SelectOption } from "./panel.js";
import { GROUND_LEVEL, SCENES, probeScene, type LoadedScene, type SceneContext, type SceneDef, type SourceKind } from "./scenes.js";

// --- State the e2e spec reads ------------------------------------------------------------------------------

interface Demo10State {
  ready: boolean;
  error?: string;
  scene?: string;
  available?: string[];
  missing?: { id: string; files: string[]; hint: string }[];
  adapter?: string;
  presentation?: "direct" | "readback";
  frames: number;
  frameMs?: number;
  resolution?: string;
  froxels?: string;
  info?: string;
  sourceKind?: SourceKind | null;
  loading: boolean;
  uncaptured: string[];
}

interface Snapshot {
  width: number;
  height: number;
  /** Pixels farther than 6 (sum of |d|) from the sky colour. */
  nonBackground: number;
  mean: [number, number, number];
  /** 32 × 18 grey thumbnail (0–255), row-major, top row first. */
  thumb: number[];
}

declare global {
  interface Window {
    __DEMO10__?: Demo10State;
    __DEMO10_API__?: { capture(): Snapshot | null; setScene(id: string): Promise<void> };
  }
}

const params = new URLSearchParams(location.search);
const PRESENTATION = params.get("presentation") === "readback" ? "readback" : "direct";
type Quality = "low" | "medium" | "high";
const QUALITY_PARAM = params.get("quality");
const SKY_COLOR = new THREE.Color(0x598eff); // his examples page background
const SKY_BYTES = [89, 142, 255];
const HDR_URL = "/fixtures/k3/sky-midday-HDR.jpg";

const QUALITY: Record<Quality, { width: number; height: number; depth: number; subSteps: number; shadowSteps: number; shadowResolution: number; emission: number; pixelRatio: number; label: string }> = {
  low: { width: 128, height: 128, depth: 16, subSteps: 8, shadowSteps: 8, shadowResolution: 32, emission: 8, pixelRatio: 1, label: "low (128×128×16)" },
  medium: { width: 256, height: 256, depth: 24, subSteps: 16, shadowSteps: 16, shadowResolution: 64, emission: 16, pixelRatio: 1, label: "medium (256×256×24)" },
  high: { width: 512, height: 512, depth: 24, subSteps: 32, shadowSteps: 32, shadowResolution: 64, emission: 32, pixelRatio: Math.min(2, window.devicePixelRatio || 1), label: "high (512×512×24, his)" },
};

type MsChoice = "off" | "diffusion" | "two-sided" | "octaves" | "neural";
const MS_VALUE: Record<MsChoice, VDBFrameMultipleScattering> = { off: false, diffusion: true, "two-sided": "two-sided", octaves: "octaves", neural: "neural" };

const controls = {
  lighting: "frame" as "volume" | "frame",
  ms: "neural" as MsChoice,
  phase: "dual-lobe" as VDBFramePhase,
  phaseG: 0.8,
  spectral: false,
  medium: "none",
  density: 1,
  sunAzimuth: 45,
  sunElevation: 55,
  sky: (params.get("sky") === "none" || params.get("sky") === "hdr" ? params.get("sky") : "hemisphere") as "hemisphere" | "hdr" | "none",
  stage: params.get("stage") !== "0",
  meshShadows: params.get("shadows") === "1",
  source: "sparse" as SourceKind,
  lod: "coarse-pyramid" as VDBSparseLod,
  hdda: false,
  temporal: params.has("temporal") ? params.get("temporal") === "1" : PRESENTATION === "direct",
  quality: (QUALITY_PARAM === "low" || QUALITY_PARAM === "high" ? QUALITY_PARAM : QUALITY_PARAM === "medium" ? "medium" : PRESENTATION === "readback" ? "low" : "medium") as Quality,
  play: params.get("play") !== "0" && PRESENTATION === "direct",
};

const state: Demo10State = { ready: false, frames: 0, loading: true, uncaptured: [], presentation: PRESENTATION };
window.__DEMO10__ = state;

const statsEl = document.querySelector<HTMLDivElement>("#stats")!;
const statusEl = document.querySelector<HTMLDivElement>("#status")!;
const controlsEl = document.querySelector<HTMLDivElement>("#controls")!;
const missingEl = document.querySelector<HTMLDivElement>("#missing")!;
const errEl = document.querySelector<HTMLDivElement>("#err")!;
const appEl = document.querySelector<HTMLDivElement>("#app")!;
if (params.get("panel") === "0") document.querySelector<HTMLDivElement>("#panel")!.style.display = "none";

function fail(message: string): void {
  errEl.textContent = message;
  state.error = message;
  state.loading = false;
}
const errorText = (err: unknown): string => (err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err));
window.addEventListener("error", (e) => fail(errorText(e.error ?? e.message)));
window.addEventListener("unhandledrejection", (e) => fail(errorText(e.reason)));

// --- sRGB encode of the read-back half-float frame (three's formula, NoToneMapping) ---------------------

const srgbByte = (v: number): number => {
  const c = v <= 0 ? 0 : v >= 1 ? 1 : v < 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
  return Math.round(c * 255);
};

async function run(): Promise<void> {
  // --- which scenes can load --------------------------------------------------------------------------
  const probes = await Promise.all(SCENES.map(async (s) => [s, await probeScene(s)] as const));
  const available = probes.filter(([, ok]) => ok).map(([s]) => s);
  const missing = probes.filter(([, ok]) => !ok).map(([s]) => s);
  state.available = available.map((s) => s.id);
  state.missing = missing.map((s) => ({ id: s.id, files: s.files, hint: s.fetchHint }));
  missingEl.textContent = missing.length
    ? `Not shown (fixtures missing):\n${missing.map((s) => `• ${s.label}: ${s.fetchHint}`).join("\n")}`
    : "";
  if (available.length === 0) throw new Error("No scene's fixtures are present. Run pnpm fixtures (see fixtures/README.md).");
  const hdrAvailable = await fetch(HDR_URL, { method: "HEAD" })
    .then((r) => r.ok)
    .catch(() => false);

  // --- renderer ---------------------------------------------------------------------------------------
  const directCanvas = PRESENTATION === "direct" ? document.createElement("canvas") : undefined;
  const created = await createDeviceRenderer(directCanvas ? { canvas: directCanvas } : {});
  const { renderer, device } = created;
  device.addEventListener("uncapturederror", (e) => {
    const message = (e as GPUUncapturedErrorEvent).error.message;
    state.uncaptured.push(message);
    console.error(`WebGPU: ${message}`);
  });
  void device.lost.then((info) => {
    if (info.reason !== "destroyed") fail(`WebGPU device lost: ${info.message}`);
  });
  state.adapter = created.adapterName || "unreported adapter";

  const display = directCanvas ?? document.createElement("canvas");
  display.id = "view";
  appEl.appendChild(display);
  const ctx2d = PRESENTATION === "readback" ? display.getContext("2d") : null;

  // One froxel renderer for the page: its settings are the panel's.
  const frameRenderer = new VDBFrameRenderer();

  // --- the stage: sky, sun, hemisphere, ground, obstacles ---------------------------------------------
  const scene = new THREE.Scene();
  scene.background = SKY_COLOR.clone();
  const sun = new THREE.DirectionalLight(0xffffff, 1.0);
  sun.shadow.mapSize.setScalar(2048);
  sun.shadow.normalBias = 0.5;
  Object.assign(sun.shadow.camera, { left: -160, right: 160, top: 160, bottom: -160, near: 1, far: 800 });
  sun.shadow.camera.updateProjectionMatrix();
  scene.add(sun, sun.target);
  const hemisphere = new THREE.HemisphereLight(0x88aaff, 0x443322, 0.5);
  scene.add(hemisphere);

  const stage = new THREE.Group();
  const groundMaterial = new THREE.MeshStandardMaterial({ color: 0x8b8f97, roughness: 0.95 });
  const propMaterial = new THREE.MeshStandardMaterial({ color: 0xc9c9c9, roughness: 0.6 });
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(1200, 1200), groundMaterial);
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = GROUND_LEVEL;
  const obstacle = new THREE.Mesh(new THREE.SphereGeometry(16, 48, 24), propMaterial);
  obstacle.position.set(66, GROUND_LEVEL + 16, 34);
  const pillar = new THREE.Mesh(new THREE.BoxGeometry(10, 64, 10), propMaterial);
  pillar.position.set(-62, GROUND_LEVEL + 32, -30);
  stage.add(ground, obstacle, pillar);
  stage.traverse((o) => {
    o.castShadow = o !== ground;
    o.receiveShadow = true;
  });
  scene.add(stage);

  const content = new THREE.Group();
  scene.add(content);

  let hdrTexture: THREE.Texture | null = null;
  async function loadHdr(): Promise<THREE.Texture> {
    hdrTexture ??= await new UltraHDRLoader().loadAsync(HDR_URL);
    hdrTexture.mapping = THREE.EquirectangularReflectionMapping;
    return hdrTexture;
  }

  // --- camera and viewport ----------------------------------------------------------------------------
  const camera = new THREE.PerspectiveCamera(60, 1, 0.5, 10000);
  camera.position.set(-100, -11, 80).multiplyScalar(1.35);
  const orbit = new OrbitControls(camera, display);
  orbit.target.set(0, -8, 0);
  orbit.enableDamping = PRESENTATION === "direct";
  orbit.maxPolarAngle = Math.PI * 0.53;
  orbit.update();

  const rt = new THREE.RenderTarget(2, 2, { type: THREE.HalfFloatType, depthBuffer: true, samples: 0 });
  let vw = 0;
  let vh = 0;
  let lastFrame: Uint8ClampedArray | null = null;
  function setViewport(): void {
    vw = Math.max(1, appEl.clientWidth || window.innerWidth);
    vh = Math.max(1, appEl.clientHeight || window.innerHeight);
    const pr = QUALITY[controls.quality].pixelRatio;
    if (PRESENTATION === "direct") {
      renderer.setPixelRatio(pr);
      renderer.setSize(vw, vh, false);
    } else {
      display.width = vw;
      display.height = vh;
      rt.setSize(vw, vh);
    }
    camera.aspect = vw / vh;
    camera.updateProjectionMatrix();
  }
  setViewport();
  window.addEventListener("resize", setViewport);

  // --- scene lifecycle --------------------------------------------------------------------------------
  let def: SceneDef = available.find((s) => s.id === params.get("scene")) ?? available[0]!;
  let loaded: LoadedScene | null = null;
  const sceneContext = (): SceneContext => ({
    renderer,
    frameRenderer,
    status: (t) => {
      statusEl.textContent = t;
    },
    paused: !controls.play,
    startFrame: Number(params.get("frame") ?? 30),
    emissionResolution: QUALITY[controls.quality].emission,
  });

  // --- the panel ---------------------------------------------------------------------------------------
  const panel = new Panel(controlsEl);
  const sceneRow = panel.select("scene", "scene", available.map((s) => ({ value: s.id, label: s.label })), def.id, (id) => void switchScene(id));
  const sourceRow = panel.select<SourceKind>("source", "source", [], controls.source, (v) => {
    controls.source = v;
    void rebuild();
  });
  const lodRow = panel.select<VDBSparseLod>(
    "lod",
    "sparse LOD",
    [
      { value: "coarse-pyramid", label: "coarse pyramid (default)" },
      { value: "node-average", label: "node average" },
      { value: "none", label: "none (level 0)" },
    ],
    controls.lod,
    (v) => {
      controls.lod = v;
      void rebuild();
    },
  );
  const hddaRow = panel.checkbox("hdda", "HDDA skip", controls.hdda, (v) => {
    controls.hdda = v;
    apply();
  });
  const frameRow = panel.range("frame", "frame", 0, 199, 1, 30, (v) => {
    const seq = loaded?.sequence;
    if (seq) void seq.seek(v);
  });
  const playRow = panel.checkbox("play", "play", controls.play, (v) => {
    controls.play = v;
    if (loaded?.sequence) loaded.sequence.playing = v;
  });

  const froxelRows: { show(v: boolean): void }[] = [];
  const froxel = <T extends { show(v: boolean): void }>(row: T): T => {
    froxelRows.push(row);
    return row;
  };
  const lightingHeading = panel.heading("Lighting and scattering");
  froxelRows.push({ show: (v) => (lightingHeading.hidden = !v) });
  froxel(
    panel.select("lighting", "lighting", [
      { value: "frame", label: "frame (shadow grid)" },
      { value: "volume", label: "volume (own march)" },
    ], controls.lighting, (v) => {
      controls.lighting = v as "volume" | "frame";
      apply();
    }),
  );
  froxel(
    panel.select<MsChoice>("ms", "multiple scat.", [
      { value: "off", label: "off (single)" },
      { value: "diffusion", label: "true (1D diffusion)" },
      { value: "two-sided", label: "two-sided" },
      { value: "octaves", label: "octaves" },
      { value: "neural", label: "neural (default)" },
    ], controls.ms, (v) => {
      controls.ms = v;
      apply();
    }),
  );
  const phaseRow = froxel(
    panel.select<VDBFramePhase>("phase", "phase", [
      { value: "isotropic", label: "isotropic" },
      { value: "henyey-greenstein", label: "Henyey-Greenstein" },
      { value: "dual-lobe", label: "dual-lobe (his demos)" },
      { value: "lorenz-mie", label: "Lorenz-Mie" },
      { value: "lorenz-mie-rgb", label: "Lorenz-Mie RGB" },
    ], controls.phase, (v) => {
      controls.phase = v;
      apply();
    }),
  );
  const phaseGRow = froxel(
    panel.range("phaseG", "phase g", -0.9, 0.95, 0.05, controls.phaseG, (v) => {
      controls.phaseG = v;
      apply();
    }),
  );
  const spectralRow = froxel(
    panel.checkbox("spectral", "spectral", controls.spectral, (v) => {
      controls.spectral = v;
      apply();
    }),
  );
  const mediumOptions: SelectOption<string>[] = [{ value: "none", label: "none" }, ...vdbFrameMediumNames.map((n) => ({ value: n, label: n }))];
  froxel(
    panel.select("medium", "medium", mediumOptions, controls.medium, (v) => {
      controls.medium = v;
      const m = vdbFrameMedium(v);
      if (m) {
        // A measured medium turns spectral on and sets the phase (his GUI's 'Medium' field).
        controls.spectral = true;
        controls.phase = m.phase;
        controls.phaseG = m.phaseG;
        spectralRow.set(true);
        phaseRow.set(m.phase);
        phaseGRow.set(m.phaseG);
      }
      apply();
    }),
  );
  froxel(
    panel.range("density", "density scale", 0.05, 4, 0.05, controls.density, (v) => {
      controls.density = v;
      apply();
    }),
  );

  panel.heading("Sun, sky and stage");
  panel.range("sunAzimuth", "sun azimuth", 0, 360, 1, controls.sunAzimuth, (v) => {
    controls.sunAzimuth = v;
    apply();
  }, (v) => `${v.toFixed(0)}°`);
  panel.range("sunElevation", "sun elevation", 2, 89, 1, controls.sunElevation, (v) => {
    controls.sunElevation = v;
    apply();
  }, (v) => `${v.toFixed(0)}°`);
  panel.select("sky", "sky light", [
    { value: "hemisphere", label: "hemisphere light" },
    ...(hdrAvailable ? [{ value: "hdr", label: "HDR sky (environment)" }] : []),
    { value: "none", label: "none (sun only)" },
  ], controls.sky, (v) => {
    controls.sky = v as typeof controls.sky;
    void applySky();
  });
  panel.checkbox("stage", "ground + props", controls.stage, (v) => {
    controls.stage = v;
    apply();
  });
  panel.checkbox("meshShadows", "mesh shadows", controls.meshShadows, (v) => {
    controls.meshShadows = v;
    apply();
  });

  const qualityHeading = panel.heading("Quality");
  froxelRows.push({ show: (v) => (qualityHeading.hidden = !v) });
  froxel(
    panel.select<Quality>("quality", "preset", (Object.keys(QUALITY) as Quality[]).map((q) => ({ value: q, label: QUALITY[q].label })), controls.quality, (v) => {
      controls.quality = v;
      setViewport();
      apply();
    }),
  );
  froxel(
    panel.checkbox("temporal", "temporal + jitter", controls.temporal, (v) => {
      controls.temporal = v;
      apply();
    }),
  );

  /** Show the rows that apply to the current scene. */
  function syncRows(): void {
    const isFroxel = def.kind === "froxel";
    for (const r of froxelRows) r.show(isFroxel);
    sourceRow.options(def.sourceKinds.map((k) => ({ value: k, label: k === "dense" ? "dense (.wvdb 8-bit textures)" : "sparse (pool, full res)" })));
    sourceRow.set(controls.source);
    sourceRow.show(isFroxel);
    sourceRow.enable(def.sourceKinds.length > 1);
    lodRow.show(isFroxel && loaded?.sourceKind === "sparse");
    hddaRow.show(isFroxel && loaded?.sourceKind === "sparse");
    const seq = loaded?.sequence;
    frameRow.show(!!seq);
    playRow.show(!!seq);
    if (seq) {
      (frameRow.input as HTMLInputElement).max = String(seq.count - 1);
      frameRow.set(seq.index());
      playRow.set(seq.playing);
    }
  }

  async function applySky(): Promise<void> {
    hemisphere.visible = controls.sky === "hemisphere";
    if (controls.sky === "hdr") {
      const tex = await loadHdr();
      scene.environment = tex;
      scene.background = tex;
      scene.environmentIntensity = 0.4; // his resources.js
    } else {
      scene.environment = null;
      scene.background = SKY_COLOR.clone();
    }
  }

  /**
   * The volume's density scale (the panel's, times a sequence's per-frame factor). A sparse source's density
   * uniform keeps the scale it was last given through `sourceUniforms.update` (its per-render stats refresh
   * re-applies that value after the frame renderer's per-draw one), so a change also goes through
   * `volume.updateSource()`; without it `densityScale` has no effect on a sparse source.
   */
  function setDensity(scene: LoadedScene): void {
    const volume = scene.volume;
    if (!volume) return;
    const value = controls.density * scene.densityFactor();
    if (volume.densityScale === value) return;
    volume.densityScale = value;
    if (volume.source.kind === "sparse") volume.updateSource();
  }

  let shadowsOn = false;
  /** Push the panel into the frame renderer's settings, the volume's props, the lights and the stage. */
  function apply(): void {
    const q = QUALITY[controls.quality];
    const s = frameRenderer.settings;
    s.width = q.width;
    s.height = q.height;
    s.depth = q.depth;
    s.subSteps = q.subSteps;
    s.shadowResolution = q.shadowResolution;
    s.lighting = controls.lighting;
    s.multipleScattering = MS_VALUE[controls.ms];
    s.temporal = controls.temporal;
    s.jitter = controls.temporal;
    s.spectral = controls.spectral;
    s.phase = controls.phase;
    s.phaseG = controls.phaseG;
    s.phaseBackG = -0.3;
    s.phaseBackWeight = 0.2;
    const medium = vdbFrameMedium(controls.medium);
    s.phaseGExponent = medium?.phaseGExponent ?? 0;

    const volume = loaded?.volume;
    if (volume && loaded) {
      volume.phase = controls.phase;
      volume.phaseG = controls.phaseG;
      volume.phaseGExponent = medium?.phaseGExponent ?? 0;
      volume.shadowSteps = q.shadowSteps;
      volume.hdda = controls.hdda;
      volume.baseColor.copy(medium?.baseColor ?? loaded.baseColor);
      if (medium) volume.extinctionScale.copy(medium.extinctionScale);
      else volume.extinctionScale.set(1, 1, 1);
      volume.angstromExponent = medium?.angstromExponent ?? 0;
      setDensity(loaded);
    }

    // Sun: azimuth about +y, elevation above the ground.
    const az = THREE.MathUtils.degToRad(controls.sunAzimuth);
    const el = THREE.MathUtils.degToRad(controls.sunElevation);
    sun.position.set(Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az)).multiplyScalar(300);
    sun.target.position.set(0, 0, 0);
    sun.updateMatrixWorld();
    sun.target.updateMatrixWorld();

    stage.visible = controls.stage;
    const shadows = controls.meshShadows;
    renderer.shadowMap.enabled = shadows;
    sun.castShadow = shadows;
    s.castShadows = shadows && def.kind === "froxel";
    if (shadows !== shadowsOn) {
      // the receivers' shaders change with the shadow map
      shadowsOn = shadows;
      groundMaterial.needsUpdate = true;
      propMaterial.needsUpdate = true;
    }
  }

  let firstLoad = true;
  async function switchScene(id: string): Promise<void> {
    const next = available.find((s) => s.id === id);
    if (!next) throw new Error(`unknown scene ${id}`);
    state.ready = false;
    state.loading = true;
    errEl.textContent = "";
    delete state.error;
    if (loaded) {
      content.remove(loaded.group);
      loaded.dispose();
      loaded = null;
    }
    def = next;
    sceneRow.set(def.id);
    state.scene = def.id;
    // the scene's own source kind first (bunny: its dense .wvdb), unless ?source= asks for another on the first load
    const asked = firstLoad ? (params.get("source") as SourceKind | null) : null;
    firstLoad = false;
    controls.source = asked && def.sourceKinds.includes(asked) ? asked : (def.sourceKinds[0] ?? "sparse");
    try {
      loaded = await def.load(sceneContext(), def.kind === "froxel" ? controls.source : null, controls.lod);
      content.add(loaded.group);
      state.sourceKind = loaded.sourceKind;
      syncRows();
      apply();
      renderedSinceLoad = 0;
    } catch (err) {
      fail(errorText(err));
      statusEl.textContent = "";
    } finally {
      state.loading = false;
    }
  }

  async function rebuild(): Promise<void> {
    if (!loaded?.rebuild) return;
    state.ready = false;
    state.loading = true;
    try {
      await loaded.rebuild(controls.source, controls.lod);
      state.sourceKind = loaded.sourceKind;
      syncRows();
      apply();
      renderedSinceLoad = 0;
    } catch (err) {
      fail(errorText(err));
    } finally {
      state.loading = false;
    }
  }

  // --- presentation ------------------------------------------------------------------------------------
  async function present(): Promise<void> {
    if (PRESENTATION === "direct") {
      renderer.setRenderTarget(null);
      renderer.render(scene, camera);
      await device.queue.onSubmittedWorkDone();
      return;
    }
    // Offscreen linear HalfFloat target with depth (the froxel integrate pass reads the scene depth), read back
    // (rows top first, padded to 256 bytes), sRGB-encoded on the CPU and drawn on the 2D canvas.
    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    const raw = (await renderer.readRenderTargetPixelsAsync(rt, 0, 0, vw, vh)) as unknown as Uint16Array;
    const tight = vw * 4;
    const stride = vh > 1 ? (raw.length - tight) / (vh - 1) : tight;
    if (!Number.isInteger(stride) || stride < tight) throw new Error(`readback layout: ${raw.length} halves for ${vw}x${vh}`);
    const out = new Uint8ClampedArray(vw * vh * 4);
    for (let y = 0; y < vh; y++) {
      for (let x = 0; x < vw; x++) {
        const s = y * stride + x * 4;
        const d = (y * vw + x) * 4;
        out[d] = srgbByte(THREE.DataUtils.fromHalfFloat(raw[s]!));
        out[d + 1] = srgbByte(THREE.DataUtils.fromHalfFloat(raw[s + 1]!));
        out[d + 2] = srgbByte(THREE.DataUtils.fromHalfFloat(raw[s + 2]!));
        out[d + 3] = 255;
      }
    }
    ctx2d!.putImageData(new ImageData(out, vw, vh), 0, 0);
    lastFrame = out;
  }

  function capture(): Snapshot | null {
    if (!lastFrame) return null;
    const w = vw;
    const h = vh;
    let nonBackground = 0;
    const mean: [number, number, number] = [0, 0, 0];
    const TW = 32;
    const TH = 18;
    const thumb = new Array<number>(TW * TH).fill(0);
    const counts = new Array<number>(TW * TH).fill(0);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        const r = lastFrame[i]!;
        const g = lastFrame[i + 1]!;
        const b = lastFrame[i + 2]!;
        if (Math.abs(r - SKY_BYTES[0]!) + Math.abs(g - SKY_BYTES[1]!) + Math.abs(b - SKY_BYTES[2]!) > 6) nonBackground++;
        mean[0] += r;
        mean[1] += g;
        mean[2] += b;
        const t = Math.min(TH - 1, Math.floor((y * TH) / h)) * TW + Math.min(TW - 1, Math.floor((x * TW) / w));
        thumb[t]! += (r + g + b) / 3;
        counts[t]!++;
      }
    }
    const n = w * h;
    return { width: w, height: h, nonBackground, mean: mean.map((v) => v / n) as [number, number, number], thumb: thumb.map((v, i) => v / Math.max(1, counts[i]!)) };
  }
  window.__DEMO10_API__ = { capture, setScene: switchScene };

  // --- loop ---------------------------------------------------------------------------------------------
  let busy = false;
  let renderedSinceLoad = 0;
  let frameMs = 0;
  let lastStats = 0;
  const loop = (now: number): void => {
    requestAnimationFrame(loop);
    if (busy || state.loading || !loaded || state.error) return;
    busy = true;
    orbit.update();
    loaded.tick?.(now);
    setDensity(loaded);
    const t0 = performance.now();
    present()
      .then(() => {
        const ms = performance.now() - t0;
        frameMs = frameMs === 0 ? ms : frameMs * 0.9 + ms * 0.1;
        state.frames++;
        state.frameMs = frameMs;
        renderedSinceLoad++;
        // two frames after a (re)load: the shaders are built and a settled frame is up
        if (renderedSinceLoad >= 2) state.ready = true;
        if (now - lastStats > 400 || renderedSinceLoad <= 2) {
          lastStats = now;
          const q = QUALITY[controls.quality];
          const px = PRESENTATION === "direct" ? `${Math.round(vw * q.pixelRatio)}×${Math.round(vh * q.pixelRatio)}` : `${vw}×${vh}`;
          state.resolution = px;
          state.froxels = `${q.width}×${q.height}×${q.depth}`;
          state.info = loaded?.info() ?? "";
          const seq = loaded?.sequence;
          if (seq && document.activeElement !== frameRow.input) frameRow.set(seq.index());
          statsEl.textContent =
            `${state.adapter} · ${PRESENTATION}\n` +
            `${px} px` +
            (def.kind === "froxel" ? ` · froxels ${state.froxels}, ${q.subSteps} sub-steps` : "") +
            `\nframe ${frameMs.toFixed(1)} ms (submit→done; provisional, D25)\n` +
            `${state.info}`;
        }
      })
      .catch((err: unknown) => fail(errorText(err)))
      .finally(() => {
        busy = false;
      });
  };

  await applySky();
  await switchScene(def.id);
  requestAnimationFrame(loop);
}

run().catch((err: unknown) => fail(errorText(err)));
