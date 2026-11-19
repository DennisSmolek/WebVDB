/**
 * Gallery demo 11 — sky clouds. The WDAS quarter cloud at full resolution from the sparse pool (24 M active voxels,
 * Fp8, no resampling) leads a skyscape of three JangaFX CloudPack clouds (dense 84×128×43 `.wvdb`, CC0), all in one
 * `VDBFrameRenderer` (neural multiple scattering, temporal; 'volume' lighting by default, see `controls.lighting`), under an HDR sky with a time-of-day
 * selector and a sun whose colour follows its elevation. Public API only (`@pmndrs/vdb`, `/tools`, `/three`).
 *
 * Gallery assets (git-ignored `fixtures/gallery/`, licences in docs/assets-licensing.md): `sky-{midday,dusk,cloudy}-HDR.jpg`
 * and `CloudPackVDB_wvdb/cloud_0{2,5,7}_variant_0000.wvdb`. Without them the page draws a procedural sky and the WDAS
 * cloud alone, and says what is missing. The WDAS grid is a fixture (`pnpm fixtures:wdas`, then Fp8, fixtures/README.md).
 *
 * Query: the gallery shell's (`?presentation=readback`, `?still=1`, `?panel=0`, `?gallery=0`, `?quality=`), plus
 * `?time=midday|dusk|cloudy`, `?elevation=<deg>`, `?lighting=volume|frame`, `?ms=neural|octaves|off`, `?temporal=0|1`. `window.__DEMO11__` is
 * the e2e state and `window.__DEMO11_API__.capture()` summarizes the last read-back frame.
 *
 * Frame times are submit-to-done wall times on this page: provisional (D25), not a benchmark.
 */
import * as THREE from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { UltraHDRLoader } from "three/addons/loaders/UltraHDRLoader.js";
import { VDBGridPool } from "@pmndrs/vdb";
import { NvdbFile, readWvdb, wvdbDenseVolume } from "@pmndrs/vdb/tools";
import {
  VDBFrameRenderer,
  VDBFrameVolume,
  VDBGridNode,
  getDevice,
  vdbDenseTextures,
  vdbSparseSource,
  type VDBFrameMultipleScattering,
  type VDBThreeVolumeSource,
} from "@pmndrs/vdb/three";
import { createDeviceRenderer } from "../device-renderer.js";
import { GALLERY_DIR, mountShowcase, PRESENTATION, Presenter, autoOrbit, fetchBytes, formatBytes, params, probe, qualityParam, type FrameSnapshot, type Quality } from "../showcase-shell.js";
import { TIMES, hdrSunAzimuth, proceduralSky, sunColor, sunDirection, type TimeOfDay } from "./sky.js";

// --- State the e2e spec reads ------------------------------------------------------------------------------

interface Demo11State {
  ready: boolean;
  error?: string;
  frames: number;
  frameMs?: number;
  adapter?: string;
  presentation: "direct" | "readback";
  /** Which inputs were found. */
  assets?: { wdas: boolean; cloudPack: number; hdr: string[] };
  sky?: "hdr" | "procedural";
  volumes?: number;
  loading: boolean;
  uncaptured: string[];
}

declare global {
  interface Window {
    __DEMO11__?: Demo11State;
    __DEMO11_API__?: { capture(): FrameSnapshot | null };
  }
}

const state: Demo11State = { ready: false, frames: 0, loading: true, uncaptured: [], presentation: PRESENTATION };
window.__DEMO11__ = state;

const shell = mountShowcase({
  number: "11",
  title: "Sky clouds",
  blurb:
    "The WDAS cloud at full resolution — 24 M voxels straight from the sparse NanoVDB pool, no resampling — leads three JangaFX clouds under an HDR sky. Sparse means the wisps and hard edges stay sharp at any distance.",
});
shell.onFail((m) => {
  state.error = m;
  state.loading = false;
});

const WDAS_URL = "/fixtures/wdas/wdas_cloud_quarter_fp8.nvdb";
const CLOUD_PACK = ["cloud_02", "cloud_05", "cloud_07"].map((n) => `${GALLERY_DIR}CloudPackVDB_wvdb/${n}_variant_0000.wvdb`);
const hdrUrl = (name: string) => `${GALLERY_DIR}${name}-HDR.jpg`;

const QUALITY: Record<Quality, { width: number; height: number; depth: number; subSteps: number; shadowSteps: number; shadowResolution: number; pixelRatio: number; label: string }> = {
  low: { width: 128, height: 128, depth: 16, subSteps: 8, shadowSteps: 8, shadowResolution: 48, pixelRatio: 1, label: "low" },
  medium: { width: 256, height: 256, depth: 32, subSteps: 16, shadowSteps: 16, shadowResolution: 64, pixelRatio: 1, label: "medium" },
  high: { width: 512, height: 512, depth: 32, subSteps: 32, shadowSteps: 32, shadowResolution: 96, pixelRatio: Math.min(2, window.devicePixelRatio || 1), label: "high" },
};

type MsChoice = "neural" | "octaves" | "off";
const MS_VALUE: Record<MsChoice, VDBFrameMultipleScattering> = { neural: "neural", octaves: "octaves", off: false };

const timeParam = params.get("time");
const msParam = params.get("ms");
const controls = {
  /**
   * 'volume' by default here, not the gallery's 'frame': the frame lighting's one shadow grid spans the whole
   * skyscape (~3 km at 64–96 voxels, ~30 WDAS voxels per shadow voxel), which flattens the WDAS billows; each
   * volume's own shadow march reads the sparse grid at full resolution.
   */
  lighting: (params.get("lighting") === "frame" ? "frame" : "volume") as "volume" | "frame",
  time: (timeParam === "dusk" || timeParam === "cloudy" ? timeParam : "midday") as TimeOfDay,
  elevation: NaN,
  ms: (msParam === "octaves" || msParam === "off" ? msParam : "neural") as MsChoice,
  quality: qualityParam(PRESENTATION === "readback" ? "low" : "medium"),
  temporal: params.has("temporal") ? params.get("temporal") === "1" : PRESENTATION === "direct",
};
controls.elevation = Number(params.get("elevation") ?? TIMES[controls.time].elevation);

/**
 * Where each cloud sits (world units; the WDAS grid's own units, ~0.83 per voxel). The camera stands near the
 * ground and looks up across the clouds; the HDR horizon stays in the lower part of the frame.
 */
const LAYOUT = {
  wdas: { center: new THREE.Vector3(-150, 320, -820), size: 640, yaw: 0.5 },
  pack: [
    { center: new THREE.Vector3(820, 430, -1300), size: 780, yaw: -0.7 },
    { center: new THREE.Vector3(-1250, 560, -2000), size: 900, yaw: 0.35 },
    { center: new THREE.Vector3(420, 860, -2700), size: 1100, yaw: 1.3 },
  ],
};
const CAMERA_POSITION = new THREE.Vector3(0, 10, 60);
const CAMERA_TARGET = new THREE.Vector3(-20, 340, -900);
/**
 * The key light's azimuth (degrees; `sunDirection`'s convention, -90° is the view's -z): front-right of the view and
 * out of frame, so the clouds are back-lit with bright rims. An HDR sky is turned to put its own sun there.
 */
const SUN_AZIMUTH = -42;

/** Put a volume in a slot: its bbox centre at `center`, longest side `size`, turned by `yaw`; Z-up files stood up. */
function place(volume: VDBFrameVolume, center: THREE.Vector3, size: number, yaw: number, zUp: boolean): THREE.Group {
  const slot = new THREE.Group();
  const up = new THREE.Group();
  const box = volume.bbox;
  const extent = box.getSize(new THREE.Vector3());
  slot.position.copy(center);
  slot.rotation.y = yaw;
  slot.scale.setScalar(size / Math.max(extent.x, extent.y, extent.z));
  up.rotation.x = zUp ? -Math.PI / 2 : 0;
  volume.position.copy(box.getCenter(new THREE.Vector3())).negate();
  up.add(volume);
  slot.add(up);
  return slot;
}

async function run(): Promise<void> {
  // --- what is here -------------------------------------------------------------------------------------
  shell.status("looking for the assets…");
  const [wdas, ...found] = await Promise.all([probe(WDAS_URL), ...CLOUD_PACK.map(probe), ...(Object.keys(TIMES) as TimeOfDay[]).map((t) => probe(hdrUrl(TIMES[t].hdr)))]);
  const packFound = found.slice(0, CLOUD_PACK.length);
  const hdrFound = (Object.keys(TIMES) as TimeOfDay[]).filter((_, i) => found[CLOUD_PACK.length + i]);
  const packUrls = CLOUD_PACK.filter((_, i) => packFound[i]);
  state.assets = { wdas: wdas!, cloudPack: packUrls.length, hdr: hdrFound };
  const missing: string[] = [];
  if (!wdas) missing.push("WDAS cloud (fixtures/wdas/wdas_cloud_quarter_fp8.nvdb: pnpm fixtures:wdas, then Fp8, fixtures/README.md)");
  if (packUrls.length < CLOUD_PACK.length) missing.push("JangaFX CloudPack clouds (fixtures/gallery/CloudPackVDB_wvdb/)");
  if (hdrFound.length < 3) missing.push(`HDR skies (fixtures/gallery/sky-*-HDR.jpg): ${hdrFound.length ? "some times use" : "using"} a procedural sky`);
  shell.note(missing.length ? `Missing, so not shown:\n${missing.map((m) => `• ${m}`).join("\n")}` : "");
  if (!wdas && packUrls.length === 0) throw new Error("No cloud to draw: neither the WDAS grid nor fixtures/gallery/CloudPackVDB_wvdb/ is present.");

  // --- renderer -----------------------------------------------------------------------------------------
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
  const presenter = new Presenter(shell.app, renderer, directCanvas);
  const rt = PRESENTATION === "readback" ? presenter.target() : null;

  const frameRenderer = new VDBFrameRenderer();
  const scene = new THREE.Scene();
  const sun = new THREE.DirectionalLight(0xffffff, 3);
  scene.add(sun, sun.target);

  // --- the clouds ---------------------------------------------------------------------------------------
  const volumes: VDBFrameVolume[] = [];
  const sources: VDBThreeVolumeSource[] = [];
  const cleanup: (() => void)[] = [];
  const info: string[] = [];
  const common = { frameRenderer, extinction: 100, angstromExponent: 0, normalizeDensity: true } as const;

  if (wdas) {
    shell.status("loading the WDAS cloud (28 MB, Fp8)…");
    const file = NvdbFile.fromArrayBuffer(await fetchBytes(WDAS_URL));
    const image = file.gridImage(0);
    const voxels = file.grids[0]?.voxelCount ?? 0;
    const pool = await VDBGridPool.create(getDevice(renderer), { label: "demo-11" });
    const id = pool.add(image);
    pool.flush();
    await pool.settled();
    const grids = new VDBGridNode(renderer, pool, { profile: "march", helpers: ["cellBoundTrilinear", "hddaCellEnd", "nodeAverage", "fastTrilinear"] });
    const source = vdbSparseSource(grids, { density: id }, { lod: "coarse-pyramid", images: { density: image } });
    const volume = new VDBFrameVolume(source, { ...common, extinction: 160 });
    scene.add(place(volume, LAYOUT.wdas.center, LAYOUT.wdas.size, LAYOUT.wdas.yaw, false));
    volumes.push(volume);
    sources.push(source);
    cleanup.push(() => {
      grids.dispose();
      pool.dispose();
    });
    const gi = pool.info(id);
    info.push(`WDAS ${(voxels / 1e6).toFixed(1)} M voxels, sparse ${formatBytes(gi.imageBytes)}`);
  }
  if (packUrls.length) {
    shell.status(`loading ${packUrls.length} JangaFX clouds…`);
    const dense = await Promise.all(packUrls.map(async (url) => wvdbDenseVolume(readWvdb(new Uint8Array(await fetchBytes(url))))));
    let bytes = 0;
    dense.forEach((d, i) => {
      const source = vdbDenseTextures(d);
      const volume = new VDBFrameVolume(source, common);
      const slot = LAYOUT.pack[i]!;
      scene.add(place(volume, slot.center, slot.size, slot.yaw, true));
      volumes.push(volume);
      sources.push(source);
      bytes += d.dimensions[0] * d.dimensions[1] * d.dimensions[2] + (d.density.mip?.data.byteLength ?? 0);
    });
    info.push(`${dense.length} × dense ${dense[0]!.dimensions.join("×")} .wvdb, ${formatBytes(bytes)}`);
  }
  state.volumes = volumes.length;

  // --- the sky ------------------------------------------------------------------------------------------
  const hdrCache = new Map<TimeOfDay, Promise<THREE.Texture>>();
  const loadHdr = (t: TimeOfDay): Promise<THREE.Texture> => {
    let p = hdrCache.get(t);
    if (!p) {
      p = new UltraHDRLoader().loadAsync(hdrUrl(TIMES[t].hdr)).then((tex) => {
        tex.mapping = THREE.EquirectangularReflectionMapping;
        return tex;
      });
      hdrCache.set(t, p);
    }
    return p;
  };
  let procedural: THREE.DataTexture | null = null;
  let skyDirty = false;
  let skyBuiltAt = 0;
  let skyToken = 0;

  /** Sky texture, environment and the sun's azimuth for the time of day (HDR when present). */
  async function applySky(): Promise<void> {
    const token = ++skyToken;
    const preset = TIMES[controls.time];
    if (hdrFound.includes(controls.time)) {
      shell.status(`loading the ${preset.label} sky…`);
      const tex = await loadHdr(controls.time);
      if (token !== skyToken) return;
      scene.background = tex;
      scene.environment = tex;
      scene.environmentIntensity = preset.environment;
      scene.backgroundIntensity = preset.background;
      // Turn the photo so its sun stands where the key light is (front-right, out of frame). three's equirect
      // rotation turns the lookup direction, so the photo turns by (its sun - ours).
      const photoSun = hdrSunAzimuth(tex);
      const turn = photoSun === null ? 0 : THREE.MathUtils.degToRad(photoSun - SUN_AZIMUTH);
      scene.backgroundRotation.set(0, turn, 0);
      scene.environmentRotation.set(0, turn, 0);
      scene.backgroundBlurriness = 0.03;
      state.sky = "hdr";
      shell.status("");
    } else {
      scene.backgroundRotation.set(0, 0, 0);
      scene.environmentRotation.set(0, 0, 0);
      scene.backgroundBlurriness = 0;
      scene.backgroundIntensity = 1;
      state.sky = "procedural";
      rebuildProcedural();
      scene.environmentIntensity = preset.proceduralEnvironment;
    }
    renderer.toneMappingExposure = preset.exposure;
    applySun();
  }

  /** The procedural sky follows the sun; the HDR keeps its own. */
  function rebuildProcedural(): void {
    const old = procedural;
    procedural = proceduralSky(controls.time, SUN_AZIMUTH, controls.elevation);
    scene.background = procedural;
    scene.environment = procedural;
    old?.dispose();
  }

  function applySun(): void {
    const { color, brightness } = sunColor(controls.elevation);
    sun.color.copy(color);
    sun.intensity = 4 * brightness * TIMES[controls.time].sun;
    sun.position.copy(sunDirection(SUN_AZIMUTH, controls.elevation)).multiplyScalar(3000).add(LAYOUT.wdas.center);
    sun.target.position.copy(LAYOUT.wdas.center);
    sun.updateMatrixWorld();
    sun.target.updateMatrixWorld();
  }

  function applySettings(): void {
    const q = QUALITY[controls.quality];
    const s = frameRenderer.settings;
    s.lighting = controls.lighting;
    s.width = q.width;
    s.height = q.height;
    s.depth = q.depth;
    s.subSteps = q.subSteps;
    s.shadowResolution = q.shadowResolution;
    s.multipleScattering = MS_VALUE[controls.ms];
    s.temporal = controls.temporal;
    s.jitter = controls.temporal;
    // A soft environment (sky light from every side) next to the sun key light: SH lights, not the photo's sun again.
    s.environmentLighting = "spherical-harmonics";
    s.environmentShOrder = 2;
    s.phase = "dual-lobe";
    s.phaseG = 0.8;
    s.phaseBackG = -0.3;
    s.phaseBackWeight = 0.2;
    for (const v of volumes) v.shadowSteps = q.shadowSteps;
  }

  // --- camera -------------------------------------------------------------------------------------------
  const camera = new THREE.PerspectiveCamera(55, 1, 5, 20000);
  camera.position.copy(CAMERA_POSITION);
  const orbit = new OrbitControls(camera, presenter.canvas);
  orbit.target.copy(CAMERA_TARGET);
  orbit.enableDamping = PRESENTATION === "direct";
  orbit.enablePan = false;
  orbit.minDistance = 400;
  orbit.maxDistance = 2600;
  // Stay under the clouds, looking up (polar angle from +y: the camera sits below its target).
  orbit.minPolarAngle = Math.PI * 0.5;
  orbit.maxPolarAngle = Math.PI * 0.62;
  orbit.update();
  autoOrbit(orbit, 0.25);

  const resize = (): void => presenter.resize(QUALITY[controls.quality].pixelRatio, camera);
  resize();
  window.addEventListener("resize", resize);

  // --- controls -----------------------------------------------------------------------------------------
  shell.select<TimeOfDay>(
    "time",
    "time of day",
    (Object.keys(TIMES) as TimeOfDay[]).map((t) => ({ value: t, label: `${TIMES[t].label}${hdrFound.includes(t) ? "" : " (procedural)"}` })),
    controls.time,
    (v) => {
      controls.time = v;
      controls.elevation = TIMES[v].elevation;
      elevationRow.set(controls.elevation);
      void applySky();
    },
  );
  const elevationRow = shell.range("elevation", "sun elevation", 1, 80, 1, controls.elevation, (v) => {
    controls.elevation = v;
    // The procedural sky follows the sun, rebuilt from the loop at most every 120 ms while the slider moves.
    if (state.sky === "procedural") skyDirty = true;
    applySun();
  }, (v) => `${v.toFixed(0)}°`);
  shell.select<"volume" | "frame">("lighting", "lighting", [
    { value: "volume", label: "full-res self-shadow" },
    { value: "frame", label: "shared shadow grid" },
  ], controls.lighting, (v) => {
    controls.lighting = v;
    applySettings();
  });
  shell.select<MsChoice>("ms", "scattering", [
    { value: "neural", label: "neural (default)" },
    { value: "octaves", label: "octaves" },
    { value: "off", label: "single only" },
  ], controls.ms, (v) => {
    controls.ms = v;
    applySettings();
  });
  shell.select<Quality>("quality", "quality", [
    { value: "low", label: "low (128² × 16)" },
    { value: "medium", label: "medium (256² × 32)" },
    { value: "high", label: "high (512² × 32)" },
  ], controls.quality, (v) => {
    controls.quality = v;
    resize();
    applySettings();
  });

  // --- loop ---------------------------------------------------------------------------------------------
  window.__DEMO11_API__ = { capture: () => presenter.capture() };
  applySettings();
  await applySky();
  scene.updateMatrixWorld(true);
  state.loading = false;
  shell.status("");

  let busy = false;
  let frameMs = 0;
  let lastStats = 0;
  const totalBytes = info.join(" · ");
  const loop = (now: number): void => {
    requestAnimationFrame(loop);
    if (busy || state.error) return;
    busy = true;
    if (skyDirty && now - skyBuiltAt > 120) {
      skyDirty = false;
      skyBuiltAt = now;
      rebuildProcedural();
    }
    orbit.update();
    const t0 = performance.now();
    const frame = async (): Promise<void> => {
      if (rt) {
        renderer.setRenderTarget(rt);
        renderer.render(scene, camera);
        renderer.setRenderTarget(null);
        presenter.show(await presenter.readTarget(rt));
      } else {
        renderer.render(scene, camera);
        await device.queue.onSubmittedWorkDone();
      }
    };
    frame()
      .then(() => {
        const ms = performance.now() - t0;
        frameMs = frameMs === 0 ? ms : frameMs * 0.9 + ms * 0.1;
        state.frames++;
        state.frameMs = frameMs;
        if (state.frames >= 2) state.ready = true;
        if (now - lastStats > 500 || state.frames <= 2) {
          lastStats = now;
          const q = QUALITY[controls.quality];
          shell.stats(
            `${state.adapter} · ${presenter.pixelWidth}×${presenter.pixelHeight} · froxels ${q.width}×${q.height}×${q.depth}\n` +
              `${frameMs.toFixed(1)} ms/frame (provisional, D25)\n${totalBytes}`,
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
    for (const v of volumes) v.dispose();
    for (const s of sources) s.dispose();
    for (const c of cleanup) c();
  });
}

run().catch((err: unknown) => shell.fail(err));
