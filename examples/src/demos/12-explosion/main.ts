/**
 * Gallery demo 12 — Explosion. JangaFX EmberGen's ground explosion (CC0, 200 `.wvdb` frames) plays as a sequence:
 * `VDBSequence` decodes frames ahead and swaps each one in place into ONE dense source (`vdbDenseTextures`), so the
 * `VDBFrameVolume` and its shaders never change. The flames (the `flames` grid, blackbody 0–3000 K) light the ground
 * and the props through `VDBFrameEmissionLight`, and with mesh shadows on the volume shadows the meshes for the
 * key light. Dusk or night HDR sky from the git-ignored `fixtures/gallery/` when present, else a procedural sky; the
 * props are `car.glb` from there, else plain crates and barrels. Public API only.
 *
 * Query (plus the shell's `presentation`, `panel`, `still`, `gallery`): `?frame=N` (the first frame, default 30),
 * `?play=0`, `?quality=low|medium|high`, `?sky=dusk|night|procedural`, `?shadows=0|1`, `?temporal=0|1`.
 * `window.__DEMO12__` is the e2e state; `window.__DEMO12_API__.capture()` summarizes the last read-back frame.
 *
 * Frame times are submit-to-done wall times on this page: provisional (D25), not a benchmark.
 */
import "../webgpu-gate.js";
import { fixtureUrl } from "../assets.js";
import * as THREE from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { readWvdb, wvdbDenseVolume, type WvdbDenseVolume } from "@pmndrs/vdb/tools";
import { VDBFrameEmissionLight, VDBFrameRenderer, VDBFrameVolume, VDBSequence, vdbDenseTextures, vdbFrameBlackbodyEmission } from "@pmndrs/vdb/three";
import { createDeviceRenderer } from "../device-renderer.js";
import { MiB, PRESENTATION, Presenter, QUALITY, STILL, autoOrbit, initialQuality, mountShowcase, params, probe, type Quality, type Snapshot } from "../showcase-shell.js";
import { horizonColor, loadHdrSky, proceduralSky } from "../gallery-sky.js";

interface Demo12State {
  ready: boolean;
  error?: string;
  loading: boolean;
  frames: number;
  /** The sequence frame on screen (-1 before the first swap). */
  frame: number;
  frameCount?: number;
  adapter?: string;
  presentation: typeof PRESENTATION;
  sky?: string;
  props?: "car.glb" | "primitives";
  /** Gallery assets that were not found (or `?gallery=0`). */
  missing: string[];
  uncaptured: string[];
  frameMs?: number;
}

declare global {
  interface Window {
    __DEMO12__?: Demo12State;
    __DEMO12_API__?: { capture(): Snapshot | null };
  }
}

const EXPLOSION_DIR = fixtureUrl("k3/ground_explosion/");
const EXPLOSION_FPS = 30;
const GALLERY = fixtureUrl("gallery/");
const SKIES = {
  night: { url: `${GALLERY}sky-night-HDR.jpg`, label: "night", envIntensity: 0.25, bgIntensity: 0.14, key: 0x9fb4ff, keyIntensity: 0.6 },
  dusk: { url: `${GALLERY}sky-dusk-HDR.jpg`, label: "dusk", envIntensity: 0.3, bgIntensity: 0.4, key: 0xffb070, keyIntensity: 1.2 },
  procedural: { url: null, label: "procedural", envIntensity: 0.6, bgIntensity: 1.0, key: 0x9fb4ff, keyIntensity: 0.6 },
} as const;
type SkyId = keyof typeof SKIES;
const CAR_URL = `${GALLERY}car.glb`;
const GROUND = -50;
/** The last frame with flames (EmberGen's ground explosion: 0 flame voxels from frame 100 on); the loop plays 0..this. */
const FIRE_LAST_FRAME = 95;

/** Longest side of the explosion box, in scene units (his frame.js `explosionSize`). */
const SIZE = 100;
/**
 * His frame.js explosion: emissive intensity 15 ('planck'), emission light intensity 25. The gallery's default
 * "emission" is 2.5× that (a night scene), with the light a further 1.6× so the glow on the ground reads.
 */
const EMISSIVE = 15;
const LIGHT = 40;
const EMISSION_DEFAULT = 2.5;
const FOG_NEAR = 260;
const FOG_FAR = 1300;

const shell = mountShowcase({ strip: "row",
  number: "12",
  title: "Explosion",
  blurb:
    "EmberGen's ground explosion plays frame by frame through one dense source, and its flames light the ground and the cars around it: an animated, self-lit volume with no shader rebuilds.",
});

const state: Demo12State = { ready: false, loading: true, frames: 0, frame: -1, presentation: PRESENTATION, missing: [], uncaptured: [] };
window.__DEMO12__ = state;
shell.onFail((message) => {
  state.error = message;
  state.loading = false;
});

const fetchBytes = async (url: string): Promise<ArrayBuffer> => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return r.arrayBuffer();
};

/** His frame.js explosion: the emission grid "flames" ÷ emissiveScale 6; early frames have no flames grid (emission 0). */
const explosionFrame = (bytes: ArrayBuffer): WvdbDenseVolume => {
  const file = readWvdb(new Uint8Array(bytes));
  const flames = file.grids.some((g) => g.name === "flames");
  return wvdbDenseVolume(file, { ...(flames ? { emission: "flames" } : {}), scales: { flames: 6.0 } });
};

async function run(): Promise<void> {
  // --- what is here ---------------------------------------------------------------------------------------
  const framesOk = await probe(`${EXPLOSION_DIR}frames.json`);
  if (!framesOk) throw new Error("The EmberGen ground explosion is missing: copy fixtures/k3/ground_explosion/ (CC0, JangaFX) from a mjurczyk/openvdb clone (fixtures/README.md).");
  const [nightOk, duskOk, carOk] = await Promise.all([probe(SKIES.night.url), probe(SKIES.dusk.url), probe(CAR_URL)]);
  if (!nightOk) state.missing.push("sky-night-HDR.jpg");
  if (!duskOk) state.missing.push("sky-dusk-HDR.jpg");
  if (!carOk) state.missing.push("car.glb");
  if (state.missing.length) shell.note(`Gallery assets not found, using fallbacks: ${state.missing.join(", ")} (fixtures/gallery/, docs/assets-licensing.md).`);
  const skyAvailable: SkyId[] = [...(nightOk ? (["night"] as const) : []), ...(duskOk ? (["dusk"] as const) : []), "procedural"];
  const skyParam = params.get("sky") as SkyId | null;

  const controls = {
    quality: initialQuality(),
    sky: skyParam && skyAvailable.includes(skyParam) ? skyParam : skyAvailable[0]!,
    shadows: params.get("shadows") !== "0",
    emission: Number(params.get("emission") ?? EMISSION_DEFAULT),
    temporal: params.has("temporal") ? params.get("temporal") === "1" : PRESENTATION === "direct",
    playing: params.get("play") !== "0" && PRESENTATION === "direct",
  };

  // --- renderer --------------------------------------------------------------------------------------------
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
  renderer.toneMappingExposure = 1.0;
  const presenter = new Presenter(shell.app, renderer, directCanvas, device);
  const frameRenderer = new VDBFrameRenderer();

  // --- stage ----------------------------------------------------------------------------------------------
  const scene = new THREE.Scene();
  const key = new THREE.DirectionalLight(0xffffff, 1);
  key.shadow.mapSize.setScalar(2048);
  key.shadow.normalBias = 0.5;
  Object.assign(key.shadow.camera, { left: -170, right: 170, top: 170, bottom: -170, near: 1, far: 900 });
  key.shadow.camera.updateProjectionMatrix();
  // Low, from behind-left of the camera's start: long shadows across the ground towards the viewer's right.
  key.position.set(-260, 70, -170);
  scene.add(key, key.target);

  const groundMaterial = new THREE.MeshStandardMaterial({ color: 0x4a443d, roughness: 0.96, metalness: 0 });
  const ground = new THREE.Mesh(new THREE.CircleGeometry(1400, 96), groundMaterial);
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = GROUND;
  ground.receiveShadow = true;
  scene.add(ground);

  const props = new THREE.Group();
  scene.add(props);
  const receivers: THREE.Material[] = [groundMaterial];
  const setShadowFlags = (root: THREE.Object3D) =>
    root.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh) return;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      for (const m of [mesh.material].flat()) if (!receivers.includes(m)) receivers.push(m);
    });

  shell.status(carOk ? "loading car.glb…" : "");
  if (carOk) {
    const gltf = await new GLTFLoader().loadAsync(CAR_URL);
    const car = gltf.scene;
    const box = new THREE.Box3().setFromObject(car);
    const size = box.getSize(new THREE.Vector3());
    const s = 30 / Math.max(size.x, size.z);
    const place = (object: THREE.Object3D, x: number, z: number, yaw: number) => {
      const holder = new THREE.Group();
      object.scale.setScalar(s);
      const c = box.getCenter(new THREE.Vector3()).multiplyScalar(s);
      object.position.set(-c.x, GROUND - box.min.y * s, -c.z);
      holder.add(object);
      holder.position.set(x, 0, z);
      holder.rotation.y = yaw;
      props.add(holder);
    };
    place(car, 61, -16, -0.2);
    place(car.clone(true), -54, 30, 1.1);
    state.props = "car.glb";
  } else {
    // Plain props: crates and fuel barrels around the blast.
    const crate = new THREE.MeshStandardMaterial({ color: 0x8a6a48, roughness: 0.8 });
    const barrel = new THREE.MeshStandardMaterial({ color: 0x7d2a22, roughness: 0.55, metalness: 0 });
    const crates: [number, number, number, number][] = [
      [64, 1, 14, 0.3],
      [81, 3, 10, -0.2],
      [70, 19, 9, 0.6],
      [-60, -30, 16, 0.1],
    ];
    for (const [x, z, a, yaw] of crates) {
      const m = new THREE.Mesh(new THREE.BoxGeometry(a, a, a), crate);
      m.position.set(x, GROUND + a / 2, z);
      m.rotation.y = yaw;
      props.add(m);
    }
    const barrels: [number, number][] = [
      [-37, 41],
      [-41, 54],
      [-50, 13],
      [74, -26],
      [-28, 58],
    ];
    for (const [x, z] of barrels) {
      const m = new THREE.Mesh(new THREE.CylinderGeometry(4.2, 4.2, 12, 28), barrel);
      m.position.set(x, GROUND + 6, z);
      props.add(m);
    }
    state.props = "primitives";
  }
  setShadowFlags(props);

  // --- sky ------------------------------------------------------------------------------------------------
  const skyTextures = new Map<SkyId, THREE.Texture>();
  async function applySky(): Promise<void> {
    const def = SKIES[controls.sky];
    let tex = skyTextures.get(controls.sky);
    if (!tex) {
      shell.status(def.url ? `loading the ${def.label} sky…` : "");
      tex = def.url ? await loadHdrSky(def.url) : proceduralSky({ zenith: "#02040b", horizon: "#1d2a40", ground: "#07080b" });
      skyTextures.set(controls.sky, tex);
      shell.status("");
    }
    scene.background = tex;
    scene.environment = tex;
    scene.environmentIntensity = def.envIntensity;
    scene.backgroundIntensity = def.bgIntensity;
    // The ground fades into the sky's horizon colour (as the background shows it).
    const fog = horizonColor(tex)?.multiplyScalar(def.bgIntensity * 0.6) ?? new THREE.Color(0x101014);
    scene.fog = new THREE.Fog(fog, FOG_NEAR, FOG_FAR);
    key.color.setHex(def.key);
    key.intensity = def.keyIntensity;
    state.sky = controls.sky;
  }

  // --- the explosion: one dense source, frames swapped in place ---------------------------------------------
  shell.status("loading the explosion…");
  const frames = ((await (await fetch(`${EXPLOSION_DIR}frames.json`)).json()) as { frames: string[] }).frames;
  state.frameCount = frames.length;
  // The fire burns out near frame 95; the last ~105 frames are smoke only. Loop the fire part (`?loop=pingpong` turns back instead of wrapping).
  const range: [number, number] = [0, Math.min(FIRE_LAST_FRAME, frames.length - 1)];
  const loopMode = params.get("loop") === "pingpong" ? "pingpong" : "repeat";
  const start = Math.min(Math.max(range[0], Number(params.get("frame") ?? 30)), range[1]);
  const bytes = new Map<number, Promise<ArrayBuffer>>();
  const fetchFrame = (i: number): Promise<ArrayBuffer> => {
    let p = bytes.get(i);
    if (!p) {
      p = fetchBytes(EXPLOSION_DIR + frames[i]);
      p.catch(() => bytes.delete(i));
      bytes.set(i, p);
    }
    return p;
  };
  // The source is made from a frame with every channel (density and flames), as the docs say.
  const first = explosionFrame(await fetchFrame(Math.max(start, 30)));
  const source = vdbDenseTextures(first);
  const volume = new VDBFrameVolume(source, {
    frameRenderer,
    // His #555555 smoke reads as a black blot at night; a lighter grey shows the volume's shape.
    baseColor: "#8c8c8c",
    extinction: 100,
    angstromExponent: 0,
    normalizeDensity: true,
    // One mean for the whole sequence, so the extinction scale does not pump from frame to frame.
    densityMean: first.meanDensity,
    temperatureRange: [0, 3000],
    emissiveIntensity: EMISSIVE,
    emissionColor: vdbFrameBlackbodyEmission("planck", 1.39),
  });
  // Z-up box, longest side SIZE, standing on the ground, centred on the origin (his frame.js placement).
  {
    const b = volume.bbox;
    const size = b.getSize(new THREE.Vector3());
    const c = b.getCenter(new THREE.Vector3());
    const s = SIZE / Math.max(size.x, size.y, size.z);
    volume.scale.setScalar(s);
    volume.rotation.set(-Math.PI / 2, 0, 0);
    volume.position.set(-c.x * s, -c.z * s + GROUND + (size.z * s) / 2, c.y * s);
  }
  scene.add(volume);
  const light = new VDBFrameEmissionLight({ intensity: LIGHT, resolution: QUALITY[controls.quality].emission, cascades: 1, range: 3, distance: 512, steps: 24 });
  scene.add(light);

  const player = new VDBSequence<WvdbDenseVolume>({
    urls: frames.map((f) => EXPLOSION_DIR + f),
    fps: EXPLOSION_FPS,
    prefetch: 8,
    range,
    loopMode,
    loader: async (_url, index) => explosionFrame(await fetchFrame(index)),
    onError: (err, index) => console.warn(`demo 12: explosion frame ${index}: ${String(err)}`),
  });
  player.start({ source });
  player.seek(start);
  if (!controls.playing) player.stop();

  // --- camera -----------------------------------------------------------------------------------------------
  const camera = new THREE.PerspectiveCamera(42, 1, 1, 6000);
  camera.position.set(100, -20, 130);
  const orbit = new OrbitControls(camera, presenter.canvas);
  orbit.target.set(0, -4, 0);
  orbit.enableDamping = PRESENTATION === "direct";
  orbit.maxPolarAngle = Math.PI * 0.52;
  orbit.minDistance = 90;
  orbit.maxDistance = 700;
  orbit.update();
  autoOrbit(orbit, 0.3);
  const resize = () => presenter.resize(QUALITY[controls.quality].pixelRatio, camera);
  resize();
  window.addEventListener("resize", resize);

  // --- controls ---------------------------------------------------------------------------------------------
  const playButton = shell.button("play", controls.playing ? "❚❚ pause" : "▶ play", () => setPlaying(!controls.playing));
  const frameRow = shell.range("frame", "frame", range[0], range[1], 1, start, (v) => {
    setPlaying(false);
    player.seek(v);
  });
  function setPlaying(v: boolean): void {
    controls.playing = v;
    if (v) player.resume();
    else player.stop();
    playButton.set(v ? "❚❚ pause" : "▶ play");
  }
  shell.range("emission", "emission", 0, 5, 0.1, controls.emission, (v) => {
    controls.emission = v;
    apply();
  }, (v) => `${v.toFixed(1)}×`);
  shell.checkbox("shadows", "mesh shadows", controls.shadows, (v) => {
    controls.shadows = v;
    apply();
  });
  shell.select<SkyId>("sky", "sky", skyAvailable.map((id) => ({ value: id, label: SKIES[id].url ? `${SKIES[id].label} HDR` : "procedural" })), controls.sky, (v) => {
    controls.sky = v;
    void applySky().catch((err: unknown) => shell.fail(String(err)));
  });
  shell.select<Quality>("quality", "quality", (["low", "medium", "high"] as const).map((q) => ({ value: q, label: QUALITY[q].label })), controls.quality, (v) => {
    controls.quality = v;
    resize();
    apply();
  });

  let shadowsOn: boolean | null = null;
  function apply(): void {
    const q = QUALITY[controls.quality];
    const s = frameRenderer.settings;
    s.width = q.width;
    s.height = q.height;
    s.depth = q.depth;
    s.subSteps = q.subSteps;
    s.shadowResolution = q.shadowResolution;
    s.lighting = "frame";
    s.multipleScattering = "neural";
    s.temporal = controls.temporal;
    s.jitter = controls.temporal;
    s.phase = "dual-lobe";
    s.phaseG = 0.8;
    s.phaseBackG = -0.3;
    s.phaseBackWeight = 0.2;
    volume.shadowSteps = q.shadowSteps;
    volume.emissiveIntensity = EMISSIVE * controls.emission;
    light.intensity = LIGHT * controls.emission;
    light.visible = controls.emission > 0;
    light.resolution = q.emission;

    const shadows = controls.shadows;
    renderer.shadowMap.enabled = shadows;
    key.castShadow = shadows;
    s.castShadows = shadows;
    if (shadows !== shadowsOn) {
      shadowsOn = shadows;
      for (const m of receivers) m.needsUpdate = true; // the receivers' shaders change with the shadow map
    }
  }

  window.__DEMO12_API__ = { capture: () => presenter.capture() };
  // `?debug=1`: the scene's pieces, for tuning from the console.
  if (params.get("debug") === "1") Object.assign(window, { __demo12: { THREE, scene, camera, orbit, volume, light, key, ground, props, controls, apply, applySky, renderer, frameRenderer, SKIES } });

  // --- loop -------------------------------------------------------------------------------------------------
  await applySky();
  apply();
  shell.status("");
  state.loading = false;

  let busy = false;
  let rendered = 0;
  let frameMs = 0;
  let lastStats = 0;
  const loop = (now: number): void => {
    requestAnimationFrame(loop);
    if (busy || state.error) return;
    busy = true;
    orbit.update();
    player.update(now);
    const shown = player.stats.frame;
    const t0 = performance.now();
    presenter
      .present(scene, camera)
      .then(() => {
        const ms = performance.now() - t0;
        frameMs = frameMs === 0 ? ms : frameMs * 0.9 + ms * 0.1;
        rendered++;
        state.frames++;
        state.frameMs = frameMs;
        state.frame = shown;
        // Two frames after the first swap: the shaders are built and a settled frame is up.
        if (rendered >= 2 && shown >= 0) state.ready = true;
        if (now - lastStats > 400 || rendered <= 2) {
          lastStats = now;
          if (shown >= 0 && document.activeElement !== frameRow.input) frameRow.set(shown);
          const dims = source.volume.dimensions;
          const texBytes = dims[0] * dims[1] * dims[2] * (source.volume.emission ? 2 : 1);
          shell.stats(
            `${state.adapter} · ${presenter.width}×${presenter.height} px · ${frameMs.toFixed(1)} ms/frame (provisional, D25)\n` +
            `dense ${dims.join("×")} r8 density${source.volume.emission ? " + flames" : ""}, ${MiB(texBytes)} on the GPU · frame ${Math.max(0, shown)} / ${frames.length}`,
          );
        }
      })
      .catch((err: unknown) => shell.fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err)))
      .finally(() => {
        busy = false;
      });
  };
  requestAnimationFrame(loop);
  if (STILL) orbit.autoRotate = false;
}

run().catch((err: unknown) => shell.fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err)));
