/**
 * Demo 13 · Level sets — a small sculpture gallery: the teapot, the Venus statue, the bunny and the icosahedron,
 * OpenVDB level sets parsed in the page and traced at full resolution on the sparse NanoVDB tree
 * (`VDBLevelSetMaterial`), standing on plinths, lit by an HDR environment (or a procedural studio), casting and
 * receiving three.js shadows. Public API only (`@pmndrs/vdb`, `/tools`, `/three`) plus three.
 *
 * Query (with the shell's `presentation`, `still`, `panel`, `gallery`): `?model=all|icosahedron|teapot|venus|bunny`
 * (the piece in focus), `?load=a,b` (load only these; e2e), `?material=clay|porcelain|bronze|chrome|glass`,
 * `?env=studio|shore|townsquare`, `?normals=1`.
 */
import "../webgpu-gate.js";
import { fixtureUrl } from "../assets.js";
import * as THREE from "three/webgpu";
import * as TSL from "three/tsl";
import { GroundedSkybox } from "three/addons/objects/GroundedSkybox.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { UltraHDRLoader } from "three/addons/loaders/UltraHDRLoader.js";
import type { VDBTslNode } from "@pmndrs/vdb/three";
import { GALLERY_ASSETS, MiB, PRESENTATION, createViewer, errorText, mountShowcase, params, probe } from "../showcase-shell.js";
import { SCULPTURES, Sculpture, resolveSculpture, type SculptureDef } from "./sculptures.js";

const { viewportSharedTexture, screenUV, vec3 } = TSL as unknown as Record<string, VDBTslNode>;

const ui = mountShowcase({
  number: "13",
  title: "Level sets",
  blurb:
    "OpenVDB level sets read by our own parser and ray traced on the GPU at full resolution, straight from the sparse NanoVDB tree: no meshing, no dense bake, three.js lights and shadows.",
  stateKey: "__DEMO13__",
});
const state = ui.state;

// --- Materials -------------------------------------------------------------------------------------------------

type MaterialId = "clay" | "porcelain" | "bronze" | "chrome" | "glass";
const MATERIALS: Record<MaterialId, { label: string; color: number; roughness: number; metalness: number }> = {
  clay: { label: "matte clay", color: 0xcfc6b6, roughness: 0.85, metalness: 0 },
  porcelain: { label: "porcelain", color: 0xf4f1ea, roughness: 0.16, metalness: 0 },
  bronze: { label: "polished bronze", color: 0xd9a066, roughness: 0.28, metalness: 1 },
  chrome: { label: "chrome", color: 0xf2f4f7, roughness: 0.04, metalness: 1 },
  glass: { label: "glass (screen-space)", color: 0x000000, roughness: 0.03, metalness: 0 },
};

// --- Environments ----------------------------------------------------------------------------------------------

type EnvId = "studio" | "shore" | "townsquare";
interface EnvDef {
  label: string;
  url?: string;
  /** Sun position in the equirect image (0–1, y down) — the shadow light points from there. */
  sun: [number, number];
  sunIntensity: number;
  envIntensity: number;
  exposure: number;
  /** Ground-projection height of the photo's camera (m). */
  height?: number;
  shadowRadius: number;
}
const ENVS: Record<EnvId, EnvDef> = {
  studio: { label: "studio (procedural)", sun: [0.62, 0.22], sunIntensity: 2.6, envIntensity: 0.55, exposure: 1.0, shadowRadius: 3 },
  shore: { label: "shore (HDR)", url: fixtureUrl("gallery/shore-HDR.jpg"), sun: [0.6, 0.34], sunIntensity: 3.2, envIntensity: 1.0, exposure: 0.85, height: 1.7, shadowRadius: 2 },
  townsquare: { label: "town square (HDR)", url: fixtureUrl("gallery/townsquare-HDR.jpg"), sun: [0.55, 0.12], sunIntensity: 0.9, envIntensity: 1.1, exposure: 1.0, height: 1.7, shadowRadius: 8 },
};

/** Direction toward an equirect pixel (three's `equirectUV` convention, u along +x → +z, v up). */
function equirectDirection(u: number, vDown: number): THREE.Vector3 {
  const az = (u - 0.5) * 2 * Math.PI;
  const el = (0.5 - vDown) * Math.PI;
  return new THREE.Vector3(Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az));
}

// --- Layout ----------------------------------------------------------------------------------------------------

const PLINTH_HEIGHT = 0.95;
const PLINTH_GAP = 1.75;
const DAIS_RADIUS = 3.35;
const plinthBase = (i: number, n: number): THREE.Vector3 => {
  const k = i - (n - 1) / 2;
  return new THREE.Vector3(k * PLINTH_GAP, PLINTH_HEIGHT, -0.32 * k * k);
};

async function run(): Promise<void> {
  // --- what can load -----------------------------------------------------------------------------------------
  const wanted = params.get("load")?.split(",").filter(Boolean);
  const defs = SCULPTURES.filter((d) => !wanted || wanted.includes(d.id));
  const resolved = await Promise.all(defs.map(async (d) => [d, await resolveSculpture(d)] as const));
  const present = resolved.filter((r): r is readonly [SculptureDef, string] => r[1] !== null);
  const missing = resolved.filter(([, u]) => u === null).map(([d]) => d);
  if (present.length === 0) throw new Error("No level-set fixtures found. Run `pnpm fixtures:vdb-samples` (fixtures/README.md).");
  const envAvailable: EnvId[] = ["studio"];
  if (GALLERY_ASSETS) {
    for (const id of ["shore", "townsquare"] as const) if (await probe(ENVS[id].url!)) envAvailable.push(id);
  }
  const notes: string[] = [];
  if (missing.length) notes.push(`Not shown (fixture missing): ${missing.map((d) => d.label).join(", ")}.`);
  if (envAvailable.length === 1)
    notes.push(GALLERY_ASSETS ? "HDR skies not found (fixtures/gallery/, local only): procedural studio lighting." : "Gallery assets off (?gallery=0): procedural studio lighting.");
  ui.note(notes.join("\n"));

  // --- viewer --------------------------------------------------------------------------------------------------
  const viewer = await createViewer(ui, { fov: 38, near: 0.05, far: 200, exposure: 1, motion: "sway", swayDegrees: 16, speed: 0.1 });
  const { renderer, scene, camera, orbit, content } = viewer;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  orbit.maxPolarAngle = Math.PI * 0.49;
  orbit.minDistance = 0.8;
  orbit.maxDistance = 14;

  // Lights: one shadow-casting sun (aimed from the sky's sun), a faint hemisphere fill for the studio.
  const sun = new THREE.DirectionalLight(0xffffff, 3);
  sun.castShadow = true;
  sun.shadow.mapSize.setScalar(PRESENTATION === "readback" ? 1024 : 2048);
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.01;
  Object.assign(sun.shadow.camera, { left: -4.6, right: 4.6, top: 3.2, bottom: -2.2, near: 0.5, far: 30 });
  sun.shadow.camera.updateProjectionMatrix();
  // The gallery is static: the shadow map (a level-set trace per texel the proxies cover) renders only after a change.
  sun.shadow.autoUpdate = false;
  let shadowFrames = 0;
  const refreshShadows = (): void => {
    shadowFrames = 3;
  };
  scene.add(sun, sun.target);
  const fill = new THREE.HemisphereLight(0xdfe6ff, 0x3b342c, 0.35);
  scene.add(fill);

  // Plinths and floor.
  const stone = new THREE.MeshStandardMaterial({ color: 0xe6e0d6, roughness: 0.6, metalness: 0 });
  const plinthTrim = new THREE.MeshStandardMaterial({ color: 0x3b3631, roughness: 0.6, metalness: 0 });
  const plinths = new THREE.Group();
  const n = present.length;
  for (let i = 0; i < n; i++) {
    const b = plinthBase(i, n);
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.78, PLINTH_HEIGHT - 0.06, 0.78), stone);
    body.position.set(b.x, (PLINTH_HEIGHT - 0.06) / 2 + 0.03, b.z);
    const cap = new THREE.Mesh(new THREE.BoxGeometry(0.86, 0.03, 0.86), plinthTrim);
    cap.position.set(b.x, PLINTH_HEIGHT - 0.015, b.z);
    const foot = new THREE.Mesh(new THREE.BoxGeometry(0.84, 0.03, 0.84), plinthTrim);
    foot.position.set(b.x, 0.015, b.z);
    for (const m of [body, cap, foot]) {
      m.castShadow = true;
      m.receiveShadow = true;
      plinths.add(m);
    }
  }
  content.add(plinths);
  const STUDIO_HORIZON = 0x5a5f6b;
  const studioFloor = new THREE.Mesh(new THREE.CircleGeometry(60, 96), new THREE.MeshStandardMaterial({ color: 0x6b6f78, roughness: 0.94, metalness: 0 }));
  studioFloor.rotation.x = -Math.PI / 2;
  studioFloor.position.y = -0.06;
  studioFloor.receiveShadow = true;
  // A low stone dais under the plinths; outside it, a shadow catcher on the projected ground (HDR skies).
  const dais = new THREE.Mesh(new THREE.CylinderGeometry(DAIS_RADIUS, DAIS_RADIUS + 0.04, 0.06, 128), new THREE.MeshStandardMaterial({ color: 0x3a3836, roughness: 0.32, metalness: 0 }));
  dais.position.y = -0.03;
  dais.receiveShadow = true;
  content.add(dais);
  const shadowCatcher = new THREE.Mesh(new THREE.RingGeometry(DAIS_RADIUS + 0.03, 14, 128, 1), new THREE.ShadowMaterial({ opacity: 0.55, color: 0x101418 }));
  shadowCatcher.rotation.x = -Math.PI / 2;
  shadowCatcher.position.y = -0.005;
  shadowCatcher.receiveShadow = true;
  scene.add(studioFloor, shadowCatcher);

  // Studio: three's RoomEnvironment through PMREM (no asset), a sky gradient by elevation (horizon to zenith) and
  // fog of the horizon colour, so the floor runs into the backdrop without a seam.
  const pmrem = new THREE.PMREMGenerator(renderer);
  const studioEnv = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  const studioBackdrop = TSL.mix(TSL.color(STUDIO_HORIZON), TSL.color(0x15171d), TSL.smoothstep(0.0, 0.6, (TSL.normalWorld as VDBTslNode).y.max(0)));
  const hdr = new Map<EnvId, THREE.Texture>();
  let skybox: GroundedSkybox | null = null;

  async function setEnvironment(id: EnvId): Promise<void> {
    const def = ENVS[id];
    if (def.url) {
      let tex = hdr.get(id);
      if (!tex) {
        ui.status(`loading the ${def.label} sky…`, null);
        tex = await new UltraHDRLoader().loadAsync(def.url);
        tex.mapping = THREE.EquirectangularReflectionMapping;
        hdr.set(id, tex);
      }
      if (skybox) {
        scene.remove(skybox);
        skybox.geometry.dispose();
        skybox.material.dispose();
      }
      skybox = new GroundedSkybox(tex, def.height!, 60);
      // Ground at y = -0.01, under the stone dais (which also hides the projection's nadir, the photo's tripod).
      skybox.position.set(0, def.height! - 0.01, 0);
      scene.add(skybox);
      scene.environment = tex;
      scene.background = null;
      scene.backgroundNode = null;
      scene.fog = null;
      studioFloor.visible = false;
      shadowCatcher.visible = true;
      fill.visible = false;
    } else {
      if (skybox) skybox.visible = false;
      scene.environment = studioEnv;
      scene.backgroundNode = studioBackdrop;
      scene.fog = new THREE.Fog(STUDIO_HORIZON, 7, 26);
      studioFloor.visible = true;
      shadowCatcher.visible = false;
      fill.visible = true;
    }
    if (skybox) skybox.visible = !!def.url;
    scene.environmentIntensity = def.envIntensity;
    renderer.toneMappingExposure = def.exposure;
    sun.intensity = def.sunIntensity;
    sun.shadow.radius = def.shadowRadius;
    sun.position.copy(equirectDirection(def.sun[0], def.sun[1]).multiplyScalar(12));
    sun.target.position.set(0, 0.8, 0);
    refreshShadows();
    state.info.env = id;
    applyMaterial(); // the glass backdrop follows the sky
  }

  // --- sculptures ------------------------------------------------------------------------------------------
  const sculptures: Sculpture[] = [];
  const controls = {
    model: params.get("model") ?? "all",
    material: (params.get("material") as MaterialId | null) ?? "porcelain",
    env: (params.get("env") as EnvId | null) ?? (envAvailable.includes("shore") ? "shore" : "studio"),
    normals: params.get("normals") === "1",
  };
  if (!MATERIALS[controls.material]) controls.material = "porcelain";
  if (!envAvailable.includes(controls.env)) controls.env = "studio";

  function applyMaterial(): void {
    const preset = MATERIALS[controls.material];
    const glass = controls.material === "glass" && !controls.normals;
    for (const s of sculptures) {
      const m = s.material;
      m.color.setHex(preset.color);
      m.roughness = preset.roughness;
      m.metalness = preset.metalness;
      const mode = controls.normals ? "normals" : glass ? "glass" : "lit";
      if (m.userData.mode !== mode) {
        m.userData.mode = mode;
        const nodes = m as unknown as Record<string, VDBTslNode | null>;
        if (mode === "normals") {
          // Unlit world normals (n * 0.5 + 0.5): the trace's own gradient.
          m.lights = false;
          m.toneMapped = false;
          nodes.colorNode = m.normalWorldNode.mul(0.5).add(0.5);
          nodes.backdropNode = null;
        } else {
          m.lights = true;
          m.toneMapped = true;
          nodes.colorNode = null;
          // Glass: the scene behind, bent by the hit's view normal and tinted; three's specular adds the reflections.
          nodes.backdropNode = glass
            ? (viewportSharedTexture as unknown as (uv: VDBTslNode) => VDBTslNode)(screenUV.sub(m.normalViewNode.xy.mul(0.06))).rgb.mul(vec3(0.82, 0.92, 0.95))
            : null;
        }
        m.needsUpdate = true;
      }
    }
    state.info.material = controls.material;
    state.info.normals = controls.normals;
  }

  // Focus: the whole gallery, or one piece.
  const galleryView = { target: new THREE.Vector3(0, 1.15, -0.3), position: new THREE.Vector3(0, 1.95, 6.3) };
  const tween = { from: null as null | { t: THREE.Vector3; p: THREE.Vector3 }, to: galleryView, start: 0, duration: 1100 };
  function focus(id: string, instant = false): void {
    controls.model = id;
    state.info.model = id;
    const s = sculptures.find((x) => x.def.id === id);
    let to = galleryView;
    if (s) {
      const c = s.box.getCenter(new THREE.Vector3());
      const size = s.box.getSize(new THREE.Vector3());
      // Fit the piece (and the plinth cap under it) with room around it, a little from the right and above.
      const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
      const half = Math.max(size.x, size.z) / 2;
      const dist = Math.max((size.y / 2 + 0.14) / tanV, half / (tanV * camera.aspect)) * 1.45 + half;
      c.y -= 0.06 * size.y;
      to = { target: c, position: c.clone().add(new THREE.Vector3(0.22, 0.2, 1).normalize().multiplyScalar(dist)) };
    }
    if (instant) {
      camera.position.copy(to.position);
      orbit.target.copy(to.target);
      camera.lookAt(to.target);
      viewer.motion.rebase();
      tween.from = null;
      return;
    }
    tween.from = { t: orbit.target.clone(), p: camera.position.clone() };
    tween.to = to;
    tween.start = performance.now();
  }
  function tickTween(now: number): void {
    if (!tween.from) return;
    const k = Math.min(1, (now - tween.start) / tween.duration);
    const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
    orbit.target.lerpVectors(tween.from.t, tween.to.target, e);
    camera.position.lerpVectors(tween.from.p, tween.to.position, e);
    camera.lookAt(orbit.target);
    if (k >= 1) {
      tween.from = null;
      viewer.motion.rebase();
    }
  }

  // --- the control strip -------------------------------------------------------------------------------------
  const panel = ui.controls;
  const modelRow = panel.select(
    "model",
    "model",
    [{ value: "all", label: "whole gallery" }, ...present.map(([d]) => ({ value: d.id, label: d.label }))],
    controls.model,
    (v) => focus(v),
  );
  panel.select<MaterialId>(
    "material",
    "material",
    (Object.keys(MATERIALS) as MaterialId[]).map((k) => ({ value: k, label: MATERIALS[k].label })),
    controls.material,
    (v) => {
      controls.material = v;
      applyMaterial();
    },
  );
  panel.select<EnvId>("env", "environment", envAvailable.map((e) => ({ value: e, label: ENVS[e].label })), controls.env, (v) => {
    controls.env = v;
    setEnvironment(v).catch((err: unknown) => ui.fail(errorText(err)));
  });
  panel.checkbox("normals", "show normals", controls.normals, (v) => {
    controls.normals = v;
    applyMaterial();
  });

  await setEnvironment(controls.env);
  focus(present.some(([d]) => d.id === controls.model) ? controls.model : "all", true);
  modelRow.set(controls.model);

  // --- stats ---------------------------------------------------------------------------------------------------
  function statsLine(): string {
    const shown = controls.model === "all" ? sculptures : sculptures.filter((s) => s.def.id === controls.model);
    if (shown.length === 0) return "loading…";
    const sum = (f: (s: Sculpture) => number): number => shown.reduce((a, s) => a + f(s), 0);
    const voxels = sum((s) => s.stats.activeVoxels);
    const one = shown.length === 1 ? shown[0]! : null;
    const coarsest = Math.max(...shown.map((s) => s.stats.bake200Coarser));
    return (
      `${one ? `${one.def.label} · ${one.stats.dims.join("×")}` : `${shown.length} pieces`} · ${(voxels / 1e6).toFixed(2)} M active voxels\n` +
      `${MiB(sum((s) => s.stats.sparseBytes))} sparse · dense f32 ${MiB(sum((s) => s.stats.denseF32Bytes))} · 200³ bake ${coarsest >= 1 ? `${coarsest.toFixed(1)}× coarser` : `${(1 / coarsest).toFixed(1)}× finer (small grid)`}`
    );
  }

  viewer.start((now) => {
    tickTween(now);
    if (shadowFrames > 0) {
      shadowFrames--;
      sun.shadow.needsUpdate = true;
    }
  }, statsLine);

  // --- load the pieces: smallest first, so the gallery fills in -------------------------------------------------
  const order = present.map(([d, url], i) => ({ d, url, i })).sort((a, b) => a.d.mb - b.d.mb);
  for (const { d, url, i } of order) {
    const s = await Sculpture.load(renderer, d, url, plinthBase(i, n), (text, f) => ui.status(text, f));
    sculptures.push(s);
    content.add(s.mesh);
    applyMaterial();
    refreshShadows();
    if (controls.model === d.id) focus(d.id, true);
  }
  ui.status("");
  state.info.loaded = sculptures.map((s) => s.def.id);
  state.info.stats = Object.fromEntries(sculptures.map((s) => [s.def.id, s.stats]));
  state.loading = false;

  // Ready after a few frames with every shader built.
  const from = state.frames;
  const wait = (): void => {
    if (state.frames >= from + 3) state.ready = true;
    else requestAnimationFrame(wait);
  };
  wait();
  (window as unknown as { __DEMO13_API__: unknown }).__DEMO13_API__ = viewer.api;
}

run().catch((err: unknown) => ui.fail(errorText(err)));
