/**
 * Demo 14 · Points — OpenVDB point data grids decoded by our own parser in a worker and drawn as one instanced
 * sprite (`VDBPoints`): the 9.5 M-point waterfall (`waterfall_points.vdb`, 157 MB, Cd / v / id) and the light
 * `sphere_points.vdb`. Colour by `Cd`, by speed `|v|`, by height or one colour; point size; density 10 / 50 / 100 %
 * (a prefix of the worker's subsample order, so a change is only an instance count). Public API only.
 *
 * Query (with the shell's `presentation`, `still`, `panel`, `gallery`): `?model=sphere|waterfall`,
 * `?color=cd|velocity|height|uniform`, `?density=10|50|100`, `?size=<multiplier>`, `?glow=1`, `?view=x,y,z` (camera direction).
 * The waterfall is never fetched until it is picked (or asked for with `?model=waterfall`).
 */
import * as THREE from "three/webgpu";
import * as TSL from "three/tsl";
import { VDBPoints, type VDBTslNode } from "@pmndrs/vdb/three";
import { MiB, createViewer, errorText, mountShowcase, params, probe } from "../showcase-shell.js";
import type { DecodeMessage, DecodedPoints } from "./decode-worker.js";

const ui = mountShowcase({
  number: "14",
  title: "Points",
  blurb:
    "OpenVDB point grids decoded by our own parser, off the main thread: all 9.5 million waterfall particles with their colour and velocity attributes, drawn as one instanced sprite.",
  stateKey: "__DEMO14__",
});
const state = ui.state;

interface ModelDef {
  id: "sphere" | "waterfall";
  label: string;
  url: string;
  mb: number;
  /** Longest side in scene units. */
  size: number;
}
const MODELS: ModelDef[] = [
  { id: "sphere", label: "sphere_points (9.8 K)", url: "/fixtures/vdb-samples/sphere_points.vdb", mb: 0.14, size: 4 },
  { id: "waterfall", label: "waterfall (9.5 M, 157 MB)", url: "/fixtures/vdb-samples/waterfall_points.vdb", mb: 157, size: 10 },
];

type ColorMode = "cd" | "velocity" | "height" | "uniform";
const COLOR_LABEL: Record<ColorMode, string> = { cd: "Cd attribute", velocity: "speed |v|", height: "height", uniform: "one colour" };
type Density = "10" | "50" | "100";

/** Colour ramps in linear RGB (stops evenly spaced). */
const SPEED_RAMP = [
  [0.004, 0.012, 0.06],
  [0.01, 0.12, 0.45],
  [0.05, 0.55, 0.95],
  [0.55, 0.9, 1.0],
  [1.0, 1.0, 1.0],
] as const;
const HEIGHT_RAMP = [
  [0.02, 0.03, 0.12],
  [0.12, 0.08, 0.5],
  [0.75, 0.18, 0.35],
  [1.0, 0.62, 0.18],
  [1.0, 0.95, 0.75],
] as const;
const UNIFORM = [0.75, 0.85, 1.0] as const;

function ramp(stops: readonly (readonly number[])[], t: number, out: Float32Array, o: number): void {
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = stops[i]!;
  const b = stops[i + 1]!;
  out[o] = a[0]! + (b[0]! - a[0]!) * f;
  out[o + 1] = a[1]! + (b[1]! - a[1]!) * f;
  out[o + 2] = a[2]! + (b[2]! - a[2]!) * f;
}

function decode(url: string, progress: (text: string, f: number | null) => void): Promise<DecodedPoints> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./decode-worker.ts", import.meta.url), { type: "module" });
    const t0 = performance.now();
    const secs = (): string => `${((performance.now() - t0) / 1000).toFixed(1)} s`;
    worker.onmessage = (e: MessageEvent<DecodeMessage>) => {
      const m = e.data;
      if (m.type === "progress") {
        if (m.stage === "download") progress(`downloading ${(m.received / 1048576).toFixed(0)}${m.total ? ` / ${(m.total / 1048576).toFixed(0)}` : ""} MB…`, m.total ? m.received / m.total : null);
        else if (m.stage === "parse") progress(`parsing the .vdb (our parser, in a worker) · ${secs()}`, null);
        else if (m.stage === "decode") progress(`decoding ${(m.count ?? 0).toLocaleString()} points + attributes · ${secs()}`, null);
        else progress(`ordering for subsampling · ${secs()}`, null);
      } else if (m.type === "done") {
        worker.terminate();
        resolve(m.result);
      } else {
        worker.terminate();
        reject(new Error(m.message));
      }
    };
    worker.onerror = (e) => {
      worker.terminate();
      reject(new Error(`decode worker: ${e.message}`));
    };
    worker.postMessage({ url });
  });
}

async function run(): Promise<void> {
  const available = (await Promise.all(MODELS.map(async (m) => [m, await probe(m.url)] as const))).filter(([, ok]) => ok).map(([m]) => m);
  if (available.length === 0) throw new Error("No point fixtures found: fixtures/vdb-samples/sphere_points.vdb (from a mjurczyk/openvdb clone) or waterfall_points.vdb (pnpm fixtures:vdb-samples).");
  const missing = MODELS.filter((m) => !available.includes(m));
  if (missing.length) ui.note(`Not shown (fixture missing): ${missing.map((m) => m.url.split("/").pop()).join(", ")}.`);

  const viewer = await createViewer(ui, { fov: 40, near: 0.05, far: 400, exposure: 1, motion: "orbit", speed: 0.06 });
  const { scene, camera, orbit, content } = viewer;
  // Night backdrop: a soft vertical gradient by view elevation (no assets).
  scene.backgroundNode = TSL.mix(TSL.color(0x1a2233), TSL.color(0x05070c), TSL.smoothstep(-0.2, 0.7, (TSL.normalWorld as VDBTslNode).y));
  orbit.minDistance = 1;
  orbit.maxDistance = 60;

  const controls = {
    model: (params.get("model") as ModelDef["id"] | null) ?? "sphere",
    color: (params.get("color") as ColorMode | null) ?? null,
    density: (params.get("density") as Density | null) ?? "100",
    size: Number(params.get("size") ?? 1),
    glow: params.get("glow") === "1",
  };
  if (!available.some((m) => m.id === controls.model)) controls.model = available[0]!.id;
  if (!["10", "50", "100"].includes(controls.density)) controls.density = "100";
  if (!(controls.size > 0)) controls.size = 1;

  interface Loaded {
    def: ModelDef;
    data: DecodedPoints;
    points: VDBPoints;
    colors: Float32Array;
    group: THREE.Group;
  }
  let loaded: Loaded | null = null;
  const cache = new Map<string, DecodedPoints>();

  const hasMode = (data: DecodedPoints, mode: ColorMode): boolean => (mode === "cd" ? !!data.cd : mode === "velocity" ? !!data.speed : true);

  function writeColors(l: Loaded, mode: ColorMode): void {
    const { data, colors } = l;
    const n = data.count;
    if (mode === "cd" && data.cd) colors.set(data.cd);
    else if (mode === "velocity" && data.speed) {
      const [lo, hi] = data.speedRange;
      const k = 1 / (hi - lo);
      for (let i = 0; i < n; i++) ramp(SPEED_RAMP, (data.speed[i]! - lo) * k, colors, i * 3);
    } else if (mode === "height") {
      const y0 = data.bounds.min[1];
      const k = 1 / Math.max(1e-9, data.bounds.max[1] - y0);
      for (let i = 0; i < n; i++) ramp(HEIGHT_RAMP, (data.positions[i * 3 + 1]! - y0) * k, colors, i * 3);
    } else {
      for (let i = 0; i < n; i++) colors.set(UNIFORM, i * 3);
    }
    l.points.colorAttribute!.needsUpdate = true;
    state.info.color = mode;
  }

  /** Instance count and point diameter for the density; size multiplies the default (half the spacing). */
  function applyDensity(l: Loaded): void {
    const { prefix, spacing } = l.data;
    const count = controls.density === "10" ? prefix.p10 : controls.density === "50" ? prefix.p50 : prefix.p100;
    l.points.pointCount = count;
    (l.points as unknown as { count: number }).count = count;
    const fraction = count / Math.max(1, prefix.p100);
    l.points.size = spacing * Math.cbrt(1 / fraction) * 0.5 * controls.size;
    state.info.drawn = count;
    state.info.density = controls.density;
  }

  function applyGlow(l: Loaded): void {
    const m = l.points.material;
    m.blending = controls.glow ? THREE.AdditiveBlending : THREE.NormalBlending;
    m.transparent = controls.glow;
    m.depthWrite = !controls.glow;
    // Additive: dim each sprite (the colour multiplies Cd), denser clouds glow brighter. Not opacity: the disc is alpha-tested.
    const dim = controls.glow ? Math.min(1, 0.35 * Math.sqrt(2e5 / Math.max(1, l.points.pointCount))) : 1;
    m.color.setScalar(dim);
    m.needsUpdate = true;
    state.info.glow = controls.glow;
  }

  async function loadModel(id: ModelDef["id"]): Promise<void> {
    const def = available.find((m) => m.id === id)!;
    state.ready = false;
    state.loading = true;
    controls.model = id;
    state.info.model = id;
    let data = cache.get(id);
    if (!data) {
      data = await decode(def.url, (t, f) => ui.status(`${def.label.split(" ")[0]}: ${t}`, f));
      cache.set(id, data);
    }
    ui.status("uploading to the GPU…", null);
    if (loaded) {
      content.remove(loaded.group);
      loaded.points.dispose();
      loaded = null;
    }
    const colors = new Float32Array(data.count * 3);
    const points = new VDBPoints({ positions: data.positions, colors, count: data.count, spacing: data.spacing });
    // Stand the cloud's box on the floor of the view: longest side `def.size`, centred.
    const min = new THREE.Vector3(...data.bounds.min);
    const max = new THREE.Vector3(...data.bounds.max);
    const size = max.clone().sub(min);
    const s = def.size / Math.max(size.x, size.y, size.z);
    points.scale.setScalar(s);
    points.position.copy(min.clone().add(max).multiplyScalar(-0.5 * s));
    const group = new THREE.Group();
    group.add(points);
    content.add(group);
    loaded = { def, data, points, colors, group };
    const mode = controls.color && hasMode(data, controls.color) ? controls.color : data.cd ? "cd" : "height";
    controls.color = mode;
    colorRow.options(colorOptions(data));
    colorRow.set(mode);
    writeColors(loaded, mode);
    applyDensity(loaded);
    applyGlow(loaded);
    // Frame the bounding sphere: a three-quarter view from slightly above.
    const radius = (size.length() * s) / 2;
    const dist = (radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2))) * (def.id === "sphere" ? 0.82 : 0.6);
    // Aim at the cloud's centroid (the 10% prefix is an even sample), not its box centre: a waterfall's box is mostly air.
    const centroid = new THREE.Vector3();
    for (let i = 0; i < data.prefix.p10; i++) centroid.add(new THREE.Vector3(data.positions[i * 3], data.positions[i * 3 + 1], data.positions[i * 3 + 2]));
    centroid.divideScalar(Math.max(1, data.prefix.p10)).applyMatrix4(points.matrixWorld.compose(points.position, points.quaternion, points.scale));
    centroid.multiplyScalar(0.5); // halfway between the box centre (the origin) and the centroid
    orbit.target.copy(centroid);
    const dir = new THREE.Vector3(...((params.get("view") ?? "-0.6,0.25,1").split(",").map(Number) as [number, number, number]));
    camera.position.copy(dir.normalize().multiplyScalar(dist).add(centroid));
    camera.lookAt(orbit.target);
    viewer.motion.rebase();
    ui.status("");
    state.info.count = data.count;
    state.info.attributes = data.attributes;
    state.loading = false;
    const from = state.frames;
    const wait = (): void => {
      if (state.frames >= from + 3 && !state.loading) state.ready = true;
      else requestAnimationFrame(wait);
    };
    wait();
  }

  // --- the control strip -------------------------------------------------------------------------------------
  const panel = ui.controls;
  const colorOptions = (data: DecodedPoints | null) =>
    (Object.keys(COLOR_LABEL) as ColorMode[]).filter((m) => !data || hasMode(data, m)).map((m) => ({ value: m, label: COLOR_LABEL[m] }));
  panel.select<ModelDef["id"]>("model", "model", available.map((m) => ({ value: m.id, label: m.label })), controls.model, (v) => {
    loadModel(v).catch((err: unknown) => ui.fail(errorText(err)));
  });
  const colorRow = panel.select<ColorMode>("color", "colour by", colorOptions(null), controls.color ?? "cd", (v) => {
    controls.color = v;
    if (loaded) writeColors(loaded, v);
  });
  panel.range("size", "point size", 0.25, 3, 0.05, controls.size, (v) => {
    controls.size = v;
    if (loaded) applyDensity(loaded);
  }, (v) => `${v.toFixed(2)}×`);
  panel.select<Density>("density", "density", [
    { value: "10", label: "10 % (every 10th)" },
    { value: "50", label: "50 % (every 2nd)" },
    { value: "100", label: "100 % (all points)" },
  ], controls.density, (v) => {
    controls.density = v;
    if (loaded) {
      applyDensity(loaded);
      applyGlow(loaded);
    }
  });
  panel.checkbox("glow", "glow (additive)", controls.glow, (v) => {
    controls.glow = v;
    if (loaded) applyGlow(loaded);
  });

  viewer.start(undefined, () => {
    if (!loaded) return "decoding…";
    const d = loaded.data;
    const drawn = loaded.points.pointCount;
    const gpu = d.count * 24;
    return (
      `${d.count.toLocaleString()} points (${d.attributes.join(", ") || "positions only"}) · drawing ${drawn.toLocaleString()}\n` +
      `${MiB(d.fileBytes)} .vdb → parsed ${(d.ms.parse / 1000).toFixed(1)} s, decoded ${(d.ms.decode / 1000).toFixed(1)} s · ${MiB(gpu)} on the GPU`
    );
  });
  (window as unknown as { __DEMO14_API__: unknown }).__DEMO14_API__ = viewer.api;
  await loadModel(controls.model);
}

run().catch((err: unknown) => ui.fail(errorText(err)));
