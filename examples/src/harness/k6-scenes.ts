/**
 * K6.R1: the reference report's own scene set (scripts/k6/reference-report.mjs), shared by the renderer page
 * (`k6-scene.ts`: `./three` `VDBFrameRenderer`, TSL or `backend: 'core'`) and the reference page (`k6-reference.ts`
 * `?case=scene`), so both read the same asset, camera, volume box, medium and lights from one definition.
 *
 * Assets (git-ignored fixtures, local only): his bunny (`fixtures/k3/bunny_cloud.wvdb`, dense), the WDAS cloud quarter
 * (`fixtures/wdas/wdas_cloud_quarter_fp8.nvdb`, sparse) and openvdb.org's smoke (`fixtures/vdb-samples/smoke.vdb`,
 * sparse). Each volume sits at its own world box (unit box → box: `localToWorld`), with no further transform.
 *
 * Lights are three's (colour × intensity). The sun is his (direction (100, 200, 100), white, 1). A point light is
 * given by the irradiance it puts at the volume's centre: 'frame' lighting attenuates with three's
 * `getDistanceAttenuation` (decay 2), so its three intensity is E · d²; 'volume' lighting has no falloff (V84), so its
 * intensity is E. Both modes then light the centre alike, and each is compared with the reference of its own mapping.
 */
import { NvdbFile, buildFromVdb, parseVdb, readWvdb, wvdbDenseVolume, type WvdbDenseVolume } from "@pmndrs/vdb/tools";
import { vdbSparseDensityStats, type VDBReferenceFrameLight } from "../../../packages/vdb/src/tools/internal.js";

export type K6Vec3 = [number, number, number];
export type K6Lighting = "volume" | "frame";

/** A light of a report scene (three units). */
export type K6Light =
  | { kind: "sun"; toLight: K6Vec3; color: K6Vec3; intensity: number }
  | { kind: "ambient"; color: K6Vec3; intensity: number }
  /** K6.Q1: three's HemisphereLight (sky colour above the plane ⟂ up, ground below). */
  | { kind: "hemisphere"; sky: K6Vec3; ground: K6Vec3; intensity: number }
  /** K6.Q1: three's LightProbe (SH9 radiance, three's order, RGB). */
  | { kind: "probe"; sh: number[]; intensity: number }
  /** `offset` from the volume centre in box diagonals; `irradiance` at the centre (see the file comment). */
  | { kind: "point"; offset: K6Vec3; color: K6Vec3; irradiance: number };

export interface K6Scene {
  name: string;
  title: string;
  asset: "bunny" | "wdas" | "smoke";
  /** 'volume' extinction (his `extinction`; normalizeDensity on: ÷ the mean non-zero density). */
  extinction: number;
  /** Linear base colour (the scattering albedo). */
  albedo: K6Vec3;
  /** Dual-lobe phase (g, back g, back weight), the volume's and the renderer's 'frame' phase. */
  phase: K6Vec3;
  lights: K6Light[];
  /** Camera direction from the box centre (normalised here) and distance in box diagonals; vertical fov 60°. */
  camera: { dir: K6Vec3; dist: number };
}

const SUN: K6Light = { kind: "sun", toLight: [100, 200, 100], color: [1, 1, 1], intensity: 1 };
const AMBIENT: K6Light = { kind: "ambient", color: [1, 1, 1], intensity: 0.4 };
// K6.Q1: a blue sky over a brown ground; a probe brighter and bluer above, warm toward +x (L1 + a little L2)
const HEMI: K6Light = { kind: "hemisphere", sky: [0.35, 0.55, 1], ground: [0.45, 0.3, 0.15], intensity: 0.5 };
const PROBE: K6Light = {
  kind: "probe",
  intensity: 1,
  // three's SphericalHarmonics3 coefficients (radiance): c0 grey-blue, c1 (y) blue up, c3 (x) warm, c6 (3z² − 1) small
  sh: [0.6, 0.65, 0.8, 0.15, 0.2, 0.35, 0, 0, 0, 0.2, 0.1, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.05, 0.05, 0, 0, 0, 0, 0, 0],
};
const PHASE: K6Vec3 = [0.8, -0.3, 0.2];
const VIEW = { dir: [0.9, 0.35, 1.1] as K6Vec3, dist: 0.8 };

const scene = (name: string, title: string, s: Omit<K6Scene, "name" | "title" | "phase" | "camera"> & Partial<Pick<K6Scene, "phase" | "camera">>): K6Scene => ({ name, title, phase: PHASE, camera: VIEW, ...s });

/** The scene set: each with his sun, and with / without ambient light; the WDAS cloud, the smoke, an absorbing and a thin bunny, a point light. */
export const K6_SCENES: Record<string, K6Scene> = Object.fromEntries(
  [
    scene("wdas", "WDAS cloud (quarter, Fp8, sparse), sun", { asset: "wdas", extinction: 40, albedo: [1, 1, 1], lights: [SUN] }),
    scene("wdas-ambient", "WDAS cloud, sun + ambient 0.4", { asset: "wdas", extinction: 40, albedo: [1, 1, 1], lights: [SUN, AMBIENT] }),
    scene("smoke", "openvdb.org smoke (sparse), sun", { asset: "smoke", extinction: 40, albedo: [1, 1, 1], lights: [SUN] }),
    scene("smoke-ambient", "openvdb.org smoke, sun + ambient 0.4", { asset: "smoke", extinction: 40, albedo: [1, 1, 1], lights: [SUN, AMBIENT] }),
    scene("bunny-absorbing", "his bunny, absorbing (albedo 0.5, extinction 100), sun", { asset: "bunny", extinction: 100, albedo: [0.5, 0.5, 0.5], lights: [SUN] }),
    scene("bunny-absorbing-ambient", "absorbing bunny, sun + ambient 0.4", { asset: "bunny", extinction: 100, albedo: [0.5, 0.5, 0.5], lights: [SUN, AMBIENT] }),
    scene("bunny-thin", "his bunny, thin (extinction 5, albedo 1), sun", { asset: "bunny", extinction: 5, albedo: [1, 1, 1], lights: [SUN] }),
    scene("bunny-thin-ambient", "thin bunny, sun + ambient 0.4", { asset: "bunny", extinction: 5, albedo: [1, 1, 1], lights: [SUN, AMBIENT] }),
    scene("smoke-hemi", "openvdb.org smoke, sun + hemisphere light (K6.Q1)", { asset: "smoke", extinction: 40, albedo: [1, 1, 1], lights: [SUN, HEMI] }),
    scene("wdas-probe", "WDAS cloud, sun + light probe (K6.Q1)", { asset: "wdas", extinction: 40, albedo: [1, 1, 1], lights: [SUN, PROBE] }),
    scene("bunny-absorbing-hemi", "absorbing bunny, sun + hemisphere light (K6.Q1)", { asset: "bunny", extinction: 100, albedo: [0.5, 0.5, 0.5], lights: [SUN, HEMI] }),
    scene("bunny-point", "his bunny (extinction 20), a point light above-left (E = 1 at the centre in both mappings)", {
      asset: "bunny",
      extinction: 20,
      albedo: [1, 1, 1],
      lights: [{ kind: "point", offset: [-0.45, 0.5, 0.25], color: [1, 0.95, 0.9], irradiance: 1 }],
    }),
  ].map((s) => [s.name, s]),
);

/** A loaded asset: the dense volume (bunny) or the sparse image (WDAS, smoke), its world box and density stats. */
export interface K6Asset {
  dense: WvdbDenseVolume | null;
  image: Uint32Array | null;
  bbox: [K6Vec3, K6Vec3];
  /** normalizeDensity's mean and the max (dense: the texels; sparse: `vdbSparseDensityStats`, normalize 'none'). */
  stats: { meanNonZero: number; max: number; activeVoxels: number };
}

async function fetchBytes(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status} (fixtures are git-ignored: copy them from the main checkout)`);
  return r.arrayBuffer();
}

/** The world box of a sparse image (index box cell corners through indexToWorld; the sparse sources' default `bbox`). */
function imageBox(indexToWorld: (p: K6Vec3) => K6Vec3, min: readonly number[], max: readonly number[]): [K6Vec3, K6Vec3] {
  const lo: K6Vec3 = [Infinity, Infinity, Infinity];
  const hi: K6Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let c = 0; c < 8; c++) {
    const p = indexToWorld([c & 1 ? max[0]! + 0.5 : min[0]! - 0.5, c & 2 ? max[1]! + 0.5 : min[1]! - 0.5, c & 4 ? max[2]! + 0.5 : min[2]! - 0.5]);
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k]!, p[k]!);
      hi[k] = Math.max(hi[k]!, p[k]!);
    }
  }
  return [lo, hi];
}

/** Load a scene's asset (the same bytes for the renderer and the reference). */
export async function k6LoadAsset(asset: K6Scene["asset"]): Promise<K6Asset> {
  if (asset === "bunny") {
    const dense = wvdbDenseVolume(readWvdb(new Uint8Array(await fetchBytes("/fixtures/k3/bunny_cloud.wvdb"))), { mipmaps: true });
    const [lo, hi] = dense.bbox ?? [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]];
    return { dense, image: null, bbox: [[...lo] as K6Vec3, [...hi] as K6Vec3], stats: { meanNonZero: dense.meanDensity, max: dense.maxDensity, activeVoxels: dense.activeTexels } };
  }
  let image: Uint32Array;
  if (asset === "wdas") image = NvdbFile.fromArrayBuffer(await fetchBytes("/fixtures/wdas/wdas_cloud_quarter_fp8.nvdb")).gridImage(0);
  else {
    const file = parseVdb(await fetchBytes("/fixtures/vdb-samples/smoke.vdb"));
    image = buildFromVdb(file.grids.find((g) => g.valueType === "float")!);
  }
  return { dense: null, image, bbox: [[0, 0, 0], [0, 0, 0]], stats: vdbSparseDensityStats(image, false) };
}

/** Fill a sparse asset's box from the pool's grid info (indexBBox, indexToWorld: row-major 3x4 or 4x4). */
export function k6SparseBox(a: K6Asset, info: { indexBBox: { min: readonly number[]; max: readonly number[] }; indexToWorld: ArrayLike<number> }): void {
  const m = info.indexToWorld;
  const at = (p: K6Vec3): K6Vec3 => [0, 1, 2].map((r) => m[r * 4]! * p[0] + m[r * 4 + 1]! * p[1] + m[r * 4 + 2]! * p[2] + m[r * 4 + 3]!) as K6Vec3;
  a.bbox = imageBox(at, info.indexBBox.min, info.indexBBox.max);
}

const norm = (v: readonly number[]): K6Vec3 => {
  const l = Math.hypot(v[0]!, v[1]!, v[2]!);
  return [v[0]! / l, v[1]! / l, v[2]! / l];
};
const cross = (a: readonly number[], b: readonly number[]): K6Vec3 => [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!];

/** The frame of a scene at an aspect: camera (column-major camera → world, WebGL projection), the box and the lights. */
export interface K6Frame {
  cameraToWorld: number[];
  projection: number[];
  near: number;
  far: number;
  fov: number;
  eye: K6Vec3;
  center: K6Vec3;
  size: K6Vec3;
  /** Unit box → world (column-major). */
  localToWorld: number[];
  /** The lights in renderer units for the lighting mode (three colour × intensity, positions in the world). */
  lights: VDBReferenceFrameLight[];
  /** The same lights as three objects take them (position, colour, intensity, decay). */
  three: { kind: "sun" | "ambient" | "point" | "hemisphere" | "probe"; position: K6Vec3; color: K6Vec3; intensity: number; decay: number; ground?: K6Vec3; sh?: number[] }[];
}

export function k6SceneFrame(s: K6Scene, a: K6Asset, aspect: number, lighting: K6Lighting): K6Frame {
  const [lo, hi] = a.bbox;
  const center: K6Vec3 = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  const size: K6Vec3 = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  const diag = Math.hypot(...size);
  const d = norm(s.camera.dir);
  const eye: K6Vec3 = [center[0] + d[0] * diag * s.camera.dist, center[1] + d[1] * diag * s.camera.dist, center[2] + d[2] * diag * s.camera.dist];
  const z = norm([eye[0] - center[0], eye[1] - center[1], eye[2] - center[2]]);
  const x = norm(cross([0, 1, 0], z));
  const y = cross(z, x);
  const near = diag * 0.01;
  const far = diag * 40;
  const fov = 60;
  const f = 1 / Math.tan((fov * Math.PI) / 360);
  const projection = [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, (2 * far * near) / (near - far), 0];
  const lights: VDBReferenceFrameLight[] = [];
  const three: K6Frame["three"] = [];
  for (const l of s.lights) {
    const c = ("color" in l ? l.color : [1, 1, 1]).map((v) => v * ("intensity" in l ? l.intensity : 1)) as K6Vec3;
    if (l.kind === "sun") {
      const p = norm(l.toLight);
      lights.push({ kind: "directional", direction: p, color: c });
      three.push({ kind: "sun", position: [center[0] + p[0] * diag, center[1] + p[1] * diag, center[2] + p[2] * diag], color: l.color, intensity: l.intensity, decay: 0 });
    } else if (l.kind === "ambient") {
      lights.push({ kind: "ambient", color: c });
      three.push({ kind: "ambient", position: center, color: l.color, intensity: l.intensity, decay: 0 });
    } else if (l.kind === "hemisphere") {
      // three's HemisphereLight: up = its position (default +y), colours × intensity
      lights.push({ kind: "hemisphere", sky: l.sky.map((v) => v * l.intensity) as K6Vec3, ground: l.ground.map((v) => v * l.intensity) as K6Vec3, up: [0, 1, 0] });
      three.push({ kind: "hemisphere", position: [0, 1, 0], color: l.sky, ground: l.ground, intensity: l.intensity, decay: 0 });
    } else if (l.kind === "probe") {
      lights.push({ kind: "probe", sh: l.sh.map((v) => v * l.intensity) });
      three.push({ kind: "probe", position: center, color: [1, 1, 1], intensity: l.intensity, decay: 0, sh: l.sh });
    } else {
      const position: K6Vec3 = [center[0] + l.offset[0] * diag, center[1] + l.offset[1] * diag, center[2] + l.offset[2] * diag];
      const dc = Math.hypot(l.offset[0] * diag, l.offset[1] * diag, l.offset[2] * diag);
      const intensity = lighting === "frame" ? l.irradiance * dc * dc : l.irradiance;
      lights.push({ kind: "point", position, color: l.color.map((v) => v * intensity) as K6Vec3, distance: 0, decay: 2 });
      three.push({ kind: "point", position, color: l.color, intensity, decay: 2 });
    }
  }
  return {
    cameraToWorld: [...x, 0, ...y, 0, ...z, 0, ...eye, 1],
    projection,
    near,
    far,
    fov,
    eye,
    center,
    size,
    localToWorld: [size[0], 0, 0, 0, 0, size[1], 0, 0, 0, 0, size[2], 0, ...center, 1],
    lights,
    three,
  };
}

/** True when a scene's lights differ between the two lighting mappings (point or spot lights). */
export const k6SceneHasFalloff = (s: K6Scene): boolean => s.lights.some((l) => l.kind === "point");
