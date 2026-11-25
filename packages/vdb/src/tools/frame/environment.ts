// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/frameEnvironment.js (all, 331 lines).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: TypeScript; `isFrameDiffusion` inlined (tools does not import the three adapter);
// the readback promise may also resolve to a Uint16Array of half floats (decoded here). Same
// directions, cells, colours and SH quadrature. Divergences: DIVERGES.md (K3.3 rows).

/**
 * Multi-directional environment light (Disney "Deep Scattering", §4.4 and §7), engine-free (K3.3):
 * the scene environment is read once into a small lat-long map (64 × 32, PMREM roughness 0.4); the
 * CPU turns it into ≤ 12 directional lights. `'directional'`: the brightest texel plus fixed sky
 * directions (zenith, nadir, a horizon ring; opposite pairs with diffusion), each texel to its
 * nearest seed, a light at the energy centroid with energy / 4. `'spherical-harmonics'`: SH L1/L2
 * projection evaluated at quadrature directions. The three adapter draws the map and passes the
 * readback through `extract`.
 */

export const frameMaxEnvironmentLights = 12;
export const frameMaxEnvironmentSamples = 8;
export const frameEnvironmentMapWidth = 64;
export const frameEnvironmentMapHeight = 32;
/** PMREM blur of the map: a small sun covers more than one texel and is not lost between texels. */
export const frameEnvironmentMapRoughness = 0.4;

export const frameEnvironmentModes = {
  outward: "outward",
  directional: "directional",
  sphericalHarmonics: "spherical-harmonics",
} as const;

export type VDBFrameEnvironmentMode = (typeof frameEnvironmentModes)[keyof typeof frameEnvironmentModes];

/** The renderer settings the environment lights read. */
export interface VDBFrameEnvironmentSettings {
  environmentLighting: string;
  environmentSamples: number;
  environmentShOrder: number;
  multipleScattering: boolean | string;
}

/** The scene fields the environment lights read (three's `Scene` fits). */
export interface VDBFrameEnvironmentScene {
  environment: { uuid: string; version: number } | null;
  environmentRotation: { x: number; y: number; z: number; order: string };
  environmentIntensity: number;
}

// NOTE Retry an empty map (environment or PMREM not ready) after this number of updates
const retryInterval = 30;

type V3 = [number, number, number];

const normalize = (x: number, y: number, z: number): V3 => {
  const length = Math.hypot(x, y, z) || 1.0;
  return [x / length, y / length, z / length];
};

/** His `isFrameDiffusion` (three `frame/phase.ts`): true, 'two-sided' and 'neural'. */
const isDiffusion = (mode: boolean | string) => !!mode && mode !== "octaves";

/** The same direction as the map shaders. u and v are 0 to 1; v = 0 is +Y. */
export const vdbFrameEnvironmentMapDirection = (u: number, v: number): V3 => {
  const phi = u * 2.0 * Math.PI;
  const theta = v * Math.PI;
  return [Math.sin(theta) * Math.cos(phi), Math.cos(theta), Math.sin(theta) * Math.sin(phi)];
};

// NOTE Real SH basis, bands 0 to 2. Same constants as three.js SphericalHarmonics3.
const getShBasis = ([x, y, z]: V3, count: number) =>
  [0.282095, 0.488603 * y, 0.488603 * z, 0.488603 * x, 1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3.0 * z * z - 1.0), 1.092548 * x * z, 0.546274 * (x * x - y * y)].slice(0, count);

// NOTE Quadrature directions. Octahedron (spherical 3-design) for L1, icosahedron (5-design) for L2.
const goldenRatio = (1.0 + Math.sqrt(5.0)) / 2.0;
const octahedron: V3[] = [
  [1, 0, 0],
  [-1, 0, 0],
  [0, 1, 0],
  [0, -1, 0],
  [0, 0, 1],
  [0, 0, -1],
];
const icosahedron: V3[] = (
  [
    [0, 1, goldenRatio],
    [0, -1, goldenRatio],
    [0, 1, -goldenRatio],
    [0, -1, -goldenRatio],
    [1, goldenRatio, 0],
    [-1, goldenRatio, 0],
    [1, -goldenRatio, 0],
    [-1, -goldenRatio, 0],
    [goldenRatio, 0, 1],
    [-goldenRatio, 0, 1],
    [goldenRatio, 0, -1],
    [-goldenRatio, 0, -1],
  ] as V3[]
).map(([x, y, z]) => normalize(x, y, z));

const luminance = (r: number, g: number, b: number) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

export const vdbFrameEnvironmentMode = (settings: Pick<VDBFrameEnvironmentSettings, "environmentLighting">): VDBFrameEnvironmentMode => {
  const mode = settings.environmentLighting;
  return mode === frameEnvironmentModes.directional || mode === frameEnvironmentModes.sphericalHarmonics ? mode : frameEnvironmentModes.outward;
};

interface Texel {
  radiance: V3;
  luminance: number;
  direction: V3;
  solidAngle: number;
}

/** One environment light: world direction to the light and its isotropic irradiance colour (energy / 4). */
export interface VDBFrameEnvironmentLight {
  direction: V3;
  color: V3;
  power?: number;
}

/** Output of {@link VDBFrameEnvironmentLights.update}. Colours include `scene.environmentIntensity`. */
export interface VDBFrameEnvironmentResult {
  mode: VDBFrameEnvironmentMode;
  count: number;
  directions: V3[];
  colors: V3[];
}

const halfToFloat = (h: number): number => {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
};

/**
 * His `FrameEnvironmentLights`: holds the lat-long map of the scene environment and the lights made
 * from it. `extract(renderer, scene)` is the engine part: it draws the map (RGB radiance, A the
 * texel's v, so the readback row order does not matter) and resolves RGBA floats.
 */
export class VDBFrameEnvironmentLights<R = unknown, S extends VDBFrameEnvironmentScene = VDBFrameEnvironmentScene> {
  key: string | null = null;
  pending = false;
  retryAt = 0;
  updates = 0;
  texels: Texel[] | null = null;
  texelsVersion = 0;
  warned = false;
  lightsKey: string | null = null;
  lights: VDBFrameEnvironmentLight[] = [];

  readonly result: VDBFrameEnvironmentResult = {
    mode: frameEnvironmentModes.outward,
    count: 0,
    directions: Array.from({ length: frameMaxEnvironmentLights }, () => [0.0, 1.0, 0.0] as V3),
    colors: Array.from({ length: frameMaxEnvironmentLights }, () => [0.0, 0.0, 0.0] as V3),
  };

  constructor(readonly extract: (renderer: R, scene: S) => Promise<ArrayLike<number>>) {}

  getKey(scene: S): string {
    const { environment, environmentRotation: rotation } = scene;
    return `${environment!.uuid}:${environment!.version}:${rotation.x}:${rotation.y}:${rotation.z}:${rotation.order}`;
  }

  /** Null when the 'outward' path must be used: another mode, no environment, or no map yet. */
  update(renderer: R, scene: S, settings: VDBFrameEnvironmentSettings): VDBFrameEnvironmentResult | null {
    const mode = vdbFrameEnvironmentMode(settings);
    this.updates++;
    if (mode === frameEnvironmentModes.outward || !scene.environment) return null;

    const key = this.getKey(scene);
    if (key !== this.key && !this.pending && this.updates >= this.retryAt) this.requestMap(renderer, scene, key);
    if (!this.texels) return null;

    // NOTE Diffusion multiple scattering needs the directions in opposite pairs (getDirectionalLights)
    const pairs = isDiffusion(settings.multipleScattering);
    const lightsKey = `${mode}:${settings.environmentSamples}:${settings.environmentShOrder}:${this.texelsVersion}:${pairs}`;
    if (lightsKey !== this.lightsKey) {
      this.lights = mode === frameEnvironmentModes.directional ? this.getDirectionalLights(settings.environmentSamples, pairs) : this.getShLights(settings.environmentShOrder);
      this.lightsKey = lightsKey;
    }

    const { result } = this;
    const intensity = scene.environmentIntensity;
    result.mode = mode;
    result.count = this.lights.length;
    this.lights.forEach(({ direction, color }, index) => {
      result.directions[index] = direction;
      result.colors[index] = color.map((value) => value * intensity) as V3;
    });
    return result;
  }

  /** Resolves when a pending map readback has settled (tests and harnesses; not his). */
  settled: Promise<void> = Promise.resolve();

  requestMap(renderer: R, scene: S, key: string): void {
    this.pending = true;
    this.settled = this.extract(renderer, scene)
      .then((pixels) => {
        this.pending = false;
        if (this.setTexels(pixels)) this.key = key;
        else this.retryAt = this.updates + retryInterval;
      })
      .catch((error: unknown) => {
        this.pending = false;
        this.retryAt = this.updates + retryInterval;
        if (!this.warned) {
          console.warn("VDBFrameRenderer: could not read the environment map.", error);
          this.warned = true;
        }
      });
  }

  /** Radiance, direction and solid angle of each texel. Returns false for an empty map. Half floats (Uint16Array) are decoded. */
  setTexels(pixels: ArrayLike<number>): boolean {
    const width = frameEnvironmentMapWidth;
    const height = frameEnvironmentMapHeight;
    const half = pixels instanceof Uint16Array;
    const at = (i: number) => (half ? halfToFloat(pixels[i]!) : pixels[i]!);
    const texels: Texel[] = [];
    let energy = 0.0;

    for (let i = 0; i < width * height; i++) {
      const r = Math.max(0.0, at(i * 4));
      const g = Math.max(0.0, at(i * 4 + 1));
      const b = Math.max(0.0, at(i * 4 + 2));
      const v = at(i * 4 + 3);
      const u = ((i % width) + 0.5) / width;
      const theta = v * Math.PI;
      const solidAngle = ((2.0 * Math.PI) / width) * (Math.PI / height) * Math.sin(theta);
      if (!Number.isFinite(r + g + b + v)) return false;
      energy += luminance(r, g, b) * solidAngle;
      texels.push({ radiance: [r, g, b], luminance: luminance(r, g, b), direction: vdbFrameEnvironmentMapDirection(u, v), solidAngle });
    }
    if (energy <= 0.0) return false;
    this.texels = texels;
    this.texelsVersion++;
    return true;
  }

  /**
   * Dominant direction (brightest texel) and fixed sky directions: zenith, nadir, then a horizon ring.
   * Each texel goes to the nearest direction; a light is at the energy centroid of its texels with
   * their energy; colour is energy / 4 (a uniform environment then gives the 'outward' light).
   * `pairs`: seeds in opposite pairs (diffusion weights of a pair sum to 2). `tie` (K6.2b, the core): near-ties go to
   * the first candidate.
   */
  getDirectionalLights(samples: number, pairs = false, tie = 0): VDBFrameEnvironmentLight[] {
    const texels = this.texels!;
    const count = Math.min(frameMaxEnvironmentSamples, Math.max(1, Math.round(samples) || 1));
    // K6.2b `tie` (the core twin): the first texel within `tie` (relative) of the maximum, and a seed must be nearer
    // by more than `tie`; 0 is his (the first exact maximum, strictly nearer)
    const brightest = Math.max(...texels.map((texel) => texel.luminance));
    const peak = tie > 0 ? texels.find((texel) => texel.luminance >= brightest * (1 - tie))! : texels.reduce((best, texel) => (texel.luminance > best.luminance ? texel : best));
    const seeds: V3[] = [peak.direction];
    const ringCount = pairs ? Math.max(0, count - 4) - (Math.max(0, count - 4) % 2) : count - 3;

    if (pairs) {
      if (count >= 2) seeds.push(peak.direction.map((value) => -value) as V3);
      if (count >= 4) seeds.push([0.0, 1.0, 0.0], [0.0, -1.0, 0.0]);
    } else {
      if (count >= 2) seeds.push([0.0, 1.0, 0.0]);
      if (count >= 3) seeds.push([0.0, -1.0, 0.0]);
    }

    // NOTE The horizon ring starts opposite the dominant direction
    const peakAzimuth = Math.atan2(peak.direction[2], peak.direction[0]);
    for (let i = 0; i < ringCount; i++) {
      const azimuth = peakAzimuth + Math.PI + (2.0 * Math.PI * i) / ringCount;
      seeds.push([Math.cos(azimuth), 0.0, Math.sin(azimuth)]);
    }

    const cells = seeds.map((seed) => ({ seed, energy: [0.0, 0.0, 0.0] as V3, centroid: [0.0, 0.0, 0.0] as V3 }));
    for (const { radiance, luminance: texelLuminance, direction, solidAngle } of texels) {
      let nearest = cells[0]!;
      let nearestDot = -Infinity;
      for (const cell of cells) {
        const dot = cell.seed[0] * direction[0] + cell.seed[1] * direction[1] + cell.seed[2] * direction[2];
        if (dot > nearestDot + tie) {
          nearest = cell;
          nearestDot = dot;
        }
      }
      for (let c = 0; c < 3; c++) {
        nearest.energy[c] = nearest.energy[c]! + radiance[c]! * solidAngle;
        nearest.centroid[c] = nearest.centroid[c]! + direction[c]! * texelLuminance * solidAngle;
      }
    }

    return cells
      .filter(({ energy }) => luminance(...energy) > 0.0)
      .map(({ seed, energy, centroid }) => ({
        direction: Math.hypot(...centroid) > 1e-6 ? normalize(...centroid) : seed,
        color: energy.map((value) => value / 4.0) as V3,
        power: luminance(...energy),
      }))
      .sort((a, b) => b.power - a.power);
  }

  /**
   * SH projection of the environment (L1: 4, L2: 9 coefficients) evaluated at quadrature directions:
   * direction m gets the colour (4π / M) L_sh(w_m) / 4. The shaders clamp the sum to ≥ 0.
   */
  getShLights(order: number): VDBFrameEnvironmentLight[] {
    const texels = this.texels!;
    const coefficientCount = Number(order) === 1 ? 4 : 9;
    const directions = coefficientCount === 4 ? octahedron : icosahedron;
    const coefficients = Array.from({ length: coefficientCount }, () => [0.0, 0.0, 0.0] as V3);
    for (const { radiance, direction, solidAngle } of texels) {
      getShBasis(direction, coefficientCount).forEach((basis, index) => {
        for (let c = 0; c < 3; c++) coefficients[index]![c] = coefficients[index]![c]! + radiance[c]! * basis * solidAngle;
      });
    }
    const weight = (4.0 * Math.PI) / directions.length / 4.0;
    return directions.map((direction) => {
      const color: V3 = [0.0, 0.0, 0.0];
      getShBasis(direction, coefficientCount).forEach((basis, index) => {
        for (let c = 0; c < 3; c++) color[c] = color[c]! + coefficients[index]![c]! * basis * weight;
      });
      return { direction, color };
    });
  }
}
