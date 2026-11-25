/**
 * Plain-data types of the core frame renderer (K6.1, design §2.1–§2.5). Matrices are column-major
 * `ArrayLike<number>` (three's `Matrix4.elements` order); colours are linear RGB. K6.1 implements
 * 'volume' lighting on dense sources (K6.2: 'frame' lighting, K6.2b: the environment, K6.5: volume shadow maps and
 * emission, K6.5b: host occluder maps).
 */

import type { VDBVolumeSlot } from "../source/types.js";
import type { VDBFramePhaseName } from "./settings.js";

export type { VDBVolumeSlot } from "../source/types.js";

/** A 3-vector. */
export type VDBFrameVec3 = readonly [number, number, number];

/** Column-major 4x4. */
export type VDBFrameMat4 = ArrayLike<number>;

/** His `frameLights` bitmask of a volume's `lights` (the same bits as `./three` `vdbFrameLights`). */
export const VDB_FRAME_LIGHT_BITS = {
  ambient: 0b100000,
  point: 0b010000,
  directional: 0b001000,
  spot: 0b000100,
  hemisphere: 0b000010,
  environment: 0b000001,
} as const;

/** Every light type (his FrameVolume default). */
export const VDB_FRAME_ALL_LIGHTS = 0b111111;

/** The camera of a frame. */
export interface VDBFrameCamera {
  /** World → view (three's `camera.matrixWorldInverse`). */
  readonly view: VDBFrameMat4;
  /** View → clip with NDC z in [0, 1] (WebGPU; three's `camera.projectionMatrix` on WebGPURenderer). */
  readonly projection: VDBFrameMat4;
  readonly near: number;
  readonly far: number;
  readonly orthographic?: boolean;
  /** The projection maps near to depth 1 and far to 0. */
  readonly reversedDepth?: boolean;
  /** Identity of the camera: the temporal history is dropped when it changes. */
  readonly id?: number | string;
}

/**
 * A volume shadow map request on a directional, spot or point light (K6.5, design §2.6): the core draws a Beer map of
 * every shadow caster for it (`settings.castShadows`), and returns it in `outputs.volumeShadows` with the view it was
 * drawn with. `size`: texels per side (`settings.castShadowCoarse`: 64). `view`:
 * - spot: the camera the host's own depth map uses (`viewProjection` column-major, WebGPU depth), so both maps share
 *   one texel grid; default a perspective camera over the cone (near 0.5, far = distance or 500).
 * - point: `near` / `far` of the cube (far divides the stored depths; default distance or 500).
 * - directional: `'fit'` (default; his cascades of the view around the casters, `castShadowCascades`,
 *   `castShadowDistance`) or explicit orthographic cascades with their view-depth `breaks`.
 */
export interface VDBVolumeShadowRequest {
  size: number;
  view?:
    | { viewProjection?: ArrayLike<number>; near: number; far: number; orthographic?: boolean }
    | { cascades: { viewProjection: ArrayLike<number>; near: number; far: number }[]; breaks: number[] }
    | "fit";
}

/** The state a volume shadow map was drawn with (what receivers project with; his 62a6e3e by construction). */
export interface VDBVolumeShadowDrawn {
  /** Per layer (cascades; spot: one), column-major, WebGPU depth. Point maps: empty. */
  readonly viewProjection: readonly Float32Array[];
  /** Ray origin per layer (cascades) or the light's position / camera position. */
  readonly origin: readonly VDBFrameVec3[];
  /** Far plane per layer: the map stores depths ÷ far. */
  readonly far: readonly number[];
  readonly forward: VDBFrameVec3;
  readonly perspective: boolean;
  /** Directional: the cascade far distances (view depth) and count the layers were drawn for. */
  readonly breaks: readonly number[];
  readonly cascadeCount: number;
}

/**
 * One light's volume shadow map (`outputs.volumeShadows`, by light id). Bind `texture` (a `'2d-array'` view; point
 * maps a `'cube'` view of its 6 layers), a linear clamp sampler and `view` (`var<uniform> VdbVolumeShadowView`), and
 * call `vdbVolumeShadow` / `vdbVolumeShadowCube` (`vdbVolumeShadowWgsl()`).
 */
export interface VDBVolumeShadowMap {
  readonly light: number | string;
  readonly kind: "directional" | "spot" | "point";
  /** rgba16float 2d with `layers` layers: (τ·first, τ·last, 0, τ), depths ÷ far. */
  readonly texture: GPUTexture;
  readonly layers: number;
  readonly size: number;
  /** The drawn view as a uniform buffer (`VdbVolumeShadowView`, rewritten each update). */
  readonly view: GPUBuffer;
  readonly drawn: VDBVolumeShadowDrawn;
  /** False when the light is not in this frame or no caster exists: receivers read 1. */
  readonly enabled: boolean;
  /** +1 per drawn piece; a new `texture` bumps `outputs.version`. */
  readonly revision: number;
  /** The update count of the last draw of each piece (the map, or each cascade), for budget checks. */
  readonly lastDraw: readonly number[];
}

/**
 * Emission colour of a volume (K6.5, design §3.8): `'white'` (default); his `'planck'` / `'soot'` spectra (the
 * blackbody table of the analytic CMF fit, D33, at the soot exponent); a 256-texel LUT (`lut`: 256 × rgb over log
 * temperature 500–40,000 K, linear between texels); or a user WGSL hook defining
 * `fn vdbUserEmissionColor(t: f32, temperature: f32) -> vec3f` (t = the normalized temperature, `temperature` in
 * kelvin; every top-level name starts with `vdbUser`), validated once (`VDB_BAD_WGSL`).
 */
export type VDBEmissionColor = "white" | { readonly spectrum: "planck" | "soot"; readonly exponent?: number } | { readonly lut: ArrayLike<number> } | { readonly wgsl: string };

/** The emission light's probe settings (his `FrameEmissionLight` props; `VDBFrameInput.emissionLight`). */
export interface VDBFrameEmissionLightSettings {
  /** Probes per axis of each cascade, 2–32 (default 16). */
  readonly resolution?: number;
  /** The last cascade's size as a multiple of the box of all emitters (default 3). */
  readonly range?: number;
  /** Probe cascades 1–4 (default 1). */
  readonly cascades?: number;
  /** The finer cascades cover this distance from the camera (default 512). */
  readonly distance?: number;
  /** Steps of each ray in a volume (default 24). */
  readonly steps?: number;
}

/** The emission light's outputs (`outputs.emissionProbes`). */
export interface VDBFrameEmissionProbes {
  /** rgba16float 3D stack: per cascade c and channel k (red, green, blue) the L1 probes at `layout.origins[3c + k]`. */
  readonly texture: GPUTexture;
  /** `VdbEmissionGrid` uniform (the receivers' `grid`). */
  readonly grid: GPUBuffer;
  /** The stack's sub-volume layout (side = the probe resolution). */
  readonly layout: { readonly size: readonly [number, number, number]; readonly origins: readonly (readonly [number, number, number])[] };
  /** The probe cascades, their centre and the emitters (the CPU copy of `grid`). */
  readonly state: {
    readonly count: number;
    readonly resolution: number;
    readonly min: readonly VDBFrameVec3[];
    readonly size: readonly VDBFrameVec3[];
    readonly center: VDBFrameVec3;
    readonly emitters: number;
  };
  /** +1 when the probes were drawn again. */
  readonly revision: number;
  /** The last update: direction passes run and whether the probes were drawn. */
  readonly lastUpdate: { readonly directions: number; readonly probes: boolean };
}

/**
 * A scene light. `direction` is the way the light travels (from the light toward its target); `up` of a
 * hemisphere light points at its sky colour; `color` is colour × intensity.
 */
export type VDBFrameLight = { readonly id: number | string; readonly mask?: number } & (
  | { readonly kind: "ambient"; readonly color: VDBFrameVec3 }
  | { readonly kind: "hemisphere"; readonly sky: VDBFrameVec3; readonly ground: VDBFrameVec3; readonly up: VDBFrameVec3 }
  | { readonly kind: "probe"; readonly sh: ArrayLike<number> }
  | { readonly kind: "directional"; readonly direction: VDBFrameVec3; readonly color: VDBFrameVec3; readonly shadow?: VDBVolumeShadowRequest }
  | {
      readonly kind: "point";
      readonly position: VDBFrameVec3;
      readonly color: VDBFrameVec3;
      readonly distance: number;
      readonly decay: number;
      readonly shadow?: VDBVolumeShadowRequest;
    }
  | {
      readonly kind: "spot";
      readonly position: VDBFrameVec3;
      readonly direction: VDBFrameVec3;
      readonly color: VDBFrameVec3;
      readonly distance: number;
      readonly decay: number;
      /** cos of the cone angle. */
      readonly cosAngle: number;
      /** cos of angle × (1 − penumbra): full intensity inside it. */
      readonly cosPenumbra: number;
      readonly shadow?: VDBVolumeShadowRequest;
    }
);

/**
 * A host environment texture (K6.2b, design §2.3, Q3): linear radiance, reduced on the GPU (no readback). Any float
 * format the device can bind as a sampled texture (read with `textureLoad` at level 0, so unfilterable formats such as
 * rgba32float work): a compute-PMREM or sky generator's output plugs in directly.
 */
export interface VDBFrameEnvironment {
  /** Identity of the source: a new id or revision re-runs the prefilter. */
  id: unknown;
  revision: number;
  /** Equirect: u = atan2(m.z, m.x) / 2π + ½ along x, row 0 the zenith (+Y). Cube: 6 layers, WebGPU's faces. */
  texture: GPUTexture;
  /** Default: 6 square layers → cube, else equirect (a `GPUTextureView` has no readable dimension). */
  projection?: "equirect" | "cube";
  /**
   * World → map, 3 × 3 column-major (m = R · d): a rotation, or a rotation with a reflection (three's flipY equirect is
   * diag(1, −1, 1) · Rᵀ, its WebGPU cube diag(−1, 1, 1) · Rᵀ). Default identity.
   */
  rotation?: Float32Array;
  /** Scales the environment lights and irradiance (three's `scene.environmentIntensity`). */
  intensity: number;
}

/**
 * A host depth shadow map (K6.5b, design §3.7): the meshes the host drew from one light, so they shadow the volumes
 * (mesh → volume shadows; `settings.occlusion`). The core tests a world position against the first layer whose clip
 * volume holds it (2 × 2 PCF, or one texel), as the host's own receivers do; outside every layer it is lit. At most 4
 * maps are bound per kernel (fixed slots; more are warned about and ignored); the core reads, never writes or frees, it.
 */
export interface VDBFrameOccluder {
  /** The {@link VDBFrameLight} `id` the map belongs to (a directional, point or spot light of this frame). */
  readonly light: number | string;
  /**
   * A `'2d-array'` view of the map: a depth texture (`'depth-only'` aspect for a depth-stencil format) or an `r32float`
   * texture, `TEXTURE_BINDING`; one layer per cascade or cube face (≤ 6). Read with `textureLoad` (no sampler).
   */
  readonly depth: GPUTextureView;
  /** Per layer: the view-projection the layer was drawn with (column-major, WebGPU depth [0, 1]); ≤ 6. */
  readonly viewProjection: readonly ArrayLike<number>[];
  /** Added to the position's depth before the test (three's `shadow.bias` sign; reversed depth: subtracted). Default 0. */
  readonly bias?: number;
  /** The map stores 1 at the near plane. Default false. */
  readonly reversedDepth?: boolean;
  /** `'bilinear'` (default): 2 × 2 percentage-closer filtering with bilinear weights; `'nearest'`: one texel. */
  readonly filter?: "bilinear" | "nearest";
  /**
   * Bump when the map's texels or matrices changed (the 'frame' transmittance redraws then); omitted: the map is taken
   * to change every update.
   */
  readonly revision?: number;
}

/** What one `update()` gets from the host. */
export interface VDBFrameInput {
  /** Monotonic: jitter index; a skipped id drops the temporal history. */
  readonly frameId: number;
  readonly camera: VDBFrameCamera;
  /** Drawing-buffer and canvas (CSS) pixels: the froxel size (`pixelsPerFroxel`). */
  readonly viewport: { readonly drawingBuffer: readonly [number, number]; readonly canvas: readonly [number, number] };
  /** Only the lights of the rendered scene. */
  readonly lights: readonly VDBFrameLight[];
  readonly environment?: VDBFrameEnvironment | null;
  /** K6.5b: the host's depth shadow maps (mesh → volume shadows), at most 4 used. */
  readonly occluders?: readonly VDBFrameOccluder[];
  /** K6.5: the scene has an emission light: draw the emission probes with these settings (null: none). */
  readonly emissionLight?: VDBFrameEmissionLightSettings | null;
}

/** Texture wrapping of a volume's lookup (his `wrap3D`). */
export type VDBFrameWrap = "clamp" | "repeat" | "mirror";

/** One volume's parameters: his `FrameVolume` props (names and defaults) for the K6.1 subset. */
export interface VDBFrameVolumeParams {
  /** The volume box: the local unit box [-0.5, 0.5]³ → world (his `volume.matrixWorld × boxMatrix`). */
  matrixWorld: VDBFrameMat4;
  baseColor: VDBFrameVec3;
  scatterColor: VDBFrameVec3;
  extinction: number;
  /** Spectral (K6.4). */
  extinctionScale: VDBFrameVec3;
  /** Spectral (K6.4). */
  angstromExponent: number;
  densityScale: number;
  normalizeDensity: boolean;
  /** The mean density `normalizeDensity` divides by; null: the source's `stats.meanNonZero`. */
  densityMean: number | null;
  densityCutoff: number;
  absorbance: number;
  opacity: number;
  emissiveIntensity: number;
  temperatureRange: readonly [number, number];
  offset3D: VDBFrameVec3;
  wrap3D: VDBFrameWrap;
  /** {@link VDB_FRAME_LIGHT_BITS}. */
  lights: number;
  shadowSteps: number;
  mipmaps: boolean;
  /** Sparse sources (K6.7); dense ignores it. */
  hdda: boolean;
  /** 'volume' lighting phase overrides (V65); null: the renderer settings'. */
  phase: VDBFramePhaseName | null;
  phaseG: number | null;
  phaseBackG: number | null;
  phaseBackWeight: number | null;
  phaseGExponent: number | null;
  /** K6.5: the emission colour (default `'white'`). */
  emissionColor: VDBEmissionColor;
}

/** His FrameVolume defaults. */
export function vdbFrameDefaultVolumeParams(): VDBFrameVolumeParams {
  return {
    matrixWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    baseColor: [1, 1, 1],
    scatterColor: [0, 0, 0],
    extinction: 100,
    extinctionScale: [1, 1, 1],
    angstromExponent: 0,
    densityScale: 1,
    normalizeDensity: true,
    densityMean: null,
    densityCutoff: 0,
    absorbance: 1,
    opacity: 1,
    emissiveIntensity: 1,
    temperatureRange: [0, 6500],
    offset3D: [0, 0, 0],
    wrap3D: "clamp",
    lights: VDB_FRAME_ALL_LIGHTS,
    shadowSteps: 32,
    mipmaps: true,
    hdda: false,
    phase: null,
    phaseG: null,
    phaseBackG: null,
    phaseBackWeight: null,
    phaseGExponent: null,
    emissionColor: "white",
  };
}

/** The host's composite targets (`render()`). */
export interface VDBFrameTargets {
  /** The colour attachment the volumes composite onto (single-sample; the format of `options.composite`). */
  readonly color: GPUTextureView;
  /** Pixels of `color` (the composite's screen size). */
  readonly size: readonly [number, number];
  /** The scene depth: a depth texture or r32float, single-sample. */
  readonly depth: GPUTextureView;
  /** The scene colour before the volumes (sRGB blending reads it); required with `srgbBlending`. */
  readonly sceneColor?: GPUTextureView;
  /** Clamp colours to [0, 1] in the sRGB blend (true without tone mapping, his rule). Default true. */
  readonly clampColors?: boolean;
  /** `colorSpace` of `color`: sRGB blending applies only for 'srgb' output (three: `outputColorSpace`). Default 'srgb'. */
  readonly outputColorSpace?: "srgb" | "linear";
}

/** The froxel layout of the outputs. */
export interface VDBFroxelLayout {
  /** Froxels across, up, and slices. */
  readonly width: number;
  readonly height: number;
  readonly depth: number;
  /** View depths of the first and last slice boundaries. */
  readonly range: readonly [number, number];
  /** Clip-space rectangle of the froxel area with one cell of padding (x0, y0, x1, y1); empty: all 0. */
  readonly screenRect: readonly [number, number, number, number];
}

/** What the renderer gives the host (design §2.5). */
export interface VDBFrameOutputs {
  /** +1 when a texture below is a new object: rebuild host bind groups. */
  readonly version: number;
  /**
   * The integrated froxels: `integrated` is W × H × (D + 1) rgba16float, boundary k = (radiance of slices
   * < k, opacity 1 − T); read with `vdbFrameIntegrateWgsl`. Null before the first update with volumes.
   */
  readonly froxels: { readonly integrated: GPUTexture; readonly integratedAlpha: GPUTexture | null; readonly layout: VDBFroxelLayout } | null;
  /** Frame uniforms (`VdbFrameUniforms`) for host shaders. */
  readonly uniforms: GPUBuffer;
  /** K6.5: the volume shadow maps by light id (lights with a `shadow` request; `settings.castShadows`). */
  readonly volumeShadows: ReadonlyMap<number | string, VDBVolumeShadowMap>;
  /** K6.5: the emission probes (with `VDBFrameInput.emissionLight` and an emitting volume), else null. */
  readonly emissionProbes: VDBFrameEmissionProbes | null;
}

/** The result of `update()`. */
export interface VDBFrameUpdate {
  readonly outputs: VDBFrameOutputs;
  /** True when kernels were recorded (some volume is in view); false: draw nothing this frame. */
  readonly recorded: boolean;
  /** Volumes drawn this frame. */
  readonly volumes: number;
}

/** Redraw counters of the 'frame' lighting caches (K6.2; zeros in K6.1). */
export interface VDBFrameRedrawStats {
  shadowDensity: { drawn: number; skipped: number };
  transmittance: { drawn: number; skipped: number };
  environmentGrid: { drawn: number; skipped: number };
  /** K6.5: volume shadow map pieces drawn and waiting for the budget in the last update (his `lastUpdate`). */
  volumeShadows: { drawn: number; waiting: number };
}

/** Handle of one volume in the renderer. */
export interface VDBFrameVolumeGpu {
  /** One writer (the host); packed at each `update()`. */
  params: VDBFrameVolumeParams;
  /** Drawn this frame (host culling, layers). */
  visible: boolean;
  /** 'frame' lighting (K6.2). */
  castsShadow: boolean;
  readonly slot: VDBVolumeSlot;
  remove(): void;
}
