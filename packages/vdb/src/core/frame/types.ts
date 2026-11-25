/**
 * Plain-data types of the core frame renderer (K6.1, design §2.1–§2.5). Matrices are column-major
 * `ArrayLike<number>` (three's `Matrix4.elements` order); colours are linear RGB. K6.1 implements
 * 'volume' lighting on dense sources; the environment, occluders and shadow requests are typed here and
 * ignored with a one-time warning until K6.2 / K6.5 / K6.5b.
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

/** A volume shadow map request (K6.5; typed, ignored in K6.1). */
export interface VDBVolumeShadowRequest {
  size: number;
  view?:
    | { viewProjection: Float32Array; near: number; far: number; orthographic: boolean }
    | { cascades: { viewProjection: Float32Array; near: number; far: number }[]; breaks: number[] }
    | "fit";
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

/** A host environment texture (K6.2; typed, ignored in K6.1). */
export interface VDBFrameEnvironment {
  id: unknown;
  revision: number;
  texture: GPUTexture;
  projection?: "equirect" | "cube";
  rotation?: Float32Array;
  intensity: number;
}

/** A host depth shadow map (K6.5b; typed, ignored in K6.1). */
export interface VDBFrameOccluder {
  light: number | string;
  depth: GPUTextureView;
  viewProjection: Float32Array[];
  orthographic: boolean;
  bias: number;
  reversedDepth: boolean;
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
  readonly occluders?: readonly VDBFrameOccluder[];
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
