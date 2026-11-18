// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameRenderer.js L79–135 (caster signatures, fullscreen geometry,
//   targets, scenes), L144–174 (getClipRect, getMaxTextureSize), L191–246 (settings, the ported
//   subset), L249–430 (fields, constructor), L431–481 (render, createProxyGeometry), L493–570
//   (isFrameLighting, renderEnvironmentGrid), L572–596 (renderEnvironmentMap), L598–645
//   (collectVolumes, usesShadowGrid, isNeural), L715–895 (getSliceRange, updateTargets,
//   updateGridTargets), L897–1005 (renderFrame, updateFroxelArea, renderTemporal), L1007–1061
//   (copyEnvironment, prepareLightScene, renderInject), L1063–1308 (renderFrameLighting,
//   transmittanceUnchanged, renderShadowDensity, updateGridVersion), L1347–1474 (pool setup,
//   updatePool, getFroxelRect, getCascadeCount, updateGrid, getGridRange), L1476–1716 (light
//   records, visitLight, updateLights), L1718–1784 (dispose);
//   src/three-openvdb/frame/webgpu/frameShaderChunks.js L800–830 (setVolumeUniforms);
//   K3.4: FrameRenderer.js L232–242 (multipleScattering 'neural' default, octave settings), L291–296 and
//   L642–699 (isNeural, the grid mip targets, renderGridMips), L1052–1056 and L1086–1098 (octaves,
//   two-sided and neural uniforms of the inject and transmittance passes), L1132 and L1201 (octaves of the
//   density and lighting passes), L506–518 and L559 (octaves in the environment grid signature and pass),
//   L1213–1226 (the neural, two-sided and octave fields of the transmittance signature);
//   K3.5: L176–189 (createAtlasTarget, isSpectralTarget), L335–337 (the spectral transmittance targets),
//   L514 and L556 (spectral in the environment grid signature and pass), L638–640 (isSpectral), L702–713
//   (getSpectralTransmittanceTarget), L725–866 (the spectral targets of updateTargets / updateGridTargets),
//   L908 (environment lights in both lighting modes), L973–1003 (the alpha targets of renderTemporal),
//   L1030 (the inject pass reads the scene environment), L1079–1196 (the slot group loop and the spectral maps
//   of renderFrameLighting), L1210–1231 (the spectral fields of the transmittance signature), L1303
//   (spectral in the grid signature);
//   K3.6: L218–230 (the castShadow* settings), L269 and L456 (volumeShadows, updated before each frame), L340–350
//   (gridRenderId / gridCamera, castVersion), L634–636 (usesShadowGrid with castShadows), L1250–1290
//   (renderShadowDensity onlyLast), L1310–1345 (prepareShadowGrid), L1719 (dispose of the volume shadows).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript; one inject, density and shadow density material per volume source (the
// source is compiled into each; DIVERGES V30); mesh shadows (K3.6, volume-shadows.ts); the neural
// network is a settable value (`neuralNetwork`; null: diffusion, with a warning); per-pass timing
// hooks and a redraw-skip counter. Divergences: DIVERGES.md (K3.1, K3.3
// and K3.4 rows), V23–V25, V30–V34.

/**
 * The froxel renderer (K3.1 `'volume'` lighting, K3.3 `'frame'` lighting; R1 §2). Each
 * `VDBFrameVolume` proxy calls {@link VDBFrameRenderer.render} from `onBeforeRender`; the first
 * proxy in a render draws the froxels into a slice atlas, and every proxy then draws the integrate
 * material, which composites the atlas over the scene up to the scene depth.
 *
 * `'frame'` lighting: shadow density of every caster into a world grid (≤ 4 cascades, redrawn only
 * when its signature changes), transmittance from each grid voxel to ≤ 6 light slots and the sky,
 * an optional environment grid ('directional' / 'spherical-harmonics' environment lights from a
 * PMREM readback), the density pass per volume, one lighting pass for all volumes, then temporal
 * reprojection.
 */

import * as THREE from "three";
import type { WebGPURenderer } from "three/webgpu";
import { NodeMaterial } from "three/webgpu";
import * as TSL from "three/tsl";
import type { VDBTslNode } from "../grid-node.js";
import { VDBFrameEnvironmentLights, frameEnvironmentMapHeight, frameEnvironmentMapWidth, frameEnvironmentModes, type VDBFrameEnvironmentResult } from "../../tools/frame/environment.js";
import {
  vdbFrameCascadeCasters,
  vdbFrameGridCascades,
  vdbFrameGridRange,
  vdbFrameGridSignatureState,
  vdbFrameUpdateGridSignatures,
  vdbSameValues,
  vdbSignaturePushElements,
  vdbSignaturePushVector,
  type VDBFrameGridSignatureState,
} from "../../tools/frame/grid.js";
import {
  createAreaUniforms,
  createEnvironmentGridUniforms,
  createFroxelUniforms,
  createGridUniforms,
  createLightUniforms,
  createTileUniforms,
  frameMaxHemisphereLights,
  frameMaxLights,
  frameMaxShadowCascades,
  frameTileVertex,
  type FrameVolumeUniforms,
} from "./chunks.js";
import { VDBFrameDensityMaterial, createDensityShadowUniforms, frameDensityEnvironment } from "./density-material.js";
import { VDBFrameEnvironmentMapMaterial } from "./environment-material.js";
import { VDBFrameEnvironmentGridMaterial } from "./environment-grid-material.js";
import { VDBFrameInjectMaterial } from "./inject-material.js";
import { VDBFrameIntegrateMaterial } from "./integrate-material.js";
import { getAtlasLayout, getFroxelSize } from "./layout.js";
import { VDBFrameLightingMaterial } from "./lighting-material.js";
import { getFramePhase, getFramePhaseSpectrum, getFrameMultipleScattering, getFrameScatteringOctaves, getFrameTwoSidedDiffusion, isFrameDiffusion, type VDBFrameMultipleScattering, type VDBFramePhase } from "./phase.js";
import { VDBFrameGridMipMaterial } from "./grid-mip-material.js";
import { vdbFrameNeuralCheck, type VDBFrameNeuralNetwork } from "../../tools/frame/neural.js";
import { vdbFrameNeuralWeights } from "../../tools/frame/neural-weights.js";
import { VDBFrameShadowDensityMaterial } from "./shadow-density-material.js";
import { getFrameExtinctionSpectrum, getFrameSpectralGroups } from "./spectrum.js";
import { VDBFrameTemporalMaterial } from "./temporal-material.js";
import { VDBFrameTransmittanceMaterial, frameNeuralMipLevels, frameShadowLightSlots } from "./transmittance-material.js";
import { VDBFrameVolumeShadows } from "./volume-shadows.js";
import { FrameVolumeCollector, getFrameFadeRadius, getFroxelArea, getFroxelRect, getFroxelSliceRange, type FrameVolumeEntry, type FrameVolumeRecord } from "./view.js";
import type { VDBFrameVolume } from "./volume.js";
import type { VDBVolumeUniforms } from "../volume-source.js";

const { outputStruct, vec4 } = TSL as unknown as Record<string, VDBTslNode>;

/** Renderer settings: his `frameRenderer.settings`, the subset ported in K3.1–K3.3 (same names and meanings). */
export interface VDBFrameRendererSettings {
  /** Froxel grid width (default 512). */
  width: number;
  /** Froxel grid height (default 512). */
  height: number;
  /** N > 0: at most one froxel per N x N canvas (CSS) pixels. 0: no cap. */
  pixelsPerFroxel: number;
  /** Slices (default 24). */
  depth: number;
  /** Most sub-steps across one slice (default 32). */
  subSteps: number;
  near: number;
  far: number;
  /** View distance where the density fades to 0 (FogExp2-like). Infinity: no fade. */
  fadeRadius: number;
  /** Golden-ratio jitter of the sub-steps, shadow start and grid marches (default false). */
  jitter: boolean;
  /** His froxel reprojection with a neighbourhood clamp (default true, his; K3.4 flipped it with `lighting` and `multipleScattering`, V33). */
  temporal: boolean;
  /** History weight of the temporal pass (0.01–1; his default 0.2). */
  temporalBlend: number;
  /** `'volume'`: each volume lights itself with its own shadow march. `'frame'`: the world shadow grid (default, his; since K3.4, V33). */
  lighting: "volume" | "frame";
  /** `'frame'` lighting: voxels on each side of each shadow grid cascade (default 64). */
  shadowResolution: number;
  /** `'frame'` lighting: shadow grid cascades, 1–4 (default 1). */
  shadowCascades: number;
  /** `'frame'` lighting: half size of the finest cascades around the camera (default 512). */
  shadowDistance: number;
  /** `'frame'` lighting: one phase for all volumes (his demos set 'dual-lobe'). */
  phase: VDBFramePhase;
  phaseG: number;
  phaseBackG: number;
  phaseBackWeight: number;
  phaseGExponent: number;
  /** Blend the volumes over the scene in sRGB, as WebGLRenderer. false blends in linear space. */
  srgbBlending: boolean;
  /**
   * `'frame'` lighting: the scene environment as lights: `'outward'` (PMREM irradiance along each
   * volume's outward direction), `'directional'` (≤ 8 lights from a 64 × 32 readback, default) or
   * `'spherical-harmonics'` (SH quadrature lights).
   */
  environmentLighting: "outward" | "directional" | "spherical-harmonics";
  /** Lights of `'directional'` (1–8, default 8). */
  environmentSamples: number;
  /** SH order of `'spherical-harmonics'` (1 or 2, default 1). */
  environmentShOrder: number;
  /**
   * Multiple scattering in the shadow marches: false (single scattering), true (1D diffusion toward the
   * light), `'two-sided'` (2D slab diffusion toward and away from the light), `'octaves'` (indirect light
   * octaves, `scatteringOctave*`) or `'neural'` (default, his: the scattered light of the scene light
   * slots from a small network in `'frame'` lighting, diffusion elsewhere; `neuralNetwork`).
   */
  multipleScattering: VDBFrameMultipleScattering;
  /** `'octaves'`: octaves N (1–8, default 4). Octave n has extinction × aⁿ, light × bⁿ, phase g × cⁿ. */
  scatteringOctaves: number;
  /** `'octaves'`: a (0–1, default 0.5). */
  scatteringOctaveExtinction: number;
  /** `'octaves'`: b (0–1, default 0.5). */
  scatteringOctaveContribution: number;
  /** `'octaves'`: c (0–1, default 0.5). */
  scatteringOctavePhase: number;
  /**
   * Spectral media (K3.5, his `spectral`, default false): per-channel extinction (each volume's
   * `extinctionScale` × (λ / 550)^−`angstromExponent`), per-channel albedo and multiple scattering. 'frame'
   * lighting draws one transmittance pass per slot group (3 targets each); the froxel atlas gets a second
   * target. `'neural'` runs as 1D diffusion (his `isNeural`). See `getFrameMediumProps` for measured media.
   */
  spectral: boolean;
  /**
   * K3.6 (his `castShadows`, default false): volumes cast shadows on meshes, for directional, point and spot lights
   * with `castShadow` (and `renderer.shadowMap.enabled`; receivers need `receiveShadow`). Beer shadow maps from the
   * world shadow grid's last cascade, multiplied into each light's shadow (`volume-shadows.ts`). Uses three r187
   * internals, self-tested (VDB_ENGINE_MISMATCH on another three).
   */
  castShadows: boolean;
  /** Texels on each side of a volume shadow map (each cascade tile and cube face; default 512). */
  castShadowMapSize: number;
  /** Directional lights: cascades of the volume shadow map (1–4, default 1). */
  castShadowCascades: number;
  /** Directional lights: the largest view distance the cascades cover (default 4000). */
  castShadowDistance: number;
  /** Grid samples on each light ray of a volume shadow map (1–256, default 32). */
  castShadowSteps: number;
  /** Map pieces drawn again in one render (a cascade tile or spot map 1, a cube map 6; default 1). */
  castShadowUpdatesPerFrame: number;
  /** Very small maps (64) and few samples (8): soft blob shadows at almost no cost (default false). */
  castShadowCoarse: boolean;
}

/** Per-pass hook (D25: timings are provisional). */
export type VDBFramePass = "froxels" | "inject" | "shadowDensity" | "gridMips" | "transmittance" | "environmentGrid" | "density" | "lighting" | "temporal" | "volumeShadows" | "emissionProbes";

export interface VDBFramePassHook {
  begin?(pass: VDBFramePass, detail?: { volume: VDBFrameVolume }): void;
  end?(pass: VDBFramePass, detail?: { volume: VDBFrameVolume }): void;
}

/** Redraw counters of the signature-cached `'frame'` passes (drawn / skipped renders). Not his. */
export interface VDBFrameRedrawStats {
  shadowDensity: { drawn: number; skipped: number };
  transmittance: { drawn: number; skipped: number };
  environmentGrid: { drawn: number; skipped: number };
}

const clearColor = new THREE.Color();
const vector3 = new THREE.Vector3();
const froxelSize = { width: 1, height: 1 };
const lightPosition = new THREE.Vector3();
const lightTarget = new THREE.Vector3();
const lightColor = new THREE.Color();
const cameraPosition = new THREE.Vector3();
const extinctionSpectrum = new THREE.Vector3();

// NOTE Instanced, then frame volume proxies skip the draw with instanceCount 0 (three.js does not draw 0 instances)
export const createFullscreenGeometry = () => {
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.instanceCount = 1;
  geometry.setAttribute("position", new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));

  return geometry;
};

const tileQuadPosition = new THREE.Float32BufferAttribute([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, 1, 0], 3);

const createRenderTarget = (width: number, height: number, options: THREE.RenderTargetOptions = {}) =>
  new THREE.RenderTarget(width, height, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    generateMipmaps: false,
    depthBuffer: false,
    ...options,
  });

// NOTE Froxel atlas. Spectral: a second target with the red and blue alpha (and the green alpha in w for blending).
const createAtlasTarget = (width: number, height: number, spectral: boolean) => {
  if (!spectral) {
    return createRenderTarget(width, height);
  }

  const target = createRenderTarget(width, height, { count: 2 });

  target.textures[1]!.format = THREE.RGFormat;

  return target;
};

const isSpectralTarget = (target: THREE.RenderTarget) => target.textures.length > 1;

const createScene = () => {
  const scene = new THREE.Scene();
  scene.matrixWorldAutoUpdate = false;

  return scene;
};

// NOTE Instanced tile quads: one instance for each slice
const createTileGeometry = () => {
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.instanceCount = 0;
  geometry.setAttribute("position", tileQuadPosition);

  return geometry;
};

// NOTE Clip rectangle of a froxel cell rectangle, padding cells larger on each side. An empty rectangle stays empty.
const clipRect = { x0: 0, y0: 0, x1: 0, y1: 0 };

const getClipRect = (rect: THREE.Vector4, width: number, height: number, padding: number) => {
  if (rect.z <= rect.x || rect.w <= rect.y) {
    clipRect.x0 = clipRect.y0 = clipRect.x1 = clipRect.y1 = 0.0;

    return clipRect;
  }

  clipRect.x0 = (Math.max(0, rect.x - padding) / width) * 2.0 - 1.0;
  clipRect.y0 = (Math.max(0, rect.y - padding) / height) * 2.0 - 1.0;
  clipRect.x1 = (Math.min(width, rect.z + padding) / width) * 2.0 - 1.0;
  clipRect.y1 = (Math.min(height, rect.w + padding) / height) * 2.0 - 1.0;

  return clipRect;
};

const getMaxTextureSize = (renderer: WebGPURenderer) => {
  const backend = renderer.backend as unknown as { device?: GPUDevice };

  return backend.device ? backend.device.limits.maxTextureDimension2D : Infinity;
};

// NOTE What the shadow density pass reads of a caster's source: its identity, version and LOD mode
// (his pushCasterMaps pushes the texture ids and versions). A new source or a replaced grid draws again.
const sourceIds = new WeakMap<object, number>();
let nextSourceId = 1;

const pushCasterSource = (values: number[], frameVolume: VDBFrameVolume) => {
  const source = frameVolume.source as unknown as { version?: number; lod?: string; lodLevels: number; kind: string; textures?: Record<string, { map: THREE.Texture; mip: THREE.Texture | null } | null> };
  let id = sourceIds.get(source);

  if (id === undefined) {
    id = nextSourceId++;
    sourceIds.set(source, id);
  }

  values.push(id, source.version ?? -1, source.lodLevels, source.lod === "node-average" ? 2 : source.lod === "none" ? 0 : 1);

  if (source.textures) {
    for (const name of ["density", "mask", "albedo"]) {
      const t = source.textures[name];

      values.push(t ? t.map.id : -1, t ? t.map.version : -1, t?.mip ? t.mip.id : -1, t?.mip ? t.mip.version : -1);
    }
  }
};

// NOTE An id for each network object (the transmittance signature: a swapped network draws again)
const networkIds = new WeakMap<object, number>();
let nextNetworkId = 1;

const networkId = (network: VDBFrameNeuralNetwork): number => {
  let id = networkIds.get(network);

  if (id === undefined) {
    id = nextNetworkId++;
    networkIds.set(network, id);
  }

  return id;
};

interface LightRecord {
  position: THREE.Vector4;
  color: THREE.Vector4;
  spot: THREE.Vector4;
  falloff: THREE.Vector2;
}

interface LightList {
  length: number;
  entries: LightRecord[];
}

interface PoolEntry {
  mesh: THREE.Mesh<THREE.InstancedBufferGeometry, THREE.Material>;
  geometry: THREE.InstancedBufferGeometry;
  frameVolume: VDBFrameVolume | null;
  volume: FrameVolumeRecord | null;
  rect: THREE.Vector4;
  sliceStart: number;
}

interface PassMaterial extends THREE.Material {
  volumeUniforms: FrameVolumeUniforms;
  sourceUniforms: VDBVolumeUniforms;
  frameRect: { value: THREE.Vector4 };
  frameSliceStart: { value: number };
}

interface LightResult {
  shadowLightCount: number;
  environmentLightCount: number;
  environmentShadowSlot: number;
  environmentGridLights: { direction: number[]; color: number[] }[];
}

type RendererLike = WebGPURenderer & { isWebGPURenderer?: boolean };
type SceneLike = THREE.Object3D & { overrideMaterial?: THREE.Material | null; environment?: THREE.Texture | null; environmentIntensity?: number; environmentRotation?: THREE.Euler };

/** His `FrameRenderer` (`'volume'` and `'frame'` lighting). One instance can serve many volumes and cameras. */
export class VDBFrameRenderer {
  settings: VDBFrameRendererSettings = {
    width: 512,
    height: 512,
    pixelsPerFroxel: 0,
    depth: 24,
    subSteps: 32,
    near: 0.1,
    far: Infinity,
    fadeRadius: Infinity,
    jitter: false,
    temporal: true,
    temporalBlend: 0.2,
    lighting: "frame",
    shadowResolution: 64,
    shadowCascades: 1,
    shadowDistance: 512,
    phase: "henyey-greenstein",
    phaseG: 0.8,
    phaseBackG: -0.3,
    phaseBackWeight: 0.2,
    phaseGExponent: 0.0,
    srgbBlending: true,
    environmentLighting: "directional",
    environmentSamples: 8,
    environmentShOrder: 1,
    multipleScattering: "neural",
    scatteringOctaves: 4,
    scatteringOctaveExtinction: 0.5,
    scatteringOctaveContribution: 0.5,
    scatteringOctavePhase: 0.5,
    spectral: false,
    castShadows: false,
    castShadowMapSize: 512,
    castShadowCascades: 1,
    castShadowDistance: 4000,
    castShadowSteps: 32,
    castShadowUpdatesPerFrame: 1,
    castShadowCoarse: false,
  };

  /**
   * The network of multipleScattering `'neural'` (default: his trained weights, `@pmndrs/vdb/tools`
   * `vdbFrameNeuralWeights`; see THIRD_PARTY.md for their provenance). Set another network of the same
   * shape to swap it, or null to run `'neural'` as 1D diffusion (warns once). Not his (his is a module import).
   */
  neuralNetwork: VDBFrameNeuralNetwork | null = vdbFrameNeuralWeights;

  /** Per-pass hooks (timing, debugging). */
  passHook: VDBFramePassHook | null = null;

  /**
   * Timing/debug hook: proxies draw the integrate pass over the last froxel atlas without rendering
   * the froxels again (the per-pass timing harness times integrate alone with it). Not his.
   */
  holdFroxels = false;

  /** Redraws and skips of the signature-cached 'frame' passes (the K3.3 redraw-skip rate). Not his. */
  readonly redrawStats: VDBFrameRedrawStats = {
    shadowDensity: { drawn: 0, skipped: 0 },
    transmittance: { drawn: 0, skipped: 0 },
    environmentGrid: { drawn: 0, skipped: 0 },
  };

  // NOTE TSL uniform nodes. Passes share them, the CPU writes .value.
  readonly froxelUniforms = createFroxelUniforms();
  // NOTE Canvas and drawing buffer sizes that settings.pixelsPerFroxel holds (layout.ts getFroxelSize)
  readonly froxelHeldSize = { canvasX: 0, canvasY: 0, bufferX: 0, bufferY: 0 };
  readonly froxelTileUniforms = createTileUniforms();
  // NOTE Froxel area of all volumes. Passes over the froxel grid draw only this area.
  readonly froxelAreaUniforms = createAreaUniforms();
  readonly froxelArea = { rect: new THREE.Vector4(), first: 0, count: 0 };
  readonly areaGeometry = createTileGeometry();
  readonly gridUniforms = createGridUniforms(this.froxelUniforms.frameJitter);
  readonly gridTileUniforms = createTileUniforms();
  readonly densityShadowUniforms = createDensityShadowUniforms();
  readonly lightUniforms = createLightUniforms();
  readonly environmentGridUniforms = createEnvironmentGridUniforms();
  // NOTE Octaves of the settings (getFrameScatteringOctaves), for the signatures
  readonly scatteringOctaves = new THREE.Vector4();

  readonly integrateMaterial: VDBFrameIntegrateMaterial;
  readonly lightingMaterial: VDBFrameLightingMaterial;
  readonly transmittanceMaterial: VDBFrameTransmittanceMaterial;
  readonly temporalMaterial: VDBFrameTemporalMaterial;

  // NOTE The light scene draws the volume inject pass or the lighting pass. The inject mesh takes the inject material of each volume.
  readonly injectScene = createScene();
  readonly injectMesh: THREE.Mesh<THREE.InstancedBufferGeometry, THREE.Material> = new THREE.Mesh(createTileGeometry(), new THREE.MeshBasicMaterial());
  readonly lightingMesh: THREE.Mesh<THREE.InstancedBufferGeometry, THREE.Material>;

  readonly densityScene = createScene();
  densityPool: PoolEntry[] = [];
  // NOTE Clears the density targets in the froxel area (one draw for all targets)
  readonly densityClearMaterial = new NodeMaterial();
  readonly densityClearScene = createScene();
  readonly densityClearMesh: THREE.Mesh<THREE.InstancedBufferGeometry, NodeMaterial>;

  readonly shadowScene = createScene();
  shadowPool: PoolEntry[] = [];

  readonly transmittanceScene = createScene();
  readonly transmittanceMesh: THREE.Mesh<THREE.InstancedBufferGeometry, THREE.Material>;

  readonly temporalScene = createScene();
  readonly temporalMesh: THREE.Mesh<THREE.InstancedBufferGeometry, THREE.Material>;
  historyTargets: [THREE.RenderTarget | null, THREE.RenderTarget | null] = [null, null];
  historyIndex = 0;
  historyValid = false;
  historyCamera: THREE.Camera | null = null;
  readonly previousViewMatrix = new THREE.Matrix4();
  readonly previousProjectionMatrix = new THREE.Matrix4();
  readonly previousRange = new THREE.Vector2(0.1, 1.0);
  jitterIndex = 0;

  // NOTE Environment lights from a small map of the scene environment (tools/frame/environment.ts)
  environmentLights: VDBFrameEnvironmentLights<RendererLike, SceneLike & VDBFrameEnvironmentSceneFields>;
  readonly environmentMapMaterial = new VDBFrameEnvironmentMapMaterial();
  readonly environmentMapScene = createScene();
  readonly environmentMapMesh: THREE.Mesh<THREE.InstancedBufferGeometry, THREE.Material>;
  readonly environmentMapCamera = new THREE.OrthographicCamera(-1.0, 1.0, 1.0, -1.0, 0.0, 1.0);
  environmentMapTarget: THREE.RenderTarget | null = null;
  environment: VDBFrameEnvironmentResult | null = null;

  // NOTE Environment grid: environment light of the last cascade voxels, from the environment lights with no slot
  readonly environmentGridMaterial: VDBFrameEnvironmentGridMaterial;
  readonly environmentGridScene = createScene();
  readonly environmentGridMesh: THREE.Mesh<THREE.InstancedBufferGeometry, THREE.Material>;
  environmentGridTarget: THREE.RenderTarget | null = null;
  environmentGridSignature: number[] = [];
  nextEnvironmentGridSignature: number[] = [];

  // NOTE multipleScattering 'neural': mips of the last grid cascade, levels 1 and up (VDBFrameGridMipMaterial)
  readonly gridMipMaterial = new VDBFrameGridMipMaterial();
  readonly gridMipScene = createScene();
  readonly gridMipMesh: THREE.Mesh<THREE.InstancedBufferGeometry, THREE.Material>;
  gridMipTargets: THREE.RenderTarget[] = [];
  gridMipSignature: number[] = [];

  atlasTarget: THREE.RenderTarget | null = null;
  densityTarget: THREE.RenderTarget | null = null;
  shadowDensityTarget: THREE.RenderTarget | null = null;
  transmittanceTarget: THREE.RenderTarget | null = null;
  // NOTE Renderer settings.spectral: 3 transmittance targets for each slot group in use (spectrum.ts).
  // Made when a group is first used. transmittanceTarget is null then.
  spectralTransmittanceTargets: THREE.RenderTarget[] = [];
  lightLimitWarned = false;
  private readonly warned = new Set<string>();

  // NOTE Shadow grid state: cascades of the last update, signatures (tools/frame/grid.ts)
  readonly gridState: VDBFrameGridSignatureState = vdbFrameGridSignatureState();
  // NOTE World shadow grid of the current render. Volume shadows make it before the first lit mesh; 'frame' lighting
  // then uses it again (gridFreshCamera). gridState.castVersion changes when the last cascade (all volumes) changes.
  gridRenderId = -1;
  gridCamera: THREE.Camera | null = null;
  gridFreshCamera: THREE.Camera | null = null;
  gridReady = false;
  /** K3.6: volume shadows on meshes (settings.castShadows). */
  volumeShadows: VDBFrameVolumeShadows;
  // NOTE gridVersion and target of the last shadow density draw. The same version is not drawn again.
  shadowDrawnVersion = -1;
  shadowDrawnTarget: THREE.RenderTarget | null = null;
  transmittanceSignature: number[] = [];
  nextTransmittanceSignature: number[] = [];

  // NOTE collectVolumes result of the current render
  readonly collected: { renderId: number; camera: THREE.Camera | null; scene: THREE.Object3D | null; result: FrameVolumeCollector<VDBFrameVolume> | null } = {
    renderId: -1,
    camera: null,
    scene: null,
    result: null,
  };
  readonly volumeCollector = new FrameVolumeCollector<VDBFrameVolume>();
  readonly sliceRange = { first: 0, count: 0 };
  readonly gridRange = { rect: new THREE.Vector4(), first: 0, count: 0 };
  readonly cascadeCasterPool: { items: { caster: FrameVolumeEntry<VDBFrameVolume>; cascade: number }[]; entries: { caster: FrameVolumeEntry<VDBFrameVolume>; cascade: number }[] } = { items: [], entries: [] };
  readonly lightState = {
    camera: null as THREE.Camera | null,
    hemiCount: 0,
    hasSky: false,
    directions: { length: 0, entries: [] } as LightList,
    points: { length: 0, entries: [] } as LightList,
    spots: { length: 0, entries: [] } as LightList,
  };
  readonly lightResult: LightResult = { shadowLightCount: 0, environmentLightCount: 0, environmentShadowSlot: 0, environmentGridLights: [] };
  readonly environmentGridLightEntries: { direction: number[]; color: number[] }[] = [];

  cachedFrame = -1;
  cachedCamera: THREE.Camera | null = null;
  rendererWarned = false;

  constructor() {
    this.integrateMaterial = new VDBFrameIntegrateMaterial({ froxelUniforms: this.froxelUniforms });
    this.lightingMaterial = new VDBFrameLightingMaterial({
      froxelUniforms: this.froxelUniforms,
      froxelTileUniforms: this.froxelTileUniforms,
      froxelAreaUniforms: this.froxelAreaUniforms,
      lightUniforms: this.lightUniforms,
      gridUniforms: this.gridUniforms,
      densityShadowUniforms: this.densityShadowUniforms,
    });
    this.transmittanceMaterial = new VDBFrameTransmittanceMaterial({ gridUniforms: this.gridUniforms, lightUniforms: this.lightUniforms });
    this.temporalMaterial = new VDBFrameTemporalMaterial({ froxelUniforms: this.froxelUniforms, froxelTileUniforms: this.froxelTileUniforms, froxelAreaUniforms: this.froxelAreaUniforms });
    this.environmentGridMaterial = new VDBFrameEnvironmentGridMaterial({ gridUniforms: this.gridUniforms, environmentGridUniforms: this.environmentGridUniforms, lightUniforms: this.lightUniforms });
    this.environmentLights = new VDBFrameEnvironmentLights((renderer, scene) => this.renderEnvironmentMap(renderer, scene));
    this.volumeShadows = new VDBFrameVolumeShadows(this);

    this.lightingMesh = new THREE.Mesh(this.areaGeometry, this.lightingMaterial);
    this.densityClearMesh = new THREE.Mesh(this.areaGeometry, this.densityClearMaterial);
    this.transmittanceMesh = new THREE.Mesh(createFullscreenGeometry(), this.transmittanceMaterial);
    this.temporalMesh = new THREE.Mesh(this.areaGeometry, this.temporalMaterial);
    this.environmentMapMesh = new THREE.Mesh(createFullscreenGeometry(), this.environmentMapMaterial);
    this.environmentGridMesh = new THREE.Mesh(createFullscreenGeometry(), this.environmentGridMaterial);
    this.gridMipMesh = new THREE.Mesh(createFullscreenGeometry(), this.gridMipMaterial);

    for (const mesh of [this.injectMesh, this.lightingMesh, this.transmittanceMesh, this.temporalMesh, this.densityClearMesh, this.environmentMapMesh, this.environmentGridMesh, this.gridMipMesh]) {
      mesh.frustumCulled = false;
    }

    this.temporalScene.add(this.temporalMesh);

    this.densityClearMaterial.vertexNode = frameTileVertex(this.froxelTileUniforms, this.froxelAreaUniforms.frameAreaRect, this.froxelAreaUniforms.frameAreaSliceStart);
    this.densityClearMaterial.fragmentNode = outputStruct(vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0));
    this.densityClearMaterial.depthTest = false;
    this.densityClearMaterial.depthWrite = false;
    this.densityClearMaterial.blending = THREE.NoBlending;
    this.densityClearScene.add(this.densityClearMesh);

    this.injectScene.add(this.injectMesh, this.lightingMesh);
    this.transmittanceScene.add(this.transmittanceMesh);
    this.environmentMapScene.add(this.environmentMapMesh);
    this.environmentGridScene.add(this.environmentGridMesh);
    this.gridMipScene.add(this.gridMipMesh);
  }

  private warnOnce(key: string, message: string): void {
    if (!this.warned.has(key)) {
      this.warned.add(key);
      console.warn(`VDBFrameRenderer: ${message}`);
    }
  }

  // NOTE Each frame volume proxy calls this. The first proxy in a frame renders the frame. Other proxies draw nothing.
  render(renderer: RendererLike, scene: SceneLike, camera: THREE.Camera, proxy: THREE.Mesh<THREE.InstancedBufferGeometry>): void {
    if (!renderer.isWebGPURenderer) {
      if (!this.rendererWarned) {
        console.error("VDBFrameVolume: needs WebGPURenderer (three/webgpu).");
        this.rendererWarned = true;
      }

      return;
    }

    const frame = renderer.info.render.calls;

    // NOTE Override passes (depth, normals) of other effects must not draw or start the frame
    if (scene.overrideMaterial) {
      proxy.geometry.instanceCount = 0;

      return;
    }

    if (frame === this.cachedFrame && camera === this.cachedCamera) {
      proxy.geometry.instanceCount = 0;

      return;
    }

    // NOTE Puts the volume shadow filter on new shadow lights (and frees the maps of removed ones)
    this.volumeShadows.update(renderer, scene);

    if (this.holdFroxels && this.atlasTarget) {
      this.cachedFrame = frame;
      this.cachedCamera = camera;
      proxy.geometry.instanceCount = 1;

      return;
    }

    const { volumes, casters } = this.collectVolumes(renderer, scene, camera);

    if (volumes.length > 0) {
      this.renderFrame(renderer, scene, camera, volumes, casters);
    }

    // NOTE Nested renders increase info.render.calls. Keep the new value. Then other proxies in this frame skip.
    this.cachedFrame = renderer.info.render.calls;
    this.cachedCamera = camera;

    proxy.geometry.instanceCount = volumes.length > 0 ? 1 : 0;
  }

  /**
   * Draws only the froxel atlas for this scene and camera (no integrate): the per-pass timing
   * harness renders this alone. Returns the number of volumes in view.
   */
  renderFroxels(renderer: RendererLike, scene: SceneLike, camera: THREE.Camera): number {
    this.collected.renderId = -1;
    const { volumes, casters } = this.collectVolumes(renderer, scene, camera);

    if (volumes.length > 0) {
      this.renderFrame(renderer, scene, camera, volumes, casters);
    }

    return volumes.length;
  }

  // NOTE A quad. The integrate material puts it on the screen rectangle of the froxel area.
  createProxyGeometry(): THREE.InstancedBufferGeometry {
    const geometry = createTileGeometry();

    geometry.instanceCount = 1;

    return geometry;
  }

  isFrameLighting(): boolean {
    return this.settings.lighting === "frame";
  }

  // NOTE The world shadow grid is used by 'frame' lighting and by volume shadows on meshes
  usesShadowGrid(): boolean {
    return this.isFrameLighting() || this.settings.castShadows;
  }

  // NOTE Visible volumes and shadow casters, one time in a render (the render id of three.js NodeFrame, internal)
  collectVolumes(renderer: RendererLike, scene: THREE.Object3D, camera: THREE.Camera): FrameVolumeCollector<VDBFrameVolume> {
    const renderId = (renderer as unknown as { _nodes?: { nodeFrame?: { renderId?: number } } })._nodes?.nodeFrame?.renderId ?? -1;
    const { collected } = this;

    if (renderId !== -1 && collected.renderId === renderId && collected.camera === camera && collected.scene === scene && collected.result) {
      return collected.result;
    }

    collected.renderId = renderId;
    collected.camera = camera;
    collected.scene = scene;
    collected.result = this.volumeCollector.collect(scene, camera as THREE.PerspectiveCamera, this.settings, this.usesShadowGrid(), null);

    if (collected.result.volumes.length > 0) {
      this.froxelUniforms.froxelRange.value.set(collected.result.near, collected.result.far);
    }

    return collected.result;
  }

  // NOTE Slices of a volume. Returns this.sliceRange ({ first, count }), used again in the next call.
  getSliceRange(volume: FrameVolumeRecord) {
    const { froxelRange, froxelGrid } = this.froxelUniforms;

    return getFroxelSliceRange(volume, froxelRange.value, froxelGrid.value.z, this.sliceRange);
  }

  private checkSettings(): void {
    const s = this.settings;
    if (this.isFrameLighting() && s.multipleScattering === "neural" && !this.neuralNetwork) {
      this.warnOnce("neural", "multipleScattering 'neural' has no network (neuralNetwork is null); the scene light slots use 1D diffusion (true).");
    }
  }

  /** The multiple scattering mode (settings.multipleScattering; 'neural' is diffusion outside the network). */
  private multipleScatteringMode(): VDBFrameMultipleScattering {
    return this.settings.multipleScattering;
  }

  isSpectral(): boolean {
    return this.settings.spectral === true;
  }

  // NOTE multipleScattering 'neural' uses the network only with grey 'frame' lighting. Other cases use diffusion.
  isNeural(): boolean {
    if (this.settings.multipleScattering !== "neural" || this.settings.spectral || !this.neuralNetwork) return false;
    const problem = vdbFrameNeuralCheck(this.neuralNetwork);
    if (problem) {
      this.warnOnce("neural-shape", `neuralNetwork ${this.neuralNetwork.name ?? ""} does not fit the shader (${problem}); using 1D diffusion.`);
      return false;
    }
    return true;
  }

  // NOTE Mips of the last grid cascade for the network. Drawn again only when the grid changes.
  renderGridMips(renderer: RendererLike, camera: THREE.Camera): void {
    const { frameGridResolution, frameGridColumns, frameGridCount } = this.gridUniforms;
    const resolution = frameGridResolution.value as number;
    const source = this.shadowDensityTarget!;
    const signature = [this.gridState.gridVersion, source.texture.id, resolution, frameGridCount.value as number, frameGridColumns.value as number];

    if (vdbSameValues(signature, this.gridMipSignature)) {
      return;
    }

    this.gridMipSignature = signature;
    this.gridMipMesh.layers.mask = camera.layers.mask;

    const material = this.gridMipMaterial;
    const { frameNeuralMipMaps, frameNeuralMipLayouts } = this.transmittanceMaterial;
    // NOTE Source of the next level: the grid atlas first, then the level before
    let previous: { target: THREE.RenderTarget; side: number; layout: { columns: number; width: number; height: number } } | null = null;

    for (let level = 1; level <= frameNeuralMipLevels; level++) {
      const side = Math.max(1, resolution >> level);
      const layout = getAtlasLayout(side, side, side);
      let target = this.gridMipTargets[level - 1];

      if (!target || target.width !== layout.width || target.height !== layout.height) {
        target?.dispose();
        target = createRenderTarget(layout.width, layout.height, { format: THREE.RGFormat });
        this.gridMipTargets[level - 1] = target;
      }

      if (previous) {
        material.frameMipSource.value = previous.target.texture;
        material.frameMipSourceResolution.value = previous.side;
        material.frameMipSourceColumns.value = previous.layout.columns;
        material.frameMipSourceTile.value = 0;
        material.frameMipSourceSize.value.set(previous.layout.width, previous.layout.height);
      } else {
        material.frameMipSource.value = source.texture;
        material.frameMipSourceResolution.value = resolution;
        material.frameMipSourceColumns.value = frameGridColumns.value as number;
        material.frameMipSourceTile.value = ((frameGridCount.value as number) - 1) * resolution;
        material.frameMipSourceSize.value.set(source.width, source.height);
      }

      material.frameMipResolution.value = side;
      material.frameMipColumns.value = layout.columns;
      material.frameMipSize.value.set(layout.width, layout.height);
      renderer.setRenderTarget(target);
      renderer.render(this.gridMipScene, camera);

      frameNeuralMipMaps[level - 1]!.value = target.texture;
      frameNeuralMipLayouts[level - 1]!.value.set(layout.columns, layout.width, layout.height);
      previous = { target, side, layout };
    }
  }

  // NOTE The 3 transmittance targets of a spectral slot group, the same size as the shadow grid
  getSpectralTransmittanceTarget(group: number): THREE.RenderTarget {
    const targets = this.spectralTransmittanceTargets;

    while (targets.length <= group) {
      const { width, height } = this.shadowDensityTarget!;

      targets.push(createRenderTarget(width, height, { count: 3 }));
    }

    return targets[group]!;
  }

  // NOTE The lighting and density passes keep their group textures bound when they do not read them: they get their
  // own placeholders (one texture each, see renderFrameLighting), not the disposed targets.
  disposeSpectralTransmittanceTargets(): void {
    this.spectralTransmittanceTargets.forEach((target) => target.dispose());
    this.spectralTransmittanceTargets = [];

    for (const map of [...this.lightingMaterial.frameSpectralTransmittanceMaps, ...this.densityShadowUniforms.frameSkyTransmittanceMaps]) {
      map.value = new THREE.Texture();
    }
  }

  updateTargets(renderer: RendererLike): void {
    const { depth, subSteps } = this.settings;
    const { width, height } = getFroxelSize(this.settings, renderer, froxelSize, this.froxelHeldSize);
    const spectral = this.isSpectral();
    const froxelAtlas = getAtlasLayout(width, height, depth);

    if (!this.atlasTarget || this.atlasTarget.width !== froxelAtlas.width || this.atlasTarget.height !== froxelAtlas.height || isSpectralTarget(this.atlasTarget) !== spectral) {
      const maxTextureSize = getMaxTextureSize(renderer);

      if (froxelAtlas.width > maxTextureSize || froxelAtlas.height > maxTextureSize) {
        console.warn("VDBFrameRenderer: froxel atlas is larger than maxTextureSize. Decrease width, height or depth.");
      }

      this.atlasTarget?.dispose();
      this.atlasTarget = createAtlasTarget(froxelAtlas.width, froxelAtlas.height, spectral);
      this.historyValid = false;
    }

    // NOTE The per-volume inject materials are set in renderInject
    this.lightingMaterial.setSpectral(spectral);
    this.temporalMaterial.setSpectral(spectral);
    this.integrateMaterial.setSpectral(spectral);

    const { froxelGrid, froxelColumns } = this.froxelUniforms;

    // NOTE A new grid layout makes the history unusable, also when the atlas keeps its size (600 x 400 to 400 x 600)
    if (froxelGrid.value.x !== width || froxelGrid.value.y !== height || froxelGrid.value.z !== depth || froxelColumns.value !== froxelAtlas.columns) {
      this.historyValid = false;
    }

    froxelGrid.value.set(width, height, depth);
    froxelColumns.value = froxelAtlas.columns;
    this.froxelUniforms.froxelSubSteps.value = Math.max(1, Math.round(subSteps));
    this.froxelTileUniforms.frameTileSize.value.set(width, height);
    this.froxelTileUniforms.frameTileColumns.value = froxelAtlas.columns;
    this.froxelTileUniforms.frameTargetSize.value.set(froxelAtlas.width, froxelAtlas.height);
    this.froxelUniforms.froxelAtlasSize.value.set(froxelAtlas.width, froxelAtlas.height);
    this.integrateMaterial.froxelAtlas.value = this.atlasTarget.textures[0];
    this.integrateMaterial.froxelAlphaAtlas.value = this.atlasTarget.textures[spectral ? 1 : 0];

    if (this.settings.temporal) {
      for (let index = 0; index < this.historyTargets.length; index++) {
        const target = this.historyTargets[index];

        if (!target || target.width !== froxelAtlas.width || target.height !== froxelAtlas.height || isSpectralTarget(target) !== spectral) {
          target?.dispose();
          this.historyTargets[index] = createAtlasTarget(froxelAtlas.width, froxelAtlas.height, spectral);
          this.historyValid = false;
        }
      }
    }

    if (this.isFrameLighting()) {
      // NOTE Spectral: 3 targets (albedo, emission, extinction of each channel), no light slot targets
      const densityCount = spectral ? 3 : 4;

      if (!this.densityTarget || this.densityTarget.width !== froxelAtlas.width || this.densityTarget.height !== froxelAtlas.height || this.densityTarget.textures.length !== densityCount) {
        this.densityTarget?.dispose();
        this.densityTarget = createRenderTarget(froxelAtlas.width, froxelAtlas.height, { count: densityCount, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
        this.densityClearMaterial.fragmentNode = outputStruct(...Array.from({ length: densityCount }, () => vec4(0.0)));
        this.densityClearMaterial.needsUpdate = true;
      }

      this.updateGridTargets();
    }
  }

  updateGridTargets(): void {
    const resolution = Math.max(2, Math.round(this.settings.shadowResolution));
    const cascades = this.getCascadeCount();
    const gridAtlas = getAtlasLayout(resolution, resolution, resolution * cascades);
    const spectral = this.isSpectral();

    if (!this.shadowDensityTarget || this.shadowDensityTarget.width !== gridAtlas.width || this.shadowDensityTarget.height !== gridAtlas.height || isSpectralTarget(this.shadowDensityTarget) !== spectral) {
      this.shadowDensityTarget?.dispose();
      this.transmittanceTarget?.dispose();
      this.disposeSpectralTransmittanceTargets();

      // NOTE Extinction and absorption. Spectral: (extinction rgb, 0) and (absorption rgb, 0).
      this.shadowDensityTarget = spectral ? createRenderTarget(gridAtlas.width, gridAtlas.height, { count: 2 }) : createRenderTarget(gridAtlas.width, gridAtlas.height, { format: THREE.RGFormat });
      // NOTE Total transmittance (2 targets), then direct transmittance of the light slots (2 targets).
      // Spectral: the slot group targets (getSpectralTransmittanceTarget).
      this.transmittanceTarget = spectral ? null : createRenderTarget(gridAtlas.width, gridAtlas.height, { count: 4 });

      // NOTE A new target is empty, so the last cascade and the environment grid are drawn again
      this.gridState.castSignature = [];
      this.environmentGridSignature = [];
    }

    // NOTE The per-volume shadow density materials are set in getShadowDensityMaterial
    this.transmittanceMaterial.setSpectral(spectral);
    this.environmentGridMaterial.setSpectral(spectral);

    // NOTE The environment grid is one cascade. A new target is empty, so the signature is reset.
    const environmentAtlas = getAtlasLayout(resolution, resolution, resolution);

    if (!this.environmentGridTarget || this.environmentGridTarget.width !== environmentAtlas.width || this.environmentGridTarget.height !== environmentAtlas.height) {
      this.environmentGridTarget?.dispose();
      this.environmentGridTarget = createRenderTarget(environmentAtlas.width, environmentAtlas.height);
      this.environmentGridSignature = [];
    }

    this.environmentGridUniforms.frameEnvironmentGridColumns.value = environmentAtlas.columns;
    this.environmentGridUniforms.frameEnvironmentGridAtlasSize.value.set(environmentAtlas.width, environmentAtlas.height);

    this.gridUniforms.frameGridResolution.value = resolution;
    this.gridUniforms.frameGridColumns.value = gridAtlas.columns;
    this.gridUniforms.frameGridAtlasSize.value.set(gridAtlas.width, gridAtlas.height);
    this.gridTileUniforms.frameTileSize.value.set(resolution, resolution);
    this.gridTileUniforms.frameTileColumns.value = gridAtlas.columns;
    this.gridTileUniforms.frameTargetSize.value.set(gridAtlas.width, gridAtlas.height);
  }

  renderFrame(renderer: RendererLike, scene: SceneLike, camera: THREE.Camera, volumes: FrameVolumeCollector<VDBFrameVolume>["volumes"], casters: FrameVolumeCollector<VDBFrameVolume>["casters"] = []): void {
    this.checkSettings();

    const currentRenderTarget = renderer.getRenderTarget();
    const currentActiveCubeFace = renderer.getActiveCubeFace();
    const currentActiveMipmapLevel = renderer.getActiveMipmapLevel();
    const currentAutoClear = renderer.autoClear;
    const currentClearAlpha = renderer.getClearAlpha();
    const currentScissorTest = renderer.getScissorTest();

    renderer.getClearColor(clearColor);

    // NOTE null: the environment uses the 'outward' path. 'volume' lighting marches every environment light in each volume.
    this.environment = this.environmentLights.update(renderer, scene as never, this.settings);

    this.updateTargets(renderer);
    this.updateFroxelArea(volumes, camera);

    const cam = camera as THREE.PerspectiveCamera & { isOrthographicCamera?: boolean };
    this.froxelUniforms.froxelProjectionInverse.value.copy(camera.projectionMatrixInverse);
    this.froxelUniforms.frameFadeRadius.value = getFrameFadeRadius(this.settings);
    this.integrateMaterial.frameCameraNear.value = cam.near;
    this.integrateMaterial.frameCameraFar.value = cam.far;
    this.integrateMaterial.frameCameraOrthographic.value = cam.isOrthographicCamera ? 1 : 0;
    this.integrateMaterial.setSrgbBlending(this.settings.srgbBlending && renderer.outputColorSpace === THREE.SRGBColorSpace, renderer.toneMapping === THREE.NoToneMapping);

    // NOTE Pass materials are unlit. Nested renders do not update shadow maps.
    renderer.autoClear = false;
    renderer.setScissorTest(false);
    renderer.setClearColor(0x000000, 0.0);

    // NOTE Golden ratio steps give a new, evenly spread jitter value for each frame
    this.froxelUniforms.frameJitter.value = this.settings.jitter ? (this.jitterIndex * 0.618034) % 1.0 : -1.0;
    this.jitterIndex = (this.jitterIndex + 1) % 1024;

    this.passHook?.begin?.("froxels");

    if (this.isFrameLighting()) {
      this.renderFrameLighting(renderer, scene, camera, volumes, casters);
    } else {
      this.renderInject(renderer, scene, camera, volumes);
    }

    if (this.settings.temporal) {
      this.passHook?.begin?.("temporal");
      this.renderTemporal(renderer, camera);
      this.passHook?.end?.("temporal");
    } else {
      this.historyValid = false;
    }

    this.passHook?.end?.("froxels");

    this.gridFreshCamera = null;

    renderer.setScissorTest(currentScissorTest);
    renderer.autoClear = currentAutoClear;
    renderer.setClearColor(clearColor, currentClearAlpha);
    renderer.setRenderTarget(currentRenderTarget, currentActiveCubeFace, currentActiveMipmapLevel);
  }

  // NOTE Froxel area of the volumes, for the passes over the froxel grid and for the integrate pass
  updateFroxelArea(volumes: FrameVolumeCollector<VDBFrameVolume>["volumes"], camera: THREE.Camera): void {
    const { froxelGrid, froxelRange } = this.froxelUniforms;
    const { x: width, y: height, z: depth } = froxelGrid.value;
    const area = getFroxelArea(volumes, camera, width, height, froxelRange.value, depth, this.froxelArea);
    const { frameAreaRect, frameAreaSliceStart } = this.froxelAreaUniforms;

    frameAreaRect.value.copy(area.rect);
    frameAreaSliceStart.value = area.first;
    this.areaGeometry.instanceCount = area.count;

    // NOTE One more cell on each side: the linear filter of the integrate pass reads one cell away
    const { x0, y0, x1, y1 } = getClipRect(area.rect, width, height, area.count > 0 ? 1 : 0);

    this.integrateMaterial.frameScreenRect.value.set(x0, y0, x1, y1);
    this.integrateMaterial.frameSliceRange.value.set(area.first, area.first + area.count);
  }

  // NOTE Blends this frame into the history. The integrate pass then reads the blended froxels.
  renderTemporal(renderer: RendererLike, camera: THREE.Camera): void {
    const material = this.temporalMaterial;
    const readTarget = this.historyTargets[this.historyIndex]!;
    const writeTarget = this.historyTargets[1 - this.historyIndex]!;
    const cam = camera as THREE.PerspectiveCamera;

    material.frameCurrentMap.value = this.atlasTarget!.textures[0];
    material.frameHistoryMap.value = readTarget.textures[0];
    material.frameCurrentAlphaMap.value = this.atlasTarget!.textures.at(-1);
    material.frameHistoryAlphaMap.value = readTarget.textures.at(-1);
    material.frameUseHistory.value = this.historyValid && this.historyCamera === camera ? 1 : 0;
    material.frameTemporalBlend.value = Math.min(1.0, Math.max(0.01, this.settings.temporalBlend));
    material.frameCameraMatrixWorld.value.copy(camera.matrixWorld);
    material.framePreviousViewMatrix.value.copy(this.previousViewMatrix);
    material.framePreviousProjectionMatrix.value.copy(this.previousProjectionMatrix);
    material.framePreviousRange.value.copy(this.previousRange);

    this.temporalMesh.layers.mask = camera.layers.mask;

    // NOTE The pass draws only the froxel area. The clear empties the other froxels.
    renderer.setRenderTarget(writeTarget);
    renderer.clear(true, false, false);
    renderer.render(this.temporalScene, camera);

    this.integrateMaterial.froxelAtlas.value = writeTarget.textures[0];
    this.integrateMaterial.froxelAlphaAtlas.value = writeTarget.textures.at(-1);

    this.historyIndex = 1 - this.historyIndex;
    this.historyValid = true;
    this.historyCamera = camera;
    this.previousViewMatrix.copy(camera.matrixWorldInverse);
    this.previousProjectionMatrix.copy(cam.projectionMatrix);
    this.previousRange.copy(this.froxelUniforms.froxelRange.value);
  }

  // NOTE PMREM rotation and intensity uniforms read the environment of the rendered pass scene
  copyEnvironment(target: THREE.Scene, scene: SceneLike): void {
    target.environment = scene.environment ?? null;
    target.environmentIntensity = scene.environmentIntensity ?? 1.0;
    if (scene.environmentRotation) target.environmentRotation.copy(scene.environmentRotation);
  }

  prepareLightScene(scene: SceneLike, camera: THREE.Camera, mesh: THREE.Mesh): LightResult {
    this.copyEnvironment(this.injectScene, scene);

    this.injectMesh.visible = mesh === this.injectMesh;
    this.lightingMesh.visible = mesh === this.lightingMesh;
    mesh.layers.mask = camera.layers.mask;

    return this.updateLights(scene, camera);
  }

  renderInject(renderer: RendererLike, scene: SceneLike, camera: THREE.Camera, volumes: FrameVolumeCollector<VDBFrameVolume>["volumes"]): void {
    const { atlasTarget, injectScene, injectMesh } = this;

    this.prepareLightScene(scene, camera, this.injectMesh);

    renderer.setRenderTarget(atlasTarget);
    renderer.clear(true, false, false);

    for (let i = 0; i < volumes.length; i++) {
      const { frameVolume, volume } = volumes[i]!;
      const injectMaterial = this.getInjectMaterial(frameVolume);

      // NOTE Draw only the froxel rectangle and the slices of this volume. One instance for each slice.
      if (!this.getFroxelRect(volume, camera, injectMaterial.frameRect.value)) {
        continue;
      }

      const { first, count } = this.getSliceRange(volume);

      injectMaterial.frameSliceStart.value = first;
      injectMesh.geometry.instanceCount = count;
      injectMesh.material = injectMaterial;

      injectMaterial.frameCameraMatrixWorld.value.copy(camera.matrixWorld);
      injectMaterial.setEnvironment(scene.environment);
      setVolumeUniforms(injectMaterial, frameVolume, volume);
      injectMaterial.frameLights.value = frameVolume.lights;
      injectMaterial.frameShadowSteps.value = frameVolume.shadowSteps;
      injectMaterial.frameMultipleScattering.value = getFrameMultipleScattering(this.multipleScatteringMode(), frameVolume);
      getFrameScatteringOctaves(this.settings, injectMaterial.frameScatteringOctaves.value);
      injectMaterial.frameTwoSided.value = getFrameTwoSidedDiffusion(this.settings.multipleScattering);

      this.passHook?.begin?.("inject", { volume: frameVolume });
      renderer.setRenderTarget(atlasTarget);
      renderer.render(injectScene, camera);
      this.passHook?.end?.("inject", { volume: frameVolume });
    }
  }

  /** The inject material of a volume: its source is compiled in (DIVERGES: one material per source). */
  getInjectMaterial(frameVolume: VDBFrameVolume): VDBFrameInjectMaterial {
    const mipmaps = this.volumeMipmaps(frameVolume);
    let material = frameVolume.injectMaterial;

    if (!material || material.froxelUniforms !== this.froxelUniforms) {
      material?.dispose();
      material = new VDBFrameInjectMaterial({
        froxelUniforms: this.froxelUniforms,
        froxelTileUniforms: this.froxelTileUniforms,
        lightUniforms: this.lightUniforms,
        source: frameVolume.source,
        sourceUniforms: frameVolume.sourceUniforms,
        mipmaps,
        hdda: frameVolume.hdda,
        spectral: this.isSpectral(),
        ...(frameVolume.emissionColor ? { emissionColor: frameVolume.emissionColor } : {}),
      });
      frameVolume.injectMaterial = material;
    }

    material.setMipmaps(mipmaps);
    material.setHdda(frameVolume.hdda);
    material.setSpectral(this.isSpectral());
    // NOTE The hemisphere and environment-light loops are compiled in only when the scene has those lights (updateLights)
    material.setLightTerms((this.lightUniforms.frameHemiCount.value as number) > 0, (this.lightUniforms.frameEnvironmentCount.value as number) > 0);

    return material;
  }

  private volumeMipmaps(frameVolume: VDBFrameVolume): boolean {
    return frameVolume.mipmaps && frameVolume.source.lodLevels > 0;
  }

  /** The 'frame' density material of a volume (source compiled in, as the inject material). */
  getDensityMaterial(frameVolume: VDBFrameVolume): VDBFrameDensityMaterial {
    const mipmaps = this.volumeMipmaps(frameVolume);
    let material = frameVolume.densityMaterial;

    if (!material || material.frameUniforms.froxelUniforms !== this.froxelUniforms) {
      material?.dispose();
      material = new VDBFrameDensityMaterial({
        froxelUniforms: this.froxelUniforms,
        froxelTileUniforms: this.froxelTileUniforms,
        gridUniforms: this.gridUniforms,
        environmentGridUniforms: this.environmentGridUniforms,
        densityShadowUniforms: this.densityShadowUniforms,
        lightUniforms: this.lightUniforms,
        source: frameVolume.source,
        sourceUniforms: frameVolume.sourceUniforms,
        mipmaps,
        ...(frameVolume.emissionColor ? { emissionColor: frameVolume.emissionColor } : {}),
      });
      frameVolume.densityMaterial = material;
    }

    material.setMipmaps(mipmaps);
    material.setSpectral(this.isSpectral());

    return material;
  }

  /** The shadow density material of a caster (source compiled in). */
  getShadowDensityMaterial(frameVolume: VDBFrameVolume): VDBFrameShadowDensityMaterial {
    const mipmaps = this.volumeMipmaps(frameVolume);
    let material = frameVolume.shadowDensityMaterial;

    if (!material || material.gridUniforms !== this.gridUniforms) {
      material?.dispose();
      material = new VDBFrameShadowDensityMaterial({
        gridUniforms: this.gridUniforms,
        gridTileUniforms: this.gridTileUniforms,
        source: frameVolume.source,
        sourceUniforms: frameVolume.sourceUniforms,
        mipmaps,
      });
      frameVolume.shadowDensityMaterial = material;
    }

    material.setMipmaps(mipmaps);
    material.setSpectral(this.isSpectral());

    return material;
  }

  // NOTE Combined lighting: density of all volumes, one shadow grid, one lighting pass.
  // Lighting and transmittance render calls do not grow with the number of volumes.
  renderFrameLighting(renderer: RendererLike, scene: SceneLike, camera: THREE.Camera, volumes: FrameVolumeCollector<VDBFrameVolume>["volumes"], casters: FrameVolumeCollector<VDBFrameVolume>["casters"]): void {
    const lights = this.prepareLightScene(scene, camera, this.lightingMesh);
    const ms = getFrameMultipleScattering(this.multipleScatteringMode(), this.settings);
    const octaves = getFrameScatteringOctaves(this.settings, this.scatteringOctaves);

    // NOTE Shadow pass. Extinction of all casters in one world grid, then transmittance to each light.
    // It is before the density pass. Then the density pass can sample the shadows of each volume.
    this.passHook?.begin?.("shadowDensity");
    const useShadows = this.gridFreshCamera === camera ? this.gridReady : this.renderShadowDensity(renderer, camera, casters);
    this.passHook?.end?.("shadowDensity");

    this.gridFreshCamera = null;
    const useEnvironmentGrid = useShadows && lights.environmentGridLights.length > 0;
    const spectral = this.isSpectral();
    const spectralGroups = getFrameSpectralGroups(lights.shadowLightCount);

    if (useShadows) {
      const material = this.transmittanceMaterial;

      material.frameShadowDensityMap.value = this.shadowDensityTarget!.textures[0];
      material.frameShadowAbsorptionMap.value = this.shadowDensityTarget!.textures.at(-1);
      material.frameShadowLightCount.value = lights.shadowLightCount;
      material.frameMultipleScattering.value = ms;
      material.frameScatteringOctaves.value.copy(octaves);
      material.frameTwoSided.value = getFrameTwoSidedDiffusion(this.settings.multipleScattering);
      this.transmittanceMesh.layers.mask = camera.layers.mask;
      material.setNeural(this.isNeural() ? this.neuralNetwork : null);

      if (material.neural) {
        this.passHook?.begin?.("gridMips");
        this.renderGridMips(renderer, camera);
        this.passHook?.end?.("gridMips");
      }

      if (this.transmittanceUnchanged(lights.shadowLightCount, spectral ? spectralGroups : 0)) {
        this.redrawStats.transmittance.skipped++;
      } else if (spectral) {
        this.redrawStats.transmittance.drawn++;
        this.passHook?.begin?.("transmittance");

        // NOTE One pass for each slot group in use, each into its own targets
        for (let group = 0; group < spectralGroups; group++) {
          material.frameSpectralGroup.value = group;

          renderer.setRenderTarget(this.getSpectralTransmittanceTarget(group));
          renderer.render(this.transmittanceScene, camera);
        }

        this.passHook?.end?.("transmittance");
      } else {
        this.redrawStats.transmittance.drawn++;
        this.passHook?.begin?.("transmittance");
        renderer.setRenderTarget(this.transmittanceTarget);
        renderer.render(this.transmittanceScene, camera);
        this.passHook?.end?.("transmittance");
      }
    }

    if (useEnvironmentGrid) {
      this.passHook?.begin?.("environmentGrid");
      this.renderEnvironmentGrid(renderer, camera, casters, lights.environmentGridLights, ms, octaves);
      this.passHook?.end?.("environmentGrid");
    }

    const { densityShadowUniforms } = this;

    densityShadowUniforms.frameUseShadows.value = useShadows ? 1 : 0;

    if (!spectral) {
      const transmittance = this.transmittanceTarget!;

      densityShadowUniforms.frameTransmittanceMap0.value = transmittance.textures[0];
      densityShadowUniforms.frameTransmittanceMap1.value = transmittance.textures[1];
      densityShadowUniforms.frameTransmittanceMap2.value = transmittance.textures[2];
      densityShadowUniforms.frameTransmittanceMap3.value = transmittance.textures[3];
    }

    densityShadowUniforms.frameShadowSceneLightCount.value = lights.environmentShadowSlot;
    getFramePhase(this.settings, densityShadowUniforms.framePhase.value);
    getFramePhaseSpectrum(this.settings, densityShadowUniforms.framePhaseSpectrum.value);
    densityShadowUniforms.frameScatteringOctaves.value.copy(octaves);
    densityShadowUniforms.frameDiffusion.value = isFrameDiffusion(this.multipleScatteringMode()) ? 1 : 0;
    densityShadowUniforms.frameEnvironmentGridMap.value = this.environmentGridTarget!.texture;
    densityShadowUniforms.frameEnvironmentMode.value = useEnvironmentGrid
      ? frameDensityEnvironment.grid
      : useShadows && lights.environmentLightCount > 0
        ? frameDensityEnvironment.none
        : frameDensityEnvironment.outward;
    densityShadowUniforms.frameCameraMatrixWorld.value.copy(camera.matrixWorld);

    // NOTE Spectral: sky up and sky down are in the first two targets of group 0 (bound also without shadows: not his)
    if (spectral) {
      const skyMaps = densityShadowUniforms.frameSkyTransmittanceMaps;

      for (let index = 0; index < skyMaps.length; index++) {
        skyMaps[index]!.value = this.getSpectralTransmittanceTarget(0).textures[index];
      }
    }

    this.copyEnvironment(this.densityScene, scene);

    // NOTE Density pass. One instanced draw for each volume, one instance for each slice.
    this.updatePool(this.densityPool, this.densityScene, volumes, camera, this.setupDensityEntry);

    this.densityClearMesh.layers.mask = camera.layers.mask;

    this.passHook?.begin?.("density");
    renderer.setRenderTarget(this.densityTarget);
    renderer.render(this.densityClearScene, camera);
    renderer.render(this.densityScene, camera);
    this.passHook?.end?.("density");

    // NOTE Lighting pass. Each froxel is lit one time from the combined data.
    const { lightingMaterial } = this;
    const density = this.densityTarget!;

    lightingMaterial.frameDensityAlbedoMap.value = density.textures[0];
    lightingMaterial.frameDensityEmissionMap.value = density.textures[1];

    if (spectral) {
      lightingMaterial.frameDensityExtinctionMap.value = density.textures[2];
      lightingMaterial.frameSpectralGroups.value = useShadows ? spectralGroups : 0;

      // NOTE Groups not in use keep the textures they have (their own targets or the initial placeholders); the lighting
      // pass does not read them. Not his (he gives them group 0's targets): three's TextureNode uniform hash is its
      // texture's uuid, so nodes that hold one texture when the shader is built share one binding, and a group that
      // came into use later (an environment light taking a slot after its readback) kept reading group 0.
      if (useShadows) {
        const maps = lightingMaterial.frameSpectralTransmittanceMaps;

        for (let index = 0; index < spectralGroups * 3; index++) {
          maps[index]!.value = this.getSpectralTransmittanceTarget(Math.floor(index / 3)).textures[index % 3];
        }
      }
    } else {
      lightingMaterial.frameDensityShadowMap0.value = density.textures[2];
      lightingMaterial.frameDensityShadowMap1.value = density.textures[3];
    }
    lightingMaterial.frameShadowLightCount.value = lights.shadowLightCount;
    lightingMaterial.frameEnvironmentLightCount.value = useShadows ? lights.environmentLightCount : 0;
    lightingMaterial.frameEnvironmentShadowSlot.value = lights.environmentShadowSlot;
    lightingMaterial.frameCameraMatrixWorld.value.copy(camera.matrixWorld);
    getFramePhase(this.settings, lightingMaterial.framePhase.value);
    getFramePhaseSpectrum(this.settings, lightingMaterial.framePhaseSpectrum.value);
    lightingMaterial.frameScatteringOctaves.value.copy(octaves);

    this.passHook?.begin?.("lighting");
    renderer.setRenderTarget(this.atlasTarget);
    renderer.clear(true, false, false);
    renderer.render(this.injectScene, camera);
    this.passHook?.end?.("lighting");
  }

  // NOTE Whether the transmittance target already holds what this render would draw: the same grid content, target,
  // shadow lights and settings. A jittered grid changes in each frame, so it is always drawn.
  transmittanceUnchanged(shadowLightCount: number, spectralGroups = 0): boolean {
    const material = this.transmittanceMaterial;
    const values = this.nextTransmittanceSignature;

    values.length = 0;
    // NOTE Spectral rendering has no single transmittance target, only the targets of its groups below
    values.push(this.gridState.gridVersion, this.shadowDensityTarget!.texture.id, this.transmittanceTarget ? this.transmittanceTarget.texture.id : -1, spectralGroups, shadowLightCount, material.frameMultipleScattering.value as number);
    // NOTE The network: on/off and which weights (a swapped network draws again)
    values.push(material.neural ? networkId(material.neural) : 0, material.frameTwoSided.value as number);
    vdbSignaturePushVector(values, material.frameScatteringOctaves.value);
    vdbSignaturePushVector(values, this.lightUniforms.frameSkyDirection.value);

    for (let group = 0; group < spectralGroups; group++) {
      values.push(this.getSpectralTransmittanceTarget(group).texture.id);
    }

    const shadowLights = material.frameShadowLights.array as THREE.Vector4[];

    for (let i = 0; i < shadowLightCount; i++) {
      vdbSignaturePushVector(values, shadowLights[i]!);
    }

    const jittered = (this.froxelUniforms.frameJitter.value as number) >= 0.0;
    const same = !jittered && vdbSameValues(values, this.transmittanceSignature);

    // NOTE A jittered draw matches no signature, so the first frame after jitter is turned off draws again
    if (jittered) {
      values.length = 0;
    }

    this.nextTransmittanceSignature = this.transmittanceSignature;
    this.transmittanceSignature = values;

    return same;
  }

  // NOTE Extinction of all casters in the world shadow grid cascades. Returns false with no casters.
  // NOTE onlyLast: only the last cascade (all volumes), drawn only when castVersion changed. Volume shadows on meshes
  // read only the last cascade; 'frame' lighting needs all cascades.
  renderShadowDensity(renderer: RendererLike, camera: THREE.Camera, casters: FrameVolumeCollector<VDBFrameVolume>["casters"], onlyLast = false): boolean {
    const ready = casters.length > 0 && this.updateGrid(casters, camera);
    const castChanged = this.updateGridVersion(ready ? casters : []);

    if (!ready) {
      return false;
    }

    if (onlyLast && !castChanged) {
      this.redrawStats.shadowDensity.skipped++;

      return true;
    }

    if (!onlyLast && this.shadowDrawnVersion === this.gridState.gridVersion && this.shadowDrawnTarget === this.shadowDensityTarget) {
      this.redrawStats.shadowDensity.skipped++;

      return true;
    }

    this.redrawStats.shadowDensity.drawn++;
    // NOTE The clear below empties every cascade, so a draw of the last one alone leaves no full grid to keep
    this.shadowDrawnVersion = onlyLast ? -1 : this.gridState.gridVersion;
    this.shadowDrawnTarget = this.shadowDensityTarget;

    const count = this.gridUniforms.frameGridCount.value as number;
    const cascadeCasters = vdbFrameCascadeCasters(casters, onlyLast ? count - 1 : 0, count, this.cascadeCasterPool);

    this.updatePool(this.shadowPool, this.shadowScene, cascadeCasters, camera, this.setupShadowEntry);

    renderer.setRenderTarget(this.shadowDensityTarget);
    renderer.clear(true, false, false);
    renderer.render(this.shadowScene, camera);

    return true;
  }

  /**
   * Makes the world shadow grid one time in a render, before the first lit mesh (the volume shadow maps need it).
   * 'frame' lighting uses it again in the same render. Returns false with no casters. `renderId`: three's node frame
   * render id (three-internals.ts).
   */
  prepareShadowGrid(renderer: RendererLike, scene: SceneLike, camera: THREE.Camera, renderId: number): boolean {
    if (this.gridRenderId === renderId && this.gridCamera === camera) {
      return this.gridReady;
    }

    this.gridRenderId = renderId;
    this.gridCamera = camera;

    const { casters } = this.collectVolumes(renderer, scene, camera);

    const currentRenderTarget = renderer.getRenderTarget();
    const currentActiveCubeFace = renderer.getActiveCubeFace();
    const currentActiveMipmapLevel = renderer.getActiveMipmapLevel();
    const currentAutoClear = renderer.autoClear;
    const currentClearAlpha = renderer.getClearAlpha();
    const currentScissorTest = renderer.getScissorTest();

    renderer.getClearColor(clearColor);
    renderer.autoClear = false;
    renderer.setScissorTest(false);
    renderer.setClearColor(0x000000, 0.0);

    this.updateGridTargets();
    this.passHook?.begin?.("shadowDensity");
    // NOTE 'volume' lighting: only the last cascade, for the volume shadow maps
    this.gridReady = this.renderShadowDensity(renderer, camera, casters, !this.isFrameLighting());
    this.passHook?.end?.("shadowDensity");
    this.gridFreshCamera = camera;

    renderer.setScissorTest(currentScissorTest);
    renderer.autoClear = currentAutoClear;
    renderer.setClearColor(clearColor, currentClearAlpha);
    renderer.setRenderTarget(currentRenderTarget, currentActiveCubeFace, currentActiveMipmapLevel);

    return this.gridReady;
  }

  // NOTE gridVersion changes when the cascades, a caster transform, a caster's data or a caster's shadow props change.
  // castVersion changes only for the last cascade. Returns true when castVersion changed.
  updateGridVersion(casters: FrameVolumeCollector<VDBFrameVolume>["casters"]): boolean {
    const { frameGridMin, frameGridSize, frameGridCount, frameGridResolution } = this.gridUniforms;

    return vdbFrameUpdateGridSignatures(
      this.gridState,
      casters,
      frameGridCount.value as number,
      frameGridResolution.value as number,
      this.isSpectral(),
      frameGridMin.array as THREE.Vector3[],
      frameGridSize.array as THREE.Vector3[],
      (values, { frameVolume, volume }) => {
        vdbSignaturePushElements(values, volume.matrixWorld.elements);
        pushCasterSource(values, frameVolume);
        values.push(frameVolume.baseColor.r, frameVolume.baseColor.g, frameVolume.baseColor.b);
        values.push(frameVolume.getDensityScale(), frameVolume.densityCutoff, frameVolume.extinction);
        vdbSignaturePushVector(values, getFrameExtinctionSpectrum(frameVolume, extinctionSpectrum));
        values.push(frameVolume.absorbance, frameVolume.mipmaps ? 1 : 0, frameVolume.wrap3D);
        vdbSignaturePushVector(values, frameVolume.offset3D);
      },
    );
  }

  // NOTE Environment grid of the last cascade. It is drawn again only when its signature changes.
  // NOTE Signature: the grid lights, the last cascade, and the caster data that the shadow density pass reads.
  renderEnvironmentGrid(renderer: RendererLike, camera: THREE.Camera, casters: FrameVolumeCollector<VDBFrameVolume>["casters"], gridLights: LightResult["environmentGridLights"], ms: number, octaves: THREE.Vector4): void {
    const { frameGridMin, frameGridSize, frameGridCount, frameGridResolution } = this.gridUniforms;
    const cascade = (frameGridCount.value as number) - 1;
    const previous = this.environmentGridSignature;
    const signature = this.nextEnvironmentGridSignature;

    this.environmentGridUniforms.frameEnvironmentGridCascade.value = cascade;

    signature.length = 0;
    signature.push(frameGridResolution.value as number, cascade, this.isSpectral() ? 1 : 0, ms);
    vdbSignaturePushVector(signature, octaves);
    vdbSignaturePushVector(signature, (frameGridMin.array as THREE.Vector3[])[cascade]!);
    vdbSignaturePushVector(signature, (frameGridSize.array as THREE.Vector3[])[cascade]!);

    for (const light of gridLights) {
      vdbSignaturePushElements(signature, light.direction);
      vdbSignaturePushElements(signature, light.color);
    }

    // NOTE The caster data the shadow density pass reads: the grid version covers it (transforms, sources, props)
    signature.push(this.gridState.castVersion, casters.length);

    if (vdbSameValues(signature, previous)) {
      this.redrawStats.environmentGrid.skipped++;

      return;
    }

    this.redrawStats.environmentGrid.drawn++;
    this.environmentGridSignature = signature;
    this.nextEnvironmentGridSignature = previous;

    const material = this.environmentGridMaterial;

    material.frameShadowDensityMap.value = this.shadowDensityTarget!.textures[0];
    material.frameShadowAbsorptionMap.value = this.shadowDensityTarget!.textures.at(-1);
    material.frameLightCount.value = gridLights.length;
    material.frameMultipleScattering.value = ms;
    material.frameScatteringOctaves.value.copy(octaves);

    const directions = material.frameLightDirections.array as THREE.Vector3[];
    const colors = material.frameLightColors.array as THREE.Vector3[];

    for (let i = 0; i < gridLights.length; i++) {
      directions[i]!.fromArray(gridLights[i]!.direction);
      colors[i]!.fromArray(gridLights[i]!.color);
    }

    this.environmentGridMesh.layers.mask = camera.layers.mask;

    renderer.setRenderTarget(this.environmentGridTarget);
    renderer.render(this.environmentGridScene, camera);
  }

  // NOTE Draws the lat-long environment map with the rotation of the scene and intensity 1.0, then reads it.
  renderEnvironmentMap(renderer: RendererLike, scene: SceneLike): Promise<ArrayLike<number>> {
    if (!this.environmentMapTarget) {
      this.environmentMapTarget = createRenderTarget(frameEnvironmentMapWidth, frameEnvironmentMapHeight, {
        type: THREE.FloatType,
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
      });
    }

    this.copyEnvironment(this.environmentMapScene, scene);
    this.environmentMapScene.environmentIntensity = 1.0;
    this.environmentMapMaterial.setEnvironment(scene.environment);

    const currentRenderTarget = renderer.getRenderTarget();

    renderer.setRenderTarget(this.environmentMapTarget);
    renderer.render(this.environmentMapScene, this.environmentMapCamera);
    renderer.setRenderTarget(currentRenderTarget);

    return renderer.readRenderTargetPixelsAsync(this.environmentMapTarget, 0, 0, frameEnvironmentMapWidth, frameEnvironmentMapHeight) as unknown as Promise<ArrayLike<number>>;
  }

  // NOTE updatePool setup of the density pass: one instance for each slice of the volume
  readonly setupDensityEntry = (entry: PoolEntry, { frameVolume, volume }: FrameVolumeEntry<VDBFrameVolume>, camera: THREE.Camera): boolean => {
    if (!this.getFroxelRect(volume, camera, entry.rect)) {
      return false;
    }

    const { first, count } = this.getSliceRange(volume);

    entry.mesh.material = this.getDensityMaterial(frameVolume);
    (entry.mesh.material as VDBFrameDensityMaterial).setEnvironment(this.densityScene.environment);
    entry.frameVolume = frameVolume;
    entry.volume = volume;
    entry.sliceStart = first;
    entry.geometry.instanceCount = count;

    return true;
  };

  // NOTE updatePool setup of the shadow density pass: the voxels of the volume in one cascade
  readonly setupShadowEntry = (entry: PoolEntry, { caster, cascade }: { caster: FrameVolumeEntry<VDBFrameVolume>; cascade: number }): boolean => {
    const { frameVolume, volume } = caster;
    const range = this.getGridRange(volume, cascade);

    if (!range) {
      return false;
    }

    entry.mesh.material = this.getShadowDensityMaterial(frameVolume);
    entry.frameVolume = frameVolume;
    entry.volume = volume;
    entry.rect.copy(range.rect);
    entry.sliceStart = range.first;
    entry.geometry.instanceCount = range.count;

    return true;
  };

  // NOTE Pool entries are reused between frames. Unused entries are hidden.
  // Each entry takes the material of its volume and sets the volume uniforms before its draw.
  updatePool<T>(pool: PoolEntry[], scene: THREE.Scene, items: readonly T[], camera: THREE.Camera, setup: (entry: PoolEntry, item: T, camera: THREE.Camera) => boolean): void {
    let used = 0;

    for (let i = 0; i < items.length; i++) {
      if (!pool[used]) {
        const geometry = new THREE.InstancedBufferGeometry();
        geometry.setAttribute("position", tileQuadPosition);

        const mesh = new THREE.Mesh<THREE.InstancedBufferGeometry, THREE.Material>(geometry, new THREE.MeshBasicMaterial());
        mesh.frustumCulled = false;
        mesh.matrixAutoUpdate = false;

        const entry: PoolEntry = { mesh, geometry, frameVolume: null, volume: null, rect: new THREE.Vector4(), sliceStart: 0 };

        mesh.onBeforeRender = () => {
          const material = mesh.material as PassMaterial;

          setVolumeUniforms(material, entry.frameVolume!, entry.volume!);
          material.frameRect.value.copy(entry.rect);
          material.frameSliceStart.value = entry.sliceStart;
        };

        scene.add(mesh);
        pool[used] = entry;
      }

      const entry = pool[used]!;

      if (setup(entry, items[i]!, camera)) {
        entry.mesh.visible = true;
        entry.mesh.layers.mask = camera.layers.mask;
        used++;
      }
    }

    for (let i = used; i < pool.length; i++) {
      pool[i]!.mesh.visible = false;
    }
  }

  // NOTE Screen rectangle of the volume in froxel cells. Writes target. Returns false when it covers no cell.
  getFroxelRect(volume: FrameVolumeRecord, camera: THREE.Camera, target: THREE.Vector4): boolean {
    const { x: width, y: height } = this.froxelUniforms.froxelGrid.value;

    return getFroxelRect(volume, camera, width, height, target);
  }

  getCascadeCount(): number {
    return Math.min(frameMaxShadowCascades, Math.max(1, Math.round(this.settings.shadowCascades)));
  }

  // NOTE Shadow grid cascades of the casters (tools/frame/grid.ts). Returns false with no casters.
  updateGrid(casters: FrameVolumeCollector<VDBFrameVolume>["casters"], camera: THREE.Camera): boolean {
    const { frameGridMin, frameGridSize, frameGridVoxel, frameGridCount, frameGridResolution } = this.gridUniforms;

    cameraPosition.setFromMatrixPosition(camera.matrixWorld);

    const count = vdbFrameGridCascades(
      casters.map((c) => c.volume.worldBox),
      cameraPosition,
      this.settings.shadowDistance,
      frameGridResolution.value as number,
      this.getCascadeCount(),
      frameGridMin.array as THREE.Vector3[],
      frameGridSize.array as THREE.Vector3[],
      frameGridVoxel.array as THREE.Vector3[],
    );

    if (count > 0) {
      frameGridCount.value = count;
    }

    return count > 0;
  }

  // NOTE Voxel rectangle and atlas slices of the volume in one shadow grid cascade
  getGridRange(volume: FrameVolumeRecord, cascade: number): { rect: THREE.Vector4; first: number; count: number } | null {
    const { frameGridMin, frameGridVoxel, frameGridResolution } = this.gridUniforms;
    const found = vdbFrameGridRange(volume.worldBox, cascade, frameGridResolution.value as number, frameGridMin.array as THREE.Vector3[], frameGridVoxel.array as THREE.Vector3[], this.gridRange);

    return found ? this.gridRange : null;
  }

  // NOTE A pooled light record of updateLights: (position, w type), (color, w spot cos), (spot direction, cos), falloff
  getLightRecord(list: LightList): LightRecord {
    if (list.length === list.entries.length) {
      list.entries.push({ position: new THREE.Vector4(), color: new THREE.Vector4(), spot: new THREE.Vector4(), falloff: new THREE.Vector2() });
    }

    const record = list.entries[list.length++]!;

    record.spot.set(0.0, 0.0, 0.0, 0.0);
    record.falloff.set(0.0, 0.0);

    return record;
  }

  // NOTE Light i of updateLights in slot order: directional, point, spot
  getSortedLight(index: number): LightRecord {
    const { directions, points, spots } = this.lightState;

    if (index < directions.length) {
      return directions.entries[index]!;
    }

    if (index < directions.length + points.length) {
      return points.entries[index - directions.length]!;
    }

    return spots.entries[index - directions.length - points.length]!;
  }

  // NOTE traverseVisible callback of updateLights. this.lightState holds the state of the current call.
  readonly visitLight = (object: THREE.Object3D): void => {
    const state = this.lightState;
    const light = object as THREE.Light & {
      isAmbientLight?: boolean;
      isHemisphereLight?: boolean;
      isLightProbe?: boolean;
      isDirectionalLight?: boolean;
      isSpotLight?: boolean;
      isPointLight?: boolean;
      groundColor?: THREE.Color;
      sh?: { coefficients: THREE.Vector3[] };
      target?: THREE.Object3D;
      angle?: number;
      penumbra?: number;
      distance?: number;
      decay?: number;
    };

    if (!light.isLight || !light.layers.test(state.camera!.layers)) {
      return;
    }

    const { frameAmbientColor, frameLightProbe, frameHemiSkyColor, frameHemiGroundColor, frameSkyDirection, frameHemiSkyColors, frameHemiGroundColors, frameHemiDirections } = this.lightUniforms;

    lightColor.copy(light.color).multiplyScalar(light.intensity);
    lightPosition.setFromMatrixPosition(light.matrixWorld);

    if (light.isAmbientLight) {
      frameAmbientColor.value.add(lightColor);
    } else if (light.isHemisphereLight) {
      const hemiCount = state.hemiCount;

      if (hemiCount < frameMaxHemisphereLights) {
        (frameHemiSkyColors.array as THREE.Vector3[])[hemiCount]!.set(lightColor.r, lightColor.g, lightColor.b);
        (frameHemiDirections.array as THREE.Vector3[])[hemiCount]!.copy(lightPosition).normalize();
      }

      frameHemiSkyColor.value.add(lightColor);
      lightColor.copy(light.groundColor!).multiplyScalar(light.intensity);
      frameHemiGroundColor.value.add(lightColor);

      if (hemiCount < frameMaxHemisphereLights) {
        (frameHemiGroundColors.array as THREE.Vector3[])[hemiCount]!.set(lightColor.r, lightColor.g, lightColor.b);
      }

      state.hemiCount++;

      if (!state.hasSky && lightPosition.lengthSq() > 0.0) {
        frameSkyDirection.value.copy(lightPosition).normalize();
        state.hasSky = true;
      }
    } else if (light.isLightProbe) {
      const { coefficients } = light.sh!;

      for (let i = 0; i < coefficients.length; i++) {
        (frameLightProbe.array as THREE.Vector3[])[i]!.addScaledVector(coefficients[i]!, light.intensity);
      }
    } else if (light.isDirectionalLight) {
      const record = this.getLightRecord(state.directions);

      lightTarget.setFromMatrixPosition(light.target!.matrixWorld);
      lightPosition.sub(lightTarget).normalize();
      record.position.set(lightPosition.x, lightPosition.y, lightPosition.z, 1.0);
      record.color.set(lightColor.r, lightColor.g, lightColor.b, 0.0);
    } else if (light.isSpotLight) {
      const record = this.getLightRecord(state.spots);

      lightTarget.setFromMatrixPosition(light.target!.matrixWorld);
      record.position.set(lightPosition.x, lightPosition.y, lightPosition.z, 3.0);
      record.color.set(lightColor.r, lightColor.g, lightColor.b, Math.cos(light.angle! * (1.0 - light.penumbra!)));
      lightTarget.subVectors(lightPosition, lightTarget).normalize();
      record.spot.set(lightTarget.x, lightTarget.y, lightTarget.z, Math.cos(light.angle!));
      record.falloff.set(light.distance!, light.decay!);
    } else if (light.isPointLight) {
      const record = this.getLightRecord(state.points);

      record.position.set(lightPosition.x, lightPosition.y, lightPosition.z, 2.0);
      record.color.set(lightColor.r, lightColor.g, lightColor.b, 0.0);
      record.falloff.set(light.distance!, light.decay!);
    }
  };

  // NOTE Scene lights into the light uniforms. Node materials in custom passes have no three.js light uniforms.
  // Shadow slots follow the light order: directional, point, spot.
  // NOTE Returns this.lightResult, used again in the next call.
  updateLights(scene: SceneLike, camera: THREE.Camera): LightResult {
    const u = this.lightUniforms;
    const state = this.lightState;
    const { directions, points, spots } = state;

    state.camera = camera;
    state.hemiCount = 0;
    state.hasSky = false;
    directions.length = 0;
    points.length = 0;
    spots.length = 0;

    u.frameAmbientColor.value.setRGB(0.0, 0.0, 0.0);
    u.frameHemiSkyColor.value.setRGB(0.0, 0.0, 0.0);
    u.frameHemiGroundColor.value.setRGB(0.0, 0.0, 0.0);
    u.frameSkyDirection.value.set(0.0, 1.0, 0.0);

    for (const v of u.frameLightProbe.array as THREE.Vector3[]) {
      v.set(0.0, 0.0, 0.0);
    }

    scene.traverseVisible(this.visitLight);
    state.camera = null;

    const { hemiCount } = state;
    const lightTotal = directions.length + points.length + spots.length;

    if ((lightTotal > frameMaxLights || hemiCount > frameMaxHemisphereLights) && !this.lightLimitWarned) {
      console.warn(`VDBFrameRenderer: too many lights. Only ${frameMaxLights} lights and ${frameMaxHemisphereLights} hemisphere lights are used.`);
      this.lightLimitWarned = true;
    }

    u.frameHemiCount.value = Math.min(hemiCount, frameMaxHemisphereLights);

    const count = Math.min(lightTotal, frameMaxLights);
    const positions = u.frameLightPositions.array as THREE.Vector4[];
    const colors = u.frameLightColors.array as THREE.Vector4[];
    const spotArray = u.frameLightSpots.array as THREE.Vector4[];
    const falloffs = u.frameLightFalloffs.array as THREE.Vector2[];

    for (let i = 0; i < count; i++) {
      const light = this.getSortedLight(i);

      positions[i]!.copy(light.position);
      colors[i]!.copy(light.color);
      spotArray[i]!.copy(light.spot);
      falloffs[i]!.copy(light.falloff);
    }

    u.frameLightCount.value = count;

    // NOTE Environment lights. 'volume' lighting marches all of them in each volume.
    const { environment } = this;
    const environmentCount = environment ? environment.count : 0;
    const environmentDirections = u.frameEnvironmentDirections.array as THREE.Vector3[];
    const environmentColors = u.frameEnvironmentColors.array as THREE.Vector3[];

    u.frameEnvironmentCount.value = environmentCount;

    for (let i = 0; i < environmentCount; i++) {
      environmentDirections[i]!.fromArray(environment!.directions[i]!);
      environmentColors[i]!.fromArray(environment!.colors[i]!);
    }

    // NOTE 'frame' lighting. Scene lights take the first shadow slots.
    // NOTE The dominant 'directional' environment light (light 0) takes the next free slot. Other environment lights go to the environment grid.
    const sceneShadowCount = Math.min(count, frameShadowLightSlots);
    const shadowLights = this.transmittanceMaterial.frameShadowLights.array as THREE.Vector4[];
    const frameLighting = this.isFrameLighting() && environmentCount > 0;
    const environmentLightCount =
      frameLighting &&
      environment!.mode === frameEnvironmentModes.directional &&
      sceneShadowCount < frameShadowLightSlots &&
      // NOTE Diffusion: all environment lights use the environment grid. Then both lights of an opposite pair get the same march.
      !isFrameDiffusion(this.multipleScatteringMode())
        ? 1
        : 0;
    const result = this.lightResult;
    const environmentGridLights = result.environmentGridLights;

    environmentGridLights.length = 0;

    for (let i = 0; i < sceneShadowCount; i++) {
      shadowLights[i]!.copy(this.getSortedLight(i).position);
    }

    if (environmentLightCount > 0) {
      shadowLights[sceneShadowCount]!.fromArray(environment!.directions[0]!).setW(1.0);
    }

    u.frameEnvironmentUnshadowed.value.set(0.0, 0.0, 0.0);

    if (frameLighting) {
      for (let i = environmentLightCount; i < environmentCount; i++) {
        const entries = this.environmentGridLightEntries;
        const index = environmentGridLights.length;

        if (index === entries.length) {
          entries.push({ direction: [], color: [] });
        }

        entries[index]!.direction = environment!.directions[i]!;
        entries[index]!.color = environment!.colors[i]!;
        environmentGridLights.push(entries[index]!);
        u.frameEnvironmentUnshadowed.value.add(environmentColors[i]!);
      }
    }

    u.frameEnvironmentUnshadowed.value.max(vector3.set(0.0, 0.0, 0.0));

    result.shadowLightCount = sceneShadowCount + environmentLightCount;
    result.environmentLightCount = environmentLightCount;
    result.environmentShadowSlot = sceneShadowCount;

    return result;
  }

  dispose(): void {
    // NOTE Puts back the shadow filter of each light; the new instance installs again on the next render
    this.volumeShadows.dispose();
    this.volumeShadows = new VDBFrameVolumeShadows(this);
    this.gridRenderId = -1;
    this.gridCamera = null;
    this.historyTargets.forEach((target) => target?.dispose());
    this.historyTargets = [null, null];
    this.historyValid = false;
    this.disposeSpectralTransmittanceTargets();

    for (const key of ["atlasTarget", "densityTarget", "shadowDensityTarget", "transmittanceTarget", "environmentMapTarget", "environmentGridTarget"] as const) {
      this[key]?.dispose();
      this[key] = null;
    }

    this.environmentLights = new VDBFrameEnvironmentLights((renderer, scene) => this.renderEnvironmentMap(renderer, scene));
    this.environment = null;
    this.environmentGridSignature = [];
    this.gridMipTargets.forEach((target) => target.dispose());
    this.gridMipTargets = [];
    this.gridMipSignature = [];

    for (const { mesh, geometry } of [...this.densityPool, ...this.shadowPool]) {
      mesh.parent?.remove(mesh);
      geometry.dispose();
    }

    this.densityPool = [];
    this.shadowPool = [];
    this.shadowDrawnVersion = -1;
    this.shadowDrawnTarget = null;

    for (const material of [this.integrateMaterial, this.lightingMaterial, this.transmittanceMaterial, this.temporalMaterial, this.densityClearMaterial, this.environmentMapMaterial, this.environmentGridMaterial, this.gridMipMaterial]) {
      material.dispose();
    }

    this.injectMesh.geometry.dispose();
    (this.injectMesh.material as THREE.Material).dispose();

    for (const mesh of [this.transmittanceMesh, this.environmentMapMesh, this.environmentGridMesh, this.gridMipMesh]) {
      mesh.geometry.dispose();
    }

    this.areaGeometry.dispose();
    this.cachedFrame = -1;
    this.cachedCamera = null;
  }
}

/** The scene fields the environment lights read (three's `Scene`). */
interface VDBFrameEnvironmentSceneFields {
  environment: THREE.Texture | null;
  environmentRotation: THREE.Euler;
  environmentIntensity: number;
}

/** frameShaderChunks.js L800–830 `setVolumeUniforms`: the values of one volume before its draw (inject, density or shadow density). */
export function setVolumeUniforms(material: { volumeUniforms: FrameVolumeUniforms; sourceUniforms: VDBVolumeUniforms }, frameVolume: VDBFrameVolume, volume: FrameVolumeRecord): void {
  const u = material.volumeUniforms;
  const s = material.sourceUniforms;

  u.frameViewToLocal.value.copy(volume.viewToLocal);
  u.frameLocalToView.value.copy(volume.localToView);
  u.frameWorldToLocal.value.copy(volume.worldToLocal);
  s.worldScale.value = volume.matrixWorld.getMaxScaleOnAxis();
  s.baseColor.value.copy(frameVolume.baseColor);
  u.frameScatterColor.value.copy(frameVolume.scatterColor);
  s.extinction.value = frameVolume.extinction;
  getFrameExtinctionSpectrum(frameVolume, u.frameExtinctionSpectrum.value);
  s.densityScale.value = frameVolume.getDensityScale();
  s.densityCutoff.value = frameVolume.densityCutoff;
  u.frameAbsorbance.value = frameVolume.absorbance;
  u.frameOpacity.value = frameVolume.opacity;
  s.emissiveIntensity.value = frameVolume.emissiveIntensity;
  s.temperatureRange.value.copy(frameVolume.temperatureRange);
  s.offset3D.value.copy(frameVolume.offset3D);
  s.wrap3D.value = frameVolume.wrap3D;
  getFramePhase(frameVolume, u.framePhase.value);
  getFramePhaseSpectrum(frameVolume, u.framePhaseSpectrum.value);
}

/** Per three renderer shared instances (`vdbFrameRenderer(renderer)`). */
const sharedByRenderer = new WeakMap<object, VDBFrameRenderer>();
let shared: VDBFrameRenderer | null = null;

/**
 * The shared {@link VDBFrameRenderer} that a `VDBFrameVolume` uses when you pass no `frameRenderer`.
 * Its `settings` are global to every volume that uses it (the froxel grid, shadows and scattering are
 * built once for all of them). Pass a three renderer to get the instance shared by that renderer only
 * (a WeakMap, so two three renderers never share targets or settings); with no argument you get the
 * one process-wide default. For volumes that need different settings, make your own
 * `new VDBFrameRenderer()` and pass it as `VDBFrameVolume({ frameRenderer })`.
 */
export function vdbFrameRenderer(renderer?: object): VDBFrameRenderer {
  if (renderer) {
    let r = sharedByRenderer.get(renderer);
    if (!r) sharedByRenderer.set(renderer, (r = new VDBFrameRenderer()));
    return r;
  }
  shared ??= new VDBFrameRenderer();
  return shared;
}
