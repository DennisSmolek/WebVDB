/**
 * K3.3 helpers of the k3-frame harness (`?lighting=frame`): GPU-vs-CPU-twin checks of mjurczyk's
 * 'frame' lighting passes, from read-back render targets.
 *
 * - `probeGrid`: the grid TSL chunks (getGridCell, findCascade, gridUvw, frameIsotropicDiffusion) in a
 *   fragment pass vs `vdbFrameGridCell` / `vdbFrameFindCascade` / `vdbFrameIsotropicDiffusion`.
 * - `transmittanceTwin`: the transmittance pass (4 MRT) vs `vdbFrameTransmittanceVoxel` marched over
 *   the read-back shadow density atlas (same input, so only the march differs).
 * - `lightingTwin`: the lighting pass (froxel atlas) vs `vdbFrameLightingFroxel` from the read-back
 *   density targets (ambient, hemisphere, directional slots, the environment slot).
 * - `syntheticEnvironment`: a deterministic equirect sky with a sun (half floats), for 'directional'
 *   environment lighting without an HDR file.
 * K3.4 (phase functions + multiple scattering):
 * - the twins above take the octave, two-sided and neural modes and the per-channel / Lorenz-Mie phase
 *   from the renderer's uniforms; `gridMipTwin`: the neural input mips vs `vdbFrameGridMipLevel`.
 * - `probeScattering`: the K3.4 TSL chunks (Mie table read, per-channel phase, octaves, two-sided
 *   diffusion, the octave shadow total and shadowed light) vs their CPU twins.
 * - `furnaceEnvironment`: a uniform equirect map of radiance L (his utils/furnace.js white furnace).
 */
import * as THREE from "three";
import { NodeMaterial, type WebGPURenderer } from "three/webgpu";
import * as TSL from "three/tsl";
import type { VDBFrameRenderer, VDBFrameVolume } from "@pmndrs/vdb/three";
import { vdbFrameChunks } from "../../../packages/vdb/src/three/internal.js";
import {
  sampleWvdbDenseTexture,
  vdbFrameGridMipLevel,
  vdbFrameHenyeyGreensteinColor,
  vdbFrameMieTexel,
  vdbFrameNeuralLight,
  vdbFrameNeuralMipResolution,
  vdbFrameOctaveLight,
  vdbFramePhaseColor,
  vdbFramePhaseWeight,
  vdbFramePhaseWeightScaled,
  vdbFrameShadowMarchDepth,
  vdbFrameShadowOpticalDepth,
  vdbFrameShadowTotal,
  vdbFrameShadowTransmittance,
  vdbFrameShadowedLightColor,
  vdbFrameSampleShadows,
  vdbFrameTwoSidedDiffusion,
  type VDBFrameNeuralMip,
  type VDBFrameTransmittanceModes,
  vdbFrameDensityFroxel,
  type VDBFrameDensityInputs,
  type WvdbDenseVolume,
  vdbFrameEnvironmentGridVoxel,
  vdbFrameFindCascade,
  vdbFrameFroxel,
  vdbFrameGridCell,
  vdbFrameGridUvw,
  vdbFrameIsotropicDiffusion,
  vdbFrameLightingFroxel,
  vdbFrameTransmittanceVoxel,
  type VDBFrameAtlas,
  type VDBFrameGridLayout,
} from "../../../packages/vdb/src/tools/internal.js";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type N = any;
type V2 = [number, number];
type V3 = [number, number, number];
type V4 = [number, number, number, number];
const { Fn, float, screenCoordinate, vec2, vec3, vec4, If, int } = TSL as unknown as Record<string, N>;

export interface TwinStats {
  compared: number;
  maxAbs: number;
  maxRel: number;
  over: number;
  /** Fraction of compared values within tolerance. */
  within: number;
  firstBad?: unknown;
}

/** Readback as tight rows (r187: rows top first, padded to 256 bytes). */
export function tightRows<T extends Uint16Array | Float32Array>(pixels: T, w: number, h: number, channels: number): T {
  const tight = w * channels;
  const stride = h > 1 ? (pixels.length - tight) / (h - 1) : tight;
  if (!Number.isInteger(stride) || stride < tight) throw new Error(`readback layout: ${pixels.length} elements for ${w}x${h}x${channels}`);
  const out = new (pixels.constructor as new (n: number) => T)(w * h * channels);
  for (let y = 0; y < h; y++) out.set(pixels.subarray(y * stride, y * stride + tight) as never, y * tight);
  return out;
}

/** A half-float render target texture as a bottom-first float atlas (the shaders' frameFragCoord rows). */
export async function readAtlas(renderer: WebGPURenderer, rt: THREE.RenderTarget, channels: number, textureIndex = 0): Promise<VDBFrameAtlas> {
  const w = rt.width;
  const h = rt.height;
  const raw = (await (renderer.readRenderTargetPixelsAsync as (...a: unknown[]) => Promise<unknown>)(rt, 0, 0, w, h, textureIndex)) as Uint16Array | Float32Array;
  const rows = tightRows(raw, w, h, channels);
  const out = new Float32Array(w * h * channels);
  for (let y = 0; y < h; y++) {
    const src = (h - 1 - y) * w * channels;
    for (let i = 0; i < w * channels; i++) {
      const v = rows[src + i]!;
      out[y * w * channels + i] = rows instanceof Uint16Array ? THREE.DataUtils.fromHalfFloat(v) : v;
    }
  }
  return { data: out, width: w, height: h, channels };
}

export function gridLayout(fr: VDBFrameRenderer): VDBFrameGridLayout {
  const g = fr.gridUniforms;
  const v3 = (a: THREE.Vector3[]) => a.map((v) => [v.x, v.y, v.z] as V3);
  return {
    count: g.frameGridCount.value as number,
    resolution: g.frameGridResolution.value as number,
    columns: g.frameGridColumns.value as number,
    atlasSize: [g.frameGridAtlasSize.value.x, g.frameGridAtlasSize.value.y],
    gridMin: v3(g.frameGridMin.array as THREE.Vector3[]),
    gridSize: v3(g.frameGridSize.array as THREE.Vector3[]),
    gridVoxel: v3(g.frameGridVoxel.array as THREE.Vector3[]),
  };
}

export function accumulate(stats: TwinStats, gpu: number, cpu: number, absTol: number, relTol: number, where: unknown): void {
  const d = Math.abs(gpu - cpu);
  stats.compared++;
  stats.maxAbs = Math.max(stats.maxAbs, d);
  stats.maxRel = Math.max(stats.maxRel, d / Math.max(1e-3, Math.abs(cpu)));
  if (d > absTol + relTol * Math.abs(cpu)) {
    stats.over++;
    stats.firstBad ??= { ...((where as object) ?? {}), gpu, cpu };
  }
}

export const newStats = (): TwinStats => ({ compared: 0, maxAbs: 0, maxRel: 0, over: 0, within: 1 });
export const finish = (s: TwinStats) => {
  s.within = s.compared > 0 ? 1 - s.over / s.compared : 0;
  return s;
};

/**
 * The transmittance pass vs its CPU twin on the read-back shadow density atlas. `step` subsamples the
 * atlas; only voxels the pass marched (some output ≠ 1) and a few skipped ones are compared.
 */
export async function transmittanceTwin(renderer: WebGPURenderer, fr: VDBFrameRenderer, step = 3): Promise<TwinStats & { marched: number }> {
  const density = await readAtlas(renderer, fr.shadowDensityTarget!, 2);
  const outs = await Promise.all([0, 1, 2, 3].map((i) => readAtlas(renderer, fr.transmittanceTarget!, 4, i)));
  const grid = gridLayout(fr);
  const tm = fr.transmittanceMaterial;
  const count = tm.frameShadowLightCount.value as number;
  const lights = (tm.frameShadowLights.array as THREE.Vector4[]).slice(0, count).map((v) => [v.x, v.y, v.z, v.w] as V4);
  const sky = fr.lightUniforms.frameSkyDirection.value;
  const ms = tm.frameMultipleScattering.value as number;
  const jitter = fr.froxelUniforms.frameJitter.value as number;
  const modes = await transmittanceModes(renderer, fr, density, grid);
  const stats = newStats();
  let marched = 0;
  for (let y = 0; y < density.height; y += step) {
    for (let x = 0; x < density.width; x += step) {
      const o = (y * density.width + x) * 4;
      const gpu = outs.map((t) => [t.data[o]!, t.data[o + 1]!, t.data[o + 2]!, t.data[o + 3]!]);
      const touched = gpu.some((v) => v.some((c) => c !== 1));
      if (touched) marched++;
      else if ((x + y) % (step * 7) !== 0) continue;
      const cpu = vdbFrameTransmittanceVoxel(density, grid, [x + 0.5, y + 0.5], lights, [sky.x, sky.y, sky.z], ms, jitter, modes);
      for (let t = 0; t < 4; t++) for (let c = 0; c < 4; c++) accumulate(stats, gpu[t]![c]!, cpu[t]![c]!, 2e-3, 4e-3, { x, y, t, c });
    }
  }
  return { ...finish(stats), marched };
}

/** The read-back neural input mips (levels 1 and up) of the renderer's last grid mip pass. */
export async function readNeuralMips(renderer: WebGPURenderer, fr: VDBFrameRenderer): Promise<VDBFrameNeuralMip[]> {
  const resolution = fr.gridUniforms.frameGridResolution.value as number;
  const layouts = fr.transmittanceMaterial.frameNeuralMipLayouts;
  return Promise.all(
    fr.gridMipTargets.map(async (t, i) => ({ atlas: await readAtlas(renderer, t, 2), resolution: vdbFrameNeuralMipResolution(resolution, i + 1), columns: layouts[i]!.value.x as number })),
  );
}

/** The K3.4 modes of the transmittance material (octaves, two-sided, the network over the read-back mips). */
async function transmittanceModes(renderer: WebGPURenderer, fr: VDBFrameRenderer, density: VDBFrameAtlas, grid: VDBFrameGridLayout): Promise<VDBFrameTransmittanceModes> {
  const tm = fr.transmittanceMaterial;
  const o = tm.frameScatteringOctaves.value;
  const modes: VDBFrameTransmittanceModes = { octaves: [o.x, o.y, o.z, o.w], twoSided: (tm.frameTwoSided.value as number) > 0.5 };
  if (tm.neural) {
    const network = tm.neural;
    const mips = await readNeuralMips(renderer, fr);
    modes.neural = (world, direction, opticalDepth) => vdbFrameNeuralLight(network, grid, density, mips, world, direction, opticalDepth);
  }
  return modes;
}

/**
 * The neural input mips (VDBFrameGridMipMaterial) vs `vdbFrameGridMipLevel`: level 1 from the read-back
 * grid atlas (the last cascade's tiles), level k from the read-back level k − 1. Every texel.
 */
export async function gridMipTwin(renderer: WebGPURenderer, fr: VDBFrameRenderer): Promise<TwinStats & { levels: number }> {
  const density = await readAtlas(renderer, fr.shadowDensityTarget!, 2);
  const mips = await readNeuralMips(renderer, fr);
  const g = fr.gridUniforms;
  const resolution = g.frameGridResolution.value as number;
  const stats = newStats();
  mips.forEach((mip, i) => {
    const source = i === 0 ? { atlas: density, resolution, columns: g.frameGridColumns.value as number, tile: ((g.frameGridCount.value as number) - 1) * resolution } : { ...mips[i - 1]!, tile: 0 };
    const cpu = vdbFrameGridMipLevel(source.atlas, source.resolution, source.columns, source.tile, mip.resolution);
    if (cpu.atlas.width !== mip.atlas.width || cpu.atlas.height !== mip.atlas.height) throw new Error(`gridMipTwin: level ${i + 1} is ${mip.atlas.width}x${mip.atlas.height}, the twin ${cpu.atlas.width}x${cpu.atlas.height}`);
    for (let k = 0; k < cpu.atlas.data.length; k++) accumulate(stats, mip.atlas.data[k]!, cpu.atlas.data[k]!, 1e-3, 2e-3, { level: i + 1, k });
  });
  return { ...finish(stats), levels: mips.length };
}

/**
 * The lighting pass vs `vdbFrameLightingFroxel` from the read-back density targets, over the froxel
 * area. Needs a scene without spot/point lights or light probes (per-channel phases: K3.4).
 */
export async function lightingTwin(renderer: WebGPURenderer, fr: VDBFrameRenderer, step = 2): Promise<TwinStats & { lit: number }> {
  const targets = await Promise.all([0, 1, 2, 3].map((i) => readAtlas(renderer, fr.densityTarget!, 4, i)));
  const atlas = await readAtlas(renderer, fr.atlasTarget!, 4);
  const fu = fr.froxelUniforms;
  const lu = fr.lightUniforms;
  const lm = fr.lightingMaterial;
  const grid: V3 = [fu.froxelGrid.value.x, fu.froxelGrid.value.y, fu.froxelGrid.value.z];
  const columns = fu.froxelColumns.value as number;
  const range: V2 = [fu.froxelRange.value.x, fu.froxelRange.value.y];
  const projectionInverse = fu.froxelProjectionInverse.value.elements;
  const cameraWorld = lm.frameCameraMatrixWorld.value.elements;
  const lightCount = lu.frameLightCount.value as number;
  const lights = (lu.frameLightPositions.array as THREE.Vector4[]).slice(0, lightCount).map((p, i) => {
    const c = (lu.frameLightColors.array as THREE.Vector4[])[i]!;
    return { position: [p.x, p.y, p.z, p.w] as V4, color: [c.x, c.y, c.z] as V3 };
  });
  const envCount = lm.frameEnvironmentLightCount.value as number;
  const envColors = (lu.frameEnvironmentColors.array as THREE.Vector3[]).slice(0, envCount).map((c) => [c.x, c.y, c.z] as V3);
  const amb = lu.frameAmbientColor.value;
  const hs = lu.frameHemiSkyColor.value;
  const hg = lu.frameHemiGroundColor.value;
  const phase = lm.framePhase.value;
  const oct = lm.frameScatteringOctaves.value;
  const spectrum = lm.framePhaseSpectrum.value;
  const colorPhase = spectrum.z > 0.5 && (fr.densityShadowUniforms.frameUseShadows.value as number) === 1;
  const directMaps = colorPhase ? await Promise.all([2, 3].map((i) => readAtlas(renderer, fr.transmittanceTarget!, 4, i))) : null;
  const grid0 = gridLayout(fr);
  const directAt = directMaps ? (world: V3) => {
    const s = vdbFrameSampleShadows(directMaps, grid0, world);
    return [s[0]![0]!, s[0]![1]!, s[0]![2]!, s[0]![3]!, s[1]![0]!, s[1]![1]!];
  } : undefined;
  const stats = newStats();
  let lit = 0;
  const w = atlas.width;
  for (let y = 0; y < atlas.height; y += step) {
    for (let x = 0; x < w; x += step) {
      const o = (y * w + x) * 4;
      const d = targets.map((t) => [t.data[o]!, t.data[o + 1]!, t.data[o + 2]!, t.data[o + 3]!] as V4);
      if (!(d[0]![3] > 0)) continue;
      const froxel = vdbFrameFroxel(grid, columns, range, projectionInverse, [x + 0.5, y + 0.5]);
      if (froxel.slice >= grid[2]) continue;
      lit++;
      const cpu = vdbFrameLightingFroxel(d as [V4, V4, V4, V4], froxel.viewNear, froxel.viewFar, cameraWorld, {
        ambient: [amb.r, amb.g, amb.b],
        hemiSky: [hs.r, hs.g, hs.b],
        hemiGround: [hg.r, hg.g, hg.b],
        lights,
        shadowLightCount: lm.frameShadowLightCount.value as number,
        phase: [phase.x, phase.y, phase.z, phase.w],
        octaves: [oct.x, oct.y, oct.z, oct.w],
        phaseSpectrum: [spectrum.x, spectrum.y, spectrum.z],
        ...(directAt ? { directAt } : {}),
        environmentColors: envColors,
        environmentShadowSlot: lm.frameEnvironmentShadowSlot.value as number,
      });
      for (let c = 0; c < 4; c++) accumulate(stats, atlas.data[o + c]!, cpu[c]!, 2e-3, 4e-3, { x, y, c });
    }
  }
  return { ...finish(stats), lit };
}

/**
 * The density pass vs `vdbFrameDensityFroxel` for one volume on the dense source (the CPU twin of his
 * sampler, `sampleWvdbDenseTexture`), with the shadow slots sampled from the read-back transmittance
 * targets. One volume only: the targets are additive.
 */
export async function densityTwin(renderer: WebGPURenderer, fr: VDBFrameRenderer, volume: VDBFrameVolume, dense: WvdbDenseVolume, step = 2): Promise<TwinStats & { covered: number }> {
  const targets = await Promise.all([0, 1, 2, 3].map((i) => readAtlas(renderer, fr.densityTarget!, 4, i)));
  const shadows = await Promise.all([0, 1, 2, 3].map((i) => readAtlas(renderer, fr.transmittanceTarget!, 4, i)));
  const fu = fr.froxelUniforms;
  const material = volume.densityMaterial!;
  const vu = material.volumeUniforms;
  const su = volume.sourceUniforms;
  const ds = fr.densityShadowUniforms;
  const lu = fr.lightUniforms;
  const tex = dense.density;
  const mipLevels = volume.mipmaps && tex.mip ? tex.mip.levels : 0;
  const densityScale = su.densityScale.value as number;
  const extinctionScale = su.extinction.value as number;
  const worldScale = su.worldScale.value as number;
  const cutoff = su.densityCutoff.value as number;
  const phase = ds.framePhase.value;
  const sceneLightCount = ds.frameShadowSceneLightCount.value as number;
  const inputs: VDBFrameDensityInputs = {
    froxelGrid: [fu.froxelGrid.value.x, fu.froxelGrid.value.y, fu.froxelGrid.value.z],
    columns: fu.froxelColumns.value as number,
    range: [fu.froxelRange.value.x, fu.froxelRange.value.y],
    projectionInverse: fu.froxelProjectionInverse.value.elements,
    cameraWorld: ds.frameCameraMatrixWorld.value.elements,
    viewToLocal: vu.frameViewToLocal.value.elements,
    maxTexels: Math.max(...dense.dimensions),
    mipLevels,
    worldScale,
    subSteps: fu.froxelSubSteps.value as number,
    jitter: fu.frameJitter.value as number,
    fadeRadius: fu.frameFadeRadius.value as number,
    opacity: vu.frameOpacity.value as number,
    baseColor: (su.baseColor.value as THREE.Color).toArray() as V3,
    scatterColor: (vu.frameScatterColor.value as THREE.Color).toArray() as V3,
    extinctionAt: (local, lod) => {
      const v = sampleWvdbDenseTexture(tex, [local[0] + 0.5, local[1] + 0.5, local[2] + 0.5], mipLevels > 0 ? lod : 0);
      return ((v > cutoff ? v : 0) * densityScale * extinctionScale) / worldScale;
    },
    shadows: (ds.frameUseShadows.value as number) === 1 ? { atlases: shadows, grid: gridLayout(fr) } : null,
    sceneLightCount,
    lightPositions: (lu.frameLightPositions.array as THREE.Vector4[]).slice(0, Math.max(sceneLightCount, 0)).map((v) => [v.x, v.y, v.z, v.w] as V4),
    phase: [phase.x, phase.y, phase.z, phase.w],
    phaseSpectrum: [ds.framePhaseSpectrum.value.x, ds.framePhaseSpectrum.value.y, ds.framePhaseSpectrum.value.z],
    octaves: [ds.frameScatteringOctaves.value.x, ds.frameScatteringOctaves.value.y, ds.frameScatteringOctaves.value.z, ds.frameScatteringOctaves.value.w],
  };
  if ((ds.frameEnvironmentMode.value as number) === 2) {
    const u = lu.frameEnvironmentUnshadowed.value;
    inputs.environmentGrid = { atlas: await readAtlas(renderer, fr.environmentGridTarget!, 4), columns: fr.environmentGridUniforms.frameEnvironmentGridColumns.value as number, unshadowed: [u.x, u.y, u.z] };
  } else if ((ds.frameEnvironmentMode.value as number) === 0 && fr.densityScene.environment) {
    throw new Error("densityTwin: the 'outward' environment term (PMREM) has no CPU twin; use 'directional' lights");
  }
  const stats = newStats();
  let covered = 0;
  const w = targets[0]!.width;
  for (let y = 0; y < targets[0]!.height; y += step) {
    for (let x = 0; x < w; x += step) {
      const o = (y * w + x) * 4;
      const gpu = targets.map((t) => [t.data[o]!, t.data[o + 1]!, t.data[o + 2]!, t.data[o + 3]!]);
      const froxelSlice = Math.floor(y / inputs.froxelGrid[1]) * inputs.columns + Math.floor(x / inputs.froxelGrid[0]);
      if (froxelSlice >= inputs.froxelGrid[2]) continue;
      const cpu = vdbFrameDensityFroxel([x + 0.5, y + 0.5], inputs) ?? [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ];
      if (cpu[0]![3] > 0 || gpu[0]![3]! > 0) covered++;
      else continue;
      for (let t = 0; t < 4; t++) for (let c = 0; c < 4; c++) accumulate(stats, gpu[t]![c]!, cpu[t]![c]!, 2e-3, 4e-3, { x, y, t, c });
    }
  }
  return { ...finish(stats), covered };
}

/** The environment grid pass vs `vdbFrameEnvironmentGridVoxel` on the read-back shadow density atlas. */
export async function environmentGridTwin(renderer: WebGPURenderer, fr: VDBFrameRenderer, step = 2): Promise<TwinStats & { marched: number }> {
  const density = await readAtlas(renderer, fr.shadowDensityTarget!, 2);
  const out = await readAtlas(renderer, fr.environmentGridTarget!, 4);
  const grid = gridLayout(fr);
  const em = fr.environmentGridMaterial;
  const n = em.frameLightCount.value as number;
  const lights = Array.from({ length: n }, (_, i) => ({
    direction: (em.frameLightDirections.array as THREE.Vector3[])[i]!.toArray() as V3,
    color: (em.frameLightColors.array as THREE.Vector3[])[i]!.toArray() as V3,
  }));
  const u = fr.lightUniforms.frameEnvironmentUnshadowed.value;
  const columns = fr.environmentGridUniforms.frameEnvironmentGridColumns.value as number;
  const ms = em.frameMultipleScattering.value as number;
  const ov = em.frameScatteringOctaves.value;
  const octaves: V4 = [ov.x, ov.y, ov.z, ov.w];
  const outside = vdbFrameShadowTotal(1, octaves);
  const stats = newStats();
  let marched = 0;
  for (let y = 0; y < out.height; y += step) {
    for (let x = 0; x < out.width; x += step) {
      const o = (y * out.width + x) * 4;
      const gpu = [out.data[o]!, out.data[o + 1]!, out.data[o + 2]!, out.data[o + 3]!];
      const cpu = vdbFrameEnvironmentGridVoxel(density, grid, columns, [x + 0.5, y + 0.5], lights, [u.x, u.y, u.z], ms, octaves);
      if (cpu[0] !== u.x * outside || cpu[1] !== u.y * outside || cpu[2] !== u.z * outside) marched++;
      for (let c = 0; c < 4; c++) accumulate(stats, gpu[c]!, cpu[c]!, 2e-3, 4e-3, { x, y, c });
    }
  }
  return { ...finish(stats), marched };
}

/** The grid TSL chunks vs their CPU twins (one RGBA32F value per channel). */
export async function probeGrid(renderer: WebGPURenderer, fr: VDBFrameRenderer): Promise<TwinStats> {
  const C = vdbFrameChunks;
  const gu = C.createGridUniforms(float(-1.0));
  // A 3-cascade layout like the renderer's: resolution 16, 2 columns? Use the renderer's own values when set.
  const layout = gridLayout(fr).count > 0 ? gridLayout(fr) : null;
  const res = 16;
  const columns = 8;
  const L: VDBFrameGridLayout = layout ?? {
    count: 3,
    resolution: res,
    columns,
    atlasSize: [res * columns, res * Math.ceil((res * 3) / columns)],
    gridMin: [
      [-4, -4, -4],
      [-8, -8, -8],
      [-16, -16, -16],
      [0, 0, 0],
    ],
    gridSize: [
      [8, 8, 8],
      [16, 16, 16],
      [32, 32, 32],
      [1, 1, 1],
    ],
    gridVoxel: [
      [0.5, 0.5, 0.5],
      [1, 1, 1],
      [2, 2, 2],
      [1, 1, 1],
    ],
  };
  gu.frameGridCount.value = L.count;
  gu.frameGridResolution.value = L.resolution;
  gu.frameGridColumns.value = L.columns;
  gu.frameGridAtlasSize.value.set(L.atlasSize[0], L.atlasSize[1]);
  for (let i = 0; i < 4; i++) {
    (gu.frameGridMin.array as THREE.Vector3[])[i]!.fromArray(L.gridMin[i] ?? [0, 0, 0]);
    (gu.frameGridSize.array as THREE.Vector3[])[i]!.fromArray(L.gridSize[i] ?? [1, 1, 1]);
    (gu.frameGridVoxel.array as THREE.Vector3[])[i]!.fromArray(L.gridVoxel[i] ?? [1, 1, 1]);
  }
  const nodes = C.createGridNodes(gu);
  const PW = 64;
  const PH = 4;
  const last = L.count - 1;
  const lm = L.gridMin[last]!;
  const ls = L.gridSize[last]!;
  // Probe world points: a line through the last cascade's box, past its sides.
  const worldAt = (a: number): V3 => [lm[0] + ls[0] * (-0.1 + 1.2 * a), lm[1] + ls[1] * (0.5 + 0.3 * Math.sin(a * 7)), lm[2] + ls[2] * (1.1 - 1.2 * a)];
  const atlasW = L.atlasSize[0];
  const atlasH = L.atlasSize[1];
  const fragAt = (a: number, row: number): V2 => [Math.floor(a * atlasW) + 0.5, Math.floor(((row * 17 + a * 61) % 1) * atlasH) + 0.5];

  const material = new NodeMaterial();
  material.fragmentNode = Fn(() => {
    const px = screenCoordinate.x.floor();
    const row = screenCoordinate.y.floor().toVar();
    const a = px.add(0.5).div(PW).toVar();
    const out = vec4(0.0).toVar();
    const world = vec3(
      float(lm[0]).add(a.mul(1.2).sub(0.1).mul(ls[0])),
      float(lm[1]).add(a.mul(7.0).sin().mul(0.3).add(0.5).mul(ls[1])),
      float(lm[2]).add(float(1.1).sub(a.mul(1.2)).mul(ls[2])),
    ).toVar();
    If(row.lessThan(0.5), () => {
      const c0 = nodes.findCascade(world, int(0)).toVar();
      const c1 = nodes.findCascade(world, int(1)).toVar();
      const uvw = nodes.gridUvw(int(last), world);
      out.assign(vec4(float(c0), float(c1), uvw.x, uvw.z));
    })
      .ElseIf(row.lessThan(2.5), () => {
        const frag = TSL.vec2(a.mul(atlasW).floor().add(0.5), row.mul(17.0).add(a.mul(61.0)).fract().mul(atlasH).floor().add(0.5));
        const cell = nodes.getGridCell(frag);
        out.assign(vec4(float(cell.cascade), cell.gridUvw));
      })
      .Else(() => {
        const f = a.mul(5.0);
        out.assign(vec4(C.frameIsotropicDiffusion(f, a.mul(3.0), a), C.frameIsotropicDiffusion(a, f, float(0.0)), C.frameShadowTotal(a, vec4(0.0)), float(0.0)));
      });
    return out;
  })();
  material.vertexNode = vec4(TSL.positionGeometry.xy, 0.0, 1.0);
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  quad.frustumCulled = false;
  const scene = new THREE.Scene();
  scene.add(quad);
  const target = new THREE.RenderTarget(PW, PH, { type: THREE.FloatType, depthBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
  renderer.setRenderTarget(target);
  renderer.render(scene, new THREE.OrthographicCamera());
  renderer.setRenderTarget(null);
  const raw = (await renderer.readRenderTargetPixelsAsync(target, 0, 0, PW, PH)) as unknown as Float32Array;
  const gpu = tightRows(raw, PW, PH, 4);
  target.dispose();
  material.dispose();
  const stats = newStats();
  for (let row = 0; row < PH; row++) {
    for (let x = 0; x < PW; x++) {
      const a = (x + 0.5) / PW;
      let cpu: number[];
      if (row === 0) {
        const w = worldAt(a);
        const uvw = vdbFrameGridUvw(L, last, w);
        cpu = [vdbFrameFindCascade(L, w, 0), vdbFrameFindCascade(L, w, 1), uvw[0], uvw[2]];
      } else if (row <= 2) {
        const cell = vdbFrameGridCell(L.resolution, L.columns, fragAt(a, row));
        cpu = [cell.cascade, ...cell.gridUvw];
      } else {
        cpu = [vdbFrameIsotropicDiffusion(a * 5, a * 3, a), vdbFrameIsotropicDiffusion(a, a * 5, 0), a, 0];
      }
      for (let c = 0; c < 4; c++) accumulate(stats, gpu[(row * PW + x) * 4 + c]!, cpu[c]!, 1e-5, 2e-5, { row, x, c });
    }
  }
  return finish(stats);
}

/**
 * A deterministic equirect sky (512 × 256 half floats): a vertical gradient (blue sky, darker ground)
 * and a small bright sun at `sun` (unit, world). Mapping: EquirectangularReflectionMapping.
 */
export function syntheticEnvironment(sun: V3 = [0.45, 0.75, 0.48]): THREE.DataTexture {
  const w = 512;
  const h = 256;
  const data = new Uint16Array(w * h * 4);
  const s = Math.hypot(...sun);
  const sd = sun.map((v) => v / s) as V3;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      // three's equirect lookup: u = atan(dir.z, dir.x) / 2π + 0.5, v = asin(dir.y) / π + 0.5 (flipY false: row 0 is v = 0, the bottom).
      const u = (x + 0.5) / w;
      const v = (y + 0.5) / h;
      const phi = (u - 0.5) * 2 * Math.PI;
      const theta = (v - 0.5) * Math.PI;
      const dir: V3 = [Math.cos(theta) * Math.cos(phi), Math.sin(theta), Math.cos(theta) * Math.sin(phi)];
      const up = dir[1];
      const sky: V3 = up >= 0 ? [0.35 + 0.25 * (1 - up), 0.55 + 0.2 * (1 - up), 1.0] : [0.25, 0.22, 0.2];
      const cos = dir[0] * sd[0] + dir[1] * sd[1] + dir[2] * sd[2];
      const sunGlow = cos > 0.995 ? 60 : 0;
      const o = (y * w + x) * 4;
      for (let c = 0; c < 3; c++) data[o + c] = THREE.DataUtils.toHalfFloat(sky[c]! + sunGlow);
      data[o + 3] = THREE.DataUtils.toHalfFloat(1);
    }
  }
  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.HalfFloatType);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.colorSpace = THREE.LinearSRGBColorSpace;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  tex.flipY = false;
  tex.needsUpdate = true;
  return tex;
}

/**
 * The K3.4 TSL chunks vs their CPU twins (one RGBA32F value per channel, 64 angles × 6 rows): the Lorenz-Mie
 * table read and phase weight, the per-channel phase (phaseGExponent and 'lorenz-mie-rgb'), the scaled
 * phase, octave light / march depth / transmittance / shadow total, two-sided diffusion and the shadowed light
 * with octaves.
 */
export async function probeScattering(renderer: WebGPURenderer): Promise<TwinStats> {
  const C = vdbFrameChunks;
  const PW = 64;
  const PH = 6;
  const dual: V4 = [0.8, -0.3, 0.2, 0];
  const mie: V4 = [0, 0, 0, 1];
  const mieRgb: V4 = [0, 0, 0, 2];
  const exponentSpectrum: V3 = [Math.pow(650 / 550, -1), Math.pow(450 / 550, -1), 1];
  const octaves: V4 = [4, 0.5, 0.5, 0.5];
  const material = new NodeMaterial();
  material.fragmentNode = Fn(() => {
    const px = screenCoordinate.x.floor();
    const row = screenCoordinate.y.floor().toVar();
    const a = px.add(0.5).div(PW).toVar();
    const out = vec4(0.0).toVar();
    const cosT = a.mul(2.0).sub(1.0).toVar();
    const sinT = float(1.0).sub(cosT.mul(cosT)).max(0.0).sqrt();
    const ray = vec3(0.0, 0.0, 1.0);
    const light = vec3(sinT, 0.0, cosT);
    const oct = vec4(...octaves);
    const tau = a.mul(6.0).toVar();
    If(row.lessThan(0.5), () => {
      // Mie table: 'lorenz-mie' weight, then the rgb table texel
      const t = C.frameMieTexel(cosT);
      out.assign(vec4(C.framePhaseWeight(vec4(...mie), ray, light), t.r, t.g, t.b));
    })
      .ElseIf(row.lessThan(1.5), () => {
        const rgb = C.framePhaseColor(vec4(...mieRgb), vec3(1.0, 1.0, 1.0), ray, light);
        const exp = C.framePhaseColor(vec4(...dual), vec3(...exponentSpectrum), ray, light);
        out.assign(vec4(rgb.b, exp.r, exp.g, exp.b));
      })
      .ElseIf(row.lessThan(2.5), () => {
        out.assign(
          vec4(
            C.framePhaseWeightScaled(vec4(...dual), cosT, float(0.5)),
            C.framePhaseWeightScaled(vec4(...mie), cosT, float(0.25)),
            C.frameOctaveLight(oct, tau, vec4(...dual), cosT),
            C.frameOctaveLight(oct, tau, null, null),
          ),
        );
      })
      .ElseIf(row.lessThan(3.5), () => {
        const depths = vec3(tau, tau.mul(0.3), tau.mul(0.6)).toVar();
        const t = C.frameShadowTransmittance(depths, float(0.42), oct);
        out.assign(vec4(C.frameShadowMarchDepth(depths, float(0.42), oct), t.x, t.y, C.frameShadowTotal(t.y, oct)));
      })
      .ElseIf(row.lessThan(4.5), () => {
        const f = a.mul(8.0).toVar();
        out.assign(
          vec4(
            C.frameTwoSidedDiffusion(float(0.0), f, a.mul(0.5), float(8.0).sub(f)),
            C.frameTwoSidedDiffusion(float(0.0), f, float(0.0), a.mul(2.0)),
            C.frameTwoSidedDiffusion(float(0.9), a.mul(0.1), float(0.0), float(0.2)),
            C.frameOctaveLight(vec4(8.0, 0.7, 0.9, 0.3), tau, vec4(...mie), cosT),
          ),
        );
      })
      .Else(() => {
        const shadow = vec2(tau.negate().exp(), float(1.0).div(tau.add(1.0))).toVar();
        const lit = C.frameShadowedLight(vec4(...dual), vec3(...exponentSpectrum), oct, ray, light, shadow);
        const plain = C.frameShadowedLight(vec4(...dual), vec3(1.0, 1.0, 0.0), vec4(0.0), ray, light, vec2(shadow.x, tau.mul(-0.5).exp()));
        out.assign(vec4(lit.r, lit.g, lit.b, plain.g));
      });
    return out;
  })();
  material.vertexNode = vec4(TSL.positionGeometry.xy, 0.0, 1.0);
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  quad.frustumCulled = false;
  const scene = new THREE.Scene();
  scene.add(quad);
  const target = new THREE.RenderTarget(PW, PH, { type: THREE.FloatType, depthBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
  renderer.setRenderTarget(target);
  renderer.render(scene, new THREE.OrthographicCamera());
  renderer.setRenderTarget(null);
  const raw = (await renderer.readRenderTargetPixelsAsync(target, 0, 0, PW, PH)) as unknown as Float32Array;
  const gpu = tightRows(raw, PW, PH, 4);
  target.dispose();
  material.dispose();
  const stats = newStats();
  for (let row = 0; row < PH; row++) {
    for (let x = 0; x < PW; x++) {
      const a = (x + 0.5) / PW;
      const cosT = a * 2 - 1;
      const sinT = Math.sqrt(Math.max(0, 1 - cosT * cosT));
      const ray: V3 = [0, 0, 1];
      const light: V3 = [sinT, 0, cosT];
      const tau = a * 6;
      let cpu: number[];
      if (row === 0) {
        const t = vdbFrameMieTexel(cosT);
        cpu = [vdbFramePhaseWeight(mie, ray, light), t[0], t[1], t[2]];
      } else if (row === 1) {
        const rgb = vdbFramePhaseColor(mieRgb, [1, 1, 1], ray, light);
        cpu = [rgb[2], ...vdbFramePhaseColor(dual, exponentSpectrum, ray, light)];
      } else if (row === 2) {
        cpu = [vdbFramePhaseWeightScaled(dual, cosT, 0.5), vdbFramePhaseWeightScaled(mie, cosT, 0.25), vdbFrameOctaveLight(octaves, tau, dual, cosT), vdbFrameOctaveLight(octaves, tau, null, 0)];
      } else if (row === 3) {
        const depths: V3 = [tau, tau * 0.3, tau * 0.6];
        const t = vdbFrameShadowTransmittance(depths, 0.42, octaves);
        cpu = [vdbFrameShadowMarchDepth(depths, 0.42, octaves), t[0], t[1], vdbFrameShadowTotal(t[1], octaves)];
      } else if (row === 4) {
        const f = a * 8;
        cpu = [vdbFrameTwoSidedDiffusion(0, f, a * 0.5, 8 - f), vdbFrameTwoSidedDiffusion(0, f, 0, a * 2), vdbFrameTwoSidedDiffusion(0.9, a * 0.1, 0, 0.2), vdbFrameOctaveLight([8, 0.7, 0.9, 0.3], tau, mie, cosT)];
      } else {
        const shadow: V2 = [Math.exp(-tau), 1 / (tau + 1)];
        const lit = vdbFrameShadowedLightColor(dual, exponentSpectrum, octaves, ray, light, shadow);
        const plain = vdbFrameShadowedLightColor(dual, [1, 1, 0], [0, 0, 0, 0], ray, light, [shadow[0], Math.exp(-tau * 0.5)]);
        cpu = [lit[0], lit[1], lit[2], plain[1]];
      }
      // f32 vs f64, relative: the Mie peak (~4e3) and HG near-forward peaks are large. The table reads (rows 0–1)
      // get 1e-3: the GPU's f32 acos moves the texel coordinate (u = √(acos(c)/π) · 2047), and the table is steep
      // around the fogbow (2.6e-4 seen at cos −0.83 on SwiftShader).
      const rel = row <= 1 ? 1e-3 : 2e-4;
      for (let c = 0; c < 4; c++) accumulate(stats, gpu[(row * PW + x) * 4 + c]!, cpu[c]!, 1e-5, rel, { row, x, c });
    }
  }
  void vdbFrameHenyeyGreensteinColor;
  void vdbFrameShadowOpticalDepth;
  return finish(stats);
}

/** His white furnace (utils/furnace.js): a uniform equirect environment of radiance `radiance` (linear, float). */
export function furnaceEnvironment(radiance = 0.5): THREE.DataTexture {
  const w = 64;
  const h = 32;
  const data = new Float32Array(w * h * 4);
  for (let i = 0; i < w * h; i++) data.set([radiance, radiance, radiance, 1.0], i * 4);
  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.FloatType);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.colorSpace = THREE.LinearSRGBColorSpace;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}
