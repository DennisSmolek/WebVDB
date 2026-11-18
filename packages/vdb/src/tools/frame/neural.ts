// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/frameNeuralChunks.js L14–129 (weight layout, dense layers, sampleLevel,
//   getLocalInputs, neuralLight), src/three-openvdb/frame/webgpu/FrameGridMipMaterial.js L10–66 (the 2x2x2
//   box mip of the last grid cascade), src/three-openvdb/frame/webgpu/FrameRenderer.js L647–699
//   (renderGridMips: level sizes and atlas layouts).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: TypeScript; f64 CPU twins of the TSL (test oracles); the network is a value
// (VDBFrameNeuralNetwork) so the weights module (./neural-weights.ts) can be swapped or left out.
// Divergences: DIVERGES.md (K3.4 rows).

/**
 * mjurczyk's `multipleScattering: 'neural'` (K3.4, R1 §5): a 20 → 32 → 32 → 1 MLP per shadow grid
 * voxel and light slot, run in the transmittance pass. Its inputs are 6 box mip levels of the last
 * shadow grid cascade sampled at the voxel and at ±h·2^k along the light, the slot's direct optical
 * depth and an albedo term; its output is the scattered light (total − direct) of the slot. This
 * module is engine-free: the network type, the forward pass, the feature sampling and the grid mip
 * pass as CPU twins of `@pmndrs/vdb/three` `frame/neural.ts` and `grid-mip-material.ts`.
 */

import type { Vec2, Vec3 } from "./index.js";
import { vdbFrameSampleGrid, vdbFrameSampleTiles, vdbFrameGridUvw, type VDBFrameAtlas, type VDBFrameGridLayout } from "./grid.js";

/** A neural indirect network: weights in his vec4 layout (w1 rows, b1, w2 rows, b2, w3, b3.x). */
export interface VDBFrameNeuralNetwork {
  /** A label for errors and signatures. */
  name?: string;
  /** Mip levels of the inputs (his 6). */
  levels: number;
  /** Inputs (his 20 = 3 × levels + 2). */
  inputs: number;
  /** Units of each hidden layer (his 32; a multiple of 4). */
  hidden: number;
  weights: Float32Array;
}

/** vec4 offsets of the weight layout (frameNeuralChunks.js L20–26). */
export function vdbFrameNeuralLayout(network: Pick<VDBFrameNeuralNetwork, "inputs" | "hidden">) {
  const inputVectors = network.inputs / 4;
  const hiddenVectors = network.hidden / 4;
  const w1 = 0;
  const b1 = network.hidden * inputVectors;
  const w2 = b1 + hiddenVectors;
  const b2 = w2 + network.hidden * hiddenVectors;
  const w3 = b2 + hiddenVectors;
  const b3 = w3 + hiddenVectors;
  return { inputVectors, hiddenVectors, w1, b1, w2, b2, w3, b3, vectors: b3 + 1 };
}

/**
 * Checks a network the shaders can compile: 3 × levels + 2 inputs in vec4s, hidden a multiple of 4,
 * the weight count of the layout. Returns a reason, or null when it fits.
 */
export function vdbFrameNeuralCheck(network: VDBFrameNeuralNetwork): string | null {
  if (!(network.levels >= 2) || network.inputs !== 3 * network.levels + 2) return `inputs must be 3 × levels + 2 (levels ${network.levels}, inputs ${network.inputs})`;
  if (network.inputs % 4 !== 0 || network.hidden % 4 !== 0 || !(network.hidden > 0)) return `inputs and hidden must be multiples of 4 (inputs ${network.inputs}, hidden ${network.hidden})`;
  if (network.levels !== 6) return `the shader packs the inputs of 6 mip levels (levels ${network.levels})`;
  const { vectors } = vdbFrameNeuralLayout(network);
  if (network.weights.length !== vectors * 4) return `expected ${vectors * 4} weights, got ${network.weights.length}`;
  return null;
}

/**
 * The MLP forward pass of `neuralLight` (f64): two ReLU layers, a linear output, then exp(min(o, 4)).
 * `inputs` is the 20-vector in the shader's order (see {@link vdbFrameNeuralInputs}).
 */
export function vdbFrameNeuralForward(network: VDBFrameNeuralNetwork, inputs: ArrayLike<number>): number {
  const L = vdbFrameNeuralLayout(network);
  const w = network.weights;
  const dense = (x: ArrayLike<number>, weightOffset: number, biasOffset: number, units: number): Float64Array => {
    const out = new Float64Array(units);
    for (let j = 0; j < units; j++) {
      const row = (weightOffset + j * (x.length / 4)) * 4;
      let s = w[(biasOffset + (j >> 2)) * 4 + (j & 3)]!;
      for (let i = 0; i < x.length; i++) s += w[row + i]! * x[i]!;
      out[j] = Math.max(s, 0);
    }
    return out;
  };
  const h1 = dense(inputs, L.w1, L.b1, network.hidden);
  const h2 = dense(h1, L.w2, L.b2, network.hidden);
  let o = w[L.b3 * 4]!;
  for (let i = 0; i < network.hidden; i++) o += w[L.w3 * 4 + i]! * h2[i]!;
  return Math.exp(Math.min(o, 4.0));
}

/** One box mip level of the neural inputs: an RG atlas, its voxels per side and tile columns. */
export interface VDBFrameNeuralMip {
  atlas: VDBFrameAtlas;
  resolution: number;
  columns: number;
}

/**
 * sampleLevel: (extinction, absorption) of a mip level at a world position, in the last cascade (level 0 is
 * the grid atlas itself); linear filter, edge texels clamped; 0 outside the cascade's [0, 1]³.
 */
export function vdbFrameNeuralSampleLevel(grid: VDBFrameGridLayout, density: VDBFrameAtlas, mips: readonly VDBFrameNeuralMip[], level: number, world: Vec3): Vec2 {
  const cascade = grid.count - 1;
  const uvw = vdbFrameGridUvw(grid, cascade, world);
  if (Math.min(uvw[0], uvw[1], uvw[2]) < 0 || Math.max(uvw[0], uvw[1], uvw[2]) > 1) return [0, 0];
  const s = level === 0 ? vdbFrameSampleGrid(density, grid, cascade, uvw) : vdbFrameSampleTiles(mips[level - 1]!.atlas, mips[level - 1]!.resolution, 0, mips[level - 1]!.columns, uvw);
  return [s[0]!, s[1]!];
}

/** The largest voxel side of the last cascade (`unit` of his chunks). */
export const vdbFrameNeuralUnit = (grid: VDBFrameGridLayout): number => {
  const v = grid.gridVoxel[grid.count - 1]!;
  return Math.max(v[0], v[1], v[2]);
};

/**
 * The 20 network inputs at a world position for a light direction (getLocalInputs + neuralLight):
 * log(1 + σ h 2^k) at the position (k = 0..5), toward the light, away from it, then
 * log(1 + direct optical depth) and √(clamp(absorption / extinction)) of level 1 (0 where level 1 is empty).
 */
export function vdbFrameNeuralInputs(
  sampleLevel: (level: number, world: Vec3) => Vec2,
  levels: number,
  h: number,
  world: Vec3,
  direction: Vec3,
  opticalDepth: number,
): number[] {
  const samples = Array.from({ length: levels }, (_, k) => sampleLevel(k, world));
  const local = samples.map((v, k) => Math.log(v[0] * (h * 2 ** k) + 1));
  const albedoSample = samples[1]!;
  const albedoInput = albedoSample[0] > 0 ? Math.sqrt(Math.min(1, Math.max(0, albedoSample[1] / Math.max(albedoSample[0], 1e-20)))) : 0;
  const toward: number[] = [];
  const away: number[] = [];
  for (let k = 0; k < levels; k++) {
    const length = h * 2 ** k;
    toward.push(Math.log(sampleLevel(k, [world[0] + direction[0] * length, world[1] + direction[1] * length, world[2] + direction[2] * length])[0] * length + 1));
    away.push(Math.log(sampleLevel(k, [world[0] - direction[0] * length, world[1] - direction[1] * length, world[2] - direction[2] * length])[0] * length + 1));
  }
  return [...local, ...toward, ...away, Math.log(opticalDepth + 1), albedoInput];
}

/** neuralLight: the network's scattered light (total − direct) of a slot at a world position. */
export function vdbFrameNeuralLight(network: VDBFrameNeuralNetwork, grid: VDBFrameGridLayout, density: VDBFrameAtlas, mips: readonly VDBFrameNeuralMip[], world: Vec3, direction: Vec3, opticalDepth: number): number {
  const sample = (level: number, p: Vec3) => vdbFrameNeuralSampleLevel(grid, density, mips, level, p);
  return vdbFrameNeuralForward(network, vdbFrameNeuralInputs(sample, network.levels, vdbFrameNeuralUnit(grid), world, direction, opticalDepth));
}

/** Atlas layout of a z-sliced grid (his core/frameLayout.js getAtlasLayout; the three adapter's `layout.ts`). */
export function vdbFrameTileAtlasLayout(width: number, height: number, depth: number): { columns: number; width: number; height: number } {
  const columns = Math.ceil(Math.sqrt((depth * height) / width));
  const rows = Math.ceil(depth / columns);
  return { columns, width: columns * width, height: rows * height };
}

/**
 * The grid mip pass (FrameGridMipMaterial) on the CPU: each texel of level k (resolution / 2^k voxels
 * per side, z slices in an atlas) is the mean of the 2 × 2 × 2 texels of level k − 1 (edges clamped).
 * `source` is the grid atlas (level 0: the last cascade's tiles from `tile`) or the level before.
 * Returns the RG atlas (bottom-first rows, as the shaders).
 */
export function vdbFrameGridMipLevel(source: VDBFrameAtlas, sourceResolution: number, sourceColumns: number, sourceTile: number, resolution: number): VDBFrameNeuralMip {
  const layout = vdbFrameTileAtlasLayout(resolution, resolution, resolution);
  const data = new Float32Array(layout.width * layout.height * 2);
  const last = sourceResolution - 1;
  const load = (x: number, y: number, z: number): Vec2 => {
    const tile = sourceTile + Math.min(z, last);
    const tx = (tile % sourceColumns) * sourceResolution + Math.min(x, last);
    const ty = Math.floor(tile / sourceColumns) * sourceResolution + Math.min(y, last);
    const o = (ty * source.width + tx) * source.channels;
    return [source.data[o]!, source.data[o + 1]!];
  };
  for (let py = 0; py < layout.height; py++) {
    for (let px = 0; px < layout.width; px++) {
      const tx = Math.floor(px / resolution);
      const ty = Math.floor(py / resolution);
      const z = ty * layout.columns + tx;
      const x = px - tx * resolution;
      const y = py - ty * resolution;
      let s0 = 0;
      let s1 = 0;
      for (let i = 0; i < 8; i++) {
        const v = load(x * 2 + (i & 1), y * 2 + ((i >> 1) & 1), z * 2 + ((i >> 2) & 1));
        s0 += v[0];
        s1 += v[1];
      }
      data[(py * layout.width + px) * 2] = s0 / 8;
      data[(py * layout.width + px) * 2 + 1] = s1 / 8;
    }
  }
  return { atlas: { data, width: layout.width, height: layout.height, channels: 2 }, resolution, columns: layout.columns };
}

/** Voxels per side of mip level k of a grid of `resolution` voxels (his `max(1, resolution >> k)`). */
export const vdbFrameNeuralMipResolution = (resolution: number, level: number): number => Math.max(1, resolution >> level);
