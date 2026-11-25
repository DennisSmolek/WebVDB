// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/frameNeuralChunks.js L14–129 (dense layers, getLocalInputs, neuralLight: the
//   inputs of his 'neural' network), src/three-openvdb/frame/webgpu/FrameGridMipMaterial.js L10–66 (the 2x2x2 box mip).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: a generic small-MLP description (layer shapes as data, any activation sequence) instead of his fixed
// 20 → 32 → 32 → 1 vec4 layout; the mips are levels of one 3D texture. CPU twins of the core's 'mlp' kernels (K6.3).
// Divergences: DIVERGES.md (K6.3 rows).

/**
 * K6.3, design §3.6 (Q5): the core's `multipleScattering: 'mlp'` is an EXPERIMENTAL generic small-MLP evaluator. A
 * network is data: its layer shapes become a uniform and its weights a storage buffer, so a new network needs no shader
 * change. The inputs are his `'neural'` features at 3·L + 2 values (L feature levels): log(1 + σ h 2^k) at the voxel,
 * h 2^k toward the light and h 2^k away from it for k < L, then log(1 + direct optical depth) and √(absorption /
 * extinction) of level 1. The output is the scattered light (total − direct) of a light slot. His trained network
 * ({@link vdbFrameMlpFromNeural} of `vdbFrameNeuralWeights`) is the first network, not the definition.
 */

import type { Vec2, Vec3 } from "./index.js";
import type { VDBFrameNeuralNetwork } from "./neural.js";
import type { VDBFrameTexture3d } from "./stack.js";
import { vdbFrameTrilinear } from "./stack.js";
import { vdbFrameGridUvw, type VDBFrameGridLayout } from "./grid.js";

/** Most layers and the widest layer of the core evaluator (two 64-wide locals). */
export const VDB_FRAME_MLP_MAX_LAYERS = 8;
export const VDB_FRAME_MLP_MAX_WIDTH = 64;
/** Feature levels of the core's mip chain (level 0 the grid, mips 1–5). */
export const VDB_FRAME_MLP_MAX_LEVELS = 6;

/** One dense layer: `outputs` = activation(W · x + b), W row-major (row j = output j). */
export interface VDBFrameMlpLayer {
  inputs: number;
  outputs: number;
  activation: "relu" | "linear";
}

/** A network for the core's experimental `'mlp'` multiple scattering. */
export interface VDBFrameMlp {
  /** A label for warnings and signatures. */
  name?: string;
  /** Feature levels L (2–6): the first layer takes 3 L + 2 inputs. */
  levels: number;
  layers: readonly VDBFrameMlpLayer[];
  /** The output transform of the last layer's first output: `'exp'` = exp(min(o, 4)) (his), `'linear'` = o. */
  output: "exp" | "linear";
  /** Per layer, in order: outputs × inputs weights (row-major), then the outputs biases. */
  weights: Float32Array;
}

/** The weight offset of each layer (its row-major W; the biases follow at offset + outputs × inputs). */
export function vdbFrameMlpOffsets(mlp: Pick<VDBFrameMlp, "layers">): number[] {
  const offsets: number[] = [];
  let at = 0;
  for (const l of mlp.layers) {
    offsets.push(at);
    at += l.outputs * (l.inputs + 1);
  }
  return offsets;
}

/** Checks a network the core evaluator runs; returns a reason, or null when it fits. */
export function vdbFrameMlpCheck(mlp: VDBFrameMlp): string | null {
  if (!Number.isInteger(mlp.levels) || mlp.levels < 2 || mlp.levels > VDB_FRAME_MLP_MAX_LEVELS) return `levels must be 2 to ${VDB_FRAME_MLP_MAX_LEVELS} (got ${mlp.levels})`;
  if (mlp.layers.length < 1 || mlp.layers.length > VDB_FRAME_MLP_MAX_LAYERS) return `1 to ${VDB_FRAME_MLP_MAX_LAYERS} layers (got ${mlp.layers.length})`;
  if (mlp.layers[0]!.inputs !== 3 * mlp.levels + 2) return `the first layer takes 3 × levels + 2 = ${3 * mlp.levels + 2} inputs (got ${mlp.layers[0]!.inputs})`;
  for (let i = 0; i < mlp.layers.length; i++) {
    const l = mlp.layers[i]!;
    if (!(l.inputs >= 1 && l.inputs <= VDB_FRAME_MLP_MAX_WIDTH && l.outputs >= 1 && l.outputs <= VDB_FRAME_MLP_MAX_WIDTH)) return `layer ${i}: widths must be 1 to ${VDB_FRAME_MLP_MAX_WIDTH}`;
    if (i > 0 && l.inputs !== mlp.layers[i - 1]!.outputs) return `layer ${i} takes ${l.inputs} inputs, the layer before gives ${mlp.layers[i - 1]!.outputs}`;
    if (l.activation !== "relu" && l.activation !== "linear") return `layer ${i}: activation must be 'relu' or 'linear'`;
  }
  if (mlp.output !== "exp" && mlp.output !== "linear") return "output must be 'exp' or 'linear'";
  const total = mlp.layers.reduce((s, l) => s + l.outputs * (l.inputs + 1), 0);
  if (mlp.weights.length !== total) return `expected ${total} weights, got ${mlp.weights.length}`;
  return null;
}

/**
 * His `'neural'` network (vec4 layout: w1 rows, b1, w2 rows, b2, w3, b3.x) as a generic MLP: the same numbers
 * (his layout is row-major with the biases after each W; the 3 pad floats after b3 are dropped).
 */
export function vdbFrameMlpFromNeural(network: VDBFrameNeuralNetwork): VDBFrameMlp {
  const n = network.inputs;
  const h = network.hidden;
  const total = h * (n + 1) + h * (h + 1) + (h + 1);
  return {
    name: network.name ?? "neural",
    levels: network.levels,
    layers: [
      { inputs: n, outputs: h, activation: "relu" },
      { inputs: h, outputs: h, activation: "relu" },
      { inputs: h, outputs: 1, activation: "linear" },
    ],
    output: "exp",
    weights: network.weights.slice(0, total),
  };
}

/** The shape uniform of the core kernel (`VdbFrameMlp`: info, then (inputs, outputs, activation, offset) per layer). */
export function vdbFrameMlpShape(mlp: VDBFrameMlp): Uint32Array {
  const u = new Uint32Array(4 + 4 * VDB_FRAME_MLP_MAX_LAYERS);
  u[0] = mlp.layers.length;
  u[1] = mlp.levels;
  u[2] = mlp.output === "exp" ? 1 : 0;
  u[3] = 3 * mlp.levels + 2;
  const offsets = vdbFrameMlpOffsets(mlp);
  mlp.layers.forEach((l, i) => {
    u[4 + 4 * i] = l.inputs;
    u[5 + 4 * i] = l.outputs;
    u[6 + 4 * i] = l.activation === "relu" ? 1 : 0;
    u[7 + 4 * i] = offsets[i]!;
  });
  return u;
}

/** The forward pass (f64): the last layer's first output through the output transform. */
export function vdbFrameMlpForward(mlp: VDBFrameMlp, inputs: ArrayLike<number>): number {
  const offsets = vdbFrameMlpOffsets(mlp);
  const w = mlp.weights;
  let x = Array.from(inputs);
  mlp.layers.forEach((l, li) => {
    const at = offsets[li]!;
    const bias = at + l.outputs * l.inputs;
    const out: number[] = [];
    for (let j = 0; j < l.outputs; j++) {
      let s = w[bias + j]!;
      for (let i = 0; i < l.inputs; i++) s += w[at + j * l.inputs + i]! * (x[i] ?? 0);
      out.push(l.activation === "relu" ? Math.max(s, 0) : s);
    }
    x = out;
  });
  return mlp.output === "exp" ? Math.exp(Math.min(x[0]!, 4)) : x[0]!;
}

/** Voxels per side of the core's feature mip chain: mip j has max(1, (R >> 1) >> j), at most 5 mips (levels 1–5). */
export function vdbFrameMlpMipSizes(resolution: number): number[] {
  const base = Math.max(1, resolution >> 1);
  const sizes = [base];
  while (sizes.length < VDB_FRAME_MLP_MAX_LEVELS - 1 && sizes[sizes.length - 1]! > 1) sizes.push(Math.max(1, sizes[sizes.length - 1]! >> 1));
  return sizes;
}

/**
 * The `gridMips` kernel (his FrameGridMipMaterial box) at texel `p` of a mip of side `size`: the mean of the 2 × 2 × 2
 * source texels 2p + i (each axis clamped to `sourceSize` − 1) of `source` read at `origin` (.xy: extinction, absorption).
 */
export function vdbFrameMlpMipTexel(source: VDBFrameTexture3d, origin: readonly number[], sourceSize: readonly number[], p: Vec3): Vec2 {
  let a = 0;
  let b = 0;
  for (let i = 0; i < 8; i++) {
    const q = [p[0] * 2 + (i & 1), p[1] * 2 + ((i >> 1) & 1), p[2] * 2 + ((i >> 2) & 1)].map((v, k) => Math.min(v, sourceSize[k]! - 1) + origin[k]!);
    const at = (q[0]! + (q[1]! + q[2]! * source.size[1]) * source.size[0]) * source.channels;
    a += source.data[at]!;
    b += source.data[at + 1]!;
  }
  return [a / 8, b / 8];
}

/**
 * sampleLevel on the core's textures: level 0 the last cascade in the shadow stack (`grid`, its origin), level k ≥ 1
 * mip min(k − 1, mips − 1) (`mips[j]`, its own texture), hardware trilinear clamped to the half-texel box; 0 outside.
 */
export function vdbFrameMlpSampleLevel(layout: VDBFrameGridLayout, grid: VDBFrameTexture3d, gridOrigin: readonly number[], mips: readonly VDBFrameTexture3d[], level: number, world: Vec3): Vec2 {
  const cascade = layout.count - 1;
  const uvw = vdbFrameGridUvw(layout, cascade, world);
  if (uvw.some((v) => v < 0 || v > 1)) return [0, 0];
  if (level === 0) {
    const r = layout.resolution;
    const s = vdbFrameTrilinear(grid, uvw.map((v, k) => gridOrigin[k]! + Math.min(Math.max(v * r, 0.5), r - 0.5)));
    return [s[0]!, s[1]!];
  }
  const t = mips[Math.min(level - 1, mips.length - 1)]!;
  const s = vdbFrameTrilinear(t, uvw.map((v, k) => Math.min(Math.max(v * t.size[k]!, 0.5), t.size[k]! - 0.5)));
  return [s[0]!, s[1]!];
}

/** The 3 L + 2 inputs at a world position for a unit direction to the light (his getLocalInputs + neuralLight). */
export function vdbFrameMlpInputs(sample: (level: number, world: Vec3) => Vec2, levels: number, h: number, world: Vec3, direction: Vec3, opticalDepth: number): number[] {
  const input = new Array<number>(3 * levels + 2).fill(0);
  for (let k = 0; k < levels; k++) {
    const reach = h * 2 ** k;
    const s = sample(k, world);
    input[k] = Math.log(s[0] * reach + 1);
    if (k === 1 && s[0] > 0) input[3 * levels + 1] = Math.sqrt(Math.min(1, Math.max(0, s[1] / Math.max(s[0], 1e-20))));
    input[levels + k] = Math.log(sample(k, [world[0] + direction[0] * reach, world[1] + direction[1] * reach, world[2] + direction[2] * reach])[0] * reach + 1);
    input[2 * levels + k] = Math.log(sample(k, [world[0] - direction[0] * reach, world[1] - direction[1] * reach, world[2] - direction[2] * reach])[0] * reach + 1);
  }
  input[3 * levels] = Math.log(opticalDepth + 1);
  return input;
}
