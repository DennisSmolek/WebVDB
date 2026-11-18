// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/frameNeuralChunks.js L1–131 (all: weight layout, dense layers,
//   sampleLevel, getLocalInputs, neuralLight).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript; the network (weights and sizes) is a parameter (VDBFrameNeuralNetwork) instead of
// a module import, so the weights module can be swapped or left out. CPU twin: tools/frame/neural.ts.
// Divergences: DIVERGES.md (K3.4 rows).

/**
 * multipleScattering `'neural'` (K3.4): the scattered light of a shadow slot from a small network
 * (`@pmndrs/vdb/tools` `vdbFrameNeuralWeights`, his `scripts/neural-indirect`). Level 0 is the last
 * cascade of the grid atlas, levels 1 and up are the atlases of {@link VDBFrameGridMipMaterial}. Values
 * outside the last cascade are 0. The weights are constants in the shader (one WGSL function,
 * `frameNeuralLight`, called per slot).
 */

import * as TSL from "three/tsl";
import type { VDBTslNode } from "../grid-node.js";
import { vdbFrameNeuralLayout, type VDBFrameNeuralNetwork } from "../../tools/frame/neural.js";
import { maxComponent, minComponent, type GridNodes, type GridUniforms } from "./chunks.js";

const { Fn, clamp, dot, exp, float, floor, log, max, min, select, sqrt, vec2, vec4 } = TSL as unknown as Record<string, VDBTslNode>;

type N = VDBTslNode;

/**
 * Network nodes over the grid uniforms and nodes (createGridNodes), the grid atlas texture node, the mip
 * texture nodes (levels 1 and up) and their atlas layout uniforms (columns, width, height).
 */
export const createNeuralNodes = (network: VDBFrameNeuralNetwork, gridUniforms: GridUniforms, grid: GridNodes, densityMap: N, mipMaps: N[], mipLayouts: N[]) => {
  const { frameGridCount, frameGridVoxel, frameGridResolution } = gridUniforms;
  const layout = vdbFrameNeuralLayout(network);
  const levels = network.levels;
  const weights = network.weights;
  const weight = (index: number): N => vec4(weights[index * 4]!, weights[index * 4 + 1]!, weights[index * 4 + 2]!, weights[index * 4 + 3]!);
  const lastCascade = () => frameGridCount.sub(1);

  // NOTE Dense layer with ReLU. inputs: vec4 nodes. Weights are constants in the shader.
  const dense = (inputs: N[], weightOffset: number, biasOffset: number): N[] => {
    const outputs: N[] = [];

    for (let group = 0; group < layout.hiddenVectors; group++) {
      const bias = weight(biasOffset + group);
      const values = [0, 1, 2, 3].map((component) => {
        const row = weightOffset + (group * 4 + component) * inputs.length;

        return inputs.reduce((sum: N, input, i) => sum.add(dot(weight(row + i), input)), bias.element(component));
      });

      outputs.push(max(vec4(...values), vec4(0.0)).toVar());
    }

    return outputs;
  };

  // NOTE (extinction, absorption) of a level at a world position. Linear filter, edge texels clamped.
  const sampleLevel = (level: number, worldPosition: N): N => {
    const cascade = lastCascade();
    const uvw = grid.gridUvw(cascade, worldPosition).toVar();
    const inside = minComponent(uvw).greaterThanEqual(0.0).and(maxComponent(uvw).lessThanEqual(1.0));
    let value: N;

    if (level === 0) {
      value = grid.sampleGrid(densityMap, cascade, uvw).rg;
    } else {
      const resolution = max(floor(frameGridResolution.div(2 ** level)), 1.0);
      const mipLayout = mipLayouts[level - 1]!;

      value = grid.sampleTiles(mipMaps[level - 1], resolution, float(0.0), mipLayout.x, mipLayout.yz, uvw).rg;
    }

    return select(inside, value, vec2(0.0));
  };

  const unit = () => maxComponent(frameGridVoxel.element(lastCascade()));

  // NOTE Inputs that do not depend on the light: (local 0 to 3), (local 4, local 5, albedo input, 0)
  const getLocalInputs = (worldPosition: N): [N, N] => {
    const h = unit().toVar();
    const samples = Array.from({ length: levels }, (_, level) => sampleLevel(level, worldPosition).toVar());
    const local = samples.map((value, level) => log(value.x.mul(h.mul(2 ** level)).add(1.0)));
    // NOTE The albedo input uses the level 1 sample
    const albedoSample = samples[1]!;
    const albedoInput = select(albedoSample.x.greaterThan(0.0), sqrt(clamp(albedoSample.y.div(max(albedoSample.x, 1e-20)), 0.0, 1.0)), float(0.0));

    return [vec4(local[0], local[1], local[2], local[3]).toVar(), vec4(local[4], local[5], albedoInput, 0.0).toVar()];
  };

  // NOTE Scattered light (total - direct) for each unit of light. direction: unit direction to the light,
  // opticalDepth: direct optical depth of the slot march.
  const neuralLight = Fn(([worldPosition, direction, opticalDepth, local0, local1]: N[]) => {
    const h = unit().toVar();
    const toward: N[] = [];
    const away: N[] = [];

    for (let level = 0; level < levels; level++) {
      const length = h.mul(2 ** level).toVar();

      toward.push(log(sampleLevel(level, worldPosition!.add(direction!.mul(length))).x.mul(length).add(1.0)));
      away.push(log(sampleLevel(level, worldPosition!.sub(direction!.mul(length))).x.mul(length).add(1.0)));
    }

    const inputs = [
      local0,
      vec4(local1!.x, local1!.y, toward[0], toward[1]),
      vec4(toward[2], toward[3], toward[4], toward[5]),
      vec4(away[0], away[1], away[2], away[3]),
      vec4(away[4], away[5], log(opticalDepth!.add(1.0)), local1!.z),
    ].map((input) => input!.toVar());

    const hidden1 = dense(inputs, layout.w1, layout.b1);
    const hidden2 = dense(hidden1, layout.w2, layout.b2);
    const output = hidden2.reduce((sum: N, value, i) => sum.add(dot(weight(layout.w3 + i), value)), weight(layout.b3).x);

    return exp(min(output, 4.0));
  }).setLayout({
    name: "frameNeuralLight",
    type: "float",
    inputs: [
      { name: "worldPosition", type: "vec3" },
      { name: "direction", type: "vec3" },
      { name: "opticalDepth", type: "float" },
      { name: "local0", type: "vec4" },
      { name: "local1", type: "vec4" },
    ],
  });

  return { getLocalInputs, neuralLight, sampleLevel };
};
