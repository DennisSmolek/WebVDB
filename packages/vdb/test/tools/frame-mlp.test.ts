import { describe, expect, it } from "vitest";
import {
  vdbFrameMlpCheck,
  vdbFrameMlpForward,
  vdbFrameMlpFromNeural,
  vdbFrameMlpInputs,
  vdbFrameMlpMipSizes,
  vdbFrameMlpShape,
  vdbFrameNeuralForward,
  vdbFrameNeuralInputs,
  vdbFrameNeuralMipResolution,
  vdbFrameNeuralWeights,
  type VDBFrameMlp,
} from "../../src/tools/internal.js";

// K6.3 (design §3.6, Q5): the core's experimental generic MLP evaluator runs his 'neural' network unchanged.
describe("K6.3 generic MLP (tools/frame/mlp.ts)", () => {
  let seed = 3;
  const r = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);

  it("his network as a generic MLP gives his forward pass (the same numbers, another layout)", () => {
    const mlp = vdbFrameMlpFromNeural(vdbFrameNeuralWeights);
    expect(vdbFrameMlpCheck(mlp)).toBeNull();
    expect(mlp.layers.map((l) => [l.inputs, l.outputs, l.activation])).toEqual([
      [20, 32, "relu"],
      [32, 32, "relu"],
      [32, 1, "linear"],
    ]);
    for (let i = 0; i < 200; i++) {
      const x = Array.from({ length: 20 }, () => r() * 3);
      expect(vdbFrameMlpForward(mlp, x)).toBeCloseTo(vdbFrameNeuralForward(vdbFrameNeuralWeights, x), 12);
    }
    // fail-before: one weight moved changes the output
    const moved: VDBFrameMlp = { ...mlp, weights: Float32Array.from(mlp.weights, (w, i) => (i === 7 ? w + 0.5 : w)) };
    const x = Array.from({ length: 20 }, () => 1);
    expect(vdbFrameMlpForward(moved, x)).not.toBeCloseTo(vdbFrameMlpForward(mlp, x), 6);
  });

  it("the inputs equal his feature order (local, toward, away, log(1 + τ), albedo)", () => {
    const sample = (k: number, p: [number, number, number]): [number, number] => [Math.abs(Math.sin(p[0] * 3 + k)) * 2, Math.abs(Math.cos(p[1] + k)) * 0.7];
    for (let i = 0; i < 50; i++) {
      const world: [number, number, number] = [r(), r(), r()];
      const d = [r() - 0.5, r() - 0.5, r() - 0.5];
      const l = Math.hypot(...d);
      const dir: [number, number, number] = [d[0]! / l, d[1]! / l, d[2]! / l];
      const ours = vdbFrameMlpInputs(sample, 6, 0.1, world, dir, 2.5);
      const his = vdbFrameNeuralInputs(sample, 6, 0.1, world, dir, 2.5);
      ours.forEach((v, k) => expect(v).toBeCloseTo(his[k]!, 12));
    }
  });

  it("the feature mips have his level sizes max(1, R >> k), the last repeating past the chain", () => {
    for (const R of [2, 7, 16, 24, 63, 64, 100]) {
      const sizes = vdbFrameMlpMipSizes(R);
      for (let level = 1; level <= 5; level++) expect(sizes[Math.min(level - 1, sizes.length - 1)]).toBe(vdbFrameNeuralMipResolution(R, level));
    }
  });

  it("checks shapes and packs the shape uniform", () => {
    const mlp = vdbFrameMlpFromNeural(vdbFrameNeuralWeights);
    expect(vdbFrameMlpCheck({ ...mlp, levels: 5 })).toMatch(/3 × levels \+ 2/);
    expect(vdbFrameMlpCheck({ ...mlp, weights: mlp.weights.slice(1) })).toMatch(/weights/);
    expect(vdbFrameMlpCheck({ ...mlp, layers: [{ inputs: 20, outputs: 65, activation: "relu" }] })).toMatch(/widths/);
    expect(Array.from(vdbFrameMlpShape(mlp).slice(0, 16))).toEqual([3, 6, 1, 20, 20, 32, 1, 0, 32, 32, 1, 672, 32, 1, 0, 1728]);
  });
});
