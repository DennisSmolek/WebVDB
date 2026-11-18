import { PerspectiveCamera } from "three";
import { describe, expect, it } from "vitest";
import {
  vdbFrameBoxSteps,
  vdbFrameDistanceFade,
  vdbFrameFroxel,
  vdbFrameFroxelSlice,
  vdbFrameHenyeyGreenstein,
  vdbFrameInjectSlice,
  vdbFrameIntegrate,
  vdbFrameNoise,
  vdbFramePhaseWeight,
  vdbFrameShadowAbsorption,
  vdbFrameShadowDepth,
  vdbFrameShadowMarch,
  vdbFrameShadowStep,
  vdbFrameShadowTransmittance,
  vdbFrameShadowedLight,
  vdbFrameSliceDepth,
  vdbFrameSliceRange,
  vdbFrameSrgbBlend,
  vdbFrameViewPosition,
  vdbSrgbEotf,
  vdbSrgbOetf,
} from "../../src/tools/internal.js";

// K3.1: the CPU twins of the froxel pipeline maths (tools/frame). These are the oracles of the TSL chunks
// (the e2e k3-frame chunk probe and integrate twin compare the GPU passes with them), so each test pins a
// property of his formula that a port error would break.

describe("froxel slice mapping", () => {
  const range: [number, number] = [0.5, 300];

  it("slice depths are logarithmic from near to far, and getFroxelSlice inverts them", () => {
    expect(vdbFrameSliceDepth(range, 0)).toBeCloseTo(0.5, 12);
    expect(vdbFrameSliceDepth(range, 1)).toBeCloseTo(300, 9);
    // Equal ratios between consecutive slice boundaries.
    const D = 24;
    const r0 = vdbFrameSliceDepth(range, 1 / D) / vdbFrameSliceDepth(range, 0);
    for (let k = 1; k < D; k++) {
      expect(vdbFrameSliceDepth(range, (k + 1) / D) / vdbFrameSliceDepth(range, k / D)).toBeCloseTo(r0, 10);
      expect(vdbFrameFroxelSlice(vdbFrameSliceDepth(range, k / D), range, D)).toBeCloseTo(k, 9);
    }
    // Depths before the near plane map to slice 0.
    expect(vdbFrameFroxelSlice(0.1, range, D)).toBe(0);
  });

  it("slice ranges cover the volume's depth range, clamped to the grid", () => {
    const D = 24;
    const near = vdbFrameSliceDepth(range, 5.5 / D);
    const far = vdbFrameSliceDepth(range, 9.2 / D);
    expect(vdbFrameSliceRange(near, far, range, D)).toEqual({ first: 5, count: 6 }); // 5..10 (ceil of 9.2)
    expect(vdbFrameSliceRange(0.01, 1e6, range, D)).toEqual({ first: 0, count: 24 });
  });

  it("view positions lie on the pixel ray at the requested view depth", () => {
    const camera = new PerspectiveCamera(60, 16 / 9, 0.01, 10000);
    camera.updateProjectionMatrix();
    const inv = camera.projectionMatrixInverse.elements;
    for (const uv of [
      [0.5, 0.5],
      [0, 0],
      [1, 0.25],
    ] as [number, number][]) {
      for (const depth of [0.5, 7, 250]) {
        const p = vdbFrameViewPosition(inv, uv, depth);
        expect(p[2]).toBeCloseTo(-depth, 6);
        // On the ray: x / -z and y / -z are the ndc times the half-extents of the frustum.
        const tanY = Math.tan((30 * Math.PI) / 180);
        expect(p[0] / depth).toBeCloseTo((uv[0] * 2 - 1) * tanY * (16 / 9), 6);
        expect(p[1] / depth).toBeCloseTo((uv[1] * 2 - 1) * tanY, 6);
      }
    }
  });

  it("frameGetFroxel: atlas pixel → tile, slice and uv", () => {
    const camera = new PerspectiveCamera(60, 1, 0.1, 100);
    camera.updateProjectionMatrix();
    // 8x4 froxels, 6 slices in 3 columns: atlas pixel (8 * 2 + 3.5, 4 * 1 + 0.5) is slice 1 * 3 + 2 = 5.
    const f = vdbFrameFroxel([8, 4, 6], 3, [0.5, 50], camera.projectionMatrixInverse.elements, [19.5, 4.5]);
    expect(f.tile).toEqual([2, 1]);
    expect(f.slice).toBe(5);
    expect(f.uv[0]).toBeCloseTo(3.5 / 8, 12);
    expect(f.uv[1]).toBeCloseTo(0.5 / 4, 12);
    expect(f.sliceNear).toBeCloseTo(vdbFrameSliceDepth([0.5, 50], 5 / 6), 12);
    expect(f.sliceFar).toBeCloseTo(50, 9);
    expect(f.viewFar[2]).toBeCloseTo(-50, 6);
  });

  it("box sub-steps: two per voxel of the read level, within [1, subSteps]", () => {
    // A slice crossing 0.1 of a 100-texel box at level 0: 10 voxels → 20 steps.
    expect(vdbFrameBoxSteps(0.1, 100, 0, 32)).toBe(20);
    expect(vdbFrameBoxSteps(0.1, 100, 1, 32)).toBe(10);
    expect(vdbFrameBoxSteps(1, 100, 0, 32)).toBe(32);
    expect(vdbFrameBoxSteps(0.001, 100, 0, 32)).toBe(2); // log2 clamps at 0: one voxel, two steps
  });
});

describe("noise and fade", () => {
  it("IGN noise is 0.5 with jitter off and in [0, 1) with jitter on", () => {
    expect(vdbFrameNoise(-1, [17, 3])).toBe(0.5);
    for (let i = 0; i < 100; i++) {
      const n = vdbFrameNoise(0.3, [i * 7 + 3, i * 13 + 5]);
      expect(n).toBeGreaterThanOrEqual(0);
      expect(n).toBeLessThan(1);
    }
  });

  it("distance fade is 1 at the camera, 0 at the radius, and off with radius 0", () => {
    expect(vdbFrameDistanceFade(50, [0, 0, 0])).toBeCloseTo(1, 12);
    expect(vdbFrameDistanceFade(50, [0, 0, -50])).toBeCloseTo(0, 12);
    expect(vdbFrameDistanceFade(50, [0, 0, -80])).toBeCloseTo(0, 12);
    expect(vdbFrameDistanceFade(0, [0, 0, -80])).toBe(1);
  });
});

describe("Henyey-Greenstein phase (relative to isotropic)", () => {
  it("g = 0 is 1 everywhere; the forward peak is (1 + g) / (1 - g)²", () => {
    for (const c of [-1, -0.3, 0, 0.7, 1]) expect(vdbFrameHenyeyGreenstein(c, 0)).toBeCloseTo(1, 12);
    expect(vdbFrameHenyeyGreenstein(1, 0.8)).toBeCloseTo(1.8 / 0.04, 9);
    expect(vdbFrameHenyeyGreenstein(-1, 0.8)).toBeCloseTo(0.2 / 1.8 ** 2, 12);
  });

  it("integrates to 1 over the sphere (mean over cos θ ∈ [-1, 1]) and has mean cosine g", () => {
    for (const g of [-0.3, 0.2, 0.8]) {
      const n = 200_000;
      let sum = 0;
      let mean = 0;
      for (let i = 0; i < n; i++) {
        const c = -1 + (2 * (i + 0.5)) / n;
        const p = vdbFrameHenyeyGreenstein(c, g);
        sum += p;
        mean += p * c;
      }
      expect(sum / n).toBeCloseTo(1, 4);
      expect(mean / n).toBeCloseTo(g, 4);
    }
  });

  it("dual-lobe is the mix of the two lobes by the back weight; isotropic is 1", () => {
    const ray: [number, number, number] = [0, 0, 1];
    const light: [number, number, number] = [0.6, 0, 0.8];
    const w = vdbFramePhaseWeight([0.8, -0.3, 0.2, 0], ray, light);
    expect(w).toBeCloseTo(0.8 * vdbFrameHenyeyGreenstein(0.8, 0.8) + 0.2 * vdbFrameHenyeyGreenstein(0.8, -0.3), 12);
    expect(vdbFramePhaseWeight([0, 0, 0, 0], ray, light)).toBe(1);
  });
});

describe("shadow march maths", () => {
  const albedo: [number, number, number] = [1, 1, 1];

  it("absorption is the part the albedo luminance does not scatter", () => {
    expect(vdbFrameShadowAbsorption(2, [1, 1, 1])).toBeCloseTo(0, 12);
    expect(vdbFrameShadowAbsorption(2, [0, 0, 0])).toBe(2);
    expect(vdbFrameShadowAbsorption(2, [0.5, 0.5, 0.5])).toBeCloseTo(1, 12);
  });

  it("single scattering (multipleScattering 0) is Beer-Lambert: total = direct", () => {
    const s = vdbFrameShadowStep(1.5, 0.3, 0);
    expect(s).toEqual([1.5, Math.sqrt(0.3 * 3 * 0.3), 0]);
    const t = vdbFrameShadowTransmittance([3, 1, 0], 0);
    expect(t[0]).toBeCloseTo(Math.exp(-3), 12);
    expect(t[1]).toBe(t[0]);
  });

  it("1D diffusion never gives less light than the direct beam, and more for a white medium", () => {
    const ms = 1 - 0.58; // dual-lobe 0.8 / -0.3 / 0.2
    for (const tau of [0.1, 1, 4, 20]) {
      const s = vdbFrameShadowStep(tau, vdbFrameShadowAbsorption(tau, [0.9, 0.9, 0.9]), ms);
      const depth = vdbFrameShadowDepth(s, ms);
      expect(depth).toBeLessThanOrEqual(tau);
      const [direct, total] = vdbFrameShadowTransmittance(s, ms);
      expect(total).toBeGreaterThanOrEqual(direct);
    }
    // White medium: effective extinction 0, total = 1 / (1 + 0.75 * reduced depth).
    const s = vdbFrameShadowStep(4, 0, ms);
    expect(vdbFrameShadowTransmittance(s, ms)[1]).toBeCloseTo(1 / (1 + 0.75 * 4 * ms), 12);
  });

  it("the march sums its steps and stops once the march depth exceeds 8", () => {
    const uniform = vdbFrameShadowMarch([0.5, 0.5, 0.5, 0.5], [0, 0, 0], 0);
    expect(uniform[0]).toBeCloseTo(Math.exp(-2), 12);
    const deep = vdbFrameShadowMarch(new Array(32).fill(1), albedo, 0);
    // Stops after the 9th step (depth 9 > 8).
    expect(deep[0]).toBeCloseTo(Math.exp(-9), 12);
  });

  it("shadowed light: phase × direct + isotropic scattered (total − direct)", () => {
    const ray: [number, number, number] = [0, 0, 1];
    const light: [number, number, number] = [0, 0, 1];
    const phase: [number, number, number, number] = [0.8, -0.3, 0.2, 0];
    const l = vdbFrameShadowedLight(phase, ray, light, [0.25, 0.5]);
    expect(l).toBeCloseTo(vdbFramePhaseWeight(phase, ray, light) * 0.25 + 0.25, 12);
  });
});

describe("inject pre-integration and the integrate composite", () => {
  it("a uniform slice: alpha 1 − exp(−Σ), lit position at the visibility-weighted mean (front-biased)", () => {
    const n = 16;
    const r = vdbFrameInjectSlice(new Array(n).fill(0.3), [0.1, 0.9])!;
    expect(r.alpha).toBeCloseTo(1 - Math.exp(-0.3 * n), 12);
    expect(r.t).toBeGreaterThan(0.1);
    expect(r.t).toBeLessThan(0.5); // dense: weighted toward the front of the segment
    // Thin medium: close to the middle.
    expect(vdbFrameInjectSlice(new Array(n).fill(1e-4), [0, 1])!.t).toBeCloseTo(0.5, 3);
    expect(vdbFrameInjectSlice(new Array(n).fill(0), [0, 1])).toBeNull();
  });

  it("front-to-back over premultiplied slices; one full slice gives its own alpha and colour", () => {
    const range: [number, number] = [1, 100];
    const D = 4;
    const one = vdbFrameIntegrate([[0.2, 0.3, 0.4, 0.5]], [1, 2], range, D, 1e20)!;
    expect(one.alpha).toBeCloseTo(0.5, 12);
    expect(one.radiance).toEqual([0.2, 0.3, 0.4]);
    const two = vdbFrameIntegrate(
      [
        [0.25, 0, 0, 0.5],
        [0.5, 0, 0, 0.5],
      ],
      [0, 2],
      range,
      D,
      1e20,
    )!;
    expect(two.alpha).toBeCloseTo(0.75, 12);
    expect(two.radiance[0]).toBeCloseTo(0.25 + 0.5 * 0.5, 12);
    expect(vdbFrameIntegrate([[0, 0, 0, 0]], [0, 1], range, D, 1e20)).toBeNull();
  });

  it("the scene depth cuts a slice: 1 − (1 − a)^coverage, and stops the march", () => {
    const range: [number, number] = [1, 100];
    const D = 2; // slice 0: [1, 10], slice 1: [10, 100]
    const half = 5.5; // coverage (5.5 - 1) / 9 = 0.5
    const r = vdbFrameIntegrate(
      [
        [0.4, 0.4, 0.4, 0.8],
        [1, 1, 1, 1],
      ],
      [0, 2],
      range,
      D,
      half,
    )!;
    const a = 1 - Math.pow(0.2, 0.5);
    expect(r.alpha).toBeCloseTo(a, 12);
    expect(r.radiance[0]).toBeCloseTo(0.4 * (a / 0.8), 12);
    // A mesh in front of the first slice: nothing.
    expect(vdbFrameIntegrate([[1, 1, 1, 1]], [0, 1], range, D, 0.5)).toBeNull();
  });

  it("sRGB blend: alpha 0 keeps the background, alpha 1 shows the volume colour; OETF/EOTF invert", () => {
    for (const x of [0, 0.001, 0.02, 0.2, 0.7, 1]) expect(vdbSrgbEotf(vdbSrgbOetf(x))).toBeCloseTo(x, 4);
    const bg: [number, number, number] = [0.1, 0.2, 0.3];
    expect(vdbFrameSrgbBlend(bg, [0, 0, 0], 1e-9).map((v) => +v.toFixed(5))).toEqual(bg.map((v) => +vdbSrgbEotf(vdbSrgbOetf(v)).toFixed(5)));
    const full = vdbFrameSrgbBlend(bg, [0.5, 0.6, 0.7], 1);
    expect(full[0]).toBeCloseTo(vdbSrgbEotf(vdbSrgbOetf(0.5)), 9);
    // The clamp: HDR volume colour saturates at 1 with srgbClamp.
    expect(vdbFrameSrgbBlend(bg, [4, 4, 4], 1)[0]).toBeCloseTo(vdbSrgbEotf(1), 9);
  });
});
