import { describe, expect, it } from "vitest";
import {
  vdbRefBeerLambert,
  vdbRefEmissionOnly,
  vdbRefInvert4,
  vdbRefPcg,
  vdbRefRng,
  vdbRefSampleHgCos,
  vdbRefPhase,
  vdbRefSingleScatter,
  vdbReferenceConstantMediumCpu,
  vdbReferenceLightsFromFrame,
  vdbReferenceRenderCpu,
  vdbReferenceSigma,
  vdbReferenceStatistics,
  VDB_REFERENCE_DIRECT_LIGHT_SCALE,
  type VDBRefVec3,
  type VDBReferenceMutation,
  type VDBReferenceScene,
  type VDBReferenceSettings,
} from "../../src/tools/internal.js";

// K6.R: the CPU twin of the reference renderer against closed forms (Beer–Lambert, emission only, white furnace)
// and a high-precision quadrature (single scattering of a directional light), each with a fail-before mutation.
// The GPU kernel runs the same cases in e2e/k6-reference.spec.ts.

const W = 8;
const H = 8;
const HALF = 1.5; // ortho half extent: some pixels miss the medium

/** Orthographic camera at z = 5 looking down −z (WebGL depth), covering [−HALF, HALF]². */
function orthoCamera(): VDBReferenceScene["camera"] {
  const n = 0.1;
  const f = 100;
  const p = [1 / HALF, 0, 0, 0, 0, 1 / HALF, 0, 0, 0, 0, -2 / (f - n), 0, 0, 0, -(f + n) / (f - n), 1];
  return { cameraToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 5, 1], projectionInverse: vdbRefInvert4(p), orthographic: true };
}

/** The ray of pixel (px, py) of {@link orthoCamera} (centre sampling). */
function orthoRay(px: number, py: number): { o: VDBRefVec3; d: VDBRefVec3 } {
  const x = (((px + 0.5) / W) * 2 - 1) * HALF;
  const y = ((1 - (py + 0.5) / H) * 2 - 1) * HALF;
  return { o: [x, y, 5], d: [0, 0, -1] };
}

const SLAB = [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const BALL = [2.4, 0, 0, 0, 0, 2.4, 0, 0, 0, 0, 2.4, 0, 0.1, -0.05, 0, 1];

interface Check {
  pixels: number;
  /** Pixels with |error| > k·σ in some channel. */
  over: number;
  maxZ: number;
  /** Mean signed error / mean σ over all pixels and channels (bias in units of σ of the image mean). */
  meanZ: number;
}

/** Compare accumulated means with `expected(px, py)` (RGBA = L, T) channel by channel: |e| ≤ k σ + abs. */
function check(scene: VDBReferenceScene, settings: VDBReferenceSettings, batches: number, expected: (px: number, py: number) => number[], channels: number[], mutate?: VDBReferenceMutation, k = 4.5, abs = 1e-6): Check {
  const accum = vdbReferenceRenderCpu(scene, scene.volumes.map(() => vdbReferenceConstantMediumCpu(1, 1)), settings, { batches, ...(mutate ? { mutate } : {}) });
  const { mean, stderr } = vdbReferenceStatistics(accum);
  let over = 0;
  let maxZ = 0;
  let sumE = 0;
  let sumS2 = 0;
  let count = 0;
  for (let py = 0; py < H; py++) for (let px = 0; px < W; px++) {
    const want = expected(px, py);
    let bad = false;
    for (const c of channels) {
      const i = (py * W + px) * 4 + c;
      const e = mean[i]! - want[c]!;
      const s = stderr[i]!;
      const z = Math.abs(e) / Math.max(s, 1e-12);
      if (Math.abs(e) > k * s + abs) bad = true;
      if (Math.abs(e) > abs) maxZ = Math.max(maxZ, z);
      sumE += e;
      sumS2 += s * s;
      count++;
    }
    if (bad) over++;
  }
  return { pixels: W * H, over, maxZ, meanZ: sumE / Math.sqrt(Math.max(sumS2, 1e-30)) };
}

/** Pass: at most 2 % of pixels over k σ (batch-mean σ has few degrees of freedom) and the whole image unbiased within 4σ. */
const passes = (c: Check) => { if (process.env.REF_DEBUG) console.log(JSON.stringify(c)); return c.over <= Math.max(1, Math.floor(c.pixels * 0.02)) && Math.abs(c.meanZ) < 4; };

describe("K6.R reference random stream", () => {
  it("is the PCG hash and gives 24-bit floats in [0, 1)", () => {
    // Known values of pcg (Jarzynski & Olano 2020): pcg(0) = 129708002, pcg(1) = 2831084092.
    expect(vdbRefPcg(0)).toBe(129708002);
    expect(vdbRefPcg(1)).toBe(2831084092);
    const r = vdbRefRng(7, 3, 11);
    let lo = 1;
    let hi = 0;
    let sum = 0;
    for (let i = 0; i < 20000; i++) {
      const x = r.next();
      expect(x * 2 ** 24).toBe(Math.floor(x * 2 ** 24));
      lo = Math.min(lo, x);
      hi = Math.max(hi, x);
      sum += x;
    }
    expect(lo).toBeGreaterThanOrEqual(0);
    expect(hi).toBeLessThan(1);
    expect(Math.abs(sum / 20000 - 0.5)).toBeLessThan(0.01);
    // Streams differ by seed, sample and pixel.
    const first = (s: number, i: number, p: number) => vdbRefRng(s, i, p).next();
    expect(new Set([first(1, 0, 0), first(2, 0, 0), first(1, 1, 0), first(1, 0, 1)]).size).toBe(4);
  });

  it("samples HG with the right mean cosine (g) and the dual-lobe phase integrates to 1", () => {
    for (const g of [-0.7, 0, 0.3, 0.85]) {
      let m = 0;
      const n = 20000;
      for (let i = 0; i < n; i++) m += vdbRefSampleHgCos(g, (i + 0.5) / n);
      expect(m / n).toBeCloseTo(g, 3);
    }
    let integral = 0;
    const n = 200000;
    for (let i = 0; i < n; i++) integral += vdbRefPhase(-1 + (2 * (i + 0.5)) / n, 0.8, -0.3, 0.2) * 2 * Math.PI * (2 / n);
    expect(integral).toBeCloseTo(1, 4);
  });

  it("maps the renderer's lights and extinction into reference units", () => {
    const { lights, environment } = vdbReferenceLightsFromFrame([
      { kind: "ambient", color: [0.3, 0.3, 0.3] },
      { kind: "hemisphere", sky: [1, 0, 0], ground: [0, 1, 0], up: [0, 1, 0] },
      { kind: "directional", direction: [1, 2, 1], color: [1, 0.5, 0.25] },
    ]);
    expect(lights).toEqual([{ kind: "directional", direction: [1, 2, 1], irradiance: [4, 2, 1] }]);
    expect(VDB_REFERENCE_DIRECT_LIGHT_SCALE).toBe(4);
    expect(environment.uniform![0]).toBeCloseTo(0.3 / Math.PI, 12);
    expect(environment.hemispheres![0]!.sky[0]).toBeCloseTo(1 / Math.PI, 12);
    // His bunny (manifest step1-*): box 46.08 x 45.68 x 34.96, densityScale 1/mean, extinction 100.
    const sigma = vdbReferenceSigma({ localToWorld: [46.08, 0, 0, 0, 0, 45.68, 0, 0, 0, 0, 34.96, 0, 0, 0, 0, 1], extinction: 100, normalizeDensity: true, meanNonZero: 0.29587744026889873 });
    expect(sigma).toBeCloseTo(100 / 0.29587744026889873 / 46.08, 9);
  });
});

describe("K6.R CPU twin against analytic cases", () => {
  const base: VDBReferenceSettings = { width: W, height: H, spp: 8, maxBounces: 1, seed: 5 };

  const beer = (shape: "box" | "sphere", mutate?: VDBReferenceMutation) => {
    const m = shape === "box" ? SLAB : BALL;
    const scene: VDBReferenceScene = { camera: orthoCamera(), volumes: [{ localToWorld: m, sigma: 1.3, majorant: 2, albedo: [0, 0, 0], shape }] };
    const sphere = shape === "sphere";
    return check(scene, base, 24, (px, py) => {
      const r = orthoRay(px, py);
      return [0, 0, 0, vdbRefBeerLambert({ localToWorld: m, sphere, sigma: 1.3 }, r.o, r.d)];
    }, [0, 1, 2, 3], mutate);
  };

  it("Beer–Lambert: ratio-tracked T of a slab and a ball = exp(−σ·chord)", () => {
    for (const shape of ["box", "sphere"] as const) {
      const c = beer(shape);
      expect(passes(c), `${shape} ${JSON.stringify(c)}`).toBe(true);
    }
  });

  it("fail-before: a wrong ratio-tracking weight fails Beer–Lambert", () => {
    expect(passes(beer("sphere", "ratio-weight"))).toBe(false);
  });

  const emission = (mutate?: VDBReferenceMutation) => {
    const le: VDBRefVec3 = [0.8, 0.5, 0.2];
    const scene: VDBReferenceScene = {
      camera: orthoCamera(),
      volumes: [{ localToWorld: BALL, sigma: 0.9, majorant: 1, albedo: [0, 0, 0], shape: "sphere", emits: true, emission: { color: le, intensity: 1, temperatureRange: [0, 3000] } }],
    };
    return check(scene, { ...base, maxBounces: 4 }, Number(process.env.REF_BATCHES ?? 24), (px, py) => {
      const r = orthoRay(px, py);
      const m = { localToWorld: BALL, sphere: true, sigma: 0.9 };
      return [...vdbRefEmissionOnly(m, r.o, r.d, le), vdbRefBeerLambert(m, r.o, r.d)];
    }, [0, 1, 2, 3], mutate);
  };

  it("emission only (albedo 0): L = Le · (1 − T)", () => {
    const c = emission();
    expect(passes(c), JSON.stringify(c)).toBe(true);
  });

  it("fail-before: emission weighted by the albedo fails the emission case", () => {
    expect(passes(emission("emission-albedo"))).toBe(false);
  });

  const single = (shape: "box" | "sphere", mutate?: VDBReferenceMutation) => {
    const m = shape === "box" ? SLAB : BALL;
    const toLight = (() => {
      const v = [0.4, 0.7, 0.6];
      const l = Math.hypot(...v);
      return v.map((x) => x / l) as VDBRefVec3;
    })();
    const albedo: VDBRefVec3 = [0.9, 0.7, 0.5];
    const phase = { g: 0.6, backG: -0.3, backWeight: 0.25 };
    const e: VDBRefVec3 = [1.5, 1.2, 1];
    const scene: VDBReferenceScene = {
      camera: orthoCamera(),
      volumes: [{ localToWorld: m, sigma: 0.8, majorant: 1.5, albedo, phase, shape }],
      lights: [{ kind: "directional", direction: toLight, irradiance: e }],
    };
    const sphere = shape === "sphere";
    return check(scene, { ...base, spp: 16 }, 24, (px, py) => {
      const r = orthoRay(px, py);
      const hm = { localToWorld: m, sphere, sigma: 0.8 };
      return [...vdbRefSingleScatter(hm, r.o, r.d, toLight, e, albedo, phase), vdbRefBeerLambert(hm, r.o, r.d)];
    }, [0, 1, 2, 3], mutate);
  };

  it("single scattering of a directional light = the quadrature (slab and ball, dual-lobe phase, RGB albedo)", () => {
    for (const shape of ["box", "sphere"] as const) {
      const c = single(shape);
      expect(passes(c), `${shape} ${JSON.stringify(c)}`).toBe(true);
    }
  });

  it("fail-before: the phase without its 1/(4π), or NEE without transmittance, fails single scattering", () => {
    expect(passes(single("sphere", "phase-norm"))).toBe(false);
    expect(passes(single("box", "nee-transmittance"))).toBe(false);
  });

  const furnace = (mutate?: VDBReferenceMutation) => {
    const le: VDBRefVec3 = [0.5, 0.5, 0.5];
    const scene: VDBReferenceScene = {
      camera: orthoCamera(),
      volumes: [{ localToWorld: BALL, sigma: 1.6, majorant: 1.25, albedo: [1, 1, 1], phase: { g: 0.5 }, shape: "sphere" }],
      environment: { uniform: le },
    };
    const accum = vdbReferenceRenderCpu(scene, [vdbReferenceConstantMediumCpu(1)], { ...base, maxBounces: 4096, spp: 8 }, { batches: 24, ...(mutate ? { mutate } : {}) });
    const { mean, stderr } = vdbReferenceStatistics(accum);
    // L + T · Le = Le per channel: σ of the sum ≤ σ_L + Le · σ_T.
    let over = 0;
    let sumE = 0;
    let sumS2 = 0;
    for (let i = 0; i < W * H; i++) {
      for (let c = 0; c < 3; c++) {
        const e = mean[i * 4 + c]! + mean[i * 4 + 3]! * le[c]! - le[c]!;
        const s = stderr[i * 4 + c]! + le[c]! * stderr[i * 4 + 3]!;
        if (Math.abs(e) > 4.5 * s + 1e-6) over++;
        sumE += e;
        sumS2 += s * s;
      }
    }
    return { over, meanZ: sumE / Math.sqrt(sumS2), stats: accum.stats };
  };

  it("white furnace: albedo 1 under a uniform environment conserves energy (L + T·Le = Le)", () => {
    const f = furnace();
    expect(f.over, JSON.stringify(f)).toBeLessThanOrEqual(1);
    expect(Math.abs(f.meanZ), JSON.stringify(f)).toBeLessThan(4);
    expect(f.stats.truncated).toBe(0);
    expect(f.stats.majorantViolations).toBe(0);
    expect(f.stats.meanDepth).toBeGreaterThan(1);
  });

  it("fail-before: accepting every tentative collision, or dropping the environment after 2 bounces, breaks the furnace", () => {
    for (const m of ["delta-accept", "escape-depth"] as const) {
      const f = furnace(m);
      expect(f.over > 0 || Math.abs(f.meanZ) >= 4, m).toBe(true);
    }
  });

  it("is deterministic: the same seed gives the same sums, another seed different ones", () => {
    const scene: VDBReferenceScene = { camera: orthoCamera(), volumes: [{ localToWorld: BALL, sigma: 1, majorant: 1.2, albedo: [0.8, 0.8, 0.8], shape: "sphere" }], environment: { uniform: [1, 1, 1] } };
    const run = (seed: number) => vdbReferenceRenderCpu(scene, [vdbReferenceConstantMediumCpu(1)], { ...base, maxBounces: 32, seed }, { batches: 2 });
    expect(Array.from(run(3).sum)).toEqual(Array.from(run(3).sum));
    expect(Array.from(run(3).sum)).not.toEqual(Array.from(run(4).sum));
  });
});
