import { describe, expect, it } from "vitest";
import {
  leafVoxels,
  sampleWvdbDenseTexture,
  vdbBuildFromDense,
  vdbFrameBlackbodyColor,
  vdbFrameMieTexel,
  vdbRefBeerLambert,
  vdbRefColorTableAt,
  vdbRefDistanceAttenuation,
  vdbRefEmissionOnly,
  vdbRefEnvironmentRadiance,
  vdbRefInvert4,
  vdbRefMajorantCell,
  vdbRefPhaseTableEval,
  vdbRefPhaseTableSample,
  vdbRefShBasis,
  vdbRefSingleScatter,
  vdbRefSingleScatterPoint,
  vdbRefVolumeInterval,
  vdbReferenceBlackbodyTable,
  vdbReferenceConstantMediumCpu,
  vdbReferenceLightsFromFrame,
  vdbReferenceMajorantGridDense,
  vdbReferenceMajorantGridSparse,
  vdbReferenceMiePhaseTable,
  vdbReferencePhaseTable,
  vdbReferenceRenderCpu,
  vdbReferenceStatistics,
  vdbSparseDensityCpu,
  type VDBRefVec3,
  type VDBReferenceMajorantGrid,
  type VDBReferenceMediumCpu,
  type VDBReferenceMutation,
  type VDBReferenceScene,
  type VDBReferenceSettings,
} from "../../src/tools/internal.js";

// K6.R1: the reference renderer's new features on the CPU twin, each against a closed form or a quadrature with a
// fail-before mutation: three's point-light falloff ('frame' lighting), the blackbody colour table, Lorenz-Mie and
// Mie RGB phase tables (evaluation and inverse-CDF sampling), the light-probe environment and per-cell majorant
// grids (dense texels, sparse node stats). The GPU kernel runs them against this twin in e2e/k6-reference.spec.ts.

const W = 8;
const H = 8;
const HALF = 1.5;

function orthoCamera(): VDBReferenceScene["camera"] {
  const n = 0.1;
  const f = 100;
  const p = [1 / HALF, 0, 0, 0, 0, 1 / HALF, 0, 0, 0, 0, -2 / (f - n), 0, 0, 0, -(f + n) / (f - n), 1];
  return { cameraToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 5, 1], projectionInverse: vdbRefInvert4(p), orthographic: true };
}

function orthoRay(px: number, py: number): { o: VDBRefVec3; d: VDBRefVec3 } {
  const x = (((px + 0.5) / W) * 2 - 1) * HALF;
  const y = ((1 - (py + 0.5) / H) * 2 - 1) * HALF;
  return { o: [x, y, 5], d: [0, 0, -1] };
}

const BALL = [2.4, 0, 0, 0, 0, 2.4, 0, 0, 0, 0, 2.4, 0, 0.1, -0.05, 0, 1];
const BOX = [2.2, 0, 0, 0, 0, 2.2, 0, 0, 0, 0, 2.2, 0, 0, 0, 0, 1];

interface Check {
  over: number;
  maxZ: number;
  meanZ: number;
}

function check(scene: VDBReferenceScene, media: VDBReferenceMediumCpu[], settings: VDBReferenceSettings, batches: number, expected: (px: number, py: number) => number[], mutate?: VDBReferenceMutation): Check & { violations: number } {
  const accum = vdbReferenceRenderCpu(scene, media, settings, { batches, ...(mutate ? { mutate } : {}) });
  const { mean, stderr } = vdbReferenceStatistics(accum);
  let over = 0;
  let maxZ = 0;
  let sumE = 0;
  let sumS2 = 0;
  for (let py = 0; py < H; py++) for (let px = 0; px < W; px++) {
    const want = expected(px, py);
    let bad = false;
    for (let c = 0; c < 4; c++) {
      const i = (py * W + px) * 4 + c;
      const e = mean[i]! - want[c]!;
      const s = stderr[i]!;
      if (Math.abs(e) > 4.5 * s + 1e-6) bad = true;
      if (Math.abs(e) > 1e-6) maxZ = Math.max(maxZ, Math.abs(e) / Math.max(s, 1e-12));
      sumE += e;
      sumS2 += s * s;
    }
    if (bad) over++;
  }
  return { over, maxZ, meanZ: sumE / Math.sqrt(Math.max(sumS2, 1e-30)), violations: accum.stats.majorantViolations };
}
const passes = (c: Check) => {
  if (process.env.REF_DEBUG) console.log(JSON.stringify(c));
  return c.over <= 1 && Math.abs(c.meanZ) < 4;
};

/** The furnace check: L + T·Le = Le per channel. */
function furnace(scene: VDBReferenceScene, media: VDBReferenceMediumCpu[], le: VDBRefVec3, batches: number, mutate?: VDBReferenceMutation) {
  const accum = vdbReferenceRenderCpu(scene, media, { width: W, height: H, spp: 8, maxBounces: 4096, seed: 3 }, { batches, ...(mutate ? { mutate } : {}) });
  const { mean, stderr } = vdbReferenceStatistics(accum);
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
}
const furnacePasses = (f: { over: number; meanZ: number }) => {
  if (process.env.REF_DEBUG) console.log(JSON.stringify(f));
  return f.over <= 2 && Math.abs(f.meanZ) < 4;
};

describe("K6.R1 light mapping per lighting mode", () => {
  it("'volume' maps point and spot lights without falloff, 'frame' with three's getDistanceAttenuation", () => {
    const lights = [
      { kind: "point" as const, position: [1, 2, 3] as VDBRefVec3, color: [1, 1, 1] as VDBRefVec3, distance: 40, decay: 1.5 },
      { kind: "spot" as const, position: [0, 5, 0] as VDBRefVec3, direction: [0, -1, 0] as VDBRefVec3, color: [2, 2, 2] as VDBRefVec3, cosAngle: 0.8, cosPenumbra: 0.9 },
      { kind: "probe" as const, sh: Array.from({ length: 27 }, (_, i) => i) },
    ];
    const v = vdbReferenceLightsFromFrame(lights);
    expect(v.lights.map((l) => ("falloff" in l ? l.falloff : undefined))).toEqual([undefined, undefined]);
    const f = vdbReferenceLightsFromFrame(lights, { lighting: "frame" });
    expect(f.lights.map((l) => ("falloff" in l ? l.falloff : undefined))).toEqual([{ decay: 1.5, cutoff: 40 }, { decay: 2, cutoff: 0 }]);
    expect(Array.from(f.environment.sh!)).toEqual(Array.from({ length: 27 }, (_, i) => i));
    // three: 1 / max(d^decay, 0.01) × (clamp(1 − (d/cutoff)⁴))²
    expect(vdbRefDistanceAttenuation(10, { decay: 2 })).toBeCloseTo(0.01, 12);
    expect(vdbRefDistanceAttenuation(0.05, { decay: 2 })).toBe(100);
    expect(vdbRefDistanceAttenuation(20, { decay: 2, cutoff: 40 })).toBeCloseTo((1 / 400) * (1 - 1 / 16) ** 2, 12);
    expect(vdbRefDistanceAttenuation(50, { decay: 2, cutoff: 40 })).toBe(0);
  });

  const pointCase = (mutate?: VDBReferenceMutation) => {
    const light = { position: [2.5, 1.8, 1.2] as VDBRefVec3, intensity: [6, 5, 4] as VDBRefVec3, falloff: { decay: 2, cutoff: 9 } };
    const albedo: VDBRefVec3 = [0.9, 0.8, 0.7];
    const phase = { g: 0.4, backG: -0.3, backWeight: 0.2 };
    const scene: VDBReferenceScene = {
      camera: orthoCamera(),
      volumes: [{ localToWorld: BALL, sigma: 0.8, majorant: 1.2, albedo, phase, shape: "sphere" }],
      lights: [{ kind: "point", ...light }],
    };
    const m = { localToWorld: BALL, sphere: true, sigma: 0.8 };
    return check(scene, [vdbReferenceConstantMediumCpu(1)], { width: W, height: H, spp: 16, maxBounces: 1, seed: 7 }, 24, (px, py) => {
      const r = orthoRay(px, py);
      return [...vdbRefSingleScatterPoint(m, r.o, r.d, light, albedo, phase), vdbRefBeerLambert(m, r.o, r.d)];
    }, mutate);
  };

  it("single scattering of a point light with three's falloff = the quadrature; without the falloff it fails", () => {
    expect(passes(pointCase())).toBe(true);
    expect(passes(pointCase("falloff"))).toBe(false);
  });
});

describe("K6.R1 blackbody emission colour table", () => {
  it("reads the renderer's blackbody colour (log-temperature blend of the same columns)", () => {
    const t = vdbReferenceBlackbodyTable(1.39);
    for (const T of [600, 1000, 1500, 2333, 5000, 12000, 39000]) {
      const a = vdbRefColorTableAt(t, T);
      const b = vdbFrameBlackbodyColor(T, 1.39);
      for (let c = 0; c < 3; c++) expect(Math.abs(a[c]! - b[c]!), `${T} K`).toBeLessThan(2e-6);
    }
  });

  it("emission only with the table: L = Le · colour(T) · (1 − T)", () => {
    const table = vdbReferenceBlackbodyTable(0);
    const value = 0.7;
    const range: [number, number] = [0, 2000];
    const tc = vdbFrameBlackbodyColor(value * 2000, 0);
    const power = value ** 4;
    const le: VDBRefVec3 = [0.8 * 3 * power * tc[0], 0.5 * 3 * power * tc[1], 0.2 * 3 * power * tc[2]];
    const scene: VDBReferenceScene = {
      camera: orthoCamera(),
      volumes: [{ localToWorld: BALL, sigma: 0.9, majorant: 1, albedo: [0, 0, 0], shape: "sphere", emits: true, emission: { color: [0.8, 0.5, 0.2], intensity: 3, temperatureRange: range, colorTable: table } }],
    };
    const m = { localToWorld: BALL, sphere: true, sigma: 0.9 };
    const c = check(scene, [vdbReferenceConstantMediumCpu(1, value)], { width: W, height: H, spp: 8, maxBounces: 4, seed: 5 }, 24, (px, py) => {
      const r = orthoRay(px, py);
      return [...vdbRefEmissionOnly(m, r.o, r.d, le), vdbRefBeerLambert(m, r.o, r.d)];
    });
    expect(passes(c), JSON.stringify(c)).toBe(true);
  });
});

describe("K6.R1 tabulated phase (Lorenz-Mie, Mie RGB)", () => {
  const mono = vdbReferenceMiePhaseTable("mono");
  const rgb = vdbReferenceMiePhaseTable("rgb");
  // A synthetic table whose channels differ strongly (isotropic, HG 0.8, HG −0.6): the per-channel weights matter.
  const hg = (c: number, g: number) => (1 - g * g) / Math.pow(1 + g * g - 2 * g * c, 1.5);
  const mixed = vdbReferencePhaseTable(
    Array.from({ length: 512 * 3 }, (_, k) => {
      const u = Math.floor(k / 3) / 511;
      const c = Math.cos(Math.PI * u * u);
      return [1, hg(c, 0.8), hg(c, -0.6)][k % 3]!;
    }),
    512,
  );

  it("evaluates his texels per steradian and every channel integrates to 1", () => {
    for (const cos of [1, 0.999, 0.9, 0.3, -0.5, -1]) {
      const t = vdbFrameMieTexel(cos);
      const m = vdbRefPhaseTableEval(mono, cos);
      const r = vdbRefPhaseTableEval(rgb, cos);
      expect(m[0] * 4 * Math.PI).toBeCloseTo(t[3], 2);
      for (let c = 0; c < 3; c++) expect(r[c]! * 4 * Math.PI).toBeCloseTo(t[c]!, 2);
    }
    // ∫ p dω over u (θ = π u²): dω = 4π² u sin θ du; fine midpoint rule
    for (const t of [mono, rgb]) {
      const s = [0, 0, 0];
      const n = 400000;
      for (let i = 0; i < n; i++) {
        const u = (i + 0.5) / n;
        const p = vdbRefPhaseTableEval(t, Math.cos(Math.PI * u * u));
        for (let c = 0; c < 3; c++) s[c] += (p[c]! * 4 * Math.PI * Math.PI * u * Math.sin(Math.PI * u * u)) / n;
      }
      for (let c = 0; c < 3; c++) expect(s[c]!).toBeCloseTo(1, 2);
    }
  });

  it("inverse-CDF sampling is unbiased per channel: E[w] = 1 and E[w cos θ] = the channel's mean cosine", () => {
    for (const t of [mono, rgb, mixed]) {
      const n = 200000;
      const ew = [0, 0, 0];
      const ewc = [0, 0, 0];
      for (let i = 0; i < n; i++) {
        const s = vdbRefPhaseTableSample(t, (i + 0.5) / n);
        for (let c = 0; c < 3; c++) {
          ew[c] += s.weight[c]! / n;
          ewc[c] += (s.weight[c]! * s.cosTheta) / n;
        }
      }
      const g = [0, 0, 0];
      const m = 400000;
      for (let i = 0; i < m; i++) {
        const u = (i + 0.5) / m;
        const cos = Math.cos(Math.PI * u * u);
        const p = vdbRefPhaseTableEval(t, cos);
        for (let c = 0; c < 3; c++) g[c] += (p[c]! * cos * 4 * Math.PI * Math.PI * u * Math.sin(Math.PI * u * u)) / m;
      }
      for (let c = 0; c < 3; c++) {
        expect(ew[c]!).toBeCloseTo(1, 2);
        expect(ewc[c]!).toBeCloseTo(g[c]!, 2);
      }
    }
    expect(Math.abs(vdbRefPhaseTableSample(mono, 0.5).weight[0] - 1)).toBeLessThan(0.05);
  });

  it("single scattering with the Mie RGB table = the quadrature", () => {
    const toLight: VDBRefVec3 = [0.36, 0.48, 0.8];
    const e: VDBRefVec3 = [1.2, 1, 0.8];
    const albedo: VDBRefVec3 = [0.9, 0.9, 0.9];
    const phase = { g: 0, table: rgb };
    // a camera ray direction close to the light's (forward scattering): tilt the camera
    const scene: VDBReferenceScene = { camera: orthoCamera(), volumes: [{ localToWorld: BALL, sigma: 0.7, majorant: 1, albedo, phase, shape: "sphere" }], lights: [{ kind: "directional", direction: toLight, irradiance: e }] };
    const m = { localToWorld: BALL, sphere: true, sigma: 0.7 };
    const c = check(scene, [vdbReferenceConstantMediumCpu(1)], { width: W, height: H, spp: 16, maxBounces: 1, seed: 9 }, 24, (px, py) => {
      const r = orthoRay(px, py);
      return [...vdbRefSingleScatter(m, r.o, r.d, toLight, e, albedo, phase), vdbRefBeerLambert(m, r.o, r.d)];
    });
    expect(passes(c), JSON.stringify(c)).toBe(true);
  });

  it("white furnace with Mie RGB (energy per channel through many bounces)", () => {
    const le: VDBRefVec3 = [0.5, 0.5, 0.5];
    const scene: VDBReferenceScene = { camera: orthoCamera(), volumes: [{ localToWorld: BALL, sigma: 1.4, majorant: 1.1, albedo: [1, 1, 1], phase: { g: 0, table: rgb }, shape: "sphere" }], environment: { uniform: le } };
    const ok = furnace(scene, [vdbReferenceConstantMediumCpu(1)], le, 24);
    expect(furnacePasses(ok), JSON.stringify(ok)).toBe(true);
    expect(ok.stats.meanDepth).toBeGreaterThan(1);
  });

  it("fail-before: one channel's sampling weight for all three moves a hemisphere-lit image (the GPU twin check sees it)", () => {
    // The furnace cannot see it (any sampled direction conserves energy); a non-uniform environment does.
    const scene: VDBReferenceScene = {
      camera: orthoCamera(),
      volumes: [{ localToWorld: BALL, sigma: 0.5, majorant: 1.1, albedo: [1, 1, 1], phase: { g: 0, table: mixed }, shape: "sphere" }],
      environment: { hemispheres: [{ sky: [0.2, 0.5, 1.5], ground: [1.6, 0.3, 0.05], up: [0.1, 0.2, 1] }] },
    };
    const run = (mutate?: VDBReferenceMutation) => vdbReferenceStatistics(vdbReferenceRenderCpu(scene, [vdbReferenceConstantMediumCpu(1)], { width: W, height: H, spp: 16, maxBounces: 64, seed: 4 }, { batches: 16, ...(mutate ? { mutate } : {}) }));
    const a = run();
    const b = run("phase-table-weight");
    // per channel, the image sum's difference in units of its σ (the twin check's image z)
    const z = [0, 1, 2].map((c) => {
      let e = 0;
      let s2 = 0;
      for (let i = c; i < a.mean.length; i += 4) {
        e += b.mean[i]! - a.mean[i]!;
        s2 += a.stderr[i]! ** 2 + b.stderr[i]! ** 2;
      }
      return e / Math.sqrt(s2);
    });
    if (process.env.REF_DEBUG) console.log(JSON.stringify(z));
    expect(Math.max(...z.map(Math.abs))).toBeGreaterThan(6);
  });
});

describe("K6.R1 light probe environment", () => {
  it("is three's SphericalHarmonics3 radiance and integrates to 4π · c₀ · Y₀₀", () => {
    const sh = Array.from({ length: 27 }, (_, i) => Math.sin(i * 1.7) * 0.3 + (i < 3 ? 1 : 0));
    const d = [0.36, 0.48, 0.8];
    const y = vdbRefShBasis(d);
    const want = [0, 1, 2].map((c) => y.reduce((s, yi, i) => s + yi * sh[i * 3 + c]!, 0));
    expect(vdbRefEnvironmentRadiance({ sh }, d)).toEqual(want);
    let s = 0;
    const n = 400;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const z = -1 + (2 * (i + 0.5)) / n;
      const phi = (2 * Math.PI * (j + 0.5)) / n;
      const r = Math.sqrt(1 - z * z);
      s += vdbRefEnvironmentRadiance({ sh }, [r * Math.cos(phi), r * Math.sin(phi), z])[0]! * ((4 * Math.PI) / (n * n));
    }
    expect(s).toBeCloseTo(4 * Math.PI * 0.282095 * sh[0]!, 3);
  });

  it("a constant probe is a uniform environment: the white furnace holds", () => {
    const c0 = 0.5 / 0.282095;
    const sh = [c0, c0, c0, ...new Array(24).fill(0)];
    const scene: VDBReferenceScene = { camera: orthoCamera(), volumes: [{ localToWorld: BALL, sigma: 1.6, majorant: 1.25, albedo: [1, 1, 1], phase: { g: 0.5 }, shape: "sphere" }], environment: { sh } };
    expect(furnacePasses(furnace(scene, [vdbReferenceConstantMediumCpu(1)], [0.5, 0.5, 0.5], 24))).toBe(true);
  });
});

describe("K6.R1 single scattering of the environment (maxBounces 1)", () => {
  it("a uniform environment through an isotropic ball = the quadrature (the last bounce still escapes and gathers it)", () => {
    const le: VDBRefVec3 = [0.6, 0.5, 0.4];
    const albedo: VDBRefVec3 = [0.9, 0.8, 0.7];
    const sigma = 0.9;
    const scene: VDBReferenceScene = { camera: orthoCamera(), volumes: [{ localToWorld: BALL, sigma, majorant: 1, albedo, phase: { g: 0 }, shape: "sphere" }], environment: { uniform: le } };
    const m = { localToWorld: BALL, sphere: true, sigma };
    const w2l = vdbRefInvert4(BALL);
    // directions: a Fibonacci sphere of 1024; isotropic phase 1/(4π)
    const dirs = Array.from({ length: 1024 }, (_, i) => {
      const z = 1 - (2 * (i + 0.5)) / 1024;
      const r = Math.sqrt(1 - z * z);
      const phi = i * Math.PI * (3 - Math.sqrt(5));
      return [r * Math.cos(phi), r * Math.sin(phi), z];
    });
    const c = check(scene, [vdbReferenceConstantMediumCpu(1)], { width: W, height: H, spp: 16, maxBounces: 1, seed: 12 }, 24, (px, py) => {
      const r = orthoRay(px, py);
      const T = vdbRefBeerLambert(m, r.o, r.d);
      const [t0, t1] = vdbRefVolumeInterval(w2l, true, r.o, r.d, 1e30);
      if (!(t0 < t1)) return [0, 0, 0, T];
      const n = 64;
      const h = (t1 - t0) / n;
      let s = 0;
      for (let k = 0; k <= n; k++) {
        const t = t0 + k * h;
        const x = [r.o[0] + t * r.d[0], r.o[1] + t * r.d[1], r.o[2] + t * r.d[2]];
        let esc = 0;
        for (const d of dirs) {
          const [, e1] = vdbRefVolumeInterval(w2l, true, x, d, 1e30);
          esc += Math.exp(-sigma * Math.max(e1, 0));
        }
        s += (k === 0 || k === n ? 1 : k % 2 ? 4 : 2) * Math.exp(-sigma * (t - t0)) * (esc / dirs.length);
      }
      const integral = (s * h) / 3;
      return [0, 1, 2].map((k) => sigma * albedo[k]! * le[k]! * integral).concat([T]);
    });
    expect(passes(c), JSON.stringify(c)).toBe(true);
  });
});

describe("K6.R1 majorant grids", () => {
  // A checker medium: value per cell of a 5 × 4 × 3 grid (0, thin, thick), the grid bound is exact per cell.
  const dims: [number, number, number] = [5, 4, 3];
  const cellValue = (i: number) => [0, 0.15, 2.2, 0.6, 1.3][(i * 3 + ((i / 5) | 0) + ((i / 20) | 0) * 2) % 5]!;
  const grid: VDBReferenceMajorantGrid = { dims, values: Float32Array.from({ length: 60 }, (_, i) => cellValue(i)) };
  const checker: VDBReferenceMediumCpu = {
    density: (uvw) => {
      const c = [0, 1, 2].map((k) => Math.min(Math.max(Math.floor(uvw[k]! * dims[k]!), 0), dims[k]! - 1));
      return cellValue(c[0]! + dims[0] * (c[1]! + dims[1] * c[2]!));
    },
  };

  it("finds the cell ahead at a face and its exit", () => {
    const g: VDBReferenceMajorantGrid = { dims: [4, 4, 4], values: new Float32Array(64) };
    // the ray x = -0.5 + t, on the face x = -0.25 at t = 0.25: the next cell (1)
    const at = vdbRefMajorantCell(g, [-0.5, 0.1, 0.1], [1, 0, 0], 0.25);
    expect(at.cell).toEqual([1, 2, 2]);
    expect(at.exit).toBeCloseTo(0.5, 12);
    const back = vdbRefMajorantCell(g, [0.5, 0.1, 0.1], [-1, 0, 0], 0.25);
    expect(back.cell).toEqual([2, 2, 2]);
    expect(back.exit).toBeCloseTo(0.5, 12);
  });

  it("white furnace through a checker medium with exact per-cell majorants: no violations, energy conserved", () => {
    const le: VDBRefVec3 = [0.5, 0.5, 0.5];
    const scene: VDBReferenceScene = {
      camera: orthoCamera(),
      volumes: [{ localToWorld: BOX, sigma: 1.5, albedo: [1, 1, 1], phase: { g: 0.3 }, majorantGrid: grid }],
      environment: { uniform: le },
    };
    const ok = furnace(scene, [checker], le, 24);
    expect(furnacePasses(ok), JSON.stringify(ok)).toBe(true);
    expect(ok.stats.majorantViolations).toBe(0);
    expect(ok.stats.meanDepth).toBeGreaterThan(1);
  });

  const checkerBeer = (mutate?: VDBReferenceMutation) => {
    // A tilted camera so rays cross cell faces on every axis; T exact: the optical depth by 20000 midpoints per ray.
    const tilt = [0.97, 0.1, 0.22];
    const cam = orthoCamera();
    const c2w = [1, 0, 0, 0, 0, 1, 0, 0, -tilt[0]! * 0 - 0.25, -0.2, 0.95, 0, 0, 0, 5, 1];
    const scene: VDBReferenceScene = { camera: { ...cam, cameraToWorld: c2w }, volumes: [{ localToWorld: BOX, sigma: 1.5, albedo: [0, 0, 0], majorantGrid: grid }] };
    const w2l = vdbRefInvert4(BOX);
    const exact = (px: number, py: number) => {
      const x = (((px + 0.5) / W) * 2 - 1) * HALF;
      const y = ((1 - (py + 0.5) / H) * 2 - 1) * HALF;
      const o = [x, y, 5];
      const dz = [-c2w[8]!, -c2w[9]!, -c2w[10]!];
      const l = Math.hypot(dz[0]!, dz[1]!, dz[2]!);
      const d = dz.map((v) => v / l);
      let tau = 0;
      const n = 20000;
      const t0 = 0;
      const t1 = 12;
      for (let i = 0; i < n; i++) {
        const t = t0 + ((i + 0.5) * (t1 - t0)) / n;
        const p = [o[0]! + t * d[0]!, o[1]! + t * d[1]!, o[2]! + t * d[2]!];
        const lp = [0, 1, 2].map((r) => w2l[r]! * p[0]! + w2l[4 + r]! * p[1]! + w2l[8 + r]! * p[2]! + w2l[12 + r]!);
        if (lp.every((v) => Math.abs(v) <= 0.5)) tau += 1.5 * checker.density(lp.map((v) => v + 0.5)) * ((t1 - t0) / n);
      }
      return [0, 0, 0, Math.exp(-tau)];
    };
    return check(scene, [checker], { width: W, height: H, spp: 8, maxBounces: 1, seed: 2 }, 24, exact, mutate);
  };

  it("Beer–Lambert through the checker medium with per-cell majorants = the exact optical depth; the wrong cell fails", () => {
    const ok = checkerBeer();
    expect(passes(ok), JSON.stringify(ok)).toBe(true);
    const bad = checkerBeer("majorant-cell");
    expect(bad.violations).toBeGreaterThan(0);
    expect(passes(bad)).toBe(false);
  });

  it("the dense grid bounds every trilinear read of its cell", () => {
    const dimsT: [number, number, number] = [19, 13, 11];
    const data = new Uint8Array(19 * 13 * 11).map((_, i) => ((i * 37) % 11 === 0 ? 255 : (i * 13) % 97));
    const density = { data, dimensions: dimsT, format: "r8unorm" as const, texelBytes: 1 as const, mip: null };
    const g = vdbReferenceMajorantGridDense({ density, mask: null }, 4);
    expect(g.dims).toEqual([5, 4, 3]);
    let violations = 0;
    let tight = 0;
    for (let i = 0; i < 20000; i++) {
      const uvw = [((i * 0.6180339) % 1), ((i * 0.4142135) % 1), ((i * 0.7320508) % 1)];
      const v = sampleWvdbDenseTexture(density, uvw, 0);
      const c = [0, 1, 2].map((k) => Math.min(Math.floor(uvw[k]! * g.dims[k]!), g.dims[k]! - 1));
      const b = g.values[c[0]! + g.dims[0] * (c[1]! + g.dims[1] * c[2]!)]!;
      if (v > b + 1e-6) violations++;
      if (v > 0.5 * b) tight++;
    }
    expect(violations).toBe(0);
    expect(tight).toBeGreaterThan(1000);
  });

  it("the sparse grid (node stats) bounds every level-0 read and equals the voxel-decoded bound on a fog grid", () => {
    const dimsG: [number, number, number] = [40, 30, 26];
    const values = new Float32Array(40 * 30 * 26);
    for (let x = 0; x < 40; x++) for (let y = 0; y < 30; y++) for (let z = 0; z < 26; z++) {
      const a = Math.max(0, 1 - Math.hypot(x - 12, y - 14, z - 10) / 9) * 1.4;
      const b = Math.max(0, 1 - Math.hypot(x - 29, y - 18, z - 16) / 7) * 0.6;
      const v = a + b;
      values[(x * 30 + y) * 26 + z] = v > 0.02 ? v : 0;
    }
    const image = vdbBuildFromDense(values, dimsG, { origin: [-5, 3, 7], voxelSize: 0.25 });
    const box = { min: [-5, 3, 7], max: [34, 32, 32] };
    const g = vdbReferenceMajorantGridSparse(image, box);
    const gv = vdbReferenceMajorantGridSparse(image, box, { from: "voxels", leafValues: (im, a) => leafVoxels(im, a).values });
    expect(g.dims).toEqual([5, 4, 4]);
    for (let i = 0; i < g.values.length; i++) expect(g.values[i]!).toBeGreaterThanOrEqual(gv.values[i]! - 1e-6);
    const vol = { lod: "none" as const, clamp01: false, density: { image, box } };
    let violations = 0;
    let empty = 0;
    for (let i = 0; i < 20000; i++) {
      const uvw = [((i * 0.6180339) % 1), ((i * 0.4142135) % 1), ((i * 0.7320508) % 1)];
      const v = vdbSparseDensityCpu(vol, uvw, 0).value;
      const c = [0, 1, 2].map((k) => Math.min(Math.floor(uvw[k]! * g.dims[k]!), g.dims[k]! - 1));
      const b = g.values[c[0]! + g.dims[0] * (c[1]! + g.dims[1] * c[2]!)]!;
      if (v > b + 1e-6) violations++;
      if (b === 0) empty++;
    }
    expect(violations).toBe(0);
    expect(empty).toBeGreaterThan(0);
  });
});
