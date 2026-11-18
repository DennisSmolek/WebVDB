import { describe, expect, it } from "vitest";
import {
  vdbFrameGridMipLevel,
  vdbFrameHenyeyGreenstein,
  vdbFrameIsotropicDiffusion,
  vdbFrameMieAsymmetry,
  vdbFrameMieChannel,
  vdbFrameMieFormat,
  vdbFrameMieTable,
  vdbFrameMieTableSize,
  vdbFrameMieTexel,
  vdbFrameMieTexels,
  vdbFrameNeuralCheck,
  vdbFrameNeuralForward,
  vdbFrameNeuralInputs,
  vdbFrameNeuralLayout,
  vdbFrameNeuralWeights,
  vdbFrameOctaveLight,
  vdbFramePhaseColor,
  vdbFramePhaseWeight,
  vdbFramePhaseWeightScaled,
  vdbFrameScatteringOctaves,
  vdbFrameShadowMarch,
  vdbFrameShadowMarchDepth,
  vdbFrameShadowOpticalDepth,
  vdbFrameShadowTotal,
  vdbFrameShadowTransmittance,
  vdbFrameShadowedLight,
  vdbFrameShadowedLightColor,
  vdbFrameTwoSidedDiffusion,
  type VDBFrameAtlas,
} from "../../src/tools/index.js";
import { frameMieRgbTable, frameMieTable } from "../../src/tools/frame/mie-table.js";

// K3.4 (R1 §6 Step 4): the CPU twins of his phase functions and multiple-scattering modes. Each test pins
// a property of his formula with an oracle that does not reuse the twin's code: the regenerated Mie table,
// sphere integrals of the phases, the geometric sums of the octaves, a finite-difference solve of the
// two-sided diffusion problem, and an independent matrix form of the network.

type V3 = [number, number, number];
type V4 = [number, number, number, number];

const N = vdbFrameMieTableSize;
const angle = (j: number) => Math.PI * (j / (N - 1)) ** 2;

/** ∫ f(cos θ) dΩ / 4π over the table angles (trapezoids in θ), as his normalization. */
function sphereMean(values: (j: number) => number): number {
  let sum = 0;
  for (let j = 1; j < N; j++) sum += 0.5 * (values(j - 1) * Math.sin(angle(j - 1)) + values(j) * Math.sin(angle(j))) * (angle(j) - angle(j - 1));
  return (2 * Math.PI * sum) / (4 * Math.PI);
}

/** ∫ f(μ) dμ / 2 on [−1, 1] (Simpson, n even): the mean of a phase / isotropic over the sphere. */
function muMean(f: (mu: number) => number, n = 20000): number {
  const h = 2 / n;
  let s = f(-1) + f(1);
  for (let i = 1; i < n; i++) s += f(-1 + i * h) * (i % 2 ? 4 : 2);
  return (s * h) / 3 / 2;
}

describe("Lorenz-Mie tables (his BHMIE generator, ported)", () => {
  it("regenerates the committed 'lorenz-mie' table digit for digit (his numbers, 6 significant digits)", () => {
    const t0 = performance.now();
    const mie = vdbFrameMieTable(vdbFrameMieChannel.wavelength, vdbFrameMieChannel.refractiveIndex);
    let differ = 0;
    for (let j = 0; j < N; j++) if (vdbFrameMieFormat(mie.table[j]!) !== Math.fround(frameMieTable[j]!) && Math.fround(vdbFrameMieFormat(mie.table[j]!)) !== frameMieTable[j]!) differ++;
    expect(differ).toBe(0);
    expect(mie.asymmetry.toFixed(4)).toBe("0.8526");
    expect(vdbFrameMieFormat(mie.peak)).toBe(2746.25);
    expect(performance.now() - t0).toBeLessThan(30_000);
  }, 60_000);

  it("a different refractive index is a different table (the check above can fail)", () => {
    const other = vdbFrameMieTable(0.55, 1.34, { tableSize: 256, radiusSteps: 120 });
    const base = vdbFrameMieTable(0.55, 1.335, { tableSize: 256, radiusSteps: 120 });
    let maxRel = 0;
    for (let j = 0; j < 256; j++) maxRel = Math.max(maxRel, Math.abs(other.table[j]! - base.table[j]!) / base.table[j]!);
    expect(maxRel).toBeGreaterThan(1e-3);
  });

  it("each committed table is a normalized phase (mean 1 over the sphere) with his asymmetries", () => {
    expect(sphereMean((j) => frameMieTable[j]!)).toBeCloseTo(1, 4);
    expect(sphereMean((j) => frameMieTable[j]! * Math.cos(angle(j)))).toBeCloseTo(vdbFrameMieAsymmetry, 3);
    const g = [0.8497, 0.8533, 0.8565];
    for (let c = 0; c < 3; c++) {
      expect(sphereMean((j) => frameMieRgbTable[j * 3 + c]!)).toBeCloseTo(1, 4);
      expect(sphereMean((j) => frameMieRgbTable[j * 3 + c]! * Math.cos(angle(j)))).toBeCloseTo(g[c]!, 3);
    }
    // Blue scatters more forward than red (forward peak 4075 vs 1980)
    expect(frameMieRgbTable[2]!).toBeGreaterThan(frameMieRgbTable[0]!);
  });

  it("the texel read: exact at texel angles, linear between, rgb in r/g/b and 'lorenz-mie' in alpha", () => {
    const texels = vdbFrameMieTexels();
    for (const j of [0, 1, 100, 1000, 2046]) {
      const t = vdbFrameMieTexel(Math.cos(angle(j)));
      expect(t[3]).toBeCloseTo(frameMieTable[j]!, 1);
      expect(t[0] / frameMieRgbTable[j * 3]!).toBeCloseTo(1, 4);
      expect(texels[j * 4 + 2]).toBe(frameMieRgbTable[j * 3 + 2]);
    }
    // Half way between texels 500 and 501 in u
    const u = 500.5 / (N - 1);
    const t = vdbFrameMieTexel(Math.cos(Math.PI * u * u));
    expect(t[3]).toBeCloseTo((frameMieTable[500]! + frameMieTable[501]!) / 2, 3);
    // cos clamps: backward is the last texel pair at its end
    expect(vdbFrameMieTexel(-1.5)[3]).toBeCloseTo(frameMieTable[N - 1]!, 4);
  });
});

describe("phase functions", () => {
  const ray: V3 = [0, 0, 1];
  const dir = (c: number): V3 => [Math.sqrt(1 - c * c), 0, c];

  it("Henyey-Greenstein / isotropic integrates to 1 with mean cosine g; dual-lobe mixes the means", () => {
    for (const g of [-0.5, 0, 0.3, 0.8]) {
      expect(muMean((mu) => vdbFrameHenyeyGreenstein(mu, g))).toBeCloseTo(1, 5);
      expect(muMean((mu) => mu * vdbFrameHenyeyGreenstein(mu, g))).toBeCloseTo(g, 4);
    }
    const dual: V4 = [0.8, -0.3, 0.2, 0];
    expect(muMean((mu) => mu * vdbFramePhaseWeight(dual, ray, dir(mu)))).toBeCloseTo(0.8 * 0.8 + 0.2 * -0.3, 4);
  });

  it("'lorenz-mie' (mode 1) is the table; mode m mixes it in by m", () => {
    for (const c of [-0.9, 0, 0.7, 0.99]) {
      expect(vdbFramePhaseWeight([0, 0, 0, 1], ray, dir(c))).toBeCloseTo(vdbFrameMieTexel(c)[3], 9);
      const hg = vdbFrameHenyeyGreenstein(c, 0.5);
      expect(vdbFramePhaseWeight([0.5, 0, 0, 0.25], ray, dir(c))).toBeCloseTo(hg + (vdbFrameMieTexel(c)[3] - hg) * 0.25, 9);
    }
  });

  it("framePhaseColor: one phase (flag 0), 'lorenz-mie-rgb' tables, phaseGExponent g · (λ / 550)^−e per channel", () => {
    const dual: V4 = [0.8, -0.3, 0.2, 0];
    const c = 0.6;
    const w = vdbFramePhaseWeight(dual, ray, dir(c));
    expect(vdbFramePhaseColor(dual, [1, 1, 0], ray, dir(c))).toEqual([w, w, w]);
    const rgb = vdbFramePhaseColor([0, 0, 0, 2], [1, 1, 1], ray, dir(c));
    const t = vdbFrameMieTexel(c);
    expect(rgb).toEqual([t[0], t[1], t[2]]);
    // His getFramePhaseSpectrum: (650/550)^−e and (450/550)^−e; green keeps g.
    const e = 1;
    const spectrum: V3 = [Math.pow(650 / 550, -e), Math.pow(450 / 550, -e), 1];
    const hg = (mu: number, g: number) => (1 - g * g) / Math.pow(1 + g * g - 2 * g * mu, 1.5);
    const lobe = (s: number) => {
      const gf = Math.min(0.99, Math.max(-0.99, 0.8 * s));
      const gb = Math.min(0.99, Math.max(-0.99, -0.3 * s));
      return hg(c, gf) * 0.8 + hg(c, gb) * 0.2;
    };
    const color = vdbFramePhaseColor(dual, spectrum, ray, dir(c));
    expect(color[0]).toBeCloseTo(lobe(spectrum[0]), 10);
    expect(color[1]).toBeCloseTo(lobe(1), 10);
    expect(color[2]).toBeCloseTo(lobe(spectrum[1]), 10);
    // Shorter wavelengths: larger g, more light near the forward direction
    const forward = vdbFramePhaseColor(dual, spectrum, ray, dir(0.995));
    expect(forward[2]).toBeGreaterThan(forward[0]);
  });

  it("framePhaseWeightScaled: every g × s; the Mie modes use HG(0.8526 s)", () => {
    const c = 0.3;
    expect(vdbFramePhaseWeightScaled([0.8, -0.3, 0.2, 0], c, 0.5)).toBeCloseTo(vdbFrameHenyeyGreenstein(c, 0.4) * 0.8 + vdbFrameHenyeyGreenstein(c, -0.15) * 0.2, 12);
    expect(vdbFramePhaseWeightScaled([0, 0, 0, 2], c, 0.5)).toBeCloseTo(vdbFrameHenyeyGreenstein(c, 0.4263), 12);
    expect(vdbFramePhaseWeightScaled([0.8, -0.3, 0.2, 0], c, 0)).toBe(1);
  });
});

describe("scattering octaves (multipleScattering 'octaves')", () => {
  const settings = { multipleScattering: "octaves", scatteringOctaves: 4, scatteringOctaveExtinction: 0.5, scatteringOctaveContribution: 0.5, scatteringOctavePhase: 0.5 };

  it("getFrameScatteringOctaves: his clamps; off for every other mode", () => {
    expect(vdbFrameScatteringOctaves(settings)).toEqual([4, 0.5, 0.5, 0.5]);
    expect(vdbFrameScatteringOctaves({ ...settings, scatteringOctaves: 20, scatteringOctaveExtinction: 2, scatteringOctavePhase: -1 })).toEqual([8, 1, 0.5, 0]);
    expect(vdbFrameScatteringOctaves({ ...settings, scatteringOctaves: 0 })[0]).toBe(1);
    for (const ms of [true, false, "two-sided", "neural"]) expect(vdbFrameScatteringOctaves({ ...settings, multipleScattering: ms })).toEqual([0, 0, 0, 0]);
  });

  it("isotropic octave light is Σ bⁿ exp(−aⁿ τ); with a = 1 the geometric sum exp(−τ) b (1 − bᴺ) / (1 − b)", () => {
    for (const tau of [0, 0.3, 2, 9]) {
      let sum = 0;
      for (let n = 1; n <= 4; n++) sum += 0.5 ** n * Math.exp(-(0.5 ** n) * tau);
      expect(vdbFrameOctaveLight([4, 0.5, 0.5, 0.5], tau, null, 0)).toBeCloseTo(sum, 12);
      expect(vdbFrameOctaveLight([5, 1, 0.6, 0.5], tau, null, 0)).toBeCloseTo((Math.exp(-tau) * 0.6 * (1 - 0.6 ** 5)) / 0.4, 12);
    }
  });

  it("phased octaves: octave n has HG(g cⁿ), so the octaves flatten toward isotropic", () => {
    const tau = 1.5;
    const c = 0.9;
    let sum = 0;
    for (let n = 1; n <= 3; n++) sum += 0.5 ** n * vdbFrameHenyeyGreenstein(c, 0.8 * 0.5 ** n) * Math.exp(-(0.5 ** n) * tau);
    expect(vdbFrameOctaveLight([3, 0.5, 0.5, 0.5], tau, [0.8, 0, 0, 0], c)).toBeCloseTo(sum, 12);
  });

  it("the shadow slot holds 1 / (1 + τ); total = exp(−τ) + octaves; the march stops on the last octave's depth", () => {
    const oct: V4 = [4, 0.5, 0.5, 0.5];
    const depths: V3 = [3, 1, 2];
    const t = vdbFrameShadowTransmittance(depths, 0.42, oct);
    expect(t[0]).toBeCloseTo(Math.exp(-3), 14);
    expect(t[1]).toBeCloseTo(1 / 4, 14);
    expect(vdbFrameShadowOpticalDepth(t[1])).toBeCloseTo(3, 12);
    expect(vdbFrameShadowTotal(t[1], oct)).toBeCloseTo(Math.exp(-3) + vdbFrameOctaveLight(oct, 3, null, 0), 12);
    expect(vdbFrameShadowTotal(0.37)).toBe(0.37);
    expect(vdbFrameShadowMarchDepth(depths, 0.42, oct)).toBeCloseTo(3 / 16, 14);
    // An unshadowed light (τ = 0) gets 1 + b + b² + … : the octaves add light outside the medium too (his).
    expect(vdbFrameShadowTotal(1, oct)).toBeCloseTo(1 + 0.5 + 0.25 + 0.125 + 0.0625, 12);
    // The inject march runs 2^4 = 16 times deeper before it stops.
    const samples = Array.from({ length: 200 }, () => 0.5);
    expect(vdbFrameShadowMarch(samples, [1, 1, 1], 0)[0]).toBeGreaterThan(Math.exp(-8.6));
    expect(vdbFrameShadowMarch(samples, [1, 1, 1], 0, oct)[0]).toBeLessThan(Math.exp(-99));
  });

  it("shadowed light: phase × direct + octave light (octaves) or + (total − direct)", () => {
    const ray: V3 = [0, 0, 1];
    const light: V3 = [0.6, 0, 0.8];
    const phase: V4 = [0.8, -0.3, 0.2, 0];
    const oct: V4 = [4, 0.5, 0.5, 0.5];
    const shadow: [number, number] = [Math.exp(-2), 1 / 3];
    const w = vdbFramePhaseWeight(phase, ray, light);
    expect(vdbFrameShadowedLight(phase, ray, light, shadow, oct)).toBeCloseTo(w * shadow[0] + vdbFrameOctaveLight(oct, 2, phase, 0.8), 12);
    expect(vdbFrameShadowedLight(phase, ray, light, [0.2, 0.5])).toBeCloseTo(w * 0.2 + 0.3, 12);
    expect(vdbFrameShadowedLightColor(phase, [1, 1, 0], oct, ray, light, shadow)[1]).toBeCloseTo(vdbFrameShadowedLight(phase, ray, light, shadow, oct), 12);
  });
});

describe("two-sided slab diffusion (multipleScattering 'two-sided')", () => {
  /** φ'' − k² φ = −3 e^(−x) on [0, L], φ(0) = z₀ φ'(0), φ(L) = −z₀ φ'(L): second-order finite differences. */
  function solveSlab(L: number, k: number, n = 4000): (x: number) => number {
    const z0 = 2 / 3;
    const h = L / n;
    // Unknowns φ_0..φ_n; ghost points from the Robin conditions (central differences at the ends).
    const a = new Float64Array(n + 1);
    const b = new Float64Array(n + 1);
    const c = new Float64Array(n + 1);
    const d = new Float64Array(n + 1);
    for (let i = 0; i <= n; i++) {
      a[i] = 1 / (h * h);
      c[i] = 1 / (h * h);
      b[i] = -2 / (h * h) - k * k;
      d[i] = -3 * Math.exp(-i * h);
    }
    // x = 0: φ'(0) = φ(0)/z0 → φ_{−1} = φ_1 − 2h φ_0 / z0
    b[0] = b[0]! - (2 * h) / z0 / (h * h);
    c[0] = 2 / (h * h);
    // x = L: φ'(L) = −φ(L)/z0 → φ_{n+1} = φ_{n−1} − 2h φ_n / z0
    b[n] = b[n]! - (2 * h) / z0 / (h * h);
    a[n] = 2 / (h * h);
    // Thomas algorithm
    const cp = new Float64Array(n + 1);
    const dp = new Float64Array(n + 1);
    cp[0] = c[0]! / b[0]!;
    dp[0] = d[0]! / b[0]!;
    for (let i = 1; i <= n; i++) {
      const m = b[i]! - a[i]! * cp[i - 1]!;
      cp[i] = c[i]! / m;
      dp[i] = (d[i]! - a[i]! * dp[i - 1]!) / m;
    }
    const phi = new Float64Array(n + 1);
    phi[n] = dp[n]!;
    for (let i = n - 1; i >= 0; i--) phi[i] = dp[i]! - cp[i]! * phi[i + 1]!;
    return (x: number) => {
      const t = Math.min(n - 1e-9, x / h);
      const i = Math.floor(t);
      return phi[i]! + (phi[i + 1]! - phi[i]!) * (t - i);
    };
  }

  it("is the solution of his boundary-value problem (an independent finite-difference solve) plus the beam", () => {
    for (const [f, back] of [
      [0.5, 3],
      [2, 2],
      [4, 0.25],
      [0.05, 6],
    ] as [number, number][]) {
      const L = f + back;
      const k = (Math.SQRT2 * Math.PI) / (L + 4 / 3);
      if (Math.abs(1 - k * k) < 0.01) continue;
      const phi = solveSlab(L, k);
      const expected = phi(f) + Math.exp(-f);
      expect(vdbFrameTwoSidedDiffusion(0, f, 0, back) / expected).toBeCloseTo(1, 4);
      // Absorption multiplies by exp(−effective front)
      expect(vdbFrameTwoSidedDiffusion(0, f, 0.7, back)).toBeCloseTo(Math.max(0, expected * Math.exp(-0.7)), 3);
    }
  });

  it("never below the direct light; continuous through the k = 1 shift; deeper is darker", () => {
    expect(vdbFrameTwoSidedDiffusion(0.95, 0.01, 0, 0.01)).toBeGreaterThanOrEqual(0.95);
    // k0 = 1 at slab = √2 π − 4/3
    const slab = Math.SQRT2 * Math.PI - 4 / 3;
    const a = vdbFrameTwoSidedDiffusion(0, slab / 2, 0, slab / 2 - 1e-3);
    const b = vdbFrameTwoSidedDiffusion(0, slab / 2, 0, slab / 2 + 1e-3);
    expect(Math.abs(a - b)).toBeLessThan(0.01);
    let last = Infinity;
    for (const f of [0.1, 0.5, 1, 2, 4]) {
      const v = vdbFrameTwoSidedDiffusion(0, f, 0, 4);
      expect(v).toBeLessThan(last);
      last = v;
    }
  });

  it("the medium behind the point adds light (back-scatter); a thin slab loses light out of its far side, below 1D", () => {
    // 1D diffusion (frameShadowDepth) of the same reduced front depth, no absorption: 1 / (1 + 0.75 f), blind to the back.
    for (const f of [0.5, 1, 3, 6]) {
      let last = 0;
      for (const back of [0.2, 1, 5, 20]) {
        const v = vdbFrameTwoSidedDiffusion(0, f, 0, back);
        expect(v).toBeGreaterThan(last);
        last = v;
      }
    }
    expect(vdbFrameTwoSidedDiffusion(0, 3, 0, 0.2)).toBeLessThan(1 / (1 + 0.75 * 3));
    expect(vdbFrameTwoSidedDiffusion(0, 3, 0, 5)).toBeGreaterThan(1 / (1 + 0.75 * 3));
    expect(Number.isFinite(vdbFrameTwoSidedDiffusion(0, 3, 0, 1e-6))).toBe(true);
    // The isotropic pair (frameIsotropicDiffusion) still sums to 2 in a white slab (the furnace property).
    expect(vdbFrameIsotropicDiffusion(3, 1, 0) + vdbFrameIsotropicDiffusion(1, 3, 0)).toBeCloseTo(2, 12);
  });
});

describe("neural indirect network (multipleScattering 'neural')", () => {
  const net = vdbFrameNeuralWeights;

  it("his network fits the shader layout: 20 → 32 → 32 → 1, 441 vec4s", () => {
    expect([net.levels, net.inputs, net.hidden]).toEqual([6, 20, 32]);
    expect(vdbFrameNeuralCheck(net)).toBeNull();
    expect(vdbFrameNeuralLayout(net)).toMatchObject({ w1: 0, b1: 160, w2: 168, b2: 424, w3: 432, b3: 440, vectors: 441 });
    expect(vdbFrameNeuralCheck({ ...net, weights: net.weights.subarray(4) })).toMatch(/expected 1764/);
    expect(vdbFrameNeuralCheck({ ...net, hidden: 30 })).toMatch(/multiples of 4/);
  });

  it("the forward pass equals his TSL's vec4 form (groups of 4 units, dot products of vec4 inputs)", () => {
    const w = net.weights;
    const vec = (i: number): V4 => [w[i * 4]!, w[i * 4 + 1]!, w[i * 4 + 2]!, w[i * 4 + 3]!];
    const dot4 = (a: V4, b: V4) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    const dense = (inputs: V4[], weightOffset: number, biasOffset: number): V4[] => {
      const out: V4[] = [];
      for (let group = 0; group < 8; group++) {
        const bias = vec(biasOffset + group);
        const values = [0, 1, 2, 3].map((component) => {
          const row = weightOffset + (group * 4 + component) * inputs.length;
          return inputs.reduce((sum, input, i) => sum + dot4(vec(row + i), input), bias[component]!);
        });
        out.push(values.map((v) => Math.max(v, 0)) as V4);
      }
      return out;
    };
    let seed = 7;
    const rand = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31) * 3;
    for (let trial = 0; trial < 20; trial++) {
      const x = Array.from({ length: 20 }, rand);
      const inputs: V4[] = [0, 1, 2, 3, 4].map((g) => x.slice(g * 4, g * 4 + 4) as V4);
      const h2 = dense(dense(inputs, 0, 160), 168, 424);
      const o = h2.reduce((sum, v, i) => sum + dot4(vec(432 + i), v), vec(440)[0]);
      expect(vdbFrameNeuralForward(net, x)).toBeCloseTo(Math.exp(Math.min(o, 4)), 10);
    }
  });

  it("inputs: log(1 + σ h 2^k) at, toward and away from the light; log(1 + τ); √(absorption / extinction) of level 1", () => {
    const sample = (level: number, p: V3): [number, number] => (p[0] > 0 ? [2 + level, 0.5] : [0, 0]);
    const x = vdbFrameNeuralInputs(sample, 6, 0.25, [1, 0, 0], [1, 0, 0], 3);
    expect(x).toHaveLength(20);
    for (let k = 0; k < 6; k++) {
      const v = Math.log((2 + k) * 0.25 * 2 ** k + 1);
      expect(x[k]).toBeCloseTo(v, 12);
      expect(x[6 + k]).toBeCloseTo(v, 12);
    }
    // Away from the light at h 2^k ≥ 1 leaves the medium (x ≤ 0) for k ≥ 2
    expect(x[12]).toBeCloseTo(Math.log(2 * 0.25 + 1), 12);
    expect(x[14]).toBe(0);
    expect(x[18]).toBeCloseTo(Math.log(4), 12);
    expect(x[19]).toBeCloseTo(Math.sqrt(0.5 / 3), 12);
  });

  it("predicts more scattered light deeper in a white medium than at its edge, bounded by exp(4)", () => {
    // A uniform white slab x ∈ [0, 20], lit along +x: a voxel near the lit face vs one deep inside.
    const sigma = 1;
    const sample = (_level: number, p: V3): [number, number] => (p[0] >= 0 && p[0] <= 20 ? [sigma, 0] : [0, 0]);
    const at = (depth: number) => vdbFrameNeuralForward(net, vdbFrameNeuralInputs(sample, 6, 0.5, [20 - depth, 0, 0], [1, 0, 0], depth * sigma));
    const edge = at(0.25);
    const deep = at(8);
    for (const v of [edge, deep]) {
      expect(v).toBeGreaterThan(0);
      expect(v).toBeLessThanOrEqual(Math.exp(4));
    }
    // Scattered light (total − direct) dominates deep inside where the direct light is ~exp(−8).
    expect(deep).toBeGreaterThan(Math.exp(-8));
  });

  it("grid mips: each texel is the 2×2×2 mean of the level before (edges clamped)", () => {
    // A 4³ grid in one cascade (2 × 2 tiles of 4 × 4), RG = (x + 4y + 16z, 1)
    const res = 4;
    const columns = 2;
    const data = new Float32Array(8 * 8 * 2);
    for (let z = 0; z < 4; z++)
      for (let y = 0; y < 4; y++)
        for (let x = 0; x < 4; x++) {
          const tx = (z % columns) * res + x;
          const ty = Math.floor(z / columns) * res + y;
          data[(ty * 8 + tx) * 2] = x + 4 * y + 16 * z;
          data[(ty * 8 + tx) * 2 + 1] = 1;
        }
    const atlas: VDBFrameAtlas = { data, width: 8, height: 8, channels: 2 };
    const mip = vdbFrameGridMipLevel(atlas, res, columns, 0, 2);
    expect([mip.atlas.width, mip.atlas.height, mip.columns]).toEqual([4, 2, 2]);
    // Voxel (0, 0, 0) of level 1: mean of x, y, z ∈ {0, 1} = 0.5 + 2 + 8
    expect(mip.atlas.data[0]).toBeCloseTo(10.5, 12);
    expect(mip.atlas.data[1]).toBe(1);
    // Voxel (1, 1, 1): x, y, z ∈ {2, 3}: 2.5 + 10 + 40; tile z = 1 is the second column
    expect(mip.atlas.data[(1 * 4 + 2 + 1) * 2]).toBeCloseTo(52.5, 12);
    // Level 2 from level 1: 1 voxel, the mean of all
    const top = vdbFrameGridMipLevel(mip.atlas, 2, mip.columns, 0, 1);
    expect(top.atlas.data[0]).toBeCloseTo(31.5, 12);
  });
});
