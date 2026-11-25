import { describe, expect, it } from "vitest";
import {
  VDBToolsError,
  vdbFloatToHalf,
  vdbFrameMultiScaleLayout,
  vdbFrameMultiScaleLight,
  vdbFrameMultiScaleMips,
  vdbFrameMultiScaleSample,
  vdbFrameMultiScaleToHalf,
  vdbFrameTexture3DLinear,
  type VDBFrameTexture3D,
} from "../../src/tools/internal.js";

// K7.3: the CPU twins of 'multi-scale' multiple scattering (mjurczyk 4e2d46f, Disney "Deep Scattering" §4.2):
// the binomial mip chain and the 5-tap-per-level lookup. Oracles: his `buildMips(…, 'binomial')` re-implemented
// from his features.mjs (clamp edges, cubic, power-of-two), mass conservation of the zero-padded chain, the
// closed form of a uniform medium, and the tap geometry. The GPU kernels run against these in e2e/k7-multi-scale.

type V3 = [number, number, number];

/** Mulberry32: a fixed-seed generator. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A 2-channel grid with a blob of seeded noise, zero within `margin` texels of the edges. */
function blob(size: V3, seed: number, margin = 0): VDBFrameTexture3D {
  const r = rng(seed);
  const data = new Float64Array(size[0] * size[1] * size[2] * 2);
  for (let z = margin; z < size[2] - margin; z++) {
    for (let y = margin; y < size[1] - margin; y++) {
      for (let x = margin; x < size[0] - margin; x++) {
        const i = (x + (y + z * size[1]) * size[0]) * 2;
        data[i] = r() * 4;
        data[i + 1] = data[i]! * r();
      }
    }
  }
  return { data, dimensions: size, channels: 2 };
}

/** His features.mjs buildMips 'binomial' (cubic, two channels, sizes resolution >> level, clamped taps), as he wrote it. */
function hisBinomialMips(resolution: number, base: ArrayLike<number>, levels: number): Float64Array[] {
  const w = [1, 3, 3, 1];
  const out: { resolution: number; data: ArrayLike<number> }[] = [{ resolution, data: base }];
  for (let level = 1; level < levels; level++) {
    const source = out[level - 1]!;
    const sourceResolution = source.resolution;
    const res = Math.max(1, resolution >> level);
    const target = new Float64Array(res * res * res * 2);
    for (let z = 0; z < res; z++) {
      for (let y = 0; y < res; y++) {
        for (let x = 0; x < res; x++) {
          let extinction = 0;
          let absorption = 0;
          const index = (x + (y + z * res) * res) * 2;
          const clampIndex = (value: number) => Math.min(Math.max(value, 0), sourceResolution - 1);
          for (let i = 0; i < 64; i++) {
            const [tx, ty, tz] = [i & 3, (i >> 2) & 3, (i >> 4) & 3];
            const sx = clampIndex(x * 2 - 1 + tx);
            const sy = clampIndex(y * 2 - 1 + ty);
            const sz = clampIndex(z * 2 - 1 + tz);
            const sourceIndex = (sx + (sy + sz * sourceResolution) * sourceResolution) * 2;
            const weight = w[tx]! * w[ty]! * w[tz]!;
            extinction += source.data[sourceIndex]! * weight;
            absorption += source.data[sourceIndex + 1]! * weight;
          }
          target[index] = extinction / 512;
          target[index + 1] = absorption / 512;
        }
      }
    }
    out.push({ resolution: res, data: target });
  }
  return out.slice(1).map((l) => l.data as Float64Array);
}

const sum = (a: ArrayLike<number>, channel: number, channels: number) => {
  let s = 0;
  for (let i = channel; i < a.length; i += channels) s += a[i]!;
  return s;
};

describe("vdbFrameMultiScaleLayout", () => {
  it("halves exactly from ⌈size / 2^(levels−1)⌉ (no drift on odd sizes); extents are ⌈size / 2^k⌉", () => {
    const l = vdbFrameMultiScaleLayout([37, 24, 50], 6);
    expect(l.mips.map((m) => m.dimensions)).toEqual([
      [32, 16, 32],
      [16, 8, 16],
      [8, 4, 8],
      [4, 2, 4],
      [2, 1, 2],
    ]);
    expect(l.mips.map((m) => m.extent)).toEqual([
      [19, 12, 25],
      [10, 6, 13],
      [5, 3, 7],
      [3, 2, 4],
      [2, 1, 2],
    ]);
    // His default shadow grid (64³, 6 levels) is his sizes exactly: 32, 16, 8, 4, 2.
    expect(vdbFrameMultiScaleLayout([64, 64, 64]).mips.map((m) => m.dimensions[0])).toEqual([32, 16, 8, 4, 2]);
    expect(() => vdbFrameMultiScaleLayout([0, 4, 4])).toThrow(VDBToolsError);
    expect(() => vdbFrameMultiScaleLayout([4, 4, 4], 1)).toThrow(/levels must be an integer from 2 to 8/);
    expect(() => vdbFrameMultiScaleLayout([4, 4, 4], 9)).toThrow(VDBToolsError);
  });
});

describe("vdbFrameMultiScaleMips (twin of csMsLevel)", () => {
  it("edge 'clamp' on a power-of-two cube = his buildMips 'binomial' (features.mjs, 4e2d46f)", () => {
    const g = blob([32, 32, 32], 7);
    const ours = vdbFrameMultiScaleMips(g, { levels: 6, edge: "clamp" });
    const his = hisBinomialMips(32, g.data, 6);
    expect(ours.levels.length).toBe(5);
    ours.levels.forEach((level, i) => {
      expect(level.length).toBe(his[i]!.length);
      let max = 0;
      for (let t = 0; t < level.length; t++) max = Math.max(max, Math.abs(level[t]! - his[i]![t]!));
      expect(max).toBeLessThan(1e-12);
    });
  });

  it("edge 'zero' keeps the mass (Σ L_k · 8 = Σ L_(k−1)) on odd, non-cubic sizes while nothing blurs past the edges; 'clamp' does not", () => {
    // Level k blurs ~3 · 2^k base texels: with density 7 texels from every face, levels 1 to 3 lose nothing (the
    // padding past ⌈size / 2^k⌉ holds what blurs out at the far faces).
    const g = blob([37, 24, 50], 11, 7);
    const zero = vdbFrameMultiScaleMips(g, { levels: 4 });
    const clamp = vdbFrameMultiScaleMips(blob([37, 24, 50], 11, 0), { levels: 6, edge: "clamp" });
    for (const c of [0, 1]) {
      let below = sum(g.data, c, 2);
      for (const level of zero.levels) {
        const s = sum(level, c, 2);
        expect(s * 8).toBeCloseTo(below, 8);
        below = s;
      }
    }
    // Clamped edges repeat the edge texels (his rule): the mass of a grid with density at its border grows.
    const g0 = blob([37, 24, 50], 11, 0);
    expect(sum(clamp.levels[0]!, 0, 2) * 8).toBeGreaterThan(sum(g0.data, 0, 2) * 1.01);
  });

  it("a constant field stays constant where every tap is inside; clamp padding repeats the last valid texel", () => {
    const size: V3 = [16, 16, 16];
    const data = new Float64Array(16 ** 3 * 2).fill(0.75);
    const g: VDBFrameTexture3D = { data, dimensions: size, channels: 2 };
    const zero = vdbFrameMultiScaleMips(g, { levels: 3 });
    // Level 1 (8³): interior texels (taps 2x − 1 … 2x + 2 inside 0 … 15) are 0.75, the borders less.
    const at = (x: number, y: number, z: number) => zero.levels[0]![(x + (y + z * 8) * 8) * 2]!;
    expect(at(3, 4, 5)).toBeCloseTo(0.75, 14);
    expect(at(0, 4, 5)).toBeCloseTo(0.75 * (7 / 8), 14);
    const clamp = vdbFrameMultiScaleMips(g, { levels: 3, edge: "clamp" });
    for (const level of clamp.levels) for (let i = 0; i < level.length; i++) expect(level[i]).toBeCloseTo(0.75, 14);
    // Odd size, clamp, 3 levels: level 1 of 5 texels has extent 3 in a padded size of 4 → texel 3 repeats texel 2.
    const odd = vdbFrameMultiScaleMips(blob([5, 5, 5], 3), { levels: 3, edge: "clamp" });
    const o = odd.layout.mips[0]!;
    expect([o.dimensions, o.extent]).toEqual([
      [4, 4, 4],
      [3, 3, 3],
    ]);
    const l1 = odd.levels[0]!;
    const t = (x: number, y: number, z: number) => (x + (y + z * 4) * 4) * 2;
    expect(l1[t(3, 1, 2)]).toBe(l1[t(2, 1, 2)]);
    expect(l1[t(3, 3, 3)]).toBe(l1[t(2, 2, 2)]);
  });

  it("a region at an origin (one cascade of a stacked grid) and the channel selection", () => {
    const stacked = blob([20, 20, 40], 5);
    const region = blob([20, 20, 20], 0);
    // Copy the second cascade (z 20 … 39) out.
    (region.data as Float64Array).set((stacked.data as Float64Array).subarray(20 * 20 * 20 * 2));
    const a = vdbFrameMultiScaleMips(stacked, { origin: [0, 0, 20], size: [20, 20, 20], levels: 4 });
    const b = vdbFrameMultiScaleMips(region, { levels: 4 });
    expect(a.levels).toEqual(b.levels);
    // channels [1]: one channel, absorption only.
    const one = vdbFrameMultiScaleMips(region, { levels: 4, channels: [1] });
    expect(one.channels).toBe(1);
    one.levels.forEach((level, i) => {
      for (let t = 0; t < level.length; t++) expect(level[t]).toBe(b.levels[i]![t * 2 + 1]);
    });
    expect(() => vdbFrameMultiScaleMips(region, { channels: [2] })).toThrow(VDBToolsError);
    expect(() => vdbFrameMultiScaleMips(stacked, { origin: [0, 0, 30], size: [20, 20, 20] })).toThrow(/outside the source/);
  });

  it("vdbFrameMultiScaleToHalf: round to nearest even, channels padded with 0", () => {
    const h = vdbFrameMultiScaleToHalf([1, 0.1, 65504, 1e-8], 2, 4);
    expect([...h]).toEqual([vdbFloatToHalf(1), vdbFloatToHalf(0.1), 0, 0, vdbFloatToHalf(65504), vdbFloatToHalf(1e-8), 0, 0]);
    expect(vdbFloatToHalf(0.1)).toBe(0x2e66);
  });
});

describe("vdbFrameTexture3DLinear (the GPU's linear filter, clamp-to-edge)", () => {
  it("texel centres at i + ½, linear between, clamped past the edges", () => {
    const t: VDBFrameTexture3D = { data: [0, 10, 20, 30], dimensions: [4, 1, 1], channels: 1 };
    expect(vdbFrameTexture3DLinear(t, [0.5, 0.5, 0.5])[0]).toBe(0);
    expect(vdbFrameTexture3DLinear(t, [1.75, 0.5, 0.5])[0]).toBeCloseTo(12.5, 12);
    expect(vdbFrameTexture3DLinear(t, [0, 0, 0])[0]).toBe(0);
    expect(vdbFrameTexture3DLinear(t, [9, 3, 2])[0]).toBe(30);
  });
});

describe("vdbFrameMultiScaleSample / vdbFrameMultiScaleLight (twins of the WGSL lookup)", () => {
  const grid = { boxMin: [10, 0, -5] as V3, boxSize: [16, 16, 16] as V3 };

  it("sample: 0 outside the box; level 0 clamps to the base region; level k reads the mip at base texel / 2^k", () => {
    // Base: two cascades stacked in z; the region is the second. Channel 0 = 1 in cascade 0, x in cascade 1.
    const base: VDBFrameTexture3D = { data: new Float64Array(16 * 16 * 32 * 2), dimensions: [16, 16, 32], channels: 2 };
    const d = base.data as Float64Array;
    for (let z = 0; z < 32; z++) for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) d[(x + (y + z * 16) * 16) * 2] = z < 16 ? 1000 : x;
    // Level 2 mip (4³ texels): channel 0 = 100 x.
    const mip2: VDBFrameTexture3D = { data: new Float64Array(64), dimensions: [4, 4, 4], channels: 1 };
    for (let i = 0; i < 64; i++) (mip2.data as Float64Array)[i] = 100 * (i % 4);
    const mip1: VDBFrameTexture3D = { data: new Float64Array(8 ** 3), dimensions: [8, 8, 8], channels: 1 };
    const textures = { base, baseOrigin: [0, 0, 16] as V3, baseSize: [16, 16, 16] as V3, mips: [mip1, mip2] };
    expect(vdbFrameMultiScaleSample(textures, grid, 0, [9.99, 8, 3])).toEqual([0, 0]);
    expect(vdbFrameMultiScaleSample(textures, grid, 0, [18, 8, 11.01])).toEqual([0, 0]);
    // World x 18.25 → base texel 8.25 → between texels 7 and 8: 7.75. z at the region's bottom edge (texel 0.0 → clamped
    // to 0.5): no bleed from cascade 0 (1000).
    const s0 = vdbFrameMultiScaleSample(textures, grid, 0, [18.25, 8, -5]);
    expect(s0[0]).toBeCloseTo(7.75, 12);
    // Level 2: base texel 8.25 → 2.0625 → mip texel coordinate 1.5625 → 100 (1 + 0.5625) = 156.25; one channel: no absorption.
    const s2 = vdbFrameMultiScaleSample(textures, grid, 2, [18.25, 8, 3]);
    expect(s2[0]).toBeCloseTo(156.25, 10);
    expect(s2[1]).toBe(0);
  });

  it("light: a uniform medium gives the closed form of 1D diffusion over h (2^L − 1) (or maxDistance)", () => {
    const sigma = 0.3;
    const alpha = 0.05;
    const g = 0.2; // 1 − asymmetry
    const h = 0.5;
    const uniform = () => [sigma, alpha] as [number, number];
    const expected = (length: number) => {
      const absorbed = alpha;
      const reduced = (sigma - absorbed) * g;
      const D = sigma * length;
      const effective = Math.sqrt(absorbed * 3 * (absorbed + reduced)) * length;
      const total = Math.min(D, effective + Math.log(reduced * length * 0.75 + 1));
      return Math.exp(-total) - Math.exp(-D);
    };
    for (const levels of [1, 3, 6]) {
      const v = vdbFrameMultiScaleLight(uniform, { unit: h, levels, multipleScattering: g }, [0, 0, 0], [0, 1, 0], 1e20);
      expect(v).toBeCloseTo(expected(h * (2 ** levels - 1)), 14);
      expect(v).toBeGreaterThan(0);
    }
    // maxDistance 5h: segments h, 2h, then 2h of the third level, none after.
    expect(vdbFrameMultiScaleLight(uniform, { unit: h, levels: 6, multipleScattering: g }, [0, 0, 0], [0, 0, 1], 5 * h)).toBeCloseTo(expected(5 * h), 14);
    // Single scattering (g = 0 → 1 − g = 0 is not used by his modes, but multipleScattering 0 means off): no scattered light.
    expect(vdbFrameMultiScaleLight(uniform, { unit: h, levels: 6, multipleScattering: 0 }, [0, 0, 0], [0, 1, 0], 1e20)).toBe(0);
    // Empty space: nothing scattered.
    expect(vdbFrameMultiScaleLight(() => [0, 0], { unit: h, levels: 6, multipleScattering: g }, [0, 0, 0], [0, 1, 0], 1e20)).toBe(0);
  });

  it("light: 5 taps per level, centred on the segment, ±h 2^k on two unit axes at right angles to the light (pole at −y too)", () => {
    const h = 1;
    for (const dir of [
      [0, 1, 0],
      [0.6, 0, 0.8],
      [0.48, -0.6, 0.64],
      [0, -1, 0],
    ] as V3[]) {
      const taps: { level: number; p: V3 }[] = [];
      vdbFrameMultiScaleLight(
        (level, p) => {
          taps.push({ level, p });
          return [1, 0];
        },
        { unit: h, levels: 4, multipleScattering: 0.5 },
        [1, 2, 3],
        dir,
        1e20,
      );
      expect(taps.length).toBe(20);
      for (let k = 0; k < 4; k++) {
        const t = taps.filter((x) => x.level === k).map((x) => x.p);
        expect(t.length).toBe(5);
        const scale = h * 2 ** k;
        const along = scale - h + scale / 2;
        const c = [1 + dir[0] * along, 2 + dir[1] * along, 3 + dir[2] * along];
        t[0]!.forEach((v, a) => expect(v).toBeCloseTo(c[a]!, 12));
        for (const p of t.slice(1)) {
          const off = p.map((v, a) => v - c[a]!);
          expect(Math.hypot(...off)).toBeCloseTo(scale, 5);
          expect(off[0]! * dir[0] + off[1]! * dir[1] + off[2]! * dir[2]).toBeCloseTo(0, 5);
        }
        // ±u and ±v pairs: opposite, and the two axes at right angles.
        const o = t.slice(1).map((p) => p.map((v, a) => v - c[a]!));
        o[0]!.forEach((v, a) => expect(v).toBeCloseTo(-o[1]![a]!, 12));
        expect(o[0]![0]! * o[2]![0]! + o[0]![1]! * o[2]![1]! + o[0]![2]! * o[2]![2]!).toBeCloseTo(0, 5);
      }
    }
  });

  it("end to end on a seeded cloud: the scattered light stays in [0, 1]", () => {
    const size: V3 = [32, 32, 32];
    const g = blob(size, 21, 4);
    const chain = vdbFrameMultiScaleMips(g, { levels: 6 });
    const textures = {
      base: g,
      baseSize: size,
      mips: chain.levels.map((data, i) => ({ data, dimensions: chain.layout.mips[i]!.dimensions, channels: 2 })),
    };
    const box = { boxMin: [0, 0, 0] as V3, boxSize: [32, 32, 32] as V3 };
    const r = rng(9);
    for (let i = 0; i < 64; i++) {
      const world: V3 = [r() * 32, r() * 32, r() * 32];
      const z = r() * 2 - 1;
      const phi = r() * 2 * Math.PI;
      const dir: V3 = [Math.sqrt(1 - z * z) * Math.cos(phi), z, Math.sqrt(1 - z * z) * Math.sin(phi)];
      const v = vdbFrameMultiScaleLight((k, p) => vdbFrameMultiScaleSample(textures, box, k, p), { unit: 1, levels: 6, multipleScattering: 0.2 }, world, dir, 1e20);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});
