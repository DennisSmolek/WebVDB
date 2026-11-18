import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import {
  WvdbFormatError,
  decodeWvdbMask,
  decodeWvdbPage,
  decodeWvdbPoints,
  encodeWvdbMask,
  encodeWvdbPage,
  encodeWvdbPoints,
  sha256Hex,
  wvdbBrickCounts,
  wvdbBrickOrigin,
  wvdbBrickTotal,
  wvdbPointLods,
} from "../../src/tools/internal.js";

// Spec references are to mjurczyk/openvdb src/docs/wvdb/README.md (v0.5.x) at 75e7345.

const brick = (fn: (i: number) => number, channels = 1): Float32Array =>
  Float32Array.from({ length: 512 * channels }, (_, i) => fn(i));

describe("wvdb brick math (§3.3)", () => {
  it("counts, total and origins follow the spec formulas", () => {
    expect(wvdbBrickCounts([37, 128, 38])).toEqual([5, 16, 5]);
    expect(wvdbBrickTotal([37, 128, 38])).toBe(400);
    expect(wvdbBrickTotal([1, 1, 1])).toBe(1);
    // index = bx + by*countX + bz*countX*countY
    const dims = [20, 30, 40];
    const [cx, cy] = wvdbBrickCounts(dims); // 3, 4
    for (const [bx, by, bz] of [[0, 0, 0], [2, 0, 0], [1, 3, 0], [2, 3, 4], [0, 1, 2]]) {
      expect(wvdbBrickOrigin(dims, bx! + by! * cx + bz! * cx * cy)).toEqual([bx! * 8, by! * 8, bz! * 8]);
    }
  });
});

describe("wvdb page chunk (§6.2, §7)", () => {
  it("8-bit: layout, rounding (halves up), end codes exact, Appendix B size", () => {
    // values 0..511 scaled into [-1, 3]; min = -1, max = 3 exactly.
    const values = brick((i) => -1 + (4 * i) / 511);
    const page = encodeWvdbPage([{ index: 7, values }], { layout: "sparse", bits: 8, channels: 1 });
    expect(page.byteLength).toBe(4 + 4 + 8 + 512);
    const view = new DataView(page.buffer);
    expect(view.getUint32(0, true)).toBe(1);
    expect(view.getUint32(4, true)).toBe(7);
    expect(view.getFloat32(8, true)).toBe(-1);
    expect(view.getFloat32(12, true)).toBe(3);
    const scale = 255 / (3 - -1);
    for (let i = 0; i < 512; i++) {
      expect(page[16 + i]).toBe(Math.min(Math.max(Math.round((Math.fround(values[i]!) - -1) * scale), 0), 255));
    }
    const decoded = decodeWvdbPage(page, { layout: "sparse", bits: 8, channels: 1, count: 1, firstIndex: 0, brickTotal: 8 });
    expect(Array.from(decoded.indices)).toEqual([7]);
    expect(decoded.values[0]).toBe(-1);
    expect(decoded.values[511]).toBe(3);
    const step = 4 / 255;
    for (let i = 0; i < 512; i++) expect(Math.abs(decoded.values[i]! - values[i]!)).toBeLessThanOrEqual(step / 2 + 1e-6);
  });

  it("16-bit codes are little-endian uint16 and tighter", () => {
    const values = brick((i) => Math.sin(i));
    const page = encodeWvdbPage([{ index: 0, values }], { layout: "dense", bits: 16, channels: 1 });
    expect(page.byteLength).toBe(4 + 8 + 1024); // dense: no indices
    const d = decodeWvdbPage(page, { layout: "dense", bits: 16, channels: 1, count: 1, firstIndex: 0, brickTotal: 1 });
    const step = (d.maxs[0]! - d.mins[0]!) / 65535;
    for (let i = 0; i < 512; i++) expect(Math.abs(d.values[i]! - values[i]!)).toBeLessThanOrEqual(step / 2 + 1e-6);
  });

  it("constant brick channels store no codes; vector ranges interleave by brick then channel", () => {
    const a = brick((i) => (i < 512 ? 0.25 : i < 1024 ? i / 1024 : 2), 3); // c0 const, c1 varies, c2 const
    const b = brick((i) => (i < 512 ? i : 5), 3); // c0 varies, c1/c2 const
    const page = encodeWvdbPage(
      [
        { index: 3, values: a },
        { index: 1, values: b },
      ],
      { layout: "sparse", bits: 8, channels: 3 },
    );
    // varying = 2 brick channels
    expect(page.byteLength).toBe(4 + 8 + 8 * 2 * 3 + 2 * 512);
    const d = decodeWvdbPage(page, { layout: "sparse", bits: 8, channels: 3, count: 2, firstIndex: 0, brickTotal: 4 });
    expect(Array.from(d.mins)).toEqual([0.25, Math.fround(512 / 1024), 2, 0, 5, 5]);
    expect(d.values.subarray(0, 512).every((v) => v === 0.25)).toBe(true);
    expect(d.values.subarray(1024, 1536).every((v) => v === 2)).toBe(true);
    expect(d.values[1536]).toBe(0);
    expect(d.values[1536 + 511]).toBe(511);
  });

  it("dense pages use implicit indices from firstIndex", () => {
    const page = encodeWvdbPage(
      [
        { index: 0, values: brick(() => 1) },
        { index: 1, values: brick(() => 2) },
      ],
      { layout: "dense", bits: 8, channels: 1 },
    );
    const d = decodeWvdbPage(page, { layout: "dense", bits: 8, channels: 1, count: 2, firstIndex: 10, brickTotal: 12 });
    expect(Array.from(d.indices)).toEqual([10, 11]);
  });

  it("rejects a wrong size, a wrong brickCount and an out-of-range index (§14)", () => {
    const page = encodeWvdbPage([{ index: 5, values: brick((i) => i) }], { layout: "sparse", bits: 8, channels: 1 });
    const opts = { layout: "sparse" as const, bits: 8 as const, channels: 1, count: 1, firstIndex: 0, brickTotal: 8 };
    expect(() => decodeWvdbPage(page.subarray(0, page.length - 1), opts)).toThrow(WvdbFormatError);
    const padded = new Uint8Array(page.length + 1);
    padded.set(page);
    expect(() => decodeWvdbPage(padded, opts)).toThrow(/Appendix B/);
    expect(() => decodeWvdbPage(page, { ...opts, count: 2 })).toThrow(WvdbFormatError);
    expect(() => decodeWvdbPage(page, { ...opts, brickTotal: 5 })).toThrow(/brickTotal/);
  });
});

describe("wvdb mask chunk (§6.3)", () => {
  it("bit 0 of word 0 is brick 0; size ceil(total/32) words", () => {
    const bytes = encodeWvdbMask([0, 31, 32, 69], 70);
    expect(bytes.byteLength).toBe(12);
    const view = new DataView(bytes.buffer);
    expect(view.getUint32(0, true)).toBe(0x80000001);
    expect(view.getUint32(4, true)).toBe(1);
    expect(view.getUint32(8, true)).toBe(1 << 5);
    expect(Array.from(decodeWvdbMask(bytes, 70))).toEqual([0x80000001, 1, 32]);
  });

  it("rejects bits after brickTotal-1 and a wrong size", () => {
    const bytes = encodeWvdbMask([69], 70);
    expect(() => decodeWvdbMask(bytes, 69)).toThrow(/bits after/);
    expect(() => decodeWvdbMask(bytes, 100)).toThrow(WvdbFormatError);
  });
});

describe("wvdb points chunk (§6.4)", () => {
  it("16-bit byte planes in the chunk box; colours round(clamp(c)*255)", () => {
    const positions = new Float32Array([0, 10, -1, 1, 10, 1, 0.5, 10, 0]);
    const colors = new Float32Array([0, 0.5, 1, 2, -1, NaN, 0.2, 0.4, 0.6]);
    const bytes = encodeWvdbPoints(positions, colors);
    expect(bytes.byteLength).toBe(4 + 24 + 6 * 3 + 3 * 3);
    const view = new DataView(bytes.buffer);
    expect(view.getUint32(0, true)).toBe(3);
    expect([0, 1, 2].map((a) => view.getFloat32(4 + a * 4, true))).toEqual([0, 10, -1]);
    expect([0, 1, 2].map((a) => view.getFloat32(16 + a * 4, true))).toEqual([1, 10, 1]);
    // x codes: 0, 65535, round(0.5*65535)=32768 -> high plane then low plane
    expect(Array.from(bytes.subarray(28, 31))).toEqual([0, 255, 128]);
    expect(Array.from(bytes.subarray(31, 34))).toEqual([0, 255, 0]);
    // y: min === max -> all codes 0
    expect(Array.from(bytes.subarray(34, 40))).toEqual([0, 0, 0, 0, 0, 0]);
    const red = bytes.subarray(46, 49);
    expect(Array.from(red)).toEqual([0, 255, 51]);
    const blue = bytes.subarray(52, 55);
    expect(Array.from(blue)).toEqual([255, 255, 153]); // NaN -> 1.0 (his encoder)
    const d = decodeWvdbPoints(bytes, { count: 3, colors: true });
    expect(d.positions[3]).toBe(1); // end code gives max exactly
    expect(d.positions[4]).toBe(10);
    expect(d.colors![1]).toBeCloseTo(128 / 255, 7);
    expect(() => decodeWvdbPoints(bytes, { count: 3, colors: false })).toThrow(WvdbFormatError);
  });

  it("assigns point i to the first LOD k where i is a multiple of 8^(n-1-k) (§3.5)", () => {
    const lods = wvdbPointLods(130, 3);
    expect(lods[0]).toEqual([0, 64, 128]);
    expect(lods[1]).toEqual([8, 16, 24, 32, 40, 48, 56, 72, 80, 88, 96, 104, 112, 120]);
    expect(lods.flat().sort((a, b) => a - b)).toEqual(Array.from({ length: 130 }, (_, i) => i));
  });
});

describe("sha256Hex", () => {
  it("matches node:crypto across block boundaries", () => {
    for (const n of [0, 1, 55, 56, 63, 64, 65, 119, 120, 1000]) {
      const bytes = Uint8Array.from({ length: n }, (_, i) => (i * 31 + 7) & 255);
      expect(sha256Hex(bytes)).toBe(createHash("sha256").update(bytes).digest("hex"));
    }
  });
});
