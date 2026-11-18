/**
 * Hostile-input hardening for point data grids (`vdb/points.ts`) and the
 * vector value paths of `readCompressedValues` (K4.2 review, to the K4.1
 * `blosc-hostile` bar).
 *
 * Every crafted file must fail with a clean `VdbFormatError` (or
 * `VdbUnsupportedError` for a real construct out of scope), in well under
 * 200 ms, without a huge allocation. A `RangeError` would mean an allocation
 * was attempted with a hostile length; the typed-array spy also records the
 * largest `Uint8Array` / `Float32Array` length constructed during the call.
 *
 * Files come from the synthetic point writer (`synthetic-points.ts`) with its
 * `tamper` hooks, which corrupt one field at the point it is written.
 */

import { describe, expect, it } from "vitest";

import { parseVdb, type VdbPointDataGrid } from "../../src/tools/index.js";
import { ByteReader } from "../../src/tools/vdb/byte-reader.js";
import { readCompressedValues, VEC3D_FORMAT, VEC3S_FORMAT } from "../../src/tools/vdb/compression.js";
import { VdbFormatError, VdbUnsupportedError } from "../../src/tools/vdb/errors.js";
import { MAX_POINT_EXPANSION, unitVecWeightTable } from "../../src/tools/vdb/points.js";
import {
  writeDescriptor,
  writeSyntheticPoints,
  type SynthPointAttr,
  type SynthPointLeaf,
  type SynthPointSpec,
} from "./synthetic-points.js";
import { bloscFrame, type Writer } from "./synthetic-vdb.js";

const TIME_BUDGET_MS = 200;
/** No hostile case may construct a typed array longer than this. */
const ALLOC_CAP = 1 << 20;

type ErrorClass = typeof VdbFormatError | typeof VdbUnsupportedError;

/** Runs `fn`, recording the largest numeric-length Uint8Array / Float32Array constructed inside it. */
function withAllocSpy(fn: () => unknown): { error: unknown; maxAlloc: number; ms: number } {
  const RealU8 = globalThis.Uint8Array;
  const RealF32 = globalThis.Float32Array;
  let maxAlloc = 0;
  const spy = <T extends abstract new (...args: never[]) => unknown>(Real: T): T =>
    class extends (Real as unknown as new (...args: unknown[]) => object) {
      constructor(...args: unknown[]) {
        if (typeof args[0] === "number" && args[0] > maxAlloc) maxAlloc = args[0];
        super(...args);
      }
    } as unknown as T;
  globalThis.Uint8Array = spy(RealU8);
  globalThis.Float32Array = spy(RealF32);
  const t = performance.now();
  let error: unknown;
  try {
    fn();
  } catch (e) {
    error = e;
  } finally {
    globalThis.Uint8Array = RealU8;
    globalThis.Float32Array = RealF32;
  }
  return { error, maxAlloc, ms: performance.now() - t };
}

function expectRejected(fn: () => unknown, message: RegExp, cls: ErrorClass = VdbFormatError): void {
  const { error, maxAlloc, ms } = withAllocSpy(fn);
  expect(error).toBeInstanceOf(cls);
  expect((error as Error).message).toMatch(message);
  expect(ms).toBeLessThan(TIME_BUDGET_MS);
  expect(maxAlloc).toBeLessThanOrEqual(ALLOC_CAP);
}

// ---------------------------------------------------------------------------
// A small valid point grid to corrupt: two leaves, P (fxpt16), id (int32),
// a group array with one group.
// ---------------------------------------------------------------------------

const ATTRS: SynthPointAttr[] = [
  { name: "P", type: "vec3s", codec: "fxpt16" },
  { name: "id", type: "int32", codec: "null" },
  { name: "__groups0", type: "uint8", codec: "grp" },
];
const GROUPS = [{ name: "red", offset: 1 }];

function offsetsFor(counts: [number, number][]): Uint32Array {
  const per = new Uint32Array(512);
  for (const [voxel, c] of counts) per[voxel] = c;
  const offsets = new Uint32Array(512);
  let total = 0;
  for (let i = 0; i < 512; i++) offsets[i] = total += per[i]!;
  return offsets;
}

function leaf(origin: [number, number, number], counts: [number, number][], first: number): SynthPointLeaf {
  const offsets = offsetsFor(counts);
  const n = offsets[511]!;
  const p = new Uint8Array(n * 6);
  const pv = new DataView(p.buffer);
  for (let i = 0; i < n * 3; i++) pv.setUint16(i * 2, (first * 3 + i) * 977, true);
  const id = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) new DataView(id.buffer).setInt32(i * 4, first + i, true);
  const groups = Uint8Array.from({ length: n }, (_, i) => ((first + i) % 2) << 1);
  return { origin, offsets, arrays: [p, id, groups] };
}

function base(over: Partial<SynthPointSpec> = {}): SynthPointSpec {
  return {
    voxelSize: 0.25,
    attributes: ATTRS,
    groups: GROUPS,
    leaves: [leaf([0, 0, 0], [[0, 2], [77, 3]], 0), leaf([8, 0, 16], [[511, 4]], 5)],
    mode: "paged",
    pageBlosc: true,
    voxelBlosc: true,
    ...over,
  };
}

const parse = (s: SynthPointSpec) => parseVdb(writeSyntheticPoints(s));
const pointsOf = (s: SynthPointSpec) => (parse(s).grids[0] as VdbPointDataGrid).points();

describe("the base files are valid", () => {
  it.each([
    ["paged, Blosc", {}],
    ["plain", { mode: "plain" as const }],
  ])("%s: parses and decodes", (_, over) => {
    const pts = pointsOf(base(over));
    expect(pts.count).toBe(9);
    expect(Array.from(pts.attributes["id"] as Int32Array)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    expect(Array.from(pts.groups["red"]!)).toEqual([0, 1, 0, 1, 0, 1, 0, 1, 0]);
  });
});

// ---------------------------------------------------------------------------
// Pass structure
// ---------------------------------------------------------------------------

describe("pass count", () => {
  it("numPasses below 4", () => {
    expectRejected(() => parse(base({ tamper: { numPasses: 3 } })), /buffer passes/);
  });

  it("numPasses too small for the descriptor's attributes", () => {
    expectRejected(() => parse(base({ tamper: { numPasses: 6 } })), /attribute arrays, but the grid writes 6 passes/);
  });

  it("numPasses 65535 over 4096 leaves costs nothing (passes past the arrays do no I/O)", () => {
    // Before the hardening every pass visited every leaf: 65535 x 4096 iterations.
    const leaves: SynthPointLeaf[] = [];
    for (let i = 0; i < 4096; i++) {
      const offsets = new Uint32Array(512).fill(1);
      leaves.push({ origin: [(i >> 8) * 8, ((i >> 4) & 15) * 8, (i & 15) * 8], offsets, arrays: [new Uint8Array(4)] });
    }
    const buffer = writeSyntheticPoints({
      voxelSize: 1,
      attributes: [{ name: "id", type: "int32", codec: "null" }],
      leaves,
      mode: "paged",
      pageBlosc: true,
      voxelBlosc: true,
      tamper: { numPasses: 65535 },
    });
    const t = performance.now();
    const grid = parseVdb(buffer).grids[0] as VdbPointDataGrid;
    expect(performance.now() - t).toBeLessThan(TIME_BUDGET_MS * 2);
    expect(grid.pointCount).toBe(4096);
  });
});

// ---------------------------------------------------------------------------
// Descriptor
// ---------------------------------------------------------------------------

/** A descriptor block written by hand: header byte, then `body`. */
const descriptorOf =
  (body: (w: Writer) => void, header = 1) =>
  (w: Writer): void => {
    w.u8(header);
    body(w);
  };

describe("descriptor", () => {
  it("attribute count of 2^40", () => {
    const descriptor = descriptorOf((w) => w.u64(1n << 40n));
    expectRejected(() => parse(base({ tamper: { descriptor } })), /claims 1099511627776 attributes/);
  });

  it("attribute index past the attribute count", () => {
    const descriptor = descriptorOf((w) => {
      w.u64(1);
      w.string("vec3s");
      w.string("fxpt16");
      w.string("P");
      w.u64(0xffffffffffffn);
    });
    expectRejected(() => parse(base({ tamper: { descriptor } })), /has index 281474976710655 of 1/);
  });

  it("the same attribute name twice", () => {
    const descriptor = descriptorOf((w) => {
      w.u64(2);
      for (let i = 0; i < 2; i++) {
        w.string("float");
        w.string("null");
      }
      w.string("a");
      w.u64(0);
      w.string("a");
      w.u64(1);
    });
    expectRejected(() => parse(base({ tamper: { descriptor } })), /"a" appears twice/);
  });

  it("group count of 2^62", () => {
    const descriptor = descriptorOf((w) => {
      w.u64(0);
      w.u64(1n << 62n);
    });
    expectRejected(() => parse(base({ tamper: { descriptor } })), /claims 4611686018427387904 groups/);
  });

  it("an attribute name whose length runs past the file", () => {
    const descriptor = descriptorOf((w) => {
      w.u64(1);
      w.u32(0xffffffff);
    });
    expectRejected(() => parse(base({ tamper: { descriptor } })), /unexpected end of buffer/);
  });

  it("header skip field past the end of the file", () => {
    const descriptor = descriptorOf((w) => {
      writeDescriptor(w, ATTRS, GROUPS);
      w.u64(0xffffffffffffffffn);
    }, 3);
    expectRejected(() => parse(base({ tamper: { descriptor } })), /skip 18446744073709551615 bytes past the end/);
  });

  it("unknown header flags", () => {
    const descriptor = descriptorOf((w) => writeDescriptor(w, ATTRS, GROUPS), 5);
    expectRejected(() => parse(base({ tamper: { descriptor } })), /header flags 0x5/, VdbUnsupportedError);
  });

  it("a 4 MiB string in the descriptor metadata is read quickly", () => {
    // Accepted: OpenVDB has no limit. (The chunked string decode is ~12x faster than the
    // per-byte loop it replaced; that only matters at tens of MiB, ~3.3 s at 64 MiB.)
    const big = "x".repeat(4 << 20);
    const descriptor = descriptorOf((w) => {
      w.u64(ATTRS.length);
      for (const a of ATTRS) {
        w.string(a.type);
        w.string(a.codec);
      }
      [...ATTRS.entries()]
        .sort(([, a], [, b]) => (a.name < b.name ? -1 : 1))
        .forEach(([i, a]) => {
          w.string(a.name);
          w.u64(i);
        });
      w.u64(1);
      w.string("red");
      w.u64(1);
      w.u32(1); // one metadata entry
      w.string("string:0");
      w.string("string");
      w.u32(big.length);
      for (let i = 0; i < big.length; i++) w.u8(120);
    });
    const buffer = writeSyntheticPoints(base({ tamper: { descriptor } }));
    const t = performance.now();
    const grid = parseVdb(buffer).grids[0] as VdbPointDataGrid;
    expect(performance.now() - t).toBeLessThan(TIME_BUDGET_MS);
    expect(grid.points().count).toBe(9);
  });
});

// ---------------------------------------------------------------------------
// Attribute array metadata
// ---------------------------------------------------------------------------

describe("attribute array metadata", () => {
  const meta = (fn: NonNullable<NonNullable<SynthPointSpec["tamper"]>["arrayMeta"]>) => base({ tamper: { arrayMeta: fn } });

  it("byte count below the 6 header bytes", () => {
    expectRejected(() => parse(meta((l, a, m) => (a === 1 ? { ...m, total: 5n } : m))), /byte count 5 is out of range/);
  });

  it("byte count of 2^63", () => {
    expectRejected(
      () => parse(meta((l, a, m) => (a === 1 ? { ...m, total: 1n << 63n } : m))),
      /byte count 9223372036854775808 is out of range/,
    );
  });

  it("byte count larger than its page (paged)", () => {
    expectRejected(() => parse(meta((l, a, m) => (a === 1 ? { ...m, total: 1n << 40n } : m))), /overruns its/);
  });

  it("byte count past the end of the file (not paged)", () => {
    const s = meta((l, a, m) => (a === 1 ? { ...m, total: 1n << 40n } : m));
    expectRejected(() => parse({ ...s, mode: "plain" }), /unexpected end of buffer/);
  });

  it("unknown serialization flags", () => {
    expectRejected(
      () => parse(meta((l, a, m) => (a === 1 ? { ...m, serialization: 0x18 } : m))),
      /serialization flags 0x18/,
      VdbUnsupportedError,
    );
  });

  it("constant stride of 0", () => {
    expectRejected(
      () => parse(meta((l, a, m) => (a === 1 ? { ...m, serialization: m.serialization | 1, stride: 0 } : m))),
      /constant stride of 0/,
    );
  });

  it("array size that disagrees with the leaf's point count", () => {
    expectRejected(() => parse(meta((l, a, m) => (a === 1 ? { ...m, size: 0xffffffff } : m))), /an attribute array of size 4294967295/);
  });

  it("uniform array with a stride of 2^32 - 1: points() stops at the output budget", () => {
    const s = base({
      mode: "plain",
      attributes: [...ATTRS, { name: "w", type: "float", codec: "null", uniform: true, stride: 2 }],
      leaves: base().leaves.map((l) => ({ ...l, arrays: [...l.arrays, new Uint8Array(4)] })),
      tamper: { arrayMeta: (l, a, m) => (a === 3 ? { ...m, stride: 0xffffffff } : m) },
    });
    const grid = parse(s).grids[0] as VdbPointDataGrid; // the stride costs nothing to parse
    expectRejected(() => grid.points(), /point attribute "w" needs \d+ bytes, past the point grid's \d+-byte output budget/);
  });
});

// ---------------------------------------------------------------------------
// Page headers
// ---------------------------------------------------------------------------

describe("page headers", () => {
  const pages = (fn: NonNullable<NonNullable<SynthPointSpec["tamper"]>["pageHeaders"]>) => base({ tamper: { pageHeaders: fn } });

  it("compressed size of 2^31 - 1 (larger than the file)", () => {
    expectRejected(
      () => parse(pages((a, h) => (a === 0 ? [[0x7fffffff, 0x7fffffff]] : h))),
      /claims 2147483647 stored bytes in a \d+-byte file/,
    );
  });

  it("raw page of 2^31 bytes (i32 minimum)", () => {
    expectRejected(() => parse(pages((a, h) => (a === 0 ? [[-0x80000000]] : h))), /claims 2147483648 stored bytes/);
  });

  it("2 GiB uncompressed from a small Blosc frame", () => {
    expectRejected(
      () => parse(pages((a, h) => (a === 0 ? [[h[0]![0]!, 0x7fffffff]] : h))),
      /claims 2147483647 bytes from a \d+-byte Blosc frame/,
    );
  });

  it("zero-sized page", () => {
    expectRejected(() => parse(pages((a, h) => (a === 0 ? [[0]] : h))), /page header 0\/0 is empty/);
  });

  it("page larger than the arrays that fill it", () => {
    expectRejected(
      () => parse(pages((a, h) => (a === 1 ? [[h[0]![0]!, h[0]![1]! + 8]] : h))),
      /point attribute 1 page headers end mid-page/,
    );
  });

  it("page smaller than the arrays assigned to it", () => {
    expectRejected(() => parse(pages((a, h) => (a === 1 ? [[h[0]![0]!, 8]] : h))), /overruns its 8-byte page/);
  });
});

// ---------------------------------------------------------------------------
// Voxel buffers and point counts
// ---------------------------------------------------------------------------

describe("voxel buffers and point counts", () => {
  it("end offsets that decrease", () => {
    const s = base({ tamper: { storedOffsets: (l, o) => (l === 0 ? (o.fill(7, 0, 5), o) : o) } });
    expectRejected(() => parse(s), /end offsets decrease/);
  });

  it("4 billion points in uniform arrays: rejected at parse, nothing allocated", () => {
    const huge = 0xffffffff;
    const s: SynthPointSpec = {
      voxelSize: 1,
      attributes: [{ name: "P", type: "vec3s", codec: "fxpt16", uniform: true }],
      leaves: [{ origin: [0, 0, 0], offsets: offsetsFor([[3, 1]]), arrays: [new Uint8Array(6)] }],
      mode: "plain",
      tamper: {
        storedOffsets: (l, o) => (o.fill(huge, 3), o),
        arrayMeta: (l, a, m) => ({ ...m, size: huge }),
      },
    };
    expectRejected(() => parse(s), /claims 4294967295 points, more than a \d+-byte file can hold/);
  });

  it("the parse-time bound is MAX_POINT_EXPANSION bytes of positions per file byte", () => {
    // Just under the bound parses (uniform P, one voxel); points() decodes within budget.
    const bytes = writeSyntheticPoints({
      voxelSize: 1,
      attributes: [{ name: "P", type: "vec3s", codec: "fxpt16", uniform: true }],
      leaves: [{ origin: [0, 0, 0], offsets: offsetsFor([[0, 1]]), arrays: [new Uint8Array(6)] }],
      mode: "plain",
    }).byteLength;
    const most = Math.floor((MAX_POINT_EXPANSION * bytes) / 12);
    const spec = (n: number): SynthPointSpec => ({
      voxelSize: 1,
      attributes: [{ name: "P", type: "vec3s", codec: "fxpt16", uniform: true }],
      leaves: [{ origin: [0, 0, 0], offsets: offsetsFor([[0, 1]]), arrays: [new Uint8Array(6)] }],
      mode: "plain",
      tamper: { storedOffsets: (l, o) => o.fill(n), arrayMeta: (l, a, m) => ({ ...m, size: n }) },
    });
    const grid = parse(spec(most)).grids[0] as VdbPointDataGrid;
    expect(grid.pointCount).toBe(most);
    expect(grid.points().positions.length).toBe(most * 3);
    expectRejected(() => parse(spec(most + 1)), /more than a \d+-byte file can hold/);
  });
});

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

describe("groups", () => {
  const withGroup = (offset: number | bigint) =>
    base({
      tamper: {
        descriptor: descriptorOf((w) => {
          w.u64(ATTRS.length);
          for (const a of ATTRS) {
            w.string(a.type);
            w.string(a.codec);
          }
          [...ATTRS.entries()]
            .sort(([, a], [, b]) => (a.name < b.name ? -1 : 1))
            .forEach(([i, a]) => {
              w.string(a.name);
              w.u64(i);
            });
          w.u64(1);
          w.string("red");
          w.u64(offset);
          w.u32(0);
        }),
      },
    });

  it("group offset 2^32 + 1 is out of range (it once wrapped to bit 1 through int32 shifts)", () => {
    const grid = parse(withGroup((1n << 32n) + 1n)).grids[0] as VdbPointDataGrid;
    expectRejected(() => grid.points(), /point group "red" offset \d+ is out of range/);
  });

  it("group offset past the group arrays", () => {
    const grid = parse(withGroup(8)).grids[0] as VdbPointDataGrid;
    expectRejected(() => grid.points(), /offset 8 is out of range/);
  });
});

// ---------------------------------------------------------------------------
// Vector value arrays (vec3s / vec3d through readCompressedValues)
// ---------------------------------------------------------------------------

describe("vector value arrays", () => {
  const maskAll = new Uint32Array(16).fill(0xffffffff);
  const opts = (format = VEC3S_FORMAT, blosc = true, useHalf = false) => ({
    fileVersion: 223,
    compression: { zip: !blosc, activeMask: false, blosc },
    useHalf,
    background: [0, 0, 0],
    numValues: 512,
    valueMask: maskAll,
    format,
  });
  /** meta byte, i64 length field, then the payload. */
  const stream = (field: bigint, payload: Uint8Array): ByteReader => {
    const bytes = new Uint8Array(1 + 8 + payload.length);
    new DataView(bytes.buffer).setBigInt64(1, field, true);
    bytes.set(payload, 9);
    return new ByteReader(bytes.buffer);
  };

  it("Blosc frame holding more than 512 vec3s", () => {
    const f = bloscFrame(new Uint8Array(513 * 12), 4);
    expectRejected(() => readCompressedValues(stream(BigInt(f.length), f), opts()), /nbytes=6156 exceeds the expected 6144/);
  });

  it("Blosc frame holding fewer than 512 vec3d", () => {
    const f = bloscFrame(new Uint8Array(511 * 24), 8);
    expectRejected(
      () => readCompressedValues(stream(BigInt(f.length), f), opts(VEC3D_FORMAT)),
      /decoded to 12264 bytes, expected 12288/,
    );
  });

  it("raw sentinel that is not a whole number of half vec3s", () => {
    expectRejected(
      () => readCompressedValues(stream(-7n, new Uint8Array(7)), opts(VEC3S_FORMAT, true, true)),
      /not a whole number of 6-byte values/,
    );
  });

  it("raw sentinel claiming more vec3s than the node holds", () => {
    expectRejected(
      () => readCompressedValues(stream(-513n * 12n, new Uint8Array(0)), opts()),
      /within the expected 512/,
    );
  });

  it("zip length field past the end of the buffer", () => {
    expectRejected(() => readCompressedValues(stream(1n << 40n, new Uint8Array(4)), opts(VEC3S_FORMAT, false)), /unexpected end/);
  });
});

// ---------------------------------------------------------------------------
// Independent oracle for the unit-vector codec
// ---------------------------------------------------------------------------

describe("QuantizedUnitVec weights", () => {
  it("equal OpenVDB's literal sNormalizationWeights entries (math/QuantizedUnitVec.cc @ a532de55)", () => {
    // Copied from the table in OpenVDB's source, not generated: an oracle independent of our formula.
    const literal: [number, number][] = [
      [0, 0.007936508395],
      [1, 0.007999744266],
      [63, 0.0112239169],
      [64, 0.01122250315],
      [127, 0.007873771712],
      [128, 0.007999744266],
      [1000, 0.009495858103],
      [4095, 0.009942499921],
      [4096, 0.01007074397],
      [5555, 0.01351598185],
      [8191, 0.01122250315],
    ];
    const table = unitVecWeightTable();
    for (const [slot, w] of literal) expect([slot, table[slot]]).toEqual([slot, Math.fround(w)]);
  });
});
