import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  VDB_GPU_LAYOUT_VERSION,
  VDB_HEADER_WORD,
  VDB_POOL_RECORD_BYTES,
  VDB_POOL_RECORD_WORDS,
  VDB_POOL_TABLE_OFFSET,
  VDB_RECORD_FLAG,
  VDB_RECORD_WORD,
  VDBGridPool,
  decodeHeader,
  decodeRecord,
  encodeRecord,
  vdbPoolLayoutWgsl,
  type VDBGridRecord,
} from "../../src/core/index.js";
import {
  VDB_RECORD_BRICK_LOG2_SHIFT,
  VDB_RECORD_BYTE_FIELD_MASK,
  VDB_RECORD_GRID_CLASS_SHIFT,
  encodeHeader,
} from "../../src/core/pool/layout.js";
import { readImageHeader } from "../../src/core/pool/image-header.js";
import { NanoVDBFile } from "../../src/tools/index.js";
import { FakeDevice } from "./fake-gpu.js";

const f32 = (x: number) => Math.fround(x);

const SAMPLE: VDBGridRecord = {
  imageOffset: 4096,
  imageBytes: 123_456,
  gridType: 14,
  live: true,
  hasStats: true,
  hasBrickMajorant: true,
  instanceTransform: true,
  gridClass: 2,
  brickLog2: 3,
  worldToIndex: [0.5, 0, 0, -1, 0, 2, 0, 3.25, 0, 0, 4, f32(0.1)],
  indexToWorld: [2, 0, 0, 2, 0, 0.5, 0, f32(-1.625), 0, 0, 0.25, f32(-0.025)],
  indexBBoxMin: [-40, -20, -30],
  indexBBoxMax: [40, 20, 30],
  background: 3,
  valueMax: f32(1.7),
  majorantOffset: 999_968,
  majorantDims: [11, 6, 9],
};

describe("pool layout (K1.3a)", () => {
  it("sizes and offsets match the design note", () => {
    expect(VDB_GPU_LAYOUT_VERSION).toBe(1);
    expect(VDB_POOL_TABLE_OFFSET).toBe(32);
    expect(VDB_POOL_RECORD_BYTES).toBe(160);
    expect(VDB_POOL_RECORD_WORDS).toBe(40);
    expect(VDB_POOL_RECORD_BYTES % 16).toBe(0);
    // Fields tile the 40 words without overlap: [first word, width].
    const widths: Record<keyof typeof VDB_RECORD_WORD, number> = {
      imageOffset: 1, imageBytes: 1, gridType: 1, flags: 1, worldToIndex: 12, indexToWorld: 12,
      indexBBoxMin: 3, background: 1, indexBBoxMax: 3, valueMax: 1, majorantOffset: 1, majorantDims: 3,
    };
    const covered = new Array<number>(VDB_POOL_RECORD_WORDS).fill(0);
    for (const [name, first] of Object.entries(VDB_RECORD_WORD)) {
      for (let k = 0; k < widths[name as keyof typeof widths]; k++) covered[first + k]!++;
    }
    expect(covered).toEqual(new Array(VDB_POOL_RECORD_WORDS).fill(1));
  });

  it("record round-trips through encode/decode", () => {
    const words = encodeRecord(SAMPLE);
    expect(words.length).toBe(VDB_POOL_RECORD_WORDS);
    expect(decodeRecord(words)).toEqual(SAMPLE);
    // At a word offset inside a larger table.
    const table = new Uint32Array(8 + 3 * 40);
    encodeRecord(SAMPLE, table, 8 + 2 * 40);
    expect(decodeRecord(table, 8 + 2 * 40)).toEqual(SAMPLE);
    expect(table.subarray(0, 8 + 2 * 40).every((w) => w === 0)).toBe(true);
  });

  it("packs flags, class and brick log2 into word 3", () => {
    const flags = encodeRecord(SAMPLE)[VDB_RECORD_WORD.flags]!;
    const F = VDB_RECORD_FLAG;
    expect(flags).toBe(F.live | F.hasStats | F.hasBrickMajorant | F.instanceTransform | (2 << 8) | (3 << 16));
    expect(() => encodeRecord({ ...SAMPLE, gridClass: 256 })).toThrow(/VDB_BAD_OPTION/);
    expect(() => encodeRecord({ ...SAMPLE, worldToIndex: [1, 2, 3] })).toThrow(/12 entries/);
  });

  it("header round-trips", () => {
    const words = encodeHeader({ gridCapacity: 16, gridCount: 3, usedBytes: 12_345_664 });
    expect(Array.from(words)).toEqual([0x50424456, 1, 16, 3, 32, 160, 12_345_664, 0]);
    expect(decodeHeader(words)).toEqual({
      magic: 0x50424456, layoutVersion: 1, gridCapacity: 16, gridCount: 3, tableOffset: 32, recordBytes: 160,
      usedBytes: 12_345_664,
    });
    // "VDBP" in memory order.
    expect(new TextDecoder().decode(new Uint8Array(words.buffer, 0, 4))).toBe("VDBP");
  });

  it("WGSL constants are generated from the same tables", () => {
    const wgsl = vdbPoolLayoutWgsl();
    expect(vdbPoolLayoutWgsl()).toBe(wgsl); // deterministic
    expect(wgsl).not.toContain("`");
    const consts = new Map<string, number>();
    for (const line of wgsl.split("\n")) {
      if (line.startsWith("//") || line === "") continue;
      const m = /^const (VDB_[A-Z0-9_]+): u32 = (0x[0-9a-f]+|\d+)u;$/.exec(line);
      expect(m, `unexpected WGSL line: ${line}`).not.toBeNull();
      expect(consts.has(m![1]!), `duplicate ${m![1]}`).toBe(false);
      consts.set(m![1]!, Number(m![2]));
    }
    const snake = (s: string) => s.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toUpperCase();
    const expected = new Map<string, number>([
      ["VDB_GPU_LAYOUT_VERSION", VDB_GPU_LAYOUT_VERSION],
      ["VDB_POOL_MAGIC", 0x50424456],
      ["VDB_POOL_ALIGN", 32],
      ["VDB_POOL_TABLE_OFFSET", VDB_POOL_TABLE_OFFSET],
      ["VDB_POOL_TABLE_WORD", VDB_POOL_TABLE_OFFSET / 4],
      ["VDB_POOL_RECORD_BYTES", VDB_POOL_RECORD_BYTES],
      ["VDB_POOL_RECORD_WORDS", VDB_POOL_RECORD_WORDS],
      ...Object.entries(VDB_HEADER_WORD).map(([k, v]) => [`VDB_HEADER_${snake(k)}`, v] as [string, number]),
      ...Object.entries(VDB_RECORD_WORD).map(([k, v]) => [`VDB_REC_${snake(k)}`, v] as [string, number]),
      ...Object.entries(VDB_RECORD_FLAG).map(([k, v]) => [`VDB_REC_FLAG_${snake(k)}`, v] as [string, number]),
      ["VDB_REC_GRID_CLASS_SHIFT", VDB_RECORD_GRID_CLASS_SHIFT],
      ["VDB_REC_BRICK_LOG2_SHIFT", VDB_RECORD_BRICK_LOG2_SHIFT],
      ["VDB_REC_BYTE_FIELD_MASK", VDB_RECORD_BYTE_FIELD_MASK],
    ]);
    expect(consts).toEqual(expected);
    // Spot checks the generator agent relies on.
    expect(wgsl).toContain("const VDB_HEADER_GRID_COUNT: u32 = 3u;");
    expect(wgsl).toContain("const VDB_REC_IMAGE_OFFSET: u32 = 0u;");
    expect(wgsl).toContain("const VDB_REC_WORLD_TO_INDEX: u32 = 4u;");
    expect(wgsl).toContain("const VDB_REC_INDEX_BBOX_MAX: u32 = 32u;");
  });
});

const fixtureUrl = new URL("../../../../fixtures/primitives/box_fog_float.nvdb", import.meta.url);

describe.skipIf(!existsSync(fixtureUrl))("golden record (box_fog_float)", () => {
  it("matches the committed words", async () => {
    const bytes = readFileSync(fixtureUrl);
    const file = NanoVDBFile.fromArrayBuffer(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    const sidecar = JSON.parse(readFileSync(new URL("box_fog_float.sidecar.json", fixtureUrl), "utf8"));
    const device = new FakeDevice();
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 4096 });
    const id = pool.add(file.gridImage(0));
    pool.flush();
    const words = device.buffers.at(-1)!.words();
    const at = VDB_POOL_TABLE_OFFSET / 4 + id * VDB_POOL_RECORD_WORDS;
    const record = Array.from(words.subarray(at, at + VDB_POOL_RECORD_WORDS));

    // prettier-ignore
    expect(record).toEqual([
      0x00000a20, // imageOffset 2592 = align32(32 + 16 * 160)
      0x00302620, // imageBytes 3155488 (sidecar gridByteSize)
      0x00000001, // Float
      0x00000203, // live | hasStats | FogVolume << 8
      0x3f800000, 0, 0, 0, 0, 0x3f800000, 0, 0, 0, 0, 0x3f800000, 0, // worldToIndex: identity (voxel size 1)
      0x3f800000, 0, 0, 0, 0, 0x3f800000, 0, 0, 0, 0, 0x3f800000, 0, // indexToWorld
      0xffffffd8, 0xffffffec, 0xffffffe2, // bbox min -40 -20 -30
      0x40400000, // background 3 (native fog root background; CLAUDE.md Traps)
      0x00000028, 0x00000014, 0x0000001e, // bbox max 40 20 30
      0x40400000, // valueMax = max(root max 1, background 3)
      0, 0, 0, 0, // no brick majorant
    ]);
    const decoded = decodeRecord(new Uint32Array(record));
    expect(decoded.imageBytes).toBe(sidecar.grid.gridByteSize);
    expect([decoded.indexBBoxMin, decoded.indexBBoxMax]).toEqual(sidecar.grid.indexBBox);
    expect(readImageHeader(file.gridImage(0)).rootMax).toBe(1);
  });
});
