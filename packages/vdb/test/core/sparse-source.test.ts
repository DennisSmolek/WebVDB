/**
 * K6.7: the core sparse source (`vdbSparseSourceGpu`) on the recording fake device: the pyramid block packer, the
 * volume table's record layout, slot allocation, the group-2 layout and bind entries, packing and growth of the
 * pyramid atlas, the retire rule, the stats readback, and the G16a zero-build swap (shader modules and pipelines
 * counted by wrapping the device's create calls). The GPU-vs-twin probe is e2e/k6-sparse.spec.ts.
 */
import { beforeEach, describe, expect, it } from "vitest";
import { VDBError, VDBGridPool } from "../../src/core/index.js";
import {
  VDB_SPARSE_RECORD_BYTES,
  VDB_SPARSE_RECORD_FIELDS,
  packSparseRecord,
  sparseRecordOffset,
  sparseRecordStructWgsl,
  sparseSourceWgsl,
  vdbSparseSourceGpu,
  type VDBSparseSourceGpu,
} from "../../src/core/internal.js";
import {
  vdbBuildFromDense,
  vdbPackPyramidBlocks,
  vdbSparseDensityStats,
  vdbBoxPyramid,
  vdbPyramidToHalf,
  vdbSparseDensityCpu,
  vdbSampleTrilinear,
} from "../../src/tools/internal.js";
import { computeFake } from "./fake-compute.js";
import { FakeDevice } from "./fake-gpu.js";
import { makeImage } from "./synthetic-image.js";
import { wgslNameProblems } from "./wgsl-names.js";

const code = (e: unknown) => (e instanceof VDBError ? e.code : String(e));
function expectCode(fn: () => unknown, c: string, message?: RegExp): void {
  let caught: unknown;
  try {
    fn();
  } catch (e) {
    caught = e;
  }
  expect(code(caught)).toBe(c);
  if (message) expect(String((caught as Error).message)).toMatch(message);
}

interface FakeTexture {
  label: string;
  width: number;
  height: number;
  depthOrArrayLayers: number;
  format: string;
  usage: number;
  destroyed: boolean;
  destroy(): void;
  createView(): { texture: FakeTexture };
}

/** computeFake plus textures, samplers, texture copies and build counting (G16a). */
function sparseFake() {
  const { fake, device } = computeFake();
  const f = fake as unknown as Record<string, unknown>;
  const textures: FakeTexture[] = [];
  const builds: string[] = [];
  const textureCopies: { texture: FakeTexture; origin: number[]; size: number[] }[] = [];
  f.createTexture = (d: GPUTextureDescriptor) => {
    const size = d.size as number[];
    const t: FakeTexture = {
      label: d.label ?? "",
      width: size[0]!,
      height: size[1] ?? 1,
      depthOrArrayLayers: size[2] ?? 1,
      format: d.format,
      usage: d.usage,
      destroyed: false,
      destroy() {
        this.destroyed = true;
      },
      createView() {
        return { texture: this };
      },
    };
    textures.push(t);
    return t;
  };
  f.createSampler = (d: GPUSamplerDescriptor) => ({ ...d });
  for (const name of ["createShaderModule", "createComputePipeline", "createRenderPipeline", "createComputePipelineAsync", "createRenderPipelineAsync"]) {
    const inner = f[name] as ((d: unknown) => unknown) | undefined;
    f[name] = (d: unknown) => {
      builds.push(name);
      return inner ? inner(d) : {};
    };
  }
  const encProto = Object.getPrototypeOf(fake.createCommandEncoder()) as Record<string, unknown>;
  encProto.clearBuffer = () => {};
  encProto.copyBufferToTexture = (_src: unknown, dst: { texture: FakeTexture; origin?: number[] }, size: number[]) => {
    textureCopies.push({ texture: dst.texture, origin: dst.origin ?? [0, 0, 0], size });
  };
  return { fake, device, textures, builds, textureCopies };
}

let fake: FakeDevice;
let device: GPUDevice;
let textures: FakeTexture[];
let builds: string[];
let textureCopies: { texture: FakeTexture; origin: number[]; size: number[] }[];
let pool: VDBGridPool;

beforeEach(async () => {
  ({ fake, device, textures, builds, textureCopies } = sparseFake());
  pool = await VDBGridPool.create(device, { initialBytes: 64 << 10 });
});

/** One encoder: the pool's update, then the source's; submitted. */
function frame(source: VDBSparseSourceGpu) {
  const encoder = fake.createCommandEncoder() as unknown as GPUCommandEncoder;
  pool.update(encoder);
  const u = source.update(encoder);
  fake.queue.submit([(encoder as unknown as { finish(): { copies: never[] } }).finish()]);
  return u;
}

const STATS = { meanNonZero: 0.5, max: 1, activeVoxels: 10 };
const box = (min: [number, number, number], max: [number, number, number]) => ({ min, max });

describe("vdbPackPyramidBlocks (K6.7)", () => {
  it("places blocks in rows, rows in layers, without overlap, deterministically", () => {
    const blocks = [
      [100, 40, 30],
      [64, 20, 15],
      [128, 128, 64],
      [9, 3, 3],
      [200, 10, 5],
    ];
    const max = [256, 256, 256];
    const p = vdbPackPyramidBlocks(blocks, max);
    expect(p).toEqual(vdbPackPyramidBlocks(blocks, max));
    const boxes = blocks.map((b, i) => ({ lo: p.origins[i]!, hi: p.origins[i]!.map((o, a) => o + b[a]!) }));
    for (const b of boxes) for (let a = 0; a < 3; a++) expect(b.hi[a]).toBeLessThanOrEqual(p.size[a]!);
    for (let i = 0; i < boxes.length; i++) {
      for (let j = i + 1; j < boxes.length; j++) {
        const overlap = [0, 1, 2].every((a) => boxes[i]!.lo[a]! < boxes[j]!.hi[a]! && boxes[j]!.lo[a]! < boxes[i]!.hi[a]!);
        expect(overlap, `${i} vs ${j}`).toBe(false);
      }
    }
    expect(p.size.every((s, a) => s <= max[a]!)).toBe(true);
  });

  it("puts a single row's blocks side by side and rejects what cannot fit", () => {
    expect(vdbPackPyramidBlocks([[10, 4, 2], [8, 6, 1]]).origins).toEqual([
      [0, 0, 0],
      [10, 0, 0],
    ]);
    expect(vdbPackPyramidBlocks([]).size).toEqual([1, 1, 1]);
    expect(() => vdbPackPyramidBlocks([[3000, 1, 1]])).toThrow(/texture limit/);
    expect(() => vdbPackPyramidBlocks([[8, 8, 8], [8, 8, 8], [8, 8, 8]], [8, 8, 16])).toThrow(/texture limit/);
  });
});

describe("the volume record (VdbSrcVolume)", () => {
  it("is generated from one field list: 16-byte fields, no padding, the packer writes each at index × 16", () => {
    expect(VDB_SPARSE_RECORD_BYTES).toBe(224);
    expect(VDB_SPARSE_RECORD_FIELDS.every((f) => f.type === "vec4u" || f.type === "vec4f")).toBe(true);
    const struct = sparseRecordStructWgsl();
    const names = [...struct.matchAll(/^\s+(\w+): (vec4[uf]),/gm)].map((m) => m[1]);
    expect(names).toEqual(VDB_SPARSE_RECORD_FIELDS.map((f) => f.name));
    const bytes = new ArrayBuffer(2 * VDB_SPARSE_RECORD_BYTES);
    const rec = {
      ids: [7, 8, 9, 15],
      albedo: [1, 2, 3, 0],
      lod: [9, 2, 7, 4.5],
      densityMin: [-3, 4, 5],
      densitySize: [10, 20, 30],
      maskMin: [1, 1, 1],
      maskSize: [2, 2, 2],
      albedoMin: [3, 3, 3],
      albedoSize: [4, 4, 4],
      emissionMin: [5, 5, 5],
      emissionSize: [6, 6, 6],
      pyrBase: [11, 12, 13],
      pyrSize: [21, 22, 23],
      pyrMip: [31, 32, 33, 64],
    } as const;
    packSparseRecord(bytes, VDB_SPARSE_RECORD_BYTES, rec);
    const u = new Uint32Array(bytes, VDB_SPARSE_RECORD_BYTES);
    const f = new Float32Array(bytes, VDB_SPARSE_RECORD_BYTES);
    expect([...u.subarray(0, 4)]).toEqual([7, 8, 9, 15]);
    expect(f[sparseRecordOffset("lod") / 4 + 3]).toBe(4.5);
    expect(f[sparseRecordOffset("densityMin") / 4]).toBe(-3);
    expect([...f.subarray(sparseRecordOffset("pyrMip") / 4, sparseRecordOffset("pyrMip") / 4 + 4)]).toEqual([31, 32, 33, 64]);
    expect(new Uint32Array(bytes, 0, VDB_SPARSE_RECORD_BYTES / 4).every((w) => w === 0)).toBe(true);
  });
});

describe("the sparse WGSL", () => {
  const slot = { group: 2, firstBinding: 0 };
  const lods = ["none", "coarse-pyramid", "node-average"] as const;
  it("declares only vdbSrc* / VdbSrc* names, no reserved words, no back-ticks, the frozen function set", () => {
    for (const lod of lods) {
      for (const profile of ["sample", "march"] as const) {
        for (const skipEmpty of [true, false]) {
          for (const on of [true, false]) {
            const wgsl = sparseSourceWgsl({
              slot,
              profile,
              lod,
              channels: { mask: on, albedo: on, emission: on },
              fastTrilinear: on,
              cacheAccessor: !on,
              clamp01: on,
              skipEmpty,
              maxVolumes: 64,
            });
            const where = `${lod} ${profile} skip=${skipEmpty} on=${on}`;
            expect(wgslNameProblems(wgsl), where).toEqual([]);
            expect(wgsl.includes("`"), where).toBe(false);
            const fns = [...wgsl.matchAll(/\bfn\s+(\w+)/g)].map((m) => m[1]!);
            expect(fns.every((n) => n.startsWith("vdbSrc")), where).toBe(true);
            expect(new Set(fns).size, where).toBe(fns.length);
            const frozen = ["vdbSrcDensity", "vdbSrcMask", "vdbSrcAlbedo", "vdbSrcEmission", "vdbSrcTexels", "vdbSrcLodLevels"];
            if (profile === "march") frozen.push("vdbSrcSkipEmpty", "vdbSrcMaxIn", "vdbSrcSkipSpan");
            for (const name of frozen) expect(fns, `${where} ${name}`).toContain(name);
            for (const sig of ["fn vdbSrcDensity(v: u32, uvw: vec3f, lod: f32) -> f32", "fn vdbSrcAlbedo(v: u32, uvw: vec3f, lod: f32) -> vec3f", "fn vdbSrcTexels(v: u32) -> vec3f"]) {
              expect(wgsl, where).toContain(sig);
            }
            if (profile === "march") {
              expect(wgsl).toContain("fn vdbSrcSkipEmpty(v: u32, o: vec3f, d: vec3f, t0: f32, t1: f32) -> f32");
              expect(wgsl).toContain("fn vdbSrcSkipSpan(v: u32, o: vec3f, d: vec3f, t0: f32, t1: f32, lod: f32, cutoff: f32) -> vec2f");
              expect(wgsl).toContain("fn vdbSrcMaxIn(v: u32, uvwMin: vec3f, uvwMax: vec3f) -> f32");
            }
            const globals = [...wgsl.matchAll(/^(?:@[^\n]*?\s)?var(?:<[^>]*>)?\s+(\w+)/gm)].map((m) => m[1]!);
            expect(globals.every((n) => n.startsWith("vdbSrc")), where).toBe(true);
          }
        }
      }
    }
  });

  it("matches its layout and bind entries binding for binding", () => {
    pool.add(makeImage({ gridType: 1, bboxMin: [0, 0, 0], bboxMax: [15, 15, 15] }));
    const s2 = { group: 2, firstBinding: 3 };
    for (const lod of lods) {
      const source = vdbSparseSourceGpu(pool, { lod, types: ["float"] });
      const wgsl = source.moduleWgsl(s2) + source.wgsl({ slot: s2, profile: "march" });
      const declared = [...wgsl.matchAll(/@group\((\d+)\) @binding\((\d+)\)/g)].map((m) => `${m[1]}:${m[2]}`).sort();
      const layout = source.layoutEntries(s2);
      expect(layout.map((e) => `2:${e.binding}`).sort(), lod).toEqual(declared);
      expect(source.bindEntries(s2).map((e) => e.binding), lod).toEqual(layout.map((e) => e.binding));
      expect(layout.map((e) => e.visibility)).toEqual(layout.map(() => 0x2 | 0x4));
      expect(layout.filter((e) => e.buffer?.type === "read-only-storage").length, "one storage buffer (D10)").toBe(1);
      expect(layout.length, lod).toBe(lod === "coarse-pyramid" ? 4 : 2);
      expect(layout.find((e) => e.buffer?.type === "uniform")?.buffer?.minBindingSize).toBe(64 * 224);
      source.dispose();
    }
  });

  it("asks for the module the K3.2 source needs", () => {
    const a = vdbSparseSourceGpu(pool, { lod: "node-average" }).module;
    expect([a.profile, [...a.helpers]]).toEqual(["march", ["cellBoundTrilinear", "hddaCellEnd", "nodeAverage", "fastTrilinear"]]);
    const b = vdbSparseSourceGpu(pool, { lod: "none", skipEmpty: false, fastTrilinear: false }).module;
    expect([b.profile, [...b.helpers]]).toEqual(["sample", []]);
  });
});

describe("vdbSparseSourceGpu volumes", () => {
  it("allocates the lowest free slot, frees on remove, caps at maxVolumes", () => {
    const id = pool.add(makeImage({ gridType: 1, bboxMin: [0, 0, 0], bboxMax: [7, 7, 7] }));
    const source = vdbSparseSourceGpu(pool, { lod: "none", maxVolumes: 3 });
    const [a, b, c] = [0, 1, 2].map(() => source.volume({ density: id }, { stats: STATS }));
    expect([a!.v, b!.v, c!.v]).toEqual([0, 1, 2]);
    expectCode(() => source.volume({ density: id }, { stats: STATS }), "VDB_POOL_FULL");
    b!.remove();
    expect(b!.removed).toBe(true);
    expect(source.volume({ density: id }, { stats: STATS }).v).toBe(1);
    expect(a!.source).toBe(source);
    expectCode(() => vdbSparseSourceGpu(pool, { maxVolumes: 1000 }), "VDB_BAD_OPTION", /maxVolumes/);
  });

  it("checks types, follows the TSL source's option rules and channel declarations", () => {
    const fp8 = pool.add(makeImage({ gridType: 14, bboxMin: [0, 0, 0], bboxMax: [7, 7, 7] }));
    const f = pool.add(makeImage({ gridType: 1, bboxMin: [0, 0, 0], bboxMax: [7, 7, 7], seed: 2 }));
    expectCode(() => vdbSparseSourceGpu(pool, { types: ["float"] }).volume({ density: fp8 }, { stats: STATS }), "VDB_TYPE_NOT_COMPILED");
    expectCode(() => vdbSparseSourceGpu(pool, { followImage: true }).volume({ density: f }, { indexBox: box([0, 0, 0], [3, 3, 3]) }), "VDB_BAD_OPTION", /followImage/);
    const fixed = vdbSparseSourceGpu(pool, { lod: "none", channels: { mask: true } });
    expect(fixed.volume({ density: f, mask: f }, { stats: STATS }).channels.mask).toBe(true);
    expectCode(() => fixed.volume({ density: f, emission: f }, { stats: STATS }), "VDB_BAD_OPTION", /channels/);
    const growing = vdbSparseSourceGpu(pool, { lod: "none" });
    const before = growing.wgsl({ slot: { group: 2, firstBinding: 0 }, profile: "march" });
    growing.volume({ density: f }, { stats: STATS });
    expect(growing.wgsl({ slot: { group: 2, firstBinding: 0 }, profile: "march" })).toBe(before);
    growing.volume({ density: f, albedo: [f, f, f] }, { stats: STATS });
    expect(growing.channels).toEqual({ mask: false, albedo: true, emission: false });
    expect(growing.wgsl({ slot: { group: 2, firstBinding: 0 }, profile: "march" })).not.toBe(before);
  });

  it("describes each volume on its slot: texels, LOD range, box, world box, record", () => {
    const id = pool.add(makeImage({ gridType: 1, bboxMin: [-4, 0, 2], bboxMax: [19, 9, 41], voxelSize: 0.5 }));
    const source = vdbSparseSourceGpu(pool, { pyramidMax: 8, types: ["float"] });
    const a = source.volume({ density: id }, { stats: STATS });
    const b = source.volume({ density: id }, { stats: STATS, indexBox: box([0, 0, 2], [7, 7, 9]) });
    expect(a.texels).toEqual([24, 10, 40]);
    expect(b.texels).toEqual([8, 8, 8]);
    expect(a.bbox).toEqual([
      [-2.25, -0.25, 0.75],
      [9.75, 4.75, 20.75],
    ]);
    // L0 = ceil(log2(40 / 8)) = 3; base 3x2x5; atlas levels ceil(log2 5) = 3.
    expect(a.lodLevels).toBe(6);
    expect(a.pyramid!.layout.dimensions).toEqual([3, 2, 5]);
    expect(source.lodLevels).toBe(6);
    expect(source.texels).toEqual([24, 10, 40]);
  });
});

describe("the packed pyramid atlas", () => {
  it("packs every volume's base and mip atlas into one r16float texture, shares equal pyramids, grows and retires", () => {
    const g1 = pool.add(makeImage({ gridType: 1, bboxMin: [0, 0, 0], bboxMax: [63, 31, 15] }));
    const g2 = pool.add(makeImage({ gridType: 1, bboxMin: [0, 0, 0], bboxMax: [15, 15, 15], seed: 3 }));
    const source = vdbSparseSourceGpu(pool, { pyramidMax: 16, types: ["float"] });
    const placeholder = source.atlas as unknown as FakeTexture;
    expect([placeholder.width, placeholder.height, placeholder.depthOrArrayLayers, placeholder.format]).toEqual([1, 1, 1, "r16float"]);
    const a = source.volume({ density: g1 }, { stats: STATS });
    const b = source.volume({ density: g2 }, { stats: STATS });
    const a2 = source.volume({ density: g1 }, { stats: STATS });
    const versionBefore = source.version;
    const u1 = frame(source);
    expect(u1.pyramidBuilds).toBe(2); // a and a2 share one pyramid
    expect(u1.rebind).toBe(true);
    expect(source.version).toBeGreaterThan(versionBefore);
    expect(a2.pyramid!.base).toEqual(a.pyramid!.base);
    // L0 2 (64/16 = 4): base 16x8x4, atlas 16x4x2; g2: L0 1: base 8x8x8, atlas 8x4x4.
    expect(a.pyramid!.layout.dimensions).toEqual([16, 8, 4]);
    expect(b.pyramid!.layout.dimensions).toEqual([8, 8, 8]);
    const atlas = source.atlas as unknown as FakeTexture;
    expect([atlas.width, atlas.height, atlas.depthOrArrayLayers]).toEqual([16 + 16 + 8 + 8, 8, 8]);
    expect([a.pyramid!.base, a.pyramid!.atlas, b.pyramid!.base, b.pyramid!.atlas]).toEqual([
      [0, 0, 0],
      [16, 0, 0],
      [32, 0, 0],
      [40, 0, 0],
    ]);
    // The copies land at the blocks' origins in the one texture.
    expect(textureCopies.filter((c) => c.texture === atlas).map((c) => c.origin)).toEqual([
      [0, 0, 0],
      [16, 0, 0],
      [32, 0, 0],
      [40, 0, 0],
    ]);
    // The record carries the placement.
    const table = fake.buffers.find((x) => x.label.endsWith("volumes"))!;
    const f = new Float32Array(table.data.buffer);
    const rec = (v: number, field: string) => [...f.subarray((v * 224 + sparseRecordOffset(field)) / 4, (v * 224 + sparseRecordOffset(field)) / 4 + 4)];
    expect(rec(b.v, "pyrBase").slice(0, 3)).toEqual([32, 0, 0]);
    expect(rec(b.v, "pyrMip")).toEqual([40, 0, 0, 8]);
    expect(rec(a.v, "lod")).toEqual([2 + 4, 2, 4, 1]);

    // Nothing changed: no build, no rebind.
    expect(frame(source)).toEqual({ pyramidBuilds: 0, relayouts: 0, rebind: false });
    // A third, larger pyramid: the atlas grows, every pyramid rebuilds into it; the old one is destroyed 2 updates later.
    const g3 = pool.add(makeImage({ gridType: 1, bboxMin: [0, 0, 0], bboxMax: [31, 31, 31], seed: 4 }));
    source.volume({ density: g3 }, { stats: STATS });
    const u3 = frame(source);
    expect(u3.rebind).toBe(true);
    expect(u3.pyramidBuilds).toBe(3);
    expect(source.atlas).not.toBe(atlas);
    expect(atlas.destroyed).toBe(false);
    frame(source);
    frame(source);
    expect(atlas.destroyed).toBe(false);
    frame(source);
    expect(atlas.destroyed).toBe(true);
    expect(placeholder.destroyed).toBe(true);
    source.dispose();
    expect(textures.filter((t) => !t.destroyed)).toEqual([]);
  });

  it("rebuilds a pyramid when its grid's image changes (a sequence frame), after the pool's update in the same encoder", () => {
    const g = pool.add(makeImage({ gridType: 1, bboxMin: [0, 0, 0], bboxMax: [15, 15, 15] }));
    const source = vdbSparseSourceGpu(pool, { types: ["float"] });
    const vol = source.volume({ density: g }, { stats: STATS });
    expect(frame(source).pyramidBuilds).toBe(1);
    const v0 = vol.version;
    pool.replace(g, makeImage({ gridType: 1, bboxMin: [0, 0, 0], bboxMax: [15, 15, 15], seed: 9 }));
    expect(frame(source).pyramidBuilds).toBe(1);
    expect(vol.version).toBeGreaterThan(v0);
    expect(frame(source).pyramidBuilds).toBe(0);
  });
});

describe("G16a: a coarse-first swap creates no shader module and no pipeline", () => {
  async function swap(followImage: boolean) {
    const preview = makeImage({ gridType: 16, bboxMin: [0, 0, 0], bboxMax: [11, 7, 5], voxelSize: 4 }); // fpn
    const full = makeImage({ gridType: 14, bboxMin: [0, 0, 0], bboxMax: [47, 31, 23], voxelSize: 1, seed: 5 }); // fp8
    const id = pool.add(preview);
    const source = vdbSparseSourceGpu(pool, { followImage, types: ["fpn", "fp8"] });
    const vol = source.volume({ density: id }, { stats: STATS });
    const slot = { group: 2, firstBinding: 0 };
    const wgslBefore = source.moduleWgsl(slot) + source.wgsl({ slot, profile: "march" });
    frame(source);
    frame(source);
    const before = builds.length;
    const texelsBefore = [...vol.texels];
    const versionBefore = vol.version;
    pool.replace(id, full);
    const u = frame(source);
    const after = builds.length - before;
    const wgslAfter = source.moduleWgsl(slot) + source.wgsl({ slot, profile: "march" });
    return { source, vol, u, after, texelsBefore, versionBefore, same: wgslAfter === wgslBefore };
  }

  it("followImage: boxes, pyramid size and LOD range move as values; 0 modules, 0 pipelines; the WGSL is unchanged", async () => {
    const { vol, u, after, texelsBefore, versionBefore, same } = await swap(true);
    expect(after).toBe(0);
    expect(same).toBe(true);
    expect(u.relayouts).toBe(1);
    expect(u.pyramidBuilds).toBe(1);
    expect(texelsBefore).toEqual([12, 8, 6]);
    expect(vol.texels).toEqual([48, 32, 24]);
    expect(vol.layoutUpdates).toBe(1);
    expect(vol.version).toBeGreaterThan(versionBefore);
    expect(vol.bbox[1]).toEqual([47.5, 31.5, 23.5]);
  });

  it("fail-before: without followImage the pyramid kernels of the new type compile at the swap (the counter sees builds)", async () => {
    const { after, u } = await swap(false);
    expect(u.relayouts).toBe(0);
    expect(after).toBeGreaterThan(0);
  });
});

describe("stats", () => {
  it("reads the density image back after the pool's update when no stats were given, and after every image change", async () => {
    const values = new Float32Array(16 * 16 * 16).map((_, i) => ((i * 7919) % 13) / 13);
    const image = vdbBuildFromDense(values, [16, 16, 16]);
    const id = pool.add(image);
    const source = vdbSparseSourceGpu(pool, { lod: "none", types: ["float"] });
    const vol = source.volume({ density: id });
    expect(vol.stats).toEqual({ meanNonZero: 0, max: 0, activeVoxels: 0 });
    frame(source); // copies the image into a staging buffer
    frame(source); // the copy was submitted: map it
    await source.settled();
    expect(vol.stats).toEqual(vdbSparseDensityStats(image));
    expect(vol.statsUpdates).toBe(1);
    const given = vdbSparseSourceGpu(pool, { lod: "none", types: ["float"] }).volume({ density: id }, { image });
    expect(given.stats).toEqual(vol.stats);
  });
});

describe("CPU twins (K6.7)", () => {
  it("density at lod 0 is the pool trilinear at the clamped sample position; past L0 it reads the pyramid", () => {
    const values = new Float32Array(24 * 20 * 16).map((_, i) => Math.abs(Math.sin(i * 0.37)));
    const image = vdbBuildFromDense(values, [24, 20, 16]);
    const b = box([0, 0, 0], [23, 19, 15]);
    const pyramid = vdbPyramidToHalf(vdbBoxPyramid(image, { indexBox: b, pyramidMax: 8 }));
    const vol = { lod: "coarse-pyramid" as const, density: { image, box: b }, pyramid };
    const uvw = [0.3, 0.71, 0.45];
    const p: [number, number, number] = [0.3 * 24 - 0.5, 0.71 * 20 - 0.5, 0.45 * 16 - 0.5];
    expect(vdbSparseDensityCpu(vol, uvw, 0).value).toBeCloseTo(vdbSampleTrilinear(image, p), 5);
    // At a base texel centre, lod L0 reads that texel exactly.
    const d = pyramid.dimensions;
    const centre = [(1 + 0.5) / d[0], (2 + 0.5) / d[1], (0 + 0.5) / d[2]];
    expect(vdbSparseDensityCpu(vol, centre, pyramid.level).value).toBeCloseTo(pyramid.data[1 + (2 + 0 * d[1]) * d[0]]!, 6);
    expect(vdbSparseDensityCpu(vol, centre, pyramid.level).filtered).toBe(true);
  });
});
