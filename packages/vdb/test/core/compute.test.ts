import { existsSync, readFileSync } from "node:fs";
import { beforeEach, describe, expect, it } from "vitest";
import { VDBError, VDBGridPool, applyValueTransform, decodeToAtlas, gridStats } from "../../src/core/index.js";
import { kernelWgsl } from "../../src/core/compute/shared.js";
import { NanoVDBFile, getDim, readValueF32 } from "../../src/tools/index.js";
import { computeFake, type FakePass } from "./fake-compute.js";
import { FakeBuffer, FakeDevice } from "./fake-gpu.js";
import { makeImage } from "./synthetic-image.js";

// K2.3: the core compute tools' host-side contract on the recording fake: what
// they validate before touching the device, what they record and where, and
// that they free exactly what they allocate. The kernels themselves run in
// e2e/compute.spec.ts against the tools CPU twins.

let fake: FakeDevice;
let device: GPUDevice;
let passes: FakePass[];
let pool: VDBGridPool;

beforeEach(async () => {
  ({ fake, device, passes } = computeFake());
  pool = await VDBGridPool.create(device, { initialBytes: 64 << 10 });
});

const code = (e: unknown) => (e instanceof VDBError ? e.code : String(e));
function expectCode(fn: () => unknown, c: string, message?: RegExp): void {
  let err: unknown;
  try {
    fn();
  } catch (e) {
    err = e;
  }
  expect(code(err)).toBe(c);
  if (message) expect((err as Error).message).toMatch(message);
}

describe("compute tools: validation before any device work", () => {
  it("rejects bad options, a foreign device, unknown and non-resident grids", async () => {
    const id = pool.add(makeImage({ gridType: 1 }));
    const created = fake.buffers.length;
    expectCode(() => gridStats(device, pool, id, { histogramBins: 0 }), "VDB_BAD_OPTION", /histogramBins/);
    expectCode(() => gridStats(device, pool, id, { histogramBins: 1.5 }), "VDB_BAD_OPTION", /histogramBins/);
    expectCode(() => decodeToAtlas(device, pool, id, { maxDim: 0 }), "VDB_BAD_OPTION", /maxDim/);
    expectCode(() => decodeToAtlas(device, pool, id, { filter: "bicubic" as never }), "VDB_BAD_OPTION", /filter/);
    expectCode(() => decodeToAtlas(device, pool, id, { format: "rgba" as never }), "VDB_BAD_OPTION", /format/);
    expectCode(() => decodeToAtlas(device, pool, id, { range: { min: 0, max: NaN } }), "VDB_BAD_OPTION", /range/);
    expectCode(() => gridStats(new FakeDevice().asGPU(), pool, id), "VDB_BAD_OPTION", /different GPUDevice/);
    expectCode(() => gridStats(device, pool, id), "VDB_BAD_OPTION", /pool.update/); // added, not flushed
    expectCode(() => gridStats(device, pool, 7), "VDB_UNKNOWN_GRID");
    const empty = pool.add(makeImage({ gridType: 1, bboxMin: [0, 0, 0], bboxMax: [-1, 5, 5] }));
    pool.flush();
    expectCode(() => decodeToAtlas(device, pool, empty), "VDB_BAD_OPTION", /empty index bbox/);
    expect(fake.buffers.length).toBe(created);
  });
});

describe("gridStats", () => {
  it("atomics path: binds the pool at slot 0, dispatches over the bbox, submits once, frees only its buffers", async () => {
    pool.add(makeImage({ gridType: 14, seed: 1 }));
    const id = pool.add(makeImage({ gridType: 1, bboxMin: [-4, -4, -4], bboxMax: [9, 3, 0], rootMin: 0.25, rootMax: 2 }));
    pool.flush();
    const before = new Set(fake.buffers);
    const submits = fake.submits;
    const job = gridStats(device, pool, id, { histogramBins: 32 });
    expect(fake.submits).toBe(submits + 1);
    expect(passes).toHaveLength(1);
    const pass = passes[0]!;
    expect(pass.pipeline.entry).toBe("csStatsFast");
    expect(pass.pipeline.code).toContain("@group(0) @binding(0) var<storage, read> vdbPool: array<u32>;");
    expect(pass.dispatch).toEqual([Math.ceil(14 / 4), Math.ceil(8 / 4), Math.ceil(5 / 4)]);
    expect(pass.groups[0]!.entries[0]).toEqual({ binding: 0, resource: { buffer: pool.buffer } });
    const mine = fake.buffers.filter((b) => !before.has(b));
    expect(mine).toHaveLength(4); // params, min/max, sums, readback
    const result = await job.read();
    // The fake runs no kernel: the outputs read back as initialized (no active voxels).
    expect(result).toMatchObject({ activeVoxelCount: 0, min: 0, max: 0, mean: 0, usedAtomicFastPath: true, range: { min: 0.25, max: 2 } });
    expect(result.histogram).toHaveLength(32);
    expect(mine.every((b) => b.destroyed)).toBe(true);
    expect(fake.buffers.filter((b) => before.has(b) && b.destroyed)).toEqual([]);
    expect(await job.read()).toBe(result); // read() is idempotent
  });

  it("dump path (root min < 0) records into the host's encoder and leaves the submit to the host", async () => {
    const id = pool.add(makeImage({ gridType: 1, rootMin: -1, rootMax: 1 }));
    const encoder = device.createCommandEncoder();
    pool.update(encoder);
    const submits = fake.submits;
    const job = gridStats(device, pool, id, { encoder });
    expect(fake.submits).toBe(submits);
    expect(passes[0]!.pipeline.entry).toBe("csStatsDump");
    const copies = (encoder as unknown as { copies: { size: number }[] }).copies;
    expect(copies.slice(-2).map((c) => c.size)).toEqual([8 * 8 * 8 * 4, 8 * 8 * 8 * 4]);
    device.queue.submit([encoder.finish()]);
    const r = await job.submitted().read();
    expect(r.usedAtomicFastPath).toBe(false);
    expect(r.activeVoxelCount).toBe(0);
  });

  it("with a host encoder, read() before submitted() rejects and maps or frees nothing", async () => {
    const id = pool.add(makeImage({ gridType: 1 }));
    pool.flush();
    const before = new Set(fake.buffers);
    const encoder = device.createCommandEncoder();
    const job = gridStats(device, pool, id, { encoder });
    const mine = fake.buffers.filter((b) => !before.has(b));
    let mapped = 0;
    for (const b of mine) {
      const map = b.mapAsync.bind(b);
      b.mapAsync = (...args: Parameters<FakeBuffer["mapAsync"]>) => (mapped++, map(...args));
    }
    await expect(job.read()).rejects.toMatchObject({ code: "VDB_BAD_OPTION", message: expect.stringMatching(/submit.*job.submitted\(\).read\(\)/) });
    expect(mapped).toBe(0);
    expect(mine.some((b) => b.destroyed)).toBe(false); // the host's submit can still use them
    device.queue.submit([encoder.finish()]);
    expect(job.submitted()).toBe(job);
    const r = await job.read();
    expect(r.histogram).toHaveLength(256);
    expect(mapped).toBe(1);
    expect(mine.every((b) => b.destroyed)).toBe(true);
    // Same contract on decodeToAtlas; without an encoder, read() needs no submitted().
    const atlasEncoder = device.createCommandEncoder();
    const atlas = decodeToAtlas(device, pool, id, { encoder: atlasEncoder });
    await expect(atlas.read()).rejects.toMatchObject({ code: "VDB_BAD_OPTION" });
    device.queue.submit([atlasEncoder.finish()]);
    expect((await atlas.submitted().read()).dims).toEqual(atlas.dims);
    expect((await decodeToAtlas(device, pool, id).read()).dims).toEqual(atlas.dims);
  });

  it("dump path: refuses a bbox over the device's binding / buffer limits before making any buffer", () => {
    const id = pool.add(makeImage({ gridType: 1, rootMin: -1, rootMax: 1, bboxMin: [0, 0, 0], bboxMax: [63, 63, 63] }));
    pool.flush();
    const limits = fake.limits as { maxStorageBufferBindingSize: number; maxBufferSize: number };
    const made = fake.buffers.length;
    limits.maxStorageBufferBindingSize = 512 << 10; // each output is 64^3 * 4 = 1 MiB
    expectCode(() => gridStats(device, pool, id), "VDB_BAD_OPTION", /dump path.*262144 voxels.*maxStorageBufferBindingSize 524288.*gridStatsCpu/);
    limits.maxStorageBufferBindingSize = 128 << 20;
    limits.maxBufferSize = 1536 << 10; // the 2 MiB readback does not fit
    expectCode(() => gridStats(device, pool, id), "VDB_BAD_OPTION", /2097152 B of readback/);
    expect(fake.buffers.length).toBe(made);
    limits.maxBufferSize = 256 << 20;
    expect(() => gridStats(device, pool, id).dispose()).not.toThrow();
  });

  it("dispose() frees the buffers and makes read() reject", async () => {
    const id = pool.add(makeImage({ gridType: 1 }));
    pool.flush();
    const before = new Set(fake.buffers);
    const job = gridStats(device, pool, id);
    job.dispose();
    expect(fake.buffers.filter((b) => !before.has(b)).every((b) => b.destroyed)).toBe(true);
    await expect(job.read()).rejects.toMatchObject({ code: "VDB_DISPOSED" });
  });
});

describe("decodeToAtlas", () => {
  it("one thread per output voxel, the root stats as the default range, and v1's dims clamp", async () => {
    const id = pool.add(makeImage({ gridType: 16, bboxMin: [0, 0, 0], bboxMax: [99, 9, 0], rootMin: 0.5, rootMax: 3 }));
    pool.flush();
    const job = decodeToAtlas(device, pool, id, { maxDim: 64, filter: "nearest" });
    expect(job.dims).toEqual([64, 10, 1]);
    expect(job.range).toEqual({ min: 0.5, max: 3 });
    expect(job.buffer.size).toBe(64 * 10 * 4);
    expect(passes[0]!.pipeline.entry).toBe("csAtlas");
    expect(passes[0]!.dispatch).toEqual([16, 3, 1]);
    const r = await job.read();
    expect(r.data).toBeInstanceOf(Uint8Array);
    expect(r.data).toHaveLength(640);
    expect(r.dims).toEqual([64, 10, 1]);
    expect(decodeToAtlas(device, pool, id, { normalize: false }).range).toEqual({ min: 0, max: 1 });
    expect(decodeToAtlas(device, pool, id, { range: { min: -2, max: 2 } }).range).toEqual({ min: -2, max: 2 });
  });

  it("readback: false keeps the GPU buffer until dispose() and has nothing to read", async () => {
    const id = pool.add(makeImage({ gridType: 1 }));
    pool.flush();
    const before = new Set(fake.buffers);
    const job = decodeToAtlas(device, pool, id, { readback: false });
    expect(fake.buffers.filter((b) => !before.has(b))).toHaveLength(2); // params + output, no staging
    expect((job.buffer as unknown as FakeBuffer).destroyed).toBe(false);
    await expect(job.read()).rejects.toMatchObject({ code: "VDB_BAD_OPTION" });
    expect((job.buffer as unknown as FakeBuffer).destroyed).toBe(false);
    job.dispose();
    expect((job.buffer as unknown as FakeBuffer).destroyed).toBe(true);
  });

  it("one generated module per (kernel, type): default options, so the WGSL size snapshot does not move", () => {
    const wgsl = kernelWgsl({ entryPoint: "x", footer: "", outputs: 0 }, "fp8");
    expect(wgsl).toContain("fn vdbSample(");
    expect(wgsl).toContain("fn vdbIsActive(");
    expect(wgsl).toContain("var<uniform> csParams: CsParams;");
  });
});

const fixture = new URL("../../../../fixtures/primitives/box_fog_float.nvdb", import.meta.url);

describe("applyValueTransform", () => {
  it("checks the transform, the grid type and the source image before staging anything", () => {
    const fp8 = pool.add(makeImage({ gridType: 14 }));
    const float = pool.add(makeImage({ gridType: 1, bytes: 1024 }));
    const rev = pool.revision;
    expectCode(() => applyValueTransform(pool, float, new Uint32Array(256), "triple" as never), "VDB_BAD_OPTION", /double, half, negate, clamp01/);
    expectCode(() => applyValueTransform(pool, fp8, new Uint32Array(256), "double"), "VDB_UNSUPPORTED_GRID_TYPE", /Float/);
    expectCode(() => applyValueTransform(pool, float, new Uint32Array(8), "double"), "VDB_BAD_OPTION", /1024 bytes/);
    expect(pool.revision).toBe(rev);
  });

  it.skipIf(!existsSync(fixture))("stages the transformed image with replace (in place: no version change)", () => {
    const image = NanoVDBFile.fromArrayBuffer(new Uint8Array(readFileSync(fixture)).buffer).gridImage(0);
    const id = pool.add(image);
    pool.flush();
    const version = pool.version;
    const out = applyValueTransform(pool, id, image, (v) => v * 3);
    expect(out).not.toBe(image);
    expect(pool.info(id).resident).toBe(false);
    pool.flush();
    expect(pool.version).toBe(version);
    const info = pool.info(id);
    expect(info.resident).toBe(true);
    const gpu = (pool.buffer as unknown as FakeBuffer).data.subarray(info.imageOffset!, info.imageOffset! + info.imageBytes);
    expect(new Uint8Array(out.buffer, out.byteOffset, out.byteLength)).toEqual(gpu);
    const leaf = ([[-40, -20, -30], [40, 20, 30], [0, 0, 0], [-40, 0, 0]] as [number, number, number][]).find((p) => getDim(image, p) === 1)!;
    expect(readValueF32(out, leaf)).toBe(Math.fround(readValueF32(image, leaf) * 3));
  }, 30_000); // CPU transform of a full fixture: slow under parallel load
});
