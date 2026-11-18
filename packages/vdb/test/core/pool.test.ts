import { existsSync, readFileSync } from "node:fs";
import { beforeEach, describe, expect, it } from "vitest";
import {
  VDBGridPool,
  VDB_POOL_RECORD_WORDS,
  VDB_RECORD_FLAG,
  decodeHeader,
  decodeRecord,
  type VDBBrickMajorant,
} from "../../src/core/internal.js";
import { NvdbFile } from "../../src/tools/index.js";
import { FakeBuffer, FakeDevice, USAGE, type FakeOp } from "./fake-gpu.js";
import { makeImage } from "./synthetic-image.js";

// ---------------------------------------------------------------------------
// helpers

function code(fn: () => unknown): string {
  try {
    fn();
  } catch (e) {
    return (e as { code?: string }).code ?? `untyped: ${(e as Error).message}`;
  }
  return "no error";
}

async function codeAsync(fn: () => Promise<unknown>): Promise<string> {
  try {
    await fn();
  } catch (e) {
    return (e as { code?: string }).code ?? `untyped: ${(e as Error).message}`;
  }
  return "no error";
}

const fake = (pool: VDBGridPool) => pool.buffer as unknown as FakeBuffer;
const recordAt = (pool: VDBGridPool, id: number) => decodeRecord(fake(pool).words(), 8 + id * VDB_POOL_RECORD_WORDS);
const recordWords = (pool: VDBGridPool, id: number) =>
  Array.from(fake(pool).words().subarray(8 + id * VDB_POOL_RECORD_WORDS, 8 + (id + 1) * VDB_POOL_RECORD_WORDS));
const header = (pool: VDBGridPool) => decodeHeader(fake(pool).words());
const bytesOf = (a: ArrayBufferView) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
/** Fast byte equality (toEqual on multi-MB arrays is slow): true, or the first differing index. */
function sameBytes(a: Uint8Array, b: Uint8Array): true | string {
  if (a.length !== b.length) return `length ${a.length} vs ${b.length}`;
  if (Buffer.from(a.buffer, a.byteOffset, a.length).equals(Buffer.from(b.buffer, b.byteOffset, b.length))) return true;
  return `first difference at byte ${a.findIndex((x, k) => x !== b[k])}`;
}

/** The bytes the pool holds for grid `id`'s image. */
function imageInPool(pool: VDBGridPool, id: number): Uint8Array {
  const info = pool.info(id);
  return fake(pool).data.subarray(info.imageOffset!, info.imageOffset! + info.imageBytes);
}

const kinds = (ops: FakeOp[], kind: FakeOp["kind"]) => ops.filter((o) => o.kind === kind);
const copies = (ops: FakeOp[]) => ops.filter((o): o is Extract<FakeOp, { kind: "copy" }> => o.kind === "copy");
const writes = (ops: FakeOp[]) => ops.filter((o): o is Extract<FakeOp, { kind: "write" }> => o.kind === "write");

let device: FakeDevice;
beforeEach(() => {
  device = new FakeDevice();
});

// With maxGrids 4 the table ends at align32(32 + 4 * 160) = 672.
const DATA_START_4 = 672;

// ---------------------------------------------------------------------------

describe("VDBGridPool.create and device checks", () => {
  it("allocates one STORAGE | COPY_DST | COPY_SRC buffer and writes an empty header", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 8192, label: "clouds" });
    expect(device.buffers).toHaveLength(1);
    const buf = fake(pool);
    expect(buf.usage).toBe(USAGE.STORAGE | USAGE.COPY_DST | USAGE.COPY_SRC);
    expect(buf.size).toBe(8192);
    expect(buf.label).toContain("clouds");
    expect(header(pool)).toEqual({
      magic: 0x50424456, layoutVersion: 1, gridCapacity: 16, gridCount: 0, tableOffset: 32, recordBytes: 160,
      usedBytes: 2592,
    });
    expect([pool.version, pool.revision, pool.layoutVersion]).toEqual([0, 0, 1]);
    expect(device.errorScopes).toBe(0); // pushed and popped
  });

  it("ORs extra usage flags and defaults to 1 MiB", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { usage: USAGE.MAP_READ });
    expect(fake(pool).usage).toBe(USAGE.STORAGE | USAGE.COPY_DST | USAGE.COPY_SRC | USAGE.MAP_READ);
    expect(pool.byteLength).toBe(1 << 20);
  });

  it("reports unsupported devices and bad options", async () => {
    expect(VDBGridPool.unsupportedReason(device.asGPU())).toBeNull();
    expect(VDBGridPool.isSupported(device.asGPU())).toBe(true);
    const none = null as unknown as GPUDevice;
    expect(VDBGridPool.unsupportedReason(none)).toMatch(/requestDevice/);
    const compat = new FakeDevice({ maxStorageBuffersPerShaderStage: 0 }).asGPU();
    expect(VDBGridPool.isSupported(compat)).toBe(false);
    expect(await codeAsync(() => VDBGridPool.create(compat))).toBe("VDB_UNSUPPORTED_DEVICE");
    expect(await codeAsync(() => VDBGridPool.create(device.asGPU(), { maxGrids: 0 }))).toBe("VDB_BAD_OPTION");
    expect(await codeAsync(() => VDBGridPool.create(device.asGPU(), { initialBytes: -5 }))).toBe("VDB_BAD_OPTION");
    expect(await codeAsync(() => VDBGridPool.create(device.asGPU(), { maxGrids: 1000, maxBytes: 4096 }))).toBe(
      "VDB_BAD_OPTION",
    );
  });

  it("requiredLimits asks for the adapter's maxima", () => {
    const adapter = { limits: { maxStorageBufferBindingSize: 2 ** 31, maxBufferSize: 2 ** 32 } } as unknown as GPUAdapter;
    expect(VDBGridPool.requiredLimits(adapter)).toEqual({ maxStorageBufferBindingSize: 2 ** 31, maxBufferSize: 2 ** 32 });
  });

  it("clamps the pool to the device binding limit", async () => {
    const small = new FakeDevice({ maxStorageBufferBindingSize: 4096 });
    const pool = await VDBGridPool.create(small.asGPU(), { maxGrids: 4 });
    expect(pool.byteLength).toBe(4096);
  });
});

describe("staging: add / replace / remove", () => {
  it("add returns an id at once and touches the GPU only in update", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 65536, maxGrids: 4 });
    const image = makeImage({ bytes: 1024, seed: 7 });
    const mark = device.log.length;
    const id = pool.add(image, { label: "a" });
    expect(id).toBe(0);
    expect(device.log.length).toBe(mark); // nothing reached the GPU
    expect(pool.info(id)).toMatchObject({ resident: false, imageOffset: DATA_START_4, label: "a", type: "float" });
    expect(pool.revision).toBe(1);

    const ops = device.updateAndSubmit(pool);
    expect(kinds(ops, "create")).toHaveLength(0);
    expect(copies(ops)).toHaveLength(0);
    const w = writes(ops);
    expect(w.map((o) => [o.offset, o.size])).toEqual([
      [DATA_START_4, 1024], // image
      [32, 160], // record 0
      [0, 32], // header
    ]);
    expect(sameBytes(imageInPool(pool, id), bytesOf(image))).toBe(true);
    expect(recordAt(pool, id)).toMatchObject({ imageOffset: DATA_START_4, imageBytes: 1024, gridType: 1, live: true });
    expect(header(pool)).toMatchObject({ gridCount: 1, usedBytes: DATA_START_4 + 1024 });
    expect(pool.info(id).resident).toBe(true);
    expect(pool.version).toBe(0);
  });

  it("replace within the reserved capacity writes in place without a version bump", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 65536, maxGrids: 4 });
    const id = pool.add(makeImage({ bytes: 1024 }), { capacity: 4096 });
    pool.flush();
    const before = pool.info(id);
    expect(before.capacity).toBe(4096);
    const next = makeImage({ bytes: 4000, seed: 9, gridType: 14 });
    pool.replace(id, next);
    const ops = device.updateAndSubmit(pool);
    expect(kinds(ops, "create")).toHaveLength(0);
    expect(writes(ops)[0]).toMatchObject({ offset: before.imageOffset, size: 4000 });
    expect(pool.info(id)).toMatchObject({ imageOffset: before.imageOffset, imageBytes: 4000, type: "fp8" });
    expect(sameBytes(imageInPool(pool, id), bytesOf(next))).toBe(true);
    expect(pool.version).toBe(0);
    expect(pool.revision).toBe(2);
    expect([...pool.types]).toEqual(["fp8"]);
  });

  it("imageRevision (K3.2a) moves on add and replace of that grid only, not on setTransform or another grid", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 65536, maxGrids: 4 });
    const a = pool.add(makeImage({ bytes: 1024 }), { capacity: 4096 });
    const b = pool.add(makeImage({ bytes: 1024, seed: 3 }));
    expect([pool.info(a).imageRevision, pool.info(b).imageRevision]).toEqual([1, 2]);
    pool.flush();
    pool.replace(b, makeImage({ bytes: 1024, seed: 4 }));
    expect([pool.info(a).imageRevision, pool.info(b).imageRevision]).toEqual([1, 3]);
    pool.setTransform(a, [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]);
    expect(pool.revision).toBe(4);
    expect(pool.info(a).imageRevision).toBe(1);
    pool.replace(a, makeImage({ bytes: 2048, seed: 5 }));
    expect(pool.info(a).imageRevision).toBe(5);
    // A reused id gets a new one.
    pool.remove(b);
    const c = pool.add(makeImage({ bytes: 1024, seed: 6 }));
    expect(c).toBe(b);
    expect(pool.info(c).imageRevision).toBe(pool.revision);
    expect(pool.info(c).imageRevision).toBeGreaterThan(5);
  });

  it("replace past the capacity moves the image (free + allocate)", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 65536, maxGrids: 4 });
    const a = pool.add(makeImage({ bytes: 1024 }));
    const b = pool.add(makeImage({ bytes: 1024, seed: 2 }));
    pool.flush();
    const big = makeImage({ bytes: 3000, seed: 3 });
    pool.replace(a, big);
    pool.flush();
    expect(pool.info(a).imageOffset).toBe(DATA_START_4 + 2048); // after b
    expect(pool.info(b).imageOffset).toBe(DATA_START_4 + 1024);
    expect(sameBytes(imageInPool(pool, a), bytesOf(big))).toBe(true);
    expect(pool.version).toBe(0);
  });

  it("remove clears the record and the id is reused", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 65536, maxGrids: 4 });
    const a = pool.add(makeImage());
    const b = pool.add(makeImage({ seed: 2 }));
    pool.flush();
    expect(header(pool).gridCount).toBe(2);
    pool.remove(b);
    pool.flush();
    expect(recordWords(pool, b).every((w) => w === 0)).toBe(true);
    expect(recordAt(pool, b).live).toBe(false);
    expect(header(pool).gridCount).toBe(1);
    expect(code(() => pool.info(b))).toBe("VDB_UNKNOWN_GRID");
    pool.remove(a);
    expect(pool.add(makeImage({ seed: 3 }))).toBe(0);
    for (const id of [-1, 1.5, 99, Number.NaN]) expect(code(() => pool.remove(id))).toBe("VDB_UNKNOWN_GRID");
  });

  it("validates images before staging", async () => {
    const pool = await VDBGridPool.create(device.asGPU());
    expect(code(() => pool.add(new Uint32Array(64)))).toBe("VDB_BAD_IMAGE");
    expect(code(() => pool.add(makeImage({ gridType: 7 })))).toBe("VDB_UNSUPPORTED_GRID_TYPE");
    expect(pool.revision).toBe(0);
    expect(pool.gridCount).toBe(0);
  });
});

describe("growth and compaction", () => {
  it("grows once per update, copies only ranges resident before it, then destroys the old buffer", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 4096, maxGrids: 4 });
    const imgA = makeImage({ bytes: 1024, seed: 1 });
    const imgB = makeImage({ bytes: 2048, seed: 2 });
    const a = pool.add(imgA);
    const b = pool.add(imgB);
    pool.flush();
    const old = fake(pool);
    const oldA = pool.info(a).imageOffset!;
    const oldB = pool.info(b).imageOffset!;

    const imgC = makeImage({ bytes: 1024, seed: 3 });
    const imgD = makeImage({ bytes: 2000, seed: 4 });
    const c = pool.add(imgC);
    const d = pool.add(imgD);
    expect(pool.info(c).imageOffset).toBeNull(); // doesn't fit: placed by update
    const ops = device.updateAndSubmit(pool);

    const created = kinds(ops, "create");
    expect(created).toHaveLength(1);
    expect(pool.version).toBe(1);
    expect(fake(pool)).not.toBe(old);
    // max(needed, 2 x current): needed = 672 + 1024 + 2048 + 1024 + 2016 = 6784 < 8192.
    expect(pool.byteLength).toBe(8192);
    expect(copies(ops).map((o) => [o.src, o.srcOffset, o.dst, o.dstOffset, o.size])).toEqual([
      [old, oldA, fake(pool), pool.info(a).imageOffset, 1024],
      [old, oldB, fake(pool), pool.info(b).imageOffset, 2048],
    ]);
    // New images are written, not copied.
    const written = writes(ops).filter((o) => o.buffer === fake(pool)).map((o) => o.offset);
    expect(written).toContain(pool.info(c).imageOffset);
    expect(written).toContain(pool.info(d).imageOffset);
    expect(writes(ops).some((o) => o.buffer === old)).toBe(false);

    // Retired, not destroyed: kept until the growth is confirmed, two updates pass and the queue drains
    // (pool-lifecycle.test.ts covers the gating).
    expect(old.destroyed).toBe(false);
    expect(pool.retiredBytes).toBe(4096);
    for (const [id, img] of [[a, imgA], [b, imgB], [c, imgC], [d, imgD]] as const) {
      expect(sameBytes(imageInPool(pool, id), bytesOf(img))).toBe(true);
    }
    expect(header(pool)).toMatchObject({ gridCapacity: 4, gridCount: 4 });

    await pool.settled();
    device.updateAndSubmit(pool);
    const next = device.updateAndSubmit(pool);
    expect(kinds(next, "destroy")).toEqual([]);
    const mark = device.log.length;
    await device.completeWork();
    expect(old.destroyed).toBe(true);
    expect(kinds(device.since(mark), "destroy")).toEqual([{ kind: "destroy", buffer: old }]);
    expect(pool.retiredBytes).toBe(0);
    expect(pool.version).toBe(1);
  });

  it("does not copy a resident range that is being rewritten in the same update", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 4096, maxGrids: 4 });
    const a = pool.add(makeImage({ bytes: 1024, seed: 1 }));
    const b = pool.add(makeImage({ bytes: 1024, seed: 2 }));
    pool.flush();
    const replacement = makeImage({ bytes: 1024, seed: 5 });
    pool.replace(a, replacement); // in place, pending
    pool.add(makeImage({ bytes: 3000, seed: 6 })); // forces growth
    const ops = device.updateAndSubmit(pool);
    expect(pool.version).toBe(1);
    expect(copies(ops)).toHaveLength(1);
    expect(copies(ops)[0]!.dstOffset).toBe(pool.info(b).imageOffset);
    expect(sameBytes(imageInPool(pool, a), bytesOf(replacement))).toBe(true);
  });

  it("doubles the table through the growth path", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 65536, maxGrids: 2 });
    const ids = [0, 1, 2].map((k) => pool.add(makeImage({ seed: k + 1 })));
    expect(ids).toEqual([0, 1, 2]);
    expect(pool.info(2).imageOffset).toBeNull();
    pool.flush();
    expect(pool.version).toBe(1);
    expect(pool.gridCapacity).toBe(4);
    expect(header(pool)).toMatchObject({ gridCapacity: 4, gridCount: 3 });
    for (const id of ids) {
      expect(recordAt(pool, id).imageOffset).toBe(pool.info(id).imageOffset);
      expect(pool.info(id).imageOffset! >= DATA_START_4).toBe(true);
    }
  });

  it("compacts into a same-size buffer when first-fit fails but free bytes suffice", async () => {
    const size = DATA_START_4 + 4 * 1024;
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: size, maxGrids: 4 });
    const imgs = [1, 2, 3, 4].map((seed) => makeImage({ bytes: 1024, seed }));
    const [a, b, c, d] = imgs.map((img) => pool.add(img)) as [number, number, number, number];
    pool.flush();
    const old = fake(pool);
    pool.remove(a);
    pool.remove(c);
    const imgE = makeImage({ bytes: 2048, seed: 5 });
    const e = pool.add(imgE);
    expect(pool.info(e).imageOffset).toBeNull(); // two 1 KiB gaps, no 2 KiB one
    const ops = device.updateAndSubmit(pool);
    expect(pool.version).toBe(1);
    expect(pool.byteLength).toBe(size);
    expect(fake(pool)).not.toBe(old);
    expect(copies(ops).map((o) => o.size)).toEqual([1024, 1024]); // b and d only
    expect(sameBytes(imageInPool(pool, b), bytesOf(imgs[1]!))).toBe(true);
    expect(sameBytes(imageInPool(pool, d), bytesOf(imgs[3]!))).toBe(true);
    expect(sameBytes(imageInPool(pool, e), bytesOf(imgE))).toBe(true);
  });

  it("places into gaps freed since staging without moving anything", async () => {
    const size = DATA_START_4 + 2 * 1024;
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: size, maxGrids: 4 });
    const a = pool.add(makeImage({ bytes: 1024, seed: 1 }));
    pool.add(makeImage({ bytes: 1024, seed: 2 }));
    pool.flush();
    const c = pool.add(makeImage({ bytes: 1024, seed: 3 }));
    expect(pool.info(c).imageOffset).toBeNull();
    pool.remove(a);
    const ops = device.updateAndSubmit(pool);
    expect(pool.version).toBe(0);
    expect(copies(ops)).toHaveLength(0);
    expect(pool.info(c).imageOffset).toBe(DATA_START_4);
  });

  it("VDB_POOL_LIMIT in update keeps the buffer and writes only the table", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 4096, maxGrids: 4, maxBytes: 8192 });
    const a = pool.add(makeImage({ bytes: 1024 }));
    pool.flush();
    const buffer = pool.buffer;
    // Each fits alone (staging accepts them); together they pass the limit: 672 + 1024 + 2 * 4000 > 8192.
    const big = pool.add(makeImage({ bytes: 4000, seed: 2 }));
    const big2 = pool.add(makeImage({ bytes: 4000, seed: 3 }));
    const revision = pool.revision;
    const mark = device.log.length;
    const encoder = device.createCommandEncoder() as unknown as GPUCommandEncoder;
    let err: Error & { code?: string } = new Error("none");
    try {
      pool.update(encoder);
    } catch (e) {
      err = e as typeof err;
    }
    expect(err.code).toBe("VDB_POOL_LIMIT");
    expect(err.message).toMatch(/8192/);
    expect(err.message).toMatch(/requiredLimits/);
    // Nothing created or copied; only the two (dead) records and the header were written.
    const ops = device.since(mark);
    expect(ops.filter((o) => o.kind !== "write")).toEqual([]);
    expect(writes(ops).every((o) => o.offset + o.size <= DATA_START_4)).toBe(true);
    expect([pool.buffer, pool.version, pool.byteLength, pool.revision, pool.retiredBytes]).toEqual([
      buffer, 0, 4096, revision, 0,
    ]);
    expect(pool.info(big).imageOffset).toBeNull();
    expect(recordAt(pool, big).live).toBe(false);
    expect(pool.info(a).resident).toBe(true);
    // Recoverable: drop a grid and update again (now it grows).
    pool.remove(big2);
    pool.flush();
    expect(pool.version).toBe(1);
    expect(pool.info(big).resident).toBe(true);
  });

  it("VDB_POOL_LIMIT also applies at the device limit, at staging", async () => {
    const small = new FakeDevice({ maxStorageBufferBindingSize: 8192 });
    const pool = await VDBGridPool.create(small.asGPU(), { initialBytes: 4096, maxGrids: 4 });
    expect(code(() => pool.add(makeImage({ bytes: 9000 })))).toBe("VDB_POOL_LIMIT");
    expect(pool.gridCount).toBe(0);
    expect(code(() => pool.flush())).toBe("no error");
    expect(small.buffers).toHaveLength(1);
  });
});

describe("dispose and host buffers", () => {
  it("dispose destroys exactly the buffers the pool created", async () => {
    const other = device.createBuffer({ size: 64, usage: USAGE.STORAGE });
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 4096, maxGrids: 4 });
    pool.add(makeImage({ bytes: 3000 }));
    pool.flush();
    pool.add(makeImage({ bytes: 3000, seed: 2 }));
    pool.flush(); // grows: the 4 KiB buffer is retired
    const [first, current] = [device.buffers[1]!, fake(pool)];
    expect(first.destroyed).toBe(false);
    pool.dispose();
    expect(first.destroyCalls).toBe(1);
    expect(current.destroyCalls).toBe(1);
    expect(other.destroyed).toBe(false);
    pool.dispose(); // idempotent
    expect(current.destroyCalls).toBe(1);
    for (const call of [() => pool.add(makeImage()), () => pool.flush(), () => pool.info(0), () => pool.remove(0)]) {
      expect(code(call)).toBe("VDB_DISPOSED");
    }
  });

  it("a host buffer is checked, never grown and never destroyed", async () => {
    const host = device.createBuffer({ size: DATA_START_4 + 2048, usage: USAGE.STORAGE | USAGE.COPY_DST });
    host.data.fill(0xab); // stale contents
    const created = device.buffers.length;
    const pool = await VDBGridPool.create(device.asGPU(), { buffer: host as unknown as GPUBuffer, maxGrids: 4 });
    expect(device.buffers.length).toBe(created);
    expect(header(pool)).toMatchObject({ gridCapacity: 4, gridCount: 0 });
    expect(fake(pool).data.subarray(32, DATA_START_4).every((x) => x === 0)).toBe(true); // table cleared

    const a = pool.add(makeImage({ bytes: 1024 }));
    pool.add(makeImage({ bytes: 1024, seed: 2 }));
    const revision = pool.revision;
    expect(code(() => pool.add(makeImage({ bytes: 1024, seed: 3 })))).toBe("VDB_POOL_FULL");
    expect(code(() => pool.replace(a, makeImage({ bytes: 2048, seed: 4 })))).toBe("VDB_POOL_FULL");
    expect(pool.revision).toBe(revision);
    expect(pool.gridCount).toBe(2);
    expect(pool.info(a)).toMatchObject({ imageOffset: DATA_START_4, imageBytes: 1024 });
    pool.flush();
    expect(device.buffers.length).toBe(created);
    pool.remove(a);
    pool.add(makeImage({ bytes: 1024, seed: 5 })); // fits the freed range again
    pool.flush();
    pool.dispose();
    expect(host.destroyCalls).toBe(0);
  });

  it("host buffer table capacity is fixed", async () => {
    const host = device.createBuffer({ size: 65536, usage: USAGE.STORAGE | USAGE.COPY_DST });
    const pool = await VDBGridPool.create(device.asGPU(), { buffer: host as unknown as GPUBuffer, maxGrids: 2 });
    pool.add(makeImage());
    pool.add(makeImage());
    expect(code(() => pool.add(makeImage()))).toBe("VDB_POOL_FULL");
  });

  it("rejects unusable host buffers", async () => {
    const make = (size: number, usage: number) => device.createBuffer({ size, usage }) as unknown as GPUBuffer;
    const ok = USAGE.STORAGE | USAGE.COPY_DST;
    expect(await codeAsync(() => VDBGridPool.create(device.asGPU(), { buffer: make(8192, USAGE.STORAGE) }))).toBe(
      "VDB_BAD_OPTION",
    );
    expect(await codeAsync(() => VDBGridPool.create(device.asGPU(), { buffer: make(512, ok) }))).toBe("VDB_BAD_OPTION");
    expect(
      await codeAsync(() => VDBGridPool.create(device.asGPU(), { buffer: make(8192, ok), initialBytes: 8192 })),
    ).toBe("VDB_BAD_OPTION");
  });
});

describe("alignment, bind entries, majorants", () => {
  it("every range starts on 32 bytes, before and after growth", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 4096, maxGrids: 4 });
    const sizes = [804, 900, 1028, 2052, 1996, 836];
    const ids = sizes.map((bytes, k) => pool.add(makeImage({ bytes, seed: k }), { capacity: bytes + 4 }));
    const majorant: VDBBrickMajorant = { brickLog2: 3, dims: [3, 1, 1], data: new Float32Array([1, 2, 3]) };
    pool.setMajorant(ids[1]!, majorant);
    for (let round = 0; round < 2; round++) {
      pool.flush();
      for (const id of ids) {
        const info = pool.info(id);
        expect(info.imageOffset! % 32, `grid ${id}`).toBe(0);
        expect(info.capacity % 32).toBe(0);
      }
      expect(pool.info(ids[1]!).majorantOffset! % 32).toBe(0);
      pool.add(makeImage({ bytes: 5000, seed: 99 })); // forces a second growth
    }
    expect(pool.version).toBe(2);
  });

  it("bindGroupEntry follows the current buffer", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 4096, maxGrids: 4 });
    const slot = { group: 2, firstBinding: 20 };
    expect(pool.bindGroupEntry(slot)).toEqual({ binding: 20, resource: { buffer: pool.buffer } });
    pool.add(makeImage({ bytes: 8000 }));
    pool.flush();
    expect(pool.bindGroupEntry(slot)).toEqual({ binding: 20, resource: { buffer: pool.buffer } });
    expect(code(() => pool.bindGroupEntry({ group: 0, firstBinding: -1 }))).toBe("VDB_BAD_OPTION");
  });

  it("stores, replaces and clears a brick majorant range", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 65536, maxGrids: 4 });
    const data = new Float32Array(24).map((_, k) => k * 0.5);
    const id = pool.add(makeImage(), { majorant: { brickLog2: 3, dims: [2, 3, 4], data } });
    pool.flush();
    const rec = recordAt(pool, id);
    expect(rec).toMatchObject({ hasBrickMajorant: true, brickLog2: 3, majorantDims: [2, 3, 4] });
    expect(rec.majorantOffset).toBe(pool.info(id).majorantOffset);
    const stored = new Float32Array(fake(pool).data.buffer, rec.majorantOffset, 24);
    expect(Array.from(stored)).toEqual(Array.from(data));

    // Same size: rewritten in place.
    const data2 = new Float32Array(24).fill(7);
    pool.setMajorant(id, { brickLog2: 3, dims: [4, 3, 2], data: data2 });
    pool.flush();
    expect(recordAt(pool, id)).toMatchObject({ majorantOffset: rec.majorantOffset, majorantDims: [4, 3, 2] });

    pool.setMajorant(id, null);
    pool.flush();
    expect(recordAt(pool, id)).toMatchObject({ hasBrickMajorant: false, brickLog2: 0, majorantOffset: 0, majorantDims: [0, 0, 0] });
    expect(pool.usedBytes).toBe(DATA_START_4 + 1024);

    pool.setMajorant(id, { brickLog2: 3, dims: [1, 1, 1], data: new Float32Array([1]) });
    pool.replace(id, makeImage({ seed: 4 })); // the old majorant described the old image
    pool.flush();
    expect(pool.info(id).hasBrickMajorant).toBe(false);

    const bad = (m: Partial<VDBBrickMajorant>) =>
      code(() => pool.setMajorant(id, { brickLog2: 3, dims: [1, 1, 1], data: new Float32Array(1), ...m }));
    expect(bad({ data: new Float32Array(2) })).toBe("VDB_BAD_OPTION");
    expect(bad({ dims: [0, 1, 1] })).toBe("VDB_BAD_OPTION");
    expect(bad({ brickLog2: -1 })).toBe("VDB_BAD_OPTION");
    expect(bad({ data: [1] as unknown as Float32Array })).toBe("VDB_BAD_OPTION");
  });

  it("valueMax: stats bound, caller bound, and no-stats", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 65536, maxGrids: 4 });
    const a = pool.add(makeImage({ rootMax: 2, background: 0.5 }));
    const b = pool.add(makeImage({ rootMax: 2 }), { valueMax: 9 });
    const c = pool.add(makeImage({ flags: 0 }));
    pool.flush();
    expect([a, b, c].map((id) => recordAt(pool, id).valueMax)).toEqual([2, 9, Infinity]);
    expect(recordAt(pool, c).hasStats).toBe(false);
  });
});

describe("transforms (f64 composition)", () => {
  /** 4x4 row-major multiply. */
  const mul = (a: number[], b: number[]) =>
    Array.from({ length: 16 }, (_, k) => {
      const r = k >> 2;
      const c = k & 3;
      return a[r * 4]! * b[c]! + a[r * 4 + 1]! * b[4 + c]! + a[r * 4 + 2]! * b[8 + c]! + a[r * 4 + 3]! * b[12 + c]!;
    });
  /** Gauss-Jordan inverse of a 4x4 row-major matrix (independent of the pool's 3x3 path). */
  const inverse = (m: number[]) => {
    const a = m.map((v, k) => [v, k]).reduce<number[][]>((rows, [v, k]) => {
      (rows[k! >> 2] ??= []).push(v!);
      return rows;
    }, []);
    a.forEach((row, r) => row.push(...[0, 1, 2, 3].map((c) => (c === r ? 1 : 0))));
    for (let c = 0; c < 4; c++) {
      let p = c;
      for (let r = c + 1; r < 4; r++) if (Math.abs(a[r]![c]!) > Math.abs(a[p]![c]!)) p = r;
      [a[c], a[p]] = [a[p]!, a[c]!];
      const d = a[c]![c]!;
      a[c] = a[c]!.map((v) => v / d);
      for (let r = 0; r < 4; r++) {
        if (r === c) continue;
        const f = a[r]![c]!;
        a[r] = a[r]!.map((v, k) => v - f * a[c]![k]!);
      }
    }
    return a.flatMap((row) => row.slice(4));
  };
  const top3 = (m: number[]) => m.slice(0, 12);
  const near = (got: ArrayLike<number>, want: number[], tol: number) => {
    for (let k = 0; k < want.length; k++) {
      expect(Math.abs(got[k]! - want[k]!), `entry ${k}: ${got[k]} vs ${want[k]}`).toBeLessThanOrEqual(
        tol * Math.max(1, Math.abs(want[k]!)),
      );
    }
  };

  // File map: a sheared, non-uniform index -> world map with an offset (row-major 3x3).
  const matD = [0.02, 0.001, 0, -0.0005, 0.03, 0.002, 0.0003, 0, 0.025];
  const translation: [number, number, number] = [12.5, -3.25, 100.125];
  // Object -> world: rotate 30° about (1, 2, 3)/|..|, scale (2, 0.5, 3), translate (-7, 4.5, 1000).
  const objectToWorldRowMajor = (() => {
    const [x, y, z] = [1, 2, 3].map((v) => v / Math.sqrt(14)) as [number, number, number];
    const t = Math.PI / 6;
    const [c, s, C] = [Math.cos(t), Math.sin(t), 1 - Math.cos(t)];
    const R = [
      [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
      [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
      [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ];
    const S = [2, 0.5, 3];
    const T = [-7, 4.5, 1000];
    return [0, 1, 2].flatMap((r) => [...[0, 1, 2].map((k) => R[r]![k]! * S[k]!), T[r]!]).concat([0, 0, 0, 1]);
  })();
  const columnMajor = (rm: number[]) => Array.from({ length: 16 }, (_, k) => rm[(k & 3) * 4 + (k >> 2)]!);
  const indexToObject = [0, 1, 2].flatMap((r) => [matD[r * 3]!, matD[r * 3 + 1]!, matD[r * 3 + 2]!, translation[r]!]).concat([0, 0, 0, 1]);

  it("matches a 4x4 f64 reference within 1e-6 and rounds to f32 in the record", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 65536, maxGrids: 4 });
    const id = pool.add(makeImage({ matD, translation }));
    pool.flush();

    // Without a transform: the file map and its inverse.
    near(pool.info(id).indexToWorld, top3(indexToObject), 1e-12);
    near(pool.info(id).worldToIndex, top3(inverse(indexToObject)), 1e-6);

    pool.setTransform(id, new Float32Array(columnMajor(objectToWorldRowMajor)));
    const ref = mul(objectToWorldRowMajor.map(Math.fround), indexToObject); // the pool sees the f32 input
    const info = pool.info(id);
    near(info.indexToWorld, top3(ref), 1e-6);
    near(info.worldToIndex, top3(inverse(ref)), 1e-6);
    pool.flush();
    const rec = recordAt(pool, id);
    expect(rec.instanceTransform).toBe(true);
    expect(rec.indexToWorld).toEqual(Array.from(info.indexToWorld, Math.fround));
    expect(rec.worldToIndex).toEqual(Array.from(info.worldToIndex, Math.fround));
    // World bbox: the 8 corners of [min, max + 1] through indexToWorld.
    const corner = [-4, -4, -4];
    const p = [0, 1, 2].map((r) => ref[r * 4]! * corner[0]! + ref[r * 4 + 1]! * corner[1]! + ref[r * 4 + 2]! * corner[2]! + ref[r * 4 + 3]!);
    for (let a = 0; a < 3; a++) expect(info.worldBBox.min[a]!).toBeLessThanOrEqual(p[a]! + 1e-9);
  });

  it("setTransform rewrites only the record (flag b3 and words 4-27), never the image", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 65536, maxGrids: 4 });
    const id = pool.add(makeImage({ matD, translation }));
    pool.flush();
    const before = recordWords(pool, id);
    const version = pool.version;
    const revision = pool.revision;
    pool.setTransform(id, columnMajor(objectToWorldRowMajor));
    const ops = device.updateAndSubmit(pool);
    expect(writes(ops).map((o) => [o.offset, o.size])).toEqual([
      [32 + id * 160, 160],
      [0, 32],
    ]);
    const after = recordWords(pool, id);
    const changed = after.map((w, k) => (w !== before[k] ? k : -1)).filter((k) => k >= 0);
    expect(changed.every((k) => k === 3 || (k >= 4 && k <= 27))).toBe(true);
    expect(after[3]! ^ before[3]!).toBe(VDB_RECORD_FLAG.instanceTransform);
    expect([pool.version, pool.revision]).toEqual([version, revision + 1]);

    pool.setTransform(id, null);
    pool.flush();
    expect(recordWords(pool, id)).toEqual(before);
  });

  it("rejects non-affine, singular and malformed matrices", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 65536, maxGrids: 4 });
    const id = pool.add(makeImage());
    const I = columnMajor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    expect(code(() => pool.setTransform(id, I))).toBe("no error");
    expect(code(() => pool.setTransform(id, I.slice(0, 12)))).toBe("VDB_BAD_OPTION");
    expect(code(() => pool.setTransform(id, I.map((v, k) => (k === 3 ? 0.5 : v))))).toBe("VDB_BAD_OPTION");
    expect(code(() => pool.setTransform(id, I.map((v, k) => (k === 0 ? 0 : v))))).toBe("VDB_BAD_OPTION");
    expect(code(() => pool.setTransform(id, I.map((v, k) => (k === 13 ? Number.NaN : v))))).toBe("VDB_BAD_OPTION");
    expect(code(() => pool.add(makeImage(), { transform: [1, 2, 3] }))).toBe("VDB_BAD_OPTION");
  });
});

describe("revision vs version", () => {
  it("revision counts content changes; version counts buffer objects", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 4096, maxGrids: 4 });
    const a = pool.add(makeImage({ bytes: 1024 }));
    pool.setTransform(a, null);
    pool.setMajorant(a, { brickLog2: 3, dims: [1, 1, 1], data: new Float32Array([1]) });
    pool.replace(a, makeImage({ bytes: 1024, seed: 2 }));
    expect([pool.revision, pool.version]).toEqual([4, 0]);
    pool.flush();
    pool.flush(); // no changes: neither moves
    expect([pool.revision, pool.version]).toEqual([4, 0]);
    pool.add(makeImage({ bytes: 4000 }));
    pool.add(makeImage({ bytes: 4000 }));
    expect(pool.version).toBe(0); // staged growth doesn't bump version
    pool.flush();
    expect([pool.revision, pool.version]).toEqual([6, 1]); // two adds, one growth
    pool.remove(a);
    pool.flush();
    expect([pool.revision, pool.version]).toEqual([7, 1]);
  });
});

// ---------------------------------------------------------------------------
// The nine primitive fixtures through growth and compaction (CPU twin of the
// e2e pool readback; fixtures are git-ignored, `pnpm fixtures:bake`).

const dir = new URL("../../../../fixtures/primitives/", import.meta.url);
const NAMES = ["box_fog", "sphere_fog", "torus_fog"].flatMap((p) => ["float", "fp8", "fpn"].map((t) => `${p}_${t}`));
const present = NAMES.every((n) => existsSync(new URL(`${n}.nvdb`, dir)));

describe.skipIf(!present)("nine fixtures in a pool started at 4 KiB", () => {
  it("every image range is byte-identical to its file after growth and compaction", async () => {
    const images = NAMES.map((n) => {
      const buf = readFileSync(new URL(`${n}.nvdb`, dir));
      return NvdbFile.fromArrayBuffer(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)).gridImage(0);
    });
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 4096, maxGrids: 4 });
    const ids: number[] = [];
    for (let k = 0; k < 9; k++) {
      ids.push(pool.add(images[k]!));
      if (k % 3 === 2) pool.flush();
    }
    pool.remove(ids[4]!); // leave a hole and re-add into it: placed first-fit, no move
    ids[4] = pool.add(images[4]!);
    pool.flush();
    expect(pool.version).toBeGreaterThanOrEqual(2);
    expect(pool.gridCapacity).toBe(16);
    expect([...pool.types].sort()).toEqual(["float", "fp8", "fpn"]);
    for (let k = 0; k < 9; k++) {
      expect(sameBytes(imageInPool(pool, ids[k]!), bytesOf(images[k]!)), NAMES[k]).toBe(true);
      expect(recordAt(pool, ids[k]!).imageOffset).toBe(pool.info(ids[k]!).imageOffset);
    }
    expect(header(pool).gridCount).toBe(9);
  });
});
