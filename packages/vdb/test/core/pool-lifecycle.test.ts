/**
 * VDBGridPool lifecycle: requests that can never fit, allocation failure during
 * growth (error scopes + rollback), and when retired buffers may be destroyed.
 * Uses the recording fake device (fake-gpu.ts).
 */
import { beforeEach, describe, expect, it } from "vitest";
import { VDBGridPool, VDB_POOL_RECORD_WORDS, decodeHeader, decodeRecord } from "../../src/core/internal.js";
import { FakeBuffer, FakeDevice, settle, type FakeOp } from "./fake-gpu.js";
import { makeImage } from "./synthetic-image.js";

function code(fn: () => unknown): string {
  try {
    fn();
  } catch (e) {
    return (e as { code?: string }).code ?? `untyped: ${(e as Error).message}`;
  }
  return "no error";
}

async function rejection(p: Promise<unknown>): Promise<Error & { code?: string }> {
  try {
    await p;
  } catch (e) {
    return e as Error & { code?: string };
  }
  throw new Error("expected a rejection");
}

const fake = (pool: VDBGridPool) => pool.buffer as unknown as FakeBuffer;
const recordIn = (buf: FakeBuffer, id: number) => decodeRecord(buf.words(), 8 + id * VDB_POOL_RECORD_WORDS);
const bytesOf = (a: ArrayBufferView) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
function same(a: Uint8Array, b: Uint8Array): boolean {
  return a.length === b.length && Buffer.from(a).equals(Buffer.from(b));
}
function imageIn(buf: FakeBuffer, pool: VDBGridPool, id: number): Uint8Array {
  const info = pool.info(id);
  return buf.data.subarray(info.imageOffset!, info.imageOffset! + info.imageBytes);
}
const kinds = (ops: FakeOp[], kind: FakeOp["kind"]) => ops.filter((o) => o.kind === kind);

/** An encoder whose submit is up to the test. */
function encoderFor(device: FakeDevice) {
  const e = device.createCommandEncoder();
  return { gpu: e as unknown as GPUCommandEncoder, finish: () => e.finish() };
}

// With maxGrids 4 the table ends at 672; with 8, at 1312.
const DATA_START_4 = 672;
const DATA_START_8 = 1312;

let device: FakeDevice;
beforeEach(() => {
  device = new FakeDevice();
});

describe("requests that can never fit throw at staging", () => {
  it("probe: an oversized add no longer blocks later updates or in-place replaces", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 8192, maxGrids: 4, maxBytes: 64 << 10 });
    const a = pool.add(makeImage({ bytes: 1024, seed: 1 }));
    pool.flush();
    const revision = pool.revision;
    expect(code(() => pool.add(makeImage({ bytes: 100 << 10, seed: 2 })))).toBe("VDB_POOL_LIMIT");
    expect([pool.revision, pool.gridCount]).toEqual([revision, 1]); // unchanged

    const a2 = makeImage({ bytes: 1024, seed: 3 });
    pool.replace(a, a2);
    expect(code(() => pool.flush())).toBe("no error");
    expect(same(imageIn(fake(pool), pool, a), bytesOf(a2))).toBe(true);
    expect(pool.info(a).resident).toBe(true);
  });

  it("counts the table size at the grid count the request needs", async () => {
    // 4 grids: data starts at 672; the 5th id needs an 8-grid table (data at 1312).
    const limit = DATA_START_8 + 4096 - 32;
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 8192, maxGrids: 4, maxBytes: limit });
    for (let k = 0; k < 4; k++) pool.add(makeImage({ bytes: 1024, seed: k }), { capacity: 0 });
    // A 4096-byte image would fit beside a 4-grid table (672 + 4096 <= limit) but not an 8-grid one.
    expect(DATA_START_4 + 4096).toBeLessThanOrEqual(limit);
    expect(code(() => pool.add(makeImage({ bytes: 4096, seed: 9 })))).toBe("VDB_POOL_LIMIT");
    expect(pool.gridCount).toBe(4);
  });

  it("replace and setMajorant check too, and leave the pool unchanged", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 4096, maxGrids: 4, maxBytes: 8192 });
    const img = makeImage({ bytes: 1024, seed: 1 });
    const a = pool.add(img);
    pool.flush();
    const before = pool.info(a);
    const revision = pool.revision;
    expect(code(() => pool.replace(a, makeImage({ bytes: 8192, seed: 2 })))).toBe("VDB_POOL_LIMIT");
    // The image and its majorant have to fit together: 672 + 1024 + 6656 > 8192.
    const cells = 6656 / 4;
    const tooBig = { brickLog2: 3, dims: [cells, 1, 1] as [number, number, number], data: new Float32Array(cells) };
    expect(code(() => pool.setMajorant(a, tooBig))).toBe("VDB_POOL_LIMIT");
    expect(code(() => pool.add(makeImage({ bytes: 1024, seed: 4 }), { majorant: tooBig }))).toBe("VDB_POOL_LIMIT");
    expect(pool.revision).toBe(revision);
    expect(pool.info(a)).toEqual(before);
    expect(code(() => pool.flush())).toBe("no error");
    expect(same(imageIn(fake(pool), pool, a), bytesOf(img))).toBe(true);
  });

  it("a relayout failure still applies independent in-place writes, then throws", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 4096, maxGrids: 4, maxBytes: 8192 });
    const a = pool.add(makeImage({ bytes: 1024, seed: 1 }));
    pool.flush();
    const buffer = fake(pool);
    // Each fits alone; together they need 672 + 1024 + 6016 + 2016 > 8192.
    const b = pool.add(makeImage({ bytes: 6000, seed: 2 }));
    const c = pool.add(makeImage({ bytes: 2000, seed: 3 }));
    const a2 = makeImage({ bytes: 1024, seed: 4 });
    pool.replace(a, a2);
    const mark = device.log.length;
    expect(code(() => pool.flush())).toBe("VDB_POOL_LIMIT");
    const ops = device.since(mark);
    expect(kinds(ops, "create")).toEqual([]);
    expect(kinds(ops, "copy")).toEqual([]);
    expect([pool.buffer, pool.version]).toEqual([buffer, 0]);
    // The in-place replace reached the GPU, with its record.
    expect(same(imageIn(buffer, pool, a), bytesOf(a2))).toBe(true);
    expect(pool.info(a).resident).toBe(true);
    // c was placed at staging (it fit the current buffer), so it went up too; b reads as dead until it fits.
    expect(pool.info(c).resident).toBe(true);
    expect(recordIn(buffer, c)).toMatchObject({ live: true, imageOffset: pool.info(c).imageOffset });
    expect(recordIn(buffer, b).live).toBe(false);
    expect(pool.info(b)).toMatchObject({ resident: false, imageOffset: null });
    expect(decodeHeader(buffer.words()).gridCount).toBe(3);

    pool.remove(c);
    pool.flush(); // b now fits after a growth
    expect(pool.version).toBe(1);
    expect(recordIn(fake(pool), b)).toMatchObject({ live: true, imageOffset: pool.info(b).imageOffset });
  });
});

describe("retired buffers outlive the work that reads them", () => {
  it("probe: update(e1) grows, update(e2), submit([e1, e2]) reads no destroyed buffer", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 4096, maxGrids: 4 });
    const imgA = makeImage({ bytes: 2048, seed: 1 });
    const a = pool.add(imgA);
    pool.flush();
    const old = fake(pool);
    const imgB = makeImage({ bytes: 3000, seed: 2 });
    const b = pool.add(imgB);
    const e1 = encoderFor(device);
    pool.update(e1.gpu); // grows: copies a out of `old`
    pool.setTransform(a, null);
    const e2 = encoderFor(device);
    pool.update(e2.gpu);
    expect(old.destroyed).toBe(false);
    expect(() => device.queue.submit([e1.finish(), e2.finish()])).not.toThrow();
    expect(same(imageIn(fake(pool), pool, a), bytesOf(imgA))).toBe(true);
    expect(same(imageIn(fake(pool), pool, b), bytesOf(imgB))).toBe(true);
  });

  it("destroys a retired buffer only after onSubmittedWorkDone, requested two updates on", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 4096, maxGrids: 4 });
    pool.add(makeImage({ bytes: 2048, seed: 1 }));
    pool.flush();
    const old = fake(pool);
    pool.add(makeImage({ bytes: 3000, seed: 2 }));
    pool.flush(); // growth: `old` retired once the allocation is confirmed
    await settle();
    expect(pool.retiredBytes).toBe(4096);
    pool.flush(); // one update later: the host may not have submitted the growth's encoder yet
    await settle();
    expect(device.workDoneRequests).toBe(0);
    await device.completeWork();
    expect(old.destroyed).toBe(false);

    pool.flush(); // two updates later: ask the queue
    expect(device.workDoneRequests).toBe(1);
    await settle();
    expect(old.destroyed).toBe(false); // the queue hasn't finished yet
    expect(pool.retiredBytes).toBe(4096);
    await device.completeWork();
    expect(old.destroyed).toBe(true);
    expect(old.destroyCalls).toBe(1);
    expect(pool.retiredBytes).toBe(0);
    pool.flush();
    expect(device.workDoneRequests).toBe(1); // nothing left to retire
  });
});

describe("allocation failure during growth", () => {
  it("rolls back to the old buffer: every resident grid intact and readable, version moves on", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 4096, maxGrids: 4 });
    const imgA = makeImage({ bytes: 1024, seed: 1 });
    const imgB = makeImage({ bytes: 2048, seed: 2 });
    const a = pool.add(imgA);
    const b = pool.add(imgB);
    pool.flush();
    const old = fake(pool);
    const recordsBefore = [recordIn(old, a), recordIn(old, b)];

    device.failCreate = () => true;
    const imgC = makeImage({ bytes: 3000, seed: 3 });
    const c = pool.add(imgC);
    pool.flush();
    expect(device.errorScopes).toBe(0); // pushed and popped
    expect(device.uncaptured.filter((m) => m.includes("out of memory"))).toEqual([]); // the pool's scope caught it
    const failed = fake(pool);
    expect(failed.invalid).toBe(true);
    expect(pool.version).toBe(1); // optimistic switch
    expect(old.destroyed).toBe(false); // kept until the scope resolves

    const err = await rejection(pool.settled());
    expect(err.code).toBe("VDB_POOL_LIMIT");
    expect(err.message).toMatch(/out of memory/);
    expect(pool.lastError).toBe(err);
    // Rolled back to the old buffer, as a new version so cached bind groups are rebuilt.
    expect(pool.buffer).toBe(old);
    expect(pool.version).toBe(2);
    expect(failed.destroyed).toBe(true);
    expect(old.destroyed).toBe(false);
    expect(pool.retiredBytes).toBe(0);
    expect(same(imageIn(old, pool, a), bytesOf(imgA))).toBe(true);
    expect(same(imageIn(old, pool, b), bytesOf(imgB))).toBe(true);
    expect([recordIn(old, a), recordIn(old, b)]).toEqual(recordsBefore);
    // c is staged again, not on the GPU.
    expect(pool.info(c)).toMatchObject({ resident: false, imageOffset: null });
    expect(recordIn(old, c).live).toBe(false);
    expect(decodeHeader(old.words())).toMatchObject({ gridCount: 3, gridCapacity: 4 });

    // The next update retries the growth; this time it succeeds.
    pool.flush();
    await pool.settled();
    expect(pool.lastError).toBeNull();
    expect(pool.version).toBe(3);
    for (const [id, img] of [[a, imgA], [b, imgB], [c, imgC]] as const) {
      expect(same(imageIn(fake(pool), pool, id), bytesOf(img))).toBe(true);
      expect(recordIn(fake(pool), id)).toMatchObject({ live: true, imageOffset: pool.info(id).imageOffset });
    }
  });

  it("restores writes made to the failed buffer before the scope resolved", async () => {
    const p = await VDBGridPool.create(device.asGPU(), { initialBytes: 4096, maxGrids: 4 });
    const imgA = makeImage({ bytes: 1024, seed: 1 });
    const a = p.add(imgA);
    p.flush();
    const old = fake(p);

    const hold = device.holdScopes();
    device.failCreate = () => true;
    const c = p.add(makeImage({ bytes: 3000, seed: 3 }));
    p.flush(); // growth into a buffer that will turn out invalid
    const a2 = makeImage({ bytes: 1024, seed: 4 });
    p.replace(a, a2); // written into the (invalid) new buffer
    const d = p.add(makeImage({ bytes: 900, seed: 5 }));
    p.flush();
    p.flush();
    await device.completeWork();
    expect(old.destroyed).toBe(false); // unconfirmed growth: the old buffer stays alive
    expect(p.retiredBytes).toBe(4096);

    await hold();
    expect(p.lastError?.code).toBe("VDB_POOL_LIMIT");
    expect(p.buffer).toBe(old);
    // a's in-place replace and d went to the failed buffer: staged again; c too.
    for (const id of [a, c, d]) expect(p.info(id).resident).toBe(false);
    expect(recordIn(old, a).live).toBe(false); // old bytes no longer match a's header
    p.flush();
    await p.settled();
    expect(same(imageIn(fake(p), p, a), bytesOf(a2))).toBe(true);
    for (const id of [a, c, d]) expect(p.info(id).resident).toBe(true);
  });

  it("a failure voids later chained growths; an earlier confirmed growth is kept", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 4096, maxGrids: 4 });
    const imgA = makeImage({ bytes: 2048, seed: 1 });
    const a = pool.add(imgA);
    pool.flush();
    const v0 = fake(pool);

    // Growth 1 succeeds, growth 2 fails, growth 3 chains off the failed one: all before any scope resolves.
    const hold = device.holdScopes();
    pool.add(makeImage({ bytes: 3000, seed: 2 }));
    pool.flush();
    const v1 = fake(pool);
    device.failCreate = () => true;
    pool.add(makeImage({ bytes: 6000, seed: 3 }));
    pool.flush();
    const v2 = fake(pool);
    pool.add(makeImage({ bytes: 16000, seed: 4 }));
    pool.flush();
    const v3 = fake(pool);
    expect(pool.version).toBe(3);
    const settled = rejection(pool.settled()); // observed now: it rejects while `hold` releases
    await hold();
    expect((await settled).code).toBe("VDB_POOL_LIMIT");

    expect(pool.buffer).toBe(v1);
    expect(pool.version).toBe(4);
    expect([v2.destroyed, v3.destroyed, v1.destroyed]).toEqual([true, true, false]);
    expect(same(imageIn(v1, pool, a), bytesOf(imgA))).toBe(true);
    expect(pool.info(1).resident).toBe(true); // written to v1 before growth 2
    expect(pool.info(2).resident).toBe(false);
    expect(pool.info(3).resident).toBe(false);
    // v0 was retired by the confirmed growth 1 and is destroyed after the queue drains.
    expect(v0.destroyed).toBe(false);
    pool.flush();
    pool.flush();
    await device.completeWork();
    expect(v0.destroyed).toBe(true);
  });
});

describe("dispose", () => {
  it("frees every buffer the pool created, including unconfirmed and retired ones, exactly once", async () => {
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 4096, maxGrids: 4 });
    pool.add(makeImage({ bytes: 2048, seed: 1 }));
    pool.flush();
    pool.add(makeImage({ bytes: 3000, seed: 2 }));
    pool.flush(); // growth 1, confirmed
    await settle();
    pool.flush();
    pool.flush(); // asks the queue about the retired buffer
    const hold = device.holdScopes();
    pool.add(makeImage({ bytes: 9000, seed: 3 }));
    pool.flush(); // growth 2, unconfirmed
    const created = device.buffers.slice();
    expect(created).toHaveLength(3);
    expect(created.some((b) => b.destroyed)).toBe(false);

    pool.dispose();
    for (const b of created) expect(b.destroyCalls).toBe(1);
    await hold();
    await device.completeWork(); // late callbacks don't destroy again
    for (const b of created) expect(b.destroyCalls).toBe(1);
    expect(code(() => pool.flush())).toBe("VDB_DISPOSED");
  });
});
