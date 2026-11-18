import { describe, expect, it, vi } from "vitest";
import { VDBError } from "../../src/core/errors.js";
import { VDBSequence } from "../../src/three/sequence.js";
import type { VDBFrameLoader, VDBSequenceTarget } from "../../src/three/sequence.js";

/**
 * Node-safe `VDBSequence` tests: the prefetch ring + frame-time scheduler +
 * stall policy, exercised with a mocked `loader`, a fake clock, and a stub
 * pool whose `replace(id, image)` records what it got — no browser, no GPU.
 * The end-to-end GPU proof (in-place `pool.replace` changing pixels, no pool
 * version change) is `e2e/demo-05.spec.ts`.
 */

/** A tagged stand-in for a decoded frame image: word 0 is the frame index. */
function grid(tag: number): Uint32Array {
  return new Uint32Array([tag]);
}

/** A stub `{ pool, id }` target recording the frames replaced into it. */
function makeTarget(id = 7): VDBSequenceTarget & { calls: number[]; ids: number[]; images: Uint32Array[] } {
  const calls: number[] = [];
  const ids: number[] = [];
  const images: Uint32Array[] = [];
  return {
    calls,
    ids,
    images,
    id,
    pool: {
      replace(gridId: number, image: Uint32Array): void {
        ids.push(gridId);
        images.push(image);
        calls.push(image[0]!);
      },
    },
  };
}

/** Loader that resolves each frame immediately to `grid(i)`. */
function immediateLoader(spy?: (i: number) => void): VDBFrameLoader {
  return (_url: string, i: number): Promise<Uint32Array> => {
    spy?.(i);
    return Promise.resolve(grid(i));
  };
}

/** Loader whose per-frame resolution you drive by hand (for stall tests). */
function deferredLoader(): { loader: VDBFrameLoader; resolve: (i: number) => void; reject: (i: number, e: unknown) => void } {
  const resolvers = new Map<number, (g: Uint32Array) => void>();
  const rejectors = new Map<number, (e: unknown) => void>();
  const loader: VDBFrameLoader = (_url, i) =>
    new Promise<Uint32Array>((res, rej) => {
      resolvers.set(i, res);
      rejectors.set(i, rej);
    });
  return {
    loader,
    resolve: (i) => resolvers.get(i)?.(grid(i)),
    reject: (i, e) => rejectors.get(i)?.(e),
  };
}

/** Let queued microtasks/timeouts run so decode `.then` callbacks populate the ring. */
const tick = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

// A source url list for N frames.
const urls = (n: number): string[] => Array.from({ length: n }, (_, i) => `frame-${i}.nvdb`);

describe("VDBSequence — construction", () => {
  it("defaults frameCount to urls.length for an array", () => {
    const seq = new VDBSequence({ urls: urls(5), loader: immediateLoader() });
    expect(seq.frameCount).toBe(5);
    expect(seq.fps).toBe(24);
    expect(seq.prefetch).toBe(3);
    expect(seq.loop).toBe(true);
  });

  it("requires an explicit frameCount when urls is a function", () => {
    expect(() => new VDBSequence({ urls: (i) => `f${i}`, loader: immediateLoader() })).toThrow(/frameCount/);
    const seq = new VDBSequence({ urls: (i) => `f${i}`, frameCount: 8, loader: immediateLoader() });
    expect(seq.frameCount).toBe(8);
  });

  it("rejects a non-positive fps", () => {
    expect(() => new VDBSequence({ urls: urls(3), fps: 0, loader: immediateLoader() })).toThrow(/fps/);
  });
});

describe("VDBSequence — prefetch ring", () => {
  it("preload decodes exactly the first N frames", async () => {
    const seen: number[] = [];
    const seq = new VDBSequence({ urls: urls(10), prefetch: 2, loader: immediateLoader((i) => seen.push(i)) });
    await seq.preload(3);
    expect(seen.sort((a, b) => a - b)).toEqual([0, 1, 2]);
  });

  it("start replaces with frame 0 and reports decodedAhead within the prefetch window", async () => {
    const seq = new VDBSequence({ urls: urls(10), prefetch: 3, loader: immediateLoader() });
    await seq.preload(4); // frames 0..3
    const target = makeTarget();
    seq.start(target, 0);
    expect(target.calls).toEqual([0]);
    expect(seq.stats.frame).toBe(0);
    expect(seq.stats.swaps).toBe(1);
    // frames 1,2,3 sit ahead, ready.
    expect(seq.stats.decodedAhead).toBe(3);
  });

  it("evicts decoded frames that fall outside the forward window", async () => {
    let decodes = 0;
    const seq = new VDBSequence({ urls: urls(6), prefetch: 1, loop: false, loader: immediateLoader(() => decodes++) });
    // Warm 0..3, but prefetch is only 1, so start(0) keeps {0,1} and evicts 2,3.
    await seq.preload(4);
    expect(decodes).toBe(4);
    const target = makeTarget();
    seq.start(target, 0);
    await tick();
    // decodedAhead counts only frame 1 (window is {0,1}).
    expect(seq.stats.decodedAhead).toBe(1);
  });
});

describe("VDBSequence — scheduler", () => {
  it("advances one frame per frame-duration and fires onFrame per new frame", async () => {
    const shown: number[] = [];
    // fps 10 => 100ms/frame exactly (no FP drift).
    const seq = new VDBSequence({
      urls: urls(4),
      fps: 10,
      prefetch: 4,
      loop: false,
      loader: immediateLoader(),
      onFrame: (s) => shown.push(s.frame),
    });
    await seq.preload(4);
    const target = makeTarget();
    seq.start(target, 0);
    seq.update(100);
    seq.update(200);
    seq.update(300);
    expect(target.calls).toEqual([0, 1, 2, 3]);
    expect(shown).toEqual([0, 1, 2, 3]);
    expect(seq.stats.stalls).toBe(0);
  });

  it("does not replace again within the same frame's time slice", async () => {
    const seq = new VDBSequence({ urls: urls(4), fps: 10, prefetch: 4, loader: immediateLoader() });
    await seq.preload(4);
    const target = makeTarget();
    seq.start(target, 0);
    seq.update(30);
    seq.update(60);
    seq.update(99);
    expect(target.calls).toEqual([0]);
    expect(seq.stats.swaps).toBe(1);
  });

  it("clamps at the last frame when loop is false", async () => {
    const seq = new VDBSequence({ urls: urls(4), fps: 10, prefetch: 4, loop: false, loader: immediateLoader() });
    await seq.preload(4);
    const target = makeTarget();
    seq.start(target, 0);
    seq.update(10_000); // way past the end
    expect(seq.stats.frame).toBe(3);
  });

  it("wraps around when loop is true", async () => {
    const seq = new VDBSequence({ urls: urls(4), fps: 10, prefetch: 4, loop: true, loader: immediateLoader() });
    await seq.preload(4);
    const target = makeTarget();
    seq.start(target, 0);
    seq.update(600); // 6 frames elapsed, wraps to frame 2
    expect(seq.stats.frame).toBe(2);
  });

  it("resume continues from the paused frame without resetting", async () => {
    const seq = new VDBSequence({ urls: urls(6), fps: 10, prefetch: 6, loop: false, loader: immediateLoader() });
    await seq.preload(6);
    const target = makeTarget();
    seq.start(target, 0);
    seq.update(200); // frame 2
    expect(seq.stats.frame).toBe(2);
    seq.stop();
    seq.update(999); // ignored while stopped
    expect(seq.stats.frame).toBe(2);
    seq.resume(1000); // re-anchor at frame 2
    expect(seq.isPlaying).toBe(true);
    seq.update(1100); // one frame later -> frame 3
    expect(seq.stats.frame).toBe(3);
    // Stats were not reset by resume. Frame 1 was dropped (0 -> 2 jump), so
    // three distinct frames were bound: 0, 2, 3.
    expect(seq.stats.swaps).toBe(3);
  });

  it("ignores update() before start() / after stop()", async () => {
    const seq = new VDBSequence({ urls: urls(4), fps: 10, prefetch: 4, loader: immediateLoader() });
    await seq.preload(4);
    expect(seq.update(100)).toBe(-1); // not playing yet
    const target = makeTarget();
    seq.start(target, 0);
    seq.update(100);
    expect(seq.stats.frame).toBe(1);
    seq.stop();
    seq.update(300); // ignored
    expect(seq.stats.frame).toBe(1);
    expect(seq.isPlaying).toBe(false);
  });
});

describe("VDBSequence — stall policy (never block)", () => {
  it("holds the last frame and counts a stall when the due frame is not decoded", async () => {
    const { loader, resolve } = deferredLoader();
    const seq = new VDBSequence({ urls: urls(4), fps: 10, prefetch: 1, loop: false, loader });

    // Warm frames 0 and 1 only.
    const pl = seq.preload(2);
    resolve(0);
    resolve(1);
    await pl;

    const target = makeTarget();
    seq.start(target, 0); // frame 0
    seq.update(100); // frame 1 (decoded)
    expect(seq.stats.frame).toBe(1);
    expect(target.calls).toEqual([0, 1]);

    // Frame 2 is due but not decoded -> stall, hold frame 1.
    seq.update(200);
    expect(seq.stats.frame).toBe(1);
    expect(seq.stats.stalls).toBeGreaterThanOrEqual(1);
    expect(target.calls).toEqual([0, 1]);

    // Once frame 2 arrives, the next update catches up to it.
    resolve(2);
    await tick();
    seq.update(200);
    expect(seq.stats.frame).toBe(2);
    expect(target.calls).toEqual([0, 1, 2]);
  });

  it("surfaces a decode failure via onError and treats that frame as a stall", async () => {
    const { loader, resolve, reject } = deferredLoader();
    const errors: number[] = [];
    const seq = new VDBSequence({
      urls: urls(3),
      fps: 10,
      prefetch: 1,
      loop: false,
      loader,
      onError: (_e, i) => errors.push(i),
    });
    const pl = seq.preload(1);
    resolve(0);
    await pl;
    const target = makeTarget();
    seq.start(target, 0);
    // frame 1 fails to decode.
    reject(1, new Error("boom"));
    await tick();
    seq.update(100);
    expect(errors).toContain(1);
    expect(seq.stats.frame).toBe(0); // held
    expect(seq.stats.stalls).toBeGreaterThanOrEqual(1);
  });
});

describe("VDBSequence — seek", () => {
  it("jumps to a frame and re-anchors the clock there", async () => {
    const seq = new VDBSequence({ urls: urls(8), fps: 10, prefetch: 8, loader: immediateLoader() });
    await seq.preload(8);
    const target = makeTarget();
    seq.start(target, 0);
    seq.seek(5, 1000);
    expect(seq.stats.frame).toBe(5);
    // From the seek anchor, one frame-duration later advances to 6.
    seq.update(1100);
    expect(seq.stats.frame).toBe(6);
  });

  it("wraps a seek index into range", async () => {
    const seq = new VDBSequence({ urls: urls(4), fps: 10, prefetch: 4, loader: immediateLoader() });
    await seq.preload(4);
    const target = makeTarget();
    seq.start(target, 0);
    seq.seek(5, 0); // wraps to 1
    expect(seq.stats.frame).toBe(1);
  });
});

describe("VDBSequence — stats", () => {
  it("reports replaceMs as a finite number and counts swaps", async () => {
    const seq = new VDBSequence({ urls: urls(4), fps: 10, prefetch: 4, loader: immediateLoader() });
    await seq.preload(4);
    const target = makeTarget();
    seq.start(target, 0);
    seq.update(100);
    seq.update(200);
    expect(Number.isFinite(seq.stats.replaceMs)).toBe(true);
    expect(seq.stats.replaceMs).toBeGreaterThanOrEqual(0);
    expect(seq.stats.swaps).toBe(3); // frames 0,1,2
  });

  it("dispose clears the ring", async () => {
    const seq = new VDBSequence({ urls: urls(4), prefetch: 4, loader: immediateLoader() });
    await seq.preload(4);
    seq.dispose();
    expect(seq.stats.decodedAhead).toBe(0);
    expect(seq.isPlaying).toBe(false);
  });

  it("uses a real, monotonically increasing clock for replaceMs when now() advances", async () => {
    let t = 0;
    const seq = new VDBSequence({ urls: urls(2), fps: 10, prefetch: 2, loader: immediateLoader(), now: () => (t += 1) });
    await seq.preload(2);
    const target = makeTarget();
    // now() ticks on every read, so the two reads around pool.replace differ.
    seq.start(target, 0);
    expect(seq.stats.replaceMs).toBeGreaterThan(0);
  });
});

describe("VDBSequence — default loader wiring", () => {
  it("passes the resolved url and index to the loader", async () => {
    const loader = vi.fn(immediateLoader());
    const seq = new VDBSequence({ urls: (i) => `custom/${i}.nvdb`, frameCount: 3, prefetch: 3, loader });
    await seq.preload(2);
    expect(loader).toHaveBeenCalledWith("custom/0.nvdb", 0);
    expect(loader).toHaveBeenCalledWith("custom/1.nvdb", 1);
  });
});

describe("VDBSequence — the pool target", () => {
  it("replaces the target id with the decoded image itself (no copy, never written)", async () => {
    const frames = [grid(0), grid(1)];
    const seq = new VDBSequence({ urls: urls(2), fps: 10, prefetch: 2, loader: (_u, i) => Promise.resolve(frames[i]!) });
    await seq.preload(2);
    const target = makeTarget(3);
    seq.start(target, 0);
    seq.update(100);
    expect(target.ids).toEqual([3, 3]);
    expect(target.images[0]).toBe(frames[0]); // held by reference: a recycling loader must wait for pool.settled()
    expect(target.images[1]).toBe(frames[1]);
    expect(Array.from(frames[0]!)).toEqual([0]);
  });

  it("rejects a target that is not { pool, id }", () => {
    const seq = new VDBSequence({ urls: urls(2), loader: immediateLoader() });
    expect(() => seq.start({ replace() {} } as never, 0)).toThrow(/\{ pool, id \}/);
    expect(() => seq.start({ pool: { replace() {} }, id: -1 }, 0)).toThrow(/grid id/);
    expect(() => seq.start({ pool: {} as never, id: 0 }, 0)).toThrow(/replace/);
  });
});

describe("VDBSequence — failed frames and disposal (K2 review)", () => {
  it("does not retry a failed frame on every update; seek and start retry it", async () => {
    const calls = new Map<number, number>();
    let failFrame1 = true;
    const errors: number[] = [];
    const loader: VDBFrameLoader = (_url, i) => {
      calls.set(i, (calls.get(i) ?? 0) + 1);
      return i === 1 && failFrame1 ? Promise.reject(new Error("boom")) : Promise.resolve(grid(i));
    };
    const seq = new VDBSequence({ urls: urls(4), fps: 10, prefetch: 2, loader, onError: (_e, i) => errors.push(i) });
    await seq.preload(3);
    await tick();
    expect(errors).toEqual([1]);
    expect(seq.stats.failed).toBe(1);
    const target = makeTarget();
    seq.start(target, 0); // start clears the failed set: frame 1 is retried once
    await tick();
    expect(calls.get(1)).toBe(2);
    expect(errors).toEqual([1, 1]);
    // Many updates while the clock points at frame 1: no further loader calls, no onError storm.
    for (let t = 100; t < 200; t += 5) seq.update(t);
    await tick();
    expect(calls.get(1)).toBe(2);
    expect(errors).toEqual([1, 1]);
    expect(seq.stats.frame).toBe(0); // held
    // The clock moves past it: playback continues with frame 2.
    seq.update(200);
    expect(seq.stats.frame).toBe(2);
    // seek retries (and this time the frame decodes).
    failFrame1 = false;
    seq.seek(1, 300);
    await tick();
    expect(calls.get(1)).toBe(3);
    expect(seq.stats.failed).toBe(0);
    seq.update(300);
    expect(seq.stats.frame).toBe(1);
  });

  it("a loader that throws synchronously is a failed frame too", async () => {
    const errors: number[] = [];
    const seq = new VDBSequence({
      urls: urls(2),
      prefetch: 1,
      loader: (_u, i) => {
        if (i === 1) throw new Error("sync");
        return Promise.resolve(grid(i));
      },
      onError: (_e, i) => errors.push(i),
    });
    await seq.preload(2);
    expect(errors).toEqual([1]);
    expect(seq.stats.failed).toBe(1);
  });

  it("drops decodes that settle after dispose(), and refuses start/seek/resume/preload", async () => {
    const { loader, resolve, reject } = deferredLoader();
    const errors: number[] = [];
    const seq = new VDBSequence({ urls: urls(4), fps: 10, prefetch: 2, loader, onError: (_e, i) => errors.push(i) });
    const target = makeTarget();
    seq.start(target, 0); // frames 0..2 in flight
    seq.dispose();
    resolve(0);
    resolve(1);
    reject(2, new Error("late"));
    await tick();
    expect(seq.stats.decodedAhead).toBe(0);
    expect(seq.stats.failed).toBe(0);
    expect(errors).toEqual([]); // the late failure is dropped, not reported
    expect(target.calls).toEqual([]);
    expect(seq.update(500)).toBe(-1);
    const disposed = (fn: () => unknown) => {
      try {
        fn();
      } catch (e) {
        return e instanceof VDBError ? e.code : String(e);
      }
      return undefined;
    };
    expect(disposed(() => seq.start(target, 0))).toBe("VDB_DISPOSED");
    expect(disposed(() => seq.seek(1, 0))).toBe("VDB_DISPOSED");
    expect(disposed(() => seq.resume(0))).toBe("VDB_DISPOSED");
    await expect(seq.preload(1)).rejects.toMatchObject({ code: "VDB_DISPOSED" });
    seq.dispose(); // idempotent
  });
});
