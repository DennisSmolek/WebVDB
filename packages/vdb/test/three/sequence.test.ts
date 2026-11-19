import { describe, expect, it, vi } from "vitest";
import { VDBError } from "../../src/core/errors.js";
import { VDBSequence } from "../../src/three/sequence.js";
import type { VDBFrameLoader, VDBSequenceLoopMode, VDBSequenceTarget } from "../../src/three/sequence.js";

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
    expect(seq.loopMode).toBe("repeat");
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
    const seq = new VDBSequence({ urls: urls(6), prefetch: 1, loopMode: "once", loader: immediateLoader(() => decodes++) });
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
      loopMode: "once",
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

  it("holds the last frame with loopMode once", async () => {
    const seq = new VDBSequence({ urls: urls(4), fps: 10, prefetch: 4, loopMode: "once", loader: immediateLoader() });
    await seq.preload(4);
    const target = makeTarget();
    seq.start(target, 0);
    seq.update(10_000); // way past the end
    expect(seq.stats.frame).toBe(3);
  });

  it("wraps around with loopMode repeat", async () => {
    const seq = new VDBSequence({ urls: urls(4), fps: 10, prefetch: 4, loopMode: "repeat", loader: immediateLoader() });
    await seq.preload(4);
    const target = makeTarget();
    seq.start(target, 0);
    seq.update(600); // 6 frames elapsed, wraps to frame 2
    expect(seq.stats.frame).toBe(2);
  });

  it("resume continues from the paused frame without resetting", async () => {
    const seq = new VDBSequence({ urls: urls(6), fps: 10, prefetch: 6, loopMode: "once", loader: immediateLoader() });
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
    const seq = new VDBSequence({ urls: urls(4), fps: 10, prefetch: 1, loopMode: "once", loader });

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
      loopMode: "once",
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

describe("VDBSequence — frame range and loop mode", () => {
  /** A started sequence at fps 10 (100 ms/frame); returns the frame shown at each of `steps` consecutive frame times. */
  async function play(opts: { n: number; range?: [number, number]; loopMode?: VDBSequenceLoopMode }, steps: number) {
    const seq = new VDBSequence({
      urls: urls(opts.n),
      fps: 10,
      prefetch: 2,
      ...(opts.range ? { range: opts.range } : {}),
      ...(opts.loopMode ? { loopMode: opts.loopMode } : {}),
      loader: immediateLoader(),
    });
    await seq.preload();
    const target = makeTarget();
    seq.start(target, 0);
    const shown = [seq.stats.frame];
    for (let k = 1; k < steps; k++) {
      await tick(); // let the ring fill so no frame is dropped to a stall
      shown.push(seq.update(k * 100));
    }
    return { seq, target, shown };
  }

  it("defaults to every frame, repeating (unchanged behaviour)", async () => {
    const { seq, shown } = await play({ n: 4 }, 9);
    expect(seq.range).toEqual([0, 3]);
    expect(seq.loopMode).toBe("repeat");
    expect(shown).toEqual([0, 1, 2, 3, 0, 1, 2, 3, 0]);
  });

  it("plays only the range and wraps inside it", async () => {
    const { seq, target, shown } = await play({ n: 10, range: [3, 5] }, 8);
    expect(seq.range).toEqual([3, 5]);
    expect(shown).toEqual([3, 4, 5, 3, 4, 5, 3, 4]);
    expect(Math.min(...target.calls)).toBe(3);
    expect(Math.max(...target.calls)).toBe(5);
  });

  it("pingpong goes forward then backward without repeating the end frames", async () => {
    const { shown } = await play({ n: 10, range: [2, 5], loopMode: "pingpong" }, 15);
    expect(shown).toEqual([2, 3, 4, 5, 4, 3, 2, 3, 4, 5, 4, 3, 2, 3, 4]);
  });

  it("pingpong over all frames; a one-frame range just holds", async () => {
    expect((await play({ n: 3, loopMode: "pingpong" }, 7)).shown).toEqual([0, 1, 2, 1, 0, 1, 2]);
    const one = await play({ n: 5, range: [2, 2], loopMode: "pingpong" }, 4);
    expect(one.shown).toEqual([2, 2, 2, 2]);
    expect(one.target.calls).toEqual([2]);
  });

  it("once holds at the range end", async () => {
    expect((await play({ n: 10, range: [1, 3], loopMode: "once" }, 6)).shown).toEqual([1, 2, 3, 3, 3, 3]);
  });

  it("setLoopMode('once') mid-play holds at the range end; back to 'repeat' plays on", async () => {
    const { seq } = await play({ n: 10, range: [2, 5] }, 3); // shown 2, 3, 4 (t = 200)
    seq.setLoopMode("once", 250);
    const shown: number[] = [];
    for (const t of [350, 450, 550, 950]) {
      await tick();
      shown.push(seq.update(t));
    }
    expect(shown).toEqual([5, 5, 5, 5]);
    seq.setLoopMode("repeat", 1000);
    for (const t of [1100, 1200]) {
      await tick();
      shown.push(seq.update(t));
    }
    expect(shown).toEqual([5, 5, 5, 5, 2, 3]);
    expect(seq.stats.stalls).toBe(0);
  });

  it("stop / resume on the backward leg of a ping-pong keeps going backward (no stall at the turn)", async () => {
    // RV finding 1: range [2,5] shows 2,3,4,5,4; paused on the way back, it used to turn forward to 5 (evicted:
    // the ring was filled backward), stall and drop the frame.
    const { seq } = await play({ n: 10, range: [2, 5], loopMode: "pingpong" }, 5);
    expect(seq.stats.frame).toBe(4);
    const stalls = seq.stats.stalls;
    seq.stop();
    seq.resume(400);
    const shown: number[] = [];
    for (const t of [500, 600, 700, 800]) {
      await tick();
      shown.push(seq.update(t));
    }
    expect(shown).toEqual([3, 2, 3, 4]);
    expect(seq.stats.stalls).toBe(stalls);
  });

  it("seek and setRange keep a ping-pong's direction", async () => {
    const { seq } = await play({ n: 10, range: [2, 6], loopMode: "pingpong" }, 6); // 2,3,4,5,6,5 (backward)
    expect(seq.stats.frame).toBe(5);
    seq.seek(4, 500);
    const shown: number[] = [seq.stats.frame];
    for (const t of [600, 700, 800]) {
      await tick();
      shown.push(seq.update(t));
    }
    expect(shown).toEqual([4, 3, 2, 3]);
    await tick();
    expect(seq.update(900)).toBe(4); // forward now
    seq.setRange([1, 8], 900);
    await tick();
    expect(seq.update(1000)).toBe(5); // still forward
    for (const t of [1100, 1200, 1300]) {
      await tick();
      seq.update(t); // 6, 7, 8
    }
    await tick();
    expect(seq.update(1400)).toBe(7); // turned at 8
    seq.setRange([1, 7], 1400);
    await tick();
    expect(seq.update(1500)).toBe(6); // still backward
    expect(seq.stats.stalls).toBe(0);
  });

  it("a clock-due frame still decoding at stop() does not land while stopped; a seek (before or after the stop) does", async () => {
    const { loader, resolve } = deferredLoader();
    const seq = new VDBSequence({ urls: urls(6), fps: 10, prefetch: 0, loader });
    const pl = seq.preload(1);
    resolve(0);
    await pl;
    const target = makeTarget();
    seq.start(target, 0);
    seq.update(100); // frame 1 is due, still decoding: stall
    expect(seq.stats.frame).toBe(0);
    seq.stop();
    resolve(1);
    await tick();
    expect(seq.stats.frame).toBe(0);
    expect(target.calls).toEqual([0]);
    seq.seek(3, 200);
    resolve(3);
    await tick();
    expect(seq.stats.frame).toBe(3);
    expect(target.calls).toEqual([0, 3]);
    // A seek while playing, then stop (demo 12's paused start): the asked-for frame still lands.
    seq.resume(300);
    seq.seek(5, 300);
    seq.stop();
    resolve(5);
    await tick();
    expect(seq.stats.frame).toBe(5);
    expect(target.calls).toEqual([0, 3, 5]);
  });

  it("prefetch follows the playback order and never leaves the range", async () => {
    const seen: number[] = [];
    const seq = new VDBSequence({ urls: urls(10), fps: 10, prefetch: 2, range: [2, 4], loopMode: "pingpong", loader: immediateLoader((i) => seen.push(i)) });
    await seq.preload();
    seq.start(makeTarget(), 0);
    await tick();
    expect([...new Set(seen)].sort((a, b) => a - b)).toEqual([2, 3, 4]);
    expect(seq.stats.stalls).toBe(0);
  });

  it("seek wraps into the clip, then clamps into the range, and continues from there", async () => {
    const seq = new VDBSequence({ urls: urls(10), fps: 10, prefetch: 2, range: [3, 6], loader: immediateLoader() });
    seq.start(makeTarget(), 0);
    seq.stop(); // a stopped player shows a seeked frame once it is decoded
    await tick();
    seq.seek(9, 0);
    await tick();
    expect(seq.stats.frame).toBe(6);
    seq.seek(-9, 0); // wraps to 1, clamps to 3
    await tick();
    expect(seq.stats.frame).toBe(3);
    seq.seek(5, 0);
    await tick();
    seq.resume(0);
    expect(seq.update(100)).toBe(6);
    await tick();
    expect(seq.update(200)).toBe(3);
  });

  it("setRange and setLoopMode re-anchor at the shown frame, clamped into the new range", async () => {
    const seq = new VDBSequence({ urls: urls(10), fps: 10, prefetch: 2, loader: immediateLoader() });
    await seq.preload(10);
    const target = makeTarget();
    seq.start(target, 0);
    await seq.preload(10);
    seq.stop();
    seq.seek(8, 0);
    seq.setRange([2, 4], 0); // 8 is outside: continues from the nearest range frame
    await tick();
    seq.resume(0);
    expect(seq.range).toEqual([2, 4]);
    expect(seq.stats.frame).toBe(4);
    expect(seq.update(100)).toBe(2);
    seq.setLoopMode("pingpong", 100);
    expect(seq.loopMode).toBe("pingpong");
    const shown: number[] = [];
    for (const t of [200, 300, 400]) {
      await tick();
      shown.push(seq.update(t));
    }
    expect(shown).toEqual([3, 4, 3]);
  });

  it("setRange works before start; start shows the range start", async () => {
    const seq = new VDBSequence({ urls: urls(10), prefetch: 1, loader: immediateLoader() });
    seq.setRange([5, 7], 0);
    await seq.preload(2);
    const target = makeTarget();
    seq.start(target, 0);
    expect(target.calls).toEqual([5]);
  });

  it("rejects bad options and setter values with VDB_BAD_OPTION", () => {
    const code = (fn: () => unknown): string | undefined => {
      try {
        fn();
      } catch (e) {
        return e instanceof VDBError ? e.code : String(e);
      }
      return undefined;
    };
    const make = (extra: object = {}) => new VDBSequence({ urls: urls(10), loader: immediateLoader(), ...extra });
    for (const range of [[-1, 3], [4, 3], [0, 10], [0.5, 3], [0, NaN], [1], "ab"]) {
      expect(code(() => make({ range })), JSON.stringify(range)).toBe("VDB_BAD_OPTION");
    }
    expect(code(() => make({ loopMode: "bounce" }))).toBe("VDB_BAD_OPTION");
    const seq = make();
    expect(code(() => seq.setRange([0, 99]))).toBe("VDB_BAD_OPTION");
    expect(code(() => seq.setLoopMode("bounce" as never))).toBe("VDB_BAD_OPTION");
    expect(seq.range).toEqual([0, 9]); // a rejected setter changes nothing
    expect(seq.loopMode).toBe("repeat");
    seq.dispose();
    expect(code(() => seq.setRange([0, 1]))).toBe("VDB_DISPOSED");
    expect(code(() => seq.setLoopMode("pingpong"))).toBe("VDB_DISPOSED");
  });
});
