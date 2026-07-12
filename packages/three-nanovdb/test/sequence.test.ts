import { describe, expect, it, vi } from "vitest";
import { NanoVDBSequence } from "../src/sequence.js";
import type { FrameLoader, SequenceTarget } from "../src/sequence.js";
import type { NanoVDBGrid } from "../src/grid.js";

/**
 * Node-safe `NanoVDBSequence` tests: the prefetch ring + frame-time scheduler +
 * stall policy, exercised with a mocked `loader`, a fake clock, and a stub
 * `SequenceTarget` — no browser, no GPU. This is the ground-truth for the
 * playback logic; the end-to-end GPU proof (real rebind changing pixels) is
 * `e2e/demo-05.spec.ts`.
 */

/** A tagged stand-in for a decoded grid — the sequence only forwards it to `rebindGrid`. */
function grid(tag: number): NanoVDBGrid {
  return { tag } as unknown as NanoVDBGrid;
}

/** A stub render target recording the grids rebound into it. */
function makeTarget(): SequenceTarget & { calls: number[] } {
  const calls: number[] = [];
  return {
    calls,
    rebindGrid(g: NanoVDBGrid): void {
      calls.push((g as unknown as { tag: number }).tag);
    },
  };
}

/** Loader that resolves each frame immediately to `grid(i)`. */
function immediateLoader(spy?: (i: number) => void): FrameLoader {
  return (_url: string, i: number): Promise<NanoVDBGrid> => {
    spy?.(i);
    return Promise.resolve(grid(i));
  };
}

/** Loader whose per-frame resolution you drive by hand (for stall tests). */
function deferredLoader(): { loader: FrameLoader; resolve: (i: number) => void; reject: (i: number, e: unknown) => void } {
  const resolvers = new Map<number, (g: NanoVDBGrid) => void>();
  const rejectors = new Map<number, (e: unknown) => void>();
  const loader: FrameLoader = (_url, i) =>
    new Promise<NanoVDBGrid>((res, rej) => {
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

describe("NanoVDBSequence — construction", () => {
  it("defaults frameCount to urls.length for an array", () => {
    const seq = new NanoVDBSequence({ urls: urls(5), loader: immediateLoader() });
    expect(seq.frameCount).toBe(5);
    expect(seq.fps).toBe(24);
    expect(seq.prefetch).toBe(3);
    expect(seq.loop).toBe(true);
  });

  it("requires an explicit frameCount when urls is a function", () => {
    expect(() => new NanoVDBSequence({ urls: (i) => `f${i}`, loader: immediateLoader() })).toThrow(/frameCount/);
    const seq = new NanoVDBSequence({ urls: (i) => `f${i}`, frameCount: 8, loader: immediateLoader() });
    expect(seq.frameCount).toBe(8);
  });

  it("rejects a non-positive fps", () => {
    expect(() => new NanoVDBSequence({ urls: urls(3), fps: 0, loader: immediateLoader() })).toThrow(/fps/);
  });
});

describe("NanoVDBSequence — prefetch ring", () => {
  it("preload decodes exactly the first N frames", async () => {
    const seen: number[] = [];
    const seq = new NanoVDBSequence({ urls: urls(10), prefetch: 2, loader: immediateLoader((i) => seen.push(i)) });
    await seq.preload(3);
    expect(seen.sort((a, b) => a - b)).toEqual([0, 1, 2]);
  });

  it("start binds frame 0 and reports decodedAhead within the prefetch window", async () => {
    const seq = new NanoVDBSequence({ urls: urls(10), prefetch: 3, loader: immediateLoader() });
    await seq.preload(4); // frames 0..3
    const target = makeTarget();
    seq.start(target, 0);
    expect(target.calls).toEqual([0]);
    expect(seq.stats.frame).toBe(0);
    expect(seq.stats.rebinds).toBe(1);
    // frames 1,2,3 sit ahead, ready.
    expect(seq.stats.decodedAhead).toBe(3);
  });

  it("evicts decoded frames that fall outside the forward window", async () => {
    let decodes = 0;
    const seq = new NanoVDBSequence({ urls: urls(6), prefetch: 1, loop: false, loader: immediateLoader(() => decodes++) });
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

describe("NanoVDBSequence — scheduler", () => {
  it("advances one frame per frame-duration and fires onFrame per new frame", async () => {
    const shown: number[] = [];
    // fps 10 => 100ms/frame exactly (no FP drift).
    const seq = new NanoVDBSequence({
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

  it("does not re-bind within the same frame's time slice", async () => {
    const seq = new NanoVDBSequence({ urls: urls(4), fps: 10, prefetch: 4, loader: immediateLoader() });
    await seq.preload(4);
    const target = makeTarget();
    seq.start(target, 0);
    seq.update(30);
    seq.update(60);
    seq.update(99);
    expect(target.calls).toEqual([0]);
    expect(seq.stats.rebinds).toBe(1);
  });

  it("clamps at the last frame when loop is false", async () => {
    const seq = new NanoVDBSequence({ urls: urls(4), fps: 10, prefetch: 4, loop: false, loader: immediateLoader() });
    await seq.preload(4);
    const target = makeTarget();
    seq.start(target, 0);
    seq.update(10_000); // way past the end
    expect(seq.stats.frame).toBe(3);
  });

  it("wraps around when loop is true", async () => {
    const seq = new NanoVDBSequence({ urls: urls(4), fps: 10, prefetch: 4, loop: true, loader: immediateLoader() });
    await seq.preload(4);
    const target = makeTarget();
    seq.start(target, 0);
    seq.update(600); // 6 frames elapsed, wraps to frame 2
    expect(seq.stats.frame).toBe(2);
  });

  it("resume continues from the paused frame without resetting", async () => {
    const seq = new NanoVDBSequence({ urls: urls(6), fps: 10, prefetch: 6, loop: false, loader: immediateLoader() });
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
    expect(seq.stats.rebinds).toBe(3);
  });

  it("ignores update() before start() / after stop()", async () => {
    const seq = new NanoVDBSequence({ urls: urls(4), fps: 10, prefetch: 4, loader: immediateLoader() });
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

describe("NanoVDBSequence — stall policy (never block)", () => {
  it("holds the last frame and counts a stall when the due frame is not decoded", async () => {
    const { loader, resolve } = deferredLoader();
    const seq = new NanoVDBSequence({ urls: urls(4), fps: 10, prefetch: 1, loop: false, loader });

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
    const seq = new NanoVDBSequence({
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

describe("NanoVDBSequence — seek", () => {
  it("jumps to a frame and re-anchors the clock there", async () => {
    const seq = new NanoVDBSequence({ urls: urls(8), fps: 10, prefetch: 8, loader: immediateLoader() });
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
    const seq = new NanoVDBSequence({ urls: urls(4), fps: 10, prefetch: 4, loader: immediateLoader() });
    await seq.preload(4);
    const target = makeTarget();
    seq.start(target, 0);
    seq.seek(5, 0); // wraps to 1
    expect(seq.stats.frame).toBe(1);
  });
});

describe("NanoVDBSequence — stats", () => {
  it("reports uploadMs as a finite number and counts rebinds", async () => {
    const seq = new NanoVDBSequence({ urls: urls(4), fps: 10, prefetch: 4, loader: immediateLoader() });
    await seq.preload(4);
    const target = makeTarget();
    seq.start(target, 0);
    seq.update(100);
    seq.update(200);
    expect(Number.isFinite(seq.stats.uploadMs)).toBe(true);
    expect(seq.stats.uploadMs).toBeGreaterThanOrEqual(0);
    expect(seq.stats.rebinds).toBe(3); // frames 0,1,2
  });

  it("dispose clears the ring", async () => {
    const seq = new NanoVDBSequence({ urls: urls(4), prefetch: 4, loader: immediateLoader() });
    await seq.preload(4);
    seq.dispose();
    expect(seq.stats.decodedAhead).toBe(0);
    expect(seq.isPlaying).toBe(false);
  });

  it("uses a real, monotonically increasing clock for uploadMs when now() advances", async () => {
    let t = 0;
    const seq = new NanoVDBSequence({ urls: urls(2), fps: 10, prefetch: 2, loader: immediateLoader(), now: () => (t += 1) });
    await seq.preload(2);
    const target = makeTarget();
    // now() ticks on every read, so the two reads around rebindGrid differ.
    seq.start(target, 0);
    expect(seq.stats.uploadMs).toBeGreaterThan(0);
  });
});

describe("NanoVDBSequence — default loader wiring", () => {
  it("passes the resolved url and index to the loader", async () => {
    const loader = vi.fn(immediateLoader());
    const seq = new NanoVDBSequence({ urls: (i) => `custom/${i}.nvdb`, frameCount: 3, prefetch: 3, loader });
    await seq.preload(2);
    expect(loader).toHaveBeenCalledWith("custom/0.nvdb", 0);
    expect(loader).toHaveBeenCalledWith("custom/1.nvdb", 1);
  });
});
