/**
 * `NanoVDBSequence` — manifest-driven animated-volume playback (docs/SPEC §3.5,
 * docs/PLAN.md Phase 7). Decodes per-frame NanoVDB grids ahead of time into a
 * bounded prefetch ring and swaps them into a `NanoVDBVolumeMaterial` via its
 * `rebindGrid()` in-place buffer update — no per-frame material rebuild.
 *
 * ## Design for testability (the scheduler is a pure-ish clock consumer)
 *
 * The player NEVER owns the animation loop or reads the wall clock itself.
 * `update(now)` takes the current time in ms; the consumer calls it from its
 * own `requestAnimationFrame` loop with `performance.now()`, and tests call it
 * with a fake, hand-advanced clock. All timing is a frame-duration accumulator
 * against the `now` handed to `start()`/`update()`. This makes the entire
 * ring + scheduler + stall logic exercisable in Vitest with a mocked `loader`
 * and no browser, no GPU (the material is reached only through the tiny
 * `SequenceTarget.rebindGrid` seam).
 *
 * ## Stall policy: never block the render loop
 *
 * If the frame that is due hasn't finished decoding, `update()` does NOT wait —
 * it holds the last-shown frame, counts a stall, and returns immediately.
 * Decoding continues in the background; the next `update()` that finds the
 * frame ready advances to wherever the clock now points (intermediate frames
 * are dropped, i.e. real-time catch-up, not slow-motion).
 */

import { NanoVDBFile } from "nanovdb-wgsl";
import { NanoVDBGrid } from "./grid.js";

/**
 * The only surface `NanoVDBSequence` needs from a render target — satisfied by
 * `NanoVDBVolumeMaterial`. Kept minimal so tests can pass a stub with no three
 * dependency.
 */
export interface SequenceTarget {
  rebindGrid(grid: NanoVDBGrid): void;
}

/** Decodes one frame's URL (or index) into a GPU-ready grid. */
export type FrameLoader = (url: string, index: number) => Promise<NanoVDBGrid>;

export interface NanoVDBSequenceOptions {
  /** Per-frame source URLs: an array, or `(i) => url`. Length/`frameCount` gate playback. */
  urls: string[] | ((index: number) => string);
  /** Total frame count. Defaults to `urls.length` when `urls` is an array (required otherwise). */
  frameCount?: number;
  /** Playback rate. Default 24. */
  fps?: number;
  /** How many frames to decode ahead of the current one. Default 3. */
  prefetch?: number;
  /** Loop back to frame 0 after the last frame. Default true. */
  loop?: boolean;
  /**
   * Decode override. Default: `NanoVDBFile.fromURL(url)` -> `NanoVDBGrid.fromFile`.
   * The demo passes an in-memory loader (build frames on the fly, no fetch);
   * the URL/fetch path is exercised via a Blob-URL manifest + this default.
   */
  loader?: FrameLoader;
  /** Clock source. Default `performance.now`. Injected as a fake in tests. */
  now?: () => number;
  /** Fired whenever a NEW frame is shown (after a successful rebind). */
  onFrame?: (stats: Readonly<NanoVDBSequenceStats>) => void;
  /** Fired if a frame fails to decode. Playback continues (that frame stays a stall). */
  onError?: (error: unknown, index: number) => void;
}

export interface NanoVDBSequenceStats {
  /** Frame index currently shown. -1 before the first frame binds. */
  frame: number;
  /** Decoded-and-ready frames sitting in the ring ahead of `frame`. */
  decodedAhead: number;
  /** Cumulative count of updates that found the due frame not yet decoded. */
  stalls: number;
  /** CPU time (ms) of the last `rebindGrid` staging copy (the GPU writeBuffer lands at next render). */
  uploadMs: number;
  /** Total successful rebinds since `start` (includes the initial frame-0 bind). */
  rebinds: number;
}

/** Default loader: real fetch + parse — the same path demo 02 uses for files. */
async function defaultLoader(url: string): Promise<NanoVDBGrid> {
  const file = await NanoVDBFile.fromURL(url);
  return NanoVDBGrid.fromFile(file, 0);
}

export class NanoVDBSequence {
  readonly frameCount: number;
  readonly fps: number;
  readonly prefetch: number;
  readonly loop: boolean;

  private readonly urlFor: (index: number) => string;
  private readonly loader: FrameLoader;
  private readonly now: () => number;
  private readonly onFrame: NanoVDBSequenceOptions["onFrame"];
  private readonly onError: NanoVDBSequenceOptions["onError"];
  private readonly frameDurationMs: number;

  /** Decoded grids by frame index. */
  private readonly ring = new Map<number, NanoVDBGrid>();
  /** In-flight decode promises by frame index (dedupe + stall detection). */
  private readonly inflight = new Map<number, Promise<void>>();

  private target: SequenceTarget | undefined;
  private playing = false;
  private startNow = 0;
  private startFrame = 0;
  private _frame = -1;
  private _stalls = 0;
  private _uploadMs = 0;
  private _rebinds = 0;

  constructor(opts: NanoVDBSequenceOptions) {
    const frameCount = opts.frameCount ?? (Array.isArray(opts.urls) ? opts.urls.length : undefined);
    if (frameCount === undefined || !Number.isInteger(frameCount) || frameCount < 1) {
      throw new Error(
        "NanoVDBSequence: `frameCount` is required (a positive integer) when `urls` is a function; " +
          "with an array of urls it defaults to urls.length.",
      );
    }
    this.frameCount = frameCount;
    this.fps = opts.fps ?? 24;
    if (this.fps <= 0) throw new Error(`NanoVDBSequence: fps must be > 0 (got ${this.fps}).`);
    this.prefetch = Math.max(0, opts.prefetch ?? 3);
    this.loop = opts.loop ?? true;
    this.frameDurationMs = 1000 / this.fps;

    const urls = opts.urls;
    this.urlFor = typeof urls === "function" ? urls : (i: number): string => urls[i] ?? "";
    this.loader = opts.loader ?? defaultLoader;
    this.now = opts.now ?? ((): number => performance.now());
    this.onFrame = opts.onFrame;
    this.onError = opts.onError;
  }

  get stats(): Readonly<NanoVDBSequenceStats> {
    return {
      frame: this._frame,
      decodedAhead: this.decodedAhead(),
      stalls: this._stalls,
      uploadMs: this._uploadMs,
      rebinds: this._rebinds,
    };
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  /** Wrap a raw (possibly negative or >= frameCount) index into [0, frameCount). */
  private wrap(index: number): number {
    const n = this.frameCount;
    return ((index % n) + n) % n;
  }

  /**
   * Ensure frames `[from .. from+prefetch]` (wrapped when looping) are decoding
   * or decoded, and evict frames outside that window to bound memory.
   */
  private ensurePrefetch(from: number): void {
    const want = new Set<number>();
    for (let k = 0; k <= this.prefetch; k++) {
      const raw = from + k;
      if (!this.loop && raw >= this.frameCount) break;
      want.add(this.wrap(raw));
    }
    // Keep the current frame available for hold-last-frame even if it slipped
    // out of the forward window (prefetch 0 edge case).
    if (this._frame >= 0) want.add(this._frame);

    for (const index of want) this.decodeFrame(index);

    // Evict decoded frames we no longer want ahead of us.
    for (const index of [...this.ring.keys()]) {
      if (!want.has(index)) this.ring.delete(index);
    }
  }

  /** Start (or return) the decode of one frame; resolves into the ring. */
  private decodeFrame(index: number): Promise<void> {
    if (this.ring.has(index)) return Promise.resolve();
    const existing = this.inflight.get(index);
    if (existing) return existing;

    const p = this.loader(this.urlFor(index), index)
      .then((grid) => {
        this.ring.set(index, grid);
      })
      .catch((err) => {
        this.onError?.(err, index);
      })
      .finally(() => {
        this.inflight.delete(index);
      });
    this.inflight.set(index, p);
    return p;
  }

  private decodedAhead(): number {
    if (this._frame < 0) return this.ring.size;
    let count = 0;
    for (let k = 1; k <= this.prefetch; k++) {
      const raw = this._frame + k;
      if (!this.loop && raw >= this.frameCount) break;
      if (this.ring.has(this.wrap(raw))) count++;
    }
    return count;
  }

  /**
   * Decode the first `count` frames and resolve once all are ready. Useful for
   * a deterministic warm start (the demo's `?test=1` awaits this so playback has
   * zero stalls). Does not start playback.
   */
  async preload(count = Math.min(this.frameCount, this.prefetch + 1)): Promise<void> {
    const n = Math.min(count, this.frameCount);
    const ps: Promise<void>[] = [];
    for (let i = 0; i < n; i++) ps.push(this.decodeFrame(this.wrap(i)));
    await Promise.all(ps);
  }

  /**
   * Begin playback into `target`. Binds frame 0 immediately if it is already
   * decoded (otherwise the first `update` that finds it ready binds it), kicks
   * off prefetch, and anchors the clock at `now` (defaults to the injected
   * clock). Call `update(now)` every animation frame thereafter.
   */
  start(target: SequenceTarget, now: number = this.now()): void {
    this.target = target;
    this.playing = true;
    this.startNow = now;
    this.startFrame = 0;
    this._frame = -1;
    this._stalls = 0;
    this._uploadMs = 0;
    this._rebinds = 0;
    this.ensurePrefetch(0);
    this.showFrame(0);
  }

  /** Stop playback. The ring is preserved so `start`/`resume`/`seek` can resume cheaply. */
  stop(): void {
    this.playing = false;
  }

  /**
   * Resume after `stop()` WITHOUT resetting to frame 0 or clearing stats:
   * re-anchors the clock at `now` from the currently-shown frame so playback
   * continues from where it paused. No-op if `start()` was never called.
   */
  resume(now: number = this.now()): void {
    if (!this.target) return;
    this.playing = true;
    this.startNow = now;
    this.startFrame = this._frame < 0 ? 0 : this._frame;
  }

  /** Jump to `index`, re-anchoring the clock there so playback continues from it. */
  seek(index: number, now: number = this.now()): void {
    const frame = this.wrap(Math.trunc(index));
    this.startNow = now;
    this.startFrame = frame;
    this.ensurePrefetch(frame);
    this.showFrame(frame);
  }

  /**
   * Advance the scheduler to time `now`. Binds the frame the clock points at if
   * it is decoded; otherwise holds the last frame and counts a stall. Never
   * blocks. Returns the frame index currently shown.
   */
  update(now: number): number {
    if (!this.playing || !this.target) return this._frame;

    const elapsed = now - this.startNow;
    const advanced = Math.floor(elapsed / this.frameDurationMs);
    const raw = this.startFrame + advanced;
    let target: number;
    if (this.loop) {
      target = this.wrap(raw);
    } else {
      target = Math.min(raw, this.frameCount - 1);
    }

    if (target === this._frame) {
      // Same frame is due; just keep prefetch warm.
      this.ensurePrefetch(this._frame < 0 ? 0 : this._frame);
      return this._frame;
    }

    this.showFrame(target);
    return this._frame;
  }

  /**
   * Bind `index` if decoded (counts a rebind, fires `onFrame`, advances
   * prefetch); otherwise count a stall and hold. Shared by start/seek/update.
   */
  private showFrame(index: number): void {
    if (!this.target) return;
    const grid = this.ring.get(index);
    if (!grid) {
      // Not decoded yet — hold last frame, count a stall, keep decoding.
      if (this._frame !== index) this._stalls++;
      this.ensurePrefetch(index);
      return;
    }
    if (index === this._frame) return;

    const t0 = this.now();
    this.target.rebindGrid(grid);
    this._uploadMs = this.now() - t0;
    this._frame = index;
    this._rebinds++;
    this.ensurePrefetch(index);
    this.onFrame?.(this.stats);
  }

  /** Drop all decoded frames and in-flight tracking (playback state kept). */
  dispose(): void {
    this.playing = false;
    this.ring.clear();
    this.inflight.clear();
  }
}
