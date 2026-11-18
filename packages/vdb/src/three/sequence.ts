/**
 * `VDBSequence`: manifest-driven animated-volume playback on a `VDBGridPool`
 * (task K2.3; v1's `NanoVDBSequence`, docs/SPEC §3.5). Decodes per-frame grid
 * images ahead of time into a bounded prefetch ring and swaps the shown one
 * into ONE pool id with `pool.replace(id, image)`.
 *
 * ## The pool contract
 *
 *   const id = pool.add(firstFrame, { capacity: largestFrameBytes });
 *   sequence.start({ pool, id });
 *   // every frame:
 *   sequence.update(performance.now());   // may call pool.replace(id, frame)
 *   pool.update(encoder) / pool.flush();  // writes the frame in place
 *   grids.update();                       // VDBGridNode: rebinds only after a growth
 *   renderer.render(scene, camera);
 *
 * Reserve `capacity` for the largest frame: every `replace` then writes in
 * place (no move, no `pool.version` change, no rebind, no new bind group). A
 * frame larger than the capacity still works, but it moves the image, and the
 * next `update` may grow the pool (a `version` bump), so keep calling
 * `grids.update()` after the pool's update either way.
 *
 * **Frame arrays are held by reference.** `replace` keeps the image until the
 * `update` that writes it, and until `pool.settled()` resolves while a growth
 * is unconfirmed (the pool may have to stage it again after a rollback). The
 * player never writes into a frame array, and its ring only drops references.
 * A custom `loader` that recycles frame buffers must not reuse an array it
 * returned before `await pool.settled()` after the update that wrote it.
 *
 * ## Design for testability (the scheduler is a pure clock consumer)
 *
 * The player never owns the animation loop or reads the wall clock itself.
 * `update(now)` takes the current time in ms; the consumer calls it from its
 * own `requestAnimationFrame` loop with `performance.now()`, and tests call it
 * with a fake, hand-advanced clock. The pool is reached only through
 * `replace(id, image)`, so tests pass a stub.
 *
 * ## Stall policy: never block the render loop
 *
 * If the frame that is due hasn't finished decoding, `update()` does not wait:
 * it holds the last-shown frame, counts a stall, and returns. Decoding
 * continues in the background; the next `update()` that finds the frame ready
 * advances to wherever the clock now points (intermediate frames are dropped:
 * real-time catch-up, not slow motion).
 */

import { VDBError } from "../core/errors.js";
import { NanoVDBFile } from "../tools/index.js";

/** The part of a `VDBGridPool` the player uses. */
export interface VDBSequencePoolLike {
  replace(id: number, image: Uint32Array): void;
}

/** Where frames go: grid `id` of `pool` (added with `capacity` for the largest frame). */
export interface VDBSequenceTarget {
  readonly pool: VDBSequencePoolLike;
  readonly id: number;
}

/** Decodes one frame's URL (or index) into a NanoVDB grid image. */
export type VDBFrameLoader = (url: string, index: number) => Promise<Uint32Array>;

export interface VDBSequenceOptions {
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
   * Decode override. Default: `NanoVDBFile.fromURL(url)` -> `gridImage(0)`.
   * The demo passes an in-memory loader (build frames on the fly, no fetch);
   * the URL/fetch path is exercised via a Blob-URL manifest + this default.
   */
  loader?: VDBFrameLoader;
  /** Clock source. Default `performance.now`. Injected as a fake in tests. */
  now?: () => number;
  /** Fired whenever a NEW frame is shown (after its `pool.replace`). */
  onFrame?: (stats: Readonly<VDBSequenceStats>) => void;
  /** Fired if a frame fails to decode. Playback continues (that frame stays a stall). */
  onError?: (error: unknown, index: number) => void;
}

export interface VDBSequenceStats {
  /** Frame index currently shown. -1 before the first frame binds. */
  frame: number;
  /** Decoded-and-ready frames sitting in the ring ahead of `frame`. */
  decodedAhead: number;
  /** Cumulative count of updates that found the due frame not yet decoded. */
  stalls: number;
  /** CPU time (ms) of the last `pool.replace` (staging only; the GPU write lands in the next pool update). */
  replaceMs: number;
  /** `pool.replace` calls since `start` (the initial frame 0 included). */
  swaps: number;
}

/** Default loader: real fetch + parse of grid 0. */
async function defaultLoader(url: string): Promise<Uint32Array> {
  const file = await NanoVDBFile.fromURL(url);
  return file.gridImage(0);
}

export class VDBSequence {
  readonly frameCount: number;
  readonly fps: number;
  readonly prefetch: number;
  readonly loop: boolean;

  private readonly urlFor: (index: number) => string;
  private readonly loader: VDBFrameLoader;
  private readonly now: () => number;
  private readonly onFrame: VDBSequenceOptions["onFrame"];
  private readonly onError: VDBSequenceOptions["onError"];
  private readonly frameDurationMs: number;

  /** Decoded frame images by frame index. */
  private readonly ring = new Map<number, Uint32Array>();
  /** In-flight decode promises by frame index (dedupe + stall detection). */
  private readonly inflight = new Map<number, Promise<void>>();

  private target: VDBSequenceTarget | undefined;
  private playing = false;
  private startNow = 0;
  private startFrame = 0;
  private _frame = -1;
  private _stalls = 0;
  private _replaceMs = 0;
  private _swaps = 0;

  constructor(opts: VDBSequenceOptions) {
    const frameCount = opts.frameCount ?? (Array.isArray(opts.urls) ? opts.urls.length : undefined);
    if (frameCount === undefined || !Number.isInteger(frameCount) || frameCount < 1) {
      throw new VDBError(
        "VDB_BAD_OPTION",
        "VDBSequence: `frameCount` is required (a positive integer) when `urls` is a function; " +
          "with an array of urls it defaults to urls.length.",
      );
    }
    this.frameCount = frameCount;
    this.fps = opts.fps ?? 24;
    if (!(this.fps > 0)) throw new VDBError("VDB_BAD_OPTION", `VDBSequence: fps must be > 0 (got ${this.fps}).`);
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

  get stats(): Readonly<VDBSequenceStats> {
    return {
      frame: this._frame,
      decodedAhead: this.decodedAhead(),
      stalls: this._stalls,
      replaceMs: this._replaceMs,
      swaps: this._swaps,
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
      .then((image) => {
        this.ring.set(index, image);
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
   * Begin playback into grid `target.id` of `target.pool`. Replaces it with
   * frame 0 at once if that is decoded (else the first `update` that finds it
   * ready does), kicks off prefetch, and anchors the clock at `now` (defaults
   * to the injected clock). Call `update(now)` every animation frame thereafter.
   */
  start(target: VDBSequenceTarget, now: number = this.now()): void {
    if (target === null || typeof target !== "object" || typeof target.pool?.replace !== "function" || !Number.isInteger(target.id) || target.id < 0) {
      throw new VDBError(
        "VDB_BAD_OPTION",
        "VDBSequence.start: target must be { pool, id } with a VDBGridPool (or an object with replace(id, image)) and a non-negative grid id.");
    }
    this.target = target;
    this.playing = true;
    this.startNow = now;
    this.startFrame = 0;
    this._frame = -1;
    this._stalls = 0;
    this._replaceMs = 0;
    this._swaps = 0;
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
   * Advance the scheduler to time `now`. Replaces the grid with the frame the
   * clock points at if it is decoded; otherwise holds the last frame and counts
   * a stall. Never blocks. Returns the frame index currently shown.
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
   * Show `index` if decoded (`pool.replace`, counts a swap, fires `onFrame`,
   * advances prefetch); otherwise count a stall and hold. Shared by
   * start/seek/update.
   */
  private showFrame(index: number): void {
    if (!this.target) return;
    const image = this.ring.get(index);
    if (!image) {
      // Not decoded yet — hold last frame, count a stall, keep decoding.
      if (this._frame !== index) this._stalls++;
      this.ensurePrefetch(index);
      return;
    }
    if (index === this._frame) return;

    const t0 = this.now();
    this.target.pool.replace(this.target.id, image);
    this._replaceMs = this.now() - t0;
    this._frame = index;
    this._swaps++;
    this.ensurePrefetch(index);
    this.onFrame?.(this.stats);
  }

  /** Drop all decoded frames and in-flight tracking (playback state kept). The pool keeps the grid. */
  dispose(): void {
    this.playing = false;
    this.ring.clear();
    this.inflight.clear();
  }
}
