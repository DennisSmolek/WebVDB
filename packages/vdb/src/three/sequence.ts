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
 *
 * ## Failed frames: no retry until `seek` / `start`
 *
 * A frame whose loader rejects goes into a failed set: `onError` fires once,
 * and the player does not ask the loader for it again (no retry storm from
 * the per-frame prefetch). While the clock points at it, `update()` holds the
 * last frame and counts stalls; playback moves on when the clock reaches the
 * next decodable frame. `seek()` and `start()` clear the set, so they retry
 * every failed frame (call `seek(stats.frame)` to retry in place).
 * `stats.failed` counts the frames in the set.
 *
 * ## Disposal
 *
 * `dispose()` drops the ring and the in-flight tracking and bumps a
 * generation token: decodes that finish later are dropped (no ring entry, no
 * `onError`). `start`, `seek`, `resume` and `preload` after `dispose()` throw
 * (or reject with) `VDB_DISPOSED`; `update` returns the last frame.
 */

import { VDBError } from "../core/errors.js";
import { NvdbFile } from "../tools/index.js";

/** The part of a `VDBGridPool` the player uses. */
export interface VDBSequencePoolLike<TFrame = Uint32Array> {
  replace(id: number, image: TFrame): void;
}

/**
 * Where frames go: grid `id` of `pool` (added with `capacity` for the largest frame), or (K5.9) a `source` with
 * `replace(frame)`: a dense `.wvdb` source (`vdbDenseTextures`) playing `WvdbDenseVolume` frames, which a
 * `VDBFrameVolume` over it follows by itself.
 */
export type VDBSequenceTarget<TFrame = Uint32Array> =
  | { readonly pool: VDBSequencePoolLike<TFrame>; readonly id: number }
  | { readonly source: { replace(frame: TFrame): void } };

/** Decodes one frame's URL (or index) into a frame: a NanoVDB grid image by default. */
export type VDBFrameLoader<TFrame = Uint32Array> = (url: string, index: number) => Promise<TFrame>;

export interface VDBSequenceOptions<TFrame = Uint32Array> {
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
   * Decode override. Default: `NvdbFile.fromURL(url)` -> `gridImage(0)`.
   * The demo passes an in-memory loader (build frames on the fly, no fetch);
   * the URL/fetch path is exercised via a Blob-URL manifest + this default.
   */
  loader?: VDBFrameLoader<TFrame>;
  /** Clock source. Default `performance.now`. Injected as a fake in tests. */
  now?: () => number;
  /** Fired whenever a NEW frame is shown (after its `pool.replace`). */
  onFrame?: (stats: Readonly<VDBSequenceStats>) => void;
  /**
   * Fired once when a frame fails to decode. The frame is not retried until
   * `seek` or `start`; playback skips it (a stall while the clock points at it).
   */
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
  /** Frames whose decode failed and that wait for a `seek` / `start` to be retried. */
  failed: number;
}

/** Default loader: real fetch + parse of grid 0. */
async function defaultLoader(url: string): Promise<Uint32Array> {
  const file = await NvdbFile.fromURL(url);
  return file.gridImage(0);
}

/**
 * Plays a frame sequence into ONE pool grid: decodes frames ahead into a
 * bounded ring and `pool.replace`s the due one, driven by your clock. See the
 * file comment for the pool contract, the stall policy, failed frames and
 * disposal.
 *
 *   const id = pool.add(firstFrame, { capacity: largestFrameBytes });
 *   const sequence = new VDBSequence({ urls, fps: 24 });
 *   sequence.start({ pool, id });
 *   // per frame: sequence.update(performance.now()); pool.update(encoder); grids.update(); render
 */
export class VDBSequence<TFrame = Uint32Array> {
  readonly frameCount: number;
  readonly fps: number;
  readonly prefetch: number;
  readonly loop: boolean;

  private readonly urlFor: (index: number) => string;
  private readonly loader: VDBFrameLoader<TFrame>;
  private readonly hasLoader: boolean;
  private readonly now: () => number;
  private readonly onFrame: VDBSequenceOptions<TFrame>["onFrame"];
  private readonly onError: VDBSequenceOptions<TFrame>["onError"];
  private readonly frameDurationMs: number;

  /** Decoded frame images by frame index. */
  private readonly ring = new Map<number, TFrame>();
  /** In-flight decode promises by frame index (dedupe + stall detection). */
  private readonly inflight = new Map<number, Promise<void>>();
  /** Frames whose decode failed: not retried until `seek` / `start`. */
  private readonly failed = new Set<number>();
  /** Bumped by `dispose()`: a decode that finishes under an older generation is dropped. */
  private generation = 0;
  private disposed = false;

  private replaceFrame: ((frame: TFrame) => void) | undefined;
  /** K5.9: a frame shown by `seek` while stopped once its decode lands. */
  private pendingFrame = -1;
  private playing = false;
  private startNow = 0;
  private startFrame = 0;
  private _frame = -1;
  private _stalls = 0;
  private _replaceMs = 0;
  private _swaps = 0;

  constructor(opts: VDBSequenceOptions<TFrame>) {
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
    this.hasLoader = !!opts.loader;
    this.loader = opts.loader ?? (defaultLoader as unknown as VDBFrameLoader<TFrame>);
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
      failed: this.failed.size,
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

  /**
   * Start (or return) the decode of one frame; resolves into the ring. A
   * failed frame is not retried (see the file comment); a decode that settles
   * after `dispose()` is dropped.
   */
  private decodeFrame(index: number): Promise<void> {
    if (this.ring.has(index) || this.failed.has(index)) return Promise.resolve();
    const existing = this.inflight.get(index);
    if (existing) return existing;

    const generation = this.generation;
    const current = (): boolean => generation === this.generation;
    let load: Promise<TFrame>;
    try {
      load = Promise.resolve(this.loader(this.urlFor(index), index));
    } catch (err) {
      load = Promise.reject(err);
    }
    const p = load
      .then((image) => {
        if (!current()) return;
        this.ring.set(index, image);
        if (index === this.pendingFrame && !this.playing) this.showFrame(index);
      })
      .catch((err) => {
        if (!current()) return;
        this.failed.add(index);
        this.onError?.(err, index);
      })
      .finally(() => {
        if (current()) this.inflight.delete(index);
      });
    this.inflight.set(index, p);
    return p;
  }

  private assertLive(what: string): void {
    if (this.disposed) throw new VDBError("VDB_DISPOSED", `VDBSequence: ${what} after dispose(). Create a new VDBSequence.`);
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
    this.assertLive("preload()");
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
   * Clears the failed set (failed frames are retried).
   *
   * @throws VDBError `VDB_DISPOSED` after `dispose()`; `VDB_BAD_OPTION` for a bad target.
   */
  start(target: VDBSequenceTarget<TFrame>, now: number = this.now()): void {
    this.assertLive("start()");
    const t = target as { pool?: { replace?: unknown }; id?: unknown; source?: { replace?: unknown } } | null;
    if (t !== null && typeof t === "object" && !("pool" in t) && typeof t.source?.replace === "function") {
      if (!this.hasLoader) throw new VDBError("VDB_BAD_OPTION", "VDBSequence.start: a { source } target plays the frames your loader returns; pass a loader (the default decodes .nvdb images).");
      const source = t.source as { replace(frame: TFrame): void };
      this.replaceFrame = (frame) => source.replace(frame);
    } else if (t === null || typeof t !== "object" || typeof t.pool?.replace !== "function" || !Number.isInteger(t.id) || (t.id as number) < 0) {
      throw new VDBError(
        "VDB_BAD_OPTION",
        "VDBSequence.start: target must be { pool, id } with a VDBGridPool (or an object with replace(id, image)) and a non-negative grid id, or { source } with replace(frame).");
    } else {
      const { pool, id } = target as { pool: VDBSequencePoolLike<TFrame>; id: number };
      this.replaceFrame = (frame) => pool.replace(id, frame);
    }
    this.playing = true;
    this.startNow = now;
    this.startFrame = 0;
    this._frame = -1;
    this._stalls = 0;
    this._replaceMs = 0;
    this._swaps = 0;
    this.failed.clear();
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
   *
   * @throws VDBError `VDB_DISPOSED` after `dispose()`.
   */
  resume(now: number = this.now()): void {
    this.assertLive("resume()");
    if (!this.replaceFrame) return;
    this.playing = true;
    this.startNow = now;
    this.startFrame = this._frame < 0 ? 0 : this._frame;
  }

  /**
   * Jump to `index`, re-anchoring the clock there so playback continues from
   * it. Clears the failed set (failed frames are retried).
   *
   * @throws VDBError `VDB_DISPOSED` after `dispose()`.
   */
  seek(index: number, now: number = this.now()): void {
    this.assertLive("seek()");
    this.failed.clear();
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
    if (!this.playing || !this.replaceFrame) return this._frame;

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
    if (!this.replaceFrame) return;
    const image = this.ring.get(index);
    if (image === undefined) {
      // Not decoded yet: hold the last frame, count a stall, keep decoding (a stopped player shows it when it lands).
      if (this._frame !== index) this._stalls++;
      this.pendingFrame = index;
      this.ensurePrefetch(index);
      return;
    }
    this.pendingFrame = -1;
    if (index === this._frame) return;

    const t0 = this.now();
    this.replaceFrame(image);
    this._replaceMs = this.now() - t0;
    this._frame = index;
    this._swaps++;
    this.ensurePrefetch(index);
    this.onFrame?.(this.stats);
  }

  /**
   * Stops playback and drops all decoded frames and in-flight tracking; later
   * decodes are dropped when they settle. The pool keeps the grid (and the
   * frame it shows). Further `start` / `seek` / `resume` / `preload` throw
   * `VDB_DISPOSED`. Idempotent.
   */
  dispose(): void {
    this.disposed = true;
    this.generation++;
    this.playing = false;
    this.replaceFrame = undefined;
    this.pendingFrame = -1;
    this.ring.clear();
    this.inflight.clear();
    this.failed.clear();
  }
}
