/**
 * `VDBGridPool`: many NanoVDB grids in one `GPUBuffer` behind a grid table
 * (docs/dev/design/K1-core.md §1, §4; D8, D10).
 *
 * Model: CPU calls (`add`, `replace`, `remove`, `setTransform`, `setMajorant`)
 * stage changes and return at once; `update(encoder)` applies them. Growth and
 * compaction happen only in `update`, by a new buffer plus copies of the ranges
 * that were resident before that update; `version` bumps then (and when a
 * failed growth rolls back to the previous buffer).
 *
 * Lifecycle (docs/archive/handoffs/K1.md, "K1.3 pool lifecycle"):
 * - A request that could never fit under the byte limit throws `VDB_POOL_LIMIT`
 *   at staging, with the pool unchanged.
 * - Growth allocates inside `out-of-memory` and `validation` error scopes. The
 *   pool switches to the new buffer at once (update is synchronous) but keeps
 *   the old one alive until the scopes resolve; on an error it rolls back to the
 *   old buffer and reports through `settled()` / `lastError`.
 * - A retired buffer is destroyed once its growth is confirmed, two `update`s
 *   have passed, and `queue.onSubmittedWorkDone()` asked at that point resolves.
 */

import { VDBError } from "../errors.js";
import { applyAffine, composeGridAffines, type GridAffines } from "./affine.js";
import { RangeAllocator, type AllocatedRange } from "./allocator.js";
import type { VDBGridTypeName, VDBSlot } from "../types.js";
import { gridClassName, readImageHeader, type VDBImageHeader } from "./image-header.js";
import {
  VDB_GPU_LAYOUT_VERSION,
  VDB_POOL_HEADER_WORDS,
  VDB_POOL_RECORD_WORDS,
  align32,
  dataStartOffset,
  encodeHeader,
  encodeRecord,
  recordByteOffset,
  type VDBGridRecord,
  type VDBVec3,
} from "./layout.js";
import {
  BUFFER_USAGE,
  POOL_BUFFER_USAGE,
  deviceByteLimit,
  floor32,
  requiredLimits,
  unsupportedReason,
  watchDevice,
} from "./limits.js";

/**
 * A dense grid of upper bounds over a grid's index space, one f32 per brick of
 * `2^brickLog2` voxels per side, x fastest (§5.1). Built by `tools`; the pool
 * only stores it.
 */
export interface VDBBrickMajorant {
  /** log2 of the brick edge in index voxels (3 for 8³ bricks). */
  brickLog2: number;
  /** Cells along x, y, z. */
  dims: readonly [number, number, number];
  /** `dims[0] * dims[1] * dims[2]` bounds, x fastest. */
  data: Float32Array;
}

/** A 4x4 column-major object-to-world matrix (three's `Matrix4.elements`), or `null` for the file's map. */
export type VDBTransformInput = Float32Array | Float64Array | readonly number[] | null;

/** Options for {@link VDBGridPool.create}. */
export interface VDBGridPoolOptions {
  /** Initial buffer bytes (default 1 MiB); the first `update` grows it to fit. */
  initialBytes?: number;
  /** Initial table capacity in grids (default 16); the table doubles through the growth path. */
  maxGrids?: number;
  /** Upper bound on the buffer size (default: the device limit). */
  maxBytes?: number;
  /** Debug label for the buffers the pool creates. */
  label?: string;
  /** Extra `GPUBufferUsage` flags ORed into `STORAGE | COPY_DST | COPY_SRC`. */
  usage?: number;
  /**
   * A host-owned buffer to use instead. It must have `STORAGE | COPY_DST`
   * usage. The pool never destroys it and never grows it: when it is full,
   * `add`/`replace`/`setMajorant` throw `VDB_POOL_FULL`.
   */
  buffer?: GPUBuffer;
}

/** Options for {@link VDBGridPool.add}. */
export interface VDBAddOptions {
  /** Bytes to reserve for the image (at least its size), so later `replace` calls fit in place. */
  capacity?: number;
  /** Instance transform (see {@link VDBGridPool.setTransform}). */
  transform?: VDBTransformInput;
  /** Brick-majorant bounds for this grid (see {@link VDBGridPool.setMajorant}). */
  majorant?: VDBBrickMajorant;
  /** Global density bound (record word 35); default `max(root max, background)` when the image has stats. */
  valueMax?: number;
  /** Free-form label, reported by `info`. */
  label?: string;
}

/** What {@link VDBGridPool.info} reports about one grid. */
export interface VDBGridInfo {
  /** The grid's pool id. */
  id: number;
  /** The grid's value type (a NanoVDB grid type name). */
  type: VDBGridTypeName;
  /** NanoVDB `GridType` id. */
  gridTypeId: number;
  /** NanoVDB `GridClass` name, e.g. `"FogVolume"`. */
  gridClass: string;
  /** NanoVDB `GridClass` id. */
  gridClassId: number;
  /** Grid name from the image header. */
  name: string;
  /** The `label` given to `add`, if any. */
  label: string | undefined;
  /** Size of the grid image in bytes. */
  imageBytes: number;
  /** Bytes reserved for the image (≥ `imageBytes`, 32 B multiple). */
  capacity: number;
  /** Byte offset of the image in the pool, or `null` until an `update` places it. */
  imageOffset: number | null;
  /** True once the image bytes are on the GPU (after the `update` that wrote them). */
  resident: boolean;
  /**
   * The pool {@link VDBGridPool.revision} at this grid's last `add` or `replace`: changes exactly
   * when the image's contents may have (not on `setTransform` / `setMajorant`). Data derived from the
   * image (a coarse pyramid, K3.2a) rebuilds when it moves; ids reused after `remove` get a new one.
   */
  imageRevision: number;
  /** True when the image carries node statistics (min/max/average/deviation). */
  hasStats: boolean;
  /** True when brick-majorant bounds are set for this grid. */
  hasBrickMajorant: boolean;
  /** Byte offset of the majorant data in the pool, or `null` when none is placed. */
  majorantOffset: number | null;
  /** Brick counts of the majorant per axis, or `null` when none. */
  majorantDims: VDBVec3 | null;
  /** True when an instance transform (`setTransform`) is set. */
  instanceTransform: boolean;
  /** The tree's background value. */
  background: number;
  /** Global density bound that marchers use (record word 35). */
  valueMax: number;
  /**
   * The root node's min/max statistics as stored in the image (f32). Meaningful
   * only when `hasStats`; without stats they are whatever the file holds
   * (usually 0). `gridStats` and `decodeToAtlas` use them as their range.
   */
  rootMin: number;
  /** Root max statistic; see `rootMin`. */
  rootMax: number;
  /** Voxel size per axis in the grid's own world units (before the instance transform). */
  voxelSize: VDBVec3;
  /** Inclusive index-space box of the active voxels. */
  indexBBox: { min: VDBVec3; max: VDBVec3 };
  /** World-space AABB of the index bbox's voxels (max corner + 1 voxel), through the record's f64 transform. */
  worldBBox: { min: VDBVec3; max: VDBVec3 };
  /** f64 affines before f32 rounding, 3x4 row-major. */
  worldToIndex: Float64Array;
  /** Index-to-world affine, 3x4 row-major, f64. */
  indexToWorld: Float64Array;
}

/** A sub-allocated range plus its GPU state. */
interface RangeState {
  range: AllocatedRange | null;
  /** The GPU holds this range's current bytes at `range.offset`. */
  resident: boolean;
}

interface MajorantState extends RangeState {
  brickLog2: number;
  dims: VDBVec3;
  bytes: number;
  pending: Float32Array | null;
  /** The data last written, kept while a growth is unconfirmed so a rollback can stage it again. */
  retained: Float32Array | null;
}

interface GridState {
  header: VDBImageHeader;
  label: string | undefined;
  capacity: number;
  image: RangeState;
  pendingImage: Uint32Array | null;
  /** The image last written, kept while a growth is unconfirmed so a rollback can stage it again. */
  retainedImage: Uint32Array | null;
  /** +1 per `replace`: tells a rollback whether the old buffer still holds this grid's current image. */
  imageSerial: number;
  /** The pool `revision` of this grid's last `add` / `replace` (K3.2a: derived data rebuilds on a change). */
  imageRevision: number;
  transform: Float64Array | null;
  affines: GridAffines;
  valueMaxOverride: number | undefined;
  majorant: MajorantState | null;
}

/** What the buffer a growth copied from holds for one grid (clean, resident ranges only). */
interface GridSnapshot {
  imageSerial: number;
  image: AllocatedRange | null;
  majorant: MajorantState | null;
  majorantRange: AllocatedRange | null;
}

/** A growth whose allocation the device hasn't confirmed yet. */
interface Growth {
  /** The buffer copied from; kept alive until this growth is confirmed or rolled back. */
  from: GPUBuffer;
  to: GPUBuffer;
  /** Table capacity of `from`. */
  capacity: number;
  /** `update` count when it happened. */
  update: number;
  snapshot: Map<GridState, GridSnapshot>;
  /** Settles in growth order; resolves to the rollback error, or `null` when confirmed or voided. */
  done: Promise<VDBError | null>;
}

/** A replaced buffer awaiting destruction. */
interface Retired {
  buffer: GPUBuffer;
  /** `update` count of the growth that replaced it. */
  since: number;
  /** `onSubmittedWorkDone` has been requested for it. */
  waiting: boolean;
}

/** Updates a retired buffer must outlive before the pool asks the queue about it (one update of slack to submit). */
const RETIRE_AFTER_UPDATES = 2;

const DEFAULT_INITIAL_BYTES = 1 << 20;
const DEFAULT_MAX_GRIDS = 16;

function badOption(message: string): never {
  throw new VDBError("VDB_BAD_OPTION", message);
}

function checkCount(name: string, value: unknown, min: number): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < min) {
    badOption(`${name} must be an integer ≥ ${min}, got ${String(value)}.`);
  }
  return value;
}

function copyTransform(m: VDBTransformInput | undefined): Float64Array | null {
  if (m === null || m === undefined) return null;
  return Float64Array.from(m);
}

function validateMajorant(m: VDBBrickMajorant): MajorantState {
  if (typeof m !== "object" || m === null) badOption("majorant must be { brickLog2, dims, data } or null.");
  checkCount("majorant.brickLog2", m.brickLog2, 0);
  if (m.brickLog2 > 10) badOption(`majorant.brickLog2 is ${m.brickLog2}; bricks larger than 1024 voxels aren't supported.`);
  if (!Array.isArray(m.dims) && !ArrayBuffer.isView(m.dims)) badOption("majorant.dims must be [x, y, z].");
  if (m.dims.length !== 3) badOption(`majorant.dims must have 3 entries, got ${m.dims.length}.`);
  const dims: VDBVec3 = [m.dims[0], m.dims[1], m.dims[2]];
  dims.forEach((d, k) => checkCount(`majorant.dims[${k}]`, d, 1));
  if (!(m.data instanceof Float32Array)) badOption("majorant.data must be a Float32Array.");
  const cells = dims[0] * dims[1] * dims[2];
  if (m.data.length !== cells) {
    badOption(`majorant.data has ${m.data.length} values; dims ${dims.join("x")} need ${cells}. Rebuild the majorant.`);
  }
  const bytes = cells * 4;
  if (bytes > 2 ** 32 - 32) badOption(`majorant needs ${bytes} bytes; the pool addresses ranges with 32 bits.`);
  return { brickLog2: m.brickLog2, dims, bytes, pending: m.data, retained: null, range: null, resident: false };
}

/**
 * `queue.writeBuffer` for any typed array. The WebGPU typings want an
 * `ArrayBuffer`-backed view; callers' arrays are typed `ArrayBufferLike`, and
 * writeBuffer copies the bytes either way.
 */
function write(queue: GPUQueue, buffer: GPUBuffer, offset: number, data: Uint32Array | Float32Array): void {
  queue.writeBuffer(buffer, offset, data as Uint32Array<ArrayBuffer>);
}

function checkValueMax(v: number | undefined): number | undefined {
  if (v === undefined) return undefined;
  if (typeof v !== "number" || Number.isNaN(v)) badOption(`valueMax must be a number, got ${String(v)}.`);
  return v;
}

/**
 * Many NanoVDB grids in one storage buffer, bound at one slot (D10). The pool
 * takes the host's `GPUDevice`, records into the host's encoder, and frees only
 * what it allocated (D8).
 *
 * Images passed to `add`/`replace` are held by reference (not copied) until the
 * next `update` writes them, and after that while a growth is unconfirmed
 * (until `settled()` resolves); don't mutate them before then.
 */
export class VDBGridPool {
  /** The device the pool allocates on. */
  readonly device: GPUDevice;

  #buffer: GPUBuffer;
  #ownsBuffer: boolean;
  #retired: Retired[] = [];
  #growths: Growth[] = [];
  #updates = 0;
  #lastError: VDBError | null = null;
  #version = 0;
  #revision = 0;
  #capacity: number;
  #limit: number;
  #label: string;
  #usage: number;
  #grids: (GridState | undefined)[] = [];
  #alloc: RangeAllocator;
  #dirtyRecords = new Set<number>();
  #headerDirty = true;
  #disposed = false;

  /**
   * Use {@link VDBGridPool.create}.
   *
   * @param device The host's device.
   * @param buffer The pool buffer.
   * @param owns True when the pool created `buffer` and frees it.
   * @param capacity Grid table capacity in grids.
   * @param limit Upper bound on the buffer size in bytes.
   * @param label Debug label of the buffers.
   * @param usage Extra `GPUBufferUsage` flags of the buffer.
   */
  private constructor(device: GPUDevice, buffer: GPUBuffer, owns: boolean, capacity: number, limit: number, label: string, usage: number) {
    this.device = device;
    this.#buffer = buffer;
    this.#ownsBuffer = owns;
    this.#capacity = capacity;
    this.#limit = limit;
    this.#label = label;
    this.#usage = usage;
    this.#alloc = new RangeAllocator(dataStartOffset(capacity), buffer.size);
  }

  /**
   * Creates a pool on `device`. Throws `VDB_UNSUPPORTED_DEVICE` (see
   * {@link unsupportedReason}) or `VDB_BAD_OPTION`. The returned pool is
   * bindable at once (an empty table).
   *
   * @param device The host's `GPUDevice`.
   * @param options Initial size, table capacity, limits, label, usage and an optional host buffer.
   */
  static async create(device: GPUDevice, options: VDBGridPoolOptions = {}): Promise<VDBGridPool> {
    const reason = unsupportedReason(device);
    if (reason !== null) throw new VDBError("VDB_UNSUPPORTED_DEVICE", reason);
    watchDevice(device);

    const capacity = checkCount("maxGrids", options.maxGrids ?? DEFAULT_MAX_GRIDS, 1);
    const usage = checkCount("usage", options.usage ?? 0, 0);
    const label = options.label ?? "vdb pool";
    const deviceLimit = deviceByteLimit(device);
    let limit = deviceLimit;
    if (options.maxBytes !== undefined) limit = Math.min(limit, floor32(checkCount("maxBytes", options.maxBytes, 1)));
    const tableEnd = dataStartOffset(capacity);

    let pool: VDBGridPool;
    if (options.buffer !== undefined) {
      if (options.initialBytes !== undefined) badOption("initialBytes and buffer are exclusive; the host buffer's size is the pool size.");
      const host = options.buffer;
      const need = BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST;
      if ((host.usage & need) !== need) {
        badOption("The host buffer needs GPUBufferUsage.STORAGE | COPY_DST. Create it with those flags.");
      }
      if (host.size % 4 !== 0) badOption(`The host buffer is ${host.size} bytes; storage buffers must be a multiple of 4.`);
      if (host.size > deviceLimit) {
        badOption(
          `The host buffer is ${host.size} bytes; this device binds at most ${deviceLimit}. ` +
            `Use a smaller buffer or request the device with VDBGridPool.requiredLimits(adapter).`,
        );
      }
      if (host.size < tableEnd) {
        badOption(`The host buffer is ${host.size} bytes; a ${capacity}-grid table needs ${tableEnd}. Lower maxGrids or enlarge it.`);
      }
      pool = new VDBGridPool(device, host, false, capacity, floor32(host.size), label, usage);
      // A host buffer may hold old data: clear the header and table.
      pool.#writeTable();
    } else {
      if (tableEnd > limit) {
        badOption(`A ${capacity}-grid table needs ${tableEnd} bytes; the pool limit is ${limit}. Lower maxGrids.`);
      }
      const initial = checkCount("initialBytes", options.initialBytes ?? DEFAULT_INITIAL_BYTES, 1);
      const size = Math.min(limit, Math.max(align32(initial), tableEnd));
      device.pushErrorScope("out-of-memory");
      const buffer = device.createBuffer({ label: `${label} (v0)`, size, usage: POOL_BUFFER_USAGE | usage });
      const oom = await device.popErrorScope();
      if (oom) {
        buffer.destroy();
        throw new VDBError("VDB_POOL_LIMIT", `The device could not allocate ${size} bytes (${oom.message}). Lower initialBytes.`);
      }
      pool = new VDBGridPool(device, buffer, true, capacity, limit, label, usage);
      pool.#writeHeader();
    }
    return pool;
  }

  /** True when `device` can hold a pool. */
  static isSupported(device: GPUDevice): boolean {
    return unsupportedReason(device) === null;
  }

  /** Why `device` can't hold a pool, or `null`. */
  static unsupportedReason(device: GPUDevice): string | null {
    return unsupportedReason(device);
  }

  /** Limits for `adapter.requestDevice({ requiredLimits })` so the pool can grow to the adapter's maxima. */
  static requiredLimits(adapter: GPUAdapter): Record<string, number> {
    return requiredLimits(adapter);
  }

  /**
   * The pool's buffer. Replaced by a growth in `update`, or by the rollback of
   * a growth whose allocation failed (asynchronously; see `settled`); `version`
   * changes each time.
   */
  get buffer(): GPUBuffer {
    return this.#buffer;
  }

  /**
   * +1 whenever `buffer` changes object: rebuild bind groups. Monotonic: a
   * failed growth's rollback to the previous buffer is a new version too.
   *
   * Read it after `update(encoder)`: call `update(encoder)` before building
   * or reusing bind groups this frame. A bind group built before `update()`
   * keeps the previous (still-alive) buffer and silently misses new grids.
   */
  get version(): number {
    return this.#version;
  }

  /** +1 on every content change (add, replace, remove, setTransform, setMajorant). */
  get revision(): number {
    return this.#revision;
  }

  /** {@link VDB_GPU_LAYOUT_VERSION}. */
  get layoutVersion(): number {
    return VDB_GPU_LAYOUT_VERSION;
  }

  /** Size of `buffer` in bytes. */
  get byteLength(): number {
    return this.#buffer.size;
  }

  /** Allocation high-water mark in bytes (header word 6). */
  get usedBytes(): number {
    return this.#alloc.highWater;
  }

  /** Bytes in replaced buffers still alive: awaiting growth confirmation or the queue (see `update`). */
  get retiredBytes(): number {
    let n = 0;
    for (const r of this.#retired) n += r.buffer.size;
    for (const g of this.#growths) n += g.from.size;
    return n;
  }

  /**
   * The error from the last growth whose allocation failed (`VDB_POOL_LIMIT`;
   * the pool rolled back to its previous buffer), or `null`. Cleared when a
   * later growth is confirmed.
   */
  get lastError(): VDBError | null {
    return this.#lastError;
  }

  /**
   * Resolves once every growth started so far is confirmed by the device;
   * rejects with the `VDB_POOL_LIMIT` error if one failed and the pool rolled
   * back (see {@link lastError}). Resolves at once when no growth is pending.
   */
  settled(): Promise<void> {
    this.#checkAlive();
    return Promise.all(this.#growths.map((g) => g.done)).then((errors) => {
      const error = errors.find((e) => e !== null);
      if (error) throw error;
    });
  }

  /** Records in the grid table. */
  get gridCapacity(): number {
    return this.#capacity;
  }

  /** Highest live id + 1 (header word 3). */
  get gridCount(): number {
    for (let id = this.#grids.length - 1; id >= 0; id--) if (this.#grids[id]) return id + 1;
    return 0;
  }

  /**
   * Grid types of the live grids now, for `vdbCheckModule` and
   * `vdbModuleWgsl({ types })`. It does not know about grids the host adds
   * later: generate the module for every type the host may add (see `add`).
   */
  get types(): ReadonlySet<VDBGridTypeName> {
    const out = new Set<VDBGridTypeName>();
    for (const g of this.#grids) if (g) out.add(g.header.gridType);
    return out;
  }

  /**
   * Stages a grid image and returns its id at once. The image is validated
   * here (`VDB_BAD_IMAGE`, `VDB_UNSUPPORTED_GRID_TYPE`) and written by the next
   * `update`. Dead ids are reused, lowest first.
   *
   * The shader reading this pool decodes only the types its module was
   * generated for; a grid of any other type reads as 0. A module must
   * therefore be generated for every type the host may add (its known set,
   * or `VDB_ALL_SCALAR_TYPES`), or regenerated and its pipeline rebuilt when
   * a new type arrives. Call `vdbCheckModule(pool, options)` after `add` to
   * catch a mismatch (`VDB_TYPE_NOT_COMPILED`).
   *
   * @param image A NanoVDB grid image (one grid) as 32-bit words.
   * @param options Reserved capacity, transform, majorant, density bound and label.
   */
  add(image: Uint32Array, options: VDBAddOptions = {}): number {
    this.#checkAlive();
    const header = readImageHeader(image);
    const capacity = align32(Math.max(header.byteLength, checkCount("capacity", options.capacity ?? 0, 0)));
    const transform = copyTransform(options.transform);
    const affines = composeGridAffines(header, transform);
    const valueMaxOverride = checkValueMax(options.valueMax);
    const majorant = options.majorant ? validateMajorant(options.majorant) : null;

    let id = this.#grids.findIndex((g) => g === undefined);
    if (id < 0) id = this.#grids.length;
    if (!this.#ownsBuffer && id >= this.#capacity) {
      throw new VDBError(
        "VDB_POOL_FULL",
        `The host buffer's table holds ${this.#capacity} grids and all are live. Remove a grid, or create the pool with a larger maxGrids.`,
      );
    }
    this.#checkCanFit(Math.max(this.#grids.length, id + 1), majorant ? [capacity, majorant.bytes] : [capacity], "image");
    const grid: GridState = {
      header,
      label: options.label,
      capacity,
      image: { range: null, resident: false },
      pendingImage: image,
      retainedImage: null,
      imageSerial: 0,
      imageRevision: 0,
      transform,
      affines,
      valueMaxOverride,
      majorant,
    };
    // Place now when it fits; a host buffer must fit now (it never grows).
    const tableFits = id < this.#capacity && this.#grids.length <= this.#capacity;
    this.#place(grid.image, capacity, "image", tableFits);
    if (majorant) {
      try {
        this.#place(majorant, majorant.bytes, "majorant", tableFits);
      } catch (e) {
        if (grid.image.range) this.#alloc.free(grid.image.range.offset);
        throw e;
      }
    }
    this.#grids[id] = grid;
    this.#touch(id);
    grid.imageRevision = this.#revision;
    return id;
  }

  /**
   * Stages a new image for grid `id`. It is written in place when it fits the
   * grid's reserved range (no `version` change: the sequence-player path), else
   * moved. A brick majorant set on the grid is dropped (it described the old
   * image); call `setMajorant` again. Transform, label and `valueMax` are kept.
   *
   * @param id The grid's pool id.
   * @param image The new grid image, as for `add`.
   */
  replace(id: number, image: Uint32Array): void {
    this.#checkAlive();
    const grid = this.#grid(id);
    const header = readImageHeader(image);
    if (align32(header.byteLength) > grid.capacity) {
      const capacity = align32(header.byteLength);
      this.#checkCanFit(this.#grids.length, [capacity], "image"); // replace drops the majorant
      const next: RangeState = { range: null, resident: false };
      this.#replaceRange(grid.image.range, () => this.#place(next, capacity, "image"));
      grid.image = next;
      grid.capacity = capacity;
    }
    grid.header = header;
    grid.pendingImage = image;
    grid.imageSerial++;
    grid.affines = composeGridAffines(header, grid.transform);
    this.#dropMajorant(grid);
    this.#touch(id);
    grid.imageRevision = this.#revision;
  }

  /** Frees grid `id`'s ranges and clears its record; the id may be reused by the next `add`. */
  remove(id: number): void {
    this.#checkAlive();
    const grid = this.#grid(id);
    if (grid.image.range) this.#alloc.free(grid.image.range.offset);
    this.#dropMajorant(grid);
    this.#grids[id] = undefined;
    while (this.#grids.length > 0 && this.#grids[this.#grids.length - 1] === undefined) this.#grids.pop();
    this.#touch(id);
  }

  /**
   * Sets grid `id`'s instance transform: a 4x4 column-major object-to-world
   * matrix (WebGPU / three convention), or `null` for the file's own map. The
   * pool composes `worldToIndex = map⁻¹ ∘ objectToWorld⁻¹` and `indexToWorld`
   * in f64 and rounds to f32. Only the record changes; the image is untouched.
   *
   * @param id The grid's pool id.
   * @param objectToWorld A 4x4 column-major matrix, or `null` for the file's own map.
   */
  setTransform(id: number, objectToWorld: VDBTransformInput): void {
    this.#checkAlive();
    const grid = this.#grid(id);
    const transform = copyTransform(objectToWorld);
    grid.affines = composeGridAffines(grid.header, transform);
    grid.transform = transform;
    this.#touch(id);
  }

  /**
   * Stores brick-majorant bounds for grid `id` in the pool (record words
   * 36–39, flag b2, brick log2 in b16–23), or clears them with `null`.
   *
   * @param id The grid's pool id.
   * @param majorant The brick bounds, or `null` to clear them.
   */
  setMajorant(id: number, majorant: VDBBrickMajorant | null): void {
    this.#checkAlive();
    const grid = this.#grid(id);
    if (majorant === null) {
      this.#dropMajorant(grid);
      this.#touch(id);
      return;
    }
    const next = validateMajorant(majorant);
    const current = grid.majorant;
    if (current?.range && align32(next.bytes) <= current.range.size) {
      next.range = current.range;
      next.resident = current.resident;
    } else {
      this.#checkCanFit(this.#grids.length, [grid.capacity, next.bytes], "majorant");
      this.#replaceRange(current?.range ?? null, () => this.#place(next, next.bytes, "majorant"));
    }
    grid.majorant = next;
    this.#touch(id);
  }

  /** Describes grid `id`. Throws `VDB_UNKNOWN_GRID` for a dead or out-of-range id. */
  info(id: number): VDBGridInfo {
    this.#checkAlive();
    const g = this.#grid(id);
    const h = g.header;
    const lo = h.indexBBoxMin;
    const hi = h.indexBBoxMax.map((v) => v + 1) as VDBVec3;
    const min: VDBVec3 = [Infinity, Infinity, Infinity];
    const max: VDBVec3 = [-Infinity, -Infinity, -Infinity];
    for (let c = 0; c < 8; c++) {
      const p = applyAffine(g.affines.indexToWorld, [c & 1 ? hi[0] : lo[0], c & 2 ? hi[1] : lo[1], c & 4 ? hi[2] : lo[2]]);
      for (let a = 0; a < 3; a++) {
        min[a] = Math.min(min[a]!, p[a]!);
        max[a] = Math.max(max[a]!, p[a]!);
      }
    }
    const m = g.majorant;
    return {
      id,
      type: h.gridType,
      gridTypeId: h.gridTypeId,
      gridClass: gridClassName(h.gridClass),
      gridClassId: h.gridClass,
      name: h.name,
      label: g.label,
      imageBytes: h.byteLength,
      capacity: g.capacity,
      imageOffset: g.image.range?.offset ?? null,
      resident: g.image.resident && g.pendingImage === null,
      imageRevision: g.imageRevision,
      hasStats: h.hasStats,
      hasBrickMajorant: m !== null,
      majorantOffset: m?.range?.offset ?? null,
      majorantDims: m ? [...m.dims] : null,
      instanceTransform: g.transform !== null,
      background: h.background,
      valueMax: this.#valueMax(g),
      rootMin: h.rootMin,
      rootMax: h.rootMax,
      voxelSize: [...h.voxelSize],
      indexBBox: { min: [...lo], max: [...h.indexBBoxMax] },
      worldBBox: { min, max },
      worldToIndex: g.affines.worldToIndex.slice(),
      indexToWorld: g.affines.indexToWorld.slice(),
    };
  }

  /**
   * Applies staged changes, recording any copies into `encoder`:
   * 1. asks the queue about buffers retired at least two updates ago (once
   *    their growth is confirmed); each is destroyed when
   *    `onSubmittedWorkDone` resolves;
   * 2. if the staged state doesn't fit, grows (or, when only fragmentation
   *    blocks a placement, compacts) into a new buffer, copying ranges that
   *    were resident before this call, and bumps `version`. The allocation is
   *    checked in error scopes; if it fails, the pool rolls back later (see
   *    {@link settled}). Planned before anything changes: when the staged
   *    grids together pass the byte limit, nothing is re-laid out;
   * 3. writes new images, majorants, dirty records and the header with
   *    `queue.writeBuffer`, for ranges placed in the current buffer (a grid
   *    still waiting for space reads as dead on the GPU).
   * Then, if step 2 hit the limit, throws `VDB_POOL_LIMIT` (buffer and
   * `version` unchanged; remove grids and update again).
   *
   * Submit `encoder` before the second `update` after this one (one update of
   * slack): a buffer this update retires stays alive until then.
   *
   * Call `update(encoder)` before building or reusing bind groups this frame,
   * then compare `version` with the version the bind group was built at. A
   * bind group built before `update()` keeps the previous (still-alive)
   * buffer and silently misses new grids: no validation error, just stale
   * data.
   */
  update(encoder: GPUCommandEncoder): void {
    this.#checkAlive();
    this.#updates++;
    this.#reapRetired();

    let limitError: VDBError | null = null;
    const needIds = this.#grids.length;
    const unplaced = this.#unplaced();
    if (needIds > this.#capacity || unplaced.length > 0) {
      if (needIds <= this.#capacity && this.#tryPlace(unplaced)) {
        // Placed in gaps freed since staging.
      } else {
        try {
          this.#relayout(encoder, needIds);
        } catch (e) {
          if (!(e instanceof VDBError) || e.code !== "VDB_POOL_LIMIT") throw e;
          limitError = e; // independent writes below still reach the current buffer
        }
      }
    }
    this.#writePending();
    if (limitError) throw limitError;
  }

  /** Convenience: creates an encoder, calls `update`, and submits it. */
  flush(): void {
    this.#checkAlive();
    const encoder = this.device.createCommandEncoder({ label: `${this.#label} flush` });
    this.update(encoder);
    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * The bind group entry for `slot`: `{ binding: slot.firstBinding, resource: { buffer } }`.
   * Takes the same `VDBSlot` as `vdbModuleWgsl({ slot })` and
   * `vdbModuleEntries(slot)`.
   *
   * It names the current `buffer`, so call `update(encoder)` before building
   * or reusing bind groups this frame, and rebuild when `version` changed. A
   * bind group built before `update()` keeps the previous (still-alive)
   * buffer and silently misses new grids.
   */
  bindGroupEntry(slot: VDBSlot): GPUBindGroupEntry {
    this.#checkAlive();
    checkCount("slot.firstBinding", slot?.firstBinding, 0);
    return { binding: slot.firstBinding, resource: { buffer: this.#buffer } };
  }

  /**
   * Destroys every buffer this pool created (current, retired, and those of an
   * unconfirmed growth) at once, never a host buffer or the device. Submit no
   * work recorded by `update` after this. Idempotent.
   */
  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    const owned = new Set<GPUBuffer>();
    for (const r of this.#retired) owned.add(r.buffer);
    for (const g of this.#growths) owned.add(g.from).add(g.to);
    if (this.#ownsBuffer) owned.add(this.#buffer);
    for (const b of owned) b.destroy();
    this.#retired = [];
    this.#growths = [];
    this.#grids = [];
    this.#dirtyRecords.clear();
  }

  // -------------------------------------------------------------------------

  #checkAlive(): void {
    if (this.#disposed) throw new VDBError("VDB_DISPOSED", "This VDBGridPool was disposed. Create a new pool.");
  }

  #grid(id: number): GridState {
    const g = Number.isInteger(id) ? this.#grids[id] : undefined;
    if (!g) {
      throw new VDBError(
        "VDB_UNKNOWN_GRID",
        `Grid id ${id} is not live in this pool (removed, or never returned by add). Use an id from add().`,
      );
    }
    return g;
  }

  #touch(id: number): void {
    this.#dirtyRecords.add(id);
    this.#headerDirty = true;
    this.#revision++;
  }

  #valueMax(g: GridState): number {
    if (g.valueMaxOverride !== undefined) return g.valueMaxOverride;
    // Without stats there is no bound in the image; Infinity tells the integrator to pass valueMax.
    return g.header.hasStats ? Math.max(g.header.rootMax, g.header.background) : Infinity;
  }

  /** Limit detail for `VDB_POOL_LIMIT` messages. */
  #limitDetail(): string {
    const { maxStorageBufferBindingSize, maxBufferSize } = this.device.limits;
    return (
      `device maxStorageBufferBindingSize ${maxStorageBufferBindingSize}, maxBufferSize ${maxBufferSize}` +
      `${this.#limit < deviceByteLimit(this.device) ? `, maxBytes ${this.#limit}` : ""}`
    );
  }

  /**
   * Throws `VDB_POOL_LIMIT` when ranges of `bytes` could not fit under the
   * limit even alone, beside a table of `ids` records (§1.5). Staged, such a
   * request would make every later `update` fail; reject it while the pool is
   * unchanged. Host buffers are checked by `#place` instead (`VDB_POOL_FULL`).
   */
  #checkCanFit(ids: number, bytes: number[], what: string): void {
    if (!this.#ownsBuffer) return;
    let capacity = this.#capacity;
    while (capacity < ids) capacity *= 2;
    const start = dataStartOffset(capacity);
    let needed = start;
    for (const b of bytes) needed += align32(b);
    if (needed <= this.#limit) return;
    throw new VDBError(
      "VDB_POOL_LIMIT",
      `This ${what} needs ${needed - start} bytes beside a ${capacity}-grid table (${start} bytes), ${needed} in all: ` +
        `more than the pool limit of ${this.#limit} bytes (${this.#limitDetail()}) even with no other grids. ` +
        `Use a smaller grid, or request the device with VDBGridPool.requiredLimits(adapter). The pool is unchanged.`,
    );
  }

  /** Frees `old`, runs `place` (which may reuse its space), and restores `old` if `place` throws. */
  #replaceRange(old: AllocatedRange | null, place: () => void): void {
    if (old) this.#alloc.free(old.offset);
    try {
      place();
    } catch (e) {
      if (old) this.#alloc.restore(old);
      throw e;
    }
  }

  /**
   * First-fit placement at staging time; a host buffer must fit now. In an
   * owned pool a range that doesn't fit (or any range while the table is
   * over capacity) stays unplaced and the next `update` re-lays out.
   */
  #place(state: RangeState, bytes: number, what: string, tableFits = this.#grids.length <= this.#capacity): void {
    const offset = tableFits ? this.#alloc.alloc(bytes) : null;
    if (offset !== null) {
      state.range = { offset, size: align32(bytes) };
      return;
    }
    if (!this.#ownsBuffer) {
      throw new VDBError(
        "VDB_POOL_FULL",
        `The host buffer (${this.#buffer.size} bytes, ${this.#alloc.allocatedBytes} allocated) has no free range of ` +
          `${align32(bytes)} bytes for this ${what}. Remove grids, or create the pool on a larger buffer (or without one, so it can grow).`,
      );
    }
    state.range = null; // placed by the next update
  }

  #dropMajorant(grid: GridState): void {
    const m = grid.majorant;
    if (m?.range) this.#alloc.free(m.range.offset);
    grid.majorant = null;
  }

  #unplaced(): { state: RangeState; bytes: number }[] {
    const out: { state: RangeState; bytes: number }[] = [];
    for (const g of this.#grids) {
      if (!g) continue;
      if (!g.image.range) out.push({ state: g.image, bytes: g.capacity });
      if (g.majorant && !g.majorant.range) out.push({ state: g.majorant, bytes: g.majorant.bytes });
    }
    return out;
  }

  /** Places every unplaced range first-fit, or changes nothing and returns false. */
  #tryPlace(unplaced: { state: RangeState; bytes: number }[]): boolean {
    const placed: number[] = [];
    for (const u of unplaced) {
      const offset = this.#alloc.alloc(u.bytes);
      if (offset === null) {
        for (const o of placed) this.#alloc.free(o);
        return false;
      }
      placed.push(offset);
    }
    unplaced.forEach((u, k) => (u.state.range = { offset: placed[k]!, size: align32(u.bytes) }));
    return true;
  }

  /** Grows or compacts into a new buffer (§1.4 step 2). Plans and checks before changing anything. */
  #relayout(encoder: GPUCommandEncoder, needIds: number): void {
    if (!this.#ownsBuffer) {
      // #place already guarantees host-buffer placements; reaching here is a pool bug.
      throw new VDBError("VDB_POOL_FULL", "The host buffer cannot be re-laid out. Remove grids or use a larger buffer.");
    }
    let capacity = this.#capacity;
    while (capacity < needIds) capacity *= 2;

    // Pack every live range in id order: image, then majorant.
    const ranges: { state: RangeState; bytes: number; copyBytes: number }[] = [];
    for (const g of this.#grids) {
      if (!g) continue;
      ranges.push({ state: g.image, bytes: g.capacity, copyBytes: g.pendingImage ? 0 : g.header.byteLength });
      if (g.majorant) ranges.push({ state: g.majorant, bytes: g.majorant.bytes, copyBytes: g.majorant.pending ? 0 : g.majorant.bytes });
    }
    const start = dataStartOffset(capacity);
    let needed = start;
    for (const r of ranges) needed += align32(r.bytes);

    const current = this.#buffer.size;
    const grows = capacity > this.#capacity || needed > current;
    const size = grows ? Math.min(this.#limit, Math.max(needed, 2 * current)) : current;
    if (needed > size) {
      throw new VDBError(
        "VDB_POOL_LIMIT",
        `The staged grids need a ${needed}-byte pool; the limit is ${this.#limit} bytes (${this.#limitDetail()}). ` +
          `Remove grids, or request the device with VDBGridPool.requiredLimits(adapter). The buffer is unchanged; ` +
          `staged changes that fit it were applied.`,
      );
    }

    // What `from` holds (clean, resident ranges), for a rollback if the allocation fails.
    const snapshot = new Map<GridState, GridSnapshot>();
    for (const g of this.#grids) {
      if (!g) continue;
      const m = g.majorant;
      snapshot.set(g, {
        imageSerial: g.imageSerial,
        image: g.image.range && g.image.resident && !g.pendingImage ? { ...g.image.range } : null,
        majorant: m,
        majorantRange: m?.range && m.resident && !m.pending ? { ...m.range } : null,
      });
    }

    // update() is synchronous: switch now, keep `from` alive, and settle when the scopes resolve.
    const device = this.device;
    device.pushErrorScope("out-of-memory");
    device.pushErrorScope("validation");
    const next = device.createBuffer({
      label: `${this.#label} (v${this.#version + 1})`,
      size,
      usage: POOL_BUFFER_USAGE | this.#usage,
    });
    const validation = device.popErrorScope();
    const oom = device.popErrorScope();
    const growth: Growth = {
      from: this.#buffer,
      to: next,
      capacity: this.#capacity,
      update: this.#updates,
      snapshot,
      done: Promise.resolve(null),
    };
    // A rejected pop (e.g. the host popped our scope) tells us nothing: treat it as success.
    const scopeError = Promise.all([oom, validation]).then(
      ([o, v]) => o ?? v,
      () => null,
    );
    const previous = this.#growths[this.#growths.length - 1]?.done ?? Promise.resolve(null);
    growth.done = Promise.all([scopeError, previous]).then(([e]) => this.#settleGrowth(growth, e));
    this.#growths.push(growth);

    const alloc = new RangeAllocator(start, size);
    for (const r of ranges) {
      const offset = alloc.alloc(r.bytes)!;
      const old = r.state.range;
      if (old && r.state.resident && r.copyBytes > 0) {
        encoder.copyBufferToBuffer(this.#buffer, old.offset, next, offset, r.copyBytes);
      }
      r.state.range = { offset, size: align32(r.bytes) };
      if (r.copyBytes === 0) r.state.resident = false;
    }

    this.#buffer = next;
    this.#alloc = alloc;
    this.#capacity = capacity;
    this.#version++;
    this.#writeTable();
  }

  /**
   * Runs when a growth's error scopes resolve, in growth order. Confirmed: its
   * old buffer is retired. Failed: roll back to that old buffer, voiding every
   * later growth (they copied from the failed buffer).
   */
  #settleGrowth(growth: Growth, scopeError: { message: string } | null): VDBError | null {
    // Disposed (dispose destroyed everything), or voided by an earlier failure.
    if (this.#disposed || this.#growths[0] !== growth) return null;
    if (!scopeError) {
      this.#growths.shift();
      this.#retired.push({ buffer: growth.from, since: growth.update, waiting: false });
      this.#lastError = null;
      if (this.#growths.length === 0) this.#releaseRetained();
      return null;
    }
    const error = new VDBError(
      "VDB_POOL_LIMIT",
      `The device could not allocate a ${growth.to.size}-byte pool buffer while growing (${scopeError.message}). ` +
        `The pool rolled back to its previous ${growth.from.size}-byte buffer as version ${this.#version + 1}; rebuild ` +
        `bind groups. Grids resident before the growth are intact; ones staged or written since are staged again. ` +
        `Remove grids or free GPU memory, then update again.`,
    );
    this.#rollBack(growth);
    this.#lastError = error;
    return error;
  }

  /** Returns to `growth.from` and what it holds; every other range is staged again. */
  #rollBack(growth: Growth): void {
    for (const g of this.#growths) g.to.destroy(); // the failed buffer and any chained after it
    this.#growths = [];
    const alloc = new RangeAllocator(dataStartOffset(growth.capacity), growth.from.size);
    for (const g of this.#grids) {
      if (!g) continue;
      const snap = growth.snapshot.get(g);
      // Same serial: no replace since the snapshot, so `from` holds the current image and nothing is pending.
      if (snap?.image && snap.imageSerial === g.imageSerial) {
        alloc.restore(snap.image);
        g.image.range = { ...snap.image };
        g.image.resident = true;
      } else {
        g.image.range = null;
        g.image.resident = false;
        g.pendingImage ??= g.retainedImage;
      }
      const m = g.majorant;
      if (!m) continue;
      if (snap?.majorantRange && snap.majorant === m) {
        alloc.restore(snap.majorantRange);
        m.range = { ...snap.majorantRange };
        m.resident = true;
      } else {
        m.range = null;
        m.resident = false;
        m.pending ??= m.retained;
      }
    }
    this.#releaseRetained();
    this.#buffer = growth.from;
    this.#alloc = alloc;
    this.#capacity = growth.capacity;
    this.#version++;
    this.#writeTable(); // `from`'s table is stale: restored grids are live, the rest read as dead until placed
  }

  #releaseRetained(): void {
    for (const g of this.#grids) {
      if (!g) continue;
      g.retainedImage = null;
      if (g.majorant) g.majorant.retained = null;
    }
  }

  /**
   * §1.4 step 1: for retired buffers that are confirmed and at least
   * {@link RETIRE_AFTER_UPDATES} updates old (so the encoder that copied out of
   * them has been submitted), asks the queue once and destroys them when that
   * work is done.
   */
  #reapRetired(): void {
    const ready = this.#retired.filter((r) => !r.waiting && this.#updates - r.since >= RETIRE_AFTER_UPDATES);
    if (ready.length === 0) return;
    for (const r of ready) r.waiting = true;
    const destroy = (): void => {
      if (this.#disposed) return; // dispose destroyed them
      this.#retired = this.#retired.filter((r) => !ready.includes(r));
      for (const r of ready) r.buffer.destroy();
    };
    this.device.queue.onSubmittedWorkDone().then(destroy, destroy);
  }

  /** Every range of `g` is placed. Until then its record stays dirty. */
  #complete(g: GridState): boolean {
    return g.image.range !== null && (g.majorant === null || g.majorant.range !== null);
  }

  /** The header's grid count: highest live id + 1, clamped to the table (ids past it can't be read). */
  #headerGridCount(): number {
    return Math.min(this.gridCount, this.#capacity);
  }

  /** Writes the header and every record (after create on a host buffer, a relayout, or a rollback). */
  #writeTable(): void {
    const words = new Uint32Array(VDB_POOL_HEADER_WORDS + this.#capacity * VDB_POOL_RECORD_WORDS);
    const dirty = new Set<number>();
    this.#grids.forEach((g, id) => {
      if (!g) return;
      if (id < this.#capacity && g.image.range) {
        encodeRecord(this.#record(g), words, VDB_POOL_HEADER_WORDS + id * VDB_POOL_RECORD_WORDS);
      }
      if (!this.#complete(g)) dirty.add(id);
    });
    encodeHeader({ gridCapacity: this.#capacity, gridCount: this.#headerGridCount(), usedBytes: this.usedBytes }, words, 0);
    write(this.device.queue, this.#buffer, 0, words);
    this.#dirtyRecords = dirty;
    this.#headerDirty = false;
  }

  #writeHeader(): void {
    const words = encodeHeader({ gridCapacity: this.#capacity, gridCount: this.#headerGridCount(), usedBytes: this.usedBytes });
    write(this.device.queue, this.#buffer, 0, words);
    this.#headerDirty = false;
  }

  /**
   * §1.4 step 3: images, majorants, dirty records, header. Only ranges placed
   * in the current buffer are written; a grid whose image is unplaced gets a
   * dead record, and its id stays dirty until every range of it is placed.
   */
  #writePending(): void {
    const queue = this.device.queue;
    const keep = this.#growths.length > 0; // a rollback may have to stage these again
    for (const g of this.#grids) {
      if (!g) continue;
      if (g.pendingImage && g.image.range) {
        write(queue, this.#buffer, g.image.range.offset, g.pendingImage);
        if (keep) g.retainedImage = g.pendingImage;
        g.pendingImage = null;
        g.image.resident = true;
      }
      const m = g.majorant;
      if (m?.pending && m.range) {
        write(queue, this.#buffer, m.range.offset, m.pending);
        if (keep) m.retained = m.pending;
        m.pending = null;
        m.resident = true;
      }
    }
    const dirty = new Set<number>();
    for (const id of [...this.#dirtyRecords].sort((a, b) => a - b)) {
      const g = this.#grids[id];
      if (g && !this.#complete(g)) dirty.add(id);
      if (id >= this.#capacity) continue; // past the table end: the relayout that grows the table writes it
      const words = new Uint32Array(VDB_POOL_RECORD_WORDS);
      if (g?.image.range) encodeRecord(this.#record(g), words, 0);
      write(queue, this.#buffer, recordByteOffset(id), words);
    }
    this.#dirtyRecords = dirty;
    if (this.#headerDirty) this.#writeHeader();
  }

  /** The record for `g`, whose image is placed. A majorant is recorded only once it is placed. */
  #record(g: GridState): VDBGridRecord {
    const h = g.header;
    const m = g.majorant?.range ? g.majorant : null;
    return {
      imageOffset: g.image.range!.offset,
      imageBytes: h.byteLength,
      gridType: h.gridTypeId,
      live: true,
      hasStats: h.hasStats,
      hasBrickMajorant: m !== null,
      instanceTransform: g.transform !== null,
      gridClass: h.gridClass,
      brickLog2: m?.brickLog2 ?? 0,
      worldToIndex: Array.from(g.affines.worldToIndex),
      indexToWorld: Array.from(g.affines.indexToWorld),
      indexBBoxMin: h.indexBBoxMin,
      indexBBoxMax: h.indexBBoxMax,
      background: h.background,
      valueMax: this.#valueMax(g),
      majorantOffset: m?.range?.offset ?? 0,
      majorantDims: m ? m.dims : [0, 0, 0],
    };
  }
}
