/**
 * `VDBGridPool`: many NanoVDB grids in one `GPUBuffer` behind a grid table
 * (docs/design/K1-core.md §1, §4; D8, D10).
 *
 * Model: CPU calls (`add`, `replace`, `remove`, `setTransform`, `setMajorant`)
 * stage changes and return at once; `update(encoder)` applies them. Growth and
 * compaction happen only in `update`, by a new buffer plus copies of the ranges
 * that were resident before that update; `version` bumps then and only then.
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
  id: number;
  type: VDBGridTypeName;
  /** NanoVDB `GridType` id. */
  gridTypeId: number;
  /** NanoVDB `GridClass` name, e.g. `"FogVolume"`. */
  gridClass: string;
  gridClassId: number;
  /** Grid name from the image header. */
  name: string;
  label: string | undefined;
  imageBytes: number;
  /** Bytes reserved for the image (≥ `imageBytes`, 32 B multiple). */
  capacity: number;
  /** Byte offset of the image in the pool, or `null` until an `update` places it. */
  imageOffset: number | null;
  /** True once the image bytes are on the GPU (after the `update` that wrote them). */
  resident: boolean;
  hasStats: boolean;
  hasBrickMajorant: boolean;
  majorantOffset: number | null;
  majorantDims: VDBVec3 | null;
  instanceTransform: boolean;
  background: number;
  valueMax: number;
  voxelSize: VDBVec3;
  indexBBox: { min: VDBVec3; max: VDBVec3 };
  /** World-space AABB of the index bbox's voxels (max corner + 1 voxel), through the record's f64 transform. */
  worldBBox: { min: VDBVec3; max: VDBVec3 };
  /** f64 affines before f32 rounding, 3x4 row-major. */
  worldToIndex: Float64Array;
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
}

interface GridState {
  header: VDBImageHeader;
  label: string | undefined;
  capacity: number;
  image: RangeState;
  pendingImage: Uint32Array | null;
  transform: Float64Array | null;
  affines: GridAffines;
  valueMaxOverride: number | undefined;
  majorant: MajorantState | null;
}

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
  return { brickLog2: m.brickLog2, dims, bytes, pending: m.data, range: null, resident: false };
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
 * next `update` writes them; don't mutate them before then.
 */
export class VDBGridPool {
  /** The device the pool allocates on. */
  readonly device: GPUDevice;

  #buffer: GPUBuffer;
  #ownsBuffer: boolean;
  #retired: GPUBuffer[] = [];
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

  /** The pool's buffer. Replaced only by `update`; then `version` changes. */
  get buffer(): GPUBuffer {
    return this.#buffer;
  }

  /** +1 whenever `buffer` is a new object: rebuild bind groups. */
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

  /** Bytes in retired buffers awaiting destruction at the next `update`. */
  get retiredBytes(): number {
    return this.#retired.reduce((n, b) => n + b.size, 0);
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

  /** Grid types of the live grids, for `vdbModuleWgsl({ types })`. */
  get types(): ReadonlySet<VDBGridTypeName> {
    const out = new Set<VDBGridTypeName>();
    for (const g of this.#grids) if (g) out.add(g.header.gridType);
    return out;
  }

  /**
   * Stages a grid image and returns its id at once. The image is validated
   * here (`VDB_BAD_IMAGE`, `VDB_UNSUPPORTED_GRID_TYPE`) and written by the next
   * `update`. Dead ids are reused, lowest first.
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
    const grid: GridState = {
      header,
      label: options.label,
      capacity,
      image: { range: null, resident: false },
      pendingImage: image,
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
    return id;
  }

  /**
   * Stages a new image for grid `id`. It is written in place when it fits the
   * grid's reserved range (no `version` change: the sequence-player path), else
   * moved. A brick majorant set on the grid is dropped (it described the old
   * image); call `setMajorant` again. Transform, label and `valueMax` are kept.
   */
  replace(id: number, image: Uint32Array): void {
    this.#checkAlive();
    const grid = this.#grid(id);
    const header = readImageHeader(image);
    if (align32(header.byteLength) > grid.capacity) {
      const capacity = align32(header.byteLength);
      const next: RangeState = { range: null, resident: false };
      this.#replaceRange(grid.image.range, () => this.#place(next, capacity, "image"));
      grid.image = next;
      grid.capacity = capacity;
    }
    grid.header = header;
    grid.pendingImage = image;
    grid.affines = composeGridAffines(header, grid.transform);
    this.#dropMajorant(grid);
    this.#touch(id);
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
      hasStats: h.hasStats,
      hasBrickMajorant: m !== null,
      majorantOffset: m?.range?.offset ?? null,
      majorantDims: m ? [...m.dims] : null,
      instanceTransform: g.transform !== null,
      background: h.background,
      valueMax: this.#valueMax(g),
      voxelSize: [...h.voxelSize],
      indexBBox: { min: [...lo], max: [...h.indexBBoxMax] },
      worldBBox: { min, max },
      worldToIndex: g.affines.worldToIndex.slice(),
      indexToWorld: g.affines.indexToWorld.slice(),
    };
  }

  /**
   * Applies staged changes, recording any copies into `encoder`:
   * 1. destroys buffers retired by the previous `update`;
   * 2. if the staged state doesn't fit, grows (or, when only fragmentation
   *    blocks a placement, compacts) into a new buffer, copying ranges that
   *    were resident before this call, and bumps `version`. Planned before
   *    anything changes: on `VDB_POOL_LIMIT` the pool is left as it was;
   * 3. writes new images, majorants, dirty records and the header with
   *    `queue.writeBuffer`.
   * Submit `encoder` before the next `update`.
   */
  update(encoder: GPUCommandEncoder): void {
    this.#checkAlive();
    for (const b of this.#retired) b.destroy();
    this.#retired = [];

    const needIds = this.#grids.length;
    const unplaced = this.#unplaced();
    if (needIds > this.#capacity || unplaced.length > 0) {
      if (needIds <= this.#capacity && this.#tryPlace(unplaced)) {
        // Placed in gaps freed since staging.
      } else {
        this.#relayout(encoder, needIds);
      }
    }
    this.#writePending();
  }

  /** Convenience: creates an encoder, calls `update`, and submits it. */
  flush(): void {
    this.#checkAlive();
    const encoder = this.device.createCommandEncoder({ label: `${this.#label} flush` });
    this.update(encoder);
    this.device.queue.submit([encoder.finish()]);
  }

  /** The bind group entry for `slot`: `{ binding: slot.firstBinding, resource: { buffer } }`. */
  bindGroupEntry(slot: VDBSlot): GPUBindGroupEntry {
    this.#checkAlive();
    checkCount("slot.firstBinding", slot?.firstBinding, 0);
    return { binding: slot.firstBinding, resource: { buffer: this.#buffer } };
  }

  /** Destroys the buffers this pool created (current and retired), never a host buffer or the device. Idempotent. */
  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    for (const b of this.#retired) b.destroy();
    this.#retired = [];
    if (this.#ownsBuffer) this.#buffer.destroy();
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
      const { maxStorageBufferBindingSize, maxBufferSize } = this.device.limits;
      throw new VDBError(
        "VDB_POOL_LIMIT",
        `The staged grids need a ${needed}-byte pool; the limit is ${this.#limit} bytes (device ` +
          `maxStorageBufferBindingSize ${maxStorageBufferBindingSize}, maxBufferSize ${maxBufferSize}` +
          `${this.#limit < deviceByteLimit(this.device) ? `, maxBytes ${this.#limit}` : ""}). Remove grids, or request ` +
          `the device with VDBGridPool.requiredLimits(adapter). The pool is unchanged.`,
      );
    }

    const next = this.device.createBuffer({
      label: `${this.#label} (v${this.#version + 1})`,
      size,
      usage: POOL_BUFFER_USAGE | this.#usage,
    });
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

    this.#retired.push(this.#buffer);
    this.#buffer = next;
    this.#alloc = alloc;
    this.#capacity = capacity;
    this.#version++;
    this.#writeTable();
  }

  /** Writes the header and every record (after create on a host buffer, and after a relayout). */
  #writeTable(): void {
    const words = new Uint32Array(VDB_POOL_HEADER_WORDS + this.#capacity * VDB_POOL_RECORD_WORDS);
    this.#grids.forEach((g, id) => {
      if (g && g.image.range) encodeRecord(this.#record(g), words, VDB_POOL_HEADER_WORDS + id * VDB_POOL_RECORD_WORDS);
    });
    encodeHeader({ gridCapacity: this.#capacity, gridCount: this.gridCount, usedBytes: this.usedBytes }, words, 0);
    write(this.device.queue, this.#buffer, 0, words);
    this.#dirtyRecords.clear();
    this.#headerDirty = false;
  }

  #writeHeader(): void {
    const words = encodeHeader({ gridCapacity: this.#capacity, gridCount: this.gridCount, usedBytes: this.usedBytes });
    write(this.device.queue, this.#buffer, 0, words);
    this.#headerDirty = false;
  }

  /** §1.4 step 3: images, majorants, dirty records, header. */
  #writePending(): void {
    const queue = this.device.queue;
    for (const g of this.#grids) {
      if (!g) continue;
      if (g.pendingImage) {
        write(queue, this.#buffer, g.image.range!.offset, g.pendingImage);
        g.pendingImage = null;
        g.image.resident = true;
      }
      const m = g.majorant;
      if (m?.pending) {
        write(queue, this.#buffer, m.range!.offset, m.pending);
        m.pending = null;
        m.resident = true;
      }
    }
    for (const id of [...this.#dirtyRecords].sort((a, b) => a - b)) {
      if (id >= this.#capacity) continue; // past a shrunk table end: nothing to clear
      const g = this.#grids[id];
      const words = new Uint32Array(VDB_POOL_RECORD_WORDS);
      if (g) encodeRecord(this.#record(g), words, 0);
      write(queue, this.#buffer, recordByteOffset(id), words);
    }
    this.#dirtyRecords.clear();
    if (this.#headerDirty) this.#writeHeader();
  }

  #record(g: GridState): VDBGridRecord {
    const h = g.header;
    const m = g.majorant;
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
