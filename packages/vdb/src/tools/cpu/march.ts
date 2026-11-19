/**
 * CPU twins (D11) of the `march` WGSL profile (docs/dev/design/K1-core.md §3.2,
 * task K1.4b): the vendored HDDA (`vdbHddaInit/Update/Step`), the node size at
 * a voxel (`vdbGetDim`), node statistics (the fork's
 * `{root,upper,lower,leaf}_get_{min,max,ave,stddev}_float`, with the K1.4b Fp
 * leaf-stats fix), `vdbNodeMax`, `vdbRayClip`, and `hddaWalk`, the canonical
 * walk the GPU probe kernel runs with the same calls. Also the opt-in helpers
 * (K1.7a): `vdbCellBoundTrilinear`, `vdbHddaCellEnd`, `vdbRayClipSupport`.
 *
 * Float arithmetic: every f32 operation of the WGSL is rounded here with
 * `Math.fround`, in the WGSL's evaluation order. GPUs may fuse a multiply-add
 * or use a less exact division (WGSL allows 2.5 ULP), so GPU parity compares
 * `t` with a tolerance, not bit for bit.
 *
 * What the node stats guarantee (NanoVDB `tools/GridStats.h`): a node's
 * min/max/ave/stddev cover the ACTIVE voxels and ACTIVE tiles of its subtree.
 * They say nothing about inactive voxels, inactive tiles or the background.
 * A trilinear sample at `p` blends voxels `floor(p)` and `floor(p) + 1`, so a
 * sample inside a leaf's cell can read a neighbouring leaf, a tile, or an
 * inactive voxel: the leaf's max does not bound it. The bound for a sample is
 * the max over its 2x2x2 support of each tap's own bound (leaf max for a leaf
 * voxel, the value for a tile or the background), and that is a bound only if
 * every inactive leaf voxel is <= its leaf's max. `test/tools/cpu-march.test.ts`
 * checks exactly these statements per fixture.
 *
 * Fp leaf stats (NanoVDB.h `LeafFnBase`, OpenVDB a532de55): uint16 codes at
 * leaf bytes 88/90/92/94; min/max/ave = code * quantum + minimum, stddev =
 * code * quantum, with the leaf's own mMinimum (byte 80) and mQuantum (84).
 */
import { VDBToolsError } from "../errors.js";
import { defineBigInt, defineNumber, gridTypeConstantsForId, gridTypeIds, type GridTypeConstants } from "./stride-tables.js";

export type Vec3 = [number, number, number];

const f = Math.fround;

const GRID_OFF_MAGIC = defineNumber("PNANOVDB_GRID_OFF_MAGIC");
const GRID_OFF_FLAGS = defineNumber("PNANOVDB_GRID_OFF_FLAGS");
const GRID_OFF_GRID_TYPE = defineNumber("PNANOVDB_GRID_OFF_GRID_TYPE");
const GRID_SIZE = defineNumber("PNANOVDB_GRID_SIZE");
const MAGIC_GRID = defineBigInt("PNANOVDB_MAGIC_GRID");
const TREE_OFF_NODE_OFFSET_ROOT = defineNumber("PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT");
const ROOT_OFF_BBOX_MIN = defineNumber("PNANOVDB_ROOT_OFF_BBOX_MIN");
const ROOT_OFF_BBOX_MAX = defineNumber("PNANOVDB_ROOT_OFF_BBOX_MAX");
const ROOT_OFF_TABLE_SIZE = defineNumber("PNANOVDB_ROOT_OFF_TABLE_SIZE");
const ROOT_TILE_OFF_KEY = defineNumber("PNANOVDB_ROOT_TILE_OFF_KEY");
const ROOT_TILE_OFF_CHILD = defineNumber("PNANOVDB_ROOT_TILE_OFF_CHILD");
const ROOT_TILE_OFF_STATE = defineNumber("PNANOVDB_ROOT_TILE_OFF_STATE");
const UPPER_OFF_VALUE_MASK = defineNumber("PNANOVDB_UPPER_OFF_VALUE_MASK");
const UPPER_OFF_CHILD_MASK = defineNumber("PNANOVDB_UPPER_OFF_CHILD_MASK");
const LOWER_OFF_VALUE_MASK = defineNumber("PNANOVDB_LOWER_OFF_VALUE_MASK");
const LOWER_OFF_CHILD_MASK = defineNumber("PNANOVDB_LOWER_OFF_CHILD_MASK");
const LEAF_OFF_VALUE_MASK = defineNumber("PNANOVDB_LEAF_OFF_VALUE_MASK");
const LEAF_TABLE_NEG_OFF_BBOX_DIF_AND_FLAGS = defineNumber("PNANOVDB_LEAF_TABLE_NEG_OFF_BBOX_DIF_AND_FLAGS");
const LEAF_TABLE_NEG_OFF_MINIMUM = defineNumber("PNANOVDB_LEAF_TABLE_NEG_OFF_MINIMUM");
const LEAF_TABLE_NEG_OFF_QUANTUM = defineNumber("PNANOVDB_LEAF_TABLE_NEG_OFF_QUANTUM");
/** PNANOVDB_GRID_FLAGS_HAS_MIN_MAX (1 << 2); not in the extracted defines. */
const GRID_FLAGS_HAS_MIN_MAX = 4;
/** PNANOVDB_HDDA_FLOAT_MAX. */
const HDDA_FLOAT_MAX = f(1e38);
const FORWARD_SCALE = f(0.999999);
const FORWARD_BIAS = f(1.0e-6);

const FP_TYPES = new Set([gridTypeIds.FP4, gridTypeIds.FP8, gridTypeIds.FP16, gridTypeIds.FPN]);
const SUPPORTED_TYPES = new Set([gridTypeIds.FLOAT, ...FP_TYPES]);

// ---------------------------------------------------------------------------
// Grid access

interface Grid {
  readonly words: Uint32Array;
  readonly f32: Float32Array;
  readonly gridType: number;
  readonly gt: GridTypeConstants;
  readonly fp: boolean;
  readonly hasStats: boolean;
  readonly root: number;
}

/** Per-image header cache, keyed by the array: images are treated as immutable once read. */
const grids = new WeakMap<Uint32Array, Grid>();

function u32(words: Uint32Array, address: number): number {
  return words[address >>> 2]!;
}

function grid(words: Uint32Array): Grid {
  let g = grids.get(words);
  if (g) return g;
  const magic = (BigInt(u32(words, GRID_OFF_MAGIC + 4)) << 32n) | BigInt(u32(words, GRID_OFF_MAGIC));
  if (magic !== MAGIC_GRID) throw new VDBToolsError("VDB_BAD_IMAGE", "march: gridWords does not start with a NanoVDB GridData block");
  const gridType = u32(words, GRID_OFF_GRID_TYPE);
  if (!SUPPORTED_TYPES.has(gridType)) throw new VDBToolsError("VDB_UNSUPPORTED_GRID_TYPE", `march: grid type ${gridType} is not Float/Fp4/Fp8/Fp16/FpN`);
  const root = GRID_SIZE + u32(words, GRID_SIZE + TREE_OFF_NODE_OFFSET_ROOT);
  g = {
    words,
    f32: new Float32Array(words.buffer, words.byteOffset, words.length),
    gridType,
    gt: gridTypeConstantsForId(gridType),
    fp: FP_TYPES.has(gridType),
    hasStats: (u32(words, GRID_OFF_FLAGS) & GRID_FLAGS_HAS_MIN_MAX) !== 0,
    root,
  };
  grids.set(words, g);
  return g;
}

function readF32(g: Grid, address: number): number {
  return g.f32[address >>> 2]!;
}

/** pnanovdb_read_uint16 (fork, K1.4b): the u16 at a 2-byte-aligned address. */
function readU16(g: Grid, address: number): number {
  return (u32(g.words, address & ~3) >>> ((address & 2) << 3)) & 0xffff;
}

function maskBit(g: Grid, base: number, n: number): boolean {
  return ((u32(g.words, base + 4 * (n >>> 5)) >>> (n & 31)) & 1) !== 0;
}

/** Child offset (low word of the i64; the fork's 32-bit addressing). */
function childOffset(g: Grid, address: number): number {
  return u32(g.words, address) | 0;
}

function coordKey(x: number, y: number, z: number): bigint {
  return BigInt(z >>> 12) | (BigInt(y >>> 12) << 21n) | (BigInt(x >>> 12) << 42n);
}

/** Where `ijk` resolves and the nodes on its path (0 = none). */
interface Path {
  /** 0 leaf voxel, 1 lower tile, 2 upper tile, 3 root tile, 4 background. */
  level: 0 | 1 | 2 | 3 | 4;
  upper: number;
  lower: number;
  leaf: number;
  /** Value slot of a tile or the background (levels 1-4). */
  valueAddress: number;
  active: boolean;
}

function locate(g: Grid, [x, y, z]: Vec3): Path {
  const gt = g.gt;
  const key = coordKey(x, y, z);
  const count = u32(g.words, g.root + ROOT_OFF_TABLE_SIZE);
  let tile = 0;
  for (let i = 0; i < count; i++) {
    const t = g.root + gt.root_size + i * gt.root_tile_size;
    const k = (BigInt(u32(g.words, t + ROOT_TILE_OFF_KEY + 4)) << 32n) | BigInt(u32(g.words, t + ROOT_TILE_OFF_KEY));
    if (k === key) {
      tile = t;
      break;
    }
  }
  const path: Path = { level: 4, upper: 0, lower: 0, leaf: 0, valueAddress: g.root + gt.root_off_background, active: false };
  if (tile === 0) return path;
  const child = childOffset(g, tile + ROOT_TILE_OFF_CHILD);
  if (child === 0 && u32(g.words, tile + ROOT_TILE_OFF_CHILD + 4) === 0) {
    path.level = 3;
    path.valueAddress = tile + gt.root_tile_off_value;
    path.active = u32(g.words, tile + ROOT_TILE_OFF_STATE) !== 0;
    return path;
  }
  path.upper = g.root + child;
  const un = (((x & 4095) >> 7) << 10) + (((y & 4095) >> 7) << 5) + ((z & 4095) >> 7);
  const ut = path.upper + gt.upper_off_table + gt.table_stride * un;
  if (!maskBit(g, path.upper + UPPER_OFF_CHILD_MASK, un)) {
    path.level = 2;
    path.valueAddress = ut;
    path.active = maskBit(g, path.upper + UPPER_OFF_VALUE_MASK, un);
    return path;
  }
  path.lower = path.upper + childOffset(g, ut);
  const ln = (((x & 127) >> 3) << 8) + (((y & 127) >> 3) << 4) + ((z & 127) >> 3);
  const lt = path.lower + gt.lower_off_table + gt.table_stride * ln;
  if (!maskBit(g, path.lower + LOWER_OFF_CHILD_MASK, ln)) {
    path.level = 1;
    path.valueAddress = lt;
    path.active = maskBit(g, path.lower + LOWER_OFF_VALUE_MASK, ln);
    return path;
  }
  path.leaf = path.lower + childOffset(g, lt);
  path.level = 0;
  path.active = maskBit(g, path.leaf + LEAF_OFF_VALUE_MASK, ((x & 7) << 6) + ((y & 7) << 3) + (z & 7));
  return path;
}

// ---------------------------------------------------------------------------
// Node size and statistics

/**
 * Twin of `vdbGetDim` (vendored `readaccessor_get_dim`): 1 inside a leaf, 8 in
 * a lower-node tile, 128 in an upper-node tile, 4096 in a root tile or the
 * background.
 */
export function getDim(gridWords: Uint32Array, ijk: Vec3): number {
  return DIM_OF_LEVEL[locate(grid(gridWords), ijk).level]!;
}

const DIM_OF_LEVEL = [1, 8, 128, 4096, 4096] as const;

/** Node levels: 0 leaf, 1 lower, 2 upper, 3 root. */
export type NodeLevel = 0 | 1 | 2 | 3;

export interface NodeStats {
  min: number;
  max: number;
  ave: number;
  stddev: number;
}

/** pnanovdb_leaf_fp_get_stat_float: `code * quantum (+ minimum)` in f32. */
function leafFpStat(g: Grid, leaf: number, address: number, withMinimum: boolean): number {
  const table = leaf + g.gt.leaf_off_table; // value_stride_bits is 0 for the Fp types
  const minimum = readF32(g, table - LEAF_TABLE_NEG_OFF_MINIMUM);
  const quantum = readF32(g, table - LEAF_TABLE_NEG_OFF_QUANTUM);
  const scaled = f(readU16(g, address) * quantum);
  return withMinimum ? f(scaled + minimum) : scaled;
}

/**
 * Twin of the fork's `{root,upper,lower,leaf}_get_{min,max,ave,stddev}_float`
 * for the node at byte `address` of the image. Root, upper and lower stats are
 * f32; Fp leaf stats are uint16 codes decoded against the leaf's quantization.
 * Meaningful only when the grid has stats (`HAS_MIN_MAX`) and the node has
 * active values.
 */
export function nodeStats(gridWords: Uint32Array, level: NodeLevel, address: number): NodeStats {
  const g = grid(gridWords);
  const gt = g.gt;
  if (level === 0 && g.fp) {
    return {
      min: leafFpStat(g, address, address + gt.leaf_off_min, true),
      max: leafFpStat(g, address, address + gt.leaf_off_max, true),
      ave: leafFpStat(g, address, address + gt.leaf_off_ave, true),
      stddev: leafFpStat(g, address, address + gt.leaf_off_stddev, false),
    };
  }
  const off =
    level === 0
      ? [gt.leaf_off_min, gt.leaf_off_max, gt.leaf_off_ave, gt.leaf_off_stddev]
      : level === 1
        ? [gt.lower_off_min, gt.lower_off_max, gt.lower_off_ave, gt.lower_off_stddev]
        : level === 2
          ? [gt.upper_off_min, gt.upper_off_max, gt.upper_off_ave, gt.upper_off_stddev]
          : [gt.root_off_min, gt.root_off_max, gt.root_off_ave, gt.root_off_stddev];
  return {
    min: readF32(g, address + off[0]!),
    max: readF32(g, address + off[1]!),
    ave: readF32(g, address + off[2]!),
    stddev: readF32(g, address + off[3]!),
  };
}

/** Whether the grid carries node stats (`HAS_MIN_MAX`; pool record flag b1). */
export function gridHasStats(gridWords: Uint32Array): boolean {
  return grid(gridWords).hasStats;
}

/** The pool's `valueMax` for this image with no caller bound (`VDBGridPool`): max(root max, background), or +Infinity without stats. */
export function defaultValueMax(gridWords: Uint32Array): number {
  const g = grid(gridWords);
  if (!g.hasStats) return Infinity;
  return Math.max(readF32(g, g.root + g.gt.root_off_max), readF32(g, g.root + g.gt.root_off_background));
}

export interface NodeMaxOptions {
  /** The record's `valueMax` (returned without stats). Default {@link defaultValueMax}. */
  valueMax?: number;
}

/**
 * Twin of `vdbNodeMax(acc, ijk, dim)`: a bound for the aligned cell of size
 * `dim` holding `ijk`. With `a = getDim(ijk)`: if `a > 1` and `dim <= a` the
 * cell is one tile and the result is its value (exact). Otherwise it is the
 * max statistic of the smallest node spanning the cell (leaf for `dim <= 8`,
 * lower `<= 128`, upper `<= 4096`, else root), which bounds that node's ACTIVE
 * values only; without stats it is `valueMax` (+Infinity unless the caller
 * gave a bound).
 */
export function nodeMax(gridWords: Uint32Array, ijk: Vec3, dim: number, options: NodeMaxOptions = {}): number {
  const g = grid(gridWords);
  const path = locate(g, ijk);
  const a = DIM_OF_LEVEL[path.level]!;
  if (a > 1 && dim <= a) return readF32(g, path.valueAddress);
  if (!g.hasStats) return options.valueMax ?? defaultValueMax(gridWords);
  if (dim <= 8) return nodeStats(gridWords, 0, path.leaf).max;
  if (dim <= 128) return nodeStats(gridWords, 1, path.lower).max;
  if (dim <= 4096) return nodeStats(gridWords, 2, path.upper).max;
  return nodeStats(gridWords, 3, g.root).max;
}

/** Twin of `vdbIsActive`. */
export function isActive(gridWords: Uint32Array, ijk: Vec3): boolean {
  return locate(grid(gridWords), ijk).active;
}

/**
 * Twin of `vdbReadValue` (f32 decode in WGSL order; any of Float/Fp4/Fp8/Fp16/FpN).
 *
 * This matches the GPU's evaluation order (f32 arithmetic), so it is the right
 * oracle for GPU parity checks. `readValue` instead returns the exact f64
 * decode; on Fp grids the two can differ by 1 ulp (f32).
 */
export function vdbReadValueF32(gridWords: Uint32Array, ijk: Vec3): number {
  const g = grid(gridWords);
  const path = locate(g, ijk);
  if (path.level !== 0) return readF32(g, path.valueAddress);
  const n = ((ijk[0] & 7) << 6) + ((ijk[1] & 7) << 3) + (ijk[2] & 7);
  return leafValue(g, path.leaf, n);
}

function leafValue(g: Grid, leaf: number, n: number): number {
  const table = leaf + g.gt.leaf_off_table;
  if (!g.fp) return readF32(g, table + ((g.gt.value_stride_bits * n) >> 3));
  let logBits: number;
  if (g.gridType === gridTypeIds.FP4) logBits = 2;
  else if (g.gridType === gridTypeIds.FP8) logBits = 3;
  else if (g.gridType === gridTypeIds.FP16) logBits = 4;
  else logBits = (u32(g.words, table - LEAF_TABLE_NEG_OFF_BBOX_DIF_AND_FLAGS) >>> 24) >>> 5;
  const bits = 1 << logBits;
  const perWordBits = 5 - logBits;
  const minimum = readF32(g, table - LEAF_TABLE_NEG_OFF_MINIMUM);
  const quantum = readF32(g, table - LEAF_TABLE_NEG_OFF_QUANTUM);
  const raw = u32(g.words, table + ((n >> perWordBits) << 2));
  const code = (raw >>> ((n & ((1 << perWordBits) - 1)) << logBits)) & ((1 << bits) - 1);
  return f(f(code * quantum) + minimum);
}

// ---------------------------------------------------------------------------
// Tree enumeration (for tests and tools)

export interface TreeNode {
  level: NodeLevel;
  /** Byte address of the node in the image. */
  address: number;
  /** Index-space origin (min corner) of the node; the root's is unused (0, 0, 0). */
  origin: Vec3;
  /** Index of the parent in the returned array; -1 for the root. */
  parent: number;
  /** Values of the node's ACTIVE tiles (root, upper, lower), with each tile's origin and size. */
  activeTiles: { value: number; origin: Vec3; dim: number }[];
}

/**
 * Every node of the tree, root first, then depth first. Leaves carry no tiles;
 * use {@link leafVoxels} for their values.
 */
export function treeNodes(gridWords: Uint32Array): TreeNode[] {
  const g = grid(gridWords);
  const gt = g.gt;
  const out: TreeNode[] = [{ level: 3, address: g.root, origin: [0, 0, 0], parent: -1, activeTiles: [] }];
  const rootNode = out[0]!;
  const count = u32(g.words, g.root + ROOT_OFF_TABLE_SIZE);
  for (let i = 0; i < count; i++) {
    const t = g.root + gt.root_size + i * gt.root_tile_size;
    const lo = u32(g.words, t + ROOT_TILE_OFF_KEY);
    const hi = u32(g.words, t + ROOT_TILE_OFF_KEY + 4);
    const key = (BigInt(hi) << 32n) | BigInt(lo);
    const origin: Vec3 = [
      Number((key >> 42n) & 0x1fffffn) << 12 | 0,
      Number((key >> 21n) & 0x1fffffn) << 12 | 0,
      Number(key & 0x1fffffn) << 12 | 0,
    ];
    const child = childOffset(g, t + ROOT_TILE_OFF_CHILD);
    if (child === 0 && u32(g.words, t + ROOT_TILE_OFF_CHILD + 4) === 0) {
      if (u32(g.words, t + ROOT_TILE_OFF_STATE) !== 0) rootNode.activeTiles.push({ value: readF32(g, t + gt.root_tile_off_value), origin, dim: 4096 });
      continue;
    }
    const upper = g.root + child;
    const ui = out.push({ level: 2, address: upper, origin, parent: 0, activeTiles: [] }) - 1;
    for (let un = 0; un < 32768; un++) {
      const uo: Vec3 = [origin[0] + ((un >> 10) & 31) * 128, origin[1] + ((un >> 5) & 31) * 128, origin[2] + (un & 31) * 128];
      const ut = upper + gt.upper_off_table + gt.table_stride * un;
      if (!maskBit(g, upper + UPPER_OFF_CHILD_MASK, un)) {
        if (maskBit(g, upper + UPPER_OFF_VALUE_MASK, un)) out[ui]!.activeTiles.push({ value: readF32(g, ut), origin: uo, dim: 128 });
        continue;
      }
      const lower = upper + childOffset(g, ut);
      const li = out.push({ level: 1, address: lower, origin: uo, parent: ui, activeTiles: [] }) - 1;
      for (let ln = 0; ln < 4096; ln++) {
        const lo3: Vec3 = [uo[0] + ((ln >> 8) & 15) * 8, uo[1] + ((ln >> 4) & 15) * 8, uo[2] + (ln & 15) * 8];
        const lt = lower + gt.lower_off_table + gt.table_stride * ln;
        if (!maskBit(g, lower + LOWER_OFF_CHILD_MASK, ln)) {
          if (maskBit(g, lower + LOWER_OFF_VALUE_MASK, ln)) out[li]!.activeTiles.push({ value: readF32(g, lt), origin: lo3, dim: 8 });
          continue;
        }
        out.push({ level: 0, address: lower + childOffset(g, lt), origin: lo3, parent: li, activeTiles: [] });
      }
    }
  }
  return out;
}

/** A leaf's 512 decoded values (f32, WGSL decode order) and active flags, x-major (`n = x*64 + y*8 + z`). */
export function leafVoxels(gridWords: Uint32Array, leafAddress: number): { values: Float32Array; active: Uint8Array } {
  const g = grid(gridWords);
  const values = new Float32Array(512);
  const active = new Uint8Array(512);
  for (let n = 0; n < 512; n++) {
    values[n] = leafValue(g, leafAddress, n);
    active[n] = maskBit(g, leafAddress + LEAF_OFF_VALUE_MASK, n) ? 1 : 0;
  }
  return { values, active };
}

// ---------------------------------------------------------------------------
// HDDA (vendored pnanovdb_hdda_*, in f32)

/** Twin of `VdbHdda` (vendored `pnanovdb_hdda_t`). */
export interface Hdda {
  dim: number;
  tmin: number;
  tmax: number;
  voxel: Vec3;
  step: Vec3;
  delta: Vec3;
  next: Vec3;
}

function posToVoxel(pos: Vec3, dim: number): Vec3 {
  const mask = ~(dim - 1);
  return [Math.floor(pos[0]) & mask, Math.floor(pos[1]) & mask, Math.floor(pos[2]) & mask];
}

function rayStart(o: Vec3, t: number, d: Vec3): Vec3 {
  return [f(o[0] + f(d[0] * t)), f(o[1] + f(d[1] * t)), f(o[2] + f(d[2] * t))];
}

const f3 = (v: readonly number[]): Vec3 => [f(v[0]!), f(v[1]!), f(v[2]!)];

/** Twin of `vdbHddaInit` (vendored `pnanovdb_hdda_init`). */
export function hddaInit(origin: Vec3, tmin: number, direction: Vec3, tmax: number, dim: number): Hdda {
  const o = f3(origin);
  const d = f3(direction);
  const h: Hdda = { dim, tmin: f(tmin), tmax: f(tmax), voxel: [0, 0, 0], step: [0, 0, 0], delta: [0, 0, 0], next: [0, 0, 0] };
  const pos = rayStart(o, h.tmin, d);
  h.voxel = posToVoxel(pos, dim);
  for (let i = 0; i < 3; i++) {
    const inv = f(1 / d[i]!);
    if (d[i] === 0) {
      h.next[i] = HDDA_FLOAT_MAX;
      h.step[i] = 0;
      h.delta[i] = 0;
    } else if (inv > 0) {
      h.step[i] = 1;
      h.next[i] = f(h.tmin + f(f(f(h.voxel[i]! + dim) - pos[i]!) * inv));
      h.delta[i] = inv;
    } else {
      h.step[i] = -1;
      h.next[i] = f(h.tmin + f(f(h.voxel[i]! - pos[i]!) * inv));
      h.delta[i] = -inv;
    }
  }
  return h;
}

/** Twin of `vdbHddaUpdate` (vendored `pnanovdb_hdda_update`): switch to `dim`; false if already there. */
export function hddaUpdate(h: Hdda, origin: Vec3, direction: Vec3, dim: number): boolean {
  if (h.dim === dim) return false;
  h.dim = dim;
  const o = f3(origin);
  const d = f3(direction);
  const pos = rayStart(o, h.tmin, d);
  h.voxel = posToVoxel(pos, dim);
  for (let i = 0; i < 3; i++) {
    if (h.step[i] === 0) continue;
    const inv = f(1 / d[i]!);
    h.next[i] = f(h.tmin + f(f(h.voxel[i]! - pos[i]!) * inv));
    if (h.step[i]! > 0) h.next[i] = f(h.next[i]! + f(dim * inv));
  }
  return true;
}

/** Twin of `vdbHddaStep` (vendored `pnanovdb_hdda_step`, with PNANOVDB_ENFORCE_FORWARD_STEPPING). */
export function hddaStep(h: Hdda): boolean {
  const n = h.next;
  const a = n[0] < n[1] && n[0] < n[2] ? 0 : n[1] < n[2] ? 1 : 2;
  if (n[a]! <= h.tmin) n[a] = f(n[a]! + f(f(h.tmin - f(FORWARD_SCALE * n[a]!)) + FORWARD_BIAS));
  h.tmin = n[a]!;
  n[a] = f(n[a]! + f(h.dim * h.delta[a]!));
  h.voxel[a] = (h.voxel[a]! + h.dim * h.step[a]!) | 0;
  return h.tmin <= h.tmax;
}

/** The grid's index bbox (root bbox, inclusive): the pool record's `indexBBoxMin/Max`. */
export function indexBBox(gridWords: Uint32Array): { min: Vec3; max: Vec3 } {
  const g = grid(gridWords);
  const i = (a: number): number => u32(g.words, a) | 0;
  return {
    min: [i(g.root + ROOT_OFF_BBOX_MIN), i(g.root + ROOT_OFF_BBOX_MIN + 4), i(g.root + ROOT_OFF_BBOX_MIN + 8)],
    max: [i(g.root + ROOT_OFF_BBOX_MAX), i(g.root + ROOT_OFF_BBOX_MAX + 4), i(g.root + ROOT_OFF_BBOX_MAX + 8)],
  };
}

/**
 * Twin of `vdbRayClip(g, o, d, &t)`: clip `[t0, t1]` to the voxel-cell box
 * `[min, max + 1]` (vendored `hdda_ray_clip`); hit if the clipped interval is
 * non-empty. NaN handling (an axis-parallel ray starting exactly on a slab
 * plane) follows JS `Math.min/max`, which a GPU need not match.
 */
export function rayClip(gridWords: Uint32Array, origin: Vec3, direction: Vec3, t: [number, number]): { hit: boolean; t0: number; t1: number } {
  const { min, max } = indexBBox(gridWords);
  return clipBox(
    [min[0], min[1], min[2]],
    [max[0] + 1, max[1] + 1, max[2] + 1],
    origin,
    direction,
    t,
  );
}

/** Options of {@link rayClipSupport}. */
export interface RayClipSupportOptions {
  /** The module's sampler (default `'trilinear'`). */
  sampler?: "trilinear" | "nearest";
}

/**
 * Twin of the opt-in `vdbRayClipSupport(g, o, d, &t)` (helper
 * `'rayClipSupport'`): clip `[t0, t1]` to the box where `vdbSample` can read
 * the grid's voxels, `[min - 1, max + 1]` for trilinear and
 * `[min - 0.5, max + 0.5]` for nearest. Unlike {@link rayClip}'s
 * `[min, max + 1]` it keeps the min-side half of the trilinear support.
 */
export function rayClipSupport(
  gridWords: Uint32Array,
  origin: Vec3,
  direction: Vec3,
  t: [number, number],
  options: RayClipSupportOptions = {},
): { hit: boolean; t0: number; t1: number } {
  const { min, max } = indexBBox(gridWords);
  const pad = options.sampler === "nearest" ? 0.5 : 1;
  return clipBox(
    [f(min[0] - pad), f(min[1] - pad), f(min[2] - pad)],
    [f(max[0] + pad), f(max[1] + pad), f(max[2] + pad)],
    origin,
    direction,
    t,
  );
}

/** The vendored `hdda_ray_clip` slab test against `[lo, hi]` (f32 box corners). */
function clipBox(lo: Vec3, hi: Vec3, origin: Vec3, direction: Vec3, t: [number, number]): { hit: boolean; t0: number; t1: number } {
  const o = f3(origin);
  const d = f3(direction);
  const near: number[] = [];
  const far: number[] = [];
  for (let i = 0; i < 3; i++) {
    const inv = f(1 / d[i]!);
    const a = f(f(lo[i]! - o[i]!) * inv);
    const b = f(f(hi[i]! - o[i]!) * inv);
    near.push(Math.min(a, b));
    far.push(Math.max(a, b));
  }
  const tnear = Math.max(near[0]!, Math.max(near[1]!, near[2]!));
  const tfar = Math.min(far[0]!, Math.min(far[1]!, far[2]!));
  const hit = tnear <= tfar;
  const t0 = Math.max(f(t[0]), tnear);
  const t1 = Math.min(f(t[1]), tfar);
  return { hit: hit && t0 <= t1, t0, t1 };
}

/**
 * Twin of the opt-in `vdbHddaCellEnd(&h)` (helper `'hddaCellEnd'`): the ray
 * `t` where the HDDA's current cell ends, `min(next, tmax)`.
 */
export function hddaCellEnd(h: Hdda): number {
  return Math.min(Math.min(h.next[0], h.next[1]), Math.min(h.next[2], h.tmax));
}

/**
 * Twin of the opt-in `vdbCellBoundTrilinear(acc, ijk, dim)` (helper
 * `'cellBoundTrilinear'`): the max of {@link nodeMax} over the 2x2x2 block of
 * aligned `dim`-cells whose min corner is the cell holding `ijk`. It bounds
 * every trilinear sample whose base voxel `floor(p)` lies in that cell,
 * provided every inactive leaf voxel is <= its leaf's max (stats bound active
 * values only; true of fog volumes).
 */
export function cellBoundTrilinear(gridWords: Uint32Array, ijk: Vec3, dim: number, options: NodeMaxOptions = {}): number {
  let m = nodeMax(gridWords, ijk, dim, options);
  for (let k = 1; k < 8; k++) {
    const c: Vec3 = [ijk[0] + (k & 1) * dim, ijk[1] + ((k >> 1) & 1) * dim, ijk[2] + ((k >> 2) & 1) * dim];
    m = Math.max(m, nodeMax(gridWords, c, dim, options));
  }
  return m;
}

/**
 * Twin of the opt-in `vdbNodeAverage(acc, ijk, dim)` (helper `'nodeAverage'`,
 * K3.2): the box mean of the aligned `dim`-cell holding `ijk` from node
 * statistics. A cell inside one tile (or the background) is the tile's value;
 * without stats (or `dim <= 1`) the value at `ijk`; `dim <= 8`: the leaf's
 * `ave * activeVoxels / 512` (exact for fog leaves, whose inactive voxels are
 * 0); larger: the lower node's `ave` (mean of its ACTIVE voxels; `dim` is meant
 * to be 8 or 128). f32 arithmetic in WGSL order.
 */
export function nodeAverage(gridWords: Uint32Array, ijk: Vec3, dim: number): number {
  const g = grid(gridWords);
  const path = locate(g, ijk);
  const a = DIM_OF_LEVEL[path.level]!;
  if ((a > 1 && dim <= a) || !g.hasStats || dim <= 1) return vdbReadValueF32(gridWords, ijk);
  if (dim <= 8) {
    let n = 0;
    for (let w = 0; w < 16; w++) {
      let v = u32(g.words, path.leaf + LEAF_OFF_VALUE_MASK + 4 * w);
      while (v) {
        v &= v - 1;
        n++;
      }
    }
    return f(nodeStats(gridWords, 0, path.leaf).ave * f(n / 512));
  }
  return nodeStats(gridWords, 1, path.lower).ave;
}

// ---------------------------------------------------------------------------
// The canonical walk

/** One HDDA cell the walk visited. */
export interface HddaSegment {
  /** Where the cell's corner resolves: 0 inside a leaf, 1 lower tile, 2 upper tile, 3 root tile or background. */
  level: 0 | 1 | 2 | 3;
  /** HDDA cell size (1, 8, 128 or 4096). */
  dim: number;
  /** Cell min corner (aligned to `dim`). */
  voxel: Vec3;
  /** Ray interval in the cell: entry `h.tmin` to `min(h.next, h.tmax)`. */
  t0: number;
  t1: number;
  /** False only when the cell is uniform (one tile or one voxel) and inactive. */
  active: boolean;
  /** `vdbNodeMax(voxel, dim)`. */
  max: number;
  /** `!active || max < threshold`: a march may skip the cell. */
  skip: boolean;
}

export interface HddaWalkOptions {
  /** Ray interval before clipping. Default `[0, 3.4e38]`. */
  tmin?: number;
  tmax?: number;
  /** Smallest cell: 8 (default) steps leaf by leaf, 1 voxel by voxel inside leaves. */
  leafDim?: 1 | 8;
  /** Cells whose bound is below this are skipped. Default -Infinity (skip only inactive cells). */
  threshold?: number;
  /** The record's `valueMax` (see {@link nodeMax}). */
  valueMax?: number;
  /** Safety cap on loop iterations. Default 1 << 16. */
  maxSteps?: number;
}

/**
 * The canonical `march` walk, built only from the profile's calls (the GPU
 * probe kernel in scripts/lib/march-probe.mjs is the same loop in WGSL):
 *
 * ```
 * t = [tmin, tmax]; if !vdbRayClip(g, o, d, &t) { return; }
 * vdbHddaInit(&h, o, t.x, d, t.y, max(vdbGetDim(&a, floor(o + d * t.x)), leafDim));
 * loop {
 *   want = max(vdbGetDim(&a, h.voxel), leafDim);       // probe the cell's corner
 *   if want != h.dim && updates < 2 { vdbHddaUpdate(&h, o, d, want); updates++; continue; }
 *   updates = 0;
 *   emit (h.voxel, h.dim, h.tmin, min(h.next, h.tmax), active, vdbNodeMax(&a, h.voxel, h.dim));
 *   if !vdbHddaStep(&h) { break; }
 * }
 * ```
 *
 * The corner decides the cell: a tile of size >= dim holding the corner holds
 * the whole aligned cell. `vdbHddaUpdate` re-derives the cell from
 * `floor(o + d * h.tmin)`, which on the entry face can name the previous cell;
 * that yields a zero-length cell and the vendored forward-stepping moves on.
 * Every emitted cell is classified from its own corner, so classification is
 * valid even for those. At most two updates per cell keep the loop finite.
 * `vdbHddaStep` continues while `tmin <= tmax`, so the last cell can be a
 * zero-length one at the exit face.
 */
export function hddaWalk(gridWords: Uint32Array, origin: Vec3, direction: Vec3, options: HddaWalkOptions = {}): HddaSegment[] {
  const leafDim = options.leafDim ?? 8;
  const threshold = options.threshold ?? -Infinity;
  const maxSteps = options.maxSteps ?? 1 << 16;
  const o = f3(origin);
  const d = f3(direction);
  const clip = rayClip(gridWords, o, d, [options.tmin ?? 0, options.tmax ?? 3.4e38]);
  if (!clip.hit) return [];
  const cellDim = (ijk: Vec3) => Math.max(getDim(gridWords, ijk), leafDim);
  const start = rayStart(o, clip.t0, d);
  const h = hddaInit(o, clip.t0, d, clip.t1, cellDim([Math.floor(start[0]), Math.floor(start[1]), Math.floor(start[2])]));
  const out: HddaSegment[] = [];
  let updates = 0;
  for (let i = 0; i < maxSteps; i++) {
    const a = getDim(gridWords, h.voxel);
    const want = Math.max(a, leafDim);
    if (want !== h.dim && updates < 2) {
      hddaUpdate(h, o, d, want);
      updates++;
      continue;
    }
    updates = 0;
    const voxel: Vec3 = [h.voxel[0], h.voxel[1], h.voxel[2]];
    const active = a >= h.dim ? isActive(gridWords, voxel) : true;
    const max = nodeMax(gridWords, voxel, h.dim, options.valueMax === undefined ? {} : { valueMax: options.valueMax });
    out.push({
      level: a === 1 ? 0 : a === 8 ? 1 : a === 128 ? 2 : 3,
      dim: h.dim,
      voxel,
      t0: h.tmin,
      t1: hddaCellEnd(h),
      active,
      max,
      skip: !active || max < threshold,
    });
    if (!hddaStep(h)) break;
  }
  return out;
}
