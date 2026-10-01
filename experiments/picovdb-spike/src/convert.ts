/**
 * NanoVDB -> PicoVDB converter, a pure-TypeScript port of upstream's native
 * converter (emcfarlane/picovdb @ f5d22e5, `src/main.zig`
 * `convertNanoVDBToPicoVDB` / `convertGrid` / `convertRootTiles` /
 * `convert{Upper,Lower,Leaf}NodesFromHandle`, plus the file encoder in
 * `src/picovdb.zig` `PicoVDBFileMutable.encode`).
 *
 * The port is **byte-faithful**: for the same input it must produce exactly
 * the bytes `zig-out/bin/picovdb convert [--type f32|u8]` writes. That is
 * oracle (b) in the spike report, asserted by `test/convert.test.ts` against
 * SHA-256 goldens recorded from a native build. Native quirks are reproduced
 * on purpose. Where a quirk breaks reads (see `ConvertDiagnostics`), it is
 * counted, not "fixed", so the oracle stays meaningful. A `layout: "fixed"`
 * option is offered separately to show what a corrected writer looks like.
 *
 * Scope is the same as native:
 * - Source grids must be NanoVDB FLOAT. Native reads every value with
 *   `pnanovdb_read_float` under `PNANOVDB_GRID_TYPE_FLOAT` strides and never
 *   checks the source type, so an Fp8/FpN input yields garbage there. Here it
 *   throws instead.
 * - FogVolume class (2) -> `GRID_TYPE_FOG_FLOAT` (raw density, inside = 1.0).
 *   Every other class -> SDF: values divided by voxel size (index-space
 *   distance), inside = -background, plus "surface" bits on leaf voxels whose
 *   sign differs from a +x/+y/+z neighbour. `valueType: "u8"` -> `GRID_TYPE_SDF_UINT8`
 *   (fixed [-3, 3] voxel range).
 * - Fog + u8 is not a real mode. Native writes u8 bytes but tags the grid
 *   FOG_FLOAT, so the GPU reads garbage. Here it throws.
 */

import { defineNumber, gridTypeConstantsFor } from "../../../packages/nanovdb-wgsl/src/cpu/stride-tables.js";
import { readValue } from "../../../packages/nanovdb-wgsl/src/cpu/read-value.js";

// ---- PicoVDB format constants (src/picovdb.zig / ts/picovdb.ts) ----------
export const PICOVDB_MAGIC = [0x6f636950, 0x30424456] as const; // "PicoVDB0"
export const GRID_TYPE_SDF_FLOAT = 1;
export const GRID_TYPE_SDF_UINT8 = 2;
export const GRID_TYPE_FOG_FLOAT = 3;
export const HEADER_SIZE = 32;
export const GRID_SIZE = 64;
export const ROOT_SIZE = 8;
export const UPPER_SIZE = 16 + 1024 * 12; // 12304
export const LOWER_SIZE = 16 + 128 * 12; // 1552
export const LEAF_SIZE = 16 + 16 * 12; // 208
export const LEVEL_SET_HALF_WIDTH = 3.0;

// ---- NanoVDB layout (FLOAT strides; native hardcodes GRID_TYPE_FLOAT) ----
const NANO_GRID_TYPE_FLOAT = 1;
const NANO_GRID_CLASS_FOG = 2;
const GT = gridTypeConstantsFor("FLOAT");
const GRID_OFF_GRID_TYPE = defineNumber("PNANOVDB_GRID_OFF_GRID_TYPE");
const GRID_OFF_GRID_CLASS = defineNumber("PNANOVDB_GRID_OFF_GRID_CLASS");
const GRID_OFF_VOXEL_SIZE = defineNumber("PNANOVDB_GRID_OFF_VOXEL_SIZE");
const NANO_GRID_SIZE = defineNumber("PNANOVDB_GRID_SIZE");
const TREE_OFF_NODE_OFFSET_ROOT = defineNumber("PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT");
const ROOT_OFF_BBOX_MIN = defineNumber("PNANOVDB_ROOT_OFF_BBOX_MIN");
const ROOT_OFF_BBOX_MAX = defineNumber("PNANOVDB_ROOT_OFF_BBOX_MAX");
const ROOT_OFF_TABLE_SIZE = defineNumber("PNANOVDB_ROOT_OFF_TABLE_SIZE");
const ROOT_TILE_OFF_KEY = defineNumber("PNANOVDB_ROOT_TILE_OFF_KEY");
const ROOT_TILE_OFF_CHILD = defineNumber("PNANOVDB_ROOT_TILE_OFF_CHILD");
const UPPER_OFF_VALUE_MASK = defineNumber("PNANOVDB_UPPER_OFF_VALUE_MASK");
const UPPER_OFF_CHILD_MASK = defineNumber("PNANOVDB_UPPER_OFF_CHILD_MASK");
const LOWER_OFF_VALUE_MASK = defineNumber("PNANOVDB_LOWER_OFF_VALUE_MASK");
const LOWER_OFF_CHILD_MASK = defineNumber("PNANOVDB_LOWER_OFF_CHILD_MASK");
const LEAF_OFF_VALUE_MASK = defineNumber("PNANOVDB_LEAF_OFF_VALUE_MASK");

export type ValueType = "f32" | "u8";

export interface ConvertOptions {
  /** Value encoding (native `--type`). Default "f32". */
  valueType?: ValueType;
  /**
   * "native" (default) reproduces upstream byte-for-byte, including the
   * value-ordering defect that breaks reads of active internal tiles in nodes
   * that also have children (see `ConvertDiagnostics.misindexedTileValues`).
   * "fixed" writes each internal node's own tile values contiguously before
   * descending into its children, so the reader's rank query is correct. That
   * is a format-compatible fix (the reader is unchanged), and not upstream
   * behaviour.
   */
  layout?: "native" | "fixed";
}

export interface GridInput {
  /** NanoVDB grid image (GridData at word 0), FLOAT only. */
  image: Uint32Array;
  /**
   * Grid class as native sees it. Native reads it from the `.nvdb`
   * FileMetaData; a raw grid buffer is assumed level set. Default: read
   * GridData.gridClass.
   */
  gridClassId?: number;
}

export interface ConvertDiagnostics {
  grids: number;
  roots: number;
  uppers: number;
  lowers: number;
  leaves: number;
  /** Values written to the data buffer, summed over grids (incl. the 2 implicit per grid). */
  dataElems: number;
  surfaceVoxels: number;
  /**
   * Root tiles without a child (constant 4096^3 tiles). Native still appends a
   * root key but no upper node, which breaks the roots<->uppers 1:1 invariant
   * the reader relies on. Non-zero means the output is corrupt for reads.
   */
  childlessRootTiles: number;
  /**
   * Active tile values in an upper/lower node whose stored position differs
   * from where the reader's rank query looks (native "layout" only). Native
   * appends a node's tile values word by word, interleaved with its
   * children's data, but indexes them as if they were contiguous.
   */
  misindexedTileValues: number;
}

export interface ConvertResult {
  bytes: Uint8Array<ArrayBuffer>;
  diagnostics: ConvertDiagnostics;
}

// ---------------------------------------------------------------------------
// Small growable byte buffer for the data section.
class ByteSink {
  buf = new Uint8Array(1 << 16);
  view = new DataView(this.buf.buffer);
  length = 0;
  private reserve(n: number): void {
    if (this.length + n <= this.buf.length) return;
    let cap = this.buf.length * 2;
    while (cap < this.length + n) cap *= 2;
    const next = new Uint8Array(cap);
    next.set(this.buf.subarray(0, this.length));
    this.buf = next;
    this.view = new DataView(next.buffer);
  }
  f32(v: number): void {
    this.reserve(4);
    this.view.setFloat32(this.length, v, true);
    this.length += 4;
  }
  u8(v: number): void {
    this.reserve(1);
    this.buf[this.length++] = v;
  }
  zeros(n: number): void {
    this.reserve(n);
    this.buf.fill(0, this.length, this.length + n);
    this.length += n;
  }
}

/** One 12-byte node element (state mask, value mask, packed local index). */
type Elements = Uint32Array; // 3 u32 per element

interface NodeRecord {
  baseInside: number;
  baseActive: number;
  elements: Elements;
}

interface Mutable {
  grids: Uint32Array[]; // 16 u32 each
  roots: [number, number][];
  uppers: NodeRecord[];
  lowers: NodeRecord[];
  leaves: NodeRecord[];
  data: ByteSink;
  diag: ConvertDiagnostics;
}

const fround = Math.fround;

/** native `appendValue`: f32 bytes, or the u8 level-set quantizer (all f32 math). */
function appendValue(m: Mutable, value: number, vt: ValueType): void {
  if (vt === "f32") {
    m.data.f32(value);
  } else {
    // @round((value / 3.0 + 1.0) * 127.5) clamped to [0, 255], in f32.
    const q = fround(fround(fround(value / LEVEL_SET_HALF_WIDTH) + 1.0) * 127.5);
    // Zig @round is half-away-from-zero; values below 0 clamp to 0 either way.
    const r = q >= 0 ? Math.floor(q + 0.5) : -Math.floor(-q + 0.5);
    m.data.u8(Math.min(255, Math.max(0, r)));
  }
}

function elemSize(vt: ValueType): number {
  return vt === "f32" ? 4 : 1;
}

function popcount(x: number): number {
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  return (((x + (x >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

// ---- NanoVDB raw reads (byte addresses into the grid image) --------------
function u32(w: Uint32Array, addr: number): number {
  return w[addr >>> 2]!;
}
function i64(w: Uint32Array, addr: number): number {
  const lo = w[addr >>> 2]!;
  const hi = w[(addr >>> 2) + 1]! | 0;
  return hi * 0x100000000 + lo;
}
const f32Views = new WeakMap<Uint32Array, Float32Array>();
function f32(w: Uint32Array, addr: number): number {
  let v = f32Views.get(w);
  if (!v) {
    v = new Float32Array(w.buffer, w.byteOffset, w.length);
    f32Views.set(w, v);
  }
  return v[addr >>> 2]!;
}

interface Ctx {
  w: Uint32Array;
  grid: Uint32Array; // the PicoVDB grid record being built (16 u32)
  vt: ValueType;
  isFog: boolean;
  voxelSize: number; // f32
  fixed: boolean;
}

function scaled(ctx: Ctx, raw: number): number {
  return ctx.isFog ? raw : fround(raw / ctx.voxelSize);
}

function relDataIndex(m: Mutable, ctx: Ctx): number {
  return ((m.data.length - ctx.grid[4]! * 16) / elemSize(ctx.vt)) >>> 0;
}

/** Shared upper/lower conversion. dim: 1024 words (upper) or 128 (lower). */
function convertInternal(
  m: Mutable,
  ctx: Ctx,
  nodeAddr: number,
  origin: [number, number, number],
  level: "upper" | "lower",
): void {
  const words = level === "upper" ? 1024 : 128;
  const valueMaskAddr = nodeAddr + (level === "upper" ? UPPER_OFF_VALUE_MASK : LOWER_OFF_VALUE_MASK);
  const childMaskAddr = nodeAddr + (level === "upper" ? UPPER_OFF_CHILD_MASK : LOWER_OFF_CHILD_MASK);
  const tableOff = level === "upper" ? GT.upper_off_table : GT.lower_off_table;
  const childList = level === "upper" ? m.lowers : m.leaves;
  const childStart = level === "upper" ? ctx.grid[2]! : ctx.grid[3]!;

  const elements = new Uint32Array(words * 3);
  const baseChild = childList.length - childStart;
  const baseValue = relDataIndex(m, ctx);
  let localState = 0;
  let localValue = 0;

  // In "fixed" layout, write this node's own tile values first (in word order),
  // then visit the children. That is what the reader's rank query assumes.
  const deferredChildren: number[] = [];

  for (let i = 0; i < words; i++) {
    const childWord = u32(ctx.w, childMaskAddr + i * 4);
    const valueWord = u32(ctx.w, valueMaskAddr + i * 4);
    let stateBits = 0;
    let valueBits = 0;
    for (let b = 0; b < 32; b++) {
      const n = i * 32 + b;
      const hasValue = (valueWord >>> b) & 1;
      const hasChild = (childWord >>> b) & 1;
      if (hasChild) {
        stateBits |= 1 << b;
        valueBits |= 1 << b;
      } else if (hasValue) {
        valueBits |= 1 << b;
        const raw = f32(ctx.w, nodeAddr + tableOff + GT.table_stride * n);
        // Diagnostic: where the reader will look vs where native writes it.
        const expected = baseValue + localValue + popcount(valueBits & ~stateBits & ((1 << b) - 1) >>> 0);
        if (!ctx.fixed && relDataIndex(m, ctx) !== expected) m.diag.misindexedTileValues++;
        appendValue(m, scaled(ctx, raw), ctx.vt);
      } else {
        const raw = f32(ctx.w, nodeAddr + tableOff + GT.table_stride * n);
        if (raw < 0) stateBits |= 1 << b;
      }
    }
    stateBits >>>= 0;
    valueBits >>>= 0;
    elements[i * 3] = stateBits;
    elements[i * 3 + 1] = valueBits;
    elements[i * 3 + 2] = ((localState << 16) | localValue) >>> 0;
    localState += popcount((valueBits & stateBits) >>> 0);
    localValue += popcount((valueBits & ~stateBits) >>> 0);

    for (let b = 0; b < 32; b++) {
      if (((childWord >>> b) & 1) === 0) continue;
      const n = i * 32 + b;
      if (ctx.fixed) deferredChildren.push(n);
      else visitChild(m, ctx, nodeAddr, origin, level, n);
    }
  }
  for (const n of deferredChildren) visitChild(m, ctx, nodeAddr, origin, level, n);

  const rec: NodeRecord = { baseInside: baseChild, baseActive: baseValue, elements };
  (level === "upper" ? m.uppers : m.lowers).push(rec);
}

function visitChild(
  m: Mutable,
  ctx: Ctx,
  nodeAddr: number,
  origin: [number, number, number],
  level: "upper" | "lower",
  n: number,
): void {
  const tableOff = level === "upper" ? GT.upper_off_table : GT.lower_off_table;
  const childAddr = nodeAddr + i64(ctx.w, nodeAddr + tableOff + GT.table_stride * n);
  if (level === "upper") {
    const o: [number, number, number] = [
      origin[0] + ((n >>> 10) & 31) * 128,
      origin[1] + ((n >>> 5) & 31) * 128,
      origin[2] + (n & 31) * 128,
    ];
    convertInternal(m, ctx, childAddr, o, "lower");
  } else {
    const o: [number, number, number] = [
      origin[0] + ((n >>> 8) & 15) * 8,
      origin[1] + ((n >>> 4) & 15) * 8,
      origin[2] + (n & 15) * 8,
    ];
    convertLeaf(m, ctx, childAddr, o);
  }
}

const NEIGHBORS: readonly [number, number, number][] = [
  [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
];

function convertLeaf(m: Mutable, ctx: Ctx, leafAddr: number, origin: [number, number, number]): void {
  const values = new Float32Array(512);
  const valueBits = new Uint32Array(16);
  const stateBits = new Uint32Array(16);

  // Phase 1: all 512 values; inactive negatives -> state ("inside implicit").
  for (let i = 0; i < 16; i++) {
    const vw = u32(ctx.w, leafAddr + LEAF_OFF_VALUE_MASK + i * 4);
    let v = 0;
    let s = 0;
    for (let b = 0; b < 32; b++) {
      const n = i * 32 + b;
      const raw = f32(ctx.w, leafAddr + GT.leaf_off_table + ((GT.value_stride_bits * n) >> 3));
      const value = scaled(ctx, raw);
      values[n] = value;
      if ((vw >>> b) & 1) v |= 1 << b;
      else if (value < 0) s |= 1 << b;
    }
    valueBits[i] = v >>> 0;
    stateBits[i] = s >>> 0;
  }

  // Phase 2 (SDF only): surface = sign differs from any of 7 +-side neighbours.
  if (!ctx.isFog) {
    for (let i = 0; i < 16; i++) {
      let surface = 0;
      const vw = valueBits[i]!;
      for (let b = 0; b < 32; b++) {
        if (((vw >>> b) & 1) === 0) continue;
        const n = i * 32 + b;
        const value = values[n]!;
        const lx = n >>> 6;
        const ly = (n >>> 3) & 7;
        const lz = n & 7;
        for (const [dx, dy, dz] of NEIGHBORS) {
          const nx = lx + dx;
          const ny = ly + dy;
          const nz = lz + dz;
          let nv: number;
          if (nx < 8 && ny < 8 && nz < 8) {
            nv = values[nx * 64 + ny * 8 + nz]!;
          } else {
            // native: pnanovdb_readaccessor_get_value_address -> read_float / voxel_size
            // (divides even for fog, but this branch is SDF-only).
            const r = readValue(ctx.w, [origin[0] + nx, origin[1] + ny, origin[2] + nz]).value;
            nv = fround(r / ctx.voxelSize);
          }
          if ((value < 0) !== (nv < 0) || (value <= 0) !== (nv <= 0)) {
            surface |= 1 << b;
            break;
          }
        }
      }
      stateBits[i] = ((stateBits[i]! & ~vw) | (surface & vw)) >>> 0;
    }
  }

  // Phase 3: elements + values (every value voxel stores its value).
  const elements = new Uint32Array(16 * 3);
  const baseValue = relDataIndex(m, ctx);
  let localValue = 0;
  let localState = 0;
  for (let i = 0; i < 16; i++) {
    const v = valueBits[i]!;
    const s = stateBits[i]!;
    elements[i * 3] = s;
    elements[i * 3 + 1] = v;
    elements[i * 3 + 2] = ((localState << 16) | localValue) >>> 0;
    localValue += popcount(v);
    localState += popcount((v & s) >>> 0);
    for (let b = 0; b < 32; b++) {
      if ((v >>> b) & 1) appendValue(m, values[i * 32 + b]!, ctx.vt);
    }
  }
  m.leaves.push({ baseInside: 0, baseActive: baseValue, elements });
}

function convertGrid(m: Mutable, input: GridInput, vt: ValueType, fixed: boolean): void {
  const w = input.image;
  const gridType = u32(w, GRID_OFF_GRID_TYPE);
  if (gridType !== NANO_GRID_TYPE_FLOAT) {
    throw new Error(
      `convert: source grid type ${gridType} is not FLOAT (1). PicoVDB's converter reads every value as ` +
        `f32 under FLOAT strides; quantized NanoVDB (Fp8/FpN) must be converted from its float source.`,
    );
  }
  const gridClassId = input.gridClassId ?? u32(w, GRID_OFF_GRID_CLASS);
  const isFog = gridClassId === NANO_GRID_CLASS_FOG;
  if (isFog && vt === "u8") {
    throw new Error(
      "convert: fog + u8 is unsupported upstream (native tags the grid FOG_FLOAT but writes level-set " +
        "u8 bytes, so every read is wrong). PicoVDB has no quantized fog encoding.",
    );
  }
  // pnanovdb_grid_get_voxel_size(.., 0): f64 voxelSize[0] cast to f32.
  const vs = fround(new Float64Array(w.buffer, w.byteOffset + GRID_OFF_VOXEL_SIZE, 1)[0]!);

  const treeAddr = NANO_GRID_SIZE;
  const rootAddr = treeAddr + i64(w, treeAddr + TREE_OFF_NODE_OFFSET_ROOT);

  if (m.data.length % 16 !== 0) throw new Error("convert: data buffer misaligned between grids");
  const grid = new Uint32Array(16);
  grid[0] = m.grids.length; // grid_index
  grid[1] = m.uppers.length; // upper_start (= root start)
  grid[2] = m.lowers.length; // lower_start
  grid[3] = m.leaves.length; // leaf_start
  grid[4] = m.data.length / 16; // data_start (16-byte units)
  grid[5] = 0; // data_elem_count (set below)
  grid[6] = isFog ? GRID_TYPE_FOG_FLOAT : vt === "f32" ? GRID_TYPE_SDF_FLOAT : GRID_TYPE_SDF_UINT8;
  for (let a = 0; a < 3; a++) {
    grid[8 + a] = u32(w, rootAddr + ROOT_OFF_BBOX_MIN + a * 4);
    grid[12 + a] = u32(w, rootAddr + ROOT_OFF_BBOX_MAX + a * 4);
  }
  const ctx: Ctx = { w, grid, vt, isFog, voxelSize: vs, fixed };
  const dataStartBytes = m.data.length;
  const leafStart = m.leaves.length;

  // Background (index 0) and inside (index 1).
  const rawBg = f32(w, rootAddr + GT.root_off_background);
  const bg = isFog ? rawBg : fround(rawBg / vs);
  appendValue(m, bg, vt);
  appendValue(m, isFog ? 1.0 : fround(-bg), vt);

  const tileCount = u32(w, rootAddr + ROOT_OFF_TABLE_SIZE);
  for (let t = 0; t < tileCount; t++) {
    const tile = rootAddr + GT.root_size + t * GT.root_tile_size;
    const k0 = u32(w, tile + ROOT_TILE_OFF_KEY);
    const k1 = u32(w, tile + ROOT_TILE_OFF_KEY + 4);
    m.roots.push([k0, k1]);
    const child = i64(w, tile + ROOT_TILE_OFF_CHILD);
    if (child === 0) {
      m.diag.childlessRootTiles++;
      continue;
    }
    const ku = k0 & 0x1fffff;
    const ju = ((k0 >>> 21) | ((k1 & 0x3ff) << 11)) >>> 0;
    const iu = k1 >>> 10;
    const origin: [number, number, number] = [(iu << 12) | 0, (ju << 12) | 0, (ku << 12) | 0];
    convertInternal(m, ctx, rootAddr + child, origin, "upper");
  }

  // Post-pass: leaf base_inside_index = running surface count.
  let surface = 0;
  for (let i = leafStart; i < m.leaves.length; i++) {
    const leaf = m.leaves[i]!;
    leaf.baseInside = surface;
    for (let j = 0; j < 16; j++) surface += popcount((leaf.elements[j * 3]! & leaf.elements[j * 3 + 1]!) >>> 0);
  }
  m.diag.surfaceVoxels += surface;

  grid[5] = (m.data.length - dataStartBytes) / elemSize(vt);
  m.diag.dataElems += grid[5];
  const pad = (16 - (m.data.length % 16)) % 16;
  m.data.zeros(pad);
  m.grids.push(grid);
}

function encode(m: Mutable): Uint8Array<ArrayBuffer> {
  const rootCount = m.roots.length;
  const rootPadded = rootCount % 2 === 1 ? rootCount + 1 : rootCount;
  const dataSize = m.data.length;
  const dataPadded = Math.ceil(dataSize / 16) * 16;
  const total =
    HEADER_SIZE +
    m.grids.length * GRID_SIZE +
    rootPadded * ROOT_SIZE +
    m.uppers.length * UPPER_SIZE +
    m.lowers.length * LOWER_SIZE +
    m.leaves.length * LEAF_SIZE +
    dataPadded;
  const out = new Uint8Array(total);
  const u = new Uint32Array(out.buffer);
  u.set([PICOVDB_MAGIC[0], PICOVDB_MAGIC[1], 0, m.grids.length, m.uppers.length, m.lowers.length, m.leaves.length, dataPadded / 16], 0);
  let w = HEADER_SIZE / 4;
  for (const g of m.grids) {
    u.set(g, w);
    w += GRID_SIZE / 4;
  }
  for (const [k0, k1] of m.roots) {
    u[w++] = k0;
    u[w++] = k1;
  }
  if (rootCount % 2 === 1) w += 2; // zero padding root
  const writeNode = (n: NodeRecord, sizeBytes: number) => {
    u[w] = n.baseInside;
    u[w + 2] = n.baseActive; // high u32s stay 0
    u.set(n.elements, w + 4);
    w += sizeBytes / 4;
  };
  for (const n of m.uppers) writeNode(n, UPPER_SIZE);
  for (const n of m.lowers) writeNode(n, LOWER_SIZE);
  for (const n of m.leaves) writeNode(n, LEAF_SIZE);
  out.set(m.data.buf.subarray(0, dataSize), w * 4);
  return out;
}

/** Converts one or more FLOAT NanoVDB grid images into a single `.pvdb` file. */
export function convertToPicoVDB(inputs: GridInput[], opts: ConvertOptions = {}): ConvertResult {
  const vt = opts.valueType ?? "f32";
  const fixed = (opts.layout ?? "native") === "fixed";
  const m: Mutable = {
    grids: [],
    roots: [],
    uppers: [],
    lowers: [],
    leaves: [],
    data: new ByteSink(),
    diag: {
      grids: 0, roots: 0, uppers: 0, lowers: 0, leaves: 0, dataElems: 0,
      surfaceVoxels: 0, childlessRootTiles: 0, misindexedTileValues: 0,
    },
  };
  for (const input of inputs) convertGrid(m, input, vt, fixed);
  const bytes = encode(m);
  Object.assign(m.diag, {
    grids: m.grids.length,
    roots: m.roots.length,
    uppers: m.uppers.length,
    lowers: m.lowers.length,
    leaves: m.leaves.length,
  });
  return { bytes, diagnostics: m.diag };
}
