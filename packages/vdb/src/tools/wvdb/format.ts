/**
 * format.ts — `.wvdb` (WebVDB, mjurczyk's spec v0.5.x) constants, header types
 * and the chunk codecs: page, mask and points chunks (spec §3.3, §6, §7).
 *
 * Portions derived from mjurczyk/openvdb (MIT), commit 75e7345:
 * `src/openvdb/wvdb/format.js` (encodeWVDBPage / decodeWVDBPage /
 * encodeWVDBMask / encodeWVDBPoints / decodeWVDBPoints). Written from the spec
 * (`src/docs/wvdb/README.md` at that commit); where the spec leaves a choice,
 * these follow his code so our bytes and decoded values match his:
 *  - min/max of a brick channel are taken over float32 values in float64 and
 *    then `Math.fround`ed; `scale = maxCode / (max - min)` in float64.
 *  - `Math.round` (halves toward +infinity) for every code, as the spec says.
 *  - decode computes `min + code * ((max - min) / maxCode)` in float64 and
 *    stores the float32 result; the end codes return min and max exactly.
 *  - a point colour that is not finite encodes as 1.0 (his encoder; the spec
 *    is silent on NaN).
 *
 * No `three`, no `node:` imports: Node and browser.
 */

import { WvdbFormatError } from "../errors.js";

/** ASCII `WVDB`. */
export const WVDB_MAGIC = "WVDB";
/** Magic + version + headerLength. */
export const WVDB_PREAMBLE_SIZE = 12;
/** Versions this reader understands (spec §11). Writers use 1 without points grids, 2 with. */
export const WVDB_SUPPORTED_VERSIONS = [1, 2] as const;
/** First version with points grids. */
export const WVDB_POINTS_VERSION = 2;

/** Voxels per brick edge (spec §3.3). */
export const WVDB_BRICK_DIM = 8;
/** Voxels per brick: `WVDB_BRICK_DIM` cubed. */
export const WVDB_BRICK_VOXELS = 512;
/** Points of one points page chunk, as his writer chooses (`1 << 18`). */
export const WVDB_POINTS_PAGE = 1 << 18;
const POINT_CODE_MAX = 0xffff;

/** Three numbers: a dimension triple, a position or an RGB value. */
export type WvdbVec3 = [number, number, number];
/** `[[minX, minY, minZ], [maxX, maxY, maxZ]]` in world units. */
export type WvdbBox = [WvdbVec3, WvdbVec3];
/** How a LOD stores its bricks: every brick (`dense`) or only the non-empty ones, listed by a mask (`sparse`). */
export type WvdbLayout = "dense" | "sparse";
/** Bits per stored density sample: 8 or 16 (spec §3). */
export type WvdbBits = 8 | 16;
/** Grid classes the spec names; any other string reads as `"unknown"` (spec §5.4). */
export type WvdbGridClass = "fog volume" | "level set" | "staggered" | "unknown";

/** A byte range of the file's data section (the compressed chunk of one page or mask). */
export interface WvdbChunkRef {
  /** Byte offset from the start of the data section. */
  offset: number;
  /** Byte length of the chunk. */
  length: number;
}
/** A page of a volume LOD: a chunk that holds up to `bake.pageBricks` bricks. */
export interface WvdbPageRef extends WvdbChunkRef {
  /** Bricks in this page. */
  bricks: number;
}
/** A page of a points LOD: a chunk that holds up to `WVDB_POINTS_PAGE` points. */
export interface WvdbPointsPageRef extends WvdbChunkRef {
  /** Points in this page. */
  points: number;
}

/** The header record of one LOD of a volume grid. */
export interface WvdbVolumeLodHeader {
  /** Voxels per axis at this LOD. */
  dimensions: WvdbVec3;
  /** Bricks stored (all of them for the dense layout, the non-empty ones for the sparse layout). */
  bricks: number;
  /** Present iff the LOD has the sparse layout. */
  mask?: WvdbChunkRef;
  /** The brick pages, in order. */
  pages: WvdbPageRef[];
}

/** The header record of a volume grid. */
export interface WvdbVolumeGridHeader {
  /** The OpenVDB grid name. */
  name: string;
  /** `"volume"`; absent in version 1 files. */
  type?: "volume";
  /** The OpenVDB grid class as a string (see {@link WvdbGridClass}). */
  class: string;
  /** Channels per voxel: 1 (scalar) or 3 (vector). Default 1. */
  channels?: 1 | 3;
  /** The value of voxels outside the grid, a number or an RGB triple. */
  background: number | WvdbVec3;
  /** The grid's active-voxel box in world units, or null when it has none. */
  bbox: WvdbBox | null;
  /** Mean density over the baked box, when the baker recorded it. */
  meanDensity?: number;
  /** Count of voxels with a non-zero density, when the baker recorded it. */
  nonZeroVoxels?: number;
  /** The LODs, coarse first (LOD `bake.lods - 1` is the finest). */
  lods: WvdbVolumeLodHeader[];
}

/** The header record of one LOD of a points grid. */
export interface WvdbPointsLodHeader {
  /** Points stored in this LOD's pages (the ones it adds to the coarser LODs). */
  points: number;
  /** The point pages, in order. */
  pages: WvdbPointsPageRef[];
}

/** The header record of a points grid (version 2). */
export interface WvdbPointsGridHeader {
  /** The OpenVDB grid name. */
  name: string;
  /** Always `"points"`. */
  type: "points";
  /** The OpenVDB grid class as a string. */
  class: string;
  /** Points in the grid, all LODs together. */
  count: number;
  /** Points in the source grid before `bake.maxPoints` thinned them. */
  sourceCount?: number;
  /** Mean point spacing in world units. */
  spacing?: number;
  /** True when each point carries an RGB colour. */
  colors: boolean;
  /** The points' box in world units, or null for an empty grid. */
  bbox: WvdbBox | null;
  /** The LODs, coarse first (LOD `bake.lods - 1` is the finest). */
  lods: WvdbPointsLodHeader[];
}

/** A grid's header record: a volume or a points grid (tell them apart with {@link isPointsGridHeader}). */
export type WvdbGridHeader = WvdbVolumeGridHeader | WvdbPointsGridHeader;

/** The settings a file was baked with (spec §5.2). */
export interface WvdbBake {
  /** Names of the grids baked, or null for every grid in the source. */
  grids: string[] | null;
  /** Voxels along the longest axis of the finest LOD. */
  resolution: number;
  /** Number of LODs baked. */
  lods: number;
  /** Bits per density sample. */
  bits: WvdbBits;
  /** Brick layout of the volume LODs. */
  layout: WvdbLayout;
  /** Bricks per page. */
  pageBricks: number;
  /** The box that was baked, in world units, or null for the grid's own box. */
  bbox?: WvdbBox | null;
  /** Cap on the points kept per points grid. */
  maxPoints?: number;
}

/** The file a `.wvdb` was baked from. */
export interface WvdbSource {
  /** The source file's name. */
  file: string;
  /** SHA-256 of the source file, hex. */
  sha256: string;
}

/** The JSON header of a `.wvdb` file (spec §5). */
export interface WvdbHeader {
  /** File version: 1, or 2 when the file has points grids. */
  version: number;
  /** The file this one was baked from. */
  source: WvdbSource;
  /** The bake settings. */
  bake: WvdbBake;
  /** Voxels per brick edge (8). */
  brickDim: number;
  /** One header per grid. */
  grids: WvdbGridHeader[];
}

/** Narrows a {@link WvdbGridHeader} to a points grid. */
export const isPointsGridHeader = (grid: WvdbGridHeader): grid is WvdbPointsGridHeader =>
  grid.type === "points";

// ---------------------------------------------------------------------------
// Brick math (spec §3.3)
// ---------------------------------------------------------------------------

export function brickCounts([x, y, z]: readonly number[]): WvdbVec3 {
  return [Math.ceil(x! / WVDB_BRICK_DIM), Math.ceil(y! / WVDB_BRICK_DIM), Math.ceil(z! / WVDB_BRICK_DIM)];
}

export function brickTotal(dimensions: readonly number[]): number {
  const [cx, cy, cz] = brickCounts(dimensions);
  return cx * cy * cz;
}

/** First voxel of brick `index` (x fastest, then y, then z). */
export function brickOrigin(dimensions: readonly number[], index: number): WvdbVec3 {
  const [countX, countY] = brickCounts(dimensions);
  return [
    (index % countX) * WVDB_BRICK_DIM,
    (Math.floor(index / countX) % countY) * WVDB_BRICK_DIM,
    Math.floor(index / (countX * countY)) * WVDB_BRICK_DIM,
  ];
}

/** Appendix B: inflated page size. `varying` = brick channels with min !== max. */
export function pageByteSize(
  count: number,
  channels: number,
  bits: WvdbBits,
  sparse: boolean,
  varying: number,
): number {
  return 4 + (sparse ? 4 * count : 0) + 8 * count * channels + varying * WVDB_BRICK_VOXELS * (bits / 8);
}

/** Appendix B: inflated points chunk size. */
export function pointsChunkByteSize(count: number, colors: boolean): number {
  return 4 + 24 + 6 * count + (colors ? 3 * count : 0);
}

/** §6.3: inflated mask size. */
export function maskByteSize(total: number): number {
  return Math.ceil(total / 32) * 4;
}

// ---------------------------------------------------------------------------
// Page chunk (spec §6.2, §7)
// ---------------------------------------------------------------------------

/** One brick for the encoder: `values` holds 512 values per channel, one channel after the other. */
export interface WvdbBrickInput {
  index: number;
  values: Float32Array;
}

export function encodePage(
  bricks: readonly WvdbBrickInput[],
  { layout, bits, channels }: { layout: WvdbLayout; bits: WvdbBits; channels: number },
): Uint8Array {
  const sparse = layout === "sparse";
  const count = bricks.length;
  const codeSize = bits / 8;
  const maxCode = bits === 16 ? 0xffff : 0xff;
  const ranges = count * channels;
  const mins = new Float64Array(ranges);
  const maxs = new Float64Array(ranges);
  let varying = 0;

  for (let b = 0; b < count; b++) {
    const values = bricks[b]!.values;
    for (let c = 0; c < channels; c++) {
      let min = Infinity;
      let max = -Infinity;
      const end = (c + 1) * WVDB_BRICK_VOXELS;
      for (let i = c * WVDB_BRICK_VOXELS; i < end; i++) {
        // Math.min/max (not `<`) so -0 and 0 order exactly as in his encoder.
        min = Math.min(min, values[i]!);
        max = Math.max(max, values[i]!);
      }
      mins[b * channels + c] = Math.fround(min);
      maxs[b * channels + c] = Math.fround(max);
      if (mins[b * channels + c] !== maxs[b * channels + c]) varying++;
    }
  }

  const size = pageByteSize(count, channels, bits, sparse, varying);
  const out = new Uint8Array(size);
  const view = new DataView(out.buffer);
  let offset = 0;
  view.setUint32(offset, count, true);
  offset += 4;
  if (sparse) {
    for (const { index } of bricks) {
      view.setUint32(offset, index, true);
      offset += 4;
    }
  }
  for (let i = 0; i < ranges; i++, offset += 4) view.setFloat32(offset, mins[i]!, true);
  for (let i = 0; i < ranges; i++, offset += 4) view.setFloat32(offset, maxs[i]!, true);

  for (let b = 0; b < count; b++) {
    const values = bricks[b]!.values;
    for (let c = 0; c < channels; c++) {
      const min = mins[b * channels + c]!;
      const max = maxs[b * channels + c]!;
      if (min === max) continue;
      const scale = maxCode / (max - min);
      const end = (c + 1) * WVDB_BRICK_VOXELS;
      for (let i = c * WVDB_BRICK_VOXELS; i < end; i++) {
        const code = Math.min(Math.max(Math.round((values[i]! - min) * scale), 0), maxCode);
        if (codeSize === 2) view.setUint16(offset, code, true);
        else out[offset] = code;
        offset += codeSize;
      }
    }
  }
  return out;
}

export interface DecodedPage {
  /** Brick indices, in page order. */
  indices: Uint32Array;
  /** Brick channel ranges, `b * channels + c`. */
  mins: Float32Array;
  maxs: Float32Array;
  /** 512 decoded values per brick channel, in brick channel order (brick 0 channel 0, brick 0 channel 1, ...). */
  values: Float32Array;
}

/**
 * Decodes one inflated page and validates its size exactly (Appendix B,
 * §14). `firstIndex` gives the implicit indices of a dense page.
 */
export function decodePage(
  bytes: Uint8Array,
  opts: { layout: WvdbLayout; bits: WvdbBits; channels: number; count: number; firstIndex: number; brickTotal: number },
): DecodedPage {
  const { layout, bits, channels, count, firstIndex, brickTotal: total } = opts;
  const sparse = layout === "sparse";
  const fixed = pageByteSize(count, channels, bits, sparse, 0);
  if (bytes.byteLength < fixed) {
    throw new WvdbFormatError(`page: ${bytes.byteLength} bytes, expected at least ${fixed} for ${count} bricks`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const stored = view.getUint32(0, true);
  if (stored !== count) {
    throw new WvdbFormatError(`page: brickCount ${stored} does not match the header (${count})`);
  }
  let offset = 4;
  const indices = new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    if (sparse) {
      indices[i] = view.getUint32(offset, true);
      offset += 4;
    } else {
      indices[i] = firstIndex + i;
    }
    if (indices[i]! >= total) {
      throw new WvdbFormatError(`page: brick index ${indices[i]} is not less than brickTotal ${total}`);
    }
  }
  const ranges = count * channels;
  const mins = new Float32Array(ranges);
  const maxs = new Float32Array(ranges);
  for (let i = 0; i < ranges; i++, offset += 4) mins[i] = view.getFloat32(offset, true);
  for (let i = 0; i < ranges; i++, offset += 4) maxs[i] = view.getFloat32(offset, true);
  let varying = 0;
  for (let i = 0; i < ranges; i++) if (mins[i] !== maxs[i]) varying++;
  const expected = pageByteSize(count, channels, bits, sparse, varying);
  if (bytes.byteLength !== expected) {
    throw new WvdbFormatError(`page: ${bytes.byteLength} bytes, expected ${expected} (Appendix B)`);
  }

  const maxCode = bits === 16 ? 0xffff : 0xff;
  const values = new Float32Array(ranges * WVDB_BRICK_VOXELS);
  for (let i = 0; i < ranges; i++) {
    const min = mins[i]!;
    const max = maxs[i]!;
    const target = i * WVDB_BRICK_VOXELS;
    if (min === max) {
      values.fill(min, target, target + WVDB_BRICK_VOXELS);
      continue;
    }
    const step = (max - min) / maxCode;
    for (let v = 0; v < WVDB_BRICK_VOXELS; v++) {
      let code: number;
      if (bits === 16) {
        code = view.getUint16(offset, true);
        offset += 2;
      } else {
        code = bytes[offset]!;
        offset += 1;
      }
      values[target + v] = code === maxCode ? max : min + code * step;
    }
  }
  return { indices, mins, maxs, values };
}

// ---------------------------------------------------------------------------
// Mask chunk (spec §6.3)
// ---------------------------------------------------------------------------

export function encodeMask(indices: Iterable<number>, total: number): Uint8Array {
  const words = new Uint32Array(Math.ceil(total / 32));
  for (const index of indices) words[index >>> 5]! |= 1 << (index & 31);
  const out = new Uint8Array(words.length * 4);
  const view = new DataView(out.buffer);
  words.forEach((word, i) => view.setUint32(i * 4, word, true));
  return out;
}

export function decodeMask(bytes: Uint8Array, total: number): Uint32Array {
  const expected = maskByteSize(total);
  if (bytes.byteLength !== expected) {
    throw new WvdbFormatError(`mask: ${bytes.byteLength} bytes, expected ${expected}`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const words = new Uint32Array(expected >>> 2);
  for (let i = 0; i < words.length; i++) words[i] = view.getUint32(i * 4, true);
  const tail = total & 31;
  if (tail !== 0 && (words[words.length - 1]! >>> tail) !== 0) {
    throw new WvdbFormatError(`mask: bits after brick ${total - 1} are set`);
  }
  return words;
}

export const isMaskBitSet = (words: Uint32Array, index: number): boolean =>
  (words[index >>> 5]! & (1 << (index & 31))) !== 0;

// ---------------------------------------------------------------------------
// Points chunk (spec §6.4)
// ---------------------------------------------------------------------------

/** `positions`: xyz per point. `colors`: rgb per point in [0, 1], or null. */
export function encodePoints(positions: Float32Array, colors: Float32Array | null): Uint8Array {
  const count = positions.length / 3;
  const out = new Uint8Array(pointsChunkByteSize(count, !!colors));
  const view = new DataView(out.buffer);
  view.setUint32(0, count, true);
  for (let axis = 0; axis < 3; axis++) {
    let min = Infinity;
    let max = -Infinity;
    for (let p = 0; p < count; p++) {
      min = Math.min(min, positions[p * 3 + axis]!);
      max = Math.max(max, positions[p * 3 + axis]!);
    }
    min = count > 0 ? Math.fround(min) : 0;
    max = count > 0 ? Math.fround(max) : 0;
    view.setFloat32(4 + axis * 4, min, true);
    view.setFloat32(16 + axis * 4, max, true);
    const scale = min === max ? 0 : POINT_CODE_MAX / (max - min);
    const high = 28 + axis * 2 * count;
    const low = high + count;
    for (let p = 0; p < count; p++) {
      const code = Math.min(Math.max(Math.round((positions[p * 3 + axis]! - min) * scale), 0), POINT_CODE_MAX);
      out[high + p] = code >>> 8;
      out[low + p] = code & 0xff;
    }
  }
  if (colors) {
    const start = 28 + 6 * count;
    for (let c = 0; c < 3; c++) {
      for (let p = 0; p < count; p++) {
        const v = colors[p * 3 + c]!;
        out[start + c * count + p] = Math.round(Math.min(Math.max(Number.isFinite(v) ? v : 1, 0), 1) * 255);
      }
    }
  }
  return out;
}

export interface DecodedPoints {
  positions: Float32Array;
  colors: Float32Array | null;
}

export function decodePoints(bytes: Uint8Array, { count, colors }: { count: number; colors: boolean }): DecodedPoints {
  const expected = pointsChunkByteSize(count, colors);
  if (bytes.byteLength !== expected) {
    throw new WvdbFormatError(`points: ${bytes.byteLength} bytes, expected ${expected}`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const stored = view.getUint32(0, true);
  if (stored !== count) {
    throw new WvdbFormatError(`points: pointCount ${stored} does not match the header (${count})`);
  }
  const positions = new Float32Array(count * 3);
  for (let axis = 0; axis < 3; axis++) {
    const min = view.getFloat32(4 + axis * 4, true);
    const max = view.getFloat32(16 + axis * 4, true);
    const step = (max - min) / POINT_CODE_MAX;
    const high = 28 + axis * 2 * count;
    const low = high + count;
    for (let p = 0; p < count; p++) {
      const code = (bytes[high + p]! << 8) | bytes[low + p]!;
      positions[p * 3 + axis] = code === POINT_CODE_MAX ? max : min + code * step;
    }
  }
  let rgb: Float32Array | null = null;
  if (colors) {
    rgb = new Float32Array(count * 3);
    const start = 28 + 6 * count;
    for (let c = 0; c < 3; c++) {
      for (let p = 0; p < count; p++) rgb[p * 3 + c] = bytes[start + c * count + p]! / 255;
    }
  }
  return { positions, colors: rgb };
}

/** A `.wvdb` file breaks the spec (bad magic, header, offsets, sizes); defined in `../errors.ts` (D32). */
export { WvdbFormatError };
