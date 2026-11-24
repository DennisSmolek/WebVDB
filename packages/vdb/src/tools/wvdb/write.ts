/**
 * write.ts — `.wvdb` writer (mjurczyk's WebVDB spec v0.5.x) from dense
 * volume levels and point lists.
 *
 * Portions derived from mjurczyk/openvdb (MIT), commit 75e7345:
 * `src/wvdb/bake.js` (downsample, getDensityStats, getBrickValues, encodeLod,
 * getPointLods, getPointsBox, the header assembly) and `writeWVDBFile` in
 * `src/openvdb/wvdb/format.js`. The goal is files his `WVDBLoader` reads, and,
 * given the same values and a zlib with the same output (Node's
 * `deflateSync(level 9)`, which his CLI uses), the same bytes. His choices we
 * keep (spec text in brackets where it only says SHOULD/MAY):
 *  - defaults: 3 LODs, 8 bits, sparse layout, 1024 bricks per page, 2^18
 *    points per points chunk;
 *  - coarse LODs are 2x box filters of the next finer LOD over the voxels
 *    that exist (§3.2.4), summed in float64, stored as float32;
 *  - a sparse LOD drops a brick only when every value (all channels,
 *    including the edge-replicated voxels outside the LOD) equals the
 *    background exactly (§3.4); a dense LOD stores every brick;
 *  - sparse pages hold the bricks with the largest sum of |v - background|
 *    first, ties in index order (§9 SHOULD);
 *  - `meanDensity` / `nonZeroVoxels` of the finest LOD for scalar grids (his
 *    8-bit `floor(clamp(v, 0, 1) * 255)` texels; level sets count the band);
 *  - header member order is his, so `JSON.stringify` gives his header bytes.
 *
 * Compression defaults to fflate `zlibSync` at level 9 (Node + browser). Its
 * deflate stream differs from Node's zlib, so for byte-identical output pass
 * `compress: (b) => zlib.deflateSync(b, { level: 9 })` in Node.
 */

import { VDBToolsError } from "../errors.js";
import { zlibSync } from "fflate";
import {
  WVDB_BRICK_DIM,
  WVDB_BRICK_VOXELS,
  WVDB_MAGIC,
  WVDB_POINTS_PAGE,
  WVDB_POINTS_VERSION,
  WVDB_PREAMBLE_SIZE,
  brickCounts,
  brickOrigin,
  encodeMask,
  encodePage,
  encodePoints,
  type WvdbBits,
  type WvdbBox,
  type WvdbChunkRef,
  type WvdbGridHeader,
  type WvdbHeader,
  type WvdbLayout,
  type WvdbPageRef,
  type WvdbPointsGridHeader,
  type WvdbPointsLodHeader,
  type WvdbPointsPageRef,
  type WvdbSource,
  type WvdbVec3,
  type WvdbVolumeGridHeader,
  type WvdbVolumeLodHeader,
} from "./format.js";

/** One dense level: channel planes one after the other, x fastest. */
export interface WvdbDenseLevel {
  /** Voxels per axis. */
  dimensions: WvdbVec3;
  /** The values: all channel planes one after the other (x fastest within a plane). */
  values: Float32Array;
  /**
   * Sparse layout only: the bricks to store, in page order (e.g. a decoded
   * level's `brickIndices`, to keep a file's order when re-encoding). Must
   * hold every brick that is not all background (§3.4). Default: his order
   * (most data first, ties by index).
   */
  brickOrder?: ArrayLike<number>;
}

/** A volume grid to write with {@link writeWvdb}. */
export interface WvdbVolumeGridInput {
  /** `"volume"` (the default). */
  type?: "volume";
  /** The grid name. */
  name: string;
  /** Default `"unknown"`. */
  class?: string;
  /** Default 1. */
  channels?: 1 | 3;
  /** One number, or 3 for a vector grid. */
  background: number | readonly number[];
  /** The grid's active-voxel box in world units, or null when it has none. */
  bbox: WvdbBox | null;
  /**
   * The finest LOD. Coarser LODs are built by `downsampleWvdbLevel` (§3.2).
   * Give `levels` instead to supply every LOD yourself.
   */
  finest?: WvdbDenseLevel;
  /** Every LOD, coarse first (`levels.length` must equal `lods`). */
  levels?: readonly WvdbDenseLevel[];
  /**
   * `meanDensity` / `nonZeroVoxels` for a scalar grid. Default true (his
   * writer always writes them for scalar grids); false omits them; an object
   * writes the given numbers.
   */
  stats?: boolean | { meanDensity: number; nonZeroVoxels: number };
}

/** A points grid to write with {@link writeWvdb} (makes the file version 2). */
export interface WvdbPointsGridInput {
  /** Always `"points"`. */
  type: "points";
  /** The grid name. */
  name: string;
  /** The grid class string. Default `"unknown"`. */
  class?: string;
  /** xyz per point, in the stored order (near indices should be near in space, §3.5). */
  positions: Float32Array;
  /** rgb per point in [0, 1], or null/absent. */
  colors?: Float32Array | null;
  /** Points of the source. Default: the point count before `maxPoints` thinning. */
  sourceCount?: number;
  /** Mean distance between stored points (world units). Default: estimated from the box (see `writeWvdb`). */
  spacing?: number;
}

/** A grid to write: a volume or a points grid. */
export type WvdbGridInput = WvdbVolumeGridInput | WvdbPointsGridInput;

/** Options of {@link writeWvdb}. */
export interface WriteWvdbOptions {
  /** LOD count. Default 3. */
  lods?: number;
  /** Quantization bits. Default 8. */
  bits?: WvdbBits;
  /** Default `"sparse"`. */
  layout?: WvdbLayout;
  /** Bricks per page chunk. Default 1024. */
  pageBricks?: number;
  /** Recorded as `bake.resolution`. Default: the longest side of the finest LOD of the first volume grid (else 128). */
  resolution?: number;
  /** Recorded as `bake.grids`. Default null. */
  gridNames?: string[] | null;
  /** Recorded as `bake.bbox` (the sequence box, §10). Default null. */
  bakeBBox?: WvdbBox | null;
  /** Keep every n-th point above this many per points grid (his `--max-points`). Default 4,000,000. */
  maxPoints?: number;
  /** `source` header member. Default `{ file: "", sha256: "" + 64 zeros }`. */
  source?: WvdbSource;
  /** zlib compressor. Default fflate `zlibSync(level 9)`. */
  compress?: (raw: Uint8Array) => Uint8Array;
}

/** What {@link writeWvdb} returns. */
export interface WriteWvdbResult {
  /** The file's bytes. */
  bytes: Uint8Array;
  /** The header that was written. */
  header: WvdbHeader;
}

const DEFAULTS = { lods: 3, bits: 8 as WvdbBits, layout: "sparse" as WvdbLayout, pageBricks: 1024, maxPoints: 4_000_000 };
const ZERO_SHA = "0".repeat(64);

const defaultCompress = (raw: Uint8Array): Uint8Array => zlibSync(raw, { level: 9 });

/** §3.2.3-5: one 2x box-filter step, each channel alone, over the voxels inside the LOD. */
export function downsampleWvdbLevel(level: WvdbDenseLevel, channels: number): WvdbDenseLevel {
  const [w, h, d] = level.dimensions;
  const dims: WvdbVec3 = [Math.ceil(w / 2), Math.ceil(h / 2), Math.ceil(d / 2)];
  const [ow, oh, od] = dims;
  const count = w * h * d;
  const outCount = ow * oh * od;
  const out = new Float32Array(outCount * channels);
  for (let c = 0; c < channels; c++) {
    const src = c * count;
    let o = c * outCount;
    for (let z = 0; z < od; z++) {
      for (let y = 0; y < oh; y++) {
        for (let x = 0; x < ow; x++) {
          let sum = 0;
          let n = 0;
          for (let dz = 0; dz < 2; dz++) {
            for (let dy = 0; dy < 2; dy++) {
              for (let dx = 0; dx < 2; dx++) {
                const sx = 2 * x + dx;
                const sy = 2 * y + dy;
                const sz = 2 * z + dz;
                if (sx < w && sy < h && sz < d) {
                  sum += level.values[src + sx + sy * w + sz * w * h]!;
                  n++;
                }
              }
            }
          }
          out[o++] = sum / n;
        }
      }
    }
  }
  return { dimensions: dims, values: out };
}

/** His `getDensityStats`: mean and count of the non-zero 8-bit density texels of one LOD. */
export function wvdbDensityStats(
  values: Float32Array,
  background: number,
  isLevelSet: boolean,
): { meanDensity: number; nonZeroVoxels: number } {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < values.length; i++) {
    const texel = isLevelSet
      ? Math.abs(values[i]!) < background
        ? 255
        : 0
      : (Math.min(Math.max(values[i]!, 0), 1) * 255) | 0;
    if (texel > 0) {
      sum += texel;
      count++;
    }
  }
  return { meanDensity: count > 0 ? sum / 255 / count : 0, nonZeroVoxels: count };
}

/** 512 values per channel; voxels outside the LOD copy the nearest voxel inside (§3.3). */
function brickValues(level: WvdbDenseLevel, index: number, channels: number): Float32Array {
  const [w, h, d] = level.dimensions;
  const [ox, oy, oz] = brickOrigin(level.dimensions, index);
  const count = w * h * d;
  const brick = new Float32Array(WVDB_BRICK_VOXELS * channels);
  let v = 0;
  for (let c = 0; c < channels; c++) {
    const plane = c * count;
    for (let z = 0; z < WVDB_BRICK_DIM; z++) {
      const sz = Math.min(oz + z, d - 1);
      for (let y = 0; y < WVDB_BRICK_DIM; y++) {
        const sy = Math.min(oy + y, h - 1);
        for (let x = 0; x < WVDB_BRICK_DIM; x++) {
          const sx = Math.min(ox + x, w - 1);
          brick[v++] = level.values[plane + sx + sy * w + sz * w * h]!;
        }
      }
    }
  }
  return brick;
}

interface EncodedLod {
  dimensions: WvdbVec3;
  bricks: number;
  mask: Uint8Array | null;
  pages: { bricks: number; bytes: Uint8Array }[];
}

function encodeLod(
  level: WvdbDenseLevel,
  background: readonly number[],
  channels: number,
  o: { layout: WvdbLayout; bits: WvdbBits; pageBricks: number; compress: (raw: Uint8Array) => Uint8Array },
): EncodedLod {
  const sparse = o.layout === "sparse";
  const [cx, cy, cz] = brickCounts(level.dimensions);
  const total = cx * cy * cz;
  const bricks: { index: number; weight: number }[] = [];
  for (let index = 0; index < total; index++) {
    const values = brickValues(level, index, channels);
    let isBackground = true;
    let weight = 0;
    for (let v = 0; v < values.length; v++) {
      const bg = background[Math.floor(v / WVDB_BRICK_VOXELS)]!;
      isBackground = isBackground && values[v] === bg;
      weight += Math.abs(values[v]! - bg);
    }
    if (sparse && isBackground) continue;
    bricks.push({ index, weight });
  }
  if (sparse) bricks.sort((a, b) => b.weight - a.weight || a.index - b.index);
  if (sparse && level.brickOrder) {
    const given = Array.from(level.brickOrder);
    const set = new Set(given);
    if (set.size !== given.length) throw new VDBToolsError("VDB_BAD_OPTION", "writeWvdb: brickOrder lists a brick twice");
    for (const i of given) {
      if (!Number.isInteger(i) || i < 0 || i >= total) throw new VDBToolsError("VDB_BAD_OPTION", `writeWvdb: brickOrder index ${i} is out of range`);
    }
    for (const b of bricks) {
      if (!set.has(b.index)) throw new VDBToolsError("VDB_BAD_OPTION", `writeWvdb: brickOrder omits brick ${b.index}, which is not all background`);
    }
    bricks.length = 0;
    for (const index of given) bricks.push({ index, weight: 0 });
  }

  const pages: EncodedLod["pages"] = [];
  for (let start = 0; start < bricks.length; start += o.pageBricks) {
    const page = bricks
      .slice(start, start + o.pageBricks)
      .map(({ index }) => ({ index, values: brickValues(level, index, channels) }));
    pages.push({ bricks: page.length, bytes: o.compress(encodePage(page, { layout: o.layout, bits: o.bits, channels })) });
  }
  return {
    dimensions: [...level.dimensions] as WvdbVec3,
    bricks: bricks.length,
    mask: sparse ? o.compress(encodeMask(bricks.map((b) => b.index), total)) : null,
    pages,
  };
}

/**
 * §3.5: point i is in the first LOD k where i is a multiple of 8^(lods-1-k).
 * @param count Points in the grid.
 * @param lods Number of LODs.
 * @returns Per LOD (coarse first), the stored indices of the points it adds.
 */
export function wvdbPointLods(count: number, lods: number): number[][] {
  return Array.from({ length: lods }, (_, lod) => {
    const stride = 8 ** (lods - 1 - lod);
    const coarser = stride * 8;
    const indices: number[] = [];
    for (let p = 0; p < count; p += stride) if (lod === 0 || p % coarser !== 0) indices.push(p);
    return indices;
  });
}

function gather(source: Float32Array, indices: readonly number[]): Float32Array {
  const out = new Float32Array(indices.length * 3);
  indices.forEach((p, i) => out.set(source.subarray(p * 3, p * 3 + 3), i * 3));
  return out;
}

function pointsBox(positions: Float32Array): WvdbBox | null {
  if (!positions.length) return null;
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i++) {
    min[i % 3] = Math.min(min[i % 3]!, positions[i]!);
    max[i % 3] = Math.max(max[i % 3]!, positions[i]!);
  }
  return [min.map(Math.fround) as WvdbVec3, max.map(Math.fround) as WvdbVec3];
}

function checkLevel(level: WvdbDenseLevel, channels: number, what: string): void {
  const [x, y, z] = level.dimensions;
  if (![x, y, z].every((n) => Number.isInteger(n) && n >= 1)) throw new VDBToolsError("VDB_BAD_OPTION", `writeWvdb: ${what}: dimensions must be integers >= 1`);
  if (level.values.length !== x * y * z * channels) {
    throw new VDBToolsError("VDB_BAD_OPTION", `writeWvdb: ${what}: ${level.values.length} values, expected ${x * y * z * channels} (X*Y*Z*channels)`);
  }
}

type PendingChunk = { bytes: Uint8Array; entry: WvdbChunkRef };

/**
 * Writes a `.wvdb` file. Volume grids give their finest LOD (or all LODs);
 * points grids give their points. All grids get `lods` LODs. Version 2 is
 * written iff a points grid is present (§11).
 *
 * Points `spacing`, when not given, is `cbrt(boxVolume / sourceCount) *
 * cbrt(stride)` (a uniform-density estimate; his CLI derives it from the
 * source voxel size instead).
 * @param grids The grids to write.
 * @param options LOD count, bits, layout, page size and the header fields.
 */
export function writeWvdb(grids: readonly WvdbGridInput[], options: WriteWvdbOptions = {}): WriteWvdbResult {
  const lods = options.lods ?? DEFAULTS.lods;
  const bits = options.bits ?? DEFAULTS.bits;
  const layout = options.layout ?? DEFAULTS.layout;
  const pageBricks = options.pageBricks ?? DEFAULTS.pageBricks;
  const maxPoints = options.maxPoints ?? DEFAULTS.maxPoints;
  const compress = options.compress ?? defaultCompress;
  if (!Number.isInteger(lods) || lods < 1) throw new VDBToolsError("VDB_BAD_OPTION", `writeWvdb: invalid LOD count ${lods}`);
  if (bits !== 8 && bits !== 16) throw new VDBToolsError("VDB_BAD_OPTION", `writeWvdb: bits must be 8 or 16 (got ${String(bits)})`);
  if (layout !== "dense" && layout !== "sparse") throw new VDBToolsError("VDB_BAD_OPTION", `writeWvdb: invalid layout ${String(layout)}`);
  if (!Number.isInteger(pageBricks) || pageBricks < 1) throw new VDBToolsError("VDB_BAD_OPTION", `writeWvdb: invalid pageBricks ${pageBricks}`);
  if (!Number.isInteger(maxPoints) || maxPoints < 1) throw new VDBToolsError("VDB_BAD_OPTION", `writeWvdb: invalid maxPoints ${maxPoints}`);
  const names = new Set<string>();
  for (const g of grids) {
    if (names.has(g.name)) throw new VDBToolsError("VDB_BAD_OPTION", `writeWvdb: grid name "${g.name}" is not unique`);
    names.add(g.name);
  }

  type Encoded =
    | { kind: "volume"; header: Omit<WvdbVolumeGridHeader, "lods">; lods: EncodedLod[] }
    | { kind: "points"; header: Omit<WvdbPointsGridHeader, "lods">; lods: { points: number; pages: { points: number; bytes: Uint8Array }[] }[] };

  let resolution = options.resolution;
  const encoded: Encoded[] = grids.map((grid): Encoded => {
    if (grid.type === "points") {
      const total = grid.positions.length / 3;
      if (!Number.isInteger(total)) throw new VDBToolsError("VDB_BAD_OPTION", `writeWvdb: "${grid.name}": positions length is not a multiple of 3`);
      if (grid.colors && grid.colors.length !== grid.positions.length) {
        throw new VDBToolsError("VDB_BAD_OPTION", `writeWvdb: "${grid.name}": colors must have 3 values per point`);
      }
      const stride = Math.max(1, Math.ceil(total / maxPoints));
      const count = total > 0 ? Math.floor((total - 1) / stride) + 1 : 0;
      let positions = grid.positions;
      let colors = grid.colors ?? null;
      if (stride > 1) {
        const keep = Array.from({ length: count }, (_, i) => i * stride);
        positions = gather(grid.positions, keep);
        colors = colors ? gather(colors, keep) : null;
      }
      const bbox = pointsBox(positions);
      let spacing = grid.spacing;
      if (spacing === undefined) {
        const size = bbox ? bbox[1].map((v, a) => v - bbox[0][a]!) : [0, 0, 0];
        const volume = size[0]! * size[1]! * size[2]!;
        const sourceCount = grid.sourceCount ?? total;
        spacing = sourceCount > 0 && volume > 0 ? Math.cbrt(volume / sourceCount) * Math.cbrt(stride) : 0;
      }
      return {
        kind: "points",
        header: {
          name: grid.name,
          type: "points",
          class: grid.class ?? "unknown",
          count,
          sourceCount: grid.sourceCount ?? total,
          spacing: Math.fround(spacing),
          colors: !!colors,
          bbox,
        },
        lods: wvdbPointLods(count, lods).map((indices) => {
          const pages: { points: number; bytes: Uint8Array }[] = [];
          for (let start = 0; start < indices.length; start += WVDB_POINTS_PAGE) {
            const page = indices.slice(start, start + WVDB_POINTS_PAGE);
            pages.push({ points: page.length, bytes: compress(encodePoints(gather(positions, page), colors ? gather(colors, page) : null)) });
          }
          return { points: indices.length, pages };
        }),
      };
    }

    const channels = grid.channels ?? 1;
    if (channels !== 1 && channels !== 3) throw new VDBToolsError("VDB_BAD_OPTION", `writeWvdb: "${grid.name}": channels must be 1 or 3`);
    const background = typeof grid.background === "number" ? [grid.background] : [...grid.background];
    if (background.length !== channels) throw new VDBToolsError("VDB_BAD_OPTION", `writeWvdb: "${grid.name}": background needs ${channels} value(s)`);
    const bg32 = background.map(Math.fround);
    let levels: WvdbDenseLevel[];
    if (grid.levels) {
      if (grid.levels.length !== lods) throw new VDBToolsError("VDB_BAD_OPTION", `writeWvdb: "${grid.name}": ${grid.levels.length} levels given, lods is ${lods}`);
      levels = [...grid.levels];
    } else if (grid.finest) {
      levels = [];
      levels[lods - 1] = grid.finest;
      for (let lod = lods - 2; lod >= 0; lod--) levels[lod] = downsampleWvdbLevel(levels[lod + 1]!, channels);
    } else {
      throw new VDBToolsError("VDB_BAD_OPTION", `writeWvdb: "${grid.name}": give finest or levels`);
    }
    levels.forEach((l, i) => checkLevel(l, channels, `"${grid.name}" LOD ${i}`));
    resolution ??= Math.max(...levels[lods - 1]!.dimensions);
    const gridClass = grid.class ?? "unknown";
    let stats: { meanDensity: number; nonZeroVoxels: number } | null = null;
    if (channels === 1 && grid.stats !== false) {
      stats =
        typeof grid.stats === "object"
          ? grid.stats
          : wvdbDensityStats(levels[lods - 1]!.values, bg32[0]!, gridClass === "level set");
    }
    return {
      kind: "volume",
      header: {
        name: grid.name,
        class: gridClass,
        channels,
        background: channels === 1 ? bg32[0]! : (bg32 as WvdbVec3),
        bbox: grid.bbox,
        ...(stats ? { meanDensity: stats.meanDensity, nonZeroVoxels: stats.nonZeroVoxels } : {}),
      },
      lods: levels.map((level) => encodeLod(level, bg32, channels, { layout, bits, pageBricks, compress })),
    };
  });

  // §9: LOD 0 of all grids, then LOD 1, ...; per grid the mask, then the pages.
  const chunks: PendingChunk[] = [];
  const headerLods: (WvdbVolumeLodHeader | WvdbPointsLodHeader)[][] = encoded.map(() => []);
  for (let lod = 0; lod < lods; lod++) {
    encoded.forEach((grid, gi) => {
      if (grid.kind === "points") {
        const { points, pages } = grid.lods[lod]!;
        headerLods[gi]![lod] = {
          points,
          pages: pages.map((page) => {
            const entry: WvdbPointsPageRef = { offset: 0, length: 0, points: page.points };
            chunks.push({ bytes: page.bytes, entry });
            return entry;
          }),
        };
        return;
      }
      const { dimensions, bricks, mask, pages } = grid.lods[lod]!;
      // Key order dimensions, bricks, mask, pages (his header bytes); pages are added after the mask.
      const entry = { dimensions, bricks } as WvdbVolumeLodHeader;
      if (mask) {
        entry.mask = { offset: 0, length: 0 };
        chunks.push({ bytes: mask, entry: entry.mask });
      }
      entry.pages = pages.map((page) => {
        const ref: WvdbPageRef = { offset: 0, length: 0, bricks: page.bricks };
        chunks.push({ bytes: page.bytes, entry: ref });
        return ref;
      });
      headerLods[gi]![lod] = entry;
    });
  }

  const hasPoints = encoded.some((g) => g.kind === "points");
  const header: WvdbHeader = {
    version: hasPoints ? WVDB_POINTS_VERSION : 1,
    source: options.source ?? { file: "", sha256: ZERO_SHA },
    bake: {
      grids: options.gridNames ?? null,
      resolution: resolution ?? 128,
      lods,
      bits,
      layout,
      pageBricks,
      bbox: options.bakeBBox ?? null,
      ...(hasPoints ? { maxPoints } : {}),
    },
    brickDim: WVDB_BRICK_DIM,
    grids: encoded.map((g, gi) => ({ ...g.header, lods: headerLods[gi] }) as WvdbGridHeader),
  };
  return { bytes: assembleWvdb(header, chunks), header };
}

/** Sets each chunk's offset/length, then lays out preamble + header JSON + chunks (§4). */
function assembleWvdb(header: WvdbHeader, chunks: PendingChunk[]): Uint8Array {
  let offset = 0;
  for (const { bytes, entry } of chunks) {
    entry.offset = offset;
    entry.length = bytes.byteLength;
    offset += bytes.byteLength;
  }
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const file = new Uint8Array(WVDB_PREAMBLE_SIZE + headerBytes.byteLength + offset);
  const view = new DataView(file.buffer);
  for (let i = 0; i < 4; i++) file[i] = WVDB_MAGIC.charCodeAt(i);
  view.setUint32(4, header.version, true);
  view.setUint32(8, headerBytes.byteLength, true);
  file.set(headerBytes, WVDB_PREAMBLE_SIZE);
  let at = WVDB_PREAMBLE_SIZE + headerBytes.byteLength;
  for (const { bytes } of chunks) {
    file.set(bytes, at);
    at += bytes.byteLength;
  }
  return file;
}
