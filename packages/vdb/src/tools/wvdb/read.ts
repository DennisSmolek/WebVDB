/**
 * read.ts — `.wvdb` reader (mjurczyk's WebVDB spec v0.5.x, file versions 1
 * and 2): header parsing and validation (§4, §5, §14), whole-file decode, and
 * coarse-first decode from a prefix, a byte range, or a stream (§9, §12.2).
 *
 * Portions derived from mjurczyk/openvdb (MIT), commit 75e7345: the chunk
 * walk mirrors `getWVDBChunks` in `src/openvdb/wvdb/format.js`, and the
 * per-LOD layout rule (a LOD is sparse iff it has a `mask`) mirrors
 * `WVDBReader.getKernelGrid` in `src/three-openvdb/wvdb/WVDBReader.js`.
 *
 * Decoded volume LODs are dense `Float32Array`s: channel planes one after the
 * other (`values[c * X*Y*Z + x + y*X + z*X*Y]`), x fastest. Voxels of bricks a
 * sparse LOD does not store hold the background (§12.2.3); voxels of a brick
 * outside the LOD are dropped (§3.3). Values are float32, computed exactly as
 * his `decodeWVDBPage` computes them.
 */

import { unzlibSync } from "fflate";
import {
  WVDB_BRICK_DIM,
  WVDB_BRICK_VOXELS,
  WVDB_MAGIC,
  WVDB_POINTS_VERSION,
  WVDB_PREAMBLE_SIZE,
  WVDB_SUPPORTED_VERSIONS,
  WvdbFormatError,
  brickCounts,
  brickOrigin,
  brickTotal,
  decodeMask,
  decodePage,
  decodePoints,
  isMaskBitSet,
  isPointsGridHeader,
  maskByteSize,
  pageByteSize,
  pointsChunkByteSize,
  type WvdbBake,
  type WvdbBits,
  type WvdbBox,
  type WvdbGridClass,
  type WvdbGridHeader,
  type WvdbHeader,
  type WvdbLayout,
  type WvdbPointsGridHeader,
  type WvdbSource,
  type WvdbVec3,
  type WvdbVolumeGridHeader,
} from "./format.js";

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

export interface WvdbHeaderInfo {
  header: WvdbHeader;
  /** File version from the preamble (equals `header.version`). */
  version: number;
  /** Absolute file offset of the data section; chunk offsets are relative to it. */
  dataStart: number;
}

export interface ReadWvdbHeaderOptions {
  /**
   * Total file length, when known. Enables the §14 checks that need it:
   * `headerLength <= fileLength - 12` and every chunk inside the data section.
   */
  fileLength?: number;
  /** Refuse a LOD whose `X * Y * Z * channels` exceeds this before allocating (§14). Default 2^28. */
  maxVoxels?: number;
  /** Also check the §9 chunk order: coarse to fine, grid order, no gaps. Default false. */
  strictChunkOrder?: boolean;
}

const DEFAULT_MAX_VOXELS = 2 ** 28;
const KNOWN_CLASSES: readonly string[] = ["fog volume", "level set", "staggered", "unknown"];

/** §5.4: any other class string reads as `"unknown"`. */
export function normalizeWvdbClass(value: string): WvdbGridClass {
  return (KNOWN_CLASSES.includes(value) ? value : "unknown") as WvdbGridClass;
}

/**
 * Reads the preamble and header. Returns `null` while `bytes` (a prefix of
 * the file) does not yet hold the whole header; throws `WvdbFormatError` on a
 * bad magic, an unsupported version, or a header that breaks §5/§14.
 */
export function readWvdbHeader(bytes: Uint8Array, opts: ReadWvdbHeaderOptions = {}): WvdbHeaderInfo | null {
  if (bytes.byteLength >= 4) {
    for (let i = 0; i < 4; i++) {
      if (bytes[i] !== WVDB_MAGIC.charCodeAt(i)) throw new WvdbFormatError("not a WVDB file (bad magic)");
    }
  }
  if (bytes.byteLength < WVDB_PREAMBLE_SIZE) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint32(4, true);
  if (!(WVDB_SUPPORTED_VERSIONS as readonly number[]).includes(version)) {
    throw new WvdbFormatError(`version ${version} is not supported (this reader reads 1 and 2)`);
  }
  const headerLength = view.getUint32(8, true);
  if (opts.fileLength !== undefined && headerLength > opts.fileLength - WVDB_PREAMBLE_SIZE) {
    throw new WvdbFormatError(`headerLength ${headerLength} is more than the file length minus 12`);
  }
  const dataStart = WVDB_PREAMBLE_SIZE + headerLength;
  if (bytes.byteLength < dataStart) return null;

  let parsed: unknown;
  try {
    // fatal: invalid UTF-8 throws; ignoreBOM: a BOM stays in the text, so JSON.parse rejects it (§4).
    const text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(
      bytes.subarray(WVDB_PREAMBLE_SIZE, dataStart),
    );
    parsed = JSON.parse(text);
  } catch (err) {
    throw new WvdbFormatError(`header is not valid UTF-8 JSON: ${(err as Error).message}`);
  }
  const header = validateHeader(parsed, version, opts);
  if (opts.fileLength !== undefined) {
    const dataLength = opts.fileLength - dataStart;
    for (const chunk of wvdbChunks(header)) {
      if (chunk.offset + chunk.length > dataLength) {
        throw new WvdbFormatError(
          `chunk (${chunk.kind}, grid ${chunk.gridIndex}, LOD ${chunk.lod}) ends at ${chunk.offset + chunk.length}, ` +
            `past the data section (${dataLength} bytes)`,
        );
      }
    }
  }
  if (opts.strictChunkOrder) checkChunkOrder(header, opts.fileLength !== undefined ? opts.fileLength - dataStart : undefined);
  return { header, version, dataStart };
}

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const isUint = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v) && v >= 0;
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

function fail(path: string, what: string): never {
  throw new WvdbFormatError(`header ${path}: ${what}`);
}

function checkBox(v: unknown, path: string): void {
  if (v === null) return;
  if (!Array.isArray(v) || v.length !== 2 || !v.every((c) => Array.isArray(c) && c.length === 3 && c.every(isNum))) {
    fail(path, "must be [[minX, minY, minZ], [maxX, maxY, maxZ]] or null");
  }
}

function checkChunkRef(v: unknown, path: string, countKey?: "bricks" | "points"): void {
  if (!isObject(v) || !isUint(v.offset) || !isUint(v.length)) fail(path, "must be { offset, length } of non-negative integers");
  if (countKey && (!isUint(v[countKey]) || (v[countKey] as number) < 1)) fail(`${path}.${countKey}`, "must be an integer >= 1");
}

function validateHeader(h: unknown, version: number, opts: ReadWvdbHeaderOptions): WvdbHeader {
  const maxVoxels = opts.maxVoxels ?? DEFAULT_MAX_VOXELS;
  if (!isObject(h)) fail("", "must be a JSON object");
  if (h.version !== version) fail(".version", `must equal the preamble version ${version} (got ${String(h.version)})`);
  if (!isObject(h.source) || typeof h.source.file !== "string" || typeof h.source.sha256 !== "string") {
    fail(".source", "must be { file: string, sha256: string }");
  }
  if (h.brickDim !== WVDB_BRICK_DIM) fail(".brickDim", `must be 8 (got ${String(h.brickDim)})`);
  const bake = h.bake;
  if (!isObject(bake)) fail(".bake", "must be an object");
  if (bake.bits !== 8 && bake.bits !== 16) fail(".bake.bits", "must be 8 or 16");
  if (!isUint(bake.lods) || bake.lods < 1) fail(".bake.lods", "must be an integer >= 1");
  if (!Array.isArray(h.grids)) fail(".grids", "must be an array");
  const lods = bake.lods;
  const names = new Set<string>();

  h.grids.forEach((g: unknown, gi: number) => {
    const p = `.grids[${gi}]`;
    if (!isObject(g)) fail(p, "must be an object");
    if (typeof g.name !== "string") fail(`${p}.name`, "must be a string");
    if (names.has(g.name)) fail(`${p}.name`, `"${g.name}" is not unique`);
    names.add(g.name);
    if (typeof g.class !== "string") fail(`${p}.class`, "must be a string");
    if (!Array.isArray(g.lods) || g.lods.length !== lods) fail(`${p}.lods`, `must have bake.lods (${lods}) entries`);
    checkBox(g.bbox, `${p}.bbox`);
    const type = g.type ?? "volume";

    if (type === "points") {
      if (version < WVDB_POINTS_VERSION) fail(`${p}.type`, "points grids need file version 2");
      if (!isUint(g.count)) fail(`${p}.count`, "must be a non-negative integer");
      if (typeof g.colors !== "boolean") fail(`${p}.colors`, "must be a boolean");
      let total = 0;
      (g.lods as unknown[]).forEach((lod, li) => {
        const lp = `${p}.lods[${li}]`;
        if (!isObject(lod) || !isUint(lod.points) || !Array.isArray(lod.pages)) fail(lp, "must be { points, pages[] }");
        let sum = 0;
        lod.pages.forEach((page: unknown, pi: number) => {
          checkChunkRef(page, `${lp}.pages[${pi}]`, "points");
          sum += (page as { points: number }).points;
        });
        if (sum !== lod.points) fail(`${lp}.points`, `${lod.points} does not equal the sum of its pages (${sum})`);
        total += sum;
      });
      if (total !== g.count) fail(`${p}.count`, `${g.count} does not equal the sum of points of all LODs (${total})`);
      return;
    }
    if (type !== "volume") fail(`${p}.type`, `unknown grid type "${String(type)}"`);

    const channels = g.channels ?? 1;
    if (channels !== 1 && channels !== 3) fail(`${p}.channels`, "must be 1 or 3");
    if (channels === 1 ? !isNum(g.background) : !(Array.isArray(g.background) && g.background.length === 3 && g.background.every(isNum))) {
      fail(`${p}.background`, channels === 1 ? "must be a number" : "must be an array of 3 numbers");
    }
    (g.lods as unknown[]).forEach((lod, li) => {
      const lp = `${p}.lods[${li}]`;
      if (!isObject(lod)) fail(lp, "must be an object");
      const dims = lod.dimensions;
      if (!Array.isArray(dims) || dims.length !== 3 || !dims.every((d) => isUint(d) && d >= 1)) {
        fail(`${lp}.dimensions`, "must be 3 integers >= 1");
      }
      const voxels = (dims[0] as number) * (dims[1] as number) * (dims[2] as number) * channels;
      if (voxels > maxVoxels) fail(`${lp}.dimensions`, `${voxels} values is more than maxVoxels (${maxVoxels})`);
      if (!isUint(lod.bricks)) fail(`${lp}.bricks`, "must be a non-negative integer");
      if (!Array.isArray(lod.pages)) fail(`${lp}.pages`, "must be an array");
      if (lod.mask !== undefined) checkChunkRef(lod.mask, `${lp}.mask`);
      let sum = 0;
      lod.pages.forEach((page: unknown, pi: number) => {
        checkChunkRef(page, `${lp}.pages[${pi}]`, "bricks");
        sum += (page as { bricks: number }).bricks;
      });
      if (sum !== lod.bricks) fail(`${lp}.bricks`, `${lod.bricks} does not equal the sum of its pages (${sum})`);
      const total = brickTotal(dims as number[]);
      if (lod.mask === undefined && lod.bricks !== total) {
        fail(`${lp}.bricks`, `a dense LOD stores all ${total} bricks (got ${lod.bricks})`);
      }
      if (lod.bricks > total) fail(`${lp}.bricks`, `${lod.bricks} is more than brickTotal ${total}`);
    });
  });
  return h as unknown as WvdbHeader;
}

// ---------------------------------------------------------------------------
// Chunks and byte ranges (§9)
// ---------------------------------------------------------------------------

export interface WvdbChunkInfo {
  gridIndex: number;
  lod: number;
  kind: "mask" | "page" | "points";
  /** Relative to dataStart. */
  offset: number;
  length: number;
  /** page: bricks in the page; points: points in the chunk; mask: 0. */
  count: number;
  /** page: index of the first brick of a dense page. points: stored index of the first point. */
  first: number;
}

/** All chunks in §9 writer order (LOD, then grid, then mask, then pages). */
export function wvdbChunks(header: WvdbHeader): WvdbChunkInfo[] {
  const out: WvdbChunkInfo[] = [];
  const lods = header.bake.lods;
  const firstPoint = header.grids.map(() => 0);
  for (let lod = 0; lod < lods; lod++) {
    header.grids.forEach((grid, gridIndex) => {
      if (isPointsGridHeader(grid)) {
        for (const page of grid.lods[lod]!.pages) {
          out.push({ gridIndex, lod, kind: "points", offset: page.offset, length: page.length, count: page.points, first: firstPoint[gridIndex]! });
          firstPoint[gridIndex]! += page.points;
        }
        return;
      }
      const entry = grid.lods[lod]!;
      if (entry.mask) out.push({ gridIndex, lod, kind: "mask", offset: entry.mask.offset, length: entry.mask.length, count: 0, first: 0 });
      let first = 0;
      for (const page of entry.pages) {
        out.push({ gridIndex, lod, kind: "page", offset: page.offset, length: page.length, count: page.bricks, first });
        first += page.bricks;
      }
    });
  }
  return out;
}

function checkChunkOrder(header: WvdbHeader, dataLength?: number): void {
  let end = 0;
  for (const c of wvdbChunks(header)) {
    if (c.offset !== end) {
      throw new WvdbFormatError(`§9 chunk order: ${c.kind} of grid ${c.gridIndex} LOD ${c.lod} is at ${c.offset}, expected ${end}`);
    }
    end += c.length;
  }
  if (dataLength !== undefined && end !== dataLength) {
    throw new WvdbFormatError(`§9 chunk order: chunks end at ${end} but the data section has ${dataLength} bytes`);
  }
}

/**
 * Absolute file byte range `[start, end)` that holds the chunks of `lod` for
 * all grids (one contiguous range in a conforming file, §9). `start === end`
 * when the LOD has no chunks.
 */
export function wvdbLodByteRange(info: WvdbHeaderInfo, lod: number): { start: number; end: number } {
  checkLodIndex(info.header, lod);
  let start = Infinity;
  let end = -Infinity;
  for (const c of wvdbChunks(info.header)) {
    if (c.lod !== lod) continue;
    start = Math.min(start, c.offset);
    end = Math.max(end, c.offset + c.length);
  }
  if (start === Infinity) {
    const at = lod === 0 ? info.dataStart : wvdbPrefixLength(info, lod - 1);
    return { start: at, end: at };
  }
  return { start: info.dataStart + start, end: info.dataStart + end };
}

/** Bytes from the start of the file needed to decode LODs 0..lod (header included). */
export function wvdbPrefixLength(info: WvdbHeaderInfo, lod: number): number {
  checkLodIndex(info.header, lod);
  let end = 0;
  for (const c of wvdbChunks(info.header)) if (c.lod <= lod) end = Math.max(end, c.offset + c.length);
  return info.dataStart + end;
}

function checkLodIndex(header: WvdbHeader, lod: number): void {
  if (!Number.isInteger(lod) || lod < 0 || lod >= header.bake.lods) {
    throw new RangeError(`wvdb: LOD ${lod} is out of range (0..${header.bake.lods - 1})`);
  }
}

// ---------------------------------------------------------------------------
// Decoded data
// ---------------------------------------------------------------------------

export interface WvdbVolumeLevel {
  kind: "volume";
  lod: number;
  dimensions: WvdbVec3;
  layout: WvdbLayout;
  channels: 1 | 3;
  brickCounts: WvdbVec3;
  brickTotal: number;
  /** Mask words of a sparse LOD (bit `index` set = stored); null for dense. */
  mask: Uint32Array | null;
  /** Stored brick indices in file order (dense: 0..brickTotal-1). */
  brickIndices: Uint32Array;
  /** Brick channel ranges in file order, `b * channels + c`. */
  mins: Float32Array;
  maxs: Float32Array;
  /** Dense decoded values: channel planes, x fastest; background where no brick is stored. */
  values: Float32Array;
}

export interface WvdbPointsLevel {
  kind: "points";
  lod: number;
  /** Stored index of the first point of this LOD within the LOD-ordered point list. */
  firstPoint: number;
  count: number;
  /** xyz per point. */
  positions: Float32Array;
  /** rgb per point in [0, 1], or null when the grid has no colours. */
  colors: Float32Array | null;
}

export type WvdbLevel = WvdbVolumeLevel | WvdbPointsLevel;

interface WvdbGridBase {
  name: string;
  /** Normalized per §5.4 (unknown strings become `"unknown"`). */
  class: WvdbGridClass;
  bbox: WvdbBox | null;
  header: WvdbGridHeader;
}

export interface WvdbVolumeGrid extends WvdbGridBase {
  kind: "volume";
  channels: 1 | 3;
  /** One value per channel. */
  background: number[];
  meanDensity?: number;
  nonZeroVoxels?: number;
  header: WvdbVolumeGridHeader;
  /** Decoded LODs, coarse first (`levels[k].lod === k`). */
  levels: WvdbVolumeLevel[];
}

export interface WvdbPointsGrid extends WvdbGridBase {
  kind: "points";
  count: number;
  sourceCount?: number;
  spacing?: number;
  colors: boolean;
  header: WvdbPointsGridHeader;
  /** Decoded LODs, coarse first. LOD k adds its points to LODs 0..k-1 (§3.5). */
  levels: WvdbPointsLevel[];
}

export type WvdbGrid = WvdbVolumeGrid | WvdbPointsGrid;

export interface WvdbFile {
  version: number;
  header: WvdbHeader;
  dataStart: number;
  source: WvdbSource;
  bake: WvdbBake;
  grids: WvdbGrid[];
  /** Number of LODs decoded (LODs 0..lodsDecoded-1). */
  lodsDecoded: number;
  /** True when every LOD of the file is decoded. */
  complete: boolean;
}

/** Inflates one zlib chunk, stopping past `maxSize` bytes (§14). */
export function inflateWvdbChunk(bytes: Uint8Array, maxSize: number): Uint8Array {
  const out = new Uint8Array(maxSize + 1);
  let result: Uint8Array;
  try {
    result = unzlibSync(bytes, { out });
  } catch (err) {
    throw new WvdbFormatError(`chunk does not inflate: ${(err as Error).message}`);
  }
  if (result.byteLength > maxSize) throw new WvdbFormatError(`chunk inflates past its size limit (${maxSize} bytes)`);
  return result;
}

const backgroundOf = (grid: WvdbVolumeGridHeader): number[] =>
  typeof grid.background === "number" ? [grid.background] : [...grid.background];

function sliceChunk(bytes: Uint8Array, byteOffset: number, info: WvdbHeaderInfo, c: WvdbChunkInfo): Uint8Array {
  const start = info.dataStart + c.offset - byteOffset;
  const end = start + c.length;
  if (start < 0 || end > bytes.byteLength) {
    throw new WvdbFormatError(
      `chunk (${c.kind}, grid ${c.gridIndex}, LOD ${c.lod}) at file bytes ${start + byteOffset}..${end + byteOffset} ` +
        `is not inside the given bytes (${byteOffset}..${byteOffset + bytes.byteLength})`,
    );
  }
  return bytes.subarray(start, end);
}

/**
 * Decodes LOD `lod` of every grid. `bytes` holds file bytes starting at file
 * offset `byteOffset` (0 for the whole file or a prefix; `range.start` for a
 * range read from `wvdbLodByteRange`). Returns one level per grid, in header
 * order.
 */
export function decodeWvdbLod(info: WvdbHeaderInfo, lod: number, bytes: Uint8Array, byteOffset = 0): WvdbLevel[] {
  checkLodIndex(info.header, lod);
  const { header } = info;
  const bits = header.bake.bits as WvdbBits;
  const chunks = wvdbChunks(header).filter((c) => c.lod === lod);
  return header.grids.map((grid, gridIndex) => {
    const own = chunks.filter((c) => c.gridIndex === gridIndex);
    return isPointsGridHeader(grid)
      ? decodePointsLod(info, grid, lod, own, bytes, byteOffset)
      : decodeVolumeLod(info, grid, lod, own, bits, bytes, byteOffset);
  });
}

function decodeVolumeLod(
  info: WvdbHeaderInfo,
  grid: WvdbVolumeGridHeader,
  lod: number,
  chunks: WvdbChunkInfo[],
  bits: WvdbBits,
  bytes: Uint8Array,
  byteOffset: number,
): WvdbVolumeLevel {
  const entry = grid.lods[lod]!;
  const channels = (grid.channels ?? 1) as 1 | 3;
  const dimensions = [...entry.dimensions] as WvdbVec3;
  const [X, Y, Z] = dimensions;
  const counts = brickCounts(dimensions);
  const total = brickTotal(dimensions);
  const layout: WvdbLayout = entry.mask ? "sparse" : "dense";
  const sparse = layout === "sparse";
  const background = backgroundOf(grid);
  const plane = X * Y * Z;
  const values = new Float32Array(plane * channels);
  for (let c = 0; c < channels; c++) values.fill(background[c]!, c * plane, (c + 1) * plane);

  let mask: Uint32Array | null = null;
  const brickIndices = new Uint32Array(entry.bricks);
  const mins = new Float32Array(entry.bricks * channels);
  const maxs = new Float32Array(entry.bricks * channels);
  const seen = sparse ? new Uint32Array(Math.ceil(total / 32)) : null;
  let stored = 0;

  for (const c of chunks) {
    const raw = sliceChunk(bytes, byteOffset, info, c);
    if (c.kind === "mask") {
      mask = decodeMask(inflateWvdbChunk(raw, maskByteSize(total)), total);
      continue;
    }
    const maxSize = pageByteSize(c.count, channels, bits, sparse, c.count * channels);
    const page = decodePage(inflateWvdbChunk(raw, maxSize), {
      layout,
      bits,
      channels,
      count: c.count,
      firstIndex: c.first,
      brickTotal: total,
    });
    brickIndices.set(page.indices, stored);
    mins.set(page.mins, stored * channels);
    maxs.set(page.maxs, stored * channels);
    stored += c.count;
    for (let b = 0; b < page.indices.length; b++) {
      const index = page.indices[b]!;
      if (seen) {
        if (isMaskBitSet(seen, index)) throw new WvdbFormatError(`LOD ${lod} of "${grid.name}": brick ${index} is stored twice`);
        seen[index >>> 5]! |= 1 << (index & 31);
      }
      const [ox, oy, oz] = brickOrigin(dimensions, index);
      const nx = Math.min(WVDB_BRICK_DIM, X - ox);
      const ny = Math.min(WVDB_BRICK_DIM, Y - oy);
      const nz = Math.min(WVDB_BRICK_DIM, Z - oz);
      for (let ch = 0; ch < channels; ch++) {
        const src = (b * channels + ch) * WVDB_BRICK_VOXELS;
        const dst = ch * plane;
        for (let k = 0; k < nz; k++) {
          for (let j = 0; j < ny; j++) {
            const row = dst + ox + (oy + j) * X + (oz + k) * X * Y;
            const from = src + WVDB_BRICK_DIM * j + 64 * k;
            for (let i = 0; i < nx; i++) values[row + i] = page.values[from + i]!;
          }
        }
      }
    }
  }

  if (sparse) {
    if (!mask) throw new WvdbFormatError(`LOD ${lod} of "${grid.name}": sparse LOD without a mask chunk`);
    for (let w = 0; w < mask.length; w++) {
      if (mask[w] !== seen![w]) {
        throw new WvdbFormatError(`LOD ${lod} of "${grid.name}": the mask does not match the stored brick indices (§6.3)`);
      }
    }
  }

  return {
    kind: "volume",
    lod,
    dimensions,
    layout,
    channels,
    brickCounts: counts,
    brickTotal: total,
    mask,
    brickIndices,
    mins,
    maxs,
    values,
  };
}

function decodePointsLod(
  info: WvdbHeaderInfo,
  grid: WvdbPointsGridHeader,
  lod: number,
  chunks: WvdbChunkInfo[],
  bytes: Uint8Array,
  byteOffset: number,
): WvdbPointsLevel {
  const count = grid.lods[lod]!.points;
  const positions = new Float32Array(count * 3);
  const colors = grid.colors ? new Float32Array(count * 3) : null;
  const firstPoint = chunks.length ? chunks[0]!.first : grid.lods.slice(0, lod).reduce((s, l) => s + l.points, 0);
  let at = 0;
  for (const c of chunks) {
    const raw = sliceChunk(bytes, byteOffset, info, c);
    const decoded = decodePoints(inflateWvdbChunk(raw, pointsChunkByteSize(c.count, grid.colors)), {
      count: c.count,
      colors: grid.colors,
    });
    positions.set(decoded.positions, at * 3);
    if (colors && decoded.colors) colors.set(decoded.colors, at * 3);
    at += c.count;
  }
  return { kind: "points", lod, firstPoint, count, positions, colors };
}

function makeGrids(header: WvdbHeader): WvdbGrid[] {
  return header.grids.map((g): WvdbGrid => {
    if (isPointsGridHeader(g)) {
      return {
        kind: "points",
        name: g.name,
        class: normalizeWvdbClass(g.class),
        bbox: g.bbox,
        count: g.count,
        ...(g.sourceCount !== undefined ? { sourceCount: g.sourceCount } : {}),
        ...(g.spacing !== undefined ? { spacing: g.spacing } : {}),
        colors: g.colors,
        header: g,
        levels: [],
      };
    }
    return {
      kind: "volume",
      name: g.name,
      class: normalizeWvdbClass(g.class),
      bbox: g.bbox,
      channels: (g.channels ?? 1) as 1 | 3,
      background: backgroundOf(g),
      ...(g.meanDensity !== undefined ? { meanDensity: g.meanDensity } : {}),
      ...(g.nonZeroVoxels !== undefined ? { nonZeroVoxels: g.nonZeroVoxels } : {}),
      header: g,
      levels: [],
    };
  });
}

function addLevels(grids: WvdbGrid[], levels: WvdbLevel[]): void {
  levels.forEach((level, i) => (grids[i]!.levels as WvdbLevel[]).push(level));
}

export interface ReadWvdbOptions extends Omit<ReadWvdbHeaderOptions, "fileLength"> {
  /** Decode LODs 0..maxLod only. Default: all. */
  maxLod?: number;
  /**
   * `bytes` is a prefix of the file (a stream so far, or a `Range: bytes=0-N`
   * read): decode the LODs it holds completely and stop. Default false: the
   * bytes are the whole file, and every chunk must be inside them.
   */
  partial?: boolean;
}

/**
 * Reads a `.wvdb` file (or, with `partial`, a prefix of one) and decodes its
 * LODs coarse to fine.
 */
export function readWvdb(input: Uint8Array | ArrayBuffer, opts: ReadWvdbOptions = {}): WvdbFile {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  const { maxLod, partial = false, ...headerOpts } = opts;
  const info = readWvdbHeader(bytes, partial ? headerOpts : { ...headerOpts, fileLength: bytes.byteLength });
  if (!info) throw new WvdbFormatError(`only ${bytes.byteLength} bytes: the header is incomplete`);
  const { header } = info;
  const grids = makeGrids(header);
  const last = Math.min(header.bake.lods - 1, maxLod ?? Infinity);
  let decoded = 0;
  for (let lod = 0; lod <= last; lod++) {
    if (partial && wvdbPrefixLength(info, lod) > bytes.byteLength) break;
    addLevels(grids, decodeWvdbLod(info, lod, bytes, 0));
    decoded++;
  }
  return {
    version: info.version,
    header,
    dataStart: info.dataStart,
    source: header.source,
    bake: header.bake,
    grids,
    lodsDecoded: decoded,
    complete: decoded === header.bake.lods,
  };
}

export interface WvdbLodEvent {
  lod: number;
  /** One level per grid, in header order. */
  levels: WvdbLevel[];
}

/**
 * Push-style coarse-first decoder for a `.wvdb` download: feed it the bytes
 * as they arrive; each `push` returns the LODs that became complete.
 *
 *     const reader = new WvdbStreamReader();
 *     for await (const chunk of response.body) for (const e of reader.push(chunk)) show(e);
 *
 * It keeps only the bytes up to the end of the last LOD it will decode.
 */
export class WvdbStreamReader {
  private buffer = new Uint8Array(0);
  private received = 0;
  private next = 0;
  private finalLod = 0;
  info: WvdbHeaderInfo | null = null;
  /** The decoded file so far (grids gain levels as LODs complete); null before the header. */
  file: WvdbFile | null = null;

  constructor(private readonly opts: Omit<ReadWvdbOptions, "partial"> = {}) {}

  get header(): WvdbHeader | null {
    return this.info?.header ?? null;
  }

  /** True when every LOD up to `maxLod` is decoded. */
  get done(): boolean {
    return this.info !== null && this.next > this.finalLod;
  }

  /** File bytes needed to finish (header known), or null before the header. */
  get bytesTotal(): number | null {
    return this.info ? wvdbPrefixLength(this.info, this.finalLod) : null;
  }

  push(chunk: Uint8Array): WvdbLodEvent[] {
    if (this.done) return [];
    const limit = this.info ? this.bytesTotal! : Infinity;
    const take = Math.max(0, Math.min(chunk.byteLength, limit - this.received));
    if (this.received + take > this.buffer.byteLength) {
      const grown = new Uint8Array(this.info ? limit : Math.max(this.buffer.byteLength * 2, this.received + take, 4096));
      grown.set(this.buffer.subarray(0, this.received));
      this.buffer = grown;
    }
    this.buffer.set(chunk.subarray(0, take), this.received);
    this.received += take;
    const view = this.buffer.subarray(0, this.received);

    if (!this.info) {
      const { maxLod, ...headerOpts } = this.opts;
      this.info = readWvdbHeader(view, headerOpts);
      if (!this.info) return [];
      const { header } = this.info;
      this.finalLod = Math.min(header.bake.lods - 1, maxLod ?? Infinity);
      this.file = {
        version: this.info.version,
        header,
        dataStart: this.info.dataStart,
        source: header.source,
        bake: header.bake,
        grids: makeGrids(header),
        lodsDecoded: 0,
        complete: false,
      };
      const total = this.bytesTotal!;
      if (this.buffer.byteLength !== total) {
        const sized = new Uint8Array(Math.max(total, this.received));
        sized.set(view);
        this.buffer = sized;
      }
    }

    const events: WvdbLodEvent[] = [];
    while (this.next <= this.finalLod && wvdbPrefixLength(this.info, this.next) <= this.received) {
      const levels = decodeWvdbLod(this.info, this.next, this.buffer.subarray(0, this.received), 0);
      addLevels(this.file!.grids, levels);
      this.file!.lodsDecoded = this.next + 1;
      this.file!.complete = this.next + 1 === this.info.header.bake.lods;
      events.push({ lod: this.next, levels });
      this.next++;
    }
    if (this.done) this.buffer = new Uint8Array(0);
    return events;
  }
}
