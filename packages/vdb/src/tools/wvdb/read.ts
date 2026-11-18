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
 * The stream reader decodes chunk by chunk (K4.13), so it holds one chunk's
 * compressed bytes at a time and can show a LOD before all its pages arrive.
 *
 * Decoded volume LODs are dense `Float32Array`s: channel planes one after the
 * other (`values[c * X*Y*Z + x + y*X + z*X*Y]`), x fastest. Voxels of bricks a
 * sparse LOD does not store hold the background (§12.2.3); voxels of a brick
 * outside the LOD are dropped (§3.3). Values are float32, computed exactly as
 * his `decodeWVDBPage` computes them.
 */

import { VDBToolsError } from "../errors.js";
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
    throw new VDBToolsError("VDB_BAD_OPTION", `wvdb: LOD ${lod} is out of range (0..${header.bake.lods - 1})`);
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
  /** Bricks decoded so far (equals the LOD's brick count once complete; less in a partial stream event). */
  bricksDecoded: number;
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
  /** Points decoded so far (equals `count` once complete; less in a partial stream event). */
  pointsDecoded: number;
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
  const chunks = wvdbChunks(info.header).filter((c) => c.lod === lod);
  const decoders = lodDecoders(info, lod);
  for (const c of chunks) decoders[c.gridIndex]!.add(c, sliceChunk(bytes, byteOffset, info, c));
  return decoders.map((d) => d.finish());
}

/** Incremental decoder of one grid's LOD: chunks in any order, then `finish`. */
interface LodDecoder {
  readonly level: WvdbLevel;
  add(c: WvdbChunkInfo, raw: Uint8Array): void;
  /** Checks the LOD is consistent (the mask matches the stored bricks) and returns it. */
  finish(): WvdbLevel;
}

function lodDecoders(info: WvdbHeaderInfo, lod: number): LodDecoder[] {
  const bits = info.header.bake.bits as WvdbBits;
  return info.header.grids.map((grid) =>
    isPointsGridHeader(grid) ? pointsLodDecoder(grid, lod) : volumeLodDecoder(grid, lod, bits),
  );
}

function volumeLodDecoder(grid: WvdbVolumeGridHeader, lod: number, bits: WvdbBits): LodDecoder {
  const entry = grid.lods[lod]!;
  const channels = (grid.channels ?? 1) as 1 | 3;
  const dimensions = [...entry.dimensions] as WvdbVec3;
  const [X, Y, Z] = dimensions;
  const total = brickTotal(dimensions);
  const layout: WvdbLayout = entry.mask ? "sparse" : "dense";
  const sparse = layout === "sparse";
  const background = backgroundOf(grid);
  const plane = X * Y * Z;
  const values = new Float32Array(plane * channels);
  for (let c = 0; c < channels; c++) values.fill(background[c]!, c * plane, (c + 1) * plane);
  const level: WvdbVolumeLevel = {
    kind: "volume",
    lod,
    dimensions,
    layout,
    channels,
    brickCounts: brickCounts(dimensions),
    brickTotal: total,
    mask: null,
    brickIndices: new Uint32Array(entry.bricks),
    mins: new Float32Array(entry.bricks * channels),
    maxs: new Float32Array(entry.bricks * channels),
    values,
    bricksDecoded: 0,
  };
  const seen = sparse ? new Uint32Array(Math.ceil(total / 32)) : null;
  let masks = 0;

  return {
    level,
    add(c, raw) {
      if (c.kind === "mask") {
        if (masks++ > 0) throw new WvdbFormatError(`LOD ${lod} of "${grid.name}": more than one mask chunk`);
        level.mask = decodeMask(inflateWvdbChunk(raw, maskByteSize(total)), total);
        return;
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
      // `c.first` is the page's position in the LOD's file order, whatever order pages decode in.
      level.brickIndices.set(page.indices, c.first);
      level.mins.set(page.mins, c.first * channels);
      level.maxs.set(page.maxs, c.first * channels);
      level.bricksDecoded += c.count;
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
    },
    finish() {
      if (sparse) {
        const mask = level.mask;
        if (!mask) throw new WvdbFormatError(`LOD ${lod} of "${grid.name}": sparse LOD without a mask chunk`);
        for (let w = 0; w < mask.length; w++) {
          if (mask[w] !== seen![w]) {
            throw new WvdbFormatError(`LOD ${lod} of "${grid.name}": the mask does not match the stored brick indices (§6.3)`);
          }
        }
      }
      return level;
    },
  };
}

function pointsLodDecoder(grid: WvdbPointsGridHeader, lod: number): LodDecoder {
  const count = grid.lods[lod]!.points;
  // Stored index of this LOD's first point: the points of the coarser LODs (§3.5; equals wvdbChunks' `first`).
  const firstPoint = grid.lods.slice(0, lod).reduce((s, l) => s + l.points, 0);
  const level: WvdbPointsLevel = {
    kind: "points",
    lod,
    firstPoint,
    count,
    positions: new Float32Array(count * 3),
    colors: grid.colors ? new Float32Array(count * 3) : null,
    pointsDecoded: 0,
  };
  return {
    level,
    add(c, raw) {
      const decoded = decodePoints(inflateWvdbChunk(raw, pointsChunkByteSize(c.count, grid.colors)), {
        count: c.count,
        colors: grid.colors,
      });
      const at = c.first - firstPoint;
      level.positions.set(decoded.positions, at * 3);
      if (level.colors && decoded.colors) level.colors.set(decoded.colors, at * 3);
      level.pointsDecoded += c.count;
    },
    finish: () => level,
  };
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
  /**
   * One level per grid, in header order. For a partial event these are the
   * LOD's levels in progress: later pages fill the same objects, and
   * `bricksDecoded` / `pointsDecoded` say how much is in.
   */
  levels: WvdbLevel[];
  /** True once every chunk of the LOD is decoded (always, unless `partialLods`). */
  complete: boolean;
  /** Partial events: the chunk just decoded (a page's bricks are `brickIndices[first .. first + count)`). */
  chunk?: WvdbChunkInfo;
}

export interface WvdbStreamReaderOptions extends Omit<ReadWvdbOptions, "partial"> {
  /**
   * Also emit an event (`complete: false`) after every chunk of a LOD decodes,
   * so a LOD can be shown before all its pages arrive (§12.2). Default false:
   * one event per finished LOD.
   */
  partialLods?: boolean;
  /** Refuse a header longer than this many bytes before buffering it. Default 16 MiB. */
  maxHeaderBytes?: number;
}

const DEFAULT_MAX_HEADER_BYTES = 16 * 1024 * 1024;

/**
 * Largest compressed size accepted for a chunk that inflates to at most `n`
 * bytes: zlib's `compressBound` is about `n + n/4096 + 13`; this allows
 * `n + n/1024 + 1 KiB`, which any deflate encoder's output fits (stored blocks
 * cost 5 bytes per 64 KiB). The stream reader holds one chunk's bytes at a
 * time, so this is its memory bound.
 */
export function wvdbChunkByteLimit(inflatedBytes: number): number {
  return inflatedBytes + Math.ceil(inflatedBytes / 1024) + 1024;
}

function inflatedLimit(header: WvdbHeader, c: WvdbChunkInfo): number {
  const grid = header.grids[c.gridIndex]!;
  if (isPointsGridHeader(grid)) return pointsChunkByteSize(c.count, grid.colors);
  const entry = grid.lods[c.lod]!;
  const total = brickTotal(entry.dimensions);
  if (c.kind === "mask") return maskByteSize(total);
  const channels = grid.channels ?? 1;
  return pageByteSize(c.count, channels, header.bake.bits as WvdbBits, entry.mask !== undefined, c.count * channels);
}

/**
 * Push-style coarse-first decoder for a `.wvdb` download: feed it the bytes
 * as they arrive; each `push` returns the LODs that became complete (and,
 * with `partialLods`, the LODs that gained a page).
 *
 *     const reader = new WvdbStreamReader({ partialLods: true });
 *     for await (const chunk of response.body) for (const e of reader.push(chunk)) show(e);
 *     reader.end();
 *
 * Chunks are decoded one at a time, in file order, as soon as their last byte
 * arrives, so the reader holds at most one chunk's compressed bytes (plus the
 * header until it parses) besides the decoded LODs: `peakBufferedBytes` is
 * bounded by the largest chunk plus one pushed buffer. Each chunk's length is
 * checked against {@link wvdbChunkByteLimit} when the header arrives. After an
 * error, every later call throws the same error.
 */
export class WvdbStreamReader {
  private buf = new Uint8Array(0);
  /** Absolute file offset of `buf[0]`, and one past the last byte held. */
  private bufStart = 0;
  private bufEnd = 0;
  private received = 0;
  private next = 0;
  private finalLod = 0;
  private chunks: WvdbChunkInfo[] = [];
  private ci = 0;
  private remaining: number[] = [];
  private decoders = new Map<number, LodDecoder[]>();
  private failure: unknown = null;
  private peak = 0;
  info: WvdbHeaderInfo | null = null;
  /** The decoded file so far (grids gain levels as LODs complete); null before the header. */
  file: WvdbFile | null = null;

  constructor(private readonly opts: WvdbStreamReaderOptions = {}) {}

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

  /** File bytes pushed so far. */
  get bytesReceived(): number {
    return this.received;
  }

  /** Compressed bytes held now, and the most ever held. */
  get bufferedBytes(): number {
    return this.bufEnd - this.bufStart;
  }

  get peakBufferedBytes(): number {
    return this.peak;
  }

  push(chunk: Uint8Array): WvdbLodEvent[] {
    if (this.failure !== null) throw this.failure;
    if (this.done) return [];
    try {
      this.append(chunk);
      const events: WvdbLodEvent[] = [];
      if (!this.info && !this.readHeader()) return events;
      this.completeLods(events);
      this.decodeChunks(events);
      if (this.done) {
        this.buf = new Uint8Array(0);
        this.bufStart = this.bufEnd = this.received;
      }
      return events;
    } catch (err) {
      this.failure = err;
      this.buf = new Uint8Array(0);
      throw err;
    }
  }

  /** Call when the stream ends: throws `WvdbFormatError` if a LOD up to `maxLod` is still missing. */
  end(): void {
    if (this.failure !== null) throw this.failure;
    if (this.done) return;
    this.failure = new WvdbFormatError(
      this.info
        ? `stream ended at byte ${this.received} before LOD ${this.next} was complete (needs ${this.bytesTotal} bytes)`
        : `stream ended at byte ${this.received} before the header was complete`,
    );
    throw this.failure;
  }

  /** Absolute offset below which no byte is needed any more. */
  private keepFrom(): number {
    if (!this.info) return 0;
    const c = this.chunks[this.ci];
    return c ? this.info.dataStart + c.offset : Infinity;
  }

  /** Drops held bytes before `keepFrom()`. */
  private compact(): void {
    const keep = this.keepFrom();
    if (keep <= this.bufStart) return;
    const drop = Math.min(keep, this.bufEnd) - this.bufStart;
    if (drop > 0) {
      this.buf.copyWithin(0, drop, this.bufEnd - this.bufStart);
      this.bufStart += drop;
    }
  }

  private append(chunk: Uint8Array): void {
    const absStart = this.received;
    const absEnd = absStart + chunk.byteLength;
    this.received = absEnd;
    this.compact();
    // Before the header, everything is kept (readHeader caps its length); after it, only up to the last needed chunk.
    const from = Math.max(absStart, this.keepFrom());
    const to = Math.min(absEnd, this.info ? this.bytesTotal! : Infinity);
    if (to <= from) return;
    // Held bytes end at absStart, or are all before keepFrom (then none are held): restart the buffer at `from`.
    if (this.bufEnd !== from) this.bufStart = this.bufEnd = from;
    const held = this.bufEnd - this.bufStart;
    const need = held + (to - from);
    if (need > this.buf.byteLength) {
      const grown = new Uint8Array(Math.max(need, Math.min(this.buf.byteLength * 2, need + (1 << 20))));
      grown.set(this.buf.subarray(0, held));
      this.buf = grown;
    }
    this.buf.set(chunk.subarray(from - absStart, to - absStart), held);
    this.bufEnd = to;
    this.peak = Math.max(this.peak, this.bufEnd - this.bufStart);
  }

  private readHeader(): boolean {
    // Before the header nothing is dropped, so the buffer starts at file offset 0.
    const held = this.buf.subarray(0, this.bufEnd - this.bufStart);
    if (held.byteLength >= WVDB_PREAMBLE_SIZE) {
      const headerLength = new DataView(held.buffer, held.byteOffset).getUint32(8, true);
      const max = this.opts.maxHeaderBytes ?? DEFAULT_MAX_HEADER_BYTES;
      if (headerLength > max) throw new WvdbFormatError(`headerLength ${headerLength} is more than maxHeaderBytes (${max})`);
    }
    const { maxLod, partialLods: _p, maxHeaderBytes: _m, ...headerOpts } = this.opts;
    const info = readWvdbHeader(held, headerOpts);
    if (!info) return false;
    const { header } = info;
    this.finalLod = Math.min(header.bake.lods - 1, maxLod ?? Infinity);
    const chunks = wvdbChunks(header).filter((c) => c.lod <= this.finalLod);
    for (const c of chunks) {
      const limit = wvdbChunkByteLimit(inflatedLimit(header, c));
      if (c.length > limit) {
        throw new WvdbFormatError(
          `chunk (${c.kind}, grid ${c.gridIndex}, LOD ${c.lod}) is ${c.length} bytes, more than a deflate of its ` +
            `${inflatedLimit(header, c)}-byte content can be (${limit})`,
        );
      }
    }
    // File order; a stable sort keeps §9 order for equal offsets.
    this.chunks = chunks.map((c, i) => ({ c, i })).sort((a, b) => a.c.offset - b.c.offset || a.i - b.i).map(({ c }) => c);
    this.remaining = Array.from({ length: this.finalLod + 1 }, () => 0);
    for (const c of this.chunks) this.remaining[c.lod]!++;
    this.info = info;
    this.file = {
      version: info.version,
      header,
      dataStart: info.dataStart,
      source: header.source,
      bake: header.bake,
      grids: makeGrids(header),
      lodsDecoded: 0,
      complete: false,
    };
    return true;
  }

  private decodersFor(lod: number): LodDecoder[] {
    let d = this.decoders.get(lod);
    if (!d) {
      d = lodDecoders(this.info!, lod);
      this.decoders.set(lod, d);
    }
    return d;
  }

  private decodeChunks(events: WvdbLodEvent[]): void {
    const info = this.info!;
    while (this.ci < this.chunks.length) {
      const c = this.chunks[this.ci]!;
      const start = info.dataStart + c.offset;
      const end = start + c.length;
      if (end > this.bufEnd) break;
      if (start < this.bufStart) throw new Error("WvdbStreamReader: internal error, chunk bytes were dropped");
      const decoders = this.decodersFor(c.lod);
      decoders[c.gridIndex]!.add(c, this.buf.subarray(start - this.bufStart, end - this.bufStart));
      this.ci++;
      this.remaining[c.lod]!--;
      if (this.opts.partialLods) events.push({ lod: c.lod, levels: decoders.map((d) => d.level), complete: false, chunk: c });
      this.completeLods(events);
    }
    // Release bytes before the next chunk now (the next push would too).
    this.compact();
  }

  private completeLods(events: WvdbLodEvent[]): void {
    while (this.next <= this.finalLod && this.remaining[this.next] === 0) {
      const lod = this.next;
      const levels = this.decodersFor(lod).map((d) => d.finish());
      this.decoders.delete(lod);
      addLevels(this.file!.grids, levels);
      this.file!.lodsDecoded = lod + 1;
      this.file!.complete = lod + 1 === this.info!.header.bake.lods;
      events.push({ lod, levels, complete: true });
      this.next++;
    }
  }
}
