/**
 * `NvdbStreamReader` (K4.16): read a `.nvdb` file from pushed bytes and emit
 * each grid as soon as its data is complete, so a coarse preview written first
 * (docs/design/K4-5-coarse-first.md, option a) can be drawn while the full
 * grid is still downloading.
 *
 *     const reader = new NvdbStreamReader();
 *     for await (const chunk of response.body) for (const g of reader.push(chunk)) show(g);
 *     reader.end(); // throws if the stream stopped mid-grid
 *
 * The framing is NanoVDB's `io::Segment` (IO.h; see `nvdb-file.ts`): a 16-byte
 * FileHeader, then a 176-byte FileMetaData + name per grid, then each grid's
 * data (the image for codec NONE; a u64 size + zlib stream for codec ZIP). A
 * file is one or more segments. Every size the reader allocates is declared
 * up front, so it checks each one before allocating:
 *
 * - `gridSize` must be a safe integer, at least a GridData header, and at most
 *   `maxGridBytes` (default 1 GiB);
 * - a ZIP payload must be at most zlib's `compressBound(gridSize)` (+64), and
 *   inflates into gridSize + 1 bytes at most (`inflateZipGrid`);
 * - a grid name is at most `maxNameBytes` (default 64 KiB).
 *
 * Memory: the reader holds the current segment's metadata and the grid it is
 * filling, never the whole file. Emitted images are handed off (a fresh buffer
 * per grid, 4-byte aligned, so `image` is a zero-copy `Uint32Array`).
 *
 * Errors are `NvdbFormatError` (a `VdbFormatError`) for malformed or
 * truncated data and `VdbUnsupportedError` for valid data out of scope
 * (BLOSC, a raw `NanoVDB1` grid buffer, unsupported grid types). After an
 * error the reader is failed: every later `push` rethrows it.
 */

import { VdbFormatError, VdbUnsupportedError } from "./vdb/errors.js";
import {
  Codec,
  FILE_HEADER_SIZE,
  FILE_METADATA_SIZE,
  GRID_DATA_SIZE,
  assertSafeGridSize,
  decodeCString,
  finishGridRecord,
  gridClassName,
  gridTypeName,
  inflateZipGrid,
  isFileHeaderMagic,
  readFileHeader,
  readFileMetaData,
  validateGridType,
  type GridMetadata,
  type ParsedFileMetaData,
} from "./nvdb-file.js";

/** A malformed or truncated `.nvdb` stream. `instanceof VdbFormatError` holds. */
export class NvdbFormatError extends VdbFormatError {
  constructor(message: string) {
    super(message);
    this.name = "NvdbFormatError";
    this.message = `@pmndrs/vdb/tools: malformed .nvdb: ${message}`;
  }
}

export interface NvdbStreamReaderOptions {
  /** Largest `gridSize` (decoded image bytes) the reader allocates. Default 1 GiB. */
  maxGridBytes?: number;
  /** Largest grid name (FileMetaData.nameSize). Default 64 KiB. */
  maxNameBytes?: number;
}

/** One grid, emitted by {@link NvdbStreamReader.push} as soon as its data is complete. */
export interface NvdbStreamGrid {
  /** Grid number in the file (counted across segments). */
  index: number;
  metadata: GridMetadata;
  /** The decoded grid image (zero-copy view of a buffer of its own). */
  image: Uint32Array;
  /** File bytes pushed when this grid completed (the end of its data). */
  bytesReceived: number;
  /** The preview level from a `<name>.lod<N>` name, or null for a full grid. */
  lod: number | null;
  /** The name without its `.lod<N>` suffix (the full grid's name). */
  baseName: string;
}

const DEFAULT_MAX_GRID_BYTES = 1 << 30;
const DEFAULT_MAX_NAME_BYTES = 1 << 16;
const ZIP_PREFIX = 8;

/** zlib's `compressBound` (the largest `compress()` output for `n` input bytes), plus slack. */
function zipBound(n: number): number {
  return n + Math.floor(n / 4096) + Math.floor(n / 16384) + Math.floor(n / 33554432) + 13 + 64;
}

const LOD_SUFFIX = /^(.*)\.lod(\d+)$/;

/** The preview name for `name` at `resolution`: `<name>.lod<resolution>` (K4.5). */
export function nvdbLodName(name: string, resolution: number): string {
  if (!Number.isInteger(resolution) || resolution < 1) {
    throw new Error(`nvdbLodName: resolution must be a positive integer, got ${resolution}`);
  }
  return `${name}.lod${resolution}`;
}

/** Splits a `<name>.lod<N>` preview name; a full grid's name gives `lod: null`. */
export function parseNvdbLodName(name: string): { baseName: string; lod: number | null } {
  const m = LOD_SUFFIX.exec(name);
  return m ? { baseName: m[1]!, lod: Number(m[2]) } : { baseName: name, lod: null };
}

type Phase = "header" | "meta" | "name" | "zipPrefix" | "grid";

interface PendingGrid {
  meta: ParsedFileMetaData;
  name: string;
}

/**
 * Push-based `.nvdb` reader: `push(chunk)` returns the grids that chunk
 * completed. Call `end()` when the stream ends to detect truncation.
 */
export class NvdbStreamReader {
  readonly #maxGridBytes: number;
  readonly #maxNameBytes: number;

  /** Small fixed-size fields (FileHeader, FileMetaData, ZIP prefix) and names. */
  #scratch = new Uint8Array(FILE_METADATA_SIZE);
  /** The buffer being filled (scratch or a grid payload) and how far. */
  #target: Uint8Array;
  #filled = 0;

  #phase: Phase = "header";
  #codec: Codec = Codec.NONE;
  #segmentGrids = 0;
  #pending: PendingGrid[] = [];
  #nextMeta: ParsedFileMetaData | null = null;
  #gridInSegment = 0;
  #received = 0;
  #segments = 0;
  #emitted = 0;
  #failed: Error | null = null;
  readonly #grids: GridMetadata[] = [];

  constructor(options: NvdbStreamReaderOptions = {}) {
    this.#maxGridBytes = options.maxGridBytes ?? DEFAULT_MAX_GRID_BYTES;
    this.#maxNameBytes = options.maxNameBytes ?? DEFAULT_MAX_NAME_BYTES;
    if (!Number.isSafeInteger(this.#maxGridBytes) || this.#maxGridBytes < GRID_DATA_SIZE) {
      throw new Error(`NvdbStreamReader: maxGridBytes must be an integer >= ${GRID_DATA_SIZE}`);
    }
    if (!Number.isSafeInteger(this.#maxNameBytes) || this.#maxNameBytes < 1) {
      throw new Error("NvdbStreamReader: maxNameBytes must be a positive integer");
    }
    this.#target = this.#scratch.subarray(0, FILE_HEADER_SIZE);
  }

  /** File bytes pushed so far. */
  get bytesReceived(): number {
    return this.#received;
  }

  /** Grids emitted so far. */
  get gridsEmitted(): number {
    return this.#emitted;
  }

  /**
   * Metadata of every grid whose FileMetaData has arrived, in file order,
   * including grids whose data has not (a segment lists all its grids before
   * any data, so the full grid's `gridByteSize` is known before its preview
   * is emitted). Names are as stored; see {@link parseNvdbLodName}.
   */
  get grids(): readonly GridMetadata[] {
    return this.#grids;
  }

  /** True between segments: every grid announced so far has been emitted. */
  get atBoundary(): boolean {
    return this.#phase === "header" && this.#filled === 0 && this.#segments > 0;
  }

  /** Pushes the next bytes of the file; returns the grids they completed, in order. */
  push(chunk: Uint8Array): NvdbStreamGrid[] {
    if (this.#failed) throw this.#failed;
    const out: NvdbStreamGrid[] = [];
    try {
      let at = 0;
      while (at < chunk.byteLength) {
        const take = Math.min(chunk.byteLength - at, this.#target.byteLength - this.#filled);
        this.#target.set(chunk.subarray(at, at + take), this.#filled);
        this.#filled += take;
        at += take;
        this.#received += take;
        if (this.#filled === this.#target.byteLength) this.#advance(out);
      }
      // A zero-length target (an empty name) completes without bytes.
      while (this.#target.byteLength === 0 && this.#filled === 0 && this.#phase !== "header") this.#advance(out);
    } catch (err) {
      this.#failed = err instanceof VdbFormatError || err instanceof VdbUnsupportedError ? err : new NvdbFormatError((err as Error).message);
      throw this.#failed;
    }
    return out;
  }

  /** Declares the end of the stream; throws `NvdbFormatError` if it stopped inside a segment. */
  end(): void {
    if (this.#failed) throw this.#failed;
    if (this.atBoundary) return;
    this.#failed = new NvdbFormatError(
      `the stream ended after ${this.#received} bytes, in the ${this.#describe()}; the file is truncated.`,
    );
    throw this.#failed;
  }

  #describe(): string {
    const grid = this.#segmentFirst() + this.#gridInSegment;
    switch (this.#phase) {
      case "header":
        return `FileHeader, ${this.#filled} of ${FILE_HEADER_SIZE} bytes`;
      case "meta":
        return `FileMetaData of grid #${this.#grids.length}, ${this.#filled} of ${FILE_METADATA_SIZE} bytes`;
      case "name":
        return `name of grid #${this.#grids.length}`;
      case "zipPrefix":
        return `ZIP size prefix of grid #${grid}`;
      case "grid":
        return `data of grid #${grid}, ${this.#filled} of ${this.#target.byteLength} bytes`;
    }
  }

  #segmentFirst(): number {
    return this.#grids.length - this.#segmentGrids;
  }

  #expect(phase: Phase, target: Uint8Array): void {
    this.#phase = phase;
    this.#target = target;
    this.#filled = 0;
  }

  #advance(out: NvdbStreamGrid[]): void {
    switch (this.#phase) {
      case "header": {
        const magic = String.fromCharCode(...this.#scratch.subarray(0, 8));
        if (!isFileHeaderMagic(magic)) {
          if (magic === "NanoVDB1") {
            throw new VdbUnsupportedError(
              ".nvdb stream: this is a raw NanoVDB grid buffer (magic NanoVDB1), not a segment file; read it whole with NanoVDBFile.fromArrayBuffer.",
            );
          }
          throw new NvdbFormatError(
            `expected a NanoVDB FileHeader (magic NanoVDB2 or NanoVDB0) at byte ${this.#received - FILE_HEADER_SIZE}, found "${magic.replace(/[^\x20-\x7e]/g, "?")}".`,
          );
        }
        const view = new DataView(this.#scratch.buffer, 0, FILE_HEADER_SIZE);
        let header: { gridCount: number; codec: Codec };
        try {
          header = readFileHeader(view, 0);
        } catch (err) {
          const msg = (err as Error).message;
          if (/BLOSC/.test(msg)) throw new VdbUnsupportedError(msg);
          throw new NvdbFormatError(msg);
        }
        if (header.gridCount === 0) throw new NvdbFormatError(`segment #${this.#segments} declares 0 grids.`);
        this.#codec = header.codec;
        this.#segmentGrids = header.gridCount;
        this.#segments++;
        this.#pending = [];
        this.#gridInSegment = 0;
        this.#expect("meta", this.#scratch.subarray(0, FILE_METADATA_SIZE));
        return;
      }
      case "meta": {
        const i = this.#grids.length;
        const meta = readFileMetaData(new DataView(this.#scratch.buffer, 0, FILE_METADATA_SIZE), 0);
        assertSafeGridSize(meta.gridSize, `grid #${i}'s FileMetaData.gridSize`);
        if (meta.gridSize < GRID_DATA_SIZE) {
          throw new NvdbFormatError(`grid #${i} declares gridSize ${meta.gridSize}, smaller than a ${GRID_DATA_SIZE}-byte GridData header.`);
        }
        if (meta.gridSize > this.#maxGridBytes) {
          throw new NvdbFormatError(
            `grid #${i} declares gridSize ${meta.gridSize} bytes, over maxGridBytes (${this.#maxGridBytes}); raise the option if the file is trusted.`,
          );
        }
        if (meta.nameSize > this.#maxNameBytes) {
          throw new NvdbFormatError(`grid #${i} declares a ${meta.nameSize}-byte name, over maxNameBytes (${this.#maxNameBytes}).`);
        }
        this.#nextMeta = meta;
        this.#expect("name", meta.nameSize <= this.#scratch.byteLength ? this.#scratch.subarray(0, meta.nameSize) : new Uint8Array(meta.nameSize));
        return;
      }
      case "name": {
        const meta = this.#nextMeta!;
        const name = decodeCString(this.#target);
        const gridType = gridTypeName(meta.gridTypeId);
        try {
          validateGridType(gridType, name);
        } catch (err) {
          throw new VdbUnsupportedError((err as Error).message);
        }
        this.#pending.push({ meta, name });
        this.#grids.push({
          name,
          gridType,
          gridClass: gridClassName(meta.gridClassId),
          worldBBox: meta.worldBBox,
          indexBBox: meta.indexBBox,
          voxelSize: meta.voxelSize,
          voxelCount: meta.voxelCount,
          gridByteSize: meta.gridSize,
        });
        this.#nextMeta = null;
        if (this.#pending.length < this.#segmentGrids) this.#expect("meta", this.#scratch.subarray(0, FILE_METADATA_SIZE));
        else this.#startGrid();
        return;
      }
      case "zipPrefix": {
        const { meta, name } = this.#pending[this.#gridInSegment]!;
        const size = Number(new DataView(this.#scratch.buffer, 0, ZIP_PREFIX).getBigUint64(0, true));
        const bound = zipBound(meta.gridSize);
        if (!Number.isSafeInteger(size) || size < 1 || size > bound) {
          throw new NvdbFormatError(
            `grid #${this.#segmentFirst() + this.#gridInSegment} ("${name}") declares a ${size}-byte ZIP payload for a ` +
              `${meta.gridSize}-byte grid (zlib's bound is ${bound}).`,
          );
        }
        this.#expect("grid", new Uint8Array(size));
        return;
      }
      case "grid": {
        const k = this.#gridInSegment;
        const index = this.#segmentFirst() + k;
        const { meta, name } = this.#pending[k]!;
        const bytes = this.#codec === Codec.ZIP ? inflateZipGrid(this.#target, meta.gridSize, index, name) : this.#target;
        const record = finishGridRecord(meta, name, index, bytes);
        this.#grids[index] = record.metadata;
        const { baseName, lod } = parseNvdbLodName(name);
        out.push({ index, metadata: record.metadata, image: record.image, bytesReceived: this.#received, lod, baseName });
        this.#emitted++;
        this.#gridInSegment++;
        if (this.#gridInSegment < this.#segmentGrids) this.#startGrid();
        else {
          this.#pending = [];
          this.#expect("header", this.#scratch.subarray(0, FILE_HEADER_SIZE));
        }
        return;
      }
    }
  }

  #startGrid(): void {
    const { meta } = this.#pending[this.#gridInSegment]!;
    if (this.#codec === Codec.ZIP) this.#expect("zipPrefix", this.#scratch.subarray(0, ZIP_PREFIX));
    else this.#expect("grid", new Uint8Array(meta.gridSize));
  }
}

/**
 * Reads a byte stream (a `fetch` body, or any async iterable of chunks) through
 * an {@link NvdbStreamReader}, yielding each grid as it completes. Stops
 * reading (and cancels the stream) when the consumer breaks out of the loop.
 */
export async function* readNvdbStream(
  source: ReadableStream<Uint8Array> | AsyncIterable<Uint8Array>,
  options: NvdbStreamReaderOptions = {},
): AsyncGenerator<NvdbStreamGrid, void, undefined> {
  const reader = new NvdbStreamReader(options);
  if (typeof (source as ReadableStream<Uint8Array>).getReader === "function") {
    const r = (source as ReadableStream<Uint8Array>).getReader();
    let finished = false;
    try {
      for (;;) {
        const { done, value } = await r.read();
        if (done) break;
        yield* reader.push(value);
      }
      finished = true;
      reader.end();
    } finally {
      if (!finished) await r.cancel().catch(() => undefined);
      r.releaseLock();
    }
    return;
  }
  for await (const chunk of source as AsyncIterable<Uint8Array>) yield* reader.push(chunk);
  reader.end();
}
