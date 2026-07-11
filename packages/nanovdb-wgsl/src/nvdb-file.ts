/**
 * `.nvdb` loader — Phase 1 (docs/PLAN.md), implementing the `NanoVDBFile`
 * contract from docs/SPEC.md §2.2: parse a `.nvdb` file (or a raw NanoVDB
 * grid buffer) into GPU-ready grid images — "a valid NanoVDB grid image in
 * a flat u32 buffer" (SPEC §1).
 *
 * Ground truth for every byte offset below is the C++ that wrote the test
 * fixtures:
 *   - fixtures/downloads/openvdb-sparse/nanovdb/nanovdb/io/IO.h
 *     (FileHeader/FileMetaData layout, Segment/Codec framing)
 *   - fixtures/downloads/openvdb-sparse/nanovdb/nanovdb/NanoVDB.h
 *     (magic numbers, Version packing, GridType/GridClass enums, GridData)
 *   - vendor/stride-tables.json (extracted PNanoVDB.h defines: GridData,
 *     TreeData and RootData offsets, ABI 32.9.1)
 *
 * ## `.nvdb` "segment" format (FileHeader magic `NanoVDB0`/`NanoVDB2`)
 *
 *     FileHeader (16B), [FileMetaData (176B), gridName (nameSize B), grid
 *     data (compressed per codec)] x gridCount
 *
 * FileHeader (`struct FileHeader`, IO.h/NanoVDB.h, 16 bytes):
 *
 *   field      | offset | size | type
 *   -----------|--------|------|------
 *   magic      | 0      | 8    | u64 (ASCII "NanoVDB0"/"NanoVDB2")
 *   version    | 8      | 4    | u32 (major<<21 | minor<<10 | patch)
 *   gridCount  | 12     | 2    | u16
 *   codec      | 14     | 2    | u16 (0=NONE, 1=ZIP, 2=BLOSC)
 *
 * FileMetaData (`struct FileMetaData`, NanoVDB.h, 176 bytes):
 *
 *   field           | offset | size | type
 *   ----------------|--------|------|------
 *   gridSize        | 0      | 8    | u64 (decompressed grid image bytes)
 *   fileSize        | 8      | 8    | u64 (on-disk bytes incl. codec framing)
 *   nameKey         | 16     | 8    | u64 (io::stringHash(gridName))
 *   voxelCount      | 24     | 8    | u64 (active voxel count)
 *   gridType        | 32     | 4    | u32 (GridType enum)
 *   gridClass       | 36     | 4    | u32 (GridClass enum)
 *   worldBBox.min   | 40     | 24   | f64 x3
 *   worldBBox.max   | 64     | 24   | f64 x3
 *   indexBBox.min   | 88     | 12   | i32 x3
 *   indexBBox.max   | 100    | 12   | i32 x3
 *   voxelSize       | 112    | 24   | f64 x3
 *   nameSize        | 136    | 4    | u32 (grid name bytes, incl. NUL)
 *   nodeCount[4]    | 140    | 16   | u32 x4 (unused by this loader)
 *   tileCount[3]    | 156    | 12   | u32 x3 (unused by this loader)
 *   codec           | 168    | 2    | u16 (mirrors FileHeader.codec)
 *   blindDataCount  | 170    | 2    | u16 (unused by this loader)
 *   version         | 172    | 4    | u32
 *                                    total: 176 bytes
 *
 * ## ZIP on-disk framing (`io::Internal::write`/`read`, IO.h)
 *
 * Per grid: an 8-byte little-endian u64 giving the *compressed* byte count,
 * followed by that many bytes of zlib-format (RFC 1950) compressed data —
 * i.e. exactly what C's `compress()`/`uncompress()` (and fflate's
 * `zlibSync`/`unzlibSync`) produce/consume. The decompressed length must
 * equal the grid's `gridSize` field. BLOSC uses the same 8-byte-prefixed
 * chunk framing but with blosc-compressed chunks; it is not implemented
 * (SPEC §7 non-goal) — callers get an actionable error instead.
 *
 * ## Raw grid buffer format (magic `NanoVDB1`, no FileHeader)
 *
 * The buffer starts directly with `GridData` (672 bytes, layout below, from
 * vendor/stride-tables.json), immediately followed by `TreeData` (64 bytes)
 * and then the rest of the tree. Multiple grids may be concatenated
 * (`GridData.mGridCount` grids total); each grid's byte length is its own
 * `mGridSize` field, so grids are found by walking `mGridSize` forward.
 * Raw buffers are always uncompressed.
 */

import { unzlibSync } from "fflate";

/** `.nvdb` FileHeader size in bytes (magic + version + grid count + codec). */
export const FILE_HEADER_SIZE = 16;

/** Per-grid FileMetaData size in bytes (excluding the trailing grid name). */
export const FILE_METADATA_SIZE = 176;

/** GridData block size in bytes (start of every NanoVDB grid image). */
export const GRID_DATA_SIZE = 672;

/**
 * NanoVDB 8-byte magic strings: `NanoVDB0` = legacy file header / grid,
 * `NanoVDB1` = raw grid buffer, `NanoVDB2` = current file header. Sniffed
 * by `NanoVDBFile.fromArrayBuffer`.
 */
export const MAGIC_ASCII = ["NanoVDB0", "NanoVDB1", "NanoVDB2"] as const;

/** File codecs we know about; v1 supports NONE and ZIP (fflate). */
export const Codec = {
  NONE: 0,
  ZIP: 1,
  BLOSC: 2,
} as const;
export type Codec = (typeof Codec)[keyof typeof Codec];

/** Grid types targeted for v1 (FogVolume rendering). */
export const SUPPORTED_GRID_TYPES = ["Float", "Fp8", "FpN"] as const;

export interface GridMetadata {
  name: string;
  gridType: string;
  gridClass: string;
  worldBBox: { min: [number, number, number]; max: [number, number, number] };
  indexBBox: { min: [number, number, number]; max: [number, number, number] };
  voxelSize: [number, number, number];
  voxelCount: number;
  gridByteSize: number;
}

// ---------------------------------------------------------------------------
// FileHeader / FileMetaData byte offsets (see module doc comment for the
// full field table; these are the raw numbers used below).
// ---------------------------------------------------------------------------

const FILE_HEADER_OFF_VERSION = 8;
const FILE_HEADER_OFF_GRID_COUNT = 12;
const FILE_HEADER_OFF_CODEC = 14;

const META_OFF_GRID_SIZE = 0;
const META_OFF_VOXEL_COUNT = 24;
const META_OFF_GRID_TYPE = 32;
const META_OFF_GRID_CLASS = 36;
const META_OFF_WORLD_BBOX_MIN = 40;
const META_OFF_WORLD_BBOX_MAX = 64;
const META_OFF_INDEX_BBOX_MIN = 88;
const META_OFF_INDEX_BBOX_MAX = 100;
const META_OFF_VOXEL_SIZE = 112;
const META_OFF_NAME_SIZE = 136;

// ---------------------------------------------------------------------------
// GridData / TreeData / RootData byte offsets, for the raw-grid-buffer path
// (magic `NanoVDB1`). Values are the PNanoVDB.h defines baked into
// vendor/stride-tables.json.
// ---------------------------------------------------------------------------

const GRID_OFF_VERSION = 16;
const GRID_OFF_GRID_INDEX = 24;
const GRID_OFF_GRID_COUNT = 28;
const GRID_OFF_GRID_SIZE = 32;
const GRID_OFF_GRID_NAME = 40;
const GRID_NAME_MAX = 256; // GridData::MaxNameSize
const GRID_OFF_WORLD_BBOX = 560;
const GRID_OFF_VOXEL_SIZE = 608;
const GRID_OFF_GRID_CLASS = 632;
const GRID_OFF_GRID_TYPE = 636;

const TREE_SIZE = 64;
const TREE_OFF_NODE_OFFSET_ROOT = 24;
const TREE_OFF_VOXEL_COUNT = 56;

const ROOT_OFF_BBOX_MIN = 0;
const ROOT_OFF_BBOX_MAX = 12;
const ROOT_BASE_SIZE = 28;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function requireBytes(buffer: ArrayBuffer, minLength: number, what: string): void {
  if (buffer.byteLength < minLength) {
    throw new Error(
      `NanoVDBFile: truncated buffer while reading ${what} ` +
        `(need ${minLength} bytes, buffer has ${buffer.byteLength})`,
    );
  }
}

function readMagic(buffer: ArrayBuffer, offset: number): string {
  const bytes = new Uint8Array(buffer, offset, 8);
  return String.fromCharCode(...bytes);
}

function decodeCString(bytes: Uint8Array): string {
  let end = bytes.indexOf(0);
  if (end < 0) end = bytes.length;
  return new TextDecoder().decode(bytes.subarray(0, end));
}

/** Packed NanoVDB `Version`: major<<21 | minor<<10 | patch. */
interface ParsedVersion {
  major: number;
  minor: number;
  patch: number;
}

function parseVersion(raw: number): ParsedVersion {
  return {
    major: (raw >>> 21) & 0x7ff,
    minor: (raw >>> 10) & 0x7ff,
    patch: raw & 0x3ff,
  };
}

function formatVersion(v: ParsedVersion): string {
  return `${v.major}.${v.minor}.${v.patch}`;
}

/** ABI major version this loader targets (vendor/stride-tables.json `$meta.abi` = "32.9.1"). */
const NANOVDB_MAJOR_VERSION = 32;

function checkVersion(raw: number): ParsedVersion {
  const v = parseVersion(raw);
  if (v.major !== NANOVDB_MAJOR_VERSION) {
    throw new Error(
      `NanoVDBFile: incompatible NanoVDB version ${formatVersion(v)} ` +
        `(this loader targets NanoVDB ${NANOVDB_MAJOR_VERSION}.x). ` +
        `Re-export the file with a matching major version of NanoVDB.`,
    );
  }
  return v;
}

// GridType enum names (NanoVDB.h `enum class GridType`), indexed by id.
// 21/22 are retired (IndexMask/OnIndexMask) and kept only as placeholders.
const GRID_TYPE_NAMES: readonly string[] = [
  "Unknown",
  "Float",
  "Double",
  "Int16",
  "Int32",
  "Int64",
  "Vec3f",
  "Vec3d",
  "Mask",
  "Half",
  "UInt32",
  "Boolean",
  "RGBA8",
  "Fp4",
  "Fp8",
  "Fp16",
  "FpN",
  "Vec4f",
  "Vec4d",
  "Index",
  "OnIndex",
  "Retired21",
  "Retired22",
  "PointIndex",
  "Vec3u8",
  "Vec3u16",
  "UInt8",
  "End",
];

// GridClass enum names (NanoVDB.h `enum class GridClass`), indexed by id.
const GRID_CLASS_NAMES: readonly string[] = [
  "Unknown",
  "LevelSet",
  "FogVolume",
  "Staggered",
  "PointIndex",
  "PointData",
  "Topology",
  "VoxelVolume",
  "IndexGrid",
  "TensorGrid",
  "VoxelBVH",
  "End",
];

function gridTypeName(id: number): string {
  return GRID_TYPE_NAMES[id] ?? `Unknown(${id})`;
}

function gridClassName(id: number): string {
  return GRID_CLASS_NAMES[id] ?? `Unknown(${id})`;
}

const SUPPORTED_GRID_TYPES_SET: readonly string[] = SUPPORTED_GRID_TYPES;

function validateGridType(gridType: string, gridName: string): void {
  if (!SUPPORTED_GRID_TYPES_SET.includes(gridType)) {
    throw new Error(
      `NanoVDBFile: grid "${gridName}" is a ${gridType} grid, which this loader does not ` +
        `support — v1 only handles FogVolume grids of type ${SUPPORTED_GRID_TYPES.join("/")}. ` +
        `Re-export as float/fp8 (or fpn), e.g. \`nanovdb_convert --fp8 in.vdb out.nvdb\`.`,
    );
  }
}

/** Zero-copy u32 view when the input is already 4-byte aligned; a copy otherwise. */
function u32View(bytes: Uint8Array): Uint32Array {
  if (bytes.byteOffset % 4 === 0 && bytes.byteLength % 4 === 0) {
    return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  }
  const copy = bytes.slice(); // fresh ArrayBuffer, byteOffset 0
  return new Uint32Array(copy.buffer, copy.byteOffset, copy.byteLength / 4);
}

interface ParsedFileMetaData {
  gridSize: number;
  voxelCount: number;
  gridTypeId: number;
  gridClassId: number;
  worldBBox: GridMetadata["worldBBox"];
  indexBBox: GridMetadata["indexBBox"];
  voxelSize: [number, number, number];
  nameSize: number;
}

function readFileMetaData(view: DataView, offset: number): ParsedFileMetaData {
  const gridSize = Number(view.getBigUint64(offset + META_OFF_GRID_SIZE, true));
  const voxelCount = Number(view.getBigUint64(offset + META_OFF_VOXEL_COUNT, true));
  const gridTypeId = view.getUint32(offset + META_OFF_GRID_TYPE, true);
  const gridClassId = view.getUint32(offset + META_OFF_GRID_CLASS, true);
  const worldBBox = {
    min: [
      view.getFloat64(offset + META_OFF_WORLD_BBOX_MIN, true),
      view.getFloat64(offset + META_OFF_WORLD_BBOX_MIN + 8, true),
      view.getFloat64(offset + META_OFF_WORLD_BBOX_MIN + 16, true),
    ] as [number, number, number],
    max: [
      view.getFloat64(offset + META_OFF_WORLD_BBOX_MAX, true),
      view.getFloat64(offset + META_OFF_WORLD_BBOX_MAX + 8, true),
      view.getFloat64(offset + META_OFF_WORLD_BBOX_MAX + 16, true),
    ] as [number, number, number],
  };
  const indexBBox = {
    min: [
      view.getInt32(offset + META_OFF_INDEX_BBOX_MIN, true),
      view.getInt32(offset + META_OFF_INDEX_BBOX_MIN + 4, true),
      view.getInt32(offset + META_OFF_INDEX_BBOX_MIN + 8, true),
    ] as [number, number, number],
    max: [
      view.getInt32(offset + META_OFF_INDEX_BBOX_MAX, true),
      view.getInt32(offset + META_OFF_INDEX_BBOX_MAX + 4, true),
      view.getInt32(offset + META_OFF_INDEX_BBOX_MAX + 8, true),
    ] as [number, number, number],
  };
  const voxelSize: [number, number, number] = [
    view.getFloat64(offset + META_OFF_VOXEL_SIZE, true),
    view.getFloat64(offset + META_OFF_VOXEL_SIZE + 8, true),
    view.getFloat64(offset + META_OFF_VOXEL_SIZE + 16, true),
  ];
  const nameSize = view.getUint32(offset + META_OFF_NAME_SIZE, true);
  return { gridSize, voxelCount, gridTypeId, gridClassId, worldBBox, indexBBox, voxelSize, nameSize };
}

interface GridRecord {
  metadata: GridMetadata;
  image: Uint32Array;
}

// ---------------------------------------------------------------------------
// Segment format: FileHeader + [FileMetaData, gridName, grid data] x N
// ---------------------------------------------------------------------------

function parseSegments(buffer: ArrayBuffer): GridRecord[] {
  requireBytes(buffer, FILE_HEADER_SIZE, "FileHeader");
  const view = new DataView(buffer);

  checkVersion(view.getUint32(FILE_HEADER_OFF_VERSION, true));
  const gridCount = view.getUint16(FILE_HEADER_OFF_GRID_COUNT, true);
  const codec = view.getUint16(FILE_HEADER_OFF_CODEC, true);

  if (codec === Codec.BLOSC) {
    throw new Error(
      "NanoVDBFile: this file uses the BLOSC codec, which is not supported in the browser. " +
        "Re-export it with codec NONE or ZIP (e.g. `nanovdb_convert --codec=zip`).",
    );
  }
  if (codec !== Codec.NONE && codec !== Codec.ZIP) {
    throw new Error(`NanoVDBFile: unknown codec value ${codec} in FileHeader`);
  }

  const records: GridRecord[] = [];
  let offset = FILE_HEADER_SIZE;

  for (let i = 0; i < gridCount; i++) {
    requireBytes(buffer, offset + FILE_METADATA_SIZE, `FileMetaData for grid #${i}`);
    const meta = readFileMetaData(view, offset);
    offset += FILE_METADATA_SIZE;

    requireBytes(buffer, offset + meta.nameSize, `grid name for grid #${i}`);
    const name = decodeCString(new Uint8Array(buffer, offset, meta.nameSize));
    offset += meta.nameSize;

    let gridBytes: Uint8Array;
    if (codec === Codec.NONE) {
      requireBytes(buffer, offset + meta.gridSize, `grid data for grid #${i} ("${name}")`);
      gridBytes = new Uint8Array(buffer, offset, meta.gridSize);
      offset += meta.gridSize;
    } else {
      // Codec.ZIP: 8-byte LE compressed-size prefix, then that many bytes of
      // zlib (RFC 1950) compressed data (io::Internal::write/read, IO.h).
      requireBytes(buffer, offset + 8, `ZIP size prefix for grid #${i} ("${name}")`);
      const compressedSize = Number(view.getBigUint64(offset, true));
      offset += 8;
      requireBytes(buffer, offset + compressedSize, `ZIP payload for grid #${i} ("${name}")`);
      const compressed = new Uint8Array(buffer, offset, compressedSize);
      let decompressed: Uint8Array;
      try {
        decompressed = unzlibSync(compressed);
      } catch (err) {
        throw new Error(
          `NanoVDBFile: failed to inflate ZIP-compressed grid #${i} ("${name}"): ${(err as Error).message}`,
        );
      }
      if (decompressed.byteLength !== meta.gridSize) {
        throw new Error(
          `NanoVDBFile: ZIP-decompressed grid #${i} ("${name}") is ${decompressed.byteLength} ` +
            `bytes, expected ${meta.gridSize}`,
        );
      }
      gridBytes = decompressed;
      offset += compressedSize;
    }

    const gridType = gridTypeName(meta.gridTypeId);
    const gridClass = gridClassName(meta.gridClassId);
    validateGridType(gridType, name);

    records.push({
      metadata: {
        name,
        gridType,
        gridClass,
        worldBBox: meta.worldBBox,
        indexBBox: meta.indexBBox,
        voxelSize: meta.voxelSize,
        voxelCount: meta.voxelCount,
        gridByteSize: meta.gridSize,
      },
      image: u32View(gridBytes),
    });
  }

  return records;
}

// ---------------------------------------------------------------------------
// Raw grid buffer format: GridData [+ TreeData + ...] x N, no FileHeader
// ---------------------------------------------------------------------------

function parseRawGridBuffer(buffer: ArrayBuffer): GridRecord[] {
  const view = new DataView(buffer);
  const records: GridRecord[] = [];

  let offset = 0;
  let index = 0;
  let expectedCount: number | undefined;

  while (offset < buffer.byteLength) {
    requireBytes(buffer, offset + GRID_DATA_SIZE, `GridData for grid #${index}`);
    const magic = readMagic(buffer, offset);
    if (magic !== "NanoVDB1" && magic !== "NanoVDB0") {
      throw new Error(`NanoVDBFile: expected a grid magic at byte ${offset}, found "${magic}"`);
    }
    checkVersion(view.getUint32(offset + GRID_OFF_VERSION, true));

    const gridCount = view.getUint32(offset + GRID_OFF_GRID_COUNT, true);
    expectedCount ??= gridCount;
    void view.getUint32(offset + GRID_OFF_GRID_INDEX, true); // mGridIndex, unused

    const gridSize = Number(view.getBigUint64(offset + GRID_OFF_GRID_SIZE, true));
    requireBytes(buffer, offset + gridSize, `grid image for grid #${index}`);

    const name = decodeCString(new Uint8Array(buffer, offset + GRID_OFF_GRID_NAME, GRID_NAME_MAX));

    const gridTypeId = view.getUint32(offset + GRID_OFF_GRID_TYPE, true);
    const gridClassId = view.getUint32(offset + GRID_OFF_GRID_CLASS, true);
    const worldBBox = {
      min: [
        view.getFloat64(offset + GRID_OFF_WORLD_BBOX, true),
        view.getFloat64(offset + GRID_OFF_WORLD_BBOX + 8, true),
        view.getFloat64(offset + GRID_OFF_WORLD_BBOX + 16, true),
      ] as [number, number, number],
      max: [
        view.getFloat64(offset + GRID_OFF_WORLD_BBOX + 24, true),
        view.getFloat64(offset + GRID_OFF_WORLD_BBOX + 32, true),
        view.getFloat64(offset + GRID_OFF_WORLD_BBOX + 40, true),
      ] as [number, number, number],
    };
    const voxelSize: [number, number, number] = [
      view.getFloat64(offset + GRID_OFF_VOXEL_SIZE, true),
      view.getFloat64(offset + GRID_OFF_VOXEL_SIZE + 8, true),
      view.getFloat64(offset + GRID_OFF_VOXEL_SIZE + 16, true),
    ];

    const treeOffset = offset + GRID_DATA_SIZE;
    requireBytes(buffer, treeOffset + TREE_SIZE, `TreeData for grid #${index}`);
    const voxelCount = Number(view.getBigUint64(treeOffset + TREE_OFF_VOXEL_COUNT, true));
    const nodeOffsetRoot = Number(view.getBigUint64(treeOffset + TREE_OFF_NODE_OFFSET_ROOT, true));
    const rootOffset = treeOffset + nodeOffsetRoot;
    requireBytes(buffer, rootOffset + ROOT_BASE_SIZE, `RootData bbox for grid #${index}`);
    const indexBBox = {
      min: [
        view.getInt32(rootOffset + ROOT_OFF_BBOX_MIN, true),
        view.getInt32(rootOffset + ROOT_OFF_BBOX_MIN + 4, true),
        view.getInt32(rootOffset + ROOT_OFF_BBOX_MIN + 8, true),
      ] as [number, number, number],
      max: [
        view.getInt32(rootOffset + ROOT_OFF_BBOX_MAX, true),
        view.getInt32(rootOffset + ROOT_OFF_BBOX_MAX + 4, true),
        view.getInt32(rootOffset + ROOT_OFF_BBOX_MAX + 8, true),
      ] as [number, number, number],
    };

    const gridType = gridTypeName(gridTypeId);
    const gridClass = gridClassName(gridClassId);
    validateGridType(gridType, name);

    const gridBytes = new Uint8Array(buffer, offset, gridSize);
    records.push({
      metadata: {
        name,
        gridType,
        gridClass,
        worldBBox,
        indexBBox,
        voxelSize,
        voxelCount,
        gridByteSize: gridSize,
      },
      image: u32View(gridBytes),
    });

    offset += gridSize;
    index += 1;
    if (expectedCount !== undefined && index >= expectedCount) break;
  }

  return records;
}

// ---------------------------------------------------------------------------
// Top-level format sniff + public API
// ---------------------------------------------------------------------------

function sniffFormat(buffer: ArrayBuffer): "segment" | "raw" {
  requireBytes(buffer, 8, "the NanoVDB magic");
  const magic = readMagic(buffer, 0);
  // "NanoVDB0" is genuinely ambiguous pre-v32.6 (used for both FileHeader and
  // raw GridData); we treat it as a (legacy) FileHeader, matching the plain
  // reading "legacy files use NanoVDB0" — none of our fixtures exercise this.
  if (magic === "NanoVDB2" || magic === "NanoVDB0") return "segment";
  if (magic === "NanoVDB1") return "raw";
  throw new Error(
    `NanoVDBFile: unrecognized magic "${magic}" at byte 0 (expected one of ` +
      `${MAGIC_ASCII.join(", ")}) — this does not look like a NanoVDB file.`,
  );
}

/**
 * Parses a `.nvdb` file (or raw grid buffer) and exposes GPU-ready grid
 * images as flat u32 views — "a valid NanoVDB grid image in a flat u32
 * buffer" is the single contract between the CPU and GPU halves (SPEC §1).
 */
export class NanoVDBFile {
  private readonly records: readonly GridRecord[];
  private readonly sourceBuffer: ArrayBuffer;

  private constructor(sourceBuffer: ArrayBuffer, records: readonly GridRecord[]) {
    this.sourceBuffer = sourceBuffer;
    this.records = records;
  }

  static async fromURL(url: string | URL): Promise<NanoVDBFile> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(
        `NanoVDBFile.fromURL: failed to fetch ${String(url)} (${response.status} ${response.statusText})`,
      );
    }
    const buffer = await response.arrayBuffer();
    return NanoVDBFile.fromArrayBuffer(buffer);
  }

  static fromArrayBuffer(buffer: ArrayBuffer): NanoVDBFile {
    const format = sniffFormat(buffer);
    const records = format === "segment" ? parseSegments(buffer) : parseRawGridBuffer(buffer);
    return new NanoVDBFile(buffer, records);
  }

  get grids(): readonly GridMetadata[] {
    return this.records.map((r) => r.metadata);
  }

  /** Zero-copy u32 view of grid `i`'s image where the codec/alignment allows it. */
  gridImage(i: number): Uint32Array {
    return this.getRecord(i, "gridImage").image;
  }

  /**
   * True when `gridImage(i)` is a zero-copy view into the buffer originally
   * passed to `fromArrayBuffer`/`fromURL`. False when a copy was required —
   * either ZIP decompression, or a codec-NONE grid segment that landed on a
   * non-4-byte-aligned offset (grid data follows a variable-length name, so
   * alignment isn't guaranteed).
   */
  isGridImageZeroCopy(i: number): boolean {
    return this.getRecord(i, "isGridImageZeroCopy").image.buffer === this.sourceBuffer;
  }

  private getRecord(i: number, caller: string): GridRecord {
    const record = this.records[i];
    if (!record) {
      throw new Error(`NanoVDBFile.${caller}: index ${i} out of range (file has ${this.records.length} grid(s))`);
    }
    return record;
  }
}
