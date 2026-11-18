/**
 * write-nvdb.ts — `writeNvdb`: pack one or more finished grid images into one
 * segment of a `.nvdb` file, in NanoVDB's own `io::Segment` layout (IO.h
 * `writeGrid`: the header and every grid's metadata first, then the grids):
 *
 *   FileHeader(16) + [FileMetaData(176) + gridName(nameSize)] x N + [grid data] x N
 *
 * With one grid this is FileHeader, FileMetaData, name, image. Grid data is
 * the image (codec NONE) or, for codec ZIP (K4.18), `io::Internal::write`'s
 * framing: a u64 LE compressed byte count, then the zlib (RFC 1950) stream
 * of the image, as zlib's `compress()` writes it. `FileMetaData.fileSize` is
 * the on-disk size of that grid's data (8 + compressed bytes for ZIP).
 *
 * FileHeader magic is `NanoVDB2` (current file format). Each grid's
 * FileMetaData is derived by reading the grid image itself (GridData / TreeData
 * / RootData), so `writeNvdb` accepts any valid image — not just ones this
 * package built — and the metadata it writes always agrees with the image
 * (which is exactly what the loader's `mGridSize` cross-check verifies).
 *
 * The 8-byte name key (`io::stringHash`) is written as 0: NanoVDB's
 * name-based `readGrid` accepts 0 as "no key" and compares the names.
 */

import { zlibSync } from "fflate";
import {
  GRID_OFF_GRID_CLASS,
  GRID_OFF_GRID_NAME,
  GRID_OFF_GRID_SIZE,
  GRID_OFF_GRID_TYPE,
  GRID_OFF_VOXEL_SIZE,
  GRID_OFF_WORLD_BBOX,
  GRID_NAME_MAX,
  GRID_SIZE,
  MAGIC_FILE,
  ROOT_OFF_BBOX_MAX,
  ROOT_OFF_BBOX_MIN,
  TREE_OFF_NODE_COUNT_LEAF,
  TREE_OFF_NODE_COUNT_LOWER,
  TREE_OFF_NODE_COUNT_UPPER,
  TREE_OFF_NODE_OFFSET_ROOT,
  TREE_OFF_TILE_COUNT_LOWER,
  TREE_OFF_TILE_COUNT_ROOT,
  TREE_OFF_TILE_COUNT_UPPER,
  TREE_OFF_VOXEL_COUNT,
  VERSION_PACKED,
} from "./bytes.js";

const FILE_HEADER_SIZE = 16;
const FILE_METADATA_SIZE = 176;

// FileHeader offsets.
const FH_OFF_VERSION = 8;
const FH_OFF_GRID_COUNT = 12;
const FH_OFF_CODEC = 14;

// FileMetaData offsets.
const MD_OFF_GRID_SIZE = 0;
const MD_OFF_FILE_SIZE = 8;
const MD_OFF_NAME_KEY = 16;
const MD_OFF_VOXEL_COUNT = 24;
const MD_OFF_GRID_TYPE = 32;
const MD_OFF_GRID_CLASS = 36;
const MD_OFF_WORLD_BBOX_MIN = 40;
const MD_OFF_WORLD_BBOX_MAX = 64;
const MD_OFF_INDEX_BBOX_MIN = 88;
const MD_OFF_INDEX_BBOX_MAX = 100;
const MD_OFF_VOXEL_SIZE = 112;
const MD_OFF_NAME_SIZE = 136;
const MD_OFF_NODE_COUNT = 140; // u32[4]: leaf, lower, upper, root
const MD_OFF_TILE_COUNT = 156; // u32[3]: lower, upper, root
const MD_OFF_CODEC = 168;
const MD_OFF_BLIND_COUNT = 170;
const MD_OFF_VERSION = 172;

const CODEC_NONE = 0;
const CODEC_ZIP = 1;

export interface WriteNvdbOptions {
  /**
   * Per-grid codec. `'none'` (default) stores the images; `'zip'` stores each
   * as NanoVDB's `Codec::ZIP` (zlib), which native tools read when built with
   * zlib and `NanoVDBFile` / `NvdbStreamReader` decode.
   */
  codec?: "none" | "zip";
  /** zlib level for `codec: 'zip'`, 0–9. Default 9 (zlib's own default, used by `compress()`, is 6). */
  zipLevel?: number;
}

interface ImageMeta {
  image: Uint32Array;
  imageBytes: Uint8Array;
  gridSize: number;
  gridType: number;
  gridClass: number;
  voxelCount: number;
  worldBBox: Float64Array; // 6
  voxelSize: Float64Array; // 3
  indexBBoxMin: [number, number, number];
  indexBBoxMax: [number, number, number];
  nodeCount: [number, number, number]; // leaf, lower, upper
  tileCount: [number, number, number]; // lower, upper, root (active tiles)
  name: string;
  nameBytes: Uint8Array; // incl. trailing NUL
}

function imageDataView(image: Uint32Array): DataView {
  return new DataView(image.buffer, image.byteOffset, image.byteLength);
}

function extractMeta(image: Uint32Array, i: number): ImageMeta {
  const dv = imageDataView(image);
  const imageBytes = new Uint8Array(image.buffer, image.byteOffset, image.byteLength);
  const gridSize = Number(dv.getBigUint64(GRID_OFF_GRID_SIZE, true));
  if (gridSize !== image.byteLength) {
    throw new Error(
      `writeNvdb: grid #${i} GridData.mGridSize (${gridSize}) !== image byte length ` +
        `(${image.byteLength}); the image is malformed and the loader's mGridSize cross-check would reject it.`,
    );
  }
  const gridType = dv.getUint32(GRID_OFF_GRID_TYPE, true);
  const gridClass = dv.getUint32(GRID_OFF_GRID_CLASS, true);

  const worldBBox = new Float64Array(6);
  for (let a = 0; a < 6; a++) worldBBox[a] = dv.getFloat64(GRID_OFF_WORLD_BBOX + a * 8, true);
  const voxelSize = new Float64Array(3);
  for (let a = 0; a < 3; a++) voxelSize[a] = dv.getFloat64(GRID_OFF_VOXEL_SIZE + a * 8, true);

  // Name: C string in the inline field.
  const nameField = new Uint8Array(image.buffer, image.byteOffset + GRID_OFF_GRID_NAME, GRID_NAME_MAX);
  let end = nameField.indexOf(0);
  if (end < 0) end = nameField.length;
  const name = new TextDecoder().decode(nameField.subarray(0, end));
  const nameBytes = new TextEncoder().encode(name);
  const nameWithNul = new Uint8Array(nameBytes.length + 1);
  nameWithNul.set(nameBytes);

  // TreeData (immediately after GridData) + RootData for counts/bbox.
  const treeOff = GRID_SIZE;
  const voxelCount = Number(dv.getBigUint64(treeOff + TREE_OFF_VOXEL_COUNT, true));
  const nodeCount: [number, number, number] = [
    dv.getUint32(treeOff + TREE_OFF_NODE_COUNT_LEAF, true),
    dv.getUint32(treeOff + TREE_OFF_NODE_COUNT_LOWER, true),
    dv.getUint32(treeOff + TREE_OFF_NODE_COUNT_UPPER, true),
  ];
  const tileCount: [number, number, number] = [
    dv.getUint32(treeOff + TREE_OFF_TILE_COUNT_LOWER, true),
    dv.getUint32(treeOff + TREE_OFF_TILE_COUNT_UPPER, true),
    dv.getUint32(treeOff + TREE_OFF_TILE_COUNT_ROOT, true),
  ];
  const rootOff = treeOff + Number(dv.getBigUint64(treeOff + TREE_OFF_NODE_OFFSET_ROOT, true));
  const indexBBoxMin: [number, number, number] = [
    dv.getInt32(rootOff + ROOT_OFF_BBOX_MIN, true),
    dv.getInt32(rootOff + ROOT_OFF_BBOX_MIN + 4, true),
    dv.getInt32(rootOff + ROOT_OFF_BBOX_MIN + 8, true),
  ];
  const indexBBoxMax: [number, number, number] = [
    dv.getInt32(rootOff + ROOT_OFF_BBOX_MAX, true),
    dv.getInt32(rootOff + ROOT_OFF_BBOX_MAX + 4, true),
    dv.getInt32(rootOff + ROOT_OFF_BBOX_MAX + 8, true),
  ];

  return {
    image,
    imageBytes,
    gridSize,
    gridType,
    gridClass,
    voxelCount,
    worldBBox,
    voxelSize,
    indexBBoxMin,
    indexBBoxMax,
    nodeCount,
    tileCount,
    name,
    nameBytes: nameWithNul,
  };
}

/** A grid's on-disk data: the image (NONE) or `u64 size + zlib stream` (ZIP). */
function gridPayload(m: ImageMeta, codec: number, level: number): Uint8Array {
  if (codec === CODEC_NONE) return m.imageBytes;
  const compressed = zlibSync(m.imageBytes, { level: level as 9 });
  const out = new Uint8Array(8 + compressed.byteLength);
  new DataView(out.buffer).setBigUint64(0, BigInt(compressed.byteLength), true);
  out.set(compressed, 8);
  return out;
}

/**
 * Serializes `images` (each a complete NanoVDB grid image) into a `.nvdb`
 * file (`NanoVDB2` header, one segment, codec NONE or ZIP) as an
 * `ArrayBuffer`. Grids keep their order, so a coarse preview written first
 * arrives first in a stream.
 */
export function writeNvdb(images: Uint32Array[], opts: WriteNvdbOptions = {}): ArrayBuffer {
  if (images.length === 0) {
    throw new Error("writeNvdb: at least one grid image is required");
  }
  if (images.length > 0xffff) {
    throw new Error(`writeNvdb: too many grids (${images.length}); FileHeader.gridCount is a u16`);
  }
  const codecName = opts.codec ?? "none";
  if (codecName !== "none" && codecName !== "zip") {
    throw new Error(`writeNvdb: unknown codec "${String(codecName)}" (expected "none" or "zip")`);
  }
  const codec = codecName === "zip" ? CODEC_ZIP : CODEC_NONE;
  const level = opts.zipLevel ?? 9;
  if (!Number.isInteger(level) || level < 0 || level > 9) {
    throw new Error(`writeNvdb: zipLevel must be an integer 0-9, got ${level}`);
  }

  const metas = images.map((img, i) => extractMeta(img, i));
  const payloads = metas.map((m) => gridPayload(m, codec, level));

  let total = FILE_HEADER_SIZE;
  for (const m of metas) total += FILE_METADATA_SIZE + m.nameBytes.length;
  for (const p of payloads) total += p.byteLength;

  const buffer = new ArrayBuffer(total);
  const dv = new DataView(buffer);
  const u8 = new Uint8Array(buffer);

  // FileHeader.
  dv.setBigUint64(0, MAGIC_FILE, true);
  dv.setUint32(FH_OFF_VERSION, VERSION_PACKED, true);
  dv.setUint16(FH_OFF_GRID_COUNT, metas.length, true);
  dv.setUint16(FH_OFF_CODEC, codec, true);

  let off = FILE_HEADER_SIZE;
  for (let g = 0; g < metas.length; g++) {
    const m = metas[g]!;
    // FileMetaData.
    dv.setBigUint64(off + MD_OFF_GRID_SIZE, BigInt(m.gridSize), true);
    dv.setBigUint64(off + MD_OFF_FILE_SIZE, BigInt(payloads[g]!.byteLength), true);
    dv.setBigUint64(off + MD_OFF_NAME_KEY, 0n, true); // advisory; loader ignores
    dv.setBigUint64(off + MD_OFF_VOXEL_COUNT, BigInt(m.voxelCount), true);
    dv.setUint32(off + MD_OFF_GRID_TYPE, m.gridType, true);
    dv.setUint32(off + MD_OFF_GRID_CLASS, m.gridClass, true);
    for (let a = 0; a < 3; a++) {
      dv.setFloat64(off + MD_OFF_WORLD_BBOX_MIN + a * 8, m.worldBBox[a]!, true);
      dv.setFloat64(off + MD_OFF_WORLD_BBOX_MAX + a * 8, m.worldBBox[3 + a]!, true);
      dv.setInt32(off + MD_OFF_INDEX_BBOX_MIN + a * 4, m.indexBBoxMin[a]!, true);
      dv.setInt32(off + MD_OFF_INDEX_BBOX_MAX + a * 4, m.indexBBoxMax[a]!, true);
      dv.setFloat64(off + MD_OFF_VOXEL_SIZE + a * 8, m.voxelSize[a]!, true);
    }
    dv.setUint32(off + MD_OFF_NAME_SIZE, m.nameBytes.length, true);
    // nodeCount[4] = leaf, lower, upper, root(=1 when the grid has any content).
    dv.setUint32(off + MD_OFF_NODE_COUNT + 0, m.nodeCount[0], true);
    dv.setUint32(off + MD_OFF_NODE_COUNT + 4, m.nodeCount[1], true);
    dv.setUint32(off + MD_OFF_NODE_COUNT + 8, m.nodeCount[2], true);
    dv.setUint32(off + MD_OFF_NODE_COUNT + 12, 1, true);
    // tileCount[3] = active tiles at lower, upper, root (copied from TreeData).
    dv.setUint32(off + MD_OFF_TILE_COUNT + 0, m.tileCount[0], true);
    dv.setUint32(off + MD_OFF_TILE_COUNT + 4, m.tileCount[1], true);
    dv.setUint32(off + MD_OFF_TILE_COUNT + 8, m.tileCount[2], true);
    dv.setUint16(off + MD_OFF_CODEC, codec, true);
    dv.setUint16(off + MD_OFF_BLIND_COUNT, 0, true);
    dv.setUint32(off + MD_OFF_VERSION, VERSION_PACKED, true);
    off += FILE_METADATA_SIZE;

    // Grid name (incl. NUL).
    u8.set(m.nameBytes, off);
    off += m.nameBytes.length;
  }

  // Grid data, in the same order as the metadata.
  for (const p of payloads) {
    u8.set(p, off);
    off += p.byteLength;
  }

  return buffer;
}
