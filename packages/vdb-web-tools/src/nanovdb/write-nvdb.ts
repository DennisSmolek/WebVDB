/**
 * write-nvdb.ts — `writeNvdb`: pack one or more finished grid images into a
 * segment-format `.nvdb` file (codec NONE), byte-for-byte in the framing the
 * repo's own `NanoVDBFile` loader parses:
 *
 *   FileHeader(16) + [FileMetaData(176) + gridName(nameSize) + image] x N
 *
 * FileHeader magic is `NanoVDB2` (current file format). Each grid's
 * FileMetaData is derived by reading the grid image itself (GridData / TreeData
 * / RootData), so `writeNvdb` accepts any valid image — not just ones this
 * package built — and the metadata it writes always agrees with the image
 * (which is exactly what the loader's `mGridSize` cross-check verifies).
 *
 * Codec is NONE only in v1 (D3 non-goal list: ZIP write can be added with fflate
 * later). The 8-byte name key (`io::stringHash`) is written as 0: the loader
 * does not consult it, and the value is documented as advisory.
 */

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

export interface WriteNvdbOptions {
  /** Reserved for future codec selection; only NONE is implemented in v1. */
  codec?: "none";
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
    name,
    nameBytes: nameWithNul,
  };
}

/**
 * Serializes `images` (each a complete NanoVDB grid image) into a
 * segment-format `.nvdb` file (`NanoVDB2` header, codec NONE) as an
 * `ArrayBuffer`.
 */
export function writeNvdb(images: Uint32Array[], _opts: WriteNvdbOptions = {}): ArrayBuffer {
  if (images.length === 0) {
    throw new Error("writeNvdb: at least one grid image is required");
  }
  if (images.length > 0xffff) {
    throw new Error(`writeNvdb: too many grids (${images.length}); FileHeader.gridCount is a u16`);
  }

  const metas = images.map((img, i) => extractMeta(img, i));

  let total = FILE_HEADER_SIZE;
  for (const m of metas) total += FILE_METADATA_SIZE + m.nameBytes.length + m.gridSize;

  const buffer = new ArrayBuffer(total);
  const dv = new DataView(buffer);
  const u8 = new Uint8Array(buffer);

  // FileHeader.
  dv.setBigUint64(0, MAGIC_FILE, true);
  dv.setUint32(FH_OFF_VERSION, VERSION_PACKED, true);
  dv.setUint16(FH_OFF_GRID_COUNT, metas.length, true);
  dv.setUint16(FH_OFF_CODEC, CODEC_NONE, true);

  let off = FILE_HEADER_SIZE;
  for (const m of metas) {
    // FileMetaData.
    dv.setBigUint64(off + MD_OFF_GRID_SIZE, BigInt(m.gridSize), true);
    dv.setBigUint64(off + MD_OFF_FILE_SIZE, BigInt(m.gridSize), true); // codec NONE
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
    // tileCount[3] all 0 (this builder emits no active tiles).
    dv.setUint32(off + MD_OFF_TILE_COUNT + 0, 0, true);
    dv.setUint32(off + MD_OFF_TILE_COUNT + 4, 0, true);
    dv.setUint32(off + MD_OFF_TILE_COUNT + 8, 0, true);
    dv.setUint16(off + MD_OFF_CODEC, CODEC_NONE, true);
    dv.setUint16(off + MD_OFF_BLIND_COUNT, 0, true);
    dv.setUint32(off + MD_OFF_VERSION, VERSION_PACKED, true);
    off += FILE_METADATA_SIZE;

    // Grid name (incl. NUL).
    u8.set(m.nameBytes, off);
    off += m.nameBytes.length;

    // Grid image bytes.
    u8.set(m.imageBytes, off);
    off += m.gridSize;
  }

  return buffer;
}
