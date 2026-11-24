/**
 * Reads the few NanoVDB `GridData` / `TreeData` / `RootData` fields a pool record
 * needs, by byte offset (docs/dev/design/K1-core.md §1.4). The core never imports
 * `tools` (D20), so the offsets are restated here; `test/core/image-header.test.ts`
 * cross-checks every one against `tools`' constants and `vendor/stride-tables.json`
 * (ABI 32.9.1, PNanoVDB.h).
 */

import { VDBError } from "../errors.js";
import { halfBitsToNumber } from "../half.js";
import { VDB_GRID_TYPE_IDS, type VDBGridTypeName } from "../types.js";
import type { VDBVec3 } from "./layout.js";

/** Byte offsets inside a NanoVDB grid image used by the pool. */
export const NANOVDB_OFFSETS = {
  /** `GridData` size; `TreeData` follows it. */
  gridSize: 672,
  gridMagic: 0,
  gridVersion: 16,
  gridFlags: 20,
  /** u64 `mGridSize`. */
  gridByteSize: 32,
  gridName: 40,
  gridNameMax: 256,
  gridMap: 296,
  mapMatD: 88,
  mapInvMatD: 160,
  mapVecD: 232,
  gridWorldBBox: 560,
  gridVoxelSize: 608,
  gridClass: 632,
  gridType: 636,
  treeSize: 64,
  /** u64, relative to `TreeData`. */
  treeNodeOffsetRoot: 24,
  rootBBoxMin: 0,
  rootBBoxMax: 12,
  /** `RootData` base size before the typed background/stats. */
  rootBaseSize: 28,
  /** Typed root fields for Float/Fp4/Fp8/Fp16/FpN (all store f32 here). */
  rootBackground: 28,
  rootMin: 32,
  rootMax: 36,
  /** Half (K7.H): background, min and max are binary16 at 28/30/32 (`pnanovdb_grid_type_constants` row 9). */
  rootHalfBackground: 28,
  rootHalfMin: 30,
  rootHalfMax: 32,
  /** Smallest `RootData` for the supported types (stride table `root_size`). */
  rootSize: 64,
} as const;

/** `GridData.mFlags` bit for min/max statistics (NanoVDB `GridFlags::HasMinMax`). */
export const NANOVDB_FLAG_HAS_MIN_MAX = 1 << 2;

/** NanoVDB ABI major this core reads. */
export const NANOVDB_ABI_MAJOR = 32;

/** `"NanoVDB1"` (grid) as two little-endian u32 words. */
const MAGIC_GRID_LO = 0x6f6e614e; // "Nano"
const MAGIC_GRID_HI_CURRENT = 0x31424456; // "VDB1"
const MAGIC_GRID_HI_LEGACY = 0x30424456; // "VDB0"

const TYPE_BY_ID = new Map<number, VDBGridTypeName>(
  (Object.entries(VDB_GRID_TYPE_IDS) as [VDBGridTypeName, number][]).map(([name, id]) => [id, name]),
);

/** Maps a NanoVDB `GridType` id to its pool type name, or `null` when the pool can't hold it. */
export function gridTypeName(id: number): VDBGridTypeName | null {
  return TYPE_BY_ID.get(id) ?? null;
}

/** NanoVDB `GridClass` names by id (NanoVDB.h). */
const GRID_CLASS_NAMES = [
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
] as const;

/** NanoVDB `GridClass` name for an id (`"Unknown(n)"` past the known range). */
export function gridClassName(id: number): string {
  return GRID_CLASS_NAMES[id] ?? `Unknown(${id})`;
}

/** What the pool reads from a grid image's header. */
export interface VDBImageHeader {
  gridType: VDBGridTypeName;
  gridTypeId: number;
  gridClass: number;
  name: string;
  /** `GridData.mFlags`. */
  flags: number;
  hasStats: boolean;
  /** Bytes in the image (`mGridSize`, equal to the array's byte length). */
  byteLength: number;
  /** f64 3x3 row-major index -> world linear part. */
  matD: Float64Array;
  /** f64 3x3 row-major world -> index linear part. */
  invMatD: Float64Array;
  /** f64 translation: `world = matD * ijk + vecD`. */
  vecD: Float64Array;
  indexBBoxMin: VDBVec3;
  indexBBoxMax: VDBVec3;
  voxelSize: VDBVec3;
  background: number;
  /** Root min statistic (f32 as stored; a Half grid's binary16 decoded exactly); meaningful only when `hasStats`. */
  rootMin: number;
  /** Root max statistic; meaningful only when `hasStats`. */
  rootMax: number;
}

function bad(message: string): never {
  throw new VDBError("VDB_BAD_IMAGE", message);
}

/**
 * Parses and validates the header of a NanoVDB grid image (one grid, as
 * `NvdbFile` from `@pmndrs/vdb/tools` returns it). Throws `VDB_BAD_IMAGE` or
 * `VDB_UNSUPPORTED_GRID_TYPE` with the fix in the message.
 */
export function readImageHeader(image: Uint32Array, what = "image"): VDBImageHeader {
  if (!(image instanceof Uint32Array)) {
    bad(`${what} must be a Uint32Array holding one NanoVDB grid (NvdbFile from @pmndrs/vdb/tools returns one per grid).`);
  }
  const O = NANOVDB_OFFSETS;
  const byteLength = image.byteLength;
  const minBytes = O.gridSize + O.treeSize + O.rootSize;
  if (byteLength < minBytes) {
    bad(`${what} is ${byteLength} bytes; a NanoVDB grid is at least ${minBytes}. Pass a whole grid image, not a slice.`);
  }
  const view = new DataView(image.buffer, image.byteOffset, byteLength);
  const u32 = (at: number) => view.getUint32(at, true);

  const magicLo = u32(O.gridMagic);
  const magicHi = u32(O.gridMagic + 4);
  if (magicLo !== MAGIC_GRID_LO || (magicHi !== MAGIC_GRID_HI_CURRENT && magicHi !== MAGIC_GRID_HI_LEGACY)) {
    bad(
      `${what} does not start with the NanoVDB grid magic "NanoVDB1". Pass a grid image (NvdbFile.grids[i].image), ` +
        `not a whole .nvdb file.`,
    );
  }
  const version = u32(O.gridVersion);
  const major = (version >>> 21) & 0x7ff;
  if (major !== NANOVDB_ABI_MAJOR) {
    bad(
      `${what} is NanoVDB ABI ${major}.${(version >>> 10) & 0x7ff}; this core reads ABI ${NANOVDB_ABI_MAJOR}. ` +
        `Re-export it with NanoVDB ${NANOVDB_ABI_MAJOR}.x.`,
    );
  }
  const sizeLo = u32(O.gridByteSize);
  const sizeHi = u32(O.gridByteSize + 4);
  if (sizeHi !== 0) {
    bad(`${what} declares a grid of 4 GiB or more; the pool addresses images with 32 bits. Split or downsample the grid.`);
  }
  if (sizeLo !== byteLength) {
    bad(
      `${what} declares mGridSize ${sizeLo} but the array holds ${byteLength} bytes. Pass exactly one grid ` +
        `(a multi-grid buffer must be split, which NvdbFile does).`,
    );
  }

  const gridTypeId = u32(O.gridType);
  const gridType = gridTypeName(gridTypeId);
  if (gridType === null) {
    throw new VDBError(
      "VDB_UNSUPPORTED_GRID_TYPE",
      `${what} has NanoVDB GridType ${gridTypeId}; the pool holds Float (1), Half (9), Fp4 (13), Fp8 (14), Fp16 (15) and FpN (16). ` +
        `Convert it, e.g. quantize() from @pmndrs/vdb/tools or nanovdb_convert --fp8.`,
    );
  }
  const flags = u32(O.gridFlags);
  const gridClass = u32(O.gridClass);
  if (gridClass > 0xff) bad(`${what} has GridClass ${gridClass}; NanoVDB classes are < 256, so the header is corrupt. Re-export it.`);

  const f64s = (at: number, n: number) => {
    const out = new Float64Array(n);
    for (let k = 0; k < n; k++) out[k] = view.getFloat64(at + 8 * k, true);
    return out;
  };
  const matD = f64s(O.gridMap + O.mapMatD, 9);
  const invMatD = f64s(O.gridMap + O.mapInvMatD, 9);
  const vecD = f64s(O.gridMap + O.mapVecD, 3);
  for (const [label, arr] of [["matD", matD], ["invMatD", invMatD], ["vecD", vecD]] as const) {
    if (!arr.every(Number.isFinite)) {
      bad(`${what} has a non-finite map (${label}). Rewrite its transform (transform() in @pmndrs/vdb/tools) and re-add it.`);
    }
  }
  const det =
    matD[0]! * (matD[4]! * matD[8]! - matD[5]! * matD[7]!) -
    matD[1]! * (matD[3]! * matD[8]! - matD[5]! * matD[6]!) +
    matD[2]! * (matD[3]! * matD[7]! - matD[4]! * matD[6]!);
  if (det === 0) bad(`${what} has a singular map (zero voxel size). Rewrite its transform and re-add it.`);
  const voxelSize = Array.from(f64s(O.gridVoxelSize, 3)) as VDBVec3;

  const rootRel = Number(view.getBigUint64(O.gridSize + O.treeNodeOffsetRoot, true));
  const root = O.gridSize + rootRel;
  if (!Number.isSafeInteger(rootRel) || root + O.rootSize > byteLength) {
    bad(`${what} has a root node offset (${rootRel}) outside the image. The file is corrupt; re-export it.`);
  }
  const i3 = (at: number): VDBVec3 => [view.getInt32(at, true), view.getInt32(at + 4, true), view.getInt32(at + 8, true)];

  const nameBytes = new Uint8Array(image.buffer, image.byteOffset + O.gridName, O.gridNameMax);
  let end = nameBytes.indexOf(0);
  if (end < 0) end = nameBytes.length;
  const name = new TextDecoder().decode(nameBytes.subarray(0, end));

  return {
    gridType,
    gridTypeId,
    gridClass,
    name,
    flags,
    hasStats: (flags & NANOVDB_FLAG_HAS_MIN_MAX) !== 0,
    byteLength,
    matD,
    invMatD,
    vecD,
    indexBBoxMin: i3(root + O.rootBBoxMin),
    indexBBoxMax: i3(root + O.rootBBoxMax),
    voxelSize,
    ...(gridType === "half"
      ? {
          background: halfBitsToNumber(view.getUint16(root + O.rootHalfBackground, true)),
          rootMin: halfBitsToNumber(view.getUint16(root + O.rootHalfMin, true)),
          rootMax: halfBitsToNumber(view.getUint16(root + O.rootHalfMax, true)),
        }
      : {
          background: view.getFloat32(root + O.rootBackground, true),
          rootMin: view.getFloat32(root + O.rootMin, true),
          rootMax: view.getFloat32(root + O.rootMax, true),
        }),
  };
}
