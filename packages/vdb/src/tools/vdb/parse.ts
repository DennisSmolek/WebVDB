/**
 * parse.ts — top-level `.vdb` container reader (docs/archive/v1/SPEC.md §4).
 *
 * File layout (format facts cross-checked between `mjurczyk/openvdb`'s
 * `OpenVDBReader`/`GridDescriptor` and `vdb-rs`'s `VdbReader::new`/
 * `read_grid_descriptors`/`read_grid_internal`):
 *
 *   magic(u64) file_version(u32) [lib_major(u32) lib_minor(u32)]
 *   has_grid_offsets(u8) [is_compressed(u8) if 220<=v<222] guid(36 chars)
 *   metadata_map grid_count(u32)
 *   grid_count x {
 *     unique_name type_name [instance_parent] grid_pos(u64) block_pos(u64) end_pos(u64)
 *     -- seek to grid_pos --
 *     [compression(u32) if v>=222] metadata_map transform tree_topology tree_buffers
 *   }
 *
 * Unlike both references' lazy/two-pass grid-descriptor tables (built for
 * random access to a single named grid), `parseVdb` decodes every grid in
 * one linear pass and defensively re-seeks to each descriptor's `end_pos`
 * afterward — the `mjurczyk/openvdb` reader does the same "hack" for
 * multi-grid files, and it's cheap insurance against any of our own
 * byte-accounting bugs desyncing the stream for the next grid.
 */

import { ByteReader } from "./byte-reader.js";
import type { CompressionFlags, ValueFormat } from "./compression.js";
import {
  compressionFromBits,
  FLOAT_FORMAT,
  HALF_FORMAT,
  PTDATAIDX32_FORMAT,
  VEC3D_FORMAT,
  VEC3I_FORMAT,
  VEC3S_FORMAT,
} from "./compression.js";
import { VdbFormatError, VDBUnsupportedError } from "./errors.js";
import { createVdbGrid } from "./grid.js";
import { readMetadataMap } from "./metadata.js";
import { readTransform } from "./transform.js";
import { readTree } from "./tree.js";
import type { VdbBBox, VdbFile, VdbGrid, VdbValueType } from "./types.js";

const MAGIC_LOW = 0x56444220; // 'V','D','B',' ' read little-endian, low 32 bits
const MIN_SUPPORTED_VERSION = 213; // OPENVDB_FILE_VERSION_ROOTNODE_MAP
/**
 * OPENVDB_FILE_VERSION (225, `OPENVDB_FILE_VERSION_HALF_GRID`), still current in OpenVDB v14.0.0.
 * A newer file fails with `VDBUnsupportedError` before anything after the header is read: a
 * future layout could misparse silently. OpenVDB 13.2+ warns for 226-299 and rejects 300+
 * (`OPENVDB_UNSUPPORTED_FILE_VERSION`); we reject from 226 (DIVERGES V70).
 */
const MAX_SUPPORTED_VERSION = 225;
const VERSION_GRID_INSTANCING = 216;
const VERSION_BOOST_UUID = 218;
const VERSION_SELECTIVE_COMPRESSION = 220;
const VERSION_NODE_MASK_COMPRESSION = 222;
const VERSION_BLOSC_COMPRESSION = 223;

const GRID_TYPE_RE = /^Tree_([A-Za-z0-9]+)_(\d+)_(\d+)_(\d+)$/;

/** On-disk value types this parser reads (the `<valueType>` of the grid type), and their layout. */
const VALUE_FORMATS = {
  float: FLOAT_FORMAT,
  vec3s: VEC3S_FORMAT,
  vec3d: VEC3D_FORMAT,
  vec3i: VEC3I_FORMAT,
  ptdataidx32: PTDATAIDX32_FORMAT,
  // K7.C: OpenVDB 13+ `HalfGrid`. Values widen exactly to f32, so it parses as a float grid
  // (`valueType: "float"`, `gridType` keeps `Tree_half_5_4_3`; DIVERGES V69).
  half: HALF_FORMAT,
} as const satisfies Record<string, ValueFormat>;

type OnDiskValueType = keyof typeof VALUE_FORMATS;

/** The parsed grid's `valueType` for each on-disk value type. */
const PARSED_VALUE_TYPE: Record<OnDiskValueType, VdbValueType> = {
  float: "float",
  vec3s: "vec3s",
  vec3d: "vec3d",
  vec3i: "vec3i",
  ptdataidx32: "ptdataidx32",
  half: "float",
};

function isSupportedValueType(t: string): t is OnDiskValueType {
  return Object.prototype.hasOwnProperty.call(VALUE_FORMATS, t);
}

function readHeaderCompression(reader: ByteReader, fileVersion: number): CompressionFlags {
  let compression: CompressionFlags = { zip: false, activeMask: true, blosc: true }; // DEFAULT_COMPRESSION
  if (fileVersion < VERSION_BLOSC_COMPRESSION) {
    compression = { zip: true, activeMask: true, blosc: false };
  }
  if (fileVersion >= VERSION_SELECTIVE_COMPRESSION && fileVersion < VERSION_NODE_MASK_COMPRESSION) {
    const isCompressed = reader.bool();
    compression = isCompressed
      ? { zip: true, activeMask: false, blosc: false }
      : { zip: false, activeMask: false, blosc: false };
  }
  return compression;
}

function readBBoxFromMetadata(metadata: Record<string, unknown>): VdbBBox | null {
  const min = metadata["file_bbox_min"];
  const max = metadata["file_bbox_max"];
  if (Array.isArray(min) && Array.isArray(max) && min.length === 3 && max.length === 3) {
    return {
      min: [min[0], min[1], min[2]] as [number, number, number],
      max: [max[0], max[1], max[2]] as [number, number, number],
    };
  }
  return null;
}

export function parseVdb(buffer: ArrayBuffer): VdbFile {
  const reader = new ByteReader(buffer);

  if (reader.length < 8) {
    throw new VdbFormatError("buffer too short to contain the magic number");
  }
  const magicLow = reader.u32();
  const magicHigh = reader.u32();
  if (magicLow !== MAGIC_LOW || magicHigh !== 0) {
    throw new VdbFormatError(
      `bad magic number (expected 0x${MAGIC_LOW.toString(16)}, got 0x${magicLow.toString(16)}) — not a .vdb file`,
    );
  }

  const fileVersion = reader.u32();
  if (fileVersion < MIN_SUPPORTED_VERSION) {
    throw new VDBUnsupportedError(
      `file version ${fileVersion} is older than the minimum supported version ${MIN_SUPPORTED_VERSION}`,
    );
  }

  if (fileVersion > MAX_SUPPORTED_VERSION) {
    throw new VDBUnsupportedError(
      `file version ${fileVersion} is newer than the maximum supported version ${MAX_SUPPORTED_VERSION} ` +
        `(the file was written by a newer OpenVDB; OpenVDB v14 still writes ${MAX_SUPPORTED_VERSION}). ` +
        `Re-save it with OpenVDB 14 or earlier.`,
    );
  }

  reader.u32(); // library version major (informational only)
  reader.u32(); // library version minor (informational only)

  const hasGridOffsets = reader.bool();
  if (!hasGridOffsets) {
    throw new VDBUnsupportedError("VDB streams without grid offsets are not supported");
  }

  const headerCompression = readHeaderCompression(reader, fileVersion);

  if (fileVersion < VERSION_BOOST_UUID) {
    throw new VDBUnsupportedError(`file version ${fileVersion} predates the Boost-UUID header layout (< 218)`);
  }
  reader.fixedString(36); // uuid (informational only)

  readMetadataMap(reader); // file-level metadata (informational only)

  const gridCount = reader.u32();
  const grids: VdbGrid[] = [];

  for (let i = 0; i < gridCount; i++) {
    const uniqueName = reader.string();
    const name = uniqueName.split("\x1e")[0] ?? uniqueName;
    let gridType = reader.string();

    let savedAsHalfFloat = false;
    if (gridType.includes("_HalfFloat")) {
      savedAsHalfFloat = true;
      gridType = gridType.split("_HalfFloat").join("");
    }

    let instanceParentName = "";
    if (fileVersion >= VERSION_GRID_INSTANCING) {
      instanceParentName = reader.string();
    }
    if (instanceParentName !== "") {
      throw new VDBUnsupportedError(`grid "${name}": instanced grids (sharing another grid's tree) are not supported`);
    }

    const gridPos = reader.u64AsNumber();
    const blockPos = reader.u64AsNumber();
    void blockPos; // not needed: we read topology+buffers in one contiguous pass
    const endPos = reader.u64AsNumber();

    reader.seek(gridPos);

    const compression =
      fileVersion >= VERSION_NODE_MASK_COMPRESSION
        ? compressionFromBits(reader.u32())
        : headerCompression;

    const metadata = readMetadataMap(reader);
    const useHalf = savedAsHalfFloat || metadata["is_saved_as_half_float"] === true;

    const match = GRID_TYPE_RE.exec(gridType);
    if (!match) {
      throw new VDBUnsupportedError(
        `grid "${name}": unrecognized grid type "${gridType}" (expected "Tree_<valueType>_<log2dim>_<log2dim>_<log2dim>")`,
      );
    }
    const [, valueType, d0, d1, d2] = match;
    if (!isSupportedValueType(valueType!)) {
      throw new VDBUnsupportedError(
        `grid "${name}": value type "${valueType}" is not supported ` +
          `(supported: ${Object.keys(VALUE_FORMATS).join(", ")})`,
      );
    }
    if (d0 !== "5" || d1 !== "4" || d2 !== "3") {
      throw new VDBUnsupportedError(
        `grid "${name}": tree configuration ${d0}-${d1}-${d2} is not supported (only the 5-4-3 tree is in scope)`,
      );
    }

    const transform = readTransform(reader);
    const { background, leaves, tiles, activeVoxelCount, pointLeaves, pointOutputBudget } = readTree(reader, {
      fileVersion,
      compression,
      useHalf,
      format: VALUE_FORMATS[valueType],
      pointData: valueType === "ptdataidx32",
    });

    grids.push(
      createVdbGrid({
        name,
        gridType,
        valueType: PARSED_VALUE_TYPE[valueType],
        transform,
        metadata,
        indexBBox: readBBoxFromMetadata(metadata),
        activeVoxelCount,
        background,
        leaves,
        tiles,
        ...(pointLeaves ? { pointLeaves, pointOutputBudget } : {}),
      }),
    );

    // Defensive re-sync (matches the `mjurczyk/openvdb` reader's own hack):
    // land exactly on the next grid descriptor regardless of any drift.
    reader.seek(endPos);
  }

  return { fileVersion, grids };
}
