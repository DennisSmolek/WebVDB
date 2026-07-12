/**
 * parse.ts — top-level `.vdb` container reader (docs/SPEC.md §4).
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
import type { CompressionFlags } from "./compression.js";
import { compressionFromBits } from "./compression.js";
import { VdbFormatError, VdbUnsupportedError } from "./errors.js";
import { VdbGridImpl } from "./grid.js";
import { readMetadataMap } from "./metadata.js";
import { readTransform } from "./transform.js";
import { readTree } from "./tree.js";
import type { VdbBBox, VdbFile, VdbGrid } from "./types.js";

const MAGIC_LOW = 0x56444220; // 'V','D','B',' ' read little-endian, low 32 bits
const MIN_SUPPORTED_VERSION = 213; // OPENVDB_FILE_VERSION_ROOTNODE_MAP
const VERSION_GRID_INSTANCING = 216;
const VERSION_BOOST_UUID = 218;
const VERSION_SELECTIVE_COMPRESSION = 220;
const VERSION_NODE_MASK_COMPRESSION = 222;
const VERSION_BLOSC_COMPRESSION = 223;

const GRID_TYPE_RE = /^Tree_([A-Za-z0-9]+)_(\d+)_(\d+)_(\d+)$/;

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
    throw new VdbUnsupportedError(
      `file version ${fileVersion} is older than the minimum supported version ${MIN_SUPPORTED_VERSION}`,
    );
  }

  reader.u32(); // library version major (informational only)
  reader.u32(); // library version minor (informational only)

  const hasGridOffsets = reader.bool();
  if (!hasGridOffsets) {
    throw new VdbUnsupportedError("VDB streams without grid offsets are not supported");
  }

  const headerCompression = readHeaderCompression(reader, fileVersion);

  if (fileVersion < VERSION_BOOST_UUID) {
    throw new VdbUnsupportedError(`file version ${fileVersion} predates the Boost-UUID header layout (< 218)`);
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
      throw new VdbUnsupportedError(`grid "${name}": instanced grids (sharing another grid's tree) are not supported`);
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
      throw new VdbUnsupportedError(
        `grid "${name}": unrecognized grid type "${gridType}" (expected "Tree_<valueType>_<log2dim>_<log2dim>_<log2dim>")`,
      );
    }
    const [, valueType, d0, d1, d2] = match;
    if (valueType !== "float") {
      throw new VdbUnsupportedError(
        `grid "${name}": value type "${valueType}" is not supported (only FloatGrid is in scope for this parser)`,
      );
    }
    if (d0 !== "5" || d1 !== "4" || d2 !== "3") {
      throw new VdbUnsupportedError(
        `grid "${name}": tree configuration ${d0}-${d1}-${d2} is not supported (only the 5-4-3 tree is in scope)`,
      );
    }

    const transform = readTransform(reader);
    const { background, leaves, activeVoxelCount } = readTree(reader, { fileVersion, compression, useHalf });

    grids.push(
      new VdbGridImpl({
        name,
        gridType,
        transform,
        metadata,
        indexBBox: readBBoxFromMetadata(metadata),
        activeVoxelCount,
        background,
        leaves,
      }),
    );

    // Defensive re-sync (matches the `mjurczyk/openvdb` reader's own hack):
    // land exactly on the next grid descriptor regardless of any drift.
    reader.seek(endPos);
  }

  return { fileVersion, grids };
}
