/**
 * nanovdb-wgsl — renderer-agnostic NanoVDB traversal WGSL module + TS loader.
 *
 * Phase 0 stub: constants and API surface only. The real `.nvdb` loader
 * lands in Phase 1 (docs/PLAN.md); the WGSL module itself is vendored in
 * `vendor/pnanovdb.wgsl` (see vendor/VENDOR.md for the fork policy).
 */

/** `.nvdb` FileHeader size in bytes (magic + version + grid count + codec). */
export const FILE_HEADER_SIZE = 16;

/** Per-grid FileMetaData size in bytes (excluding the trailing grid name). */
export const FILE_METADATA_SIZE = 176;

/** GridData block size in bytes (start of every NanoVDB grid image). */
export const GRID_DATA_SIZE = 672;

/**
 * NanoVDB 8-byte magic strings: `NanoVDB0` = file header, `NanoVDB1` =
 * raw grid buffer, `NanoVDB2` = file+grid. Sniffed by the Phase 1 loader.
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

/**
 * Parses a `.nvdb` file (or raw grid buffer) and exposes GPU-ready grid
 * images as flat u32 views — "a valid NanoVDB grid image in a flat u32
 * buffer" is the single contract between the CPU and GPU halves (SPEC §1).
 *
 * Phase 1 deliverable — every member currently throws.
 */
export class NanoVDBFile {
  static async fromURL(_url: string | URL): Promise<NanoVDBFile> {
    throw new Error("NanoVDBFile.fromURL: not implemented (Phase 1)");
  }

  static fromArrayBuffer(_buffer: ArrayBuffer): NanoVDBFile {
    throw new Error("NanoVDBFile.fromArrayBuffer: not implemented (Phase 1)");
  }

  get grids(): readonly GridMetadata[] {
    throw new Error("NanoVDBFile.grids: not implemented (Phase 1)");
  }

  /** Zero-copy u32 view of grid `i`'s image where the codec allows it. */
  gridImage(_i: number): Uint32Array {
    throw new Error("NanoVDBFile.gridImage: not implemented (Phase 1)");
  }
}

/**
 * Resolvable URL of the vendored WGSL module, for runtimes that fetch it
 * (Node tests, non-bundler setups). Vite/bundler consumers should import
 * `nanovdb-wgsl/pnanovdb.wgsl?raw` instead.
 */
export const pnanovdbWgslUrl = new URL("../vendor/pnanovdb.wgsl", import.meta.url);
