/**
 * errors.ts — the one error hierarchy of `@pmndrs/vdb/tools` (D32).
 *
 * Every error the tools throw on bad input is a `VDBToolsError` with a `code`, so one
 * `instanceof VDBToolsError` check covers every format:
 *
 * - `VdbFormatError` (`VDB_BAD_VDB`), `NvdbFormatError` (`VDB_BAD_NVDB`) and `WvdbFormatError`
 *   (`VDB_BAD_WVDB`): the bytes are not a valid file of that format (bad magic, truncated, corrupt).
 * - `VDBUnsupportedError` (`VDB_UNSUPPORTED`): a real construct of the format that these tools don't
 *   handle (BLOSC `.nvdb`, a codec, a tree configuration, a grid type a reader can't decode).
 * - `VDBToolsError` itself: an argument or option is invalid (`VDB_BAD_OPTION`), a grid image is not a
 *   NanoVDB grid image (`VDB_BAD_IMAGE`), or an operation doesn't take the image's grid type
 *   (`VDB_UNSUPPORTED_GRID_TYPE`).
 *
 * The core (`@pmndrs/vdb`) has its own `VDBError` with the same message shape; tools never import the
 * core (D20), so the two are separate classes. The codes that mean the same thing share a name.
 */

/** What went wrong, for `VDBToolsError.code`. */
export type VDBToolsErrorCode =
  /** Malformed `.vdb` bytes (`VdbFormatError`). */
  | "VDB_BAD_VDB"
  /** Malformed `.nvdb` bytes (`NvdbFormatError`). */
  | "VDB_BAD_NVDB"
  /** Malformed `.wvdb` bytes (`WvdbFormatError`). */
  | "VDB_BAD_WVDB"
  /** Valid data the tools don't handle (`VDBUnsupportedError`). */
  | "VDB_UNSUPPORTED"
  /** The operation doesn't take this grid type (e.g. a Float-only CPU reader given a Vec3f grid). */
  | "VDB_UNSUPPORTED_GRID_TYPE"
  /** A `Uint32Array` argument is not a NanoVDB grid image. */
  | "VDB_BAD_IMAGE"
  /** An argument or option is out of range or inconsistent. */
  | "VDB_BAD_OPTION";

/** Base class of every error `@pmndrs/vdb/tools` throws on bad input. */
export class VDBToolsError extends Error {
  readonly code: VDBToolsErrorCode;

  constructor(code: VDBToolsErrorCode, message: string) {
    super(`@pmndrs/vdb/tools [${code}]: ${message}`);
    this.name = "VDBToolsError";
    this.code = code;
  }
}

/** The bytes are not a valid OpenVDB `.vdb` stream (bad magic, truncated buffer, corrupt masks). */
export class VdbFormatError extends VDBToolsError {
  constructor(message: string) {
    super("VDB_BAD_VDB", `malformed .vdb: ${message}`);
    this.name = "VdbFormatError";
  }
}

/** A malformed or truncated `.nvdb` (K4.20), from `NvdbFile` and the stream reader alike. */
export class NvdbFormatError extends VDBToolsError {
  constructor(message: string) {
    super("VDB_BAD_NVDB", `malformed .nvdb: ${message}`);
    this.name = "NvdbFormatError";
  }
}

/** A `.wvdb` file breaks the spec (bad magic, header, offsets, sizes). */
export class WvdbFormatError extends VDBToolsError {
  constructor(message: string) {
    super("VDB_BAD_WVDB", `malformed .wvdb: ${message}`);
    this.name = "WvdbFormatError";
  }
}

/**
 * A real construct of the format that is out of these tools' scope (BLOSC `.nvdb`, point grids'
 * unsupported codecs, non-5-4-3 trees, rotated transforms, ...), per SPEC's "throw clear errors" contract.
 */
export class VDBUnsupportedError extends VDBToolsError {
  constructor(message: string) {
    super("VDB_UNSUPPORTED", `not supported: ${message}`);
    this.name = "VDBUnsupportedError";
  }
}
