/**
 * Error type for the engine-free core (docs/dev/design/K1-core.md §4).
 * Every message should say what to do next.
 */
export type VDBErrorCode =
  | "VDB_UNSUPPORTED_DEVICE"
  | "VDB_BAD_IMAGE"
  | "VDB_UNSUPPORTED_GRID_TYPE"
  | "VDB_POOL_LIMIT"
  | "VDB_POOL_FULL"
  | "VDB_UNKNOWN_GRID"
  | "VDB_NO_STATS"
  | "VDB_TYPE_NOT_COMPILED"
  | "VDB_LAYOUT_MISMATCH"
  | "VDB_BAD_OPTION"
  | "VDB_DISPOSED"
  /** User WGSL (the frame renderer's emission colour hook) failed validation; the message names the line. */
  | "VDB_BAD_WGSL"
  /** An engine adapter found the engine's internals not behaving as the adapter expects (e.g. a renderer version it wasn't built for). */
  | "VDB_ENGINE_MISMATCH";

/** The error every core API throws, with a machine-readable {@link VDBErrorCode} in `code`. */
export class VDBError extends Error {
  /** The error kind; the message is prefixed with it. */
  readonly code: VDBErrorCode;

  /**
   * Creates the error; the message is prefixed with `@pmndrs/vdb [code]`.
   *
   * @param code The error kind.
   * @param message What went wrong and how to fix it.
   */
  constructor(code: VDBErrorCode, message: string) {
    super(`@pmndrs/vdb [${code}]: ${message}`);
    this.name = "VDBError";
    this.code = code;
  }
}
