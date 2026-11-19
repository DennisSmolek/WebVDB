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
  /** An engine adapter found the engine's internals not behaving as the adapter expects (e.g. a renderer version it wasn't built for). */
  | "VDB_ENGINE_MISMATCH";

export class VDBError extends Error {
  readonly code: VDBErrorCode;

  constructor(code: VDBErrorCode, message: string) {
    super(`@pmndrs/vdb [${code}]: ${message}`);
    this.name = "VDBError";
    this.code = code;
  }
}
