/**
 * errors.ts — error taxonomy for the `.vdb` parser.
 *
 * `VdbFormatError` is for things that mean "this isn't a valid/parseable VDB
 * stream at all" (bad magic, truncated buffer, corrupt masks). `VdbUnsupportedError`
 * is for "this is a real VDB construct, but out of this parser's scope" (blosc,
 * point grids, tiles, non-5-4-3 trees, rotated transforms, ...) — per SPEC's
 * "throw clear errors" contract for anything beyond the v1 FloatGrid slice.
 */

export class VdbFormatError extends Error {
  constructor(message: string) {
    super(`vdb-web-tools: malformed .vdb: ${message}`);
    this.name = "VdbFormatError";
  }
}

export class VdbUnsupportedError extends Error {
  constructor(message: string) {
    super(`vdb-web-tools: not supported: ${message}`);
    this.name = "VdbUnsupportedError";
  }
}
