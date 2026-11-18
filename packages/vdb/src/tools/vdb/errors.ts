/**
 * errors.ts — the `.vdb` parser's errors, from the tools' one hierarchy (`../errors.ts`, D32).
 *
 * `VdbFormatError` is for things that mean "this isn't a valid/parseable VDB
 * stream at all" (bad magic, truncated buffer, corrupt masks). `VDBUnsupportedError`
 * is for "this is a real VDB construct, but out of this parser's scope" (blosc
 * codecs, tiles, non-5-4-3 trees, rotated transforms, ...) — per SPEC's
 * "throw clear errors" contract.
 */
export { VdbFormatError, VDBUnsupportedError } from "../errors.js";
