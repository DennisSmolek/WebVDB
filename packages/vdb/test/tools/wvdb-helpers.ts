// Shared helpers for the .wvdb tests (K4.3). Not a test file.
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

/** Committed copies of his CC0 `.wvdb` files (see fixtures/wvdb/README.md). */
export const WVDB_FIXTURES = path.join(here, "fixtures", "wvdb");

export function fixture(name: string): Uint8Array {
  return new Uint8Array(readFileSync(path.join(WVDB_FIXTURES, name)));
}

/** His example assets in a mjurczyk/openvdb checkout named by MJURCZYK_OPENVDB_DIR, or null. */
export function hisAsset(name: string): string | null {
  const root = process.env.MJURCZYK_OPENVDB_DIR;
  if (!root) return null;
  const p = path.join(root, "src", "docs", "examples", "public", "assets", name);
  return existsSync(p) ? p : null;
}

/** Our fetched openvdb.org sample (pnpm fixtures), else his checkout's copy, else null. */
export function vdbSample(name: string): string | null {
  const ours = path.resolve(here, "../../../../fixtures/vdb-samples", name);
  if (existsSync(ours)) return ours;
  return hisAsset(name);
}

export function readBytes(p: string): Uint8Array {
  return new Uint8Array(readFileSync(p));
}

export const bytesEqual = (a: ArrayLike<number>, b: ArrayLike<number>): boolean => {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
};

/** Header JSON with chunk offsets/lengths zeroed (they depend on the zlib build). */
export const headerWithoutOffsets = (h: unknown): unknown =>
  JSON.parse(JSON.stringify(h, (k, v) => (k === "offset" || k === "length" ? 0 : v)));
