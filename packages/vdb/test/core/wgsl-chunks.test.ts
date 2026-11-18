import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
// @ts-expect-error: repo-level ESM helper without type declarations
import { extractWgslChunks, generateChunksTs, pinnedCommitFromVendorDoc, sha256 } from "../../../../scripts/lib/wgsl-chunks.mjs";
import { PNANOVDB_CHUNKS, PNANOVDB_SOURCE_SHA256 } from "../../src/core/wgsl/pnanovdb.chunks.generated.js";

const wgslUrl = new URL("../../vendor/pnanovdb.wgsl", import.meta.url);
const vendorDocUrl = new URL("../../vendor/VENDOR.md", import.meta.url);
const chunksUrl = new URL("../../src/core/wgsl/pnanovdb.chunks.generated.ts", import.meta.url);

interface Chunk {
  name: string;
  kind: string;
  deps: string[];
  text: string;
}

describe("pnanovdb.chunks.generated.ts (K1.4a declaration table)", () => {
  it("is in sync with the vendored file (regen: node scripts/extract-wgsl-chunks.mjs)", async () => {
    const source = await readFile(wgslUrl, "utf8");
    const pinnedCommit = pinnedCommitFromVendorDoc(await readFile(vendorDocUrl, "utf8"));
    const fresh = generateChunksTs(source, { pinnedCommit });
    expect(fresh).toBe(await readFile(chunksUrl, "utf8"));
    expect(PNANOVDB_SOURCE_SHA256).toBe(sha256(source));
  });

  it("covers every top-level declaration once, in file order", async () => {
    const source = await readFile(wgslUrl, "utf8");
    const chunks: Chunk[] = extractWgslChunks(source);
    expect(chunks).toHaveLength(293); // design note §3.5: 293 declarations
    expect(new Set(chunks.map((c) => c.name)).size).toBe(chunks.length);
    // Every declaration the regex can see at column 0 is in the table, in order.
    const topLevel = [...source.matchAll(/^(fn|struct|const|alias)\s+(\w+)/gm)].map((m) => m[2]);
    expect(chunks.map((c) => c.name)).toEqual(topLevel);
    for (const c of PNANOVDB_CHUNKS) expect(c.name).toMatch(/^(pnanovdb_|PNANOVDB_)/);
  });

  it("drops comments and keeps declaration text otherwise verbatim", () => {
    for (const c of PNANOVDB_CHUNKS) {
      expect(c.text).not.toMatch(/\/\/|\/\*/);
      expect(c.text).not.toContain("`");
      expect(c.text.startsWith(`${c.kind} ${c.name}`)).toBe(true);
    }
    const read = PNANOVDB_CHUNKS.find((c) => c.name === "pnanovdb_buf_read_uint32");
    expect(read?.text).toContain("nanovdb_buffer[(buf.byte_offset / 4u) + (byte_offset / 4u)]");
  });

  it("records dependencies on other declarations only", () => {
    const names = new Set(PNANOVDB_CHUNKS.map((c) => c.name));
    for (const c of PNANOVDB_CHUNKS) {
      for (const d of c.deps) {
        expect(names.has(d)).toBe(true);
        expect(d).not.toBe(c.name);
      }
    }
    const typed = PNANOVDB_CHUNKS.find((c) => c.name === "pnanovdb_sample_trilinear_typed");
    expect(typed?.deps).toContain("pnanovdb_readaccessor_get_value_float");
    const tileZero = PNANOVDB_CHUNKS.find((c) => c.name === "pnanovdb_root_get_tile_zero");
    expect(tileZero?.deps).toContain("pnanovdb_grid_type_constants");
  });
});
