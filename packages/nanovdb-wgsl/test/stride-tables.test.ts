import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
// @ts-expect-error — repo-level ESM helper without type declarations
import { extractStrideTables } from "../../../scripts/lib/pnanovdb-extract.mjs";

const headerUrl = new URL("../vendor/upstream/PNanoVDB.h", import.meta.url);
const jsonUrl = new URL("../vendor/stride-tables.json", import.meta.url);

async function loadTables() {
  return JSON.parse(await readFile(jsonUrl, "utf8"));
}

describe("vendored PNanoVDB.h + extracted stride tables", () => {
  it("header carries upstream license and the ABI the study pinned", async () => {
    const header = await readFile(headerUrl, "utf8");
    expect(header).toContain("SPDX-License-Identifier: Apache-2.0");
    expect(header).toContain("Copyright Contributors to the OpenVDB Project");
    expect(header).toContain("#define PNANOVDB_MAJOR_VERSION_NUMBER 32");
  });

  it("stride-tables.json is in sync with the vendored header (regen: node scripts/extract-stride-tables.mjs)", async () => {
    const header = await readFile(headerUrl, "utf8");
    const committed = await loadTables();
    const fresh = extractStrideTables(header);

    const { $meta: committedMeta, ...committedRest } = committed;
    const { $meta: _freshMeta, ...freshRest } = fresh;
    expect(committedRest).toEqual(freshRest);

    // Provenance must match the actual vendored bytes and the pinned ABI.
    const sha256 = createHash("sha256").update(header).digest("hex");
    expect(committedMeta.sha256).toBe(sha256);
    expect(committedMeta.abi).toBe("32.9.1");
    expect(committedMeta.commit).toBe("a532de5526ef791280b6483a872336a811a68542");
  });

  it("spot-checks values against upstream's documented layout", async () => {
    const t = await loadTables();
    // GridData/TreeData block sizes (FEASIBILITY.md §3 layout diagram).
    expect(t.defines.PNANOVDB_GRID_SIZE).toBe(672);
    expect(t.defines.PNANOVDB_TREE_SIZE).toBe(64);
    // v1 grid types exist with their NanoVDB.h ids.
    expect(t.gridTypes.FLOAT).toBe(1);
    expect(t.gridTypes.FP8).toBe(14);
    expect(t.gridTypes.FPN).toBe(16);
    // Float grid: leaf value table starts at +96, 32-bit values, 8³ leaf = 2144 B.
    const f = t.gridTypeConstants.FLOAT;
    expect(f.leaf_off_table).toBe(96);
    expect(f.value_stride_bits).toBe(32);
    expect(f.leaf_size).toBe(2144);
    // Quantized types: fixed stride lives in the aux array (constants-table
    // value_stride_bits is 0 for them). Fp8 = 8 bits/code → 608 B leaf;
    // FpN is variable-width (0 in both places).
    expect(t.gridTypeConstants.FP8.value_stride_bits).toBe(0);
    expect(t.auxArrays.grid_type_value_strides_bits[t.gridTypes.FP8]).toBe(8);
    expect(t.gridTypeConstants.FP8.leaf_size).toBe(608);
    expect(t.auxArrays.grid_type_value_strides_bits[t.gridTypes.FPN]).toBe(0);
  });

  it("every constants row has every struct field", async () => {
    const t = await loadTables();
    for (const [name, row] of Object.entries(t.gridTypeConstants)) {
      expect(Object.keys(row as object), name).toEqual(t.constantsFields);
    }
  });
});
