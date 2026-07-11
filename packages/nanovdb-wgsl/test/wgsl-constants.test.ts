import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
// @ts-expect-error — repo-level ESM helper without type declarations
import { generateWgslConstants } from "../../../scripts/lib/gen-wgsl-constants.mjs";

const jsonUrl = new URL("../vendor/stride-tables.json", import.meta.url);
const wgslUrl = new URL("../src/wgsl/pnanovdb-constants.generated.wgsl", import.meta.url);

async function loadTables() {
  return JSON.parse(await readFile(jsonUrl, "utf8"));
}

describe("wgsl-constants generator", () => {
  it("generated file is in sync", async () => {
    const committed = await readFile(wgslUrl, "utf8");
    const tables = await loadTables();
    const fresh = generateWgslConstants(tables);

    expect(fresh).toBe(committed);
  });

  it("spot checks", async () => {
    const tables = await loadTables();
    const generated = generateWgslConstants(tables);

    expect(generated).toContain("const PNANOVDB_GRID_SIZE: u32 = 672u;");
    expect(generated).toContain("const PNANOVDB_FLOAT_LEAF_OFF_TABLE: u32 = 96u;");
    expect(generated).toContain("fn pnanovdb_grid_type_value_strides_bits(grid_type: u32) -> u32 {");
  });
});
