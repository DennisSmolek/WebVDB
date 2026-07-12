import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
// @ts-expect-error — repo-level ESM helper without type declarations
import { generateStrideTablesTs } from "../../../scripts/lib/gen-stride-tables-ts.mjs";

const jsonUrl = new URL("../vendor/stride-tables.json", import.meta.url);
const tsUrl = new URL("../src/cpu/stride-tables.generated.ts", import.meta.url);

async function loadTables() {
  return JSON.parse(await readFile(jsonUrl, "utf8"));
}

describe("stride-tables.generated.ts generator", () => {
  it("generated file is in sync (regen: node scripts/extract-stride-tables.mjs)", async () => {
    const committed = await readFile(tsUrl, "utf8");
    const tables = await loadTables();
    const fresh = generateStrideTablesTs(tables);

    expect(fresh).toBe(committed);
  });

  it("mirrors the JSON exactly", async () => {
    const tables = await loadTables();
    const generated = generateStrideTablesTs(tables);

    // Strip the header comment + `export const strideTables = ... as const;`
    // wrapper and confirm what's left round-trips through JSON parsing back
    // to the exact same data as the source JSON.
    const match = /export const strideTables = ([\s\S]*) as const;\n$/.exec(generated);
    expect(match).not.toBeNull();
    const embedded = JSON.parse(match![1]!);
    expect(embedded).toEqual(tables);
  });
});
