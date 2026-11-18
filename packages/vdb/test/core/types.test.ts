import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import * as core from "../../src/core/index.js";
import { VDB_GRID_TYPE_IDS, VDB_GRID_TYPE_NAMES, type VDBGridTypeName } from "../../src/core/types.js";
import { gridTypeName } from "../../src/core/pool/image-header.js";
import { PNANOVDB_CHUNKS } from "../../src/core/wgsl/pnanovdb.chunks.generated.js";
import * as bytes from "../../src/tools/nanovdb/bytes.js";
import { gridTypeIds } from "../../src/tools/cpu/stride-tables.js";

const header = readFileSync(new URL("../../vendor/upstream/PNanoVDB.h", import.meta.url), "utf8");
const stride = JSON.parse(readFileSync(new URL("../../vendor/stride-tables.json", import.meta.url), "utf8"));

describe("shared grid-type table (core/types.ts)", () => {
  it("names, ids and the union agree", () => {
    expect([...VDB_GRID_TYPE_NAMES].sort()).toEqual(Object.keys(VDB_GRID_TYPE_IDS).sort());
    expect(new Set(Object.values(VDB_GRID_TYPE_IDS)).size).toBe(VDB_GRID_TYPE_NAMES.length);
    for (const name of VDB_GRID_TYPE_NAMES) expect(gridTypeName(VDB_GRID_TYPE_IDS[name])).toBe(name);
    expect(gridTypeName(2)).toBeNull();
  });

  it("matches the NanoVDB GridType ids in PNanoVDB.h, the stride tables, the vendored WGSL and tools", () => {
    const wgslConst = new Map(
      PNANOVDB_CHUNKS.map((c) => [c.name, /=\s*(\d+)u;/.exec(c.text)?.[1]] as const),
    );
    for (const name of VDB_GRID_TYPE_NAMES) {
      const id = VDB_GRID_TYPE_IDS[name];
      const upper = name.toUpperCase();
      expect(new RegExp(`#define\\s+PNANOVDB_GRID_TYPE_${upper}\\s+${id}\\b`).test(header), `PNanoVDB.h ${upper}`).toBe(true);
      expect(stride.gridTypes[upper], `stride-tables.json ${upper}`).toBe(id);
      expect((gridTypeIds as Record<string, number>)[upper], `tools ${upper}`).toBe(id);
      expect(wgslConst.get(`PNANOVDB_GRID_TYPE_${upper}`), `vendored WGSL ${upper}`).toBe(String(id));
    }
    expect(bytes.GRID_TYPE_FLOAT).toBe(VDB_GRID_TYPE_IDS.float);
    expect(bytes.GRID_TYPE_FP8).toBe(VDB_GRID_TYPE_IDS.fp8);
    expect(bytes.GRID_TYPE_FPN).toBe(VDB_GRID_TYPE_IDS.fpn);
  });

  it("is exported once from core/index.ts", () => {
    expect(core.VDB_GRID_TYPE_IDS).toBe(VDB_GRID_TYPE_IDS);
    expect(core.VDB_GRID_TYPE_NAMES).toBe(VDB_GRID_TYPE_NAMES);
    const name: VDBGridTypeName = "fp16";
    expect(core.VDB_GRID_TYPE_IDS[name]).toBe(15);
  });
});
