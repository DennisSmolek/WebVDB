import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  NANOVDB_FLAG_HAS_MIN_MAX,
  NANOVDB_OFFSETS as O,
  readImageHeader,
} from "../../src/core/pool/image-header.js";
import { VDB_GRID_TYPE_IDS, type VDBGridTypeName } from "../../src/core/types.js";
// Tests may import tools (the core may not, D20): this is the cross-check.
import * as bytes from "../../src/tools/nanovdb/bytes.js";
import { NvdbFile } from "../../src/tools/index.js";
import { makeImage } from "./synthetic-image.js";

const strideTables = JSON.parse(readFileSync(new URL("../../vendor/stride-tables.json", import.meta.url), "utf8"));
const defines: Record<string, number> = strideTables.defines;

describe("image-header offsets match tools and PNanoVDB.h", () => {
  it("GridData / TreeData / RootData offsets", () => {
    expect(O.gridSize).toBe(bytes.GRID_SIZE);
    expect(O.gridMagic).toBe(bytes.GRID_OFF_MAGIC);
    expect(O.gridVersion).toBe(bytes.GRID_OFF_VERSION);
    expect(O.gridFlags).toBe(bytes.GRID_OFF_FLAGS);
    expect(O.gridByteSize).toBe(bytes.GRID_OFF_GRID_SIZE);
    expect(O.gridName).toBe(bytes.GRID_OFF_GRID_NAME);
    expect(O.gridNameMax).toBe(bytes.GRID_NAME_MAX);
    expect(O.gridMap).toBe(bytes.GRID_OFF_MAP);
    expect(O.mapMatD).toBe(bytes.MAP_OFF_MATD);
    expect(O.mapInvMatD).toBe(bytes.MAP_OFF_INVMATD);
    expect(O.mapVecD).toBe(bytes.MAP_OFF_VECD);
    expect(O.gridWorldBBox).toBe(bytes.GRID_OFF_WORLD_BBOX);
    expect(O.gridVoxelSize).toBe(bytes.GRID_OFF_VOXEL_SIZE);
    expect(O.gridClass).toBe(bytes.GRID_OFF_GRID_CLASS);
    expect(O.gridType).toBe(bytes.GRID_OFF_GRID_TYPE);
    expect(O.treeSize).toBe(bytes.TREE_SIZE);
    expect(O.treeNodeOffsetRoot).toBe(bytes.TREE_OFF_NODE_OFFSET_ROOT);
    expect(O.rootBBoxMin).toBe(bytes.ROOT_OFF_BBOX_MIN);
    expect(O.rootBBoxMax).toBe(bytes.ROOT_OFF_BBOX_MAX);
    expect(O.rootBaseSize).toBe(defines.PNANOVDB_ROOT_BASE_SIZE);
    expect(O.gridSize).toBe(defines.PNANOVDB_GRID_SIZE);
    expect(O.gridMap).toBe(defines.PNANOVDB_GRID_OFF_MAP);
    expect(O.mapMatD).toBe(defines.PNANOVDB_MAP_OFF_MATD);
    expect(O.mapInvMatD).toBe(defines.PNANOVDB_MAP_OFF_INVMATD);
    expect(O.mapVecD).toBe(defines.PNANOVDB_MAP_OFF_VECD);
    expect(O.treeNodeOffsetRoot).toBe(defines.PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT);
    expect(NANOVDB_FLAG_HAS_MIN_MAX & bytes.GRID_FLAGS_STATS_BBOX_BREADTH_FIRST).toBe(NANOVDB_FLAG_HAS_MIN_MAX);
  });

  it("typed root fields are f32 at 28/32/36 (Half: binary16 at 28/30/32) for every supported type", () => {
    for (const name of Object.keys(VDB_GRID_TYPE_IDS)) {
      const c = strideTables.gridTypeConstants[name.toUpperCase()];
      expect(c, name).toBeTruthy();
      const half = name === "half";
      expect(c.root_off_background, name).toBe(half ? O.rootHalfBackground : O.rootBackground);
      expect(c.root_off_min, name).toBe(half ? O.rootHalfMin : O.rootMin);
      expect(c.root_off_max, name).toBe(half ? O.rootHalfMax : O.rootMax);
      expect(c.root_size, name).toBe(O.rootSize);
      expect(strideTables.gridTypes[name.toUpperCase()], name).toBe(VDB_GRID_TYPE_IDS[name as VDBGridTypeName]);
    }
  });
});

describe("readImageHeader validation", () => {
  it("reads a synthetic image", () => {
    const h = readImageHeader(makeImage({ bytes: 2048, gridType: 14, background: 0.5, rootMin: -0.25, rootMax: 2, name: "density" }));
    expect(h).toMatchObject({ gridType: "fp8", gridTypeId: 14, gridClass: 2, name: "density", byteLength: 2048 });
    expect(h.hasStats).toBe(true);
    expect(h.background).toBe(0.5);
    expect(h.rootMin).toBe(-0.25);
    expect(h.rootMax).toBe(2);
    expect(h.indexBBoxMin).toEqual([-4, -4, -4]);
  });

  it("reads a Half image's binary16 background and root min/max (K7.H)", () => {
    const img = makeImage({ bytes: 2048, gridType: 9 });
    const v = new DataView(img.buffer);
    const root = 736;
    v.setUint32(root + 28, 0, true);
    v.setUint32(root + 32, 0, true);
    v.setUint16(root + 28, 0x3800, true); // background 0.5
    v.setUint16(root + 30, 0xb400, true); // min -0.25
    v.setUint16(root + 32, 0x7bff, true); // max 65504
    const h = readImageHeader(img);
    expect(h.gridType).toBe("half");
    expect(h.background).toBe(0.5);
    expect(h.rootMin).toBe(-0.25);
    expect(h.rootMax).toBe(65504);
  });

  const code = (fn: () => unknown) => {
    try {
      fn();
    } catch (e) {
      return (e as { code?: string }).code;
    }
    return "no error";
  };

  it("rejects bad images with actionable codes", () => {
    const img = () => makeImage({ bytes: 1024 });
    const badMagic = img();
    badMagic[1] = 0x32424456; // "VDB2": a file header, not a grid
    expect(code(() => readImageHeader(badMagic))).toBe("VDB_BAD_IMAGE");
    const badVersion = img();
    new DataView(badVersion.buffer).setUint32(16, 31 << 21, true);
    expect(code(() => readImageHeader(badVersion))).toBe("VDB_BAD_IMAGE");
    const badSize = img();
    badSize[8] = 4096; // mGridSize lo
    expect(code(() => readImageHeader(badSize))).toBe("VDB_BAD_IMAGE");
    const hugeSize = img();
    hugeSize[9] = 1; // mGridSize hi: ≥ 4 GiB
    expect(code(() => readImageHeader(hugeSize))).toBe("VDB_BAD_IMAGE");
    expect(code(() => readImageHeader(new Uint32Array(100)))).toBe("VDB_BAD_IMAGE");
    expect(code(() => readImageHeader(makeImage({ gridType: 2 })))).toBe("VDB_UNSUPPORTED_GRID_TYPE");
    expect(code(() => readImageHeader(makeImage({ matD: [1, 0, 0, 0, 0, 0, 0, 0, 1] })))).toBe("VDB_BAD_IMAGE");
    expect(code(() => readImageHeader(makeImage({ gridClass: 300 })))).toBe("VDB_BAD_IMAGE");
    expect(code(() => readImageHeader(img()))).toBe("no error");
  });
});

const dir = new URL("../../../../fixtures/primitives/", import.meta.url);
const NAMES = ["box_fog", "sphere_fog", "torus_fog"].flatMap((p) => ["float", "fp8", "fpn"].map((t) => `${p}_${t}`));
const present = NAMES.every((n) => existsSync(new URL(`${n}.nvdb`, dir)));

describe.skipIf(!present)("readImageHeader on the nine primitive fixtures", () => {
  it.each(NAMES)("%s agrees with NvdbFile and the sidecar", (name) => {
    const buf = readFileSync(new URL(`${name}.nvdb`, dir));
    const file = NvdbFile.fromArrayBuffer(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    const meta = file.grids[0]!;
    const sidecar = JSON.parse(readFileSync(new URL(`${name}.sidecar.json`, dir), "utf8")).grid;
    const h = readImageHeader(file.gridImage(0));
    expect(h.gridType).toBe(sidecar.type);
    expect(h.name).toBe(meta.name);
    expect(h.byteLength).toBe(meta.gridByteSize);
    expect([h.indexBBoxMin, h.indexBBoxMax]).toEqual([meta.indexBBox.min, meta.indexBBox.max]);
    expect(h.voxelSize).toEqual(meta.voxelSize);
    expect(h.gridClass).toBe(2); // FogVolume
    expect(h.hasStats).toBe(true);
    // The map's index -> world sends the index bbox to the file's world bbox (max + 1 voxel; CLAUDE.md Traps).
    for (let a = 0; a < 3; a++) {
      const w = (i: number) => h.matD[a * 4]! * i + h.vecD[a]!; // diagonal maps in these fixtures
      expect(w(h.indexBBoxMin[a]!)).toBeCloseTo(meta.worldBBox.min[a]!, 9);
    }
    expect(h.rootMax).toBeGreaterThan(0);
    expect(h.rootMin).toBeLessThanOrEqual(h.rootMax);
  });
});
