import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  bakeWvdb,
  vdbBuildFromDense,
  buildFromVdb,
  nvdbToWvdb,
  parseVdb,
  readWvdb,
  sampleWvdbDense,
  vdbToWvdb,
  wvdbChunks,
  wvdbDensityStats,
  wvdbFinestDimensions,
  wvdbSourceFromNvdb,
  type WvdbSampleSource,
  type WvdbVolumeGrid,
} from "../../src/tools/internal.js";
import { WVDB_FIXTURES, bytesEqual, headerWithoutOffsets, readBytes, vdbSample } from "./wvdb-helpers.js";

const identity = (b: Uint8Array) => b;

describe("bake geometry (§3.2.1)", () => {
  it("finest dims: `resolution` on the longest side, the rest rounded, cube when the box is empty", () => {
    // his Appendix A box
    expect(wvdbFinestDimensions([[-111.5, -114.5, -159.5], [109.5, 141.5, 122.5]], 128)).toEqual([100, 116, 128]);
    expect(wvdbFinestDimensions([[0, 0, 0], [10, 0.01, 1]], 64)).toEqual([64, 1, 6]);
    expect(wvdbFinestDimensions(null, 32)).toEqual([32, 32, 32]);
  });

  it("point-samples voxel centres with Math.round in index space; level sets follow §8", () => {
    const seen: number[][] = [];
    const src: WvdbSampleSource = {
      name: "ls",
      class: "level set",
      background: 1,
      bbox: [[0, 0, 0], [3, 1, 1]],
      sample(w) {
        seen.push([...w]);
        // inside the band at x < 1.5 (active); outside elsewhere (inactive), negative at x > 2
        if (w[0] < 1.5) return { value: w[0] < 0.6 ? 5 : -0.999, active: true };
        return { value: w[0] > 2 ? -0.4 : 0.4, active: false };
      },
    };
    const v = sampleWvdbDense(src, src.bbox, [3, 1, 1], 8);
    expect(seen.map((p) => p[0])).toEqual([0.5, 1.5, 2.5]);
    const a = 1 - 2 / 255;
    expect(v[0]).toBe(Math.fround(a)); // active, clamped to +a
    expect(v[1]).toBe(1); // inactive, positive -> +b
    expect(v[2]).toBe(-1); // inactive, negative -> -b
  });
});

describe("NanoVDB grid → .wvdb through the CPU oracle", () => {
  it("matches a direct dense source on a dense-built grid, and round-trips", () => {
    const dims: [number, number, number] = [20, 13, 9];
    const values = new Float32Array(dims[0] * dims[1] * dims[2]);
    // vdbBuildFromDense indexes x-major (x slowest); fill by coordinates
    const at = (x: number, y: number, z: number) => (x * dims[1] + y) * dims[2] + z;
    const f = (x: number, y: number, z: number) => (Math.hypot(x - 10, y - 6, z - 4) < 5 ? Math.fround(0.1 + (x + y + z) / 50) : 0);
    for (let x = 0; x < dims[0]; x++) for (let y = 0; y < dims[1]; y++) for (let z = 0; z < dims[2]; z++) values[at(x, y, z)] = f(x, y, z);
    const image = vdbBuildFromDense(values, dims, { voxelSize: 0.5, worldOrigin: [1, 2, 3], gridName: "density" });
    const src = wvdbSourceFromNvdb(image);
    expect(src.name).toBe("density");
    expect(src.class).toBe("fog volume");
    // active box -> world box (voxel centres): index (6..14, 2..10, 0..8) * 0.5 + origin
    expect(src.bbox).toEqual([[4, 3, 3], [8, 7, 7]]);
    const direct: WvdbSampleSource = {
      ...src,
      sample(w) {
        const ijk = [0, 1, 2].map((a) => Math.round((w[a]! - [1, 2, 3][a]!) * 2));
        const inside = ijk.every((c, a) => c >= 0 && c < dims[a]!);
        const value = inside ? f(ijk[0]!, ijk[1]!, ijk[2]!) : 0;
        return { value, active: value !== 0 };
      },
    };
    const a = nvdbToWvdb(image, { resolution: 24, compress: identity });
    const b = bakeWvdb([direct], { resolution: 24, compress: identity });
    expect(bytesEqual(a.bytes, b.bytes)).toBe(true);
    const file = readWvdb(nvdbToWvdb(image, { resolution: 24 }).bytes);
    const g = file.grids[0] as WvdbVolumeGrid;
    expect(g.levels[2]!.dimensions).toEqual(wvdbFinestDimensions(src.bbox, 24));
    expect(g.nonZeroVoxels).toBeGreaterThan(0);
  });
});

// smoke.vdb: our pinned openvdb.org sample (pnpm fixtures) or his checkout (MJURCZYK_OPENVDB_DIR).
const smokeVdb = vdbSample("smoke.vdb");

describe.skipIf(!smokeVdb)("smoke.vdb → .wvdb reproduces his committed smoke.wvdb", () => {
  const digest = JSON.parse(readFileSync(path.join(WVDB_FIXTURES, "smoke.wvdb.digest.json"), "utf8"));

  it("same source digest, box, dims, brick sets and order, and every inflated chunk (SHA-256)", () => {
    const bytes = readBytes(smokeVdb!);
    // His smoke.wvdb predates his cube-texel fix: it is resolution^3, so pass the dims.
    const ours = vdbToWvdb(bytes, { fileName: "smoke.vdb", dimensions: [128, 128, 128], compress: identity });
    const expected = digest.headerWithoutOffsets;
    const got = headerWithoutOffsets(ours.header) as typeof expected;
    // His header stats were recomputed later from his decoded data; ours come from the source values.
    const { meanDensity: _m, nonZeroVoxels: _n, ...ourGrid } = got.grids[0];
    const { meanDensity, nonZeroVoxels, ...hisGrid } = expected.grids[0];
    expect({ ...got, grids: [ourGrid] }).toEqual({ ...expected, grids: [hisGrid] });

    const dataStart = 12 + new DataView(ours.bytes.buffer).getUint32(8, true);
    const chunks = wvdbChunks(ours.header);
    expect(chunks.length).toBe(digest.chunks.length);
    chunks.forEach((c, i) => {
      const raw = ours.bytes.subarray(dataStart + c.offset, dataStart + c.offset + c.length);
      expect(raw.length).toBe(digest.chunks[i].inflatedBytes);
      expect(createHash("sha256").update(raw).digest("hex")).toBe(digest.chunks[i].inflatedSha256);
    });

    // ... and his stats are the stats of the decoded finest LOD
    const decoded = readWvdb(vdbToWvdb(bytes, { fileName: "smoke.vdb", dimensions: [128, 128, 128] }).bytes);
    const s = wvdbDensityStats((decoded.grids[0] as WvdbVolumeGrid).levels[2]!.values, 0, false);
    expect(s).toEqual({ meanDensity, nonZeroVoxels });
  });

  it("the NanoVDB path (buildFromVdb → CPU oracle) gives the same bytes as the .vdb path", () => {
    const bytes = readBytes(smokeVdb!);
    const file = parseVdb(bytes.slice().buffer);
    const a = vdbToWvdb(file, { compress: identity });
    const b = nvdbToWvdb(buildFromVdb(file.grids[0]!), { compress: identity });
    expect(bytesEqual(a.bytes, b.bytes)).toBe(true);
    // default (current) dims are cube texels: 128 on the longest side
    expect(a.header.grids[0]!.lods.at(-1)).toMatchObject({ dimensions: wvdbFinestDimensions(a.header.grids[0]!.bbox, 128) });
  });
});
