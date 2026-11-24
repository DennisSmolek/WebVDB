/**
 * K7.C: files written by native OpenVDB / NanoVDB v14.0.0 and NVIDIA Warp open here, with the
 * values native tools read from them.
 *
 * The fixtures are small and committed (`test/tools/fixtures/native-v14/`; README there names the
 * generators, versions and commands):
 *
 * - `sphere_fog_blosc.nvdb`, `two_grids_blosc.nvdb`: NanoVDB 32.11 `io::writeGrid(.., Codec::BLOSC)`
 *   (LZ4 frames from the Blosc OpenVDB v14 vendors). The decoded images are checked byte for byte
 *   against the native images' SHA-256, and values against a native accessor.
 * - `warp_sphere_blosclz.nvdb`: Warp 1.19 `Volume.save_to_nvdb(codec="blosc")` (python-blosc's
 *   BloscLZ, one frame per grid, `NanoVDB0` header, zero index bbox in FileMetaData).
 * - `half_{none,zip,blosc}.vdb`: OpenVDB v14 `HalfGrid`s (`Tree_half_5_4_3`, file version 225), a
 *   fog grid with a lower-node tile and a narrow-band level set (inactive values of both signs).
 */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  buildFromVdb,
  NvdbFile,
  NvdbStreamReader,
  parseVdb,
  vdbReadValueCpu,
  VDBToolsError,
  VDBUnsupportedError,
  type VdbFloatGrid,
} from "../../src/tools/internal.js";

const dir = new URL("./fixtures/native-v14/", import.meta.url);
const bytes = (name: string): Uint8Array => new Uint8Array(readFileSync(new URL(name, dir)));
const buffer = (name: string): ArrayBuffer => {
  const b = bytes(name);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
};
const json = <T>(name: string): T => JSON.parse(readFileSync(new URL(name, dir), "utf8")) as T;

interface Probe {
  ijk: [number, number, number];
  value: number;
  active?: boolean;
}

const nvdbProbes = json<Record<string, Probe[] | string>>("nvdb-probes.json");
const images = json<Record<string, { bytes: number; sha256: string; sha256HeaderNormalized: string }>>("nvdb-images.json");
const warp = json<{ imageBytes: number; imageSha256: string; probes: Probe[] }>("warp-probes.json");
const vdbProbes = json<Record<string, { background: number; activeVoxelCount: number; probes: Probe[] } | string | number>>(
  "vdb-probes.json",
);

function imageBytes(image: Uint32Array): Uint8Array {
  return new Uint8Array(image.buffer, image.byteOffset, image.byteLength);
}

function sha256(b: Uint8Array): string {
  return createHash("sha256").update(b).digest("hex");
}

/** SHA-256 with the header fields merging grids rewrites (checksum, version, flags, index, count) zeroed. */
function sha256Normalized(b: Uint8Array): string {
  const copy = b.slice();
  copy.fill(0, 8, 32);
  return sha256(copy);
}

function expectProbes(image: Uint32Array, probes: Probe[]): void {
  for (const p of probes) {
    const got = vdbReadValueCpu(image, p.ijk);
    expect(got.value, `value at ${p.ijk}`).toBeCloseTo(p.value, 6);
    if (p.active !== undefined) expect(got.active, `active at ${p.ijk}`).toBe(p.active);
  }
}

describe("NanoVDB v14 BLOSC (LZ4) .nvdb", () => {
  it("one grid: decodes to the native image byte for byte, and reads native's values", () => {
    const file = NvdbFile.fromArrayBuffer(buffer("sphere_fog_blosc.nvdb"));
    expect(file.grids).toHaveLength(1);
    const meta = file.grids[0]!;
    expect(meta).toMatchObject({ name: "density", gridType: "Float", gridClass: "FogVolume", voxelCount: 2109 });
    expect(meta.indexBBox).toEqual({ min: [8, 8, 8], max: [24, 24, 24] });
    const image = imageBytes(file.gridImage(0));
    expect(image.byteLength).toBe(images["sphere_fog_blosc.density"]!.bytes);
    expect(sha256(image)).toBe(images["sphere_fog_blosc.density"]!.sha256);
    expect(file.isGridImageZeroCopy(0)).toBe(false);
    expectProbes(file.gridImage(0), nvdbProbes["density"] as Probe[]);
  });

  it("two grids in one segment (Float + Fp8)", () => {
    const file = NvdbFile.fromArrayBuffer(buffer("two_grids_blosc.nvdb"));
    expect(file.grids.map((g) => [g.name, g.gridType])).toEqual([
      ["density", "Float"],
      ["density_fp8", "Fp8"],
    ]);
    expect(sha256Normalized(imageBytes(file.gridImage(0)))).toBe(images["sphere_fog_blosc.density"]!.sha256HeaderNormalized);
    expect(sha256Normalized(imageBytes(file.gridImage(1)))).toBe(images["two_grids_blosc.density_fp8"]!.sha256HeaderNormalized);
    expectProbes(file.gridImage(0), nvdbProbes["density"] as Probe[]);
    expectProbes(file.gridImage(1), nvdbProbes["density_fp8"] as Probe[]);
  });

  it("the stream reader emits the same images, fed one byte at a time and in one push", () => {
    const whole = bytes("two_grids_blosc.nvdb");
    const ref = NvdbFile.fromArrayBuffer(buffer("two_grids_blosc.nvdb"));
    for (const chunk of [1, 7, 4096, whole.byteLength]) {
      const reader = new NvdbStreamReader();
      const out = [];
      for (let at = 0; at < whole.byteLength; at += chunk) out.push(...reader.push(whole.subarray(at, at + chunk)));
      reader.end();
      expect(out.map((g) => g.index)).toEqual([0, 1]);
      for (const g of out) expect(sha256(imageBytes(g.image))).toBe(sha256(imageBytes(ref.gridImage(g.index))));
      expect(reader.bytesTotal).toBe(whole.byteLength);
    }
  });
});

describe("NVIDIA Warp BloscLZ .nvdb (NanoVDB 14 reads these)", () => {
  it("decodes to Warp's own image and values", () => {
    const file = NvdbFile.fromArrayBuffer(buffer("warp_sphere_blosclz.nvdb"));
    expect(file.grids).toHaveLength(1);
    const image = file.gridImage(0);
    expect(image.byteLength).toBe(warp.imageBytes);
    expect(sha256(imageBytes(image))).toBe(warp.imageSha256);
    expectProbes(image, warp.probes);
  });

  it("fills the index bbox and voxel count Warp leaves zero in FileMetaData from the image", () => {
    const meta = NvdbFile.fromArrayBuffer(buffer("warp_sphere_blosclz.nvdb")).grids[0]!;
    expect(meta.name).toBe("");
    expect(meta.indexBBox.max.some((c) => c !== 0)).toBe(true);
    expect(meta.voxelCount).toBeGreaterThan(0);
  });

  it("streams too", () => {
    const whole = bytes("warp_sphere_blosclz.nvdb");
    const reader = new NvdbStreamReader();
    const grids = [];
    for (let at = 0; at < whole.byteLength; at += 100) grids.push(...reader.push(whole.subarray(at, at + 100)));
    reader.end();
    expect(grids).toHaveLength(1);
    expect(sha256(imageBytes(grids[0]!.image))).toBe(warp.imageSha256);
  });
});

describe("OpenVDB v14 HalfGrid .vdb (Tree_half_5_4_3)", () => {
  for (const codec of ["none", "zip", "blosc"]) {
    it(`${codec}: parses as float grids with native's values, tiles and inactive values`, () => {
      const file = parseVdb(buffer(`half_${codec}.vdb`));
      expect(file.fileVersion).toBe(225);
      expect(file.grids.map((g) => [g.name, g.gridType, g.valueType])).toEqual([
        ["fog", "Tree_half_5_4_3", "float"],
        ["ls", "Tree_half_5_4_3", "float"],
      ]);
      for (const grid of file.grids as VdbFloatGrid[]) {
        const expected = vdbProbes[grid.name] as { background: number; activeVoxelCount: number; probes: Probe[] };
        expect(grid.background).toBe(expected.background);
        expect(grid.activeVoxelCount).toBe(BigInt(expected.activeVoxelCount));
        for (const p of expected.probes) {
          // Halves widen to f32 exactly: equal, not close.
          expect(grid.readValue(p.ijk), `${grid.name} at ${p.ijk}`).toEqual({ value: Math.fround(p.value), active: p.active });
        }
      }
    });
  }

  it("all three codecs decode to the same grids", () => {
    const values = (codec: string) =>
      (parseVdb(buffer(`half_${codec}.vdb`)).grids as VdbFloatGrid[]).map((g) => [...g.iterLeaves()].map((l) => [...l.values]));
    const none = values("none");
    expect(values("zip")).toEqual(none);
    expect(values("blosc")).toEqual(none);
  });

  it("file versions above 225 fail with a coded VDBUnsupportedError before the grids are read", () => {
    // OpenVDB 13.2+ warns for 226-299 and rejects 300+; we reject anything newer than we know (DIVERGES V70).
    for (const version of [226, 300, 0xffffffff]) {
      const b = bytes("half_none.vdb");
      new DataView(b.buffer, b.byteOffset).setUint32(8, version, true);
      let err: unknown;
      try {
        parseVdb(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer);
      } catch (e) {
        err = e;
      }
      expect(err).toBeInstanceOf(VDBUnsupportedError);
      expect((err as VDBToolsError).code).toBe("VDB_UNSUPPORTED");
      expect((err as Error).message).toMatch(new RegExp(`file version ${version} is newer than the maximum supported version 225`));
    }
  });

  it("buildFromVdb makes a Float NanoVDB grid that reads the same values", () => {
    const grid = parseVdb(buffer("half_blosc.vdb")).grids.find((g) => g.name === "ls") as VdbFloatGrid;
    const image = buildFromVdb(grid);
    const expected = vdbProbes["ls"] as { probes: Probe[] };
    for (const p of expected.probes) {
      expect(vdbReadValueCpu(image, p.ijk), `at ${p.ijk}`).toEqual({ value: Math.fround(p.value), active: p.active });
    }
  });
});
