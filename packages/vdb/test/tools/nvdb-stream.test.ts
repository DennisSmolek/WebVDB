/**
 * K4.16 / K4.18: `writeNvdb` in NanoVDB's own segment layout with codec ZIP,
 * `NvdbStreamReader` (push bytes, emit each grid when complete, hostile-input
 * checks before allocation) and `vdbBakeCoarseFirst` (`[<name>.lod48, <name>]`).
 */
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";
import { unzlibSync } from "fflate";
import { describe, expect, it } from "vitest";
import {
  vdbBakeCoarseFirst,
  vdbBuildFromDense,
  NvdbFile,
  NvdbFormatError,
  NvdbStreamReader,
  nvdbResponseLength,
  type NvdbProgress,
  nvdbLodName,
  parseNvdbLodName,
  parseVdb,
  readNvdbStream,
  vdbReadValueCpu,
  VDBToolsError,
  VDBUnsupportedError,
  writeNvdb,
  type NvdbStreamGrid,
} from "../../src/tools/index.js";


/** A soft ball of radius r in a dims^3 block: a FLOAT image named `name`. */
function ball(dims: number, name: string, voxelSize = 1): Uint32Array {
  const v = new Float32Array(dims ** 3);
  const c = (dims - 1) / 2;
  for (let x = 0; x < dims; x++)
    for (let y = 0; y < dims; y++)
      for (let z = 0; z < dims; z++) {
        const d = Math.hypot(x - c, y - c, z - c) / (dims / 2);
        v[(x * dims + y) * dims + z] = d < 1 ? Math.round((1 - d) * 64) / 64 : 0;
      }
  return vdbBuildFromDense(v, [dims, dims, dims], { gridName: name, voxelSize, origin: [-4, 3, 10] });
}

const u8 = (image: Uint32Array) => new Uint8Array(image.buffer, image.byteOffset, image.byteLength);
const same = (a: Uint32Array, b: Uint32Array) => Buffer.from(u8(a)).equals(Buffer.from(u8(b)));
const dv = (b: Uint8Array) => new DataView(b.buffer, b.byteOffset, b.byteLength);

function pushAll(bytes: Uint8Array, step: number, reader = new NvdbStreamReader()): NvdbStreamGrid[] {
  const out: NvdbStreamGrid[] = [];
  for (let off = 0; off < bytes.length; off += step) out.push(...reader.push(bytes.subarray(off, off + step)));
  reader.end();
  return out;
}

const A = ball(24, "density.lod8", 3);
const B = ball(40, "density", 1);

describe("writeNvdb: native segment layout and Codec.ZIP (K4.18)", () => {
  it("a single grid keeps the FileHeader, FileMetaData, name, image layout", () => {
    const file = new Uint8Array(writeNvdb([B]));
    const name = new TextEncoder().encode("density\0");
    expect(file.length).toBe(16 + 176 + name.length + B.byteLength);
    expect(Buffer.from(file.subarray(16 + 176, 16 + 176 + name.length)).equals(Buffer.from(name))).toBe(true);
    expect(Buffer.from(file.subarray(16 + 176 + name.length)).equals(Buffer.from(u8(B)))).toBe(true);
    expect(Number(dv(file).getBigUint64(16 + 8, true))).toBe(B.byteLength); // fileSize = gridSize for NONE
  });

  it("several grids: every FileMetaData + name first, then the grid data in the same order (io::Segment)", () => {
    const file = new Uint8Array(writeNvdb([A, B]));
    const n0 = "density.lod8\0".length;
    const n1 = "density\0".length;
    const meta1 = 16 + 176 + n0;
    expect(dv(file).getUint16(12, true)).toBe(2);
    expect(Number(dv(file).getBigUint64(meta1, true))).toBe(B.byteLength); // grid 1's FileMetaData.gridSize
    const data0 = meta1 + 176 + n1;
    expect(Buffer.from(file.subarray(data0, data0 + A.byteLength)).equals(Buffer.from(u8(A)))).toBe(true);
    expect(Buffer.from(file.subarray(data0 + A.byteLength)).equals(Buffer.from(u8(B)))).toBe(true);
    const parsed = NvdbFile.fromArrayBuffer(file.buffer);
    expect(parsed.grids.map((g) => g.name)).toEqual(["density.lod8", "density"]);
    expect(same(parsed.gridImage(0), A) && same(parsed.gridImage(1), B)).toBe(true);
  });

  it("codec zip: u64 size + zlib stream per grid; header, FileMetaData codec and fileSize agree; images round-trip", () => {
    const file = new Uint8Array(writeNvdb([A, B], { codec: "zip" }));
    const view = dv(file);
    expect(view.getUint16(14, true)).toBe(1); // FileHeader.codec = ZIP
    let at = 16;
    const fileSizes: number[] = [];
    for (const name of ["density.lod8", "density"]) {
      expect(view.getUint16(at + 168, true)).toBe(1); // FileMetaData.codec
      fileSizes.push(Number(view.getBigUint64(at + 8, true)));
      at += 176 + name.length + 1;
    }
    for (const [k, img] of [A, B].entries()) {
      const size = Number(view.getBigUint64(at, true));
      expect(fileSizes[k]).toBe(8 + size);
      expect(Buffer.from(unzlibSync(file.subarray(at + 8, at + 8 + size))).equals(Buffer.from(u8(img)))).toBe(true);
      at += 8 + size;
    }
    expect(at).toBe(file.length);
    const parsed = NvdbFile.fromArrayBuffer(file.buffer);
    expect(same(parsed.gridImage(0), A) && same(parsed.gridImage(1), B)).toBe(true);
    // The 270 KB upper node is nearly all zeros: ZIP makes the file much smaller.
    expect(file.length).toBeLessThan(writeNvdb([A, B]).byteLength / 5);
  });

  it("rejects an unknown codec and a bad zipLevel", () => {
    expect(() => writeNvdb([B], { codec: "blosc" as never })).toThrow(/unknown codec/);
    expect(() => writeNvdb([B], { codec: "zip", zipLevel: 10 })).toThrow(/zipLevel/);
  });

  it("NvdbFile reads a file of several segments (writeGrids writes one per handle)", () => {
    const one = new Uint8Array(writeNvdb([A], { codec: "zip" }));
    const two = new Uint8Array(writeNvdb([B]));
    const cat = new Uint8Array(one.length + two.length);
    cat.set(one);
    cat.set(two, one.length);
    const parsed = NvdbFile.fromArrayBuffer(cat.buffer);
    expect(parsed.grids.map((g) => g.name)).toEqual(["density.lod8", "density"]);
    expect(same(parsed.gridImage(1), B)).toBe(true);
    const junk = new Uint8Array(two.length + 20);
    junk.set(two);
    expect(() => NvdbFile.fromArrayBuffer(junk.buffer)).toThrow(/not another segment/);
  });
});

describe("NvdbStreamReader (K4.16)", () => {
  for (const codec of ["none", "zip"] as const) {
    it(`codec ${codec}: emits each grid as soon as its data is complete, at any chunk size`, () => {
      const file = new Uint8Array(writeNvdb([A, B], { codec }));
      const whole = NvdbFile.fromArrayBuffer(file.slice().buffer);
      for (const step of [1, 7, 176, 4096, file.length]) {
        const events = pushAll(file, step);
        expect(events.map((e) => e.metadata.name)).toEqual(["density.lod8", "density"]);
        expect(events.map((e) => e.lod)).toEqual([8, null]);
        expect(events.map((e) => e.baseName)).toEqual(["density", "density"]);
        expect(events.map((e) => e.index)).toEqual([0, 1]);
        expect(same(events[0]!.image, A) && same(events[1]!.image, B)).toBe(true);
        expect(events[0]!.metadata).toEqual(whole.grids[0]);
        expect(events[1]!.bytesReceived).toBe(file.length);
        expect(events[0]!.image.byteOffset % 4).toBe(0);
      }
      // The preview completes exactly at the end of its data: one byte fewer emits nothing.
      const reader = new NvdbStreamReader();
      const end0 = pushAll(file, 4096)[0]!.bytesReceived;
      expect(reader.push(file.subarray(0, end0 - 1))).toEqual([]);
      // Both FileMetaData are known before any grid data: the full grid's size is there for `capacity`.
      expect(reader.grids.map((g) => [g.name, g.gridByteSize])).toEqual([
        ["density.lod8", A.byteLength],
        ["density", B.byteLength],
      ]);
      const got = reader.push(file.subarray(end0 - 1, end0));
      expect(got.map((g) => g.metadata.name)).toEqual(["density.lod8"]);
      expect(got[0]!.bytesReceived).toBe(end0);
      expect(reader.atBoundary).toBe(false);
      expect(reader.push(file.subarray(end0)).length).toBe(1);
      expect(reader.atBoundary).toBe(true);
      reader.end();
    });
  }

  it("reads several segments and counts grids across them", () => {
    const one = new Uint8Array(writeNvdb([A], { codec: "zip" }));
    const two = new Uint8Array(writeNvdb([B]));
    const cat = new Uint8Array([...one, ...two]);
    const events = pushAll(cat, 333);
    expect(events.map((e) => [e.index, e.metadata.name])).toEqual([
      [0, "density.lod8"],
      [1, "density"],
    ]);
    expect(events[0]!.bytesReceived).toBe(one.length);
  });

  it("readNvdbStream reads a ReadableStream and cancels it when the consumer stops", async () => {
    const file = new Uint8Array(writeNvdb([A, B], { codec: "zip" }));
    let cancelled = false;
    let pulls = 0;
    const stream = () =>
      new ReadableStream<Uint8Array>({
        pull(c) {
          const off = pulls++ * 100;
          if (off >= file.length) c.close();
          else c.enqueue(file.slice(off, off + 100));
        },
        cancel() {
          cancelled = true;
        },
      });
    const names: string[] = [];
    for await (const g of readNvdbStream(stream())) names.push(g.metadata.name);
    expect(names).toEqual(["density.lod8", "density"]);

    pulls = 0;
    for await (const g of readNvdbStream(stream())) {
      expect(g.lod).toBe(8);
      break;
    }
    expect(cancelled).toBe(true);
    expect(pulls * 100).toBeLessThan(file.length);
  });

  describe("hostile and truncated input: a clean NvdbFormatError before any large allocation", () => {
    const file = () => new Uint8Array(writeNvdb([A, B], { codec: "zip" }));

    it("truncation is reported by end(), naming where the stream stopped", () => {
      for (const cut of [0, 10, 100, 16 + 176 + 3, 600, file().length - 1]) {
        const reader = new NvdbStreamReader();
        reader.push(file().subarray(0, cut));
        expect(() => reader.end()).toThrow(NvdbFormatError);
        expect(() => reader.end()).toThrow(/truncated/);
      }
    });

    it("bad magic, raw grid buffers, an unknown codec, zero grids, a bad version", () => {
      const bad = file();
      bad[0] = 0x58;
      expect(() => new NvdbStreamReader().push(bad)).toThrow(NvdbFormatError);
      expect(() => new NvdbStreamReader().push(u8(B))).toThrow(VDBUnsupportedError); // NanoVDB1 raw grid
      const codec = file();
      dv(codec).setUint16(14, 7, true);
      expect(() => new NvdbStreamReader().push(codec)).toThrow(NvdbFormatError);
      const zero = file();
      dv(zero).setUint16(12, 0, true);
      expect(() => new NvdbStreamReader().push(zero)).toThrow(/0 grids/);
      const version = file();
      dv(version).setUint32(8, 31 << 21, true);
      expect(() => new NvdbStreamReader().push(version)).toThrow(/incompatible NanoVDB version/);
      expect(() => new NvdbStreamReader().push(version)).toThrow(VDBUnsupportedError);
    });

    it("declared sizes are checked before allocation: gridSize (u64, max, minimum), nameSize, ZIP payload", () => {
      const huge = file();
      dv(huge).setBigUint64(16, 1n << 62n, true);
      expect(() => new NvdbStreamReader().push(huge)).toThrow(/unrepresentable gridSize/);

      const big = file();
      dv(big).setBigUint64(16 + 176 + "density.lod8\0".length, 3n << 30n, true);
      const r = new NvdbStreamReader();
      expect(() => r.push(big.subarray(0, 16 + 2 * 176 + 13 + 8))).toThrow(/over maxGridBytes/);
      expect(() => new NvdbStreamReader({ maxGridBytes: 4096 }).push(file())).toThrow(/maxGridBytes \(4096\)/);

      const tiny = file();
      dv(tiny).setBigUint64(16, 100n, true);
      expect(() => new NvdbStreamReader().push(tiny)).toThrow(/smaller than a 672-byte GridData/);

      const name = file();
      dv(name).setUint32(16 + 136, 1 << 20, true);
      expect(() => new NvdbStreamReader().push(name)).toThrow(/maxNameBytes/);

      const zip = file();
      const data0 = 16 + 2 * 176 + "density.lod8\0".length + "density\0".length;
      dv(zip).setBigUint64(data0, BigInt(A.byteLength * 2), true);
      expect(() => new NvdbStreamReader().push(zip)).toThrow(/ZIP payload/);
    });

    it("a zip bomb (payload inflating past gridSize) and a lying gridSize are rejected", () => {
      // A declared gridSize smaller than the real image: inflate is bounded and stops.
      const bomb = file();
      dv(bomb).setBigUint64(16, BigInt(A.byteLength - 64), true);
      expect(() => new NvdbStreamReader().push(bomb)).toThrow(/exceeds its declared gridSize/);

      // A NONE file whose FileMetaData.gridSize disagrees with the image's own mGridSize.
      const none = new Uint8Array(writeNvdb([B]));
      const lie = new Uint8Array(none.length + 8);
      lie.set(none);
      dv(lie).setBigUint64(16, BigInt(B.byteLength + 8), true);
      expect(() => new NvdbStreamReader().push(lie)).toThrow(/does not match its own internal GridData.mGridSize/);

      // A corrupt deflate stream.
      const corrupt = file();
      const data0 = 16 + 2 * 176 + "density.lod8\0".length + "density\0".length;
      corrupt.fill(0xff, data0 + 8 + 2, data0 + 8 + 40);
      expect(() => new NvdbStreamReader().push(corrupt)).toThrow(NvdbFormatError);
    });

    it("an unsupported grid type is VDBUnsupportedError, at its FileMetaData", () => {
      const vec = file();
      dv(vec).setUint32(16 + 32, 6, true); // Vec3f
      expect(() => new NvdbStreamReader().push(vec.subarray(0, 16 + 176 + 13))).toThrow(VDBUnsupportedError);
    });

    it("a failed reader rethrows on every later call", () => {
      const reader = new NvdbStreamReader();
      const bad = file();
      bad[1] = 0;
      expect(() => reader.push(bad)).toThrow(NvdbFormatError);
      expect(() => reader.push(new Uint8Array(1))).toThrow(NvdbFormatError);
      expect(() => reader.end()).toThrow(NvdbFormatError);
    });

    it("random corruption never escapes as anything but an NvdbFormatError / VDBUnsupportedError (both VDBToolsError)", () => {
      let seed = 12345;
      const rand = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32);
      const base = file();
      for (let trial = 0; trial < 200; trial++) {
        const f = base.slice();
        const at = Math.floor(rand() * Math.min(f.length, 1200));
        f[at] = Math.floor(rand() * 256);
        const reader = new NvdbStreamReader();
        try {
          pushAll(f, 1 + Math.floor(rand() * 3000), reader);
        } catch (err) {
          expect(err instanceof NvdbFormatError || err instanceof VDBUnsupportedError, String(err)).toBe(true);
          expect(err instanceof VDBToolsError, String(err)).toBe(true);
        }
      }
    });
  });

  it("LOD names", () => {
    expect(nvdbLodName("density", 48)).toBe("density.lod48");
    expect(parseNvdbLodName("density.lod48")).toEqual({ baseName: "density", lod: 48 });
    expect(parseNvdbLodName("a.lod4.lod16")).toEqual({ baseName: "a.lod4", lod: 16 });
    expect(parseNvdbLodName("density")).toEqual({ baseName: "density", lod: null });
    expect(parseNvdbLodName("density.lodx")).toEqual({ baseName: "density.lodx", lod: null });
    expect(() => nvdbLodName("d", 0)).toThrow();
  });
});

describe("vdbBakeCoarseFirst (K4.16)", () => {
  it("writes [<name>.lod<r>..., <name>] coarsest first; the previews cover the full grid's world box", () => {
    const full = ball(40, "density", 0.5);
    const bytes = vdbBakeCoarseFirst(full, { previewResolution: [16, 8], codec: "zip" });
    const events = pushAll(bytes, 1000);
    expect(events.map((e) => e.metadata.name)).toEqual(["density.lod8", "density.lod16", "density"]);
    expect(events.map((e) => e.metadata.gridType)).toEqual(["FpN", "FpN", "Float"]);
    expect(same(events[2]!.image, full)).toBe(true);
    // Cell boxes: NanoVDB's worldBBox runs from the centre of the min voxel to one voxel past the
    // centre of the max voxel (voxel i is [i - 1/2, i + 1/2] around its centre), so shift by -vs/2.
    const cells = (e: NvdbStreamGrid) => {
      const { worldBBox: w, voxelSize: vs } = e.metadata;
      return { min: w.min.map((v, a) => v - vs[a]! / 2), max: w.max.map((v, a) => v - vs[a]! / 2) };
    };
    const fb = cells(events[2]!);
    for (const e of events.slice(0, 2)) {
      const b = cells(e);
      for (let a = 0; a < 3; a++) {
        expect(b.min[a]).toBeCloseTo(fb.min[a]!, 9); // the box is kept at the min side ...
        expect(b.max[a]).toBeGreaterThanOrEqual(fb.max[a]! - 1e-9); // ... and grown to whole voxels at the max side,
        expect(b.max[a]! - fb.max[a]!).toBeLessThan(e.metadata.voxelSize[a]!); // by less than one preview voxel.
      }
    }
    expect(events[0]!.metadata.voxelSize[0]).toBeCloseTo((40 * 0.5) / 8, 9);
    // The preview reads the ball's centre value approximately (box filter, FpN).
    const centre = vdbReadValueCpu(events[1]!.image, [8, 8, 8]).value;
    expect(centre).toBeGreaterThan(0.6);
  });

  it("defaults: one preview at 48, box, FpN, codec zip; fullQuantize; option errors", () => {
    const full = ball(64, "smoke");
    const bytes = vdbBakeCoarseFirst(full);
    expect(dv(bytes).getUint16(14, true)).toBe(1);
    const events = pushAll(bytes, 1 << 16);
    expect(events.map((e) => [e.metadata.name, e.metadata.gridType])).toEqual([
      ["smoke.lod48", "FpN"],
      ["smoke", "Float"],
    ]);
    const fp8 = pushAll(vdbBakeCoarseFirst(full, { fullQuantize: "fp8", quantize: "none", codec: "none" }), 1 << 16);
    expect(fp8.map((e) => e.metadata.gridType)).toEqual(["Float", "Fp8"]);
    expect(() => vdbBakeCoarseFirst(full, { previewResolution: [] })).toThrow(/empty/);
    expect(() => vdbBakeCoarseFirst(full, { previewResolution: [8, 8] })).toThrow(/duplicates/);
    expect(() => vdbBakeCoarseFirst(full, { previewResolution: 0 })).toThrow(/positive/);
  });

  const SMOKE = fileURLToPath(new URL("../../../../fixtures/vdb-samples/smoke.vdb", import.meta.url));
  it.skipIf(!existsSync(SMOKE))("smoke: first paint is under his 14.8 KiB LOD0 with ZIP inside the file, and with gzip on the wire", () => {
    const buf = readFileSync(SMOKE);
    const vdb = parseVdb(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    const GATE = 14.8 * 1024;

    const zip = vdbBakeCoarseFirst(vdb);
    const z = pushAll(zip, 512);
    expect(z.map((e) => [e.metadata.name, e.metadata.gridType])).toEqual([
      ["density.lod48", "FpN"],
      ["density", "FpN"],
    ]);
    expect(z[0]!.bytesReceived).toBeLessThan(GATE); // raw transfer, no HTTP compression
    expect(zip.length).toBeLessThan(700 * 1024);

    const none = vdbBakeCoarseFirst(vdb, { codec: "none" });
    const n = pushAll(none, 1 << 16);
    expect(n[0]!.bytesReceived).toBeGreaterThan(300 * 1024); // the raw upper node
    // With gzip on the wire the preview needs about its gzip size (the design's 11.9 KiB + the second FileMetaData).
    expect(gzipSync(none.subarray(0, n[0]!.bytesReceived), { level: 9 }).length).toBeLessThan(GATE);
    for (const k of [0, 1]) expect(same(z[k]!.image, n[k]!.image)).toBe(true);
  });
});


describe("progress: bytesReceived and bytesTotal (K4.20)", () => {
  for (const codec of ["none", "zip"] as const) {
    it(`codec ${codec}: bytesTotal is the file size once the metadata is in; onProgress fires per push`, () => {
      const file = new Uint8Array(writeNvdb([A, B], { codec }));
      const seen: NvdbProgress[] = [];
      const reader = new NvdbStreamReader({ onProgress: (p) => seen.push(p) });
      expect(reader.bytesTotal).toBeNull();
      const grids = pushAll(file, 100, reader);
      const metaEnd = 16 + 2 * 176 + "density.lod8\0".length + "density\0".length;
      const known = seen.filter((p) => p.bytesTotal !== null);
      expect(seen.length).toBe(Math.ceil(file.length / 100));
      expect(seen[0]!.bytesTotal).toBeNull();
      expect(seen[0]!.bytesReceived).toBe(100);
      expect(seen.at(-1)).toEqual({ bytesReceived: file.length, bytesTotal: file.length, gridsEmitted: 2 });
      expect(known.every((p) => p.bytesTotal === file.length)).toBe(true);
      expect(known[0]!.bytesReceived).toBeGreaterThanOrEqual(metaEnd);
      expect(known[0]!.bytesReceived - 100).toBeLessThan(metaEnd);
      expect(grids.map((g) => g.bytesTotal)).toEqual([file.length, file.length]);
      expect(reader.progress.bytesTotal).toBe(file.length);
    });
  }

  it("several segments: the total is the end of the latest segment seen, the bytesTotal option wins", () => {
    const one = new Uint8Array(writeNvdb([B]));
    const two = new Uint8Array(one.length * 2);
    two.set(one, 0);
    two.set(one, one.length);
    const reader = new NvdbStreamReader();
    const totals: (number | null)[] = [];
    for (const g of pushAll(two, 4096, reader)) totals.push(g.bytesTotal);
    expect(totals).toEqual([one.length, two.length]);
    const declared = new NvdbStreamReader({ bytesTotal: 12345 });
    declared.push(one.subarray(0, 100));
    expect(declared.bytesTotal).toBe(12345);
    expect(() => new NvdbStreamReader({ bytesTotal: -1 })).toThrow(/bytesTotal/);
  });

  it("nvdbResponseLength: Content-Length unless the body is content-encoded", () => {
    const h = (o: Record<string, string>) => ({ headers: new Headers(o) });
    expect(nvdbResponseLength(h({ "content-length": "1234" }))).toBe(1234);
    expect(nvdbResponseLength(h({ "content-length": "1234", "content-encoding": "gzip" }))).toBeNull();
    expect(nvdbResponseLength(h({ "content-length": "1234", "content-encoding": "identity" }))).toBe(1234);
    expect(nvdbResponseLength(h({}))).toBeNull();
    expect(nvdbResponseLength(h({ "content-length": "abc" }))).toBeNull();
  });

  it("readNvdbStream takes a Response and reports its Content-Length", async () => {
    const file = new Uint8Array(writeNvdb([A, B], { codec: "zip" }));
    const seen: NvdbProgress[] = [];
    const res = new Response(file as BlobPart, { headers: { "content-length": String(file.length) } });
    const grids: NvdbStreamGrid[] = [];
    for await (const g of readNvdbStream(res, { onProgress: (p) => seen.push(p) })) grids.push(g);
    expect(grids.length).toBe(2);
    expect(seen[0]!.bytesTotal).toBe(file.length); // known from the first chunk
    expect(grids[0]!.bytesTotal).toBe(file.length);
  });
});
