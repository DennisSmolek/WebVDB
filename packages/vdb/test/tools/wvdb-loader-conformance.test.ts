import { describe, expect, it } from "vitest";
import { readWvdb, writeWvdb, type WvdbGridInput, type WvdbVec3, type WvdbVolumeGrid, type WriteWvdbOptions } from "../../src/tools/index.js";
import { bytesEqual, fixture } from "./wvdb-helpers.js";

/*
 * K4.3 gate "a .wvdb we write loads in his WVDBLoader" — STATIC conformance.
 *
 * We may not run his code, so this test re-implements, field by field, the
 * parse path of his loader at mjurczyk/openvdb 75e7345 (read, not executed):
 *   src/three-openvdb/wvdb/WVDBLoader.js   load → WVDBReader.load, readGrids, attach
 *   src/three-openvdb/wvdb/WVDBReader.js   receive/readWVDBHeader, prepare, getKernelGrid,
 *                                          getBackgroundTexel, decode (offset order), apply
 *   src/three-openvdb/wvdb/wvdbKernel.js   inflate (DecompressionStream('deflate')), chunk,
 *                                          writeBricks, clearMissingBricks, toTexels
 *   src/openvdb/wvdb/format.js             readWVDBHeader, getWVDBChunks, decodeWVDBPage,
 *                                          decodeWVDBMask, decodeWVDBPoints
 * and asserts every field, offset and flag it reads is present, typed and
 * consistent in files our writer produces. The decode below follows his
 * functions (his byte reads, his loops), not our reader, and its float output
 * is compared with readWvdb. A real load in his WVDBLoader on the owner's
 * machine is still needed (docs/handoffs/K4-3.md).
 */

interface HisChunk {
  gridIndex: number;
  lod: number;
  kind: "mask" | "page" | "points";
  offset: number;
  length: number;
  firstIndex?: number;
  firstPoint?: number;
}

async function inflateLikeHisKernel(bytes: Uint8Array): Promise<Uint8Array> {
  // wvdbKernel.inflate: new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'))
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Mirrors his loader's parse path over a whole file; returns per grid per LOD the decoded floats. */
async function hisLoaderMirror(bytes: Uint8Array) {
  // format.js readWVDBHeader
  expect(new TextDecoder().decode(bytes.subarray(0, 4))).toBe("WVDB");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint32(4, true);
  expect(version).toBeLessThanOrEqual(2); // wvdbVersion = 2
  const dataStart = 12 + view.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(12, dataStart)));

  // WVDBReader.prepare: bake.lods, bake.bits, grids[]
  expect(Number.isInteger(header.bake.lods)).toBe(true);
  expect([8, 16]).toContain(header.bake.bits);
  const finalLod = header.bake.lods - 1;
  expect(Array.isArray(header.grids)).toBe(true);

  // format.js getWVDBChunks (then sorted by offset)
  const chunks: HisChunk[] = [];
  header.grids.forEach((grid: any, gridIndex: number) => {
    expect(grid.lods.length).toBe(header.bake.lods);
    let firstPoint = 0;
    grid.lods.forEach(({ mask, pages }: any, lod: number) => {
      if (grid.type === "points") {
        pages.forEach(({ offset, length, points }: any) => {
          expect(Number.isInteger(points) && points > 0).toBe(true);
          chunks.push({ gridIndex, lod, kind: "points", offset, length, firstPoint });
          firstPoint += points;
        });
        return;
      }
      if (mask) chunks.push({ gridIndex, lod, kind: "mask", offset: mask.offset, length: mask.length });
      let firstIndex = 0;
      pages.forEach(({ offset, length, bricks }: any) => {
        expect(Number.isInteger(bricks) && bricks > 0).toBe(true);
        chunks.push({ gridIndex, lod, kind: "page", offset, length, firstIndex });
        firstIndex += bricks;
      });
    });
  });
  chunks.sort((a, b) => a.offset - b.offset);
  for (const c of chunks) expect(Number.isInteger(c.offset) && Number.isInteger(c.length)).toBe(true);
  // WVDBReader.decode walks chunks in offset order and waits on bytes in that order: LODs must not interleave.
  for (let i = 1; i < chunks.length; i++) expect(chunks[i]!.lod).toBeGreaterThanOrEqual(chunks[i - 1]!.lod);
  // bytesTotal: the reader stops the download here
  const bytesTotal = dataStart + chunks.reduce((end, { offset, length }) => Math.max(end, offset + length), 0);
  expect(bytesTotal).toBe(bytes.length);

  // WVDBReader.prepare / getKernelGrid / getBackgroundTexel / WVDBLoader.readGrids field reads
  header.grids.forEach((grid: any) => {
    expect(typeof grid.name).toBe("string");
    expect(typeof grid.class).toBe("string");
    expect(grid.bbox === null || (Array.isArray(grid.bbox) && grid.bbox.length === 2)).toBe(true);
    if (grid.type === "points") {
      expect(typeof grid.colors).toBe("boolean");
      expect(Number.isInteger(grid.count)).toBe(true);
      // WVDBLoader createPointsGridData.spacing: grid.spacing * cbrt(totalCount / count)
      expect(Number.isFinite(grid.spacing)).toBe(true);
      grid.lods.forEach((l: any) => expect(Number.isInteger(l.points)).toBe(true));
      return;
    }
    const channels = grid.channels || 1;
    expect([1, 3]).toContain(channels);
    // getBackgroundTexel: channels 3 -> background.map(toByte); else toByte(background / scale)
    if (channels === 3) expect(Array.isArray(grid.background) && grid.background.length === 3).toBe(true);
    else expect(typeof grid.background).toBe("number");
    // densityStats: typeof meanDensity === 'number' -> sum = meanDensity * nonZeroVoxels
    if (typeof grid.meanDensity === "number") expect(Number.isInteger(grid.nonZeroVoxels)).toBe(true);
    grid.lods.forEach((lod: any) => {
      expect(lod.dimensions.length).toBe(3);
      // A LOD gives its first texture after its last page, or after its mask when it has no pages
      // (wvdbKernel.chunk). With neither, the grid would never reach lod 0 and onLoad never fires.
      expect(!!lod.mask || lod.pages.length > 0).toBe(true);
    });
  });

  // wvdbKernel.chunk per chunk, in offset order
  const levels: Float32Array[][] = header.grids.map(() => []);
  const masks: (Uint32Array | null)[][] = header.grids.map(() => []);
  for (const chunk of chunks) {
    const grid = header.grids[chunk.gridIndex];
    const data = await inflateLikeHisKernel(bytes.slice(dataStart + chunk.offset, dataStart + chunk.offset + chunk.length));
    if (chunk.kind === "points") {
      // decodeWVDBPoints
      const v = new DataView(data.buffer, data.byteOffset, data.byteLength);
      const count = v.getUint32(0, true);
      expect(data.byteLength).toBe(28 + 6 * count + (grid.colors ? 3 * count : 0));
      const positions = new Float32Array(count * 3);
      for (let axis = 0; axis < 3; axis++) {
        const min = v.getFloat32(4 + axis * 4, true);
        const max = v.getFloat32(16 + axis * 4, true);
        const step = (max - min) / 0xffff;
        const high = 28 + axis * 2 * count;
        for (let p = 0; p < count; p++) {
          const code = (data[high + p]! << 8) | data[high + count + p]!;
          positions[p * 3 + axis] = code === 0xffff ? max : min + code * step;
        }
      }
      const level = (levels[chunk.gridIndex]![chunk.lod] ??= new Float32Array(grid.lods[chunk.lod].points * 3));
      const lodStart = grid.lods.slice(0, chunk.lod).reduce((s: number, l: any) => s + l.points, 0);
      level.set(positions, (chunk.firstPoint! - lodStart) * 3);
      continue;
    }
    const lod = grid.lods[chunk.lod];
    const [width, height, depth] = lod.dimensions;
    const channels = grid.channels || 1;
    const total = Math.ceil(width / 8) * Math.ceil(height / 8) * Math.ceil(depth / 8);
    const plane = width * height * depth;
    const background: number[] = channels === 3 ? grid.background : [grid.background];
    const level = (levels[chunk.gridIndex]![chunk.lod] ??= (() => {
      const l = new Float32Array(plane * channels);
      background.forEach((b, c) => l.fill(b, c * plane, (c + 1) * plane));
      return l;
    })());
    if (chunk.kind === "mask") {
      // decodeWVDBMask: words = byteLength >>> 2; clearMissingBricks reads words[index >>> 5] for index < total
      expect(data.byteLength % 4).toBe(0);
      expect(data.byteLength >>> 2).toBeGreaterThanOrEqual(Math.ceil(total / 32));
      const v = new DataView(data.buffer, data.byteOffset, data.byteLength);
      masks[chunk.gridIndex]![chunk.lod] = Uint32Array.from({ length: data.byteLength >>> 2 }, (_, i) => v.getUint32(i * 4, true));
      continue;
    }
    // decodeWVDBPage, layout from `mask ? sparse : dense` (getKernelGrid)
    const sparse = !!lod.mask;
    const v = new DataView(data.buffer, data.byteOffset, data.byteLength);
    const count = v.getUint32(0, true);
    const pageRef = lod.pages.find((p: any) => p.offset === chunk.offset);
    expect(count).toBe(pageRef.bricks);
    let offset = 4;
    const indices: number[] = [];
    for (let i = 0; i < count; i++) {
      if (sparse) {
        indices.push(v.getUint32(offset, true));
        offset += 4;
      } else indices.push(chunk.firstIndex! + i);
    }
    for (const index of indices) expect(index).toBeLessThan(total);
    const ranges = count * channels;
    const mins = Array.from({ length: ranges }, (_, i) => v.getFloat32(offset + i * 4, true));
    offset += 4 * ranges;
    const maxs = Array.from({ length: ranges }, (_, i) => v.getFloat32(offset + i * 4, true));
    offset += 4 * ranges;
    const maxCode = (1 << header.bake.bits) - 1;
    const values = new Float32Array(ranges * 512);
    for (let i = 0; i < ranges; i++) {
      if (mins[i] === maxs[i]) {
        values.fill(mins[i]!, i * 512, i * 512 + 512);
        continue;
      }
      const step = (maxs[i]! - mins[i]!) / maxCode;
      for (let voxel = 0; voxel < 512; voxel++) {
        let code: number;
        if (header.bake.bits === 16) {
          code = v.getUint16(offset, true);
          offset += 2;
        } else code = data[offset++]!;
        values[i * 512 + voxel] = code === maxCode ? maxs[i]! : mins[i]! + code * step;
      }
    }
    // his decoder does not check the size; ours must leave no trailing bytes it would silently ignore
    expect(offset).toBe(data.byteLength);
    // wvdbKernel.writeBricks (before toTexels), per channel
    const countX = Math.ceil(width / 8);
    const countY = Math.ceil(height / 8);
    indices.forEach((index, b) => {
      const ox = (index % countX) * 8;
      const oy = (Math.floor(index / countX) % countY) * 8;
      const oz = Math.floor(index / (countX * countY)) * 8;
      for (let c = 0; c < channels; c++) {
        let voxel = (b * channels + c) * 512;
        for (let z = 0; z < 8; z++)
          for (let y = 0; y < 8; y++) {
            const inside = oy + y < height && oz + z < depth;
            for (let x = 0; x < 8; x++, voxel++) {
              if (inside && ox + x < width) level[c * plane + ox + x + (oy + y) * width + (oz + z) * width * height] = values[voxel]!;
            }
          }
      }
      if (sparse) {
        const words = masks[chunk.gridIndex]![chunk.lod]!;
        expect(words).toBeDefined(); // the mask chunk comes before the pages (offset order)
        expect((words[index >>> 5]! & (1 << (index & 31))) !== 0).toBe(true);
      }
    });
  }
  return { header, levels };
}

const cases: { name: string; grids: WvdbGridInput[]; opts: WriteWvdbOptions }[] = (() => {
  const dims: WvdbVec3 = [27, 19, 33];
  const n = dims[0] * dims[1] * dims[2];
  const blob = Float32Array.from({ length: n }, (_, i) => {
    const x = i % dims[0];
    const y = Math.floor(i / dims[0]) % dims[1];
    const z = Math.floor(i / (dims[0] * dims[1]));
    const d = Math.hypot(x - 13, y - 9, z - 16) / 10;
    return d < 1 ? 1 - d : 0;
  });
  const rgb = Float32Array.from({ length: n * 3 }, (_, i) => ((i * 13) % 17) / 16);
  const box: [WvdbVec3, WvdbVec3] = [[-1, -1, -1], [1.7, 0.9, 2.3]];
  return [
    { name: "sparse 8-bit, several pages", grids: [{ name: "density", class: "fog volume", background: 0, bbox: box, finest: { dimensions: dims, values: blob } }], opts: { pageBricks: 4 } },
    { name: "dense 16-bit", grids: [{ name: "density", background: 0, bbox: box, finest: { dimensions: dims, values: blob } }], opts: { layout: "dense", bits: 16 } },
    {
      name: "vector + scalar + empty grid",
      grids: [
        { name: "density", background: 0, bbox: box, finest: { dimensions: dims, values: blob } },
        { name: "Cd", channels: 3, background: [1, 1, 1], bbox: box, finest: { dimensions: dims, values: rgb } },
        { name: "flames", background: 0, bbox: box, finest: { dimensions: dims, values: new Float32Array(n) } },
      ],
      opts: {},
    },
    {
      name: "points (colours) + volume, version 2",
      grids: [
        { type: "points", name: "points", positions: Float32Array.from({ length: 3 * 5000 }, (_, i) => Math.cos(i * 0.37) * 3), colors: Float32Array.from({ length: 3 * 5000 }, (_, i) => (i % 5) / 4) },
        { name: "density", background: 0, bbox: box, finest: { dimensions: dims, values: blob } },
      ],
      opts: { lods: 4 },
    },
  ];
})();

describe("static conformance with his WVDBLoader parse path (75e7345)", () => {
  it.each(cases)("$name: every field his loader reads is present and decodes to our values", async ({ grids, opts }) => {
    const { bytes } = writeWvdb(grids, opts);
    const mirror = await hisLoaderMirror(bytes);
    const ours = readWvdb(bytes);
    ours.grids.forEach((g, gi) => {
      g.levels.forEach((l, k) => {
        const theirs = mirror.levels[gi]![k];
        if (l.kind === "volume") expect(theirs).toEqual(l.values);
        else expect(theirs ?? new Float32Array(0)).toEqual(l.positions);
      });
    });
  });

  it("the mirror also accepts his own committed file (sanity check of the mirror)", async () => {
    const bytes = fixture("ground_explosion_0010.wvdb");
    const mirror = await hisLoaderMirror(bytes);
    const ours = readWvdb(bytes);
    // A loop, not toEqual: toEqual on this many values took ~4 s (over the 5 s timeout under load).
    const want = (ours.grids[0] as WvdbVolumeGrid).levels[2]!.values;
    const got = mirror.levels[0]![2]!;
    expect(got.length).toBe(want.length);
    expect(bytesEqual(got, want)).toBe(true);
  });
});
