/**
 * K3.2: the sparse volume source's host side in Node (no GPU): option checks, the info it derives from
 * the pool (texels, LOD levels, stats, box maps, GPU bytes), the coarse pyramid, and VDBFrameVolume
 * taking it. The shaders run in e2e/k3-sparse.spec.ts.
 */

import { describe, expect, it } from "vitest";
import { VDBFrameRenderer, VDBFrameVolume, vdbSparseSource, vdbVolumeUniforms, type VDBGridNode } from "../../src/three/index.js";
import { readWvdb, wvdbDenseToNvdb, wvdbDenseVolume } from "../../src/tools/index.js";
import { fixture } from "../tools/wvdb-helpers.js";

const dense = wvdbDenseVolume(readWvdb(fixture("cloud_04_variant_0000.wvdb")), { mipmaps: false });
const grid = wvdbDenseToNvdb(dense);

/** A VDBGridNode stand-in: the source reads only `module`, `options` and `pool.info`. */
function fakeGrids(helpers: string[] = ["cellBoundTrilinear", "hddaCellEnd", "nodeAverage"], profile = "march", hasStats = true): VDBGridNode {
  const info = () => ({
    indexBBox: grid.indexBBox,
    indexToWorld: new Float64Array([2, 0, 0, 10, 0, 2, 0, 20, 0, 0, 2, 30]),
    hasStats,
    capacity: 123_456,
    type: "float",
  });
  return { module: {}, options: { profile, helpers }, pool: { info } } as unknown as VDBGridNode;
}

describe("vdbSparseSource", () => {
  it("checks the VDBGridNode's profile and helpers, the LOD name and the stats input", () => {
    expect(() => vdbSparseSource(fakeGrids([], "sample"), { density: 0 }, { images: { density: grid.image } })).toThrow(/profile 'march'/);
    expect(() => vdbSparseSource(fakeGrids(["cellBoundTrilinear", "hddaCellEnd"]), { density: 0 }, { lod: "node-average", images: { density: grid.image } })).toThrow(/nodeAverage/);
    // The default LOD is the coarse pyramid, built on the GPU by default (K3.2a): it needs the renderer, not the nodeAverage helper.
    expect(() => vdbSparseSource(fakeGrids(["cellBoundTrilinear", "hddaCellEnd"]), { density: 0 }, { images: { density: grid.image } })).toThrow(/initialized WebGPURenderer/);
    expect(vdbSparseSource(fakeGrids(["cellBoundTrilinear", "hddaCellEnd"]), { density: 0 }, { images: { density: grid.image }, pyramidBuild: "cpu" }).lod).toBe("coarse-pyramid");
    expect(() => vdbSparseSource(fakeGrids(), { density: 0 }, { stats: { meanNonZero: 1, max: 1, activeVoxels: 1 } })).toThrow(/the default/);
    expect(() => vdbSparseSource(fakeGrids(), { density: 0 }, { pyramidBuild: "cpu", stats: { meanNonZero: 1, max: 1, activeVoxels: 1 } })).toThrow(/images.density/);
    expect(() => vdbSparseSource(fakeGrids(), { density: 0 }, { pyramidBuild: "gl" as never, images: { density: grid.image } })).toThrow(/unknown pyramidBuild/);
    expect(() => vdbSparseSource(fakeGrids(["cellBoundTrilinear", "hddaCellEnd"]), { density: 0 }, { lod: "none", images: { density: grid.image } })).not.toThrow();
    expect(() => vdbSparseSource(fakeGrids(), { density: 0 }, { lod: "mips" as never, images: { density: grid.image } })).toThrow(/unknown lod/);
    expect(() => vdbSparseSource(fakeGrids(), { density: 0 }, { lod: "none" })).toThrow(/images.density/);
    expect(() => vdbSparseSource(fakeGrids(), { density: 0 }, { lod: "coarse-pyramid", stats: { meanNonZero: 1, max: 1, activeVoxels: 1 } })).toThrow(/coarse-pyramid/);
    expect(() => vdbSparseSource(fakeGrids(undefined, "march", false), { density: 0 }, { lod: "node-average", images: { density: grid.image } })).toThrow(/statistics/);
  });

  it("derives texels, stats (mean of the non-zero voxels: his rule), the box maps and the world box", () => {
    const s = vdbSparseSource(fakeGrids(), { density: 0 }, { lod: "none", images: { density: grid.image }, indexBox: grid.textureIndexBox });
    expect(s.kind).toBe("sparse");
    expect(s.texels).toEqual([...dense.dimensions]);
    expect(s.lodLevels).toBe(0);
    expect(s.stats.activeVoxels).toBe(dense.activeTexels);
    // The same texels as f32 instead of bytes / 255 in f64: equal to ~1e-7.
    expect(s.stats.meanNonZero).toBeCloseTo(dense.meanDensity, 6);
    expect(s.stats.max).toBeCloseTo(dense.maxDensity, 6);
    // Local unit box → sample index space: local −½ → −½ (the min cell face), +½ → N − ½.
    const m = s.boxToSource;
    expect(m[0]! * -0.5 + m[12]!).toBeCloseTo(-0.5, 6);
    expect(m[5]! * 0.5 + m[13]!).toBeCloseTo(dense.dimensions[1] - 0.5, 6);
    // World box: the cell box [−½, N − ½] through world = 2·ijk + (10, 20, 30).
    expect(s.bbox[0]).toEqual([9, 19, 29]);
    expect(s.bbox[1]).toEqual([10 + 2 * (dense.dimensions[0] - 0.5), 20 + 2 * (dense.dimensions[1] - 0.5), 30 + 2 * (dense.dimensions[2] - 0.5)]);
    expect(s.gpuBytes).toEqual({ pool: 123_456, pyramid: 0, pyramidScratch: 0 });
    expect(s.channels).toEqual({ mask: false, albedo: false, emission: false });
  });

  it("K3.2d level-0 sampler: vdbSampleFast when the node has 'fastTrilinear' (asked for without it: an error), one accessor per invocation by default", () => {
    const base = { lod: "none" as const, images: { density: grid.image } };
    const plain = vdbSparseSource(fakeGrids(), { density: 0 }, base);
    expect([plain.fastTrilinear, plain.cacheAccessor]).toEqual([false, true]);
    const fast = vdbSparseSource(fakeGrids(["cellBoundTrilinear", "hddaCellEnd", "nodeAverage", "fastTrilinear"]), { density: 0 }, base);
    expect([fast.fastTrilinear, fast.cacheAccessor]).toEqual([true, true]);
    const off = vdbSparseSource(fakeGrids(["cellBoundTrilinear", "hddaCellEnd", "fastTrilinear"]), { density: 0 }, { ...base, fastTrilinear: false, cacheAccessor: false });
    expect([off.fastTrilinear, off.cacheAccessor]).toEqual([false, false]);
    expect(() => vdbSparseSource(fakeGrids(), { density: 0 }, { ...base, fastTrilinear: true })).toThrow(/helper 'fastTrilinear'/);
  });

  it("K3.2e: stats follow the density image; without a pool revision or a resident image there is nothing to read back", async () => {
    const s = vdbSparseSource(fakeGrids(), { density: 0 }, { lod: "none", images: { density: grid.image } });
    expect(s.statsUpdates).toBe(0);
    // The fake pool has no revision, device or buffer: refreshStats does nothing (and never throws).
    await expect(s.refreshStats()).resolves.toBe(false);
    expect(s.statsUpdates).toBe(0);
  });

  it("K5.9a: the density uniform's render update keeps a per-draw value (densityScale works as on dense); a new mean is applied once, when its readback lands", async () => {
    // A pool with a revision, a device and a buffer: refreshStats reads the "replaced" image back (a fake map).
    const halved = wvdbDenseToNvdb({ ...dense, density: { ...dense.density, data: dense.density.data.map((b) => b >> 1) as Uint8Array<ArrayBuffer> } }).image;
    const pool = {
      revision: 1,
      imageRevision: 1,
      device: {
        createBuffer: () => ({ mapAsync: async () => undefined, getMappedRange: () => halved.buffer, destroy: () => undefined }),
        createCommandEncoder: () => ({ copyBufferToBuffer: () => undefined, finish: () => ({}) }),
        queue: { submit: () => undefined },
      },
      buffer: {},
      info: () => ({ indexBBox: grid.indexBBox, indexToWorld: new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]), hasStats: true, capacity: 1, type: "float", imageRevision: pool.imageRevision, resident: true, imageOffset: 0, imageBytes: halved.byteLength }),
    };
    const grids = { module: {}, options: { profile: "march", helpers: ["cellBoundTrilinear", "hddaCellEnd"] }, pool } as unknown as VDBGridNode;
    const s = vdbSparseSource(grids, { density: 0 }, { lod: "none", images: { density: grid.image } });
    const volume = new VDBFrameVolume(s, { frameRenderer: new VDBFrameRenderer() });
    const u = volume.sourceUniforms.densityScale as unknown as { value: number; update(frame: unknown): void };
    const mean0 = s.stats.meanNonZero;
    // The frame renderer writes the volume's scale before each draw (setVolumeUniforms); the render update follows.
    volume.densityScale = 3;
    u.value = volume.getDensityScale();
    u.update({});
    expect(u.value).toBeCloseTo(3 / mean0, 9);
    // An image change: the render update starts the readback; the new mean lands once, then per-draw values hold again.
    pool.revision = 2;
    pool.imageRevision = 2;
    u.update({});
    await s.refreshStats();
    expect(s.statsUpdates).toBe(1);
    const mean1 = s.stats.meanNonZero;
    expect(mean1).toBeLessThan(mean0 * 0.6);
    expect(u.value).toBeCloseTo(1 / mean1, 6); // uniforms used directly: their own scale (1) over the new mean
    u.value = volume.getDensityScale();
    u.update({});
    expect(u.value).toBeCloseTo(3 / mean1, 6);
    volume.dispose();
  });

  it("LOD levels: node-average 7; coarse-pyramid L₀ + atlas levels, r16f bytes counted", () => {
    expect(vdbSparseSource(fakeGrids(), { density: 0 }, { lod: "node-average", images: { density: grid.image } }).lodLevels).toBe(7);
    const p = vdbSparseSource(fakeGrids(), { density: 0 }, { lod: "coarse-pyramid", pyramidBuild: "cpu", images: { density: grid.image }, indexBox: grid.textureIndexBox, pyramidMax: 32 });
    // 128 voxels on the longest side, pyramidMax 32: L₀ = 2, base 21x32x11, atlas 5 levels.
    expect(p.pyramid!.level).toBe(2);
    expect(p.pyramid!.dimensions).toEqual([21, 32, 11]);
    expect(p.lodLevels).toBe(2 + p.pyramid!.atlas!.levels);
    expect(p.gpuBytes.pyramid).toBe(2 * (21 * 32 * 11 + p.pyramid!.atlas!.data!.length));
    expect(p.pyramid!.build).toBe("cpu");
    expect(p.pyramidBuilds).toBe(0);
    p.dispose();
  });

  it("VDBFrameVolume takes it: default box from the source, uniforms normalized by its mean", () => {
    const s = vdbSparseSource(fakeGrids(), { density: 0 }, { lod: "none", images: { density: grid.image } });
    const volume = new VDBFrameVolume(s, { frameRenderer: new VDBFrameRenderer() });
    expect(volume.meanDensity).toBe(s.stats.meanNonZero);
    expect(volume.hdda).toBe(false); // D30 / D31: off by default
    const u = vdbVolumeUniforms(s, { densityScale: 2 });
    expect(u.densityScale.value).toBeCloseTo(2 / s.stats.meanNonZero, 9);
    u.update(s, { densityScale: 4 });
    expect(u.densityScale.value).toBeCloseTo(4 / s.stats.meanNonZero, 9);
    expect(u.mipLevels.density.value).toBe(0);
    volume.dispose();
  });

  it("G16a followImage: a replaced image's boxes are re-read once it is resident; the uniforms and the volume's box follow", () => {
    // A pool id whose image is replaced by one with another box and voxel size (a coarse-first swap).
    const preview = { min: [0, 0, 0] as [number, number, number], max: [11, 23, 5] as [number, number, number] };
    const full = { min: [-2, 1, 0] as [number, number, number], max: [47, 95, 23] as [number, number, number] };
    const pool = {
      revision: 1,
      image: { box: preview, scale: 4, imageRevision: 1, resident: true },
      info: () => {
        const { box, scale, imageRevision, resident } = pool.image;
        return { indexBBox: box, indexToWorld: new Float64Array([scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, scale, 0]), hasStats: true, capacity: 64, type: "fpn", imageRevision, resident };
      },
    };
    const grids = { module: {}, options: { profile: "march", helpers: ["cellBoundTrilinear", "hddaCellEnd"] }, pool } as unknown as VDBGridNode;
    const base = { lod: "none" as const, images: { density: grid.image } };
    for (const fixed of [{ indexBox: full }, { bbox: [[0, 0, 0], [1, 1, 1]] as [[number, number, number], [number, number, number]] }, { auxIndexBoxes: {} }]) {
      expect(() => vdbSparseSource(grids, { density: 0 }, { ...base, followImage: true, ...fixed })).toThrow(/followImage re-reads every box/);
    }
    expect(() => vdbSparseSource(grids, { density: 0 }, { lod: "coarse-pyramid", pyramidBuild: "cpu", images: { density: grid.image }, followImage: true })).toThrow(/pyramidBuild 'gpu'/);

    const fixedSource = vdbSparseSource(grids, { density: 0 }, base);
    const s = vdbSparseSource(grids, { density: 0 }, { ...base, followImage: true });
    expect([s.followImage, fixedSource.followImage]).toEqual([true, false]);
    const volume = new VDBFrameVolume(s, { frameRenderer: new VDBFrameRenderer() });
    const u = volume.sourceUniforms as unknown as { boxes: { density: { min: { value: { toArray(): number[] } }; size: { value: { toArray(): number[] } } } }; gridTexels: { value: { toArray(): number[] } } };
    expect(s.version).toBe(1);
    expect(volume.bbox.min.toArray()).toEqual([-2, -2, -2]); // the preview's cell box [−½, max + ½] × 4
    expect(u.gridTexels.value.toArray()).toEqual([12, 24, 6]);

    // Replaced but not written yet: the GPU still holds the preview, so the boxes stay.
    pool.revision = 2;
    pool.image = { box: full, scale: 1, imageRevision: 2, resident: false };
    volume.syncSource();
    expect([s.version, s.layoutUpdates]).toEqual([1, 0]);
    expect(s.indexBox).toEqual(preview);

    // Written (pool.update): the next frame re-reads everything.
    pool.image.resident = true;
    volume.syncSource();
    expect([s.version, s.layoutUpdates]).toEqual([2, 1]);
    expect(s.indexBox).toEqual(full);
    expect(s.texels).toEqual([50, 95, 24]);
    expect(s.bbox).toEqual([
      [-2.5, 0.5, -0.5],
      [47.5, 95.5, 23.5],
    ]);
    expect(u.boxes.density.min.value.toArray()).toEqual([-2, 1, 0]);
    expect(u.boxes.density.size.value.toArray()).toEqual([50, 95, 24]);
    expect(u.gridTexels.value.toArray()).toEqual([50, 95, 24]);
    expect(volume.bbox.min.toArray()).toEqual([-2.5, 0.5, -0.5]);
    expect(volume.bbox.max.toArray()).toEqual([47.5, 95.5, 23.5]);
    // The unit box maps onto the new box: its centre and size.
    const e = volume.volumes[0]!.matrix.elements;
    expect([e[0], e[5], e[10], e[12], e[13], e[14]]).toEqual([50, 95, 24, 22.5, 48, 11.5]);
    // Nothing more to do until the image changes again.
    volume.syncSource();
    expect(s.layoutUpdates).toBe(1);

    // Without followImage the box stays as made (the K3 behaviour: a sequence keeps the first frame's box).
    expect(fixedSource.indexBox).toEqual(preview);
    expect(fixedSource.version).toBe(2);
    volume.dispose();
  });

  it("RV: a grid type the VDBGridNode does not compile throws VDB_TYPE_NOT_COMPILED (at creation, and when followImage re-reads a new image) instead of reading 0", () => {
    const pool = {
      revision: 1,
      image: { type: "fpn", imageRevision: 1, resident: true },
      info: () => ({ indexBBox: grid.indexBBox, indexToWorld: new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]), hasStats: true, capacity: 64, ...pool.image }),
    };
    const node = (types: string[]) => ({ module: {}, options: { profile: "march", helpers: ["cellBoundTrilinear", "hddaCellEnd"], types }, pool }) as unknown as VDBGridNode;
    const base = { lod: "none" as const, images: { density: grid.image }, followImage: true };
    const code = (fn: () => unknown): string | undefined => {
      try {
        fn();
      } catch (e) {
        return (e as { code?: string }).code;
      }
      return undefined;
    };
    expect(code(() => vdbSparseSource(node(["fp8"]), { density: 0 }, base))).toBe("VDB_TYPE_NOT_COMPILED");
    const s = vdbSparseSource(node(["fpn"]), { density: 0 }, base);
    // The coarse-first swap: an fpn preview, then an fp8 full grid in the same id, with a node made for the preview only.
    pool.revision = 2;
    pool.image = { type: "fp8", imageRevision: 2, resident: true };
    let message = "";
    try {
      void s.version;
    } catch (e) {
      message = String((e as Error).message);
      expect((e as { code?: string }).code).toBe("VDB_TYPE_NOT_COMPILED");
    }
    expect(message).toMatch(/fp8/);
    expect(message).toMatch(/VDBCoarseFirstEvent\.types/);
    expect(s.layoutUpdates).toBe(0); // the layout is not re-read through a module that reads 0
    // A node made with the family's types (event.types) swaps.
    pool.revision = 1;
    pool.image = { type: "fpn", imageRevision: 1, resident: true };
    const ok = vdbSparseSource(node(["fp8", "fpn"]), { density: 0 }, base);
    pool.revision = 2;
    pool.image = { type: "fp8", imageRevision: 2, resident: true };
    expect(ok.version).toBe(2);
    expect(ok.layoutUpdates).toBe(1);
  });

  it("RV: uniforms dispose their listeners; volumes made and disposed over one source leave its listener counts flat", () => {
    const pool = {
      revision: 1,
      info: () => ({ indexBBox: grid.indexBBox, indexToWorld: new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]), hasStats: true, capacity: 64, type: "float", imageRevision: 1, resident: true }),
    };
    const grids = { module: {}, options: { profile: "march", helpers: ["cellBoundTrilinear", "hddaCellEnd"] }, pool } as unknown as VDBGridNode;
    const s = vdbSparseSource(grids, { density: 0 }, { lod: "none", images: { density: grid.image }, followImage: true });
    const counts = () => {
      const r = s as unknown as { statsListeners: Set<unknown>; layoutListeners: Set<unknown> };
      return [r.statsListeners.size, r.layoutListeners.size];
    };
    const frameRenderer = new VDBFrameRenderer();
    const keep = new VDBFrameVolume(s, { frameRenderer });
    expect(counts()).toEqual([1, 1]);
    for (let i = 0; i < 50; i++) new VDBFrameVolume(s, { frameRenderer }).dispose();
    expect(counts()).toEqual([1, 1]);
    const u = vdbVolumeUniforms(s);
    expect(counts()).toEqual([2, 2]);
    u.dispose();
    u.dispose(); // idempotent
    expect(counts()).toEqual([1, 1]);
    keep.dispose();
    expect(counts()).toEqual([0, 0]);
  });
});
