/**
 * K3.2: the sparse volume source's host side in Node (no GPU): option checks, the info it derives from
 * the pool (texels, LOD levels, stats, box maps, GPU bytes), the coarse pyramid, and VDBFrameVolume
 * taking it. The shaders run in e2e/k3-sparse.spec.ts.
 */

import { describe, expect, it } from "vitest";
import { VDBFrameRenderer, VDBFrameVolume, vdbSparseSource, vdbVolumeUniforms, type VDBGridNode } from "../../src/three/index.js";
import { readWvdb, wvdbDenseToNanoVdb, wvdbDenseVolume } from "../../src/tools/index.js";
import { fixture } from "../tools/wvdb-helpers.js";

const dense = wvdbDenseVolume(readWvdb(fixture("cloud_04_variant_0000.wvdb")), { mipmaps: false });
const grid = wvdbDenseToNanoVdb(dense);

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
});
