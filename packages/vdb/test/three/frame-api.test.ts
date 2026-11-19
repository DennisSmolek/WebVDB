/**
 * K5.9: the froxel renderer API polish found by demo 10, host side (no GPU): one place for the phase, a medium in
 * one call, a sequence-wide density mean, the resolved box, a dense frame swap that follows by itself (and tolerates
 * a frame without a channel), and a sparse copy of a `.wvdb` aligned in one spread.
 */

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { readWvdb, writeWvdb, wvdbDenseToNvdb, wvdbDenseVolume, type WvdbDenseVolume } from "../../src/tools/index.js";
import { VDBFrameRenderer, VDBFrameVolume, VDBSequence, vdbDenseTextures, vdbFrameMedium } from "../../src/three/index.js";
import { setVolumeUniforms, vdbFrameChunks } from "../../src/three/frame/index.js";
import { fixture } from "../tools/wvdb-helpers.js";

const cloud = wvdbDenseVolume(readWvdb(fixture("cloud_04_variant_0000.wvdb")));

/** A small dense frame: density 0.2 everywhere, optionally a "flames" emission grid. */
function frame(dims: [number, number, number], flames: boolean, value = 0.2): WvdbDenseVolume {
  const n = dims[0] * dims[1] * dims[2];
  const box: [[number, number, number], [number, number, number]] = [[0, 0, 0], [1, 1, 1]];
  const grids = [{ name: "density", class: "fog volume" as const, background: 0, bbox: box, finest: { dimensions: dims, values: new Float32Array(n).fill(value) } }];
  if (flames) grids.push({ name: "flames", class: "fog volume" as const, background: 0, bbox: box, finest: { dimensions: dims, values: new Float32Array(n).fill(0.5) } });
  const { bytes } = writeWvdb(grids, { lods: 1 });
  const file = readWvdb(bytes);
  return wvdbDenseVolume(file, flames ? { emission: "flames" } : {});
}

describe("K5.9 VDBFrameVolume API", () => {
  it("phase: the renderer's settings for every volume; a volume's own fields override it ('volume' lighting), field by field", () => {
    const fr = new VDBFrameRenderer();
    const v = new VDBFrameVolume(vdbDenseTextures(cloud), { frameRenderer: fr });
    fr.settings.phase = "dual-lobe";
    fr.settings.phaseG = 0.6;
    expect(v.getPhase()).toMatchObject({ phase: "dual-lobe", phaseG: 0.6, phaseBackG: -0.3 });
    v.phaseG = 0.3;
    expect(v.getPhase()).toMatchObject({ phase: "dual-lobe", phaseG: 0.3 });
    const own = new VDBFrameVolume(vdbDenseTextures(cloud), { frameRenderer: fr, phase: "isotropic" });
    expect(own.getPhase().phase).toBe("isotropic");
    v.dispose();
    own.dispose();
  });

  it("setMedium: albedo, extinction spectrum, the renderer's phase and spectral in one call; null restores; unknown throws", () => {
    const fr = new VDBFrameRenderer();
    const v = new VDBFrameVolume(vdbDenseTextures(cloud), { frameRenderer: fr, baseColor: 0x808080, angstromExponent: 0, phaseG: 0.1 });
    const props = v.setMedium("dark-smoke")!;
    const want = vdbFrameMedium("dark-smoke")!;
    expect(props.phase).toBe(want.phase);
    expect(v.baseColor.equals(want.baseColor)).toBe(true);
    expect(v.extinctionScale.equals(want.extinctionScale)).toBe(true);
    expect(v.angstromExponent).toBe(want.angstromExponent);
    expect([fr.settings.phase, fr.settings.phaseG, fr.settings.phaseGExponent, fr.settings.spectral]).toEqual([want.phase, want.phaseG, want.phaseGExponent, true]);
    // The volume's own phase override is cleared: the medium's phase reaches 'volume' lighting too.
    expect(v.phaseG).toBe(null);
    expect(v.getPhase()).toMatchObject({ phase: want.phase, phaseG: want.phaseG, phaseGExponent: want.phaseGExponent });
    expect(v.setMedium(null)).toBe(null);
    expect(v.baseColor.getHex()).toBe(0x808080);
    expect(v.angstromExponent).toBe(0);
    expect(() => v.setMedium("plasma")).toThrow(/unknown medium/);
    v.dispose();
  });

  it("bbox: the resolved box (the source's by default, the prop's when given); densityMean fixes the mean normalizeDensity divides by", () => {
    const v = new VDBFrameVolume(vdbDenseTextures(cloud), { frameRenderer: new VDBFrameRenderer() });
    expect(v.bbox.min.toArray()).toEqual(cloud.bbox![0]);
    expect(v.bbox.max.toArray()).toEqual(cloud.bbox![1]);
    const given = new VDBFrameVolume(vdbDenseTextures(cloud), { frameRenderer: new VDBFrameRenderer(), bbox: [[0, 0, 0], [2, 4, 8]] });
    expect(given.bbox.equals(new THREE.Box3(new THREE.Vector3(0, 0, 0), new THREE.Vector3(2, 4, 8)))).toBe(true);
    expect(v.getDensityScale()).toBeCloseTo(1 / cloud.meanDensity, 9);
    v.densityMean = 0.5;
    v.densityScale = 2;
    expect(v.meanDensity).toBe(0.5);
    expect(v.getDensityScale()).toBe(4);
    v.dispose();
    given.dispose();
  });

  it("dense frames: a replace reaches the uniforms by itself (syncSource), a frame without a channel reads its absent value, an extra channel is ignored", () => {
    const first = frame([8, 8, 8], true);
    const source = vdbDenseTextures(first);
    const v = new VDBFrameVolume(source, { frameRenderer: new VDBFrameRenderer() });
    const u = v.sourceUniforms;
    // A frame of another size without flames: the emission texture is a constant 0 of the new size.
    source.replace(frame([16, 8, 8], false, 0.4));
    const emission = source.textures.emission!.map.image as { data: Uint8Array; width: number };
    expect(emission.width).toBe(16);
    expect(emission.data.every((b) => b === 0)).toBe(true);
    expect(source.version).toBe(1);
    expect(u.gridTexels.value.x).toBe(8);
    // The renderer's per-draw setter syncs the uniforms (no volume.updateSource()).
    const fake = { volumeUniforms: vdbFrameChunks.createFrameVolumeUniforms(), sourceUniforms: u };
    setVolumeUniforms(fake, v, v.volumes[0]!);
    expect(u.gridTexels.value.x).toBe(16);
    expect(u.densityScale.value).toBeCloseTo(1 / 0.4, 2);
    // A source made without emission ignores a frame's flames (warned once) instead of throwing.
    const plain = vdbDenseTextures(frame([8, 8, 8], false));
    expect(() => plain.replace(frame([8, 8, 8], true))).not.toThrow();
    expect(plain.textures.emission).toBe(null);
    v.dispose();
  });

  it("VDBSequence plays dense frames into a { source } target (needs a loader)", async () => {
    const frames = [frame([8, 8, 8], true, 0.1), frame([8, 8, 8], false, 0.3), frame([8, 8, 8], true, 0.5)];
    const source = vdbDenseTextures(frames[0]!);
    let clock = 0;
    const seq = new VDBSequence<WvdbDenseVolume>({ urls: ["a", "b", "c"], fps: 10, loader: async (_u, i) => frames[i]!, now: () => clock });
    expect(() => new VDBSequence<WvdbDenseVolume>({ urls: ["a"] }).start({ source })).toThrow(/loader/);
    seq.start({ source });
    await seq.preload(3);
    clock = 150;
    expect(seq.update(clock)).toBe(1);
    expect(source.volume).toBe(frames[1]);
    // A seek while stopped shows the frame once it is decoded.
    seq.stop();
    seq.seek(2);
    expect(source.volume).toBe(frames[2]);
    seq.dispose();
  });

  it("wvdbDenseToNvdb.sparseSourceOptions: the texture box, the dense stats and the .wvdb box", () => {
    const grid = wvdbDenseToNvdb(cloud);
    expect(grid.sparseSourceOptions.indexBox).toEqual(grid.textureIndexBox);
    expect(grid.sparseSourceOptions.stats).toEqual({ meanNonZero: cloud.meanDensity, max: cloud.maxDensity, activeVoxels: cloud.activeTexels });
    expect(grid.sparseSourceOptions.bbox).toEqual(cloud.bbox);
  });
});
