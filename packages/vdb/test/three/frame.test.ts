/**
 * K3.1: the froxel renderer's host side in Node (no GPU): his layout, phase packing, multiple scattering
 * value and view code (frustum clip, slice range, froxel rectangle and area) on a real VDBFrameVolume over
 * the committed CC0 cloud. The passes themselves run in e2e/k3-frame.spec.ts.
 */

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { readWvdb, vdbFrameSliceDepth, vdbFrameSliceRange, wvdbDenseVolume } from "../../src/tools/index.js";
import { VDBFrameRenderer, VDBFrameVolume, vdbDenseTextures, vdbFrameAllLights } from "../../src/three/index.js";
import {
  getAtlasLayout,
  getFrameMultipleScattering,
  getFramePhase,
  getFroxelArea,
  getFroxelRect,
  getFroxelSliceRange,
} from "../../src/three/frame/index.js";
import { FrameVolumeCollector } from "../../src/three/frame/view.js";
import { getFroxelSize } from "../../src/three/frame/layout.js";
import { fixture } from "../tools/wvdb-helpers.js";

const dual = { phase: "dual-lobe" as const, phaseG: 0.8, phaseBackG: -0.3, phaseBackWeight: 0.2, phaseGExponent: 0 };

describe("@pmndrs/vdb/three K3.1 exports", () => {
  it("exports the froxel renderer surface", async () => {
    const three = (await import("../../src/three/index.js")) as Record<string, unknown>;
    for (const name of ["VDBFrameVolume", "VDBFrameRenderer", "vdbFrameRenderer", "VDBFrameInjectMaterial", "VDBFrameIntegrateMaterial", "vdbFroxelRect"]) {
      expect(typeof three[name], name).toBe("function");
    }
    expect(typeof three.vdbFrameChunks).toBe("object");
  });
});

describe("frame layout and phase packing", () => {
  it("atlas layout: his 512x512x24 grid is 5 x 5 tiles", () => {
    expect(getAtlasLayout(512, 512, 24)).toEqual({ columns: 5, width: 2560, height: 2560 });
    expect(getAtlasLayout(160, 96, 24)).toEqual({ columns: 4, width: 640, height: 576 });
  });

  it("froxel size is capped by the drawing buffer, and by pixelsPerFroxel", () => {
    const renderer = {
      getDrawingBufferSize: (t: THREE.Vector2) => t.set(320, 180),
      getSize: (t: THREE.Vector2) => t.set(320, 180),
    };
    expect(getFroxelSize({ width: 512, height: 512, pixelsPerFroxel: 0 }, renderer)).toEqual({ width: 320, height: 180 });
    expect(getFroxelSize({ width: 512, height: 512, pixelsPerFroxel: 4 }, renderer)).toEqual({ width: 80, height: 45 });
  });

  it("phase vectors and the multiple scattering value (1 − g for the diffusion modes)", () => {
    expect(getFramePhase(dual, new THREE.Vector4()).toArray()).toEqual([0.8, -0.3, 0.2, 0]);
    expect(getFramePhase({ ...dual, phase: "henyey-greenstein" }, new THREE.Vector4()).toArray()).toEqual([0.8, 0, 0, 0]);
    expect(getFramePhase({ ...dual, phase: "isotropic" }, new THREE.Vector4()).toArray()).toEqual([0, 0, 0, 0]);
    expect(getFramePhase({ ...dual, phaseG: 5 }, new THREE.Vector4()).x).toBe(0.99);
    // Dual-lobe asymmetry 0.8 · 0.8 + (−0.3) · 0.2 = 0.58.
    expect(getFrameMultipleScattering(true, dual)).toBeCloseTo(0.42, 12);
    // 'neural' is diffusion outside his network ('volume' lighting never runs the network).
    expect(getFrameMultipleScattering("neural", dual)).toBeCloseTo(0.42, 12);
    expect(getFrameMultipleScattering(false, dual)).toBe(0);
    expect(getFrameMultipleScattering("octaves", dual)).toBe(0);
  });

  it("the host slice range matches the CPU twin", () => {
    const range = new THREE.Vector2(0.5, 300);
    for (const [near, far] of [
      [3, 7],
      [0.2, 1000],
      [120, 121],
    ] as [number, number][]) {
      const host = getFroxelSliceRange({ viewNear: near, viewFar: far }, range, 24, { first: 0, count: 0 });
      expect(host).toEqual(vdbFrameSliceRange(near, far, [0.5, 300], 24));
    }
  });
});

describe("VDBFrameVolume and the view code", () => {
  const dense = wvdbDenseVolume(readWvdb(fixture("cloud_04_variant_0000.wvdb")));
  const source = vdbDenseTextures(dense);
  const bbox = new THREE.Box3(new THREE.Vector3(...dense.bbox![0]), new THREE.Vector3(...dense.bbox![1]));

  it("has his props and defaults, the source's mean density, and a proxy on the shared integrate material", () => {
    const frameRenderer = new VDBFrameRenderer();
    const v = new VDBFrameVolume(source, { frameRenderer });
    expect(v.isVDBFrameVolume).toBe(true);
    expect(v.extinction).toBe(100);
    expect(v.phase).toBe("henyey-greenstein");
    expect(v.phaseG).toBe(0.8);
    expect(v.shadowSteps).toBe(32);
    expect(v.lights).toBe(vdbFrameAllLights);
    expect(v.meanDensity).toBeCloseTo(dense.meanDensity, 12);
    expect(v.getDensityScale()).toBeCloseTo(1 / dense.meanDensity, 9);
    v.normalizeDensity = false;
    expect(v.getDensityScale()).toBe(1);
    expect(v.proxy.material).toBe(frameRenderer.integrateMaterial);
    expect(v.proxy.renderOrder).toBe(1_000_000);
    // The box: his FrameVolume matrix of the unit box (centre, size) from the .wvdb bbox.
    const box = new THREE.Box3(new THREE.Vector3(-0.5, -0.5, -0.5), new THREE.Vector3(0.5, 0.5, 0.5)).applyMatrix4(v.volumes[0]!.matrix);
    expect(box.min.distanceTo(bbox.min)).toBeLessThan(1e-9);
    expect(box.max.distanceTo(bbox.max)).toBeLessThan(1e-9);
    expect(() => new VDBFrameVolume({} as never)).toThrow(/vdbDenseTextures/);
    v.dispose();
    frameRenderer.dispose();
  });

  it("the collector clips the box to the frustum; slice range and froxel rect cover the volume", () => {
    const v = new VDBFrameVolume(source, { frameRenderer: new VDBFrameRenderer() });
    const scene = new THREE.Scene();
    scene.add(v);
    const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.01, 10000);
    const center = bbox.getCenter(new THREE.Vector3());
    const diag = bbox.getSize(new THREE.Vector3()).length();
    camera.position.copy(center).add(new THREE.Vector3(0, 0, diag * 1.5));
    camera.lookAt(center);
    camera.updateMatrixWorld();
    scene.updateMatrixWorld(true);

    const collector = new FrameVolumeCollector();
    const c = collector.collect(scene, camera, { near: 0.1, far: Infinity }, false);
    expect(c.volumes.length).toBe(1);
    const half = bbox.getSize(new THREE.Vector3()).z / 2;
    // Looking down −z at the box centre: the depth range is the box's z extent around the distance.
    expect(c.near).toBeCloseTo(diag * 1.5 - half, 6);
    expect(c.far).toBeCloseTo(diag * 1.5 + half, 6);

    const range = new THREE.Vector2(c.near, c.far);
    const slices = getFroxelSliceRange(c.volumes[0]!.volume, range, 24, { first: 0, count: 0 });
    expect(slices).toEqual({ first: 0, count: 24 });
    // Slice depths at the ends are the near and far of the range.
    expect(vdbFrameSliceDepth([c.near, c.far], 1)).toBeCloseTo(c.far, 6);

    const rect = new THREE.Vector4();
    expect(getFroxelRect(c.volumes[0]!.volume, camera, 512, 512, rect)).toBe(true);
    // Centred on screen, with one cell of padding.
    expect((rect.x + rect.z) / 2).toBeCloseTo(256, -1);
    expect((rect.y + rect.w) / 2).toBeCloseTo(256, -1);
    const area = getFroxelArea(c.volumes, camera, 512, 512, range, 24, { rect: new THREE.Vector4(), first: 0, count: 0 });
    expect(area.rect.x).toBe(Math.max(0, rect.x - 1));
    expect(area.count).toBe(24);

    // Behind the camera: nothing collected.
    camera.position.copy(center).add(new THREE.Vector3(0, 0, diag * 1.5));
    camera.lookAt(center.clone().add(new THREE.Vector3(0, 0, diag * 10)));
    camera.updateMatrixWorld();
    expect(collector.collect(scene, camera, { near: 0.1, far: Infinity }, false).volumes.length).toBe(0);
    // A hidden volume is not collected either.
    camera.lookAt(center);
    camera.updateMatrixWorld();
    v.visible = false;
    expect(collector.collect(scene, camera, { near: 0.1, far: Infinity }, false).volumes.length).toBe(0);
  });
});
