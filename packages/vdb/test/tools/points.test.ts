/**
 * K6.6: the point-sprite twins (`pointSpriteSize`, `pointSpriteVertex`, `pointSpriteClip`) against an
 * independent f64 screen-space construction with three's camera: every sprite is a `sizePx`-wide square
 * around the point's projected pixel, `sizePx = max(pixelScale / viewDepth, minSize)`. The GPU half runs in
 * e2e/k6-surface.spec.ts (the sprite probe and the raw point pass).
 */

import { Matrix4, PerspectiveCamera, Vector3, WebGPUCoordinateSystem } from "three";
import { describe, expect, it } from "vitest";
import { pointSpriteClip, pointSpriteSize, pointSpriteVertex } from "../../src/tools/internal.js";

describe("K6.6 point sprite twins", () => {
  it("the size is pixelScale / viewDepth with a minimum, in f32", () => {
    expect(pointSpriteSize(100, 4, 1)).toBe(25);
    expect(pointSpriteSize(1, 4, 1)).toBe(1);
    expect(pointSpriteSize(1, 4, 0)).toBe(0.25);
    expect(pointSpriteSize(1, 3, 0)).toBe(Math.fround(1 / 3));
  });

  it("a corner is the centre plus corner · size pixels on screen at any depth (three's camera, f64 oracle)", () => {
    const camera = new PerspectiveCamera(55, 16 / 9, 0.1, 500);
    camera.coordinateSystem = WebGPUCoordinateSystem;
    camera.position.set(3, -2, 40);
    camera.lookAt(0, 1, 0);
    camera.updateMatrixWorld(true);
    camera.updateProjectionMatrix();
    const model = new Matrix4().makeScale(2, 2, 2).setPosition(1, 0, -3);
    const modelView = new Matrix4().multiplyMatrices(camera.matrixWorldInverse, model).elements;
    const viewport: [number, number] = [1280, 720];
    const pixelScale = 300;
    for (const local of [[0, 0, 0], [4, -3, 2], [-6, 5, -9], [0.5, 0.5, 18]]) {
      const world = new Vector3(...local).applyMatrix4(model);
      const ndc = world.clone().project(camera);
      const centre = [((ndc.x + 1) / 2) * viewport[0], ((1 - ndc.y) / 2) * viewport[1]];
      const depth = -world.clone().applyMatrix4(camera.matrixWorldInverse).z;
      const expected = Math.max(pixelScale / depth, 1);
      for (const corner of [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]] as [number, number][]) {
        const { clip, sizePx } = pointSpriteClip(local, modelView, camera.projectionMatrix.elements, corner, pixelScale, 1, viewport);
        expect(Math.abs(sizePx - expected) / expected).toBeLessThan(1e-6);
        const px = [((clip[0] / clip[3] + 1) / 2) * viewport[0], ((1 - clip[1] / clip[3]) / 2) * viewport[1]];
        // corner +x is right, +y is up on screen (pixel y down).
        expect(Math.abs(px[0]! - (centre[0]! + corner[0] * expected))).toBeLessThan(1e-3);
        expect(Math.abs(px[1]! - (centre[1]! - corner[1] * expected))).toBeLessThan(1e-3);
        expect(Math.abs(clip[2] / clip[3] - ndc.z)).toBeLessThan(1e-6);
      }
    }
  });

  it("the vertex keeps z and w and scales the offset by w (the perspective divide cancels it)", () => {
    const v = pointSpriteVertex([1, 2, 3, 4], [0.5, -0.5], 10, [200, 100]);
    expect(v).toEqual([Math.fround(1 + 0.5 * 10 / 100 * 4), Math.fround(2 - 0.5 * 10 / 50 * 4), 3, 4]);
  });
});
