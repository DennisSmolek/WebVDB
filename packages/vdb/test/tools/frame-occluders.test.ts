import { describe, expect, it } from "vitest";
import { Matrix4, OrthographicCamera, PerspectiveCamera, Vector3, WebGPUCoordinateSystem } from "three";
import { vdbFrameOccludedTotal, vdbFrameOccluderLayer, vdbFrameOccluderVisibility, type VDBFrameOccluderCpu } from "../../src/tools/internal.js";

// K6.5b: the CPU twin of the core's host occluder lookup (core/frame/wgsl/occluders.ts). The kernel is compared with it
// on the GPU (e2e/k6-core.spec.ts `&occluders=1`); here: three's conventions and the filter.

/** A light camera's view-projection (three, WebGPU depth). */
function viewProjection(camera: OrthographicCamera | PerspectiveCamera): Float64Array {
  camera.coordinateSystem = WebGPUCoordinateSystem;
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  return Float64Array.from(new Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).elements);
}

/** An orthographic sun 10 units above the origin looking down, a 4 × 4 unit map of n × n texels. */
function sun(n: number, depth: (x: number, y: number) => number, o: Partial<VDBFrameOccluderCpu> = {}): VDBFrameOccluderCpu {
  const camera = new OrthographicCamera(-2, 2, 2, -2, 0.5, 20);
  camera.position.set(0, 10, 0);
  camera.up.set(0, 0, -1);
  camera.lookAt(0, 0, 0);
  return { viewProjection: [viewProjection(camera)], bias: 0, reversedDepth: false, filter: "bilinear", size: [n, n], depthAt: (_l, x, y) => depth(x, y), ...o };
}

/** The NDC depth of world height y under the sun above (near 0.5, far 20 at y = 10). */
const depthOfHeight = (y: number) => (10 - y - 0.5) / 19.5;

describe("K6.5b vdbFrameOccluderVisibility", () => {
  // An occluder at height 2 over x, z in [−1, 1) (texels 2..5 of 8), nothing elsewhere (depth 1)
  const occluder = sun(8, (x, y) => (x >= 2 && x < 6 && y >= 2 && y < 6 ? depthOfHeight(2) : 1));

  it("under an occluder 0, above it or beside it 1, outside the map 1", () => {
    expect(vdbFrameOccluderVisibility(occluder, [0, 0, 0])).toBe(0);
    expect(vdbFrameOccluderVisibility(occluder, [0, 3, 0])).toBe(1);
    expect(vdbFrameOccluderVisibility(occluder, [1.75, 0, 0])).toBe(1);
    expect(vdbFrameOccluderVisibility(occluder, [5, 0, 0])).toBe(1);
  });

  it("bilinear PCF between texel centres; one texel with 'nearest'", () => {
    // x = 1: the edge between texel 5 (occluded) and 6 (lit): halfway between the centres
    expect(vdbFrameOccluderVisibility(occluder, [1, 0, 0])).toBeCloseTo(0.5, 12);
    expect(vdbFrameOccluderVisibility(occluder, [1.125, 0, 0])).toBeCloseTo(0.75, 12);
    expect(vdbFrameOccluderVisibility({ ...occluder, filter: "nearest" }, [1.125, 0, 0])).toBe(1);
    expect(vdbFrameOccluderVisibility({ ...occluder, filter: "nearest" }, [0.875, 0, 0])).toBe(0);
  });

  it("the bias moves the test (three's sign: added); reversed depth flips it", () => {
    // a position just below the occluder's surface is occluded; a bias that lifts it past the surface lights it
    const below = 2 - 0.01;
    expect(vdbFrameOccluderVisibility(occluder, [0, below, 0])).toBe(0);
    expect(vdbFrameOccluderVisibility({ ...occluder, bias: -0.002 }, [0, below, 0])).toBe(1);
    const reversed = sun(8, (x, y) => (x >= 2 && x < 6 && y >= 2 && y < 6 ? 1 - depthOfHeight(2) : 0), { reversedDepth: true });
    // the twin compares the matrix's depth: a reversed map needs a reversed matrix; flip z of the clip
    const flip = new Matrix4().set(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 1, 0, 0, 0, 1);
    const m = new Matrix4().fromArray(Array.from(reversed.viewProjection[0]!));
    const r = { ...reversed, viewProjection: [Float64Array.from(new Matrix4().multiplyMatrices(flip, m).elements)] };
    expect(vdbFrameOccluderVisibility(r, [0, 0, 0])).toBe(0);
    expect(vdbFrameOccluderVisibility(r, [0, 3, 0])).toBe(1);
  });

  it("the first layer that holds the position is used (cascades, cube faces)", () => {
    const fine = new OrthographicCamera(-1, 1, 1, -1, 0.5, 20);
    fine.position.set(0, 10, 0);
    fine.up.set(0, 0, -1);
    fine.lookAt(0, 0, 0);
    const layers = [viewProjection(fine), occluder.viewProjection[0]!];
    expect(vdbFrameOccluderLayer(layers, [0.5, 0, 0])?.layer).toBe(0);
    expect(vdbFrameOccluderLayer(layers, [1.5, 0, 0])?.layer).toBe(1);
    expect(vdbFrameOccluderLayer(layers, [3, 0, 0])).toBeNull();
    // a point light's 90° face: behind the camera (w ≤ 0) is outside
    const face = new PerspectiveCamera(90, 1, 0.1, 10);
    face.position.set(0, 0, 0);
    face.lookAt(new Vector3(0, 0, -1));
    expect(vdbFrameOccluderLayer([viewProjection(face)], [0, 0, 1])).toBeNull();
    expect(vdbFrameOccluderLayer([viewProjection(face)], [0.5, -0.5, -1])?.uv).toEqual([0.75, 0.75]);
  });

  it("the transmittance target keeps the direct part and occludes the scattered part", () => {
    expect(vdbFrameOccludedTotal(0.3, 0.7, 1)).toBe(0.7);
    expect(vdbFrameOccludedTotal(0.3, 0.7, 0)).toBe(0.3);
    expect(vdbFrameOccludedTotal(0.3, 0.7, 0.5)).toBeCloseTo(0.5, 15);
  });
});
