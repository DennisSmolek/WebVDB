import { describe, expect, it } from "vitest";
import {
  VDB_FRAME_ENV_WORK_HEIGHT,
  VDB_FRAME_ENV_WORK_WIDTH,
  vdbFrameCubeTexel,
  vdbFrameEnvironmentAreaTexel,
  vdbFrameEnvironmentBlurTexel,
  vdbFrameEnvironmentCubeTexel,
  vdbFrameEnvironmentDensityTerm,
  vdbFrameEnvironmentIrradiance,
  vdbFrameEnvironmentIrradianceSh,
  vdbFrameEnvironmentReduce,
  vdbFrameEnvToMap,
  vdbFrameEnvToWorld,
  vdbFrameEnvWorkSolidAngle,
  vdbFrameEquirectDirection,
  type VDBFrameEnvImage,
  type VDBFrameEnvironmentGpuResult,
} from "../../src/tools/internal.js";
import { vdbFrameEnvironmentProjection } from "../../src/core/frame/environment.js";

// K6.2b: the CPU twins of the core's environment kernels (design §3.5). The kernels themselves are compared with these
// twins on the GPU (e2e/k6-core.spec.ts `?probe=environment`); here: conventions and the white-furnace identities.

const uniform = (w: number, h: number, c: number[]): VDBFrameEnvImage => {
  const data = new Float64Array(w * h * 4);
  for (let i = 0; i < w * h; i++) data.set([...c, 1], i * 4);
  return { data, width: w, height: h };
};
const close = (a: readonly number[], b: readonly number[], eps = 1e-9) => a.forEach((v, i) => expect(v).toBeCloseTo(b[i]!, -Math.log10(eps)));

describe("K6.2b environment twins", () => {
  it("equirect map directions: u = atan2(z, x) / 2pi + 1/2, row 0 the zenith", () => {
    close(vdbFrameEquirectDirection(0.5, 0.5), [1, 0, 0]);
    close(vdbFrameEquirectDirection(0.75, 0.5), [0, 0, 1]);
    close(vdbFrameEquirectDirection(0.3, 0), [0, 1, 0]);
    close(vdbFrameEquirectDirection(0.3, 1), [0, -1, 0]);
    // the work rows' exact solid angles sum to the sphere
    let total = 0;
    for (let y = 0; y < VDB_FRAME_ENV_WORK_HEIGHT; y++) total += vdbFrameEnvWorkSolidAngle(y) * VDB_FRAME_ENV_WORK_WIDTH;
    expect(total).toBeCloseTo(4 * Math.PI, 10);
  });

  it("cube faces follow WebGPU's order and orientation", () => {
    const S = 8;
    expect(vdbFrameCubeTexel([1, 0, 0], S)).toEqual({ face: 0, x: 4, y: 4 });
    expect(vdbFrameCubeTexel([-1, 0, 0], S).face).toBe(1);
    expect(vdbFrameCubeTexel([0, 1, 0], S).face).toBe(2);
    expect(vdbFrameCubeTexel([0, -1, 0], S).face).toBe(3);
    expect(vdbFrameCubeTexel([0, 0, 1], S).face).toBe(4);
    expect(vdbFrameCubeTexel([0, 0, -1], S).face).toBe(5);
    // +X face: up (+y) is the top row, +z is the left column (s = -z)
    expect(vdbFrameCubeTexel([1, 0.9, 0], S).y).toBe(0);
    expect(vdbFrameCubeTexel([1, 0, 0.9], S).x).toBe(0);
    // +Y face: +z is the bottom row
    expect(vdbFrameCubeTexel([0, 1, 0.9], S).y).toBe(S - 1);
  });

  it("rotations: world → map and back", () => {
    const a = 0.4;
    const r = [Math.cos(a), 0, -Math.sin(a), 0, -1, 0, Math.sin(a), 0, Math.cos(a)]; // a reflection of y and a turn about y
    const d: [number, number, number] = [0.3, -0.5, 0.81];
    close(vdbFrameEnvToWorld(r, vdbFrameEnvToMap(r, d)), d);
  });

  it("white furnace: a uniform map stays uniform and gives E = pi L everywhere", () => {
    const L = [0.7, 0.5, 0.25];
    const src = uniform(300, 150, L);
    close(vdbFrameEnvironmentAreaTexel(src, 17, 40).slice(0, 3), L);
    const faces = Array.from({ length: 6 }, () => uniform(16, 16, L));
    close(vdbFrameEnvironmentCubeTexel(faces, 16, 90, 3).slice(0, 3), L);
    const work = uniform(VDB_FRAME_ENV_WORK_WIDTH, VDB_FRAME_ENV_WORK_HEIGHT, L);
    close(vdbFrameEnvironmentBlurTexel(work, [1, 0, 0, 0, 1, 0, 0, 0, 1], 10, 20).slice(0, 3), L);
    const sh = vdbFrameEnvironmentIrradianceSh(work, [1, 0, 0, 0, 1, 0, 0, 0, 1]);
    for (const n of [[0, 1, 0], [1, 0, 0], [0.6, 0, -0.8]] as [number, number, number][]) close(vdbFrameEnvironmentIrradiance(sh, n), L.map((v) => Math.PI * v), 1e-3); // three's 6-digit SH constants
    // his 'directional' cells: the light colours sum to E / 4 * 4 = pi L (energy / 4 per cell, energy 4 pi L in all)
    const map = new Float64Array(64 * 32 * 4);
    for (let i = 0; i < 64 * 32; i++) map.set([...L, 1], i * 4);
    const out = vdbFrameEnvironmentReduce(map, sh, { environmentLighting: "directional", environmentSamples: 8, environmentShOrder: 1, pairs: false, intensity: 2 });
    expect(out.valid).toBe(true);
    expect(out.count).toBe(8);
    const sum = [0, 1, 2].map((c) => out.colors.reduce((s, col) => s + col[c]!, 0));
    close(sum, L.map((v) => 2 * Math.PI * v), 1e-2);
  });

  it("an empty map is not valid: no lights, no SH", () => {
    const work = uniform(VDB_FRAME_ENV_WORK_WIDTH, VDB_FRAME_ENV_WORK_HEIGHT, [0, 0, 0]);
    const out = vdbFrameEnvironmentReduce(new Float64Array(64 * 32 * 4), vdbFrameEnvironmentIrradianceSh(work, [1, 0, 0, 0, 1, 0, 0, 0, 1]), { environmentLighting: "directional", environmentSamples: 8, environmentShOrder: 1, pairs: false, intensity: 1 });
    expect(out.valid).toBe(false);
    expect(out.count).toBe(0);
  });

  it("the density term picks the environment grid, nothing (slot light only) or 'outward' as his density pass", () => {
    const env: VDBFrameEnvironmentGpuResult = { valid: true, mode: 1, count: 2, directions: [[0, 1, 0], [1, 0, 0]], colors: [[1, 1, 1], [0.5, 0.25, 0.125]], sh: Array.from({ length: 9 }, (_, k) => (k === 0 ? [1, 1, 1] : [0, 0, 0])) };
    const grid = () => [0.3, 0.2, 0.1] as [number, number, number];
    const outside = () => null;
    const flags = { useShadows: true, slot: true, grid: true, diffusion: false };
    expect(vdbFrameEnvironmentDensityTerm(env, flags, [0, 1, 0], [0, 1, 0], [1, 1], grid)).toEqual([0.3, 0.2, 0.1]);
    // outside the grid: the unshadowed sum of the grid lights (after the slot light)
    expect(vdbFrameEnvironmentDensityTerm(env, flags, [0, 1, 0], [0, 1, 0], [1, 1], outside)).toEqual([0.5, 0.25, 0.125]);
    // one light, in the slot: the lighting adds it
    expect(vdbFrameEnvironmentDensityTerm({ ...env, count: 1 }, flags, [0, 1, 0], [0, 1, 0], [1, 1], grid)).toEqual([0, 0, 0]);
    // no shadow grid: 'outward' SH (E of a constant SH0 = 0.886227 · c0), shadowed between sky down and up
    const e = vdbFrameEnvironmentDensityTerm(env, { ...flags, useShadows: false }, [0, -1, 0], [0, 1, 0], [0.8, 0.4], grid);
    close(e, [0.886227 * 0.4, 0.886227 * 0.4, 0.886227 * 0.4]);
  });

  it("the projection of a host texture: 6 square layers are a cube, else an equirect (design §2.3)", () => {
    const t = (w: number, h: number, d: number) => ({ texture: { width: w, height: h, depthOrArrayLayers: d } as GPUTexture });
    expect(vdbFrameEnvironmentProjection(t(64, 64, 6))).toBe("cube");
    expect(vdbFrameEnvironmentProjection(t(128, 64, 1))).toBe("equirect");
    expect(vdbFrameEnvironmentProjection({ ...t(64, 64, 6), projection: "equirect" })).toBe("equirect");
  });
});
