import { PerspectiveCamera } from "three";
import { describe, expect, it } from "vitest";
import {
  VDBFrameEnvironmentLights,
  vdbFrameAtlasBilinear,
  vdbFrameCascadeCasters,
  vdbFrameEnvironmentGridVoxel,
  vdbFrameEnvironmentMapDirection,
  vdbFrameFindCascade,
  vdbFrameGridCascades,
  vdbFrameGridCell,
  vdbFrameGridMarch,
  vdbFrameGridRange,
  vdbFrameGridSignatureState,
  vdbFrameGridUvw,
  vdbFrameIsotropicDiffusion,
  vdbFrameLightingFroxel,
  vdbFrameReproject,
  vdbFrameSampleTiles,
  vdbFrameTransmittanceVoxel,
  vdbFrameUpdateGridSignatures,
  vdbFrameEnvironmentMapWidth,
  vdbFrameEnvironmentMapHeight,
  type VDBFrameAtlas,
  type VDBFrameGridLayout,
} from "../../src/tools/index.js";

// K3.3: the engine-free maths of mjurczyk's 'frame' lighting (tools/frame/grid.ts, environment.ts): his shadow grid
// cascades, caster ranges and signatures, and the CPU twins the e2e harness compares the GPU passes with. Each test
// pins a property of his formula that a port error would break.

type V3 = [number, number, number];
const v3 = (x = 0, y = 0, z = 0) => ({ x, y, z });
const box = (min: V3, max: V3) => ({ min: v3(...min), max: v3(...max) });
const slots = (n: number) => Array.from({ length: n }, () => v3());

function cascades(boxes: ReturnType<typeof box>[], camera: V3, distance: number, res: number, count: number) {
  const gridMin = slots(4);
  const gridSize = slots(4);
  const gridVoxel = slots(4);
  const n = vdbFrameGridCascades(boxes, v3(...camera), distance, res, count, gridMin, gridSize, gridVoxel);
  return { n, gridMin: gridMin.slice(0, n), gridSize: gridSize.slice(0, n), gridVoxel: gridVoxel.slice(0, n) };
}

describe("shadow grid cascades (frameGrid.updateGridCascades)", () => {
  it("no casters: 0; one cascade fits every caster with power-of-two voxels and snapped bounds", () => {
    expect(cascades([], [0, 0, 0], 512, 64, 1).n).toBe(0);
    const c = cascades([box([-50, -50, -30], [50, 40, 30]), box([10, -10, 0], [70, 10, 20])], [0, 0, 0], 512, 64, 1);
    expect(c.n).toBe(1);
    for (const a of ["x", "y", "z"] as const) {
      const voxel = c.gridVoxel[0]![a];
      expect(Math.log2(voxel) % 1).toBe(0);
      expect(c.gridMin[0]![a] / voxel).toBe(Math.floor(c.gridMin[0]![a] / voxel));
      expect(c.gridSize[0]![a]).toBe(voxel * 64);
    }
    // x: [-50, 70] → extent 120 / 63 → voxel 2 (2^ceil(log2 1.905)); covers min..max.
    expect(c.gridVoxel[0]!.x).toBe(2);
    expect(c.gridMin[0]!.x).toBeLessThanOrEqual(-50);
    expect(c.gridMin[0]!.x + c.gridSize[0]!.x).toBeGreaterThanOrEqual(70);
  });

  it("finer cascades sit around the camera, each half the size of the next; a camera far away keeps only the last", () => {
    const casters = [box([-64, -64, -64], [64, 64, 64])];
    const near = cascades(casters, [10, 5, -3], 512, 64, 4);
    expect(near.n).toBe(4);
    for (let i = 0; i + 1 < near.n; i++) {
      expect(near.gridVoxel[i]!.x).toBeLessThanOrEqual(near.gridVoxel[i + 1]!.x);
      // The camera is inside every fine cascade.
      expect(near.gridMin[i]!.x).toBeLessThanOrEqual(10);
      expect(near.gridMin[i]!.x + near.gridSize[i]!.x).toBeGreaterThanOrEqual(10);
    }
    // his demo camera at (-100, -11, 80) is outside the fine cascades of a 128-unit caster: only the last remains
    expect(cascades(casters, [-100, -11, 80], 512, 64, 4).n).toBe(1);
  });

  it("a fine cascade equal to the casters' box is skipped (the last one covers it)", () => {
    // A fine cascade is at most a quarter of the casters' size, so it equals their box only when that box is a point
    // at the camera; his check keeps it from being drawn twice.
    expect(cascades([box([1, 1, 1], [1, 1, 1])], [1, 1, 1], 512, 32, 3).n).toBe(1);
    expect(cascades([box([1, 1, 1], [1, 1, 1])], [0, 0, 0], 512, 32, 3).n).toBe(1);
  });

  it("caster range: voxel rectangle and atlas slices of a box in a cascade; outside is false", () => {
    const gridMin = [v3(-8, -8, -8), v3(-32, -32, -32)];
    const gridVoxel = [v3(0.25, 0.25, 0.25), v3(1, 1, 1)];
    const target = { rect: [0, 0, 0, 0] as [number, number, number, number], first: 0, count: 0 };
    expect(vdbFrameGridRange(box([-1.1, 0, 2], [1, 0.5, 3.01]), 1, 64, gridMin, gridVoxel, target)).toBe(true);
    expect(target.rect).toEqual([30, 32, 33, 33]);
    expect(target.first).toBe(64 + 34);
    expect(target.count).toBe(36 - 34);
    expect(vdbFrameGridRange(box([100, 100, 100], [110, 110, 110]), 0, 64, gridMin, gridVoxel, target)).toBe(false);
  });

  it("cascade casters: one item per caster per cascade from the first", () => {
    const pool = { items: [] as { caster: string; cascade: number }[], entries: [] as { caster: string; cascade: number }[] };
    const items = vdbFrameCascadeCasters(["a", "b"], 1, 3, pool);
    expect(items.map((i) => `${i.caster}${i.cascade}`)).toEqual(["a1", "b1", "a2", "b2"]);
    expect(vdbFrameCascadeCasters(["a"], 0, 1, pool).length).toBe(1);
    expect(pool.entries.length).toBe(4);
  });

  it("signatures: versions change only when the grid or a caster changes; castVersion only for the last cascade", () => {
    const state = vdbFrameGridSignatureState();
    const gridMin = [v3(0, 0, 0), v3(-4, -4, -4)];
    const gridSize = [v3(8, 8, 8), v3(16, 16, 16)];
    const casters = [{ m: 1 }];
    const push = (values: number[], c: { m: number }) => values.push(c.m);
    expect(vdbFrameUpdateGridSignatures(state, casters, 2, 64, false, gridMin, gridSize, push)).toBe(true);
    expect([state.gridVersion, state.castVersion]).toEqual([1, 1]);
    // Same input: no change (the redraw is skipped)
    expect(vdbFrameUpdateGridSignatures(state, casters, 2, 64, false, gridMin, gridSize, push)).toBe(false);
    expect([state.gridVersion, state.castVersion]).toEqual([1, 1]);
    // A fine cascade moved (camera): gridVersion only
    gridMin[0] = v3(1, 0, 0);
    expect(vdbFrameUpdateGridSignatures(state, casters, 2, 64, false, gridMin, gridSize, push)).toBe(false);
    expect([state.gridVersion, state.castVersion]).toEqual([2, 1]);
    // A caster changed: both
    casters[0]!.m = 2;
    expect(vdbFrameUpdateGridSignatures(state, casters, 2, 64, false, gridMin, gridSize, push)).toBe(true);
    expect([state.gridVersion, state.castVersion]).toEqual([3, 2]);
  });
});

// A 2-cascade layout of resolution 8 in an atlas of 4 columns (16 tiles → 4 x 4 tiles = 32 x 32 texels).
const layout: VDBFrameGridLayout = {
  count: 2,
  resolution: 8,
  columns: 4,
  atlasSize: [32, 32],
  gridMin: [
    [-2, -2, -2],
    [-4, -4, -4],
  ],
  gridSize: [
    [4, 4, 4],
    [8, 8, 8],
  ],
  gridVoxel: [
    [0.5, 0.5, 0.5],
    [1, 1, 1],
  ],
};

/** An RG atlas with (extinction, absorption) from a function of the voxel centre, for the layout above. */
function atlasOf(f: (world: V3, cascade: number) => [number, number]): VDBFrameAtlas {
  const data = new Float32Array(32 * 32 * 2);
  for (let y = 0; y < 32; y++) {
    for (let x = 0; x < 32; x++) {
      const cell = vdbFrameGridCell(8, 4, [x + 0.5, y + 0.5]);
      if (cell.cascade >= 2) continue;
      const m = layout.gridMin[cell.cascade]!;
      const s = layout.gridSize[cell.cascade]!;
      const w: V3 = [m[0] + cell.gridUvw[0] * s[0], m[1] + cell.gridUvw[1] * s[1], m[2] + cell.gridUvw[2] * s[2]];
      const v = f(w, cell.cascade);
      data[(y * 32 + x) * 2] = v[0];
      data[(y * 32 + x) * 2 + 1] = v[1];
    }
  }
  return { data, width: 32, height: 32, channels: 2 };
}

describe("grid shader twins", () => {
  it("getGridCell: the atlas pixel of tile z in cascade c is voxel z of c, at the voxel centre", () => {
    // tile index 8 + 3 = cascade 1, z 3: tile (3, 2) → pixels x 24..31, y 16..23
    const cell = vdbFrameGridCell(8, 4, [24 + 2.5, 16 + 5.5]);
    expect(cell.cascade).toBe(1);
    expect(cell.gridUvw).toEqual([2.5 / 8, 5.5 / 8, 3.5 / 8]);
  });

  it("findCascade: the finest cascade that holds the point half a voxel inside; −1 outside every cascade", () => {
    expect(vdbFrameFindCascade(layout, [0, 0, 0])).toBe(0);
    expect(vdbFrameFindCascade(layout, [0, 0, 0], 1)).toBe(1);
    // In cascade 0 but within half a voxel of its side: the next one
    expect(vdbFrameFindCascade(layout, [1.9, 0, 0])).toBe(1);
    expect(vdbFrameFindCascade(layout, [3, 3, 3])).toBe(1);
    expect(vdbFrameFindCascade(layout, [3.9, 0, 0])).toBe(-1);
    expect(vdbFrameGridUvw(layout, 1, [0, 0, 0])).toEqual([0.5, 0.5, 0.5]);
  });

  it("sampleTiles: exact at voxel centres, linear between slices, clamped at the sides", () => {
    const atlas = atlasOf((w) => [w[0] + 10 * w[2], 1]);
    const at = (uvw: V3) => vdbFrameSampleTiles(atlas, 8, 8, 4, uvw)[0]!;
    // Cascade 1 voxel (3, 4, 5) centre: world (-4 + 3.5, -4 + 4.5, -4 + 5.5)
    expect(at([3.5 / 8, 4.5 / 8, 5.5 / 8])).toBeCloseTo(-0.5 + 15, 12);
    // Half way between slices 5 and 6 (and x half way): the field is linear
    expect(at([4 / 8, 4.5 / 8, 6 / 8])).toBeCloseTo(0 + 20, 12);
    // Beyond the side, x clamps to the edge voxel centre
    expect(at([1.2, 4.5 / 8, 5.5 / 8])).toBeCloseTo(3.5 + 15, 12);
    expect(vdbFrameAtlasBilinear(atlas, 0.5, 0.5)).toHaveLength(2);
  });

  it("march: a uniform medium gives exp(−σ L) to the grid side (direct), and diffusion only adds light", () => {
    const sigma = 0.3;
    const ext = (w: V3): [number, number] => (vdbFrameFindCascade(layout, w, 0) < 0 ? [0, 0] : [sigma, sigma * 0.1]);
    // From the centre along +x: cascade 0 ends at 1.75 (half-voxel inset), then cascade 1 to 3.5.
    const single = vdbFrameGridMarch(ext, layout, [0, 0, 0], [1, 0, 0], 1e20, 0.5, 0);
    expect(single[0]).toBeCloseTo(Math.exp(-sigma * 4), 6);
    expect(single[1]).toBeCloseTo(single[0], 12);
    const ms = vdbFrameGridMarch(ext, layout, [0, 0, 0], [1, 0, 0], 1e20, 0.5, 0.42);
    expect(ms[0]).toBeCloseTo(single[0], 12);
    expect(ms[1]).toBeGreaterThan(ms[0]);
    // maxDistance stops the march (point lights)
    expect(vdbFrameGridMarch(ext, layout, [0, 0, 0], [1, 0, 0], 1, 0.5, 0)[0]).toBeCloseTo(Math.exp(-sigma), 6);
  });

  it("isotropic diffusion: opposite directions of a white (non-absorbing) medium sum to 2 (white furnace)", () => {
    for (const [f, b] of [
      [0, 0],
      [1.5, 0.2],
      [7, 3],
    ] as const) {
      expect(vdbFrameIsotropicDiffusion(f, b, 0) + vdbFrameIsotropicDiffusion(b, f, 0)).toBeCloseTo(2, 12);
    }
    expect(vdbFrameIsotropicDiffusion(0, 0, 0)).toBe(1);
    expect(vdbFrameIsotropicDiffusion(1, 1, 2)).toBeCloseTo(Math.exp(-2), 12);
  });

  it("transmittance voxel: empty voxels keep 1; slots pack (sky up, sky down, total 0, total 1), totals, directs", () => {
    const atlas = atlasOf((w) => (Math.hypot(...w) < 1.5 ? [0.8, 0.0] : [0, 0]));
    const far = vdbFrameTransmittanceVoxel(atlas, layout, [0.5, 0.5], [], [0, 1, 0], 0, -1);
    expect(far.flat().every((v) => v === 1)).toBe(true);
    // Cascade 0 voxel (4, 4, 4) at the centre: tile 4 = (0, 1)
    const out = vdbFrameTransmittanceVoxel(atlas, layout, [4.5, 8 + 4.5], [[0, 0, 1, 1]], [0, 1, 0], 0, -1);
    expect(out[0][2]).toBeLessThan(1); // slot 0 total (light toward +z)
    expect(out[2][0]).toBeCloseTo(out[0][2], 12); // direct = total without diffusion
    // Sky up and down are shadowed; the voxel (0.25, 0.25, 0.25) is nearer the top of the sphere than its bottom.
    expect(out[0][0]).toBeLessThan(1);
    expect(out[0][0]).toBeGreaterThan(out[0][1]);
    expect(out[0][3]).toBe(1); // no slot 1
    expect(out[3]).toEqual([1, 1, 1, 1]);
  });

  it("environment grid voxel: the unshadowed sum outside the casters; inside, colour × transmittance", () => {
    const atlas = atlasOf((w) => (Math.hypot(...w) < 3 ? [0.5, 0.0] : [0, 0]));
    const lights = [{ direction: [0, 1, 0] as V3, color: [1, 2, 3] as V3 }];
    // Env grid atlas = the last cascade alone (8 tiles in 4 columns); voxel (4, 4, 4): tile 4 = (0, 1)
    const inside = vdbFrameEnvironmentGridVoxel(atlas, layout, 4, [4.5, 8 + 4.5], lights, [1, 2, 3], 0);
    expect(inside[0]).toBeGreaterThan(0);
    expect(inside[0]).toBeLessThan(1);
    expect(inside[1] / inside[0]).toBeCloseTo(2, 12);
    const empty = atlasOf(() => [0, 0]);
    expect(vdbFrameEnvironmentGridVoxel(empty, layout, 4, [4.5, 8 + 4.5], lights, [1, 2, 3], 0)).toEqual([1, 2, 3, 1]);
  });

  it("lighting froxel: no light gives the emission; alpha is 1 − exp(−σ · length)", () => {
    const sigma = 2;
    const density: [number, number, number, number][] = [
      [0.5 * sigma, 0.5 * sigma, 0.5 * sigma, sigma],
      [0.1 * sigma, 0.2 * sigma, 0.3 * sigma, 0.5 * sigma],
      [sigma, sigma, sigma, sigma],
      [sigma, sigma, sigma, sigma],
    ];
    const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const none = { ambient: [0, 0, 0] as V3, hemiSky: [0, 0, 0] as V3, hemiGround: [0, 0, 0] as V3, lights: [], shadowLightCount: 0, phase: [0, 0, 0, 0] as [number, number, number, number], environmentColors: [], environmentShadowSlot: 0 };
    const out = vdbFrameLightingFroxel(density as never, [0, 0, -1], [0, 0, -1.5], identity, none);
    const alpha = 1 - Math.exp(-sigma * 0.5);
    expect(out[3]).toBeCloseTo(alpha, 12);
    expect(out[0] / alpha).toBeCloseTo(0.1, 12);
    // Ambient π gives albedo × 1 more; a shadowed directional slot at 0.25 adds 0.25 × albedo / π × colour
    const lit = vdbFrameLightingFroxel(density as never, [0, 0, -1], [0, 0, -1.5], identity, { ...none, ambient: [Math.PI, Math.PI, Math.PI] });
    expect(lit[1] / alpha).toBeCloseTo(0.2 + 0.5, 12);
    expect(vdbFrameLightingFroxel(density.map(() => [0, 0, 0, 0]) as never, [0, 0, -1], [0, 0, -2], identity, none)).toEqual([0, 0, 0, 0]);
  });

  it("temporal reprojection: a still camera reprojects each froxel centre onto itself with the base blend", () => {
    const camera = new PerspectiveCamera(60, 1.5, 0.1, 100);
    camera.position.set(1, 2, 3);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld(true);
    const range: [number, number] = [0.5, 40];
    const grid: V3 = [32, 24, 16];
    // A froxel centre: slice 7 of 16 at uv (0.3, 0.6) → world, by the shaders' maths (vdbFrameViewPosition)
    const t = (7 + 0.5) / 16;
    const depth = range[0] * Math.pow(range[1] / range[0], t);
    const ndc = [0.3 * 2 - 1, 0.6 * 2 - 1];
    const tanY = Math.tan((30 * Math.PI) / 180);
    const view: V3 = [ndc[0]! * tanY * 1.5 * depth, ndc[1]! * tanY * depth, -depth];
    const world = view.map((_, r) => {
      const m = camera.matrixWorld.elements;
      return m[r]! * view[0] + m[4 + r]! * view[1] + m[8 + r]! * view[2] + m[12 + r]!;
    }) as V3;
    const r = vdbFrameReproject(world, camera.matrixWorldInverse.elements, camera.projectionMatrix.elements, range, grid, { uv: [0.3, 0.6], slice: 7 }, 0.2)!;
    expect(r.uv[0]).toBeCloseTo(0.3, 9);
    expect(r.uv[1]).toBeCloseTo(0.6, 9);
    expect(r.slice).toBeCloseTo(7, 9);
    expect(r.blend).toBeCloseTo(0.2, 6);
    // Behind the previous camera: no history
    expect(vdbFrameReproject([1 + 0, 2 + 2, 3 + 6], camera.matrixWorldInverse.elements, camera.projectionMatrix.elements, range, grid, { uv: [0.5, 0.5], slice: 0 }, 0.2)).toBeNull();
  });
});

describe("environment lights (frameEnvironment.js)", () => {
  const W = vdbFrameEnvironmentMapWidth;
  const H = vdbFrameEnvironmentMapHeight;
  /** RGBA floats of the 64 × 32 map; alpha is v (the texel's row centre), the shaders' convention. */
  const map = (radiance: (dir: V3) => V3, rowsTopFirst = true) => {
    const px = new Float32Array(W * H * 4);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const v = rowsTopFirst ? (y + 0.5) / H : 1 - (y + 0.5) / H;
        const L = radiance(vdbFrameEnvironmentMapDirection((x + 0.5) / W, v));
        px.set([...L, v], (y * W + x) * 4);
      }
    }
    return px;
  };
  const lightsOf = (px: Float32Array) => {
    const e = new VDBFrameEnvironmentLights(async () => px);
    expect(e.setTexels(px)).toBe(true);
    return e;
  };

  it("a uniform environment L: the light colours sum to π L, the 'outward' irradiance (directional and SH)", () => {
    const e = lightsOf(map(() => [2, 2, 2]));
    for (const lights of [e.getDirectionalLights(8), e.getDirectionalLights(8, true), e.getShLights(1), e.getShLights(2)]) {
      const sum = lights.reduce((s, l) => s + l.color[1], 0);
      expect(sum).toBeCloseTo(Math.PI * 2, 1);
    }
    expect(new VDBFrameEnvironmentLights(async () => new Float32Array(0)).setTexels(new Float32Array(W * H * 4))).toBe(false);
  });

  it("a sun: the first (strongest) directional light points at it; the row order of the readback does not matter", () => {
    const sun: V3 = [0.4, 0.8, -0.45];
    const n = Math.hypot(...sun);
    const s = sun.map((v) => v / n) as V3;
    const radiance = (d: V3): V3 => (d[0] * s[0] + d[1] * s[1] + d[2] * s[2] > 0.99 ? [500, 480, 450] : [0.3, 0.4, 0.6]);
    for (const top of [true, false]) {
      const lights = lightsOf(map(radiance, top)).getDirectionalLights(8);
      expect(lights.length).toBe(8);
      const d = lights[0]!.direction;
      expect(d[0] * s[0] + d[1] * s[1] + d[2] * s[2]).toBeGreaterThan(0.995);
    }
    // pairs: the seeds come in opposite pairs (the second cell is opposite the sun)
    const paired = lightsOf(map(radiance)).getDirectionalLights(8, true);
    expect(paired.length).toBe(8);
  });

  it("update: 'outward' or no environment is null; the map is requested once and the colours scale by environmentIntensity", async () => {
    const px = map(() => [1, 1, 1]);
    let calls = 0;
    const e = new VDBFrameEnvironmentLights<null>(async () => {
      calls++;
      return px;
    });
    const scene = { environment: { uuid: "a", version: 1 }, environmentRotation: { x: 0, y: 0, z: 0, order: "XYZ" }, environmentIntensity: 0.4 };
    const settings = { environmentLighting: "directional", environmentSamples: 8, environmentShOrder: 1, multipleScattering: true };
    expect(e.update(null, { ...scene, environment: null }, settings)).toBeNull();
    expect(e.update(null, scene, { ...settings, environmentLighting: "outward" })).toBeNull();
    expect(e.update(null, scene, settings)).toBeNull(); // requested, not there yet
    await e.settled;
    const r = e.update(null, scene, settings)!;
    expect(calls).toBe(1);
    expect(r.mode).toBe("directional");
    expect(r.colors.slice(0, r.count).reduce((s, c) => s + c[0], 0)).toBeCloseTo(Math.PI * 0.4, 1);
  });
});
