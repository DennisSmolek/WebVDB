/**
 * K6.6: the core level-set surface WGSL and raw pass, the point-sprite WGSL and raw pass (design §3.9), in
 * Node: composed WGSL (names, sizes, budget), the proxy box's winding, and what the passes record on a
 * recording device (uniform layout, dynamic offsets, bind-group rebuilds, pipeline reuse, errors, dispose).
 * The shaders run on a GPU in e2e/k6-surface.spec.ts against the tools twins.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { VDB_ALL_SCALAR_TYPES, VDB_GPU_LAYOUT_VERSION, VDBError, vdbModuleWgsl, type VDBGridTypeName } from "../../src/core/index.js";
import {
  VDB_LEVEL_SET_BOX_TRIANGLES,
  VDB_LEVEL_SET_PASS_DRAW,
  VDB_LEVEL_SET_PASS_FRAME,
  VDB_POINT_PASS_FRAME,
  vdbCreateLevelSetPassGpu,
  vdbCreatePointPassGpu,
  vdbLevelSetPassWgsl,
  vdbLevelSetProxyCorners,
  vdbLevelSetSurfaceWgsl,
  vdbPointPassWgsl,
  vdbPointPixelScale,
  vdbPointSpriteWgsl,
} from "../../src/core/internal.js";
import { vdbMat4Invert, vdbMat4Multiply } from "../../src/tools/frame/view.js";
import { wgslNameProblems } from "./wgsl-names.js";
// @ts-expect-error: repo-level ESM helper without type declarations
import { wgslSizes } from "../../../../scripts/lib/wgsl-variants.mjs";

type V3 = [number, number, number];

// ---------------------------------------------------------------------------------------------------------
// A recording device: enough of GPUDevice for the passes and the pipeline cache.

interface Recorded {
  calls: { op: string; args: unknown[] }[];
  buffers: { label: string; size: number; destroyed: boolean; data: ArrayBuffer }[];
  pipelines: number;
  modules: number;
  bindGroups: { label: string; entries: GPUBindGroupEntry[] }[];
}

function recordingDevice(): { device: GPUDevice; rec: Recorded } {
  const rec: Recorded = { calls: [], buffers: [], pipelines: 0, modules: 0, bindGroups: [] };
  const device = {
    label: "recording",
    createBindGroupLayout: (d: GPUBindGroupLayoutDescriptor) => ({ d }),
    createPipelineLayout: (d: GPUPipelineLayoutDescriptor) => ({ d }),
    createShaderModule: (d: GPUShaderModuleDescriptor) => {
      rec.modules++;
      return { d };
    },
    createRenderPipeline: (d: GPURenderPipelineDescriptor) => {
      rec.pipelines++;
      return { d };
    },
    createRenderPipelineAsync: async (d: GPURenderPipelineDescriptor) => {
      rec.pipelines++;
      return { d };
    },
    createBuffer: (d: GPUBufferDescriptor) => {
      const b = { label: d.label ?? "", size: d.size, destroyed: false, data: new ArrayBuffer(d.size), destroy: () => void (b.destroyed = true) };
      rec.buffers.push(b);
      return b;
    },
    createBindGroup: (d: GPUBindGroupDescriptor) => {
      rec.bindGroups.push({ label: d.label ?? "", entries: [...d.entries] });
      return { d };
    },
    pushErrorScope: () => {},
    popErrorScope: async () => null,
    queue: {
      writeBuffer: (b: { data: ArrayBuffer }, offset: number, data: ArrayBuffer | ArrayBufferView, dataOffset = 0, size?: number) => {
        const src = data instanceof ArrayBuffer ? new Uint8Array(data, dataOffset, size) : new Uint8Array(data.buffer, data.byteOffset + dataOffset, size ?? data.byteLength);
        new Uint8Array(b.data).set(src, offset);
      },
    },
  };
  return { device: device as unknown as GPUDevice, rec };
}

function recordingPass(rec: Recorded): GPURenderPassEncoder {
  const log =
    (op: string) =>
    (...args: unknown[]) =>
      rec.calls.push({ op, args });
  return { setPipeline: log("setPipeline"), setBindGroup: log("setBindGroup"), setVertexBuffer: log("setVertexBuffer"), draw: log("draw") } as unknown as GPURenderPassEncoder;
}

/** A pool stand-in: one 3x4 affine per grid (row-major), a box, a voxel size. */
function fakePool(types: VDBGridTypeName[] = ["float"]) {
  const affine = [0.5, 0, 0, 1, 0, 0.5, 0, 2, 0, 0, 0.5, 3];
  const pool = {
    buffer: { label: "pool A" } as unknown as GPUBuffer,
    version: 1,
    layoutVersion: VDB_GPU_LAYOUT_VERSION,
    types: new Set(types) as ReadonlySet<VDBGridTypeName>,
    info: (_id: number) => ({ indexBBox: { min: [-10, -20, -30], max: [10, 20, 30] }, indexToWorld: affine, voxelSize: [0.5, 0.5, 0.5] }),
  };
  return pool;
}

// Column-major camera: a look-at view and a WebGPU perspective (z in [0, 1]).
function lookAt(eye: V3, target: V3): Float64Array {
  const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const norm = (a: V3): V3 => {
    const l = Math.hypot(...a);
    return [a[0] / l, a[1] / l, a[2] / l];
  };
  const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const z = norm(sub(eye, target));
  const x = norm(cross([0, 1, 0], z));
  const y = cross(z, x);
  const world = new Float64Array([...x, 0, ...y, 0, ...z, 0, ...eye, 1]);
  return vdbMat4Invert(world);
}
function perspective(fovYDeg: number, aspect: number, near: number, far: number): Float64Array {
  const f = 1 / Math.tan((fovYDeg * Math.PI) / 360);
  return new Float64Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0]);
}

const readF32 = (data: ArrayBuffer, byteOffset: number, n: number) => [...new Float32Array(data, byteOffset, n)];
const readU32 = (data: ArrayBuffer, byteOffset: number, n: number) => [...new Uint32Array(data, byteOffset, n)];
const close = (a: readonly number[], b: ArrayLike<number>, eps = 1e-5) => a.every((v, i) => Math.abs(v - b[i]!) <= eps * Math.max(1, Math.abs(b[i]!)));

// ---------------------------------------------------------------------------------------------------------

describe("K6.6 WGSL composition", () => {
  const variants: [string, string][] = [
    ["level-set pass gbuffer (float)", vdbLevelSetPassWgsl({ types: ["float"], output: "gbuffer" })],
    ["level-set pass shade (float)", vdbLevelSetPassWgsl({ types: ["float"], output: "shade" })],
    ["level-set pass shade (all scalar types)", vdbLevelSetPassWgsl({ types: VDB_ALL_SCALAR_TYPES, output: "shade" })],
    ["point pass", vdbPointPassWgsl(false)],
    ["point pass (colors)", vdbPointPassWgsl(true)],
  ];

  it.each(variants)("%s: no reserved or clashing names, no back-ticks, every function declared once", (_label, code) => {
    expect(wgslNameProblems(code)).toEqual([]);
    expect(code).not.toContain("`");
    const fns = [...code.matchAll(/\bfn\s+(\w+)/g)].map((m) => m[1]!);
    expect(new Set(fns).size).toBe(fns.length);
  });

  it("the pass includes the surface text verbatim after one module, and calls the surface once", () => {
    const code = vdbLevelSetPassWgsl({ types: ["float"], output: "gbuffer" });
    expect(code).toContain(vdbLevelSetSurfaceWgsl());
    expect(code.match(/vdbLevelSetSurface\(vdbLsDraw/g)?.length).toBe(1);
    expect(code.match(/@group\(1\) @binding\(0\) var<storage, read> vdbPool/g)?.length).toBe(1);
    // The surface text has no bindings and uses only the module's public API (plus its own names).
    expect(vdbLevelSetSurfaceWgsl()).not.toMatch(/@group|@binding|pnanovdb|vdb_[a-z]/);
    expect(vdbPointSpriteWgsl()).not.toMatch(/@group|@binding/);
  });

  // D21: bytes are budgeted. design §5.3: the level-set surface = its module's budget + 4 KiB, i.e. the surface
  // functions and entries add at most 4 KiB to the module (march + levelSet, budgeted by the generator's own
  // snapshot), and the float pass fits the float row (31.5 KiB) + 4; source-free kernels 24 KiB.
  const SNAPSHOT_URL = new URL("../../src/core/surface/size-snapshot.json", import.meta.url);
  const sizes = Object.fromEntries(variants.map(([label, code]) => [label, wgslSizes(code) as { raw: number; min: number; gz: number }]));
  const BUDGET = { levelSetFloat: (31.5 + 4) * 1024, added: 4 * 1024, points: 24 * 1024 };
  const moduleMin = (types: readonly VDBGridTypeName[]) =>
    (wgslSizes(vdbModuleWgsl({ slot: { group: 1, firstBinding: 0 }, types, profile: "march", helpers: ["levelSet"] })) as { min: number }).min;
  const modules = { float: moduleMin(["float"]), all: moduleMin(VDB_ALL_SCALAR_TYPES) };
  it("matches the committed snapshot (K6_WRITE_SIZES=1 rewrites it) and fits the budgets", () => {
    if (process.env.K6_WRITE_SIZES === "1") writeFileSync(SNAPSHOT_URL, `${JSON.stringify({ unit: "bytes of minified WGSL (scripts/lib/wgsl-variants.mjs wgslSizes)", passes: sizes }, null, 2)}\n`);
    const snap = JSON.parse(readFileSync(SNAPSHOT_URL, "utf8")) as { passes: Record<string, { min: number }> };
    expect(Object.keys(snap.passes).sort()).toEqual(Object.keys(sizes).sort());
    for (const [label, s] of Object.entries(sizes)) {
      expect(s.min, `${label}: WGSL size changed; if intended run K6_WRITE_SIZES=1 pnpm vitest run packages/vdb/test/core/surface.test.ts and commit`).toBe(snap.passes[label]!.min);
      if (label.startsWith("point")) expect(s.min).toBeLessThanOrEqual(BUDGET.points);
      else {
        const module = label.includes("all scalar") ? modules.all : modules.float;
        expect(s.min - module, `${label}: bytes added to the module`).toBeLessThanOrEqual(BUDGET.added);
        if (!label.includes("all scalar")) expect(s.min).toBeLessThanOrEqual(BUDGET.levelSetFloat);
      }
    }
  });

  it("uniform structs fit their slots", () => {
    expect(VDB_LEVEL_SET_PASS_FRAME.size).toBeLessThanOrEqual(512);
    expect(VDB_LEVEL_SET_PASS_DRAW.size).toBe(256);
    expect(VDB_LEVEL_SET_PASS_DRAW.fields.get("corners")!.offset).toBe(0);
    expect(VDB_POINT_PASS_FRAME.size).toBe(176);
  });
});

describe("K6.6 level-set proxy box", () => {
  const cornerOf = (k: number): V3 => [k & 1, (k >> 1) & 1, (k >> 2) & 1];
  const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

  /** Every triangle's normal points away from the box centre (counter-clockwise from outside). */
  function outward(points: V3[]): boolean {
    const c = points.reduce<V3>((s, p) => [s[0] + p[0] / 8, s[1] + p[1] / 8, s[2] + p[2] / 8], [0, 0, 0]);
    for (let t = 0; t < 36; t += 3) {
      const [a, b, d] = [0, 1, 2].map((j) => points[VDB_LEVEL_SET_BOX_TRIANGLES[t + j]!]!) as [V3, V3, V3];
      const n = cross(sub(b, a), sub(d, a));
      const centroid: V3 = [(a[0] + b[0] + d[0]) / 3, (a[1] + b[1] + d[1]) / 3, (a[2] + b[2] + d[2]) / 3];
      if (!(dot(n, sub(centroid, c)) > 0)) return false;
    }
    return true;
  }

  it("its 12 triangles cover the 6 faces, counter-clockwise from outside", () => {
    expect(VDB_LEVEL_SET_BOX_TRIANGLES).toHaveLength(36);
    expect(outward([0, 1, 2, 3, 4, 5, 6, 7].map(cornerOf))).toBe(true);
    const faces = new Set<string>();
    for (let t = 0; t < 36; t += 3) {
      const cs = [0, 1, 2].map((j) => cornerOf(VDB_LEVEL_SET_BOX_TRIANGLES[t + j]!));
      for (let axis = 0; axis < 3; axis++) if (cs.every((c) => c[axis] === cs[0]![axis])) faces.add(`${axis}${cs[0]![axis]}`);
    }
    expect(faces.size).toBe(6);
  });

  it("corners are the index box +-1.5 through the affine; a mirroring affine keeps the winding outward", () => {
    const info = fakePool().info(0);
    const c = vdbLevelSetProxyCorners(info);
    expect([...c.subarray(0, 4)]).toEqual([0.5 * -11.5 + 1, 0.5 * -21.5 + 2, 0.5 * -31.5 + 3, 1]);
    expect([...c.subarray(28, 32)]).toEqual([0.5 * 11.5 + 1, 0.5 * 21.5 + 2, 0.5 * 31.5 + 3, 1]);
    const pts = (m: Float32Array) => [0, 1, 2, 3, 4, 5, 6, 7].map((k) => [m[k * 4]!, m[k * 4 + 1]!, m[k * 4 + 2]!] as V3);
    expect(outward(pts(c))).toBe(true);
    const mirrored = { ...info, indexToWorld: [-0.5, 0, 0, 1, 0, 0.5, 0, 2, 0, 0, 0.5, 3] };
    expect(outward(pts(vdbLevelSetProxyCorners(mirrored)))).toBe(true);
    // Fail-before: without the swap the mirrored box is inside out.
    const unswapped = new Float32Array(32);
    for (let k = 0; k < 8; k++) unswapped.set(vdbLevelSetProxyCorners(mirrored).subarray((k ^ 1) * 4, (k ^ 1) * 4 + 4), k * 4);
    expect(outward(pts(unswapped))).toBe(false);
  });
});

describe("K6.6 vdbCreateLevelSetPassGpu", () => {
  const view = lookAt([3, 4, 20], [1, 2, 3]);
  const projection = perspective(50, 16 / 9, 0.1, 100);
  const frame = { view, projection, viewport: [0, 0, 320, 180] as const };

  it("packs the camera, lights and draws, binds draws by dynamic offset, and reuses the pipeline", () => {
    const { device, rec } = recordingDevice();
    const pool = fakePool();
    const ls = vdbCreateLevelSetPassGpu(device, pool, { colorFormat: "rgba8unorm", depthFormat: "depth24plus", maxSamplesCap: 64 });
    expect(ls.types).toEqual(["float"]);
    expect(ls.spec.depthStencil).toEqual({ format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" });
    expect(ls.spec.primitive).toEqual({ topology: "triangle-list", cullMode: "front", frontFace: "ccw" });
    ls.record(recordingPass(rec), { ...frame, light: { direction: [0, -2, 0], color: [1, 0.5, 0.25] }, ambient: [0.1, 0.2, 0.3] }, [{ grid: 0 }, { grid: 3, maxSamples: 1000, sdfToIndex: 4, color: [1, 0, 0], normalStep: 0.5 }]);
    expect(rec.pipelines).toBe(1);
    const uniforms = rec.buffers.find((b) => b.label.includes("uniforms"))!;
    const F = VDB_LEVEL_SET_PASS_FRAME.fields;
    expect(close(readF32(uniforms.data, F.get("viewProjection")!.offset, 16), vdbMat4Multiply(projection, view))).toBe(true);
    expect(close(readF32(uniforms.data, F.get("projectionInverse")!.offset, 16), vdbMat4Invert(projection))).toBe(true);
    expect(close(readF32(uniforms.data, F.get("cameraWorld")!.offset, 16), vdbMat4Invert(view))).toBe(true);
    expect(readF32(uniforms.data, F.get("viewport")!.offset, 4)).toEqual([0, 0, 320, 180]);
    expect(readF32(uniforms.data, F.get("lightDirection")!.offset, 4)).toEqual([0, -1, 0, 0]);
    expect(readF32(uniforms.data, F.get("lightColor")!.offset, 3)).toEqual([1, 0.5, 0.25]);
    expect(close(readF32(uniforms.data, F.get("ambient")!.offset, 3), [0.1, 0.2, 0.3])).toBe(true);
    expect(readU32(uniforms.data, F.get("flags")!.offset, 1)).toEqual([0]);
    const D = VDB_LEVEL_SET_PASS_DRAW.fields;
    const d0 = 512;
    const d1 = 768;
    // Default maxSamples 512, clamped to the compiled cap 64.
    expect(readU32(uniforms.data, d0 + D.get("grid")!.offset, 2)).toEqual([0, 64]);
    expect(readF32(uniforms.data, d0 + D.get("sdfToIndex")!.offset, 2)).toEqual([2, 1]);
    expect(readF32(uniforms.data, d0 + D.get("color")!.offset, 4)).toEqual([1, 1, 1, 1]);
    // maxSamples clamps to the compiled cap; the corners are the proxy box.
    expect(readU32(uniforms.data, d1 + D.get("grid")!.offset, 2)).toEqual([3, 64]);
    expect(readF32(uniforms.data, d1 + D.get("sdfToIndex")!.offset, 2)).toEqual([4, 0.5]);
    expect(readF32(uniforms.data, d1 + D.get("corners")!.offset, 32)).toEqual([...vdbLevelSetProxyCorners(pool.info(3))]);
    const draws = rec.calls.filter((c) => c.op === "draw");
    expect(draws.map((c) => c.args)).toEqual([[36], [36]]);
    const groups0 = rec.calls.filter((c) => c.op === "setBindGroup" && c.args[0] === 0).map((c) => c.args[2]);
    expect(groups0).toEqual([[512], [768]]);

    // A second record: same pipeline, same pool group; a pool growth (new version and buffer) rebuilds group 1 only.
    const groupsBefore = rec.bindGroups.length;
    ls.record(recordingPass(rec), frame, [{ grid: 0 }]);
    expect(rec.pipelines).toBe(1);
    expect(rec.bindGroups.length).toBe(groupsBefore);
    pool.version = 2;
    pool.buffer = { label: "pool B" } as unknown as GPUBuffer;
    ls.record(recordingPass(rec), frame, [{ grid: 0 }]);
    expect(rec.bindGroups.length).toBe(groupsBefore + 1);
    expect(rec.bindGroups.at(-1)!.entries[0]!.resource).toEqual({ buffer: pool.buffer });
    expect(rec.pipelines).toBe(1);
  });

  it("orthographic projections set the flag; no light means black direct light; growth retires the old buffer", () => {
    const { device, rec } = recordingDevice();
    const ls = vdbCreateLevelSetPassGpu(device, fakePool(), { colorFormat: "rgba16float", depthFormat: "depth32float", output: "gbuffer" });
    const ortho = new Float64Array([0.1, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, -0.01, 0, 0, 0, 0.5, 1]);
    ls.record(recordingPass(rec), { ...frame, projection: ortho, light: null }, [{ grid: 0 }]);
    const first = rec.buffers.find((b) => b.label.includes("uniforms"))!;
    expect(readU32(first.data, VDB_LEVEL_SET_PASS_FRAME.fields.get("flags")!.offset, 1)).toEqual([1]);
    expect(readF32(first.data, VDB_LEVEL_SET_PASS_FRAME.fields.get("lightColor")!.offset, 3)).toEqual([0, 0, 0]);
    // 40 draws outgrow the first buffer (4 draws): replaced, the old one destroyed two records later.
    ls.record(recordingPass(rec), frame, Array.from({ length: 40 }, () => ({ grid: 0 })));
    expect(first.destroyed).toBe(false);
    ls.record(recordingPass(rec), frame, [{ grid: 0 }]);
    expect(first.destroyed).toBe(false);
    ls.record(recordingPass(rec), frame, [{ grid: 0 }]);
    expect(first.destroyed).toBe(true);
    ls.dispose();
    expect(rec.buffers.every((b) => b.destroyed)).toBe(true);
    expect(() => ls.record(recordingPass(rec), frame, [{ grid: 0 }])).toThrow(VDBError);
    ls.dispose();
  });

  it("rejects bad options and draws; a pool type the module lacks throws VDB_TYPE_NOT_COMPILED", async () => {
    const { device, rec } = recordingDevice();
    const code = (fn: () => unknown) => {
      try {
        fn();
      } catch (e) {
        return (e as VDBError).code;
      }
      return undefined;
    };
    expect(code(() => vdbCreateLevelSetPassGpu(device, fakePool(), { colorFormat: "rgba8unorm" } as never))).toBe("VDB_BAD_OPTION");
    expect(code(() => vdbCreateLevelSetPassGpu(device, fakePool(), { colorFormat: "rgba8unorm", depthFormat: "depth24plus", output: "x" as never }))).toBe("VDB_BAD_OPTION");
    expect(code(() => vdbCreateLevelSetPassGpu(device, fakePool(), { colorFormat: "rgba8unorm", depthFormat: "depth24plus", maxSamplesCap: 0 }))).toBe("VDB_BAD_OPTION");
    const pool = fakePool(["float"]);
    const ls = vdbCreateLevelSetPassGpu(device, pool, { colorFormat: "rgba8unorm", depthFormat: "depth24plus" });
    expect(code(() => ls.record(recordingPass(rec), frame, [{ grid: -1 }]))).toBe("VDB_BAD_OPTION");
    expect(code(() => ls.record(recordingPass(rec), { ...frame, viewport: [0, 0, 0, 10] }, [{ grid: 0 }]))).toBe("VDB_BAD_OPTION");
    expect(code(() => ls.record(recordingPass(rec), frame, [{ grid: 0, sdfToIndex: -1 }]))).toBe("VDB_BAD_OPTION");
    pool.types = new Set<VDBGridTypeName>(["float", "fp8"]);
    expect(code(() => ls.record(recordingPass(rec), frame, [{ grid: 0 }]))).toBe("VDB_TYPE_NOT_COMPILED");
    // No draws: nothing recorded.
    const before = rec.calls.length;
    vdbCreateLevelSetPassGpu(device, fakePool(), { colorFormat: "rgba8unorm", depthFormat: "depth24plus" }).record(recordingPass(rec), frame, []);
    expect(rec.calls.length).toBe(before);
    await ls.prepare();
  });
});

describe("K6.6 vdbCreatePointPassGpu", () => {
  const view = lookAt([0, 0, 12], [0, 0, 0]);
  const projection = perspective(60, 1, 0.1, 100);

  it("draws 6 vertices per point instance, picks the colour pipeline, packs the frame", () => {
    const { device, rec } = recordingDevice();
    const pts = vdbCreatePointPassGpu(device, { colorFormat: "rgba8unorm", depthFormat: "depth24plus" });
    expect(pts.specs[0].buffers).toHaveLength(1);
    expect(pts.specs[1].buffers).toHaveLength(2);
    pts.setPoints({ positions: new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2]), colors: new Float32Array(9).fill(0.5) });
    expect(pts.count).toBe(3);
    const model = new Float64Array([2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]);
    pts.record(recordingPass(rec), { view, projection, model, viewport: [0, 0, 128, 128], diameter: 0.5 });
    expect(rec.calls.filter((c) => c.op === "draw").map((c) => c.args)).toEqual([[6, 3]]);
    expect(rec.calls.filter((c) => c.op === "setVertexBuffer")).toHaveLength(2);
    const uniforms = rec.buffers.find((b) => b.label.includes("uniforms"))!;
    const F = VDB_POINT_PASS_FRAME.fields;
    expect(close(readF32(uniforms.data, F.get("modelView")!.offset, 16), vdbMat4Multiply(view, model))).toBe(true);
    const scale = readF32(uniforms.data, F.get("params")!.offset, 2);
    // diameter · maxScale · projection[5] · height / 2, min size 1.
    expect(scale[0]).toBeCloseTo(0.5 * 2 * projection[5]! * 64, 3);
    expect(scale[0]).toBeCloseTo(vdbPointPixelScale(0.5, projection, 128, model), 3);
    expect(scale[1]).toBe(1);
    // Without colours: the one-attribute pipeline.
    pts.setPoints({ positions: new Float32Array([0, 0, 0]) });
    pts.record(recordingPass(rec), { view, projection, viewport: [0, 0, 128, 128], pixelScale: 10 });
    expect(rec.calls.filter((c) => c.op === "setVertexBuffer")).toHaveLength(3);
    expect(rec.pipelines).toBe(2);
    pts.dispose();
    expect(rec.buffers.every((b) => b.destroyed)).toBe(true);
  });

  it("rejects bad data", () => {
    const { device } = recordingDevice();
    const pts = vdbCreatePointPassGpu(device, { colorFormat: "rgba8unorm" });
    expect(() => pts.setPoints({ positions: new Float32Array(4) })).toThrow(VDBError);
    expect(() => pts.setPoints({ positions: new Float32Array(6), count: 3 })).toThrow(VDBError);
    expect(() => pts.setPoints({ positions: new Float32Array(6), colors: new Float32Array(3) })).toThrow(VDBError);
    expect(() => vdbCreatePointPassGpu(device, {} as never)).toThrow(VDBError);
  });
});
