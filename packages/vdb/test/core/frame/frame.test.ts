import { readFileSync, writeFileSync } from "node:fs";
import { Camera, Matrix4, PerspectiveCamera, Quaternion, Vector3, Vector4, WebGPUCoordinateSystem } from "three";
import { describe, expect, it } from "vitest";
import { vdbDenseSource } from "../../../src/core/source/dense.js";
import {
  ENV_GROUP0_LAYOUT,
  GROUP0_LAYOUT,
  VDB_FRAME_GROUP,
  VDB_FRAME_LIGHTS,
  VDB_FRAME_UNIFORMS,
  VDB_FRAME_VOLUME,
  VDB_FRAME_VOLUME_BYTES,
  VDBRetireQueue,
  VDBUniformWriter,
  vdbFrameChunkProbeSpec,
  vdbFrameCompositeSpec,
  vdbFrameDefaultSettings,
  vdbFrameEnvBlurSpec,
  vdbFrameEnvPrefilterSpec,
  vdbFrameEnvReduceSpec,
  vdbFrameEnvironmentGridSpec,
  vdbFrameGatherSpec,
  vdbFrameGridDensitySpec,
  vdbFrameTransmittanceSpec,
  vdbFrameVolumeShadowSpec,
  vdbFrameEmissionDirectionSpec,
  vdbFrameEmissionReduceSpec,
  vdbFrameEmissionProbeSpec,
  vdbVolumeShadowWgsl,
  vdbEmissionProbeWgsl,
  vdbFrameIntegrateWgsl,
  vdbFrameResolveSpec,
  vdbUniformStruct,
  type VDBComputeSpec,
  type VDBRenderSpec,
} from "../../../src/core/frame/index.js";
import { VDBFrameRenderer } from "../../../src/three/frame/renderer.js";
import { getFroxelArea, getFroxelRect, getFroxelSliceRange } from "../../../src/three/frame/view.js";
import { getFroxelSize } from "../../../src/three/frame/layout.js";
import {
  vdbFrameClipCamera,
  vdbFrameClipVolume,
  vdbFrameFroxelArea,
  vdbFrameFroxelRect,
  vdbFrameFroxelSize,
  vdbFrameVolumeSlices,
  vdbFroxelHeldSize,
} from "../../../src/tools/frame/view.js";
import { wgslNameProblems } from "../wgsl-names.js";
// @ts-expect-error: repo-level ESM helper without type declarations
import { wgslSizes } from "../../../../../scripts/lib/wgsl-variants.mjs";

/** A device stub that is enough for vdbDenseSource (textures, a sampler, writeTexture). */
const fakeDevice = () =>
  ({
    createTexture: (d: GPUTextureDescriptor) => ({ ...d, createView: () => ({}), destroy() {} }),
    createSampler: () => ({}),
    queue: { writeTexture() {} },
  }) as unknown as GPUDevice;

function denseVolume(o: { albedo?: boolean; mask?: boolean; emission?: boolean; mips?: boolean } = {}) {
  const tex = (k: 1 | 4) => ({
    data: new Uint8Array(8 * 8 * 8 * k).fill(100),
    dimensions: [8, 8, 8] as [number, number, number],
    format: k === 1 ? ("r8unorm" as const) : ("rgba8unorm" as const),
    texelBytes: k,
    mip: o.mips ? { data: new Uint8Array(8 * 4 * 4 * k), dimensions: [8, 4, 4] as [number, number, number], levels: 3 } : null,
  });
  return {
    kind: "dense",
    normalize: "clamp01",
    lod: 0,
    dimensions: [8, 8, 8],
    bbox: null,
    density: tex(1),
    mask: o.mask ? tex(1) : null,
    albedo: o.albedo ? tex(4) : null,
    emission: o.emission ? tex(1) : null,
    densityStats: { sum: 1, count: 1 },
    densityStatsSource: "texels",
    meanDensity: 0.4,
    maxDensity: 0.4,
    activeTexels: 512,
    gridNames: { density: "density" },
  } as never;
}

const sources = () => [vdbDenseSource(fakeDevice(), denseVolume({ albedo: true, mask: true, emission: true, mips: true })), vdbDenseSource(fakeDevice(), denseVolume())];

/** A user emission colour hook (K6.5): every top-level name starts with vdbUser. */
const TEST_HOOK = "fn vdbUserEmissionColor(t: f32, temperature: f32) -> vec3f {\n  return vec3f(t, 0.5 * t, 0.25) * clamp(temperature / 3000.0, 0.0, 1.0);\n}\n";

/** Every K6.1 pipeline (the key matrix): gather × mix × multi × source layout, resolve × temporal, composite × depth × sRGB. */
function allSpecs(): { compute: VDBComputeSpec[]; render: VDBRenderSpec[] } {
  const compute: VDBComputeSpec[] = [];
  for (const s of sources()) {
    for (const mix of ["mixture", "over"] as const) for (const multi of [false, true]) compute.push(vdbFrameGatherSpec(s, { mix, multi }));
    // K6.2 'frame' lighting: the fused gather and the shadow grid density per source layout
    for (const multi of [false, true]) compute.push(vdbFrameGatherSpec(s, { mix: "mixture", multi, lighting: "frame" }), vdbFrameGridDensitySpec(s, { multi }));
    // K6.5: the emission direction pass (sources with emission), with and without a user colour hook
    if (s.channels.emission) {
      compute.push(vdbFrameEmissionDirectionSpec(s, {}), vdbFrameEmissionDirectionSpec(s, { hook: TEST_HOOK }));
      compute.push(vdbFrameGatherSpec(s, { mix: "mixture", multi: false, hook: TEST_HOOK }), vdbFrameGatherSpec(s, { mix: "mixture", multi: false, lighting: "frame", hook: TEST_HOOK }));
    }
  }
  compute.push(vdbFrameTransmittanceSpec(), vdbFrameResolveSpec({ temporal: false }), vdbFrameResolveSpec({ temporal: true }));
  // K6.2b: the environment kernels (source-free)
  compute.push(vdbFrameEnvPrefilterSpec("equirect"), vdbFrameEnvPrefilterSpec("cube"), vdbFrameEnvBlurSpec(), vdbFrameEnvReduceSpec(), vdbFrameEnvironmentGridSpec());
  compute.push(vdbFrameVolumeShadowSpec(), vdbFrameEmissionReduceSpec(), vdbFrameEmissionProbeSpec());
  const render: VDBRenderSpec[] = [];
  for (const depth of ["depth", "r32float"] as const) for (const srgb of [false, true]) render.push(vdbFrameCompositeSpec({ colorFormat: "rgba16float", depth, srgb }));
  return { compute, render };
}

describe("K6.1 uniform structs", () => {
  it("lays fields out by WGSL's uniform rules", () => {
    const s = vdbUniformStruct("T", [
      { name: "a", type: "f32" },
      { name: "b", type: "vec3f" },
      { name: "c", type: "vec2f" },
      { name: "d", type: "mat4x4f" },
      { name: "e", type: "u32" },
      { name: "f", type: "vec4f", count: 3 },
    ]);
    const at = (n: string) => s.fields.get(n)!.offset;
    expect([at("a"), at("b"), at("c"), at("d"), at("e"), at("f")]).toEqual([0, 16, 32, 48, 112, 128]);
    expect(s.size).toBe(176);
    expect(s.wgsl).toContain("f: array<vec4f, 3>,");
  });

  it("the frame structs: the volume record is 512 B, the group record fits its 256 B stride, offsets follow the WGSL order", () => {
    expect(VDB_FRAME_VOLUME.size).toBe(VDB_FRAME_VOLUME_BYTES);
    expect(VDB_FRAME_GROUP.size).toBeLessThanOrEqual(256);
    for (const s of [VDB_FRAME_UNIFORMS, VDB_FRAME_LIGHTS, VDB_FRAME_VOLUME, VDB_FRAME_GROUP]) {
      const names = [...s.wgsl.matchAll(/^ {2}(\w+):/gm)].map((m) => m[1]);
      expect(names.slice(0, s.fields.size)).toEqual([...s.fields.keys()]);
      const offsets = [...s.fields.values()].map((f) => f.offset);
      expect(offsets).toEqual([...offsets].sort((a, b) => a - b));
      for (const f of s.fields.values()) expect(f.offset % (f.type.startsWith("vec3") || f.type.startsWith("vec4") || f.type.startsWith("mat") ? 16 : 4)).toBe(0);
    }
    expect(VDB_FRAME_UNIFORMS.fields.get("grid")!.offset).toBe(256);
    expect(VDB_FRAME_LIGHTS.fields.get("position")!.offset).toBe(32 + 9 * 16 + 3 * 4 * 16);
  });

  it("the packer writes floats and integers at the struct offsets", () => {
    const w = new VDBUniformWriter(2 * VDB_FRAME_VOLUME_BYTES);
    w.set(VDB_FRAME_VOLUME, VDB_FRAME_VOLUME_BYTES, "slices", [3, 4, 5, 6]);
    w.set(VDB_FRAME_VOLUME, VDB_FRAME_VOLUME_BYTES, "density", [0.5, 0.25, 2, 1]);
    const slices = (VDB_FRAME_VOLUME_BYTES + VDB_FRAME_VOLUME.fields.get("slices")!.offset) / 4;
    const density = (VDB_FRAME_VOLUME_BYTES + VDB_FRAME_VOLUME.fields.get("density")!.offset) / 4;
    expect([...w.u32.subarray(slices, slices + 4)]).toEqual([3, 4, 5, 6]);
    expect([...w.f32.subarray(density, density + 4)]).toEqual([0.5, 0.25, 2, 1]);
  });
});

describe("K6.1 composed WGSL", () => {
  const { compute, render } = allSpecs();
  const texts = [...compute, ...render, vdbFrameChunkProbeSpec()].map((s) => [s.label, s.code] as const);
  texts.push(["vdbFrameIntegrateWgsl", vdbFrameIntegrateWgsl()], ["vdbVolumeShadowWgsl", vdbVolumeShadowWgsl()], ["vdbEmissionProbeWgsl", vdbEmissionProbeWgsl()]);

  it.each(texts)("%s: no reserved words, no function/variable clashes, no back-ticks, no duplicate declarations", (_, code) => {
    expect(wgslNameProblems(code)).toEqual([]);
    expect(code).not.toContain("`");
    const fns = [...code.matchAll(/\bfn\s+(\w+)/g)].map((m) => m[1]);
    expect(fns.filter((f, i) => fns.indexOf(f) !== i)).toEqual([]);
    const structs = [...code.matchAll(/\bstruct\s+(\w+)/g)].map((m) => m[1]);
    expect(structs.filter((f, i) => structs.indexOf(f) !== i)).toEqual([]);
    for (const word of ["active", "set", "filter", "target"]) expect(code).not.toMatch(new RegExp(`\\b(?:let|var|fn)\\s+${word}\\b`));
  });

  it("the same options give the same bytes (fixed chunk order)", () => {
    const again = allSpecs();
    expect(again.compute.map((s) => s.code)).toEqual(compute.map((s) => s.code));
  });
});

describe("K6.1 device limits (design §1)", () => {
  const { compute, render } = allSpecs();
  it.each([...compute, ...render].map((s) => [s.label, s] as const))("%s fits the core limits", (_, spec) => {
    expect(spec.groups.length).toBeLessThanOrEqual(3);
    const all = spec.groups.flatMap((g) => [...g.entries]);
    expect(all.filter((e) => e.storageTexture).length).toBeLessThanOrEqual(2);
    // Sampled textures: WebGPU's default (and compat's) maxSampledTexturesPerShaderStage is 16. K6.5b's 4 occluder
    // slots raise the worst kernel (the 'frame' gather of a dense source with mask, albedo and emission and their mips:
    // 8 source + transmittance + environment grid + emission LUT + 4 occluders) to 15; the design's own cap was 12
    // before the occluders (design §3.2).
    expect(all.filter((e) => e.texture).length).toBeLessThanOrEqual(16);
    expect(all.filter((e) => e.buffer?.type === "storage" || e.buffer?.type === "read-only-storage").length).toBeLessThanOrEqual(1);
    expect(all.filter((e) => e.buffer?.type === "uniform").length).toBeLessThanOrEqual(5);
    expect(all.filter((e) => e.sampler).length).toBeLessThanOrEqual(16);
    // the environment prefilter / blur / reduce kernels have their own group 0 (their parameters; K6.2b)
    expect([GROUP0_LAYOUT, ENV_GROUP0_LAYOUT]).toContain(spec.groups[0]);
    if (!/environment (prefilter|blur|reduce)/.test(spec.label)) expect(spec.groups[0]).toBe(GROUP0_LAYOUT);
    for (const g of spec.groups) {
      const bindings = [...g.entries].map((e) => e.binding).sort((a, b) => a - b);
      expect(new Set(bindings).size).toBe(bindings.length);
    }
    for (const m of spec.code.matchAll(/@workgroup_size\((\d+),\s*(\d+),\s*(\d+)\)/g)) expect(Number(m[1]) * Number(m[2]) * Number(m[3])).toBe(64);
    // every declared @group is a layout group (no gaps, nothing past the layouts)
    const groups = new Set([...spec.code.matchAll(/@group\((\d+)\)/g)].map((m) => Number(m[1])));
    for (const g of groups) expect(g).toBeLessThan(spec.groups.length);
  });
});

describe("K6.5b sampled-texture headroom", () => {
  it("the worst kernel samples 15 textures (16 is the WebGPU default limit): one left", () => {
    const { compute } = allSpecs();
    const counts = compute.map((s) => [s.label, s.groups.flatMap((g) => [...g.entries]).filter((e) => e.texture).length] as const);
    const worst = Math.max(...counts.map(([, n]) => n));
    expect(worst).toBe(15);
    expect(counts.filter(([, n]) => n === worst).every(([label]) => label.startsWith("vdb frame gather (frame lighting"))).toBe(true);
  });
});

describe("K6.1 retire queue", () => {
  it("destroys a retired object two updates later; drain destroys the rest", () => {
    const q = new VDBRetireQueue();
    const made = Array.from({ length: 3 }, () => ({ destroyed: 0, destroy() {
      this.destroyed++;
    } }));
    q.retire(made[0]);
    q.advance();
    expect(made[0]!.destroyed).toBe(0);
    q.retire(made[1]);
    q.advance();
    expect(made[0]!.destroyed).toBe(1);
    expect(made[1]!.destroyed).toBe(0);
    q.retire(made[2]);
    q.drain();
    expect(made.map((m) => m.destroyed)).toEqual([1, 1, 1]);
    expect(q.pending).toBe(0);
  });
});

describe("K6.1 settings", () => {
  it("the core defaults equal ./three's VDBFrameRenderer settings, field by field, plus froxelMix and occlusion", () => {
    const core = vdbFrameDefaultSettings();
    const three = new VDBFrameRenderer().settings as unknown as Record<string, unknown>;
    const { froxelMix, occlusion, ...shared } = core;
    expect(froxelMix).toBe("mixture");
    // K6.5b: host occluder maps shadow the volumes by default (core only)
    expect(occlusion).toBe(true);
    expect(shared).toEqual(three);
  });
});

describe("K6.1 tools/frame/view.ts equals three/frame/view.ts", () => {
  let seed = 9;
  const r = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
  it("clip depths, rects, slices, areas and froxel sizes on random cameras and boxes", () => {
    let compared = 0;
    for (let trial = 0; trial < 200; trial++) {
      const camera = new PerspectiveCamera(30 + r() * 60, 0.5 + r() * 1.5, 0.05 + r(), 50 + r() * 200);
      camera.coordinateSystem = WebGPUCoordinateSystem;
      camera.position.set(r() * 10 - 5, r() * 10 - 5, r() * 10 - 5);
      camera.lookAt(r() - 0.5, r() - 0.5, r() - 0.5);
      camera.updateMatrixWorld(true);
      camera.updateProjectionMatrix();
      const box = new Matrix4().compose(
        new Vector3(r() * 4 - 2, r() * 4 - 2, r() * 4 - 2),
        new Quaternion(r() - 0.5, r() - 0.5, r() - 0.5, 1).normalize(),
        new Vector3(0.5 + r() * 3, 0.5 + r() * 3, 0.5 + r() * 3),
      );
      const clip = vdbFrameClipCamera(camera.matrixWorldInverse.elements, camera.projectionMatrix.elements, { coordinateSystem: "webgpu" });
      const ours = vdbFrameClipVolume(clip, box.elements);
      // three: the collector's maths on one record
      const localToView = new Matrix4().multiplyMatrices(camera.matrixWorldInverse, box);
      const W = 64 + Math.floor(r() * 400);
      const H = 64 + Math.floor(r() * 400);
      if (!ours) continue;
      const theirsRect = new Vector4();
      const okThree = getFroxelRect({ localToView }, camera as Camera, W, H, theirsRect);
      const oursRect = vdbFrameFroxelRect(ours.localToView, camera.projectionMatrix.elements, W, H);
      expect(!!oursRect).toBe(okThree);
      if (oursRect) expect(oursRect).toEqual([theirsRect.x, theirsRect.y, theirsRect.z, theirsRect.w]);
      const range = { x: Math.max(0.1, ours.viewNear), y: Math.max(ours.viewFar, ours.viewNear * 1.0001 + 0.2) };
      const D = 8 + Math.floor(r() * 40);
      const target = { first: 0, count: 0 };
      getFroxelSliceRange({ viewNear: ours.viewNear, viewFar: ours.viewFar }, range, D, target);
      const slices = vdbFrameVolumeSlices(ours.viewNear, ours.viewFar, [range.x, range.y], D);
      expect(slices).toEqual(target);
      const area = { rect: new Vector4(), first: 0, count: 0 };
      getFroxelArea([{ volume: { localToView, viewNear: ours.viewNear, viewFar: ours.viewFar } }], camera as Camera, W, H, range, D, area);
      const a = vdbFrameFroxelArea([{ rect: oursRect, ...slices }], W, H, D);
      expect([...a.rect, a.first, a.count]).toEqual([area.rect.x, area.rect.y, area.rect.z, area.rect.w, area.first, area.count]);
      compared++;
    }
    expect(compared).toBeGreaterThan(40);
  });

  it("froxel sizes and the held size follow layout.ts getFroxelSize", () => {
    const held = { canvasX: 0, canvasY: 0, bufferX: 0, bufferY: 0 };
    const ours = vdbFroxelHeldSize();
    for (const [bw, bh, cw, ch, ppf] of [
      [1280, 720, 640, 360, 0],
      [1280, 720, 640, 360, 4],
      [1300, 700, 650, 350, 4],
      [2000, 700, 1000, 350, 4],
      [300, 200, 300, 200, 2],
    ] as const) {
      const settings = { width: 512, height: 512, pixelsPerFroxel: ppf };
      const three = getFroxelSize(settings, { getDrawingBufferSize: (t) => t.set(bw, bh), getSize: (t) => t.set(cw, ch) }, { width: 1, height: 1 }, held);
      expect(vdbFrameFroxelSize(settings, { drawingBuffer: [bw, bh], canvas: [cw, ch] }, ours)).toEqual(three);
    }
    expect(ours).toEqual(held);
  });
});

describe("K6.1 WGSL sizes (D21; design §5.3)", () => {
  const SNAPSHOT_URL = new URL("../../../src/core/frame/size-snapshot.json", import.meta.url);
  const { compute, render } = allSpecs();
  const sizes = Object.fromEntries([...compute, ...render, vdbFrameChunkProbeSpec()].map((s) => [s.label, (wgslSizes(s.code) as { min: number; gz: number })]));
  const CAP_SOURCE = 56 * 1024;
  const CAP_FREE = 24 * 1024;

  it("matches the committed snapshot (K6_WRITE_SIZES=1 rewrites it) and fits the caps", () => {
    if (process.env.K6_WRITE_SIZES === "1") writeFileSync(SNAPSHOT_URL, `${JSON.stringify({ unit: "bytes of minified WGSL (scripts/lib/wgsl-variants.mjs wgslSizes)", kernels: sizes }, null, 2)}\n`);
    const snap = JSON.parse(readFileSync(SNAPSHOT_URL, "utf8")) as { kernels: Record<string, { min: number; gz: number }> };
    expect(Object.keys(snap.kernels).sort()).toEqual(Object.keys(sizes).sort());
    for (const [label, s] of Object.entries(sizes)) {
      expect(s.min, `${label}: WGSL size changed; if intended run K6_WRITE_SIZES=1 pnpm vitest run packages/vdb/test/core/frame and commit`).toBe(snap.kernels[label]!.min);
      expect(s.min).toBeLessThanOrEqual(label.includes("gather") || label.includes("grid density") || label.includes("emission direction") ? CAP_SOURCE : CAP_FREE);
    }
  });
});
