/**
 * K6.R: the reference renderer's GPU half without a GPU: the composed WGSL (names, back-ticks, one track and one
 * density call site, every fail-before patch applies exactly once) and the uniform packer against the WGSL struct
 * layout. The kernel runs against its CPU twin in e2e/k6-reference.spec.ts.
 */
import { describe, expect, it } from "vitest";
import {
  VDB_REFERENCE_MUTATIONS,
  VDB_REF_HEMI_BYTES,
  VDB_REF_LIGHT_BYTES,
  VDB_REF_UNIFORM_OFFSETS,
  VDB_REF_VOLUME_BYTES,
  vdbReferencePackUniforms,
  vdbReferenceTableLayout,
  vdbReferenceWgsl,
  type VDBReferenceSourceGpu,
} from "../../src/core/internal.js";
import { VDB_REFERENCE_MAX_HEMISPHERES, VDB_REFERENCE_MAX_LIGHTS, VDB_REFERENCE_MAX_VOLUMES, vdbRefInvert4, vdbReferenceBlackbodyTable, vdbReferenceMiePhaseTable, type VDBReferenceMajorantGrid } from "../../src/tools/internal.js";
import { wgslNameProblems } from "./wgsl-names.js";

/** A source with the group-2 functions as constants (the WGSL text only). */
const stubSource = {
  kind: "dense",
  texels: [1, 1, 1],
  lodLevels: 0,
  channels: { mask: false, albedo: false, emission: false },
  stats: { meanNonZero: 1, max: 1, activeVoxels: 1 },
  boxToSource: new Float32Array(16),
  version: 0,
  wgsl: () => `fn vdbSrcDensity(v: u32, uvw: vec3f, lod: f32) -> f32 { return 1.0; }
fn vdbSrcMask(v: u32, uvw: vec3f, lod: f32) -> f32 { return 1.0; }
fn vdbSrcAlbedo(v: u32, uvw: vec3f, lod: f32) -> vec3f { return vec3f(1.0); }
fn vdbSrcEmission(v: u32, uvw: vec3f, lod: f32) -> f32 { return 0.0; }`,
  layoutEntries: () => [],
  bindEntries: () => [],
  dispose: () => {},
} as unknown as VDBReferenceSourceGpu;

/** WGSL struct layout (sizes and alignments of the scalar, vector, matrix, array and struct members we use). */
function structLayout(wgsl: string, name: string): { offsets: Record<string, number>; size: number; align: number } {
  const body = new RegExp(`struct ${name} \\{([\\s\\S]*?)\\n\\}`).exec(wgsl)![1]!;
  const offsets: Record<string, number> = {};
  let offset = 0;
  let maxAlign = 4;
  for (const m of body.matchAll(/(\w+): ([\w<>, ]+),/g)) {
    const [, field, type] = m as unknown as [string, string, string];
    const { size, align } = typeLayout(wgsl, type.trim());
    offset = Math.ceil(offset / align) * align;
    offsets[field] = offset;
    offset += size;
    maxAlign = Math.max(maxAlign, align);
  }
  return { offsets, size: Math.ceil(offset / maxAlign) * maxAlign, align: maxAlign };
}

function typeLayout(wgsl: string, type: string): { size: number; align: number } {
  if (type === "u32" || type === "f32") return { size: 4, align: 4 };
  if (type === "vec2u" || type === "vec2f") return { size: 8, align: 8 };
  if (type === "vec4f" || type === "vec4u") return { size: 16, align: 16 };
  if (type === "mat4x4f") return { size: 64, align: 16 };
  const arr = /^array<(\w+), (\d+)>$/.exec(type);
  if (arr) {
    const e = typeLayout(wgsl, arr[1]!);
    const stride = Math.ceil(e.size / e.align) * e.align;
    return { size: stride * Number(arr[2]), align: e.align };
  }
  const s = structLayout(wgsl, type);
  return { size: s.size, align: s.align };
}

describe("K6.R reference kernel WGSL", () => {
  const wgsl = vdbReferenceWgsl(stubSource);

  it("has no reserved names, no back-ticks, one entry point", () => {
    expect(wgslNameProblems(wgsl)).toEqual([]);
    expect(wgsl.includes("`")).toBe(false);
    expect(wgsl.match(/@compute/g)?.length).toBe(1);
  });

  it("reaches the tracker and the source from one call site each (SwiftShader compile time)", () => {
    const kernel = wgsl.slice(wgsl.indexOf("K6.R reference renderer"));
    expect(kernel.match(/vdbRefTrack\(/g)?.length).toBe(2); // declaration + one call
    for (const f of ["vdbSrcDensity", "vdbSrcMask", "vdbSrcAlbedo", "vdbSrcEmission", "vdbRefSigma", "vdbRefSample"]) {
      expect(kernel.match(new RegExp(`${f}\\(`, "g"))?.length, f).toBe(f === "vdbRefSigma" || f === "vdbRefSample" ? 2 : 1);
    }
  });

  it("applies every fail-before patch exactly once", () => {
    for (const [name, [from]] of Object.entries(VDB_REFERENCE_MUTATIONS)) expect(wgsl.split(from).length - 1, name).toBe(1);
  });

  it("lays out VdbRefUniforms as the packer writes it", () => {
    const u = structLayout(wgsl, "VdbRefUniforms");
    const O = VDB_REF_UNIFORM_OFFSETS;
    expect(u.offsets).toMatchObject({
      cameraToWorld: O.cameraToWorld,
      projectionInverse: O.projectionInverse,
      size: O.size,
      seed: O.seed,
      spp: O.spp,
      maxBounces: O.maxBounces,
      rrStart: O.rrStart,
      volumeCount: O.volumeCount,
      lightCount: O.lightCount,
      hemiCount: O.hemiCount,
      flags: O.flags,
      envSize: O.envSize,
      uniformEnv: O.uniformEnv,
      envRotation0: O.envRotation,
      sh: O.sh,
      hemis: O.hemis,
      volumes: O.volumes,
      lights: O.lights,
    });
    expect(u.size).toBe(O.size_);
    expect(structLayout(wgsl, "VdbRefVolume").size).toBe(VDB_REF_VOLUME_BYTES);
    expect(structLayout(wgsl, "VdbRefVolume").offsets).toMatchObject({ ids: 144, grid: 160, tables: 176 });
    expect(structLayout(wgsl, "VdbRefLight").size).toBe(VDB_REF_LIGHT_BYTES);
    expect(structLayout(wgsl, "VdbRefHemi").size).toBe(VDB_REF_HEMI_BYTES);
    expect(O.lights + VDB_REFERENCE_MAX_LIGHTS * VDB_REF_LIGHT_BYTES).toBe(O.size_);
    expect(O.volumes + VDB_REFERENCE_MAX_VOLUMES * VDB_REF_VOLUME_BYTES).toBe(O.lights);
    expect(O.hemis + VDB_REFERENCE_MAX_HEMISPHERES * VDB_REF_HEMI_BYTES).toBe(O.volumes);
  });

  it("packs the scene where the struct expects it", () => {
    const l2w = [2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 1, 2, 3, 1];
    const buf = vdbReferencePackUniforms(
      {
        camera: { cameraToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 9, 1], projectionInverse: l2w, orthographic: true },
        volumes: [{ localToWorld: l2w, sigma: 2, majorant: 1.5, cutoff: 0.1, albedo: [0.5, 0.6, 0.7], phase: { g: 0.3, backG: -0.2, backWeight: 0.4 }, emits: true, emission: { color: [1, 2, 3], intensity: 2, temperatureRange: [100, 900] }, shape: "sphere", v: 5 }],
        lights: [{ kind: "spot", position: [1, 2, 3], direction: [0, 0, -2], intensity: [4, 5, 6], cosOuter: 0.5, cosInner: 0.7, inverseSquare: true }],
        environment: { uniform: [0.1, 0.2, 0.3], hemispheres: [{ sky: [1, 1, 1], ground: [0, 0, 0], up: [0, 2, 0] }] },
      },
      { width: 7, height: 5, seed: 9, spp: 3, maxBounces: 11, rrStart: 2, pixelFilter: "box" },
    );
    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    const O = VDB_REF_UNIFORM_OFFSETS;
    expect([u[O.size / 4], u[O.size / 4 + 1], u[O.seed / 4], u[O.spp / 4], u[O.maxBounces / 4], u[O.rrStart / 4], u[O.volumeCount / 4], u[O.lightCount / 4], u[O.hemiCount / 4], u[O.flags / 4]]).toEqual([7, 5, 9, 3, 11, 2, 1, 1, 1, 1 | 2]);
    expect(f[O.cameraToWorld / 4 + 14]).toBe(9);
    const v = O.volumes / 4;
    const inv = vdbRefInvert4(l2w);
    for (let i = 0; i < 16; i++) expect(f[v + i]).toBeCloseTo(inv[i]!, 6);
    expect(Array.from(f.subarray(v + 16, v + 36))).toEqual(
      [2, 3, 0.1, 1, 0.5, 0.6, 0.7, 1, 0.3, -0.2, 0.4, 0, 2, 4, 6, 0, 100, 900, 0, 0].map(Math.fround),
    );
    expect(u[v + 36]).toBe(5);
    const l = O.lights / 4;
    expect(Array.from(f.subarray(l, l + 16))).toEqual([1, 2, 3, 2, 0, 0, -1, 0.5, 4, 5, 6, 0.7, 1, 0, 0, 0].map(Math.fround));
    const h = O.hemis / 4;
    expect(Array.from(f.subarray(h + 8, h + 12))).toEqual([0, 1, 0, 0]);
    expect(Array.from(f.subarray(O.uniformEnv / 4, O.uniformEnv / 4 + 4))).toEqual([0.1, 0.2, 0.3, 1].map(Math.fround));
  });

  it("K6.R1: packs tables, majorant grids, the light probe and three's falloff", () => {
    const l2w = [2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 1, 2, 3, 1];
    const mie = vdbReferenceMiePhaseTable("rgb");
    const bb = vdbReferenceBlackbodyTable(1.39);
    const g1: VDBReferenceMajorantGrid = { dims: [2, 3, 4], values: Float32Array.from({ length: 24 }, (_, i) => i) };
    const g2: VDBReferenceMajorantGrid = { dims: [5, 1, 2], values: Float32Array.from({ length: 10 }, (_, i) => 100 + i) };
    const sh = Array.from({ length: 27 }, (_, i) => i / 10);
    const scene = {
      camera: { cameraToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 9, 1], projectionInverse: l2w },
      volumes: [
        { localToWorld: l2w, sigma: 2, phase: { g: 0, table: mie }, majorantGrid: g1, emits: true, emission: { colorTable: bb, temperatureRange: [0, 3000] as [number, number] } },
        { localToWorld: l2w, sigma: 2, phase: { g: 0, table: mie }, majorantGrid: g2 },
        { localToWorld: l2w, sigma: 2, majorantGrid: g1 },
      ],
      lights: [{ kind: "point" as const, position: [1, 2, 3] as [number, number, number], intensity: [4, 5, 6] as [number, number, number], falloff: { decay: 2, cutoff: 30 } }],
      environment: { sh },
    };
    const layout = vdbReferenceTableLayout(scene);
    expect(layout.tables.width).toBe(mie.size);
    expect(layout.tables.height).toBe(2); // one Mie row (shared), one blackbody row
    const row = (r: number, i: number) => Array.from(layout.tables.data.subarray((r * layout.tables.width + i) * 4, (r * layout.tables.width + i) * 4 + 4));
    expect(row(0, 7)).toEqual([mie.values[21]!, mie.values[22]!, mie.values[23]!, mie.cdf[7]!]);
    expect(row(1, 3)).toEqual([bb.values[9]!, bb.values[10]!, bb.values[11]!, 0]);
    expect([layout.majorants.width, layout.majorants.height, layout.majorants.depth]).toEqual([5, 3, 6]);
    // g1 cell (1, 2, 3) at z 3; g2 cell (4, 0, 1) at z 4 + 1
    expect(layout.majorants.data[(3 * 3 + 2) * 5 + 1]).toBe(1 + 2 * 2 + 3 * 6);
    expect(layout.majorants.data[(5 * 3 + 0) * 5 + 4]).toBe(109);
    const buf = vdbReferencePackUniforms(scene, { width: 4, height: 4 }, layout);
    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    const O = VDB_REF_UNIFORM_OFFSETS;
    expect(u[O.flags / 4]! & 8).toBe(8);
    expect(Array.from(f.subarray(O.sh / 4 + 4 * 8, O.sh / 4 + 4 * 8 + 4))).toEqual([2.4, 2.5, 2.6, 0].map(Math.fround));
    const vol = (i: number) => (O.volumes + i * VDB_REF_VOLUME_BYTES) / 4;
    expect(Array.from(u.subarray(vol(0) + 36, vol(0) + 48))).toEqual([0, 2, 3, 4, 0, 0, 0, 0, 1, mie.size, 2, bb.size]);
    expect(Array.from(u.subarray(vol(1) + 36, vol(1) + 48))).toEqual([0, 5, 1, 2, 4, 0, 0, 0, 1, mie.size, 0, 0]);
    expect(Array.from(u.subarray(vol(2) + 36, vol(2) + 48))).toEqual([0, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0]);
    expect(f[vol(0) + 34]).toBeCloseTo(Math.log(bb.minTemperature), 5);
    expect(f[vol(0) + 35]).toBeCloseTo(Math.log(bb.maxTemperature), 5);
    const l = O.lights / 4;
    expect(Array.from(f.subarray(l + 12, l + 16))).toEqual([2, 30, 2, 0]);
  });
});
