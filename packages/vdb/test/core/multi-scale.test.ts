import { beforeEach, describe, expect, it } from "vitest";
import { VDBError } from "../../src/core/index.js";
import {
  VDB_FRAME_MULTI_SCALE_PARAMS_BYTES,
  multiScaleMipsWgsl,
  vdbCreateMultiScaleMips,
  vdbFrameMultiScaleParams,
  vdbFrameMultiScaleWgsl,
  vdbModuleWgsl,
} from "../../src/core/internal.js";
import { computeFake, type FakePass } from "./fake-compute.js";
import type { FakeDevice } from "./fake-gpu.js";

// K7.3: the multi-scale mip chain's host-side contract on the recording fake (validation, what it records, the
// retire rule, ownership) and the lookup WGSL text. The kernels run against the tools twins in e2e/k7-multi-scale.

interface FakeTexture {
  label: string;
  format: string;
  dimension: string;
  width: number;
  height: number;
  depthOrArrayLayers: number;
  mipLevelCount: number;
  usage: number;
  destroyed: boolean;
  createView(d?: object): object;
  destroy(): void;
}

let fake: FakeDevice;
let device: GPUDevice;
let passes: FakePass[];
let textures: FakeTexture[];
let textureCopies: { mipLevel: number; size: number[]; bytesPerRow: number }[];

function makeTexture(d: { label?: string; size: number[]; dimension?: string; format: string; mipLevelCount?: number; usage: number }): FakeTexture {
  const t: FakeTexture = {
    label: d.label ?? "",
    format: d.format,
    dimension: d.dimension ?? "2d",
    width: d.size[0]!,
    height: d.size[1] ?? 1,
    depthOrArrayLayers: d.size[2] ?? 1,
    mipLevelCount: d.mipLevelCount ?? 1,
    usage: d.usage,
    destroyed: false,
    createView: (v?: object) => ({ texture: t, ...v }),
    destroy: () => {
      t.destroyed = true;
    },
  };
  return t;
}

beforeEach(() => {
  ({ fake, device, passes } = computeFake());
  textures = [];
  textureCopies = [];
  const f = fake as unknown as Record<string, unknown>;
  f.createTexture = (d: Parameters<typeof makeTexture>[0]) => {
    const t = makeTexture(d);
    textures.push(t);
    return t;
  };
  const encProto = Object.getPrototypeOf(fake.createCommandEncoder()) as Record<string, unknown>;
  encProto.copyBufferToTexture = (src: { bytesPerRow: number }, dst: { mipLevel: number }, size: number[]) => {
    textureCopies.push({ mipLevel: dst.mipLevel, size, bytesPerRow: src.bytesPerRow });
  };
});

const code = (e: unknown) => (e instanceof VDBError ? e.code : String(e));
function expectCode(fn: () => unknown, c: string, message?: RegExp): void {
  let err: unknown;
  try {
    fn();
  } catch (e) {
    err = e;
  }
  expect(code(err)).toBe(c);
  if (message) expect((err as Error).message).toMatch(message);
}

const source = () => makeTexture({ size: [64, 64, 128], dimension: "3d", format: "rg16float", usage: 0x4 | 0x10 });
const encoder = () => fake.createCommandEncoder() as unknown as GPUCommandEncoder;
const submit = (e: GPUCommandEncoder) => fake.queue.submit([(e as unknown as { finish(): { copies: never[] } }).finish()]);

describe("vdbCreateMultiScaleMips: options and records", () => {
  it("picks the format by channel count, allocates one texture (levels − 1 mips) and its scratch", () => {
    const m = vdbCreateMultiScaleMips(device, { size: [64, 64, 64] });
    expect([m.format, m.channels, m.layout.levels]).toEqual(["rg16float", 2, 6]);
    expect(textures.length).toBe(1);
    expect([textures[0]!.width, textures[0]!.height, textures[0]!.depthOrArrayLayers, textures[0]!.mipLevelCount, textures[0]!.dimension]).toEqual([32, 32, 32, 5, "3d"]);
    // Levels 1–5: 32³ + 16³ + 8³ + 4³ + 2³ texels × 4 bytes (rg16float).
    expect(m.gpuBytes.textures).toBe((32 ** 3 + 16 ** 3 + 8 ** 3 + 4 ** 3 + 2 ** 3) * 4);
    expect(vdbCreateMultiScaleMips(device, { size: [8, 8, 8], channels: [1] }).format).toBe("r16float");
    expect(vdbCreateMultiScaleMips(device, { size: [8, 8, 8], channels: [0, 1, 2] }).format).toBe("rgba16float");
    expect(vdbCreateMultiScaleMips(device, { size: [8, 8, 8], format: "rgba16float" }).format).toBe("rgba16float");
  });

  it("rejects bad options with VDB_BAD_OPTION before allocating", () => {
    const before = textures.length + fake.buffers.length;
    expectCode(() => vdbCreateMultiScaleMips({} as GPUDevice, { size: [8, 8, 8] }), "VDB_BAD_OPTION", /GPUDevice/);
    expectCode(() => vdbCreateMultiScaleMips(device, { size: [8, 8, 0] }), "VDB_BAD_OPTION", /size must be 3 positive integers/);
    expectCode(() => vdbCreateMultiScaleMips(device, { size: [8, 8, 8], levels: 9 }), "VDB_BAD_OPTION", /levels/);
    expectCode(() => vdbCreateMultiScaleMips(device, { size: [8, 8, 8], channels: [4] }), "VDB_BAD_OPTION", /channels/);
    expectCode(() => vdbCreateMultiScaleMips(device, { size: [8, 8, 8], format: "r16float" }), "VDB_BAD_OPTION", /holds 1 channel/);
    expectCode(() => vdbCreateMultiScaleMips(device, { size: [8, 8, 8], edge: "mirror" as "zero" }), "VDB_BAD_OPTION", /edge/);
    expect(textures.length + fake.buffers.length).toBe(before);
  });

  it("records levels then packs in one pass, one copy per mip; checks the source and its region", () => {
    const m = vdbCreateMultiScaleMips(device, { size: [64, 64, 64] });
    const e = encoder();
    m.record(e, source() as unknown as GPUTexture, { origin: [0, 0, 64] });
    expect(passes.length).toBe(1);
    expect(textureCopies.map((c) => c.mipLevel)).toEqual([0, 1, 2, 3, 4]);
    expect(textureCopies.every((c) => c.bytesPerRow % 256 === 0)).toBe(true);
    expect(textureCopies[0]!.size).toEqual([32, 32, 32]);
    expectCode(() => m.record(encoder(), source() as unknown as GPUTexture, { origin: [0, 0, 65] }), "VDB_BAD_OPTION", /outside the source/);
    const flat = makeTexture({ size: [64, 64, 64], format: "rg16float", usage: 0x4 });
    expectCode(() => m.record(encoder(), flat as unknown as GPUTexture), "VDB_BAD_OPTION", /3D texture/);
    const ints = makeTexture({ size: [64, 64, 64], dimension: "3d", format: "rg32uint", usage: 0x4 });
    expectCode(() => m.record(encoder(), ints as unknown as GPUTexture), "VDB_BAD_OPTION", /float format/);
  });
});

describe("vdbCreateMultiScaleMips: retire rule and ownership", () => {
  it("resize retires the old texture: destroyed after two more records and the queue's done, version bumps", async () => {
    const m = vdbCreateMultiScaleMips(device, { size: [32, 32, 32] });
    const src = source() as unknown as GPUTexture;
    m.record(encoder(), src);
    const old = textures[0]!;
    const oldView = m.view;
    m.resize([64, 64, 64]);
    expect(m.version).toBe(1);
    expect(m.view).not.toBe(oldView);
    expect(m.texture).not.toBe(old as unknown as GPUTexture);
    const e1 = encoder();
    m.record(e1, src);
    submit(e1);
    await fake.completeWork();
    expect(old.destroyed).toBe(false); // the frame in flight may still bind the old view
    const e2 = encoder();
    m.record(e2, src);
    submit(e2);
    expect(old.destroyed).toBe(false); // asked the queue, not yet answered
    await fake.completeWork();
    expect(old.destroyed).toBe(true);
    expect((m.texture as unknown as FakeTexture).destroyed).toBe(false);
    m.destroy();
    expect((m.texture as unknown as FakeTexture).destroyed).toBe(true);
    expect(fake.buffers.every((b) => b.destroyed)).toBe(true);
    expectCode(() => m.record(encoder(), src), "VDB_DISPOSED");
  });

  it("destroy frees retired resources at once; a host target is never destroyed and must match", () => {
    const m = vdbCreateMultiScaleMips(device, { size: [16, 16, 16] });
    m.resize([32, 32, 32]);
    m.destroy();
    expect(textures.every((t) => t.destroyed)).toBe(true);

    const target = makeTexture({ size: [8, 8, 8], dimension: "3d", format: "rg16float", mipLevelCount: 3, usage: 0x2 | 0x4 });
    const t = vdbCreateMultiScaleMips(device, { size: [16, 16, 16], levels: 4, target: target as unknown as GPUTexture });
    expect(t.texture).toBe(target as unknown as GPUTexture);
    expectCode(() => t.resize([32, 32, 32]), "VDB_BAD_OPTION", /host target/);
    t.destroy();
    expect(target.destroyed).toBe(false);
    const wrong = makeTexture({ size: [8, 8, 8], dimension: "3d", format: "rg16float", mipLevelCount: 2, usage: 0x2 });
    expectCode(() => vdbCreateMultiScaleMips(device, { size: [16, 16, 16], levels: 4, target: wrong as unknown as GPUTexture }), "VDB_BAD_OPTION", /3 mip levels/);
  });
});

describe("the WGSL", () => {
  it("build kernels and lookup are text outside the VDB module: default module bytes do not change", () => {
    const before = vdbModuleWgsl({ slot: { group: 0, firstBinding: 0 }, types: ["float"], profile: "march" });
    const lookup = vdbFrameMultiScaleWgsl();
    expect(vdbModuleWgsl({ slot: { group: 0, firstBinding: 0 }, types: ["float"], profile: "march" })).toBe(before);
    expect(before).not.toContain("MultiScale");
    for (const name of ["struct VdbFrameMultiScale", "fn vdbFrameMultiScaleSample(", "fn vdbFrameMultiScaleLight("]) expect(lookup).toContain(name);
    // Every module-scope symbol of the lookup is vdb-prefixed (one shader can hold it next to the VDB module).
    const symbols = [...lookup.matchAll(/^(?:fn|struct|const|var|override)\s+([A-Za-z_][A-Za-z0-9_]*)/gm)].map((x) => x[1]!);
    expect(symbols.length).toBe(3);
    for (const s of symbols) expect(s).toMatch(/^(vdb|Vdb)/);
    expect(multiScaleMipsWgsl()).toContain("fn csMsLevel(");
    // Channel options: base .xz, one-channel mips read no absorption.
    const custom = vdbFrameMultiScaleWgsl({ baseChannels: [0, 2], mipChannels: 1 });
    expect(custom).toContain("vec2f(s.x, s.z)");
    expect(custom).toContain("vec2f(s.x, 0.0)");
    expectCode(() => vdbFrameMultiScaleWgsl({ baseChannels: [0, 1, 2] }), "VDB_BAD_OPTION");
  });

  it("vdbFrameMultiScaleParams packs the WGSL struct (64 bytes)", () => {
    const b = vdbFrameMultiScaleParams({ boxMin: [1, 2, 3], boxSize: [4, 5, 6], unit: 0.5, levels: 6, multipleScattering: 0.2, baseOrigin: [0, 0, 64], baseSize: [64, 64, 64] });
    expect(b.byteLength).toBe(VDB_FRAME_MULTI_SCALE_PARAMS_BYTES);
    const f = new Float32Array(b);
    const u = new Uint32Array(b);
    expect([...f.slice(0, 3), u[3], ...f.slice(4, 8), ...f.slice(8, 12), ...f.slice(12, 15)]).toEqual([1, 2, 3, 6, 4, 5, 6, 0.5, 0, 0, 64, Math.fround(0.2), 64, 64, 64]);
  });
});
