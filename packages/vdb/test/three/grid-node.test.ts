/**
 * K2.1/K2.2 in Node, against three r187's own attribute bookkeeping
 * (`Attributes`, `WebGPUAttributeUtils`) on a stand-in backend: getDevice /
 * getGPUTexture, VDBGridNode binding, rebinding on a pool version change,
 * three's disposal paths never destroying the pool's buffer, the loud
 * compatibility check, option validation and module memoization. The shaders
 * themselves run on a GPU in e2e/three-grid-node.spec.ts.
 */

import { describe, expect, it } from "vitest";
import { VDBGridPool, VDBError, vdbModuleWgsl } from "../../src/core/index.js";
import { VDBGridNode, getDevice, getGPUTexture, vdbReadValue, vdbSample } from "../../src/three/index.js";
import { FakeDevice } from "../core/fake-gpu.js";
import { makeImage } from "../core/synthetic-image.js";

const scope = globalThis as unknown as Record<string, unknown>;
scope.GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
scope.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };

type ThreeAttributes = { update(attribute: object, type: number): void; has(attribute: object): boolean; dispose(): void };
const attributesModule = "three/src/renderers/common/Attributes.js";
const utilsModule = "three/src/renderers/webgpu/utils/WebGPUAttributeUtils.js";
const constantsModule = "three/src/renderers/common/Constants.js";
const { default: Attributes } = (await import(attributesModule)) as { default: new (backend: object, info: object) => ThreeAttributes };
const { default: WebGPUAttributeUtils } = (await import(utilsModule)) as {
  default: new (backend: object) => { createAttribute(attribute: object, usage: number): void; destroyAttribute(attribute: object): void };
};
const { AttributeType } = (await import(constantsModule)) as { AttributeType: { STORAGE: number } };

interface TestBuffer {
  label: string;
  destroyed: boolean;
  destroy(): void;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
}

function makeBuffer(label: string): TestBuffer {
  const b: TestBuffer = {
    label,
    destroyed: false,
    destroy: () => {
      b.destroyed = true;
    },
    getMappedRange: () => new ArrayBuffer(16),
    unmap: () => {},
  };
  return b;
}

/** A WebGPURenderer stand-in whose attribute paths are three's own (r187). */
function fakeRenderer(device?: object) {
  let data = new WeakMap<object, Record<string, unknown> & { buffer?: GPUBuffer }>();
  const created: TestBuffer[] = [];
  // A real fake device (pool tests), else a recorder of three's / the destroy guard's buffers.
  device ??= {
    createBuffer: (d: GPUBufferDescriptor) => {
      const b = makeBuffer(d.label ?? "");
      created.push(b);
      return b;
    },
  };
  const backend = {
    isWebGPUBackend: true,
    device,
    get(object: object) {
      let entry = data.get(object);
      if (!entry) data.set(object, (entry = {}));
      return entry;
    },
    delete(object: object) {
      data.delete(object);
    },
    getBufferAttribute: (attribute: object) => attribute,
    createStorageAttribute(attribute: object) {
      utils.createAttribute(attribute, 0x80);
    },
    destroyAttribute(attribute: object) {
      utils.destroyAttribute(attribute);
    },
    forget() {
      data = new WeakMap();
    },
  };
  const utils = new WebGPUAttributeUtils(backend);
  const attributes = new Attributes(backend, { createStorageAttribute() {}, destroyAttribute() {} });
  const renderer = { backend, _attributes: attributes };
  /** What three does when a shader binds a storage node: make (or take) the attribute's buffer. */
  const use = (attribute: object) => attributes.update(attribute, AttributeType.STORAGE);
  return { renderer, backend, device, created, attributes, use };
}

/** The pool surface VDBGridNode reads, as a plain object. */
function fakePool(device: object, types: string[] = ["float"]) {
  return {
    device: device as GPUDevice,
    buffer: makeBuffer("pool v1") as unknown as GPUBuffer,
    version: 1,
    layoutVersion: 1,
    types: new Set(types) as ReadonlySet<never>,
  };
}

const code = (fn: () => unknown): string | undefined => {
  try {
    fn();
  } catch (e) {
    return e instanceof VDBError ? e.code : `not a VDBError: ${String(e)}`;
  }
  return undefined;
};

const boundBuffer = (backend: { get(o: object): { buffer?: unknown } }, grids: VDBGridNode) => backend.get(grids.storage.value as object).buffer;

describe("getDevice / getGPUTexture (K2.1)", () => {
  it("returns the renderer's device and a texture's GPUTexture; fails clearly otherwise", () => {
    const { renderer, backend, device } = fakeRenderer();
    expect(getDevice(renderer)).toBe(device);

    const tex = { name: "rt", uuid: "u1" };
    const gpuTexture = {} as GPUTexture;
    backend.get(tex).texture = gpuTexture;
    expect(getGPUTexture(renderer, tex)).toBe(gpuTexture);
    expect(() => getGPUTexture(renderer, { uuid: "u2" })).toThrow(/no GPUTexture behind "u2".*initTexture/);

    // Before `await renderer.init()` the WebGPU backend's device is null.
    const uninit = { backend: { ...backend, device: null } };
    expect(() => getDevice(uninit)).toThrow("requires an initialized WebGPURenderer (await renderer.init())");
    expect(code(() => getDevice(uninit))).toBe("VDB_UNSUPPORTED_DEVICE");
    // The WebGL fallback backend.
    expect(() => getDevice({ backend: { isWebGLBackend: true } })).toThrow(/requires an initialized WebGPURenderer/);
    expect(() => getGPUTexture({ backend: {} }, tex)).toThrow(/getGPUTexture requires an initialized WebGPURenderer/);
  });
});

describe("VDBGridNode (K2.2)", () => {
  it("binds the pool buffer, rebinds on a new version, never destroys pool buffers", () => {
    const { renderer, backend, use, created } = fakeRenderer();
    const pool = fakePool(backend.device);
    const grids = new VDBGridNode(renderer, pool);
    expect(boundBuffer(backend, grids)).toBe(pool.buffer);
    expect(grids.version).toBe(1);
    expect(grids.update()).toBe(false);

    use(grids.storage.value as object);
    expect(created).toHaveLength(0); // three took the pool's buffer as made
    expect(grids.update()).toBe(false); // verified, nothing changed

    const first = grids.storage.value;
    const old = pool.buffer;
    pool.buffer = makeBuffer("pool v2") as unknown as GPUBuffer;
    pool.version = 2;
    expect(grids.update()).toBe(true);
    expect(grids.storage.value).not.toBe(first);
    expect(boundBuffer(backend, grids)).toBe(pool.buffer);
    expect(grids.version).toBe(2);
    expect(grids.update()).toBe(false);
    // The old stand-in was retired through three (Attributes.delete -> destroyAttribute): a throwaway died, not the pool's buffer.
    expect((old as unknown as TestBuffer).destroyed).toBe(false);
    expect(created.map((b) => [b.label, b.destroyed])).toEqual([["@pmndrs/vdb/three retired stand-in", true]]);

    // A rollback (version bump back to an older buffer object) rebinds too.
    pool.buffer = old;
    pool.version = 3;
    expect(grids.update()).toBe(true);
    expect(boundBuffer(backend, grids)).toBe(old);

    grids.dispose();
    grids.dispose();
    expect((pool.buffer as unknown as TestBuffer).destroyed).toBe(false);
    expect(code(() => grids.update())).toBe("VDB_DISPOSED");
  });

  it("survives renderer.dispose() (Attributes.dispose) and a stand-in's dispose()", () => {
    const { renderer, backend, use, attributes, created } = fakeRenderer();
    const pool = fakePool(backend.device);
    const a = new VDBGridNode(renderer, pool);
    const b = new VDBGridNode(renderer, pool, { name: "vdbPoolB" });
    use(a.storage.value as object);
    use(b.storage.value as object);
    expect(created).toHaveLength(0);

    (a.storage.value as { dispose(): void }).dispose();
    expect((pool.buffer as unknown as TestBuffer).destroyed).toBe(false);
    expect(created).toHaveLength(1);

    attributes.dispose(); // what renderer.dispose() runs
    expect((pool.buffer as unknown as TestBuffer).destroyed).toBe(false);
    expect(created).toHaveLength(2);
    expect(created.every((t) => t.destroyed)).toBe(true);
  });

  it("follows a real VDBGridPool growth on a fake device", async () => {
    const device = new FakeDevice();
    const { renderer, backend } = fakeRenderer(device.asGPU());
    const pool = await VDBGridPool.create(device.asGPU(), { initialBytes: 16384 });
    pool.add(makeImage({ bytes: 1024, seed: 1 }));
    pool.flush();
    await pool.settled();
    const grids = new VDBGridNode(renderer, pool);
    expect(grids.options.types).toEqual(["float"]);
    expect(boundBuffer(backend, grids)).toBe(pool.buffer);

    pool.add(makeImage({ bytes: 1024, seed: 2 }));
    pool.flush(); // fits: same buffer, revision only
    expect(grids.update()).toBe(false);

    const before = pool.version;
    pool.add(makeImage({ bytes: 16384, seed: 3 }));
    pool.flush(); // grows
    expect(pool.version).toBeGreaterThan(before);
    expect(grids.update()).toBe(true);
    expect(boundBuffer(backend, grids)).toBe(pool.buffer);
    expect(grids.version).toBe(pool.version);
  });

  it("throws a clear error when three bound something else to a stand-in it used", () => {
    const { renderer, backend, use } = fakeRenderer();
    const pool = fakePool(backend.device);
    const grids = new VDBGridNode(renderer, pool);
    // Not used by three yet: nothing to check.
    backend.get(grids.storage.value as object).buffer = undefined;
    expect(grids.update()).toBe(false);
    // A three that looks the buffer up elsewhere makes its own and binds that.
    backend.forget();
    use(grids.storage.value as object);
    expect(() => grids.update()).toThrow(/three r\d+ did not bind the pool buffer.*unsupported/);

    // Or keeps ours in .buffer but made another one.
    const second = new VDBGridNode(renderer, pool);
    use(second.storage.value as object);
    backend.get(second.storage.value as object).gpuBuffer = makeBuffer("three's own");
    expect(code(() => second.update())).toBe("VDB_UNSUPPORTED_DEVICE");

    // Internals missing at construction.
    expect(() => new VDBGridNode({ backend: { ...backend, destroyAttribute: undefined }, _attributes: renderer._attributes } as never, pool)).toThrow(
      /no backend.destroyAttribute.*r187/,
    );
    expect(() => new VDBGridNode({ backend } as never, pool)).toThrow(/no renderer._attributes/);
  });

  it("validates options and the pool", () => {
    const { renderer, backend } = fakeRenderer();
    const pool = fakePool(backend.device, ["float", "fp8"]);
    const make = (options: object, p: object = pool) => code(() => new VDBGridNode(renderer, p as never, options as never));
    expect(make({})).toBeUndefined();
    expect(make({ slot: { group: 0, firstBinding: 0 } })).toBe("VDB_BAD_OPTION");
    expect(make({ buffer: "x.value" })).toBe("VDB_BAD_OPTION");
    expect(make({ profiel: "sample" })).toBe("VDB_BAD_OPTION");
    expect(make({ profile: "majorant" })).toBe("VDB_BAD_OPTION");
    expect(make({ profile: "nope" })).toBe("VDB_BAD_OPTION");
    expect(make({ types: ["vec3"] })).toBe("VDB_BAD_OPTION");
    expect(make({ types: [] })).toBe("VDB_BAD_OPTION");
    expect(make({ sampler: "cubic" })).toBe("VDB_BAD_OPTION");
    expect(make({ name: "set" })).toBe("VDB_BAD_OPTION"); // a WGSL reserved word
    expect(make({ name: "my pool" })).toBe("VDB_BAD_OPTION");
    expect(make({}, fakePool(backend.device, []))).toBe("VDB_BAD_OPTION"); // nothing to compile
    expect(make({ types: ["float"] }, fakePool(backend.device, []))).toBeUndefined();
    expect(make({}, { ...pool, device: {} })).toBe("VDB_BAD_OPTION"); // another device
    expect(make({}, { ...pool, layoutVersion: 2 })).toBe("VDB_LAYOUT_MISMATCH");
    expect(make({ types: ["float"] }, { ...pool, layoutVersion: 2 })).toBe("VDB_LAYOUT_MISMATCH");
    expect(code(() => new VDBGridNode({ backend: { isWebGPUBackend: true, device: null } } as never, pool))).toBe("VDB_UNSUPPORTED_DEVICE");

    // Explicit types are the caller's choice; check() reports grids they would read as 0.
    const floatOnly = new VDBGridNode(renderer, pool, { types: ["float"] });
    expect(code(() => floatOnly.check())).toBe("VDB_TYPE_NOT_COMPILED");
    expect(code(() => new VDBGridNode(renderer, pool).check())).toBeUndefined();

    // vdbSample needs sample+; vdbReadValue works on read.
    const read = new VDBGridNode(renderer, pool, { profile: "read" });
    expect(code(() => vdbSample(read, 0, null))).toBe("VDB_BAD_OPTION");
    expect(code(() => vdbSample(new VDBGridNode(renderer, pool), 0, null, { space: "object" as never }))).toBe("VDB_BAD_OPTION");
    expect(code(() => vdbReadValue(read, -1, null))).toBe("VDB_BAD_OPTION");
  });

  it("generates the module through the storage node's reference, memoized per reference", () => {
    const { renderer, backend } = fakeRenderer();
    const pool = fakePool(backend.device, ["fp8", "float"]);
    const grids = new VDBGridNode(renderer, pool, { profile: "march", outside: "zero" });
    expect(grids.options).toEqual({ types: ["float", "fp8"], profile: "march", outside: "zero" });
    const mod = grids.module as unknown as { moduleFor(ref: string): string; vdbPool: unknown; isCodeNode: boolean; language: string };
    expect(mod.isCodeNode).toBe(true);
    expect(mod.language).toBe("wgsl");
    expect(mod.vdbPool).toBe(grids.storage); // a node property: three emits the binding with the module

    const a = mod.moduleFor("vdbPool.value");
    expect(mod.moduleFor("vdbPool.value")).toBe(a);
    expect(a).toBe(vdbModuleWgsl({ buffer: "vdbPool.value", types: ["float", "fp8"], profile: "march", outside: "zero" }));
    expect(a).not.toMatch(/@group|@binding/);
    expect(a.match(/vdbPool\.value\[/g)).toHaveLength(1); // the one access site

    const b = mod.moduleFor("vdbPool_1.value");
    expect(b).not.toBe(a);
    expect(b.replace("vdbPool_1.value[", "vdbPool.value[")).toBe(a);
  });

  it("makes each TSL helper's wgslFn once per node", () => {
    const { renderer, backend } = fakeRenderer();
    const grids = new VDBGridNode(renderer, fakePool(backend.device));
    const call = vdbSample(grids, 3, null) as { isNode?: boolean; functionNode?: unknown };
    expect(call.isNode).toBe(true);
    expect((vdbSample(grids, 0, null) as typeof call).functionNode).toBe(call.functionNode);
    expect((vdbSample(grids, 0, null, { space: "index" }) as typeof call).functionNode).not.toBe(call.functionNode);
    expect((vdbReadValue(grids, 0, null) as typeof call).isNode).toBe(true);
    expect(grids.helper("readValue")).toBe(grids.helper("readValue"));
  });
});
