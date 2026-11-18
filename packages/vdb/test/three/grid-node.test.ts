/**
 * K2.1/K2.2 in Node, against three r187's own attribute bookkeeping
 * (`Attributes`, `WebGPUAttributeUtils`) on a stand-in backend: getDevice /
 * getGPUTexture, VDBGridNode binding, rebinding on a pool version change,
 * three's disposal paths never destroying the pool's buffer, the loud
 * compatibility check, option validation and module memoization. The shaders
 * themselves run on a GPU in e2e/three-grid-node.spec.ts.
 */

import { describe, expect, it, vi } from "vitest";
import { VDBGridPool, VDBError, vdbModuleWgsl } from "../../src/core/index.js";
import { VDBGridNode, getDevice, getGPUTexture, vdbReadValue, vdbSample } from "../../src/three/index.js";
import { resetThreeRevisionWarning, warnThreeRevision } from "../../src/three/grid-node.js";
import { FakeDevice } from "../core/fake-gpu.js";
import { makeImage } from "../core/synthetic-image.js";
import { SELF_TEST_LABEL, fakeRenderer, makeBuffer, type TestBuffer } from "./fake-three.js";

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
    const { renderer, backend, use, created: all } = fakeRenderer();
    const pool = fakePool(backend.device);
    const grids = new VDBGridNode(renderer, pool);
    // The self-test: its buffer, and the throwaway the guard swapped in for it; both destroyed.
    expect(all.map((b) => [b.label, b.destroyed])).toEqual([
      [SELF_TEST_LABEL, true],
      ["@pmndrs/vdb/three retired stand-in", true],
    ]);
    const made = () => all.slice(2); // buffers after the self-test's
    expect(boundBuffer(backend, grids)).toBe(pool.buffer);
    expect(grids.version).toBe(1);
    expect(grids.update()).toBe(false);

    use(grids.storage.value as object);
    expect(made()).toHaveLength(0); // three took the pool's buffer as made
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
    expect(made().map((b) => [b.label, b.destroyed])).toEqual([["@pmndrs/vdb/three retired stand-in", true]]);

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
    const { renderer, backend, use, attributes, created: all } = fakeRenderer();
    const pool = fakePool(backend.device);
    const a = new VDBGridNode(renderer, pool);
    const b = new VDBGridNode(renderer, pool, { name: "vdbPoolB" });
    const made = () => all.slice(2); // buffers after the self-test's
    use(a.storage.value as object);
    use(b.storage.value as object);
    expect(made()).toHaveLength(0);

    (a.storage.value as { dispose(): void }).dispose();
    expect((pool.buffer as unknown as TestBuffer).destroyed).toBe(false);
    expect(made()).toHaveLength(1);

    attributes.dispose(); // what renderer.dispose() runs
    expect((pool.buffer as unknown as TestBuffer).destroyed).toBe(false);
    expect(made()).toHaveLength(2);
    expect(made().every((t) => t.destroyed)).toBe(true);
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
    expect(code(() => second.update())).toBe("VDB_ENGINE_MISMATCH");

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

describe("VDBGridNode guards (K2 review)", () => {
  /** The module node's per-shader check, called the way `generate` calls it first. */
  type ModuleNode = { checkBuild(builder: object): void; generate(builder: object): string; vdbPool: { build: unknown } };
  const moduleOf = (grids: VDBGridNode) => grids.module as unknown as ModuleNode;

  it("refuses a shader another renderer builds (two renderers, one device): one VDBGridNode per renderer", () => {
    const a = fakeRenderer();
    const b = fakeRenderer(a.device);
    const pool = fakePool(a.device);
    const grids = new VDBGridNode(a.renderer, pool);
    expect(grids.renderer).toBe(a.renderer);
    const mod = moduleOf(grids);
    expect(() => mod.checkBuild({ renderer: b.renderer })).toThrow(/another renderer.*one VDBGridNode per renderer/);
    expect(code(() => mod.checkBuild({ renderer: b.renderer }))).toBe("VDB_BAD_OPTION");
    expect(code(() => mod.checkBuild({ renderer: a.renderer }))).toBeUndefined();
    // generate() checks before it builds anything (the stubbed build would throw "built").
    mod.vdbPool.build = () => {
      throw new Error("built");
    };
    expect(code(() => mod.generate({ renderer: b.renderer }))).toBe("VDB_BAD_OPTION");
    expect(code(() => mod.generate({ renderer: a.renderer }))).toBe("not a VDBError: Error: built");
  });

  it("refuses a second VDB module in one shader; the same node twice is fine", () => {
    const { renderer, backend } = fakeRenderer();
    const pool = fakePool(backend.device);
    const one = moduleOf(new VDBGridNode(renderer, pool));
    const two = moduleOf(new VDBGridNode(renderer, pool, { name: "vdbPool2" }));
    const shader = { renderer };
    one.checkBuild(shader);
    one.checkBuild(shader); // same module again (another stage, another include)
    expect(() => two.checkBuild(shader)).toThrow(/two VDB modules.*One VDB module per shader/);
    expect(code(() => two.checkBuild(shader))).toBe("VDB_BAD_OPTION");
    expect(code(() => two.checkBuild({ renderer }))).toBeUndefined(); // another shader
  });

  it("self-tests three's attribute path once per backend before binding the pool", () => {
    const { renderer, backend, created } = fakeRenderer();
    const pool = fakePool(backend.device);
    new VDBGridNode(renderer, pool);
    expect(created.map((b) => b.label)).toEqual([SELF_TEST_LABEL, "@pmndrs/vdb/three retired stand-in"]);
    expect(created.every((b) => b.destroyed)).toBe(true); // the dummy is destroyed by the test itself, after it passed
    new VDBGridNode(renderer, pool);
    expect(created).toHaveLength(2); // cached per backend
    new VDBGridNode(fakeRenderer(backend.device).renderer, pool);
    expect(created.filter((b) => b.label === SELF_TEST_LABEL)).toHaveLength(2); // another backend, another test
  });

  it("throws VDB_ENGINE_MISMATCH before exposing the pool buffer when three destroys behind the guard", () => {
    const { renderer, backend, attributes, utils, created } = fakeRenderer();
    // A three whose Attributes.delete destroys the buffer itself instead of calling backend.destroyAttribute.
    attributes.delete = (attribute: object) => {
      utils.destroyAttribute(attribute);
      return null;
    };
    const records: object[] = [];
    const get = backend.get.bind(backend);
    backend.get = (object: object) => {
      records.push(object);
      return get(object);
    };
    const pool = fakePool(backend.device);
    expect(() => new VDBGridNode(renderer, pool)).toThrow(/destroyed a storage attribute's GPUBuffer without going through backend.destroyAttribute/);
    expect(code(() => new VDBGridNode(renderer, pool))).toBe("VDB_ENGINE_MISMATCH"); // cached
    expect(created.filter((b) => b.label === SELF_TEST_LABEL)).toHaveLength(1);
    // The pool's buffer never reached a backend record, and was not destroyed.
    expect(records.some((o) => get(o).buffer === pool.buffer)).toBe(false);
    expect((pool.buffer as unknown as TestBuffer).destroyed).toBe(false);
  });

  it("throws VDB_ENGINE_MISMATCH when three makes its own buffer for a stand-in", () => {
    const { renderer, backend } = fakeRenderer();
    backend.createStorageAttribute = (attribute: object) => {
      backend.get(attribute).buffer = makeBuffer("three's own") as unknown as GPUBuffer;
    };
    expect(() => new VDBGridNode(renderer, fakePool(backend.device))).toThrow(/made its own GPUBuffer/);
  });

  it("warns once about an unsupported three revision", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      resetThreeRevisionWarning();
      expect(warnThreeRevision("187")).toBe(false);
      expect(warnThreeRevision("188")).toBe(true);
      expect(warnThreeRevision("189")).toBe(false);
      expect(warn).toHaveBeenCalledTimes(1);
      expect(String(warn.mock.calls[0]![0])).toMatch(/three r188 is outside the supported range \(r187/);
    } finally {
      resetThreeRevisionWarning();
      warn.mockRestore();
    }
  });

  it("warns once per backend when the renderer owns its device (renderer.dispose() destroys the pool's buffer)", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const own = fakeRenderer(undefined, { parameters: {} });
      const pool = fakePool(own.device);
      new VDBGridNode(own.renderer, pool);
      new VDBGridNode(own.renderer, pool);
      expect(warn).toHaveBeenCalledTimes(1);
      expect(String(warn.mock.calls[0]![0])).toMatch(/created its own GPUDevice.*renderer.dispose\(\) destroys/);
      const given = fakeRenderer(undefined, { parameters: { device: {} } });
      new VDBGridNode(given.renderer, fakePool(given.device));
      const plain = fakeRenderer(); // no parameters: nothing to tell
      new VDBGridNode(plain.renderer, fakePool(plain.device));
      expect(warn).toHaveBeenCalledTimes(1);
    } finally {
      warn.mockRestore();
    }
  });
});
