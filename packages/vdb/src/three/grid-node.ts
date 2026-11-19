/**
 * `VDBGridNode`: a `VDBGridPool` inside three's node system (task K2.2,
 * docs/dev/design/K1-core.md §6). Follows `@pmndrs/bvh`'s `src/three/trace-nodes.ts`.
 *
 *   const grids = new VDBGridNode(renderer, pool, { profile: 'sample' });
 *   // TSL:
 *   const density = vdbSample(grids, gridId, positionWorld);
 *   // or WGSL, calling the generated module by name (one module per shader):
 *   const shade = wgslFn(`fn shade(g: u32, p: vec3f) -> f32 {
 *       var a = vdbAccessorInit(g);
 *       return vdbSample(&a, vdbWorldToIndex(g, p));
 *   }`, [grids.module]);
 *   // each frame, after pool.update(encoder) was submitted (or pool.flush()):
 *   grids.update();
 *
 * **Binding.** `grids.storage` is a read-only three storage node named
 * `vdbPool` (`storage(standIn, 'uint', 0).toReadOnly().setName('vdbPool')`).
 * Its `StorageBufferAttribute` is a stand-in: before three first sees it, the
 * backend's record for it gets `buffer = pool.buffer`, and three's
 * `createAttribute` only makes a buffer when that field is unset. So three
 * binds the pool's buffer as is: no copy, no CPU write, one binding (D10).
 * three emits `struct vdbPoolStruct { value: array<u32> }` at a group and
 * binding it chooses; the module reads it through the storage node's property
 * reference (`vdbPool.value`), its one access site (the generator's `buffer`
 * mode). The name is fixed, so shader text is the same on every load.
 *
 * **Module.** `grids.module` is a `CodeNode` holding the storage node as a node
 * property, so three sets it up as a child and emits its binding in any shader
 * that includes the module. Its WGSL is `vdbModuleWgsl({ ...options, buffer:
 * <the storage node's property reference> })`, memoized per reference.
 * `wgslFn` parses only its first `fn`; every helper lives in this module.
 *
 * **Rebinding.** `update()` compares `pool.version` (and the buffer object). When
 * the pool replaced its buffer (growth, compaction, rollback), the storage
 * node's value becomes a new stand-in bound to the new buffer and three makes
 * new bind groups at the next use. Nothing recompiles. `pool.revision` changes
 * (add, replace, remove, setTransform into the same buffer) need nothing.
 * Skipping `update()` after a growth leaves three binding the old buffer, which
 * the pool destroys a couple of updates later.
 *
 * **Disposal.** The pool owns its buffer. three tracks the stand-ins like its own
 * attributes, so `renderer.dispose()` or a stand-in's `dispose()` would destroy
 * the pool's buffer. `backend.destroyAttribute` is wrapped once per backend:
 * for a stand-in, a throwaway buffer takes the pool's place before three
 * destroys it. (A renderer that created its own device still destroys that
 * device in `renderer.dispose()`; see `device.ts`.)
 *
 * **One renderer, one module per shader.** A `VDBGridNode` binds through one
 * renderer's backend: build a second node for a second renderer (even on the
 * same device). A shader holds at most one VDB module, so every grid one
 * shader reads goes through one `VDBGridNode`. Both are checked when three
 * builds a shader (`VDB_BAD_OPTION`).
 *
 * This file reaches into three's WebGPU backend, which is not public API. It
 * is written against three r187 (peer range >=0.187.0 <0.188.0, D23), and
 * every internal it uses is listed in `threeInternals` and checked loudly:
 * another three revision warns once, and the first `VDBGridNode` on a backend
 * runs a self-test of the disposal path before any pool buffer reaches three
 * (`VDB_ENGINE_MISMATCH` if three would destroy a buffer behind the guard).
 */

import { CodeNode, REVISION, StorageBufferAttribute, type Node, type NodeBuilder } from "three/webgpu";
import { storage, uint, wgslFn } from "three/tsl";
import { VDBError } from "../core/errors.js";
import { vdbCheckModule, vdbModuleWgsl } from "../core/wgsl/generate.js";
import type { VDBModuleHelper, VDBModuleOptions, VDBModuleProfile } from "../core/wgsl/types.js";
import type { VDBGridTypeName } from "../core/types.js";
import { webgpuBackendOf, type VDBThreeRendererLike } from "./device.js";

/** What `VDBGridNode` needs from a pool; a `VDBGridPool` is one. */
export interface VDBGridNodePoolLike {
  readonly device: GPUDevice;
  /** Replaced only when `version` changes. */
  readonly buffer: GPUBuffer;
  readonly version: number;
  readonly layoutVersion: number;
  readonly types: Iterable<VDBGridTypeName>;
}

/** Options of {@link VDBGridNode}: the module options, minus the binding (the adapter owns it). */
export interface VDBGridNodeOptions {
  /** Grid types the module decodes; others read 0. Default: the pool's types now (`[...pool.types]`, checked). */
  readonly types?: Iterable<VDBGridTypeName> | undefined;
  /** `'read'`, `'sample'` (default) or `'march'`. `vdbSample` needs `sample`+. */
  readonly profile?: Exclude<VDBModuleProfile, "majorant"> | undefined;
  readonly sampler?: VDBModuleOptions["sampler"];
  readonly isActive?: VDBModuleOptions["isActive"];
  readonly outside?: VDBModuleOptions["outside"];
  /**
   * Opt-in WGSL helpers for the module (`'rayClipSupport'`, and with `march`
   * `'cellBoundTrilinear'`, `'hddaCellEnd'`). `VDBVolumeMaterial` needs
   * `'rayClipSupport'`. Default none.
   */
  readonly helpers?: Iterable<VDBModuleHelper> | undefined;
  /** Name of the storage binding three emits. Default `'vdbPool'`. */
  readonly name?: string | undefined;
}

/** The parts of three's `WebGPURenderer` this file uses (see {@link threeInternals}). */
interface RendererLike {
  readonly backend: BackendLike;
  readonly _attributes?: { has(attribute: object): boolean; update(attribute: object, type: number): void };
}

interface BackendLike {
  readonly isWebGPUBackend?: boolean;
  readonly device?: GPUDevice | null;
  /** The renderer's constructor parameters; `device` is undefined when the renderer made its own device. */
  readonly parameters?: { readonly device?: GPUDevice };
  get(object: object): Record<string, unknown> & { buffer?: GPUBuffer };
  destroyAttribute?(attribute: object): void;
}

/** A three storage node, as far as this file uses one. */
type StorageNode = Node & { value: unknown; setName(name: string): unknown; build(builder: NodeBuilder, output?: string): unknown };

/**
 * A TSL node, loosely typed: three's TSL types do not follow a value returned
 * through a `wgslFn`, so the helpers take and return this (as `@pmndrs/bvh` does).
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type VDBTslNode = any;

const unsupported = (what: string): VDBError =>
  new VDBError(
    "VDB_ENGINE_MISMATCH",
    `three adapter: three r${REVISION} ${what}. VDBGridNode binds the pool's buffer through three's WebGPU backend internals, ` +
      "written against three r187 (peer range >=0.187.0 <0.188.0); this three version is unsupported.",
  );

/** Stand-ins per backend whose buffer is a pool's: `destroyAttribute` swaps in a throwaway for them. */
const guarded = new WeakMap<BackendLike, WeakSet<object>>();
/** Self-test result per backend: true, or the error every later construction rethrows. */
const selfTested = new WeakMap<BackendLike, true | VDBError>();
/** Backends already warned about owning their device. */
const ownDeviceWarned = new WeakSet<BackendLike>();
/** The VDB module each shader build has generated (one per shader). */
const moduleOfBuild = new WeakMap<object, VDBModuleNode>();

/** three r187's `AttributeType.STORAGE` (`renderers/common/Constants.js`; not exported from `three/webgpu`). */
const ATTRIBUTE_TYPE_STORAGE = 3;
/** The three revision this adapter is written against. */
export const VDB_THREE_REVISION = "187";
/** Label of the self-test buffer (4 bytes, once per backend). */
export const VDB_THREE_SELF_TEST_LABEL = "@pmndrs/vdb/three self-test";

let revisionWarned = false;

/**
 * @internal Warns once per page when `revision` is not the three revision this
 * adapter is written against. Returns true when it warned.
 */
export function warnThreeRevision(revision: string): boolean {
  if (revision === VDB_THREE_REVISION || revisionWarned) return false;
  revisionWarned = true;
  console.warn(
    `@pmndrs/vdb/three: three r${revision} is outside the supported range (r${VDB_THREE_REVISION}; peer >=0.187.0 <0.188.0). ` +
      "VDBGridNode binds the pool's buffer through three's WebGPU backend internals; it self-tests them and throws " +
      "VDB_ENGINE_MISMATCH when they differ, but other internals may have changed too.",
  );
  return true;
}

/** @internal Resets the once-per-page revision warning (tests). */
export function resetThreeRevisionWarning(): void {
  revisionWarned = false;
}

/**
 * Every three internal VDBGridNode relies on, checked against three r187's source:
 *
 * 1. `backend.get(attribute).buffer` is a storage attribute's `GPUBuffer`.
 *    `WebGPUAttributeUtils.createAttribute` makes one only when it is undefined,
 *    and `WebGPUBindingUtils.createBindGroup` binds it for a storage binding
 *    (`backend.get(binding.attribute).buffer`).
 * 2. `backend.destroyAttribute(attribute)` destroys that buffer
 *    (`WebGPUAttributeUtils.destroyAttribute`). `Attributes.delete` calls it on
 *    the attribute's `dispose` event and for every tracked attribute in
 *    `renderer.dispose()` (`Attributes.dispose`).
 * 3. `renderer._attributes.has(attribute)` is true once three used the attribute.
 * 4. `Bindings` rebuilds a storage binding's bind group when the storage node's
 *    value (`NodeStorageBuffer.attribute`) is a different attribute.
 * 5. `renderer._attributes.update(attribute, AttributeType.STORAGE)` is how
 *    three first uses a storage attribute (`createStorageAttribute`, which
 *    keeps a buffer already in the record), and registers the attribute's
 *    `dispose` event to `Attributes.delete`.
 * 6. `NodeBuilder.renderer` is the renderer building the shader.
 *
 * The constructor checks 2, 3 and 5 exist, then (once per backend) runs
 * {@link threeInternals.selfTest}: a 4-byte buffer goes through 1, 5 and 2
 * as a stand-in would, and must come out undestroyed. `update()` checks 1 for
 * every stand-in three used (its buffer is still the pool's, and three made
 * no other). The module node checks 6 per shader.
 */
const threeInternals = {
  check(renderer: RendererLike): void {
    if (typeof renderer.backend.destroyAttribute !== "function") throw unsupported("has no backend.destroyAttribute");
    if (typeof renderer._attributes?.has !== "function" || typeof renderer._attributes.update !== "function") {
      throw unsupported("has no renderer._attributes (has/update)");
    }
  },
  /**
   * Puts a 4-byte buffer through three's attribute path exactly as a stand-in
   * goes (bind, guard, `_attributes.update(attr, STORAGE)`, `attr.dispose()`)
   * and checks three took it as made and never destroyed it. Its instance
   * `destroy` only records the call until the test ends. Cached per backend;
   * a failure is rethrown on every later construction on that backend.
   *
   * @throws VDBError `VDB_ENGINE_MISMATCH` when three bypasses the guard (or
   *   makes its own buffer, or keeps a disposed attribute).
   */
  selfTest(renderer: RendererLike): void {
    const backend = renderer.backend;
    const cached = selfTested.get(backend);
    if (cached === true) return;
    if (cached) throw cached;
    let error: VDBError | null = null;
    const dummy = backend.device!.createBuffer({ label: VDB_THREE_SELF_TEST_LABEL, size: 4, usage: 0x80 /* GPUBufferUsage.STORAGE */ });
    const realDestroy = dummy.destroy;
    let destroyedByThree = false;
    (dummy as { destroy: () => void }).destroy = () => {
      destroyedByThree = true;
    };
    try {
      const attribute = new StorageBufferAttribute(new Uint32Array(1), 1);
      attribute.name = VDB_THREE_SELF_TEST_LABEL;
      this.bind(backend, attribute, dummy);
      this.guardDestroy(backend, attribute);
      renderer._attributes!.update(attribute, ATTRIBUTE_TYPE_STORAGE);
      if (!this.usedByThree(renderer, attribute)) {
        error = unsupported("did not track a storage attribute in renderer._attributes after _attributes.update");
      } else if (this.bufferOf(backend, attribute) !== dummy || this.otherBuffer(backend, attribute, dummy)) {
        error = unsupported("made its own GPUBuffer for a storage attribute whose backend record already held one");
      }
      attribute.dispose(); // three: Attributes.delete -> backend.destroyAttribute (the guard swaps a throwaway in)
      if (!error && destroyedByThree) {
        error = unsupported("destroyed a storage attribute's GPUBuffer without going through backend.destroyAttribute (the destroy guard)");
      } else if (!error && this.usedByThree(renderer, attribute)) {
        error = unsupported("kept a disposed storage attribute in renderer._attributes");
      }
    } catch (e) {
      error = e instanceof VDBError ? e : unsupported(`failed the adapter's attribute self-test (${e instanceof Error ? e.message : String(e)})`);
    } finally {
      (dummy as { destroy: () => void }).destroy = realDestroy;
      dummy.destroy();
    }
    selfTested.set(backend, error ?? true);
    if (error) throw error;
  },
  bufferOf(backend: BackendLike, attribute: object): GPUBuffer | undefined {
    return backend.get(attribute).buffer;
  },
  bind(backend: BackendLike, attribute: object, buffer: GPUBuffer): void {
    backend.get(attribute).buffer = buffer;
  },
  usedByThree(renderer: RendererLike, attribute: object): boolean {
    return renderer._attributes!.has(attribute);
  },
  /** A GPUBuffer in the backend's record other than `ours` (three made its own). */
  otherBuffer(backend: BackendLike, attribute: object, ours: GPUBuffer): boolean {
    return Object.values(backend.get(attribute)).some(
      (value) => value !== ours && typeof (value as { getMappedRange?: unknown } | null)?.getMappedRange === "function",
    );
  },
  /** Makes `backend.destroyAttribute` spare `attribute`'s buffer: a throwaway takes its place first. */
  guardDestroy(backend: BackendLike, attribute: object): void {
    let set = guarded.get(backend);
    if (!set) {
      const ours = (set = new WeakSet());
      guarded.set(backend, ours);
      const destroy = backend.destroyAttribute!;
      (backend as { destroyAttribute: (target: object) => void }).destroyAttribute = function (this: BackendLike, target: object) {
        if (ours.has(target)) {
          ours.delete(target);
          backend.get(target).buffer = backend.device!.createBuffer({
            label: "@pmndrs/vdb/three retired stand-in",
            size: 16,
            usage: 0x80 /* GPUBufferUsage.STORAGE */,
          });
        }
        return destroy.call(this, target);
      };
    }
    set.add(attribute);
  },
};

/** Module options after validation, without the binding. */
type ModuleOptions = Omit<VDBModuleOptions, "slot" | "buffer" | "helpers"> & {
  readonly types: readonly VDBGridTypeName[];
  readonly profile: VDBModuleProfile;
  readonly helpers?: readonly VDBModuleHelper[];
};

const OPTION_KEYS = new Set(["types", "profile", "sampler", "isActive", "outside", "helpers", "name"]);

function bad(message: string): VDBError {
  return new VDBError("VDB_BAD_OPTION", `VDBGridNode: ${message}`);
}

/** The generated module as a three code node: its buffer reference is the storage node's, per shader. */
class VDBModuleNode extends CodeNode {
  /** Node-valued property: three sets it up and analyzes it as this node's child. */
  vdbPool: StorageNode;
  private readonly _options: ModuleOptions;
  private readonly _byRef = new Map<string, string>();
  /** The backend the stand-ins are bound through; a shader built by another renderer would bind nothing. */
  private readonly _backend: object;

  constructor(storageNode: StorageNode, options: ModuleOptions, backend: object) {
    super("", [], "wgsl");
    this.vdbPool = storageNode;
    this._options = options;
    this._backend = backend;
  }

  /**
   * The per-shader checks, before anything is emitted: the shader is built by
   * the renderer this node binds through, and it holds no other VDB module.
   */
  checkBuild(builder: NodeBuilder): void {
    const backend = (builder as unknown as { renderer?: { backend?: object } }).renderer?.backend;
    if (backend !== this._backend) {
      throw bad(
        "a shader built by another renderer uses this node. A VDBGridNode binds the pool's buffer through the renderer " +
          "it was made with; on any other renderer it would bind nothing and read 0. Create one VDBGridNode per renderer " +
          "(new VDBGridNode(otherRenderer, pool, options)).",
      );
    }
    const first = moduleOfBuild.get(builder);
    if (first !== undefined && first !== this) {
      throw bad(
        "two VDB modules (two VDBGridNodes) in one shader. One VDB module per shader: generated symbols are declared " +
          "once, so a second module redeclares them. Read every grid of the shader through one VDBGridNode (by id), " +
          "made with the largest profile, types and helpers the shader needs.",
      );
    }
    moduleOfBuild.set(builder, this);
  }

  /** The module for a buffer reference (`'vdbPool.value'` in three r187). Memoized per reference. */
  moduleFor(ref: string): string {
    let text = this._byRef.get(ref);
    if (text === undefined) {
      text = vdbModuleWgsl({ ...this._options, buffer: ref });
      this._byRef.set(ref, text);
    }
    return text;
  }

  override generate(builder: NodeBuilder): string {
    this.checkBuild(builder);
    const ref = this.vdbPool.build(builder, "property");
    if (typeof ref !== "string") throw unsupported("built the pool storage node to a non-string property reference");
    const text = this.moduleFor(ref);
    const nodeCode = (builder as unknown as { getCodeFromNode(node: Node, type: string): { code: string } }).getCodeFromNode(
      this,
      this.getNodeType(builder),
    );
    nodeCode.code = text;
    return text;
  }
}

/**
 * Warns once per backend when the renderer made its own device (no `device`
 * parameter): `renderer.dispose()` then destroys that device, and with it the
 * pool's buffer.
 */
function warnOwnDevice(backend: BackendLike): void {
  if (backend.parameters === undefined || backend.parameters.device !== undefined || ownDeviceWarned.has(backend)) return;
  ownDeviceWarned.add(backend);
  console.warn(
    "@pmndrs/vdb/three: this WebGPURenderer created its own GPUDevice, so renderer.dispose() destroys that device and every " +
      "buffer on it, the VDBGridPool's included. For a pool that outlives the renderer, create the device yourself " +
      "(adapter.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(adapter) })) and pass new WebGPURenderer({ device }).",
  );
}

type HelperKind = "sampleWorld" | "sampleIndex" | "readValue";

const HELPER_WGSL: Readonly<Record<HelperKind, string>> = {
  sampleWorld: /* wgsl */ `fn vdbThreeSampleWorld(g: u32, p: vec3f) -> f32 {
    var a = vdbAccessorInit(g);
    return vdbSample(&a, vdbWorldToIndex(g, p));
}`,
  sampleIndex: /* wgsl */ `fn vdbThreeSampleIndex(g: u32, xyz: vec3f) -> f32 {
    var a = vdbAccessorInit(g);
    return vdbSample(&a, xyz);
}`,
  readValue: /* wgsl */ `fn vdbThreeReadValue(g: u32, ijk: vec3i) -> f32 {
    var a = vdbAccessorInit(g);
    return vdbReadValue(&a, ijk);
}`,
};

/**
 * A `VDBGridPool` in three's node system; see the file comment. Not itself a
 * three node: it holds the storage node (`storage`), the module code node
 * (`module`) and the TSL helpers' functions.
 */
export class VDBGridNode {
  /** The pool's buffer as a read-only three storage node (`array<u32>`). */
  readonly storage: StorageNode;
  /** The generated WGSL module (`vdbAccessorInit`, `vdbSample`, ...) as a three code node: include it in a `wgslFn`. */
  readonly module: Node;
  /** The module options this node compiles (validated; `types` sorted). */
  readonly options: Readonly<ModuleOptions>;
  /** Name of the storage binding three emits. */
  readonly name: string;
  /**
   * The renderer this node binds through. Only shaders this renderer builds
   * may use the node; make one `VDBGridNode` per renderer.
   */
  readonly renderer: VDBThreeRendererLike;

  private readonly three: RendererLike;
  private boundVersion: number;
  private bound: GPUBuffer;
  private disposed = false;
  private readonly verified = new WeakSet<object>();
  private readonly helpers = new Map<HelperKind, (...args: unknown[]) => VDBTslNode>();

  constructor(
    renderer: VDBThreeRendererLike,
    readonly pool: VDBGridNodePoolLike,
    options: VDBGridNodeOptions = {},
  ) {
    const backend = webgpuBackendOf(renderer, "VDBGridNode");
    warnThreeRevision(REVISION);
    this.renderer = renderer;
    this.three = renderer as unknown as RendererLike;
    threeInternals.check(this.three);
    if (pool === null || typeof pool !== "object" || typeof pool.version !== "number") {
      throw bad("pool must be a VDBGridPool (or an object with device, buffer, version, layoutVersion, types).");
    }
    if (pool.device !== backend.device) {
      throw bad(
        "the pool was created on a different GPUDevice than the renderer's. Create it with VDBGridPool.create(getDevice(renderer)).",
      );
    }
    if (options === null || typeof options !== "object") throw bad("options must be an object.");
    for (const key of Object.keys(options)) {
      if (key === "slot" || key === "buffer") {
        throw bad(`'${key}' is not an option here: three assigns the binding, and VDBGridNode reads it through its storage node.`);
      }
      if (!OPTION_KEYS.has(key)) throw bad(`unknown option '${key}'; use ${[...OPTION_KEYS].join(", ")}.`);
    }
    if ((options.profile as string | undefined) === "majorant") {
      throw new VDBError("VDB_BAD_OPTION", "VDBGridNode: profile 'majorant' is deferred past 0.1 (D22). Use 'read', 'sample' or 'march'.");
    }
    const name = options.name ?? "vdbPool";
    const explicitTypes = options.types !== undefined;
    const poolTypes = [...pool.types];
    if (!explicitTypes && poolTypes.length === 0) {
      throw bad("the pool holds no grids yet, so there are no types to compile. Pass types (e.g. ['float', 'fp8']) or add grids first.");
    }
    const raw: VDBModuleOptions = {
      buffer: `${name}.value`,
      types: explicitTypes ? options.types! : poolTypes,
      profile: options.profile ?? "sample",
      sampler: options.sampler,
      isActive: options.isActive,
      outside: options.outside,
      // Materialize once (a one-shot iterable would be empty after validation); non-iterables reach the validator as is.
      helpers:
        options.helpers != null && typeof options.helpers !== "string" && typeof options.helpers[Symbol.iterator] === "function"
          ? [...options.helpers]
          : options.helpers,
    };
    // Validates every option (VDB_BAD_OPTION), the name as a WGSL identifier included.
    vdbModuleWgsl(raw);
    // With the default types, the module must decode every grid the pool holds now.
    // Explicit types are the caller's choice (a pool may be shared by modules
    // compiling different subsets); the layout must match either way.
    vdbCheckModule(explicitTypes ? { types: [], layoutVersion: pool.layoutVersion } : pool, raw);
    this.name = name;
    this.options = Object.freeze({
      types: Object.freeze([...new Set(raw.types)].sort()) as readonly VDBGridTypeName[],
      profile: raw.profile,
      ...(raw.sampler !== undefined ? { sampler: raw.sampler } : {}),
      ...(raw.isActive !== undefined ? { isActive: raw.isActive } : {}),
      ...(raw.outside !== undefined ? { outside: raw.outside } : {}),
      ...(raw.helpers !== undefined ? { helpers: Object.freeze([...new Set(raw.helpers)].sort()) as readonly VDBModuleHelper[] } : {}),
    });

    // Before any pool buffer reaches three: the destroy guard must hold on this backend.
    threeInternals.selfTest(this.three);
    warnOwnDevice(this.three.backend);

    this.bound = pool.buffer;
    this.boundVersion = pool.version;
    const node = storage(this.standIn(this.bound), "uint", 0).toReadOnly() as unknown as StorageNode;
    node.setName(name);
    this.storage = node;
    this.module = new VDBModuleNode(node, this.options, this.three.backend);
  }

  /** The pool version bound now (the pool's at construction or the last {@link update} that rebound). */
  get version(): number {
    return this.boundVersion;
  }

  /**
   * Rebinds the storage node when the pool replaced its buffer (`version`
   * changed, or the buffer object differs). Returns true when it rebound.
   * Cheap when nothing changed; call it every frame before three renders or
   * computes with this node.
   *
   * @throws VDBError `VDB_ENGINE_MISMATCH` when three bound something other
   *   than the pool's buffer to a stand-in it used (a three whose internals
   *   differ; see `threeInternals`); `VDB_DISPOSED` after {@link dispose}.
   */
  update(): boolean {
    if (this.disposed) throw new VDBError("VDB_DISPOSED", "VDBGridNode: update() after dispose().");
    this.checkBinding();
    const buffer = this.pool.buffer;
    if (this.pool.version === this.boundVersion && buffer === this.bound) return false;
    const old = this.storage.value as StorageBufferAttribute;
    this.storage.value = this.standIn(buffer);
    this.retire(old);
    this.bound = buffer;
    this.boundVersion = this.pool.version;
    return true;
  }

  /**
   * Checks that this module can read every grid the pool holds now.
   *
   * @throws VDBError `VDB_TYPE_NOT_COMPILED` (those grids would read 0) or `VDB_LAYOUT_MISMATCH`.
   */
  check(): void {
    vdbCheckModule(this.pool, { types: this.options.types });
  }

  /** Releases the stand-in attribute. The pool's buffer is not destroyed (it is the pool's). */
  dispose(): void {
    if (this.disposed) return;
    this.retire(this.storage.value as StorageBufferAttribute);
    this.disposed = true;
  }

  /** @internal The `wgslFn` behind a TSL helper, made once per node (the module is included once per shader). */
  helper(kind: HelperKind): (...args: unknown[]) => VDBTslNode {
    let fn = this.helpers.get(kind);
    if (fn === undefined) {
      if (kind !== "readValue" && this.options.profile === "read") {
        throw bad("vdbSample needs profile 'sample' or 'march'; this VDBGridNode was made with profile 'read'.");
      }
      fn = wgslFn(HELPER_WGSL[kind], [this.module] as never) as unknown as (...args: unknown[]) => VDBTslNode;
      this.helpers.set(kind, fn);
    }
    return fn;
  }

  /** The stand-in three used still has the pool's buffer, and three made none of its own. */
  private checkBinding(): void {
    const attribute = this.storage.value as StorageBufferAttribute;
    if (this.verified.has(attribute) || !threeInternals.usedByThree(this.three, attribute)) return;
    const backend = this.three.backend;
    if (threeInternals.bufferOf(backend, attribute) !== this.bound || threeInternals.otherBuffer(backend, attribute, this.bound)) {
      throw unsupported("did not bind the pool buffer VDBGridNode set (backend.get(attribute).buffer)");
    }
    this.verified.add(attribute);
  }

  /** A `StorageBufferAttribute` whose GPU buffer is `buffer`: three takes it as made, never creates one, never destroys it. */
  private standIn(buffer: GPUBuffer): StorageBufferAttribute {
    const attribute = new StorageBufferAttribute(new Uint32Array(1), 1);
    attribute.name = buffer.label || this.name;
    threeInternals.bind(this.three.backend, attribute, buffer);
    threeInternals.guardDestroy(this.three.backend, attribute);
    return attribute;
  }

  /** Lets three forget an attribute: the destroy guard swaps a throwaway in for the pool's buffer, which three then destroys. */
  private retire(attribute: StorageBufferAttribute): void {
    attribute.dispose();
  }
}

/** A grid id as a TSL node: a number becomes `uint(n)`. */
function gridId(index: VDBTslNode | number): VDBTslNode {
  if (typeof index === "number") {
    if (!Number.isInteger(index) || index < 0) throw bad(`grid index must be a non-negative integer, got ${index}.`);
    return uint(index);
  }
  return index;
}

/** Options of {@link vdbSample}. */
export interface VDBSampleOptions {
  /** `'world'` (default): `position` is world space, through the grid's record transform (`setTransform`). `'index'`: index space. */
  readonly space?: "world" | "index" | undefined;
}

/**
 * TSL: the grid's value at `position` (a `vec3` node) with the node's sampler
 * (`trilinear` or `nearest`) and `outside` mode. `gridIndex` is a `uint` node
 * or a number (the id `pool.add` returned). Needs profile `sample` or `march`.
 * A dead id, or a grid type the module does not compile, reads 0.
 */
export function vdbSample(grids: VDBGridNode, gridIndex: VDBTslNode | number, position: VDBTslNode, options: VDBSampleOptions = {}): VDBTslNode {
  const space = options.space ?? "world";
  if (space !== "world" && space !== "index") throw bad(`vdbSample space must be 'world' or 'index', got ${JSON.stringify(space)}.`);
  return grids.helper(space === "world" ? "sampleWorld" : "sampleIndex")(gridId(gridIndex), position);
}

/**
 * TSL: the grid's value at index coordinate `ijk` (an `ivec3` node; a float
 * vector converts by truncation). Any profile. A dead id, or a grid type the
 * module does not compile, reads 0.
 */
export function vdbReadValue(grids: VDBGridNode, gridIndex: VDBTslNode | number, ijk: VDBTslNode): VDBTslNode {
  return grids.helper("readValue")(gridId(gridIndex), ijk);
}
