/**
 * A `WebGPURenderer` stand-in whose attribute paths are three r187's own
 * (`Attributes`, `WebGPUAttributeUtils`; bvh's method), for Node tests of the
 * `./three` adapter. `VDBGridNode`'s construction-time self-test runs on it.
 */

const scope = globalThis as unknown as Record<string, unknown>;
scope.GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
scope.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };

export type ThreeAttributes = { update(attribute: object, type: number): void; has(attribute: object): boolean; delete(attribute: object): unknown; dispose(): void };
const attributesModule = "three/src/renderers/common/Attributes.js";
const utilsModule = "three/src/renderers/webgpu/utils/WebGPUAttributeUtils.js";
const constantsModule = "three/src/renderers/common/Constants.js";
const { default: Attributes } = (await import(attributesModule)) as { default: new (backend: object, info: object) => ThreeAttributes };
const { default: WebGPUAttributeUtils } = (await import(utilsModule)) as {
  default: new (backend: object) => { createAttribute(attribute: object, usage: number): void; destroyAttribute(attribute: object): void };
};
export const { AttributeType } = (await import(constantsModule)) as { AttributeType: { STORAGE: number } };

export interface TestBuffer {
  label: string;
  destroyed: boolean;
  destroy(): void;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
}

export function makeBuffer(label: string): TestBuffer {
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

/** Label of the self-test buffer `VDBGridNode` makes once per backend. */
export const SELF_TEST_LABEL = "@pmndrs/vdb/three self-test";

/**
 * A WebGPURenderer stand-in whose attribute paths are three's own (r187).
 * `created` records every buffer made on the default recording device: the
 * first `VDBGridNode` adds two (the self-test buffer and the throwaway the
 * destroy guard swapped in for it).
 */
export function fakeRenderer(device?: object, options: { parameters?: { device?: object } } = {}) {
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
    ...(options.parameters ? { parameters: options.parameters } : {}),
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
  return { renderer, backend, device, created, attributes, utils, use };
}
