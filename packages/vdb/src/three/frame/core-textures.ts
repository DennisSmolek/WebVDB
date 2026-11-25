/**
 * Binding the core frame renderer's `GPUTexture`s into three r187 (K6.1, design §6; owner Q2): a stand-in
 * per output, three's `ExternalTexture` (r187: `backend.createTexture` takes `sourceTexture` as the GPU
 * texture and `destroyTexture` never destroys it) flagged `is3DTexture`, so a `texture3D()` node binds
 * the core texture with a 3D view. When `outputs.version` moves, a new stand-in replaces the old one (three
 * cannot re-point an initialized ExternalTexture) and the node's `value` swaps, which rebuilds its bind
 * group. Once per backend a self-test puts a throwaway texture through the same path and checks three
 * took it as given and did not destroy it, else `VDB_ENGINE_MISMATCH`.
 */

import { ClampToEdgeWrapping, ExternalTexture, LinearFilter, REVISION } from "three";
import { VDBError } from "../../core/errors.js";

interface BackendLike {
  get(object: object): Record<string, unknown>;
}
interface RendererLike {
  readonly backend: object;
  initTexture(texture: object): void;
}

/** Label of the once-per-backend self-test texture (1³; tests that count textures skip it). */
export const VDB_THREE_CORE_TEXTURE_SELF_TEST_LABEL = "@pmndrs/vdb/three core-texture self-test";

const selfTested = new WeakMap<object, true | VDBError>();

const mismatch = (what: string) =>
  new VDBError(
    "VDB_ENGINE_MISMATCH",
    `three adapter: three r${REVISION} ${what}. The core frame renderer binds its GPUTextures into three through ExternalTexture ` +
      "(written against three r187); this three version is unsupported for backend 'core'.",
  );

/** A three stand-in for a core 3D texture. */
export function vdbCoreTextureStandIn(gpu: GPUTexture): ExternalTexture {
  const t = new ExternalTexture(gpu);
  const f = t as unknown as Record<string, unknown>;
  f.is3DTexture = true;
  f.wrapR = ClampToEdgeWrapping;
  t.wrapS = t.wrapT = ClampToEdgeWrapping;
  t.minFilter = t.magFilter = LinearFilter;
  t.generateMipmaps = false;
  t.flipY = false;
  t.name = gpu.label;
  f.image = { width: gpu.width, height: gpu.height, depth: gpu.depthOrArrayLayers };
  return t;
}

/**
 * Checks once per backend that three binds an ExternalTexture's `sourceTexture` as given and never destroys
 * it (dispose included).
 *
 * @throws VDBError `VDB_ENGINE_MISMATCH` when three's internals differ.
 */
export function vdbCoreTextureSelfTest(renderer: RendererLike, device: GPUDevice): void {
  const backend = renderer.backend as BackendLike;
  const cached = selfTested.get(backend);
  if (cached === true) return;
  if (cached) throw cached;
  let error: VDBError | null = null;
  const dummy = device.createTexture({ label: VDB_THREE_CORE_TEXTURE_SELF_TEST_LABEL, size: [1, 1, 1], dimension: "3d", format: "rgba16float", usage: 0x04 /* TEXTURE_BINDING */ });
  const realDestroy = dummy.destroy;
  let destroyedByThree = false;
  (dummy as { destroy: () => void }).destroy = () => {
    destroyedByThree = true;
  };
  try {
    if (typeof backend.get !== "function" || typeof renderer.initTexture !== "function") throw mismatch("has no backend.get or renderer.initTexture");
    const t = vdbCoreTextureStandIn(dummy);
    renderer.initTexture(t);
    if (backend.get(t).texture !== dummy) error = mismatch("did not bind an ExternalTexture's sourceTexture as its GPU texture");
    t.dispose();
    if (!error && destroyedByThree) error = mismatch("destroyed an ExternalTexture's sourceTexture on dispose");
  } catch (e) {
    error = e instanceof VDBError ? e : mismatch(`failed the core-texture self-test (${e instanceof Error ? e.message : String(e)})`);
  } finally {
    (dummy as { destroy: () => void }).destroy = realDestroy;
    dummy.destroy();
  }
  selfTested.set(backend, error ?? true);
  if (error) throw error;
}

/** One core output bound into three: `texture` follows the core texture (a new stand-in when it changes). */
export class VDBCoreTextureBinding {
  #gpu: GPUTexture | null = null;
  #standIn: ExternalTexture | null = null;

  /** The current stand-in (null before the first {@link set}). */
  get texture(): ExternalTexture | null {
    return this.#standIn;
  }

  /** Points the binding at `gpu`; returns true when the stand-in changed (swap the node's `value`). */
  set(gpu: GPUTexture): boolean {
    if (gpu === this.#gpu) return false;
    // ExternalTexture.dispose never destroys the core texture (the self-test checked it); the core retires it.
    this.#standIn?.dispose();
    this.#gpu = gpu;
    this.#standIn = vdbCoreTextureStandIn(gpu);
    return true;
  }

  /** Drops the stand-in (the core texture belongs to the core renderer). */
  dispose(): void {
    this.#standIn?.dispose();
    this.#standIn = null;
    this.#gpu = null;
  }
}
